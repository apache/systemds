/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 * 
 *   http://www.apache.org/licenses/LICENSE-2.0
 * 
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

package org.apache.sysml.hops.codegen.template;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.LinkedList;

import org.apache.sysml.hops.AggUnaryOp;
import org.apache.sysml.hops.BinaryOp;
import org.apache.sysml.hops.Hop;
import org.apache.sysml.hops.UnaryOp;
import org.apache.sysml.hops.Hop.AggOp;
import org.apache.sysml.hops.Hop.Direction;
import org.apache.sysml.hops.Hop.OpOp2;
import org.apache.sysml.hops.TernaryOp;
import org.apache.sysml.hops.codegen.cplan.CNode;
import org.apache.sysml.hops.codegen.cplan.CNodeBinary;
import org.apache.sysml.hops.codegen.cplan.CNodeBinary.BinType;
import org.apache.sysml.hops.codegen.cplan.CNodeCell;
import org.apache.sysml.hops.codegen.cplan.CNodeData;
import org.apache.sysml.hops.codegen.cplan.CNodeTpl;
import org.apache.sysml.hops.codegen.cplan.CNodeUnary;
import org.apache.sysml.hops.codegen.cplan.CNodeUnary.UnaryType;
import org.apache.sysml.hops.codegen.template.CPlanMemoTable.MemoTableEntry;
import org.apache.sysml.hops.codegen.cplan.CNodeTernary;
import org.apache.sysml.hops.codegen.cplan.CNodeTernary.TernaryType;
import org.apache.sysml.hops.rewrite.HopRewriteUtils;
import org.apache.sysml.parser.Expression.DataType;
import org.apache.sysml.runtime.matrix.data.Pair;

public class CellTpl extends BaseTpl 
{	
	public CellTpl() {
		super(TemplateType.CellTpl);
	}

	@Override
	public boolean open(Hop hop) {
		return isValidOperation(hop);
	}

	@Override
	public boolean fuse(Hop hop, Hop input) {
		return !isClosed() && (isValidOperation(hop) 
			|| ( hop instanceof AggUnaryOp && ((AggUnaryOp) hop).getOp() == AggOp.SUM 
				&& ((AggUnaryOp) hop).getDirection()!= Direction.Col));
	}

	@Override
	public boolean merge(Hop hop, Hop input) {
		//merge of other cell tpl possible
		return (!isClosed() && isValidOperation(hop));
	}

	@Override
	public CloseType close(Hop hop) {
		//need to close cell tpl after aggregation, see fuse for exact properties
		if( hop instanceof AggUnaryOp && isValidOperation(hop.getInput().get(0)) )
			return CloseType.CLOSED_VALID;
		else if( hop instanceof AggUnaryOp )
			return CloseType.CLOSED_INVALID;
		else
			return CloseType.OPEN;
	}

	@Override
	public Pair<Hop[], CNodeTpl> constructCplan(Hop hop, CPlanMemoTable memo, boolean compileLiterals) 
	{
		//recursively process required cplan output
		HashSet<Hop> inHops = new HashSet<Hop>();
		HashMap<Long, CNode> tmp = new HashMap<Long, CNode>();
		hop.resetVisitStatus();
		rConstructCplan(hop, memo, tmp, inHops, compileLiterals);
		hop.resetVisitStatus();
		
		//reorder inputs (ensure matrices/vectors come first, prune literals)
		LinkedList<Hop> sinHops = new LinkedList<Hop>();
		for( int i : new int[]{0,1,2} ) //matrices, vectors, scalars
			for( Hop h : inHops ) //matrices
				if( (i==0 && h.getDataType().isMatrix() && !TemplateUtils.isVector(h))
					|| (i==1 && h.getDataType().isMatrix() && TemplateUtils.isVector(h))
					|| (i==2 && h.getDataType().isScalar() && !tmp.get(h.getHopID()).isLiteral())) {
					sinHops.add(h);
				}
		
		//construct template node
		ArrayList<CNode> inputs = new ArrayList<CNode>();
		for( Hop in : sinHops )
			inputs.add(tmp.get(in.getHopID()));
		CNode output = tmp.get(hop.getHopID());
		CNodeCell tpl = new CNodeCell(inputs, output);
		tpl.setCellType(TemplateUtils.getCellType(hop));
		
		// return cplan instance
		return new Pair<Hop[],CNodeTpl>(sinHops.toArray(new Hop[0]), tpl);
	}
	
	private void rConstructCplan(Hop hop, CPlanMemoTable memo, HashMap<Long, CNode> tmp, HashSet<Hop> inHops, boolean compileLiterals) 
	{
		//recursively process required childs
		MemoTableEntry me = memo.getBest(hop.getHopID(), TemplateType.CellTpl);
		for( int i=0; i<hop.getInput().size(); i++ ) {
			Hop c = hop.getInput().get(i);
			if( me.isPlanRef(i) )
				rConstructCplan(c, memo, tmp, inHops, compileLiterals);
			else {
				CNodeData cdata = TemplateUtils.createCNodeData(c, compileLiterals);	
				tmp.put(c.getHopID(), cdata);
				inHops.add(c);
			}
		}
		
		//construct cnode for current hop
		CNode out = null;
		if(hop instanceof UnaryOp)
		{
			CNode cdata1 = tmp.get(hop.getInput().get(0).getHopID());
			if( TemplateUtils.isColVector(cdata1) )
				cdata1 = new CNodeUnary(cdata1, UnaryType.LOOKUP_R);
			else if( cdata1 instanceof CNodeData && hop.getInput().get(0).getDataType().isMatrix() )
				cdata1 = new CNodeUnary(cdata1, UnaryType.LOOKUP_RC);
			
			String primitiveOpName = ((UnaryOp)hop).getOp().toString();
			out = new CNodeUnary(cdata1, UnaryType.valueOf(primitiveOpName));
		}
		else if(hop instanceof BinaryOp)
		{
			BinaryOp bop = (BinaryOp) hop;
			CNode cdata1 = tmp.get(hop.getInput().get(0).getHopID());
			CNode cdata2 = tmp.get(hop.getInput().get(1).getHopID());
			String primitiveOpName = bop.getOp().toString();
			
			//cdata1 is vector
			if( TemplateUtils.isColVector(cdata1) )
				cdata1 = new CNodeUnary(cdata1, UnaryType.LOOKUP_R);
			else if( cdata1 instanceof CNodeData && hop.getInput().get(0).getDataType().isMatrix() )
				cdata1 = new CNodeUnary(cdata1, UnaryType.LOOKUP_RC);
			
			//cdata2 is vector
			if( TemplateUtils.isColVector(cdata2) )
				cdata2 = new CNodeUnary(cdata2, UnaryType.LOOKUP_R);
			else if( cdata2 instanceof CNodeData && hop.getInput().get(1).getDataType().isMatrix() )
				cdata2 = new CNodeUnary(cdata2, UnaryType.LOOKUP_RC);
			
			if( bop.getOp()==OpOp2.POW && cdata2.isLiteral() && cdata2.getVarname().equals("2") )
				out = new CNodeUnary(cdata1, UnaryType.POW2);
			else if( bop.getOp()==OpOp2.MULT && cdata2.isLiteral() && cdata2.getVarname().equals("2") )
				out = new CNodeUnary(cdata1, UnaryType.MULT2);
			else //default binary	
				out = new CNodeBinary(cdata1, cdata2, BinType.valueOf(primitiveOpName));
		}
		else if(hop instanceof TernaryOp) 
		{
			TernaryOp top = (TernaryOp) hop;
			CNode cdata1 = tmp.get(hop.getInput().get(0).getHopID());
			CNode cdata2 = tmp.get(hop.getInput().get(1).getHopID());
			CNode cdata3 = tmp.get(hop.getInput().get(2).getHopID());
			
			//cdata1 is vector
			if( TemplateUtils.isColVector(cdata1) )
				cdata1 = new CNodeUnary(cdata1, UnaryType.LOOKUP_R);
			else if( cdata1 instanceof CNodeData && hop.getInput().get(0).getDataType().isMatrix() )
				cdata1 = new CNodeUnary(cdata1, UnaryType.LOOKUP_RC);
			
			//cdata3 is vector
			if( TemplateUtils.isColVector(cdata3) )
				cdata3 = new CNodeUnary(cdata3, UnaryType.LOOKUP_R);
			else if( cdata3 instanceof CNodeData && hop.getInput().get(2).getDataType().isMatrix() )
				cdata3 = new CNodeUnary(cdata3, UnaryType.LOOKUP_RC);
			
			//construct ternary cnode, primitive operation derived from OpOp3
			out = new CNodeTernary(cdata1, cdata2, cdata3, 
					TernaryType.valueOf(top.getOp().toString()));
		}
		else if (hop instanceof AggUnaryOp && ((AggUnaryOp)hop).getOp() == AggOp.SUM
			&& (((AggUnaryOp) hop).getDirection() == Direction.RowCol 
			|| ((AggUnaryOp) hop).getDirection() == Direction.Row) )
		{
			out = tmp.get(hop.getInput().get(0).getHopID());
		}
		
		tmp.put(hop.getHopID(), out);
	}
	
	private static boolean isValidOperation(Hop hop) 
	{	
		//prepare indicators for binary operations
		boolean isBinaryMatrixScalar = false;
		boolean isBinaryMatrixVector = false;
		boolean isBinaryMatrixMatrixDense = false;
		if( hop instanceof BinaryOp && hop.getDataType().isMatrix() ) {
			Hop left = hop.getInput().get(0);
			Hop right = hop.getInput().get(1);
			DataType ldt = left.getDataType();
			DataType rdt = right.getDataType();
			
			isBinaryMatrixScalar = (ldt.isScalar() || rdt.isScalar());	
			isBinaryMatrixVector = hop.dimsKnown() 
				&& ((ldt.isMatrix() && TemplateUtils.isVectorOrScalar(right) && !TemplateUtils.isBinaryMatrixRowVector(hop)) 
				|| (rdt.isMatrix() && TemplateUtils.isVectorOrScalar(left) && !TemplateUtils.isBinaryMatrixRowVector(hop)) );
			isBinaryMatrixMatrixDense = hop.dimsKnown() && HopRewriteUtils.isEqualSize(left, right)
				&& ldt.isMatrix() && rdt.isMatrix() && !HopRewriteUtils.isSparse(left) && !HopRewriteUtils.isSparse(right);
		}
				
		//prepare indicators for ternary operations
		boolean isTernaryVectorScalarVector = false;
		boolean isTernaryMatrixScalarMatrixDense = false;
		if( hop instanceof TernaryOp && hop.getInput().size()==3 && hop.dimsKnown() 
			&& HopRewriteUtils.checkInputDataTypes(hop, DataType.MATRIX, DataType.SCALAR, DataType.MATRIX)) {
			Hop left = hop.getInput().get(0);
			Hop right = hop.getInput().get(2);
			
			isTernaryVectorScalarVector = TemplateUtils.isVector(left) && TemplateUtils.isVector(right);
			isTernaryMatrixScalarMatrixDense = HopRewriteUtils.isEqualSize(left, right) 
				&& !HopRewriteUtils.isSparse(left) && !HopRewriteUtils.isSparse(right);
		}
		
		//check supported unary, binary, ternary operations
		return hop.getDataType() == DataType.MATRIX && TemplateUtils.isOperationSupported(hop) && (hop instanceof UnaryOp 
				|| isBinaryMatrixScalar || isBinaryMatrixVector || isBinaryMatrixMatrixDense 
				|| isTernaryVectorScalarVector || isTernaryMatrixScalarMatrixDense);	
	}
}
