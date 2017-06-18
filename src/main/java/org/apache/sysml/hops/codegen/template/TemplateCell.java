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
import java.util.Comparator;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.stream.Collectors;

import org.apache.sysml.hops.AggBinaryOp;
import org.apache.sysml.hops.AggUnaryOp;
import org.apache.sysml.hops.BinaryOp;
import org.apache.sysml.hops.DataOp;
import org.apache.sysml.hops.Hop;
import org.apache.sysml.hops.UnaryOp;
import org.apache.sysml.hops.Hop.AggOp;
import org.apache.sysml.hops.Hop.Direction;
import org.apache.sysml.hops.Hop.OpOp2;
import org.apache.sysml.hops.Hop.ParamBuiltinOp;
import org.apache.sysml.hops.IndexingOp;
import org.apache.sysml.hops.LiteralOp;
import org.apache.sysml.hops.ParameterizedBuiltinOp;
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

public class TemplateCell extends TemplateBase 
{	
	private static final AggOp[] SUPPORTED_AGG = 
			new AggOp[]{AggOp.SUM, AggOp.SUM_SQ, AggOp.MIN, AggOp.MAX};
	
	public TemplateCell() {
		super(TemplateType.CellTpl);
	}
	
	public TemplateCell(boolean closed) {
		super(TemplateType.CellTpl, closed);
	}
	
	public TemplateCell(TemplateType type, boolean closed) {
		super(type, closed);
	}
	

	@Override
	public boolean open(Hop hop) {
		return hop.dimsKnown() && isValidOperation(hop)
				&& !(hop.getDim1()==1 && hop.getDim2()==1) 	
			|| (hop instanceof IndexingOp && (((IndexingOp)hop)
				.isColLowerEqualsUpper() || hop.getDim2()==1));
	}

	@Override
	public boolean fuse(Hop hop, Hop input) {
		return !isClosed() && (isValidOperation(hop) 
			|| (HopRewriteUtils.isAggUnaryOp(hop, SUPPORTED_AGG) 
				&& ((AggUnaryOp) hop).getDirection()!= Direction.Col)
			|| (HopRewriteUtils.isMatrixMultiply(hop) && hop.getDim1()==1 && hop.getDim2()==1)
				&& HopRewriteUtils.isTransposeOperation(hop.getInput().get(0)));
	}

	@Override
	public boolean merge(Hop hop, Hop input) {
		//merge of other cell tpl possible
		return (!isClosed() && isValidOperation(hop));
	}

	@Override
	public CloseType close(Hop hop) {
		//need to close cell tpl after aggregation, see fuse for exact properties
		if( (HopRewriteUtils.isAggUnaryOp(hop, SUPPORTED_AGG) 
				&& ((AggUnaryOp) hop).getDirection()!= Direction.Col)
			|| (HopRewriteUtils.isMatrixMultiply(hop) && hop.getDim1()==1 && hop.getDim2()==1) )
			return CloseType.CLOSED_VALID;
		else if( hop instanceof AggUnaryOp || hop instanceof AggBinaryOp )
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
		
		//reorder inputs (ensure matrices/vectors come first) and prune literals
		//note: we order by number of cells and subsequently sparsity to ensure
		//that sparse inputs are used as the main input w/o unnecessary conversion
		List<Hop> sinHops = inHops.stream()
			.filter(h -> !(h.getDataType().isScalar() && tmp.get(h.getHopID()).isLiteral()))
			.sorted(new HopInputComparator()).collect(Collectors.toList());
		
		//construct template node
		ArrayList<CNode> inputs = new ArrayList<CNode>();
		for( Hop in : sinHops )
			inputs.add(tmp.get(in.getHopID()));
		CNode output = tmp.get(hop.getHopID());
		CNodeCell tpl = new CNodeCell(inputs, output);
		tpl.setCellType(TemplateUtils.getCellType(hop));
		tpl.setAggOp(TemplateUtils.getAggOp(hop));
		tpl.setSparseSafe((HopRewriteUtils.isBinary(hop, OpOp2.MULT) && hop.getInput().contains(sinHops.get(0)))
				|| (HopRewriteUtils.isBinary(hop, OpOp2.DIV) && hop.getInput().get(0) == sinHops.get(0)));
		tpl.setRequiresCastDtm(hop instanceof AggBinaryOp);
		tpl.setBeginLine(hop.getBeginLine());
		
		// return cplan instance
		return new Pair<Hop[],CNodeTpl>(sinHops.toArray(new Hop[0]), tpl);
	}
	
	protected void rConstructCplan(Hop hop, CPlanMemoTable memo, HashMap<Long, CNode> tmp, HashSet<Hop> inHops, boolean compileLiterals) 
	{
		//memoization for common subexpression elimination and to avoid redundant work 
		if( tmp.containsKey(hop.getHopID()) )
			return;
		
		MemoTableEntry me = memo.getBest(hop.getHopID(), TemplateType.CellTpl);
		
		//recursively process required childs
		if( me!=null && (me.type == TemplateType.RowTpl || me.type == TemplateType.OuterProdTpl) ) {
			CNodeData cdata = TemplateUtils.createCNodeData(hop, compileLiterals);	
			tmp.put(hop.getHopID(), cdata);
			inHops.add(hop);
			return;
		}
		for( int i=0; i<hop.getInput().size(); i++ ) {
			Hop c = hop.getInput().get(i);
			if( me!=null && me.isPlanRef(i) && !(c instanceof DataOp)
				&& (me.type!=TemplateType.MultiAggTpl || memo.contains(c.getHopID(), TemplateType.CellTpl)))
				rConstructCplan(c, memo, tmp, inHops, compileLiterals);
			else if( me!=null && me.type==TemplateType.MultiAggTpl && HopRewriteUtils.isMatrixMultiply(hop) && i==0 )
				rConstructCplan(c.getInput().get(0), memo, tmp, inHops, compileLiterals);
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
			cdata1 = TemplateUtils.wrapLookupIfNecessary(cdata1, hop.getInput().get(0));
			
			String primitiveOpName = ((UnaryOp)hop).getOp().name();
			out = new CNodeUnary(cdata1, UnaryType.valueOf(primitiveOpName));
		}
		else if(hop instanceof BinaryOp)
		{
			BinaryOp bop = (BinaryOp) hop;
			CNode cdata1 = tmp.get(hop.getInput().get(0).getHopID());
			CNode cdata2 = tmp.get(hop.getInput().get(1).getHopID());
			String primitiveOpName = bop.getOp().name();
			
			//add lookups if required
			cdata1 = TemplateUtils.wrapLookupIfNecessary(cdata1, hop.getInput().get(0));
			cdata2 = TemplateUtils.wrapLookupIfNecessary(cdata2, hop.getInput().get(1));
			
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
			
			//add lookups if required
			cdata1 = TemplateUtils.wrapLookupIfNecessary(cdata1, hop.getInput().get(0));
			cdata3 = TemplateUtils.wrapLookupIfNecessary(cdata3, hop.getInput().get(2));
			
			//construct ternary cnode, primitive operation derived from OpOp3
			out = new CNodeTernary(cdata1, cdata2, cdata3, 
					TernaryType.valueOf(top.getOp().name()));
		}
		else if( hop instanceof ParameterizedBuiltinOp ) 
		{
			CNode cdata1 = tmp.get(((ParameterizedBuiltinOp)hop).getTargetHop().getHopID());
			cdata1 = TemplateUtils.wrapLookupIfNecessary(cdata1, hop.getInput().get(0));
			
			CNode cdata2 = tmp.get(((ParameterizedBuiltinOp)hop).getParameterHop("pattern").getHopID());
			CNode cdata3 = tmp.get(((ParameterizedBuiltinOp)hop).getParameterHop("replacement").getHopID());
			TernaryType ttype = (cdata2.isLiteral() && cdata2.getVarname().equals("Double.NaN")) ? 
					TernaryType.REPLACE_NAN : TernaryType.REPLACE;
			out = new CNodeTernary(cdata1, cdata2, cdata3, ttype);
		}
		else if( hop instanceof IndexingOp ) 
		{
			CNode cdata1 = tmp.get(hop.getInput().get(0).getHopID());
			out = new CNodeTernary(cdata1, 
					TemplateUtils.createCNodeData(new LiteralOp(hop.getInput().get(0).getDim2()), true), 
					TemplateUtils.createCNodeData(hop.getInput().get(4), true),
					TernaryType.LOOKUP_RC1);
		}
		else if( HopRewriteUtils.isTransposeOperation(hop) ) 
		{
			out = TemplateUtils.skipTranspose(tmp.get(hop.getHopID()), 
				hop, tmp, compileLiterals);
			if( out instanceof CNodeData && !inHops.contains(hop.getInput().get(0)) )
				inHops.add(hop.getInput().get(0));
		}
		else if( hop instanceof AggUnaryOp )
		{
			//aggregation handled in template implementation (note: we do not compile 
			//^2 of SUM_SQ into the operator to simplify the detection of single operators)
			out = tmp.get(hop.getInput().get(0).getHopID());
		}
		else if( hop instanceof AggBinaryOp ) {
			//guaranteed to be a dot product, so there are two cases:
			//(1) t(X)%*%X -> sum(X^2) and t(X) %*% Y -> sum(X*Y)
			if( HopRewriteUtils.isTransposeOfItself(hop.getInput().get(0), hop.getInput().get(1)) ) {
				CNode cdata1 = tmp.get(hop.getInput().get(1).getHopID());
				if( TemplateUtils.isColVector(cdata1) )
					cdata1 = new CNodeUnary(cdata1, UnaryType.LOOKUP_R);
				out = new CNodeUnary(cdata1, UnaryType.POW2);
			}
			else {
				CNode cdata1 = TemplateUtils.skipTranspose(tmp.get(hop.getInput().get(0).getHopID()), 
						hop.getInput().get(0), tmp, compileLiterals);
				if( cdata1 instanceof CNodeData && !inHops.contains(hop.getInput().get(0).getInput().get(0)) )
					inHops.add(hop.getInput().get(0).getInput().get(0));
				if( TemplateUtils.isColVector(cdata1) )
					cdata1 = new CNodeUnary(cdata1, UnaryType.LOOKUP_R);
				CNode cdata2 = tmp.get(hop.getInput().get(1).getHopID());
				if( TemplateUtils.isColVector(cdata2) )
					cdata2 = new CNodeUnary(cdata2, UnaryType.LOOKUP_R);
				out = new CNodeBinary(cdata1, cdata2, BinType.MULT);
			}
		} 
		
		tmp.put(hop.getHopID(), out);
	}
	
	protected static boolean isValidOperation(Hop hop) 
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
				&& ((ldt.isMatrix() && TemplateUtils.isVectorOrScalar(right)) 
				|| (rdt.isMatrix() && TemplateUtils.isVectorOrScalar(left)) );
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
				|| isTernaryVectorScalarVector || isTernaryMatrixScalarMatrixDense
				|| (hop instanceof ParameterizedBuiltinOp && ((ParameterizedBuiltinOp)hop).getOp()==ParamBuiltinOp.REPLACE));	
	}
	
	/**
	 * Comparator to order input hops of the cell template. We try to order 
	 * matrices-vectors-scalars via sorting by number of cells and for 
	 * equal number of cells by sparsity to prefer sparse inputs as the main 
	 * input for sparsity exploitation.
	 */
	public static class HopInputComparator implements Comparator<Hop> 
	{
		@Override
		public int compare(Hop h1, Hop h2) {
			long ncells1 = h1.getDataType()==DataType.SCALAR ? Long.MIN_VALUE : 
				h1.dimsKnown() ? h1.getDim1()*h1.getDim2() : Long.MAX_VALUE;
			long ncells2 = h2.getDataType()==DataType.SCALAR ? Long.MIN_VALUE :
				h2.dimsKnown() ? h2.getDim1()*h2.getDim2() : Long.MAX_VALUE;
			if( ncells1 > ncells2 ) 
				return -1;
			else if( ncells1 < ncells2) 
				return 1;
			return Long.compare(
				h1.dimsKnown(true) ? h1.getNnz() : ncells1, 
				h2.dimsKnown(true) ? h2.getNnz() : ncells2);
		}
	}
}
