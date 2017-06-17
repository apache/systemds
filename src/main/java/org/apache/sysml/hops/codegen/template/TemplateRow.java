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
import org.apache.sysml.hops.Hop;
import org.apache.sysml.hops.IndexingOp;
import org.apache.sysml.hops.LiteralOp;
import org.apache.sysml.hops.ParameterizedBuiltinOp;
import org.apache.sysml.hops.TernaryOp;
import org.apache.sysml.hops.UnaryOp;
import org.apache.sysml.hops.codegen.cplan.CNode;
import org.apache.sysml.hops.codegen.cplan.CNodeBinary;
import org.apache.sysml.hops.codegen.cplan.CNodeBinary.BinType;
import org.apache.sysml.hops.codegen.cplan.CNodeTernary.TernaryType;
import org.apache.sysml.hops.codegen.cplan.CNodeData;
import org.apache.sysml.hops.codegen.cplan.CNodeRow;
import org.apache.sysml.hops.codegen.cplan.CNodeTernary;
import org.apache.sysml.hops.codegen.cplan.CNodeTpl;
import org.apache.sysml.hops.codegen.cplan.CNodeUnary;
import org.apache.sysml.hops.codegen.cplan.CNodeUnary.UnaryType;
import org.apache.sysml.hops.codegen.template.CPlanMemoTable.MemoTableEntry;
import org.apache.sysml.hops.rewrite.HopRewriteUtils;
import org.apache.sysml.hops.Hop.AggOp;
import org.apache.sysml.hops.Hop.Direction;
import org.apache.sysml.hops.Hop.OpOp1;
import org.apache.sysml.hops.Hop.OpOp2;
import org.apache.sysml.parser.Expression.DataType;
import org.apache.sysml.runtime.matrix.data.Pair;

public class TemplateRow extends TemplateBase 
{
	private static final Hop.AggOp[] SUPPORTED_ROW_AGG = new AggOp[]{AggOp.SUM, AggOp.MIN, AggOp.MAX};
	private static final Hop.OpOp1[] SUPPORTED_VECT_UNARY = new OpOp1[]{
			OpOp1.EXP, OpOp1.SQRT, OpOp1.LOG, OpOp1.ABS, OpOp1.ROUND, OpOp1.CEIL, OpOp1.FLOOR, OpOp1.SIGN};
	private static final Hop.OpOp2[] SUPPORTED_VECT_BINARY = new OpOp2[]{
			OpOp2.MULT, OpOp2.DIV, OpOp2.MINUS, OpOp2.PLUS, OpOp2.POW, OpOp2.MIN, OpOp2.MAX,
			OpOp2.EQUAL, OpOp2.NOTEQUAL, OpOp2.LESS, OpOp2.LESSEQUAL, OpOp2.GREATER, OpOp2.GREATEREQUAL};
	
	public TemplateRow() {
		super(TemplateType.RowTpl);
	}
	
	public TemplateRow(boolean closed) {
		super(TemplateType.RowTpl, closed);
	}
	
	@Override
	public boolean open(Hop hop) {
		return (hop instanceof BinaryOp && hop.dimsKnown() && hop.getInput().get(0).getDim2()>1 
				&& hop.getInput().get(1).getDim2()==1 && TemplateCell.isValidOperation(hop)) 
			|| (hop instanceof AggBinaryOp && hop.dimsKnown() && hop.getDim2()==1
				&& hop.getInput().get(0).getDim1()>1 && hop.getInput().get(0).getDim2()>1)
			|| (hop instanceof AggUnaryOp && ((AggUnaryOp)hop).getDirection()!=Direction.RowCol 
				&& hop.getInput().get(0).getDim1()>1 && hop.getInput().get(0).getDim2()>1
				&& HopRewriteUtils.isAggUnaryOp(hop, SUPPORTED_ROW_AGG));
	}

	@Override
	public boolean fuse(Hop hop, Hop input) {
		return !isClosed() && 
			(  (hop instanceof BinaryOp && TemplateUtils.isOperationSupported(hop) 
				&& (HopRewriteUtils.isBinaryMatrixColVectorOperation(hop)
					|| HopRewriteUtils.isBinaryMatrixScalarOperation(hop)
					|| HopRewriteUtils.isBinaryMatrixMatrixOperationWithSharedInput(hop)) ) 
			|| (HopRewriteUtils.isBinary(hop, OpOp2.CBIND) && hop.getInput().indexOf(input)==0
				&& input.getDim2()==1 && hop.getInput().get(1).getDim2()==1
				&& HopRewriteUtils.isEmpty(hop.getInput().get(1)))
			|| ((hop instanceof UnaryOp || hop instanceof ParameterizedBuiltinOp) 
					&& TemplateCell.isValidOperation(hop))		
			|| (hop instanceof AggUnaryOp && ((AggUnaryOp)hop).getDirection()!=Direction.RowCol
				&& HopRewriteUtils.isAggUnaryOp(hop, SUPPORTED_ROW_AGG))
			|| (hop instanceof AggBinaryOp && hop.getDim1()>1 && hop.getDim2()==1
				&& HopRewriteUtils.isTransposeOperation(hop.getInput().get(0))));
	}

	@Override
	public boolean merge(Hop hop, Hop input) {
		//merge rowagg tpl with cell tpl if input is a vector
		return !isClosed() &&
			((hop instanceof BinaryOp && TemplateUtils.isOperationSupported(hop)
				&& (input.getDim2()==1 //matrix-scalar/vector-vector ops )
					|| HopRewriteUtils.isBinaryMatrixMatrixOperationWithSharedInput(hop)))
			 ||(hop instanceof AggBinaryOp && input.getDim2()==1
				&& HopRewriteUtils.isTransposeOperation(hop.getInput().get(0))));
	}

	@Override
	public CloseType close(Hop hop) {
		//close on column aggregate (e.g., colSums, t(X)%*%y)
		if(    (hop instanceof AggUnaryOp && ((AggUnaryOp)hop).getDirection()==Direction.Col)
			|| (hop instanceof AggBinaryOp && HopRewriteUtils.isTransposeOperation(hop.getInput().get(0)))
			|| HopRewriteUtils.isBinary(hop, OpOp2.CBIND) )
			return CloseType.CLOSED_VALID;
		else
			return CloseType.OPEN;
	}

	@Override
	public Pair<Hop[], CNodeTpl> constructCplan(Hop hop, CPlanMemoTable memo, boolean compileLiterals) {
		//recursively process required cplan output
		HashSet<Hop> inHops = new HashSet<Hop>();
		HashMap<String, Hop> inHops2 = new HashMap<String,Hop>();
		HashMap<Long, CNode> tmp = new HashMap<Long, CNode>();
		hop.resetVisitStatus();
		rConstructCplan(hop, memo, tmp, inHops, inHops2, compileLiterals);
		hop.resetVisitStatus();
		
		//reorder inputs (ensure matrix is first input, and other inputs ordered by size)
		List<Hop> sinHops = inHops.stream()
			.filter(h -> !(h.getDataType().isScalar() && tmp.get(h.getHopID()).isLiteral()))
			.sorted(new HopInputComparator(inHops2.get("X"))).collect(Collectors.toList());
		
		//construct template node
		ArrayList<CNode> inputs = new ArrayList<CNode>();
		for( Hop in : sinHops )
			inputs.add(tmp.get(in.getHopID()));
		CNode output = tmp.get(hop.getHopID());
		CNodeRow tpl = new CNodeRow(inputs, output);
		tpl.setRowType(TemplateUtils.getRowType(hop, sinHops.get(0)));
		tpl.setNumVectorIntermediates(TemplateUtils
			.countVectorIntermediates(output, new HashSet<Long>()));
		tpl.getOutput().resetVisitStatus();
		tpl.rReorderCommutativeBinaryOps(tpl.getOutput(), sinHops.get(0).getHopID());
		tpl.setBeginLine(hop.getBeginLine());
		
		// return cplan instance
		return new Pair<Hop[],CNodeTpl>(sinHops.toArray(new Hop[0]), tpl);
	}

	private void rConstructCplan(Hop hop, CPlanMemoTable memo, HashMap<Long, CNode> tmp, HashSet<Hop> inHops, HashMap<String, Hop> inHops2, boolean compileLiterals) 
	{	
		//memoization for common subexpression elimination and to avoid redundant work 
		if( tmp.containsKey(hop.getHopID()) )
			return;
		
		//recursively process required childs
		MemoTableEntry me = memo.getBest(hop.getHopID(), TemplateType.RowTpl);
		for( int i=0; i<hop.getInput().size(); i++ ) {
			Hop c = hop.getInput().get(i);
			if( me!=null && me.isPlanRef(i) )
				rConstructCplan(c, memo, tmp, inHops, inHops2, compileLiterals);
			else {
				CNodeData cdata = TemplateUtils.createCNodeData(c, compileLiterals);	
				tmp.put(c.getHopID(), cdata);
				inHops.add(c);
			}
		}
		
		//construct cnode for current hop
		CNode out = null;
		if(hop instanceof AggUnaryOp)
		{
			CNode cdata1 = tmp.get(hop.getInput().get(0).getHopID());
			if( ((AggUnaryOp)hop).getDirection() == Direction.Row && HopRewriteUtils.isAggUnaryOp(hop, SUPPORTED_ROW_AGG) ) {
				if(hop.getInput().get(0).getDim2()==1)
					out = (cdata1.getDataType()==DataType.SCALAR) ? cdata1 : new CNodeUnary(cdata1,UnaryType.LOOKUP_R);
				else {
					String opcode = "ROW_"+((AggUnaryOp)hop).getOp().name().toUpperCase()+"S";
					out = new CNodeUnary(cdata1, UnaryType.valueOf(opcode));
					if( cdata1 instanceof CNodeData && inHops2.isEmpty() )
						inHops2.put("X", hop.getInput().get(0));
				}
			}
			else  if (((AggUnaryOp)hop).getDirection() == Direction.Col && ((AggUnaryOp)hop).getOp() == AggOp.SUM ) {
				//vector add without temporary copy
				if( cdata1 instanceof CNodeBinary && ((CNodeBinary)cdata1).getType().isVectorScalarPrimitive() )
					out = new CNodeBinary(cdata1.getInput().get(0), cdata1.getInput().get(1), 
							((CNodeBinary)cdata1).getType().getVectorAddPrimitive());
				else	
					out = cdata1;
			}
		}
		else if(hop instanceof AggBinaryOp)
		{
			CNode cdata1 = tmp.get(hop.getInput().get(0).getHopID());
			CNode cdata2 = tmp.get(hop.getInput().get(1).getHopID());
			
			if( HopRewriteUtils.isTransposeOperation(hop.getInput().get(0)) )
			{
				//correct input under transpose
				cdata1 = TemplateUtils.skipTranspose(cdata1, hop.getInput().get(0), tmp, compileLiterals);
				inHops.remove(hop.getInput().get(0)); 
				inHops.add(hop.getInput().get(0).getInput().get(0));
				
				//note: vectorMultAdd applicable to vector-scalar, and vector-vector
				out = new CNodeBinary(cdata1, cdata2, BinType.VECT_MULT_ADD);
				inHops2.put("X", hop.getInput().get(0).getInput().get(0));
			}
			else
			{
				if(hop.getInput().get(0).getDim2()==1 && hop.getInput().get(1).getDim2()==1)
					out = new CNodeBinary((cdata1.getDataType()==DataType.SCALAR)? cdata1 : new CNodeUnary(cdata1, UnaryType.LOOKUP0),
						(cdata2.getDataType()==DataType.SCALAR)? cdata2 : new CNodeUnary(cdata2, UnaryType.LOOKUP0), BinType.MULT);
				else {
					out = new CNodeBinary(cdata1, cdata2, BinType.DOT_PRODUCT);
					inHops2.put("X", hop.getInput().get(0));
				}
			}
		}
		else if(hop instanceof UnaryOp)
		{
			CNode cdata1 = tmp.get(hop.getInput().get(0).getHopID());
			
			// if one input is a matrix then we need to do vector by scalar operations
			if(hop.getInput().get(0).getDim1() > 1 && hop.getInput().get(0).getDim2() > 1 ) 
			{
				if( HopRewriteUtils.isUnary(hop, SUPPORTED_VECT_UNARY) ) {
					String opname = "VECT_"+((UnaryOp)hop).getOp().name();
					out = new CNodeUnary(cdata1, UnaryType.valueOf(opname));
					if( cdata1 instanceof CNodeData && inHops2.isEmpty() )
						inHops2.put("X", hop.getInput().get(0));
				}
				else 
					throw new RuntimeException("Unsupported unary matrix "
							+ "operation: " + ((UnaryOp)hop).getOp().name());
			}
			else //general scalar case
			{
				cdata1 = TemplateUtils.wrapLookupIfNecessary(cdata1, hop.getInput().get(0));
				
				String primitiveOpName = ((UnaryOp)hop).getOp().toString();
				out = new CNodeUnary(cdata1, UnaryType.valueOf(primitiveOpName));
			}
		}
		else if(HopRewriteUtils.isBinary(hop, OpOp2.CBIND)) 
		{
			//special case for cbind with zeros
			CNode cdata1 = tmp.get(hop.getInput().get(0).getHopID());
			out = new CNodeUnary(cdata1, UnaryType.CBIND0);
		}
		else if(hop instanceof BinaryOp)
		{
			CNode cdata1 = tmp.get(hop.getInput().get(0).getHopID());
			CNode cdata2 = tmp.get(hop.getInput().get(1).getHopID());
			
			// if one input is a matrix then we need to do vector by scalar operations
			if( (hop.getInput().get(0).getDim1() > 1 && hop.getInput().get(0).getDim2() > 1)
				|| (hop.getInput().get(1).getDim1() > 1 && hop.getInput().get(1).getDim2() > 1))
			{
				if( HopRewriteUtils.isBinary(hop, SUPPORTED_VECT_BINARY) ) {
					if( TemplateUtils.isMatrix(cdata1) && TemplateUtils.isMatrix(cdata2) ) {
						String opname = "VECT_"+((BinaryOp)hop).getOp().name();
						out = new CNodeBinary(cdata1, cdata2, BinType.valueOf(opname));
					}
					else {
						String opname = "VECT_"+((BinaryOp)hop).getOp().name()+"_SCALAR";
						if( TemplateUtils.isColVector(cdata1) )
							cdata1 = new CNodeUnary(cdata1, UnaryType.LOOKUP_R);
						if( TemplateUtils.isColVector(cdata2) )
							cdata2 = new CNodeUnary(cdata2, UnaryType.LOOKUP_R);
						out = new CNodeBinary(cdata1, cdata2, BinType.valueOf(opname));
					}
					if( cdata1 instanceof CNodeData && inHops2.isEmpty() 
						&& !(cdata1.getDataType()==DataType.SCALAR) ) {
						inHops2.put("X", hop.getInput().get(0));
					}
				}
				else 
					throw new RuntimeException("Unsupported binary matrix "
							+ "operation: " + ((BinaryOp)hop).getOp().name());
			}
			else //one input is a vector/scalar other is a scalar
			{
				String primitiveOpName = ((BinaryOp)hop).getOp().toString();
				if( TemplateUtils.isColVector(cdata1) )
					cdata1 = new CNodeUnary(cdata1, UnaryType.LOOKUP_R);
				if( TemplateUtils.isColVector(cdata2) //vector or vector can be inferred from lhs
					|| (TemplateUtils.isColVector(hop.getInput().get(0)) && cdata2 instanceof CNodeData
						&& hop.getInput().get(1).getDataType().isMatrix()))
					cdata2 = new CNodeUnary(cdata2, UnaryType.LOOKUP_R);
				out = new CNodeBinary(cdata1, cdata2, BinType.valueOf(primitiveOpName));	
			}
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
					TernaryType.valueOf(top.getOp().toString()));
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
		
		if( out == null ) {
			throw new RuntimeException(hop.getHopID()+" "+hop.getOpString());
		}
		
		if( out.getDataType().isMatrix() ) {
			out.setNumRows(hop.getDim1());
			out.setNumCols(hop.getDim2());
		}
		
		tmp.put(hop.getHopID(), out);
	}
	
	/**
	 * Comparator to order input hops of the row aggregate template. We try 
	 * to order matrices-vectors-scalars via sorting by number of cells but 
	 * we keep the given main input always at the first position.
	 */
	public static class HopInputComparator implements Comparator<Hop> 
	{
		private final Hop _X;
		
		public HopInputComparator(Hop X) {
			_X = X;
		}
		
		@Override
		public int compare(Hop h1, Hop h2) {
			long ncells1 = h1.getDataType()==DataType.SCALAR ? Long.MIN_VALUE : 
				(h1==_X) ? Long.MAX_VALUE : 
				h1.dimsKnown() ? h1.getDim1()*h1.getDim2() : Long.MAX_VALUE-1;
			long ncells2 = h2.getDataType()==DataType.SCALAR ? Long.MIN_VALUE : 
				(h2==_X) ? Long.MAX_VALUE : 
				h2.dimsKnown() ? h2.getDim1()*h2.getDim2() : Long.MAX_VALUE-1;
			return (ncells1 > ncells2) ? -1 : (ncells1 < ncells2) ? 1 : 0; 
		}
	}
}
