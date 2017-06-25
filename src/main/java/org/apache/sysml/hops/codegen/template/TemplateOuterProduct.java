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

import org.apache.sysml.hops.AggBinaryOp;
import org.apache.sysml.hops.AggUnaryOp;
import org.apache.sysml.hops.BinaryOp;
import org.apache.sysml.hops.Hop;
import org.apache.sysml.hops.UnaryOp;
import org.apache.sysml.hops.Hop.AggOp;
import org.apache.sysml.hops.Hop.Direction;
import org.apache.sysml.hops.Hop.OpOp2;
import org.apache.sysml.hops.codegen.cplan.CNode;
import org.apache.sysml.hops.codegen.cplan.CNodeBinary;
import org.apache.sysml.hops.codegen.cplan.CNodeBinary.BinType;
import org.apache.sysml.hops.codegen.cplan.CNodeUnary;
import org.apache.sysml.hops.codegen.cplan.CNodeUnary.UnaryType;
import org.apache.sysml.hops.codegen.template.CPlanMemoTable.MemoTableEntry;
import org.apache.sysml.hops.codegen.cplan.CNodeData;
import org.apache.sysml.hops.codegen.cplan.CNodeOuterProduct;
import org.apache.sysml.hops.codegen.cplan.CNodeTpl;
import org.apache.sysml.hops.rewrite.HopRewriteUtils;
import org.apache.sysml.runtime.codegen.SpoofOuterProduct.OutProdType;
import org.apache.sysml.runtime.matrix.data.Pair;

public class TemplateOuterProduct extends TemplateBase {
	
	public TemplateOuterProduct() {
		super(TemplateType.OuterProdTpl);
	}
	
	public TemplateOuterProduct(boolean closed) {
		super(TemplateType.OuterProdTpl, closed);
	}

	@Override
	public boolean open(Hop hop) {
		//open on outer product like matrix mult (output larger than common dim)
		return HopRewriteUtils.isOuterProductLikeMM(hop)
			&& hop.getDim1()>256 && hop.getDim2() > 256;
	}

	@Override
	public boolean fuse(Hop hop, Hop input) {
		return !isClosed() 
			&&((hop instanceof UnaryOp && TemplateUtils.isOperationSupported(hop))  
			|| (hop instanceof BinaryOp && TemplateUtils.isOperationSupported(hop)
				&& (TemplateUtils.isBinaryMatrixColVector(hop) || HopRewriteUtils.isBinaryMatrixScalarOperation(hop)
				|| (HopRewriteUtils.isBinaryMatrixMatrixOperation(hop) && HopRewriteUtils.isBinary(hop, OpOp2.MULT, OpOp2.DIV)) )) 
			|| HopRewriteUtils.isTransposeOperation(hop) 
			|| (hop instanceof AggBinaryOp && !HopRewriteUtils.isOuterProductLikeMM(hop))
			|| (hop instanceof AggUnaryOp && ((AggUnaryOp)hop).getDirection()==Direction.RowCol));
	}

	@Override
	public boolean merge(Hop hop, Hop input) {
		return !isClosed() && 
			(TemplateUtils.isBinaryMatrixRowVector(hop)
			|| HopRewriteUtils.isBinaryMatrixScalarOperation(hop));
	}

	@Override
	public CloseType close(Hop hop) {
		// close on second matrix multiply (after open) or unary aggregate
		if( hop instanceof AggUnaryOp && HopRewriteUtils.isOuterProductLikeMM(hop.getInput().get(0))
			|| (hop instanceof AggBinaryOp && (HopRewriteUtils.isOuterProductLikeMM(hop.getInput().get(0))
				|| HopRewriteUtils.isOuterProductLikeMM(hop.getInput().get(1)))) )
			return CloseType.CLOSED_INVALID;
		else if( (hop instanceof AggUnaryOp) 
			|| (hop instanceof AggBinaryOp && !HopRewriteUtils.isOuterProductLikeMM(hop) 
					&& !HopRewriteUtils.isTransposeOperation(hop.getParent().get(0)))
			|| (HopRewriteUtils.isTransposeOperation(hop) && hop.getInput().get(0) instanceof AggBinaryOp
					&& !HopRewriteUtils.isOuterProductLikeMM(hop.getInput().get(0)) ))
			return CloseType.CLOSED_VALID;
		else
			return CloseType.OPEN;
	}

	public Pair<Hop[], CNodeTpl> constructCplan(Hop hop, CPlanMemoTable memo, boolean compileLiterals) 
	{
		//recursively process required cplan output
		HashSet<Hop> inHops = new HashSet<Hop>();
		HashMap<String,Hop> inHops2 = new HashMap<String, Hop>();
		HashMap<Long, CNode> tmp = new HashMap<Long, CNode>();
		hop.resetVisitStatus();
		rConstructCplan(hop, memo, tmp, inHops, inHops2, compileLiterals);
		hop.resetVisitStatus();
		
		//reorder inputs (ensure matrix is first input)
		Hop X = inHops2.get("_X");
		Hop U = inHops2.get("_U");
		Hop V = inHops2.get("_V");
		LinkedList<Hop> sinHops = new LinkedList<Hop>(inHops);
		sinHops.remove(V); sinHops.addFirst(V);
		sinHops.remove(U); sinHops.addFirst(U);
		sinHops.remove(X); sinHops.addFirst(X);
		
		//construct template node
		ArrayList<CNode> inputs = new ArrayList<CNode>();
		for( Hop in : sinHops )
			if( in != null )
				inputs.add(tmp.get(in.getHopID()));

		CNode output = tmp.get(hop.getHopID());
		CNodeOuterProduct tpl = new CNodeOuterProduct(inputs, output);
		tpl.setOutProdType(TemplateUtils.getOuterProductType(X, U, V, hop));
		tpl.setTransposeOutput(!HopRewriteUtils.isTransposeOperation(hop)
			&& tpl.getOutProdType()==OutProdType.LEFT_OUTER_PRODUCT);
		tpl.setBeginLine(hop.getBeginLine());
		
		
		return new Pair<Hop[],CNodeTpl>(sinHops.toArray(new Hop[0]), tpl);
	}
	
	private void rConstructCplan(Hop hop, CPlanMemoTable memo, HashMap<Long, CNode> tmp, HashSet<Hop> inHops, HashMap<String, Hop> inHops2, boolean compileLiterals) 
	{
		//memoization for common subexpression elimination and to avoid redundant work 
		if( tmp.containsKey(hop.getHopID()) )
			return;
		
		//recursively process required childs
		MemoTableEntry me = memo.getBest(hop.getHopID(), TemplateType.OuterProdTpl);
		for( int i=0; i<hop.getInput().size(); i++ ) {
			Hop c = hop.getInput().get(i);
			if( me.isPlanRef(i) )
				rConstructCplan(c, memo, tmp, inHops, inHops2, compileLiterals);
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
			String primitiveOpName = ((UnaryOp)hop).getOp().toString();
			out = new CNodeUnary(cdata1, UnaryType.valueOf(primitiveOpName));
		}
		else if(hop instanceof BinaryOp)
		{
			CNode cdata1 = tmp.get(hop.getInput().get(0).getHopID());
			CNode cdata2 = tmp.get(hop.getInput().get(1).getHopID());
			String primitiveOpName = ((BinaryOp)hop).getOp().toString();
			
			if( HopRewriteUtils.isEqualSize(hop.getInput().get(0), hop.getInput().get(1)) ) {
				Hop main = hop.getInput().get((cdata1 instanceof CNodeData) ? 0 : 1);
				inHops2.put("_X", main);
			}
			
			//add lookups if required
			cdata1 = TemplateUtils.wrapLookupIfNecessary(cdata1, hop.getInput().get(0));
			cdata2 = TemplateUtils.wrapLookupIfNecessary(cdata2, hop.getInput().get(1));
			
			out = new CNodeBinary(cdata1, cdata2, BinType.valueOf(primitiveOpName));
		}
		else if(hop instanceof AggBinaryOp)
		{
			CNode cdata1 = tmp.get(hop.getInput().get(0).getHopID());
			CNode cdata2 = tmp.get(hop.getInput().get(1).getHopID());
			
			//handle transpose in outer or final product
			cdata1 = TemplateUtils.skipTranspose(cdata1, hop.getInput().get(0), tmp, compileLiterals);
			cdata2 = TemplateUtils.skipTranspose(cdata2, hop.getInput().get(1), tmp, compileLiterals);
			
			//outer product U%*%t(V), see open
			if( HopRewriteUtils.isOuterProductLikeMM(hop) )
			{
				//keep U and V for later reference
				inHops2.put("_U", hop.getInput().get(0));
				if( HopRewriteUtils.isTransposeOperation(hop.getInput().get(1)) )
					inHops2.put("_V", hop.getInput().get(1).getInput().get(0));
				else
					inHops2.put("_V", hop.getInput().get(1));
				
				out = new CNodeBinary(cdata1, cdata2, BinType.DOT_PRODUCT);
			}
			//final left/right matrix mult, see close
			else {
				if( cdata1.getDataType().isScalar() )
					out = new CNodeBinary(cdata2, cdata1, BinType.VECT_MULT_ADD);	
				else
					out = new CNodeBinary(cdata1, cdata2, BinType.VECT_MULT_ADD);	
			}
		}
		else if( HopRewriteUtils.isTransposeOperation(hop) ) 
		{
			out = tmp.get(hop.getInput().get(0).getHopID());	
		}
		else if( hop instanceof AggUnaryOp && ((AggUnaryOp)hop).getOp() == AggOp.SUM
			&& ((AggUnaryOp)hop).getDirection() == Direction.RowCol )
		{
			out = tmp.get(hop.getInput().get(0).getHopID());
		}
		
		tmp.put(hop.getHopID(), out);
	}

	protected static MemoTableEntry dropAlternativePlan(CPlanMemoTable memo, MemoTableEntry me1, MemoTableEntry me2) {
		//if there are two alternative sub plans with references to disjoint outer product plans
		//drop the one that would render the other invalid
		if( me1.countPlanRefs()==1 && me2.countPlanRefs()==1 
			&& me1.getPlanRefIndex() != me2.getPlanRefIndex() ) 
		{
			Hop c1 = memo._hopRefs.get(me1.input(me1.getPlanRefIndex()));
			Hop c2 = memo._hopRefs.get(me2.input(me2.getPlanRefIndex()));
			
			if( memo.contains(c1.getHopID(), TemplateType.OuterProdTpl) 
				&& memo.contains(c2.getHopID(), TemplateType.OuterProdTpl) )
			{
				if( HopRewriteUtils.isBinaryMatrixMatrixOperation(c1) 
					&& HopRewriteUtils.isBinary(c1, OpOp2.MULT, OpOp2.DIV) )
					return me1;
				if( HopRewriteUtils.isBinaryMatrixMatrixOperation(c2) 
					&& HopRewriteUtils.isBinary(c2, OpOp2.MULT, OpOp2.DIV) )
					return me2;
			}
		}
		return null;
	}
}
