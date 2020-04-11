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

package org.apache.sysds.hops.codegen.template;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.LinkedList;

import org.apache.sysds.common.Types.AggOp;
import org.apache.sysds.common.Types.Direction;
import org.apache.sysds.common.Types.OpOp2;
import org.apache.sysds.hops.AggBinaryOp;
import org.apache.sysds.hops.AggUnaryOp;
import org.apache.sysds.hops.BinaryOp;
import org.apache.sysds.hops.Hop;
import org.apache.sysds.hops.UnaryOp;
import org.apache.sysds.hops.codegen.cplan.CNode;
import org.apache.sysds.hops.codegen.cplan.CNodeBinary;
import org.apache.sysds.hops.codegen.cplan.CNodeData;
import org.apache.sysds.hops.codegen.cplan.CNodeOuterProduct;
import org.apache.sysds.hops.codegen.cplan.CNodeTpl;
import org.apache.sysds.hops.codegen.cplan.CNodeUnary;
import org.apache.sysds.hops.codegen.cplan.CNodeBinary.BinType;
import org.apache.sysds.hops.codegen.cplan.CNodeUnary.UnaryType;
import org.apache.sysds.hops.codegen.template.CPlanMemoTable.MemoTableEntry;
import org.apache.sysds.hops.rewrite.HopRewriteUtils;
import org.apache.sysds.lops.MMTSJ;
import org.apache.sysds.runtime.codegen.SpoofOuterProduct.OutProdType;
import org.apache.sysds.runtime.matrix.data.Pair;

public class TemplateOuterProduct extends TemplateBase {

	MMTSJ.MMTSJType mmtsj = MMTSJ.MMTSJType.NONE;

	public TemplateOuterProduct() {
		super(TemplateType.OUTER);
	}
	
	public TemplateOuterProduct(CloseType ctype) {
		super(TemplateType.OUTER, ctype);
	}

	@Override
	public boolean open(Hop hop) {
		//open on (1) outer product like matrix mult (output larger than common dim)
		//or (2) binary outer operation
		return (HopRewriteUtils.isOuterProductLikeMM(hop)
				|| HopRewriteUtils.isOuterBinary(hop))
			&& hop.getDim1() > 256 && hop.getDim2() > 256;
	}

	@Override
	public boolean fuse(Hop hop, Hop input) {
		return !isClosed()
			&& ((hop instanceof UnaryOp && TemplateUtils.isOperationSupported(hop))
				|| (hop instanceof BinaryOp && TemplateUtils.isOperationSupported(hop)
					&& (TemplateUtils.isBinaryMatrixColVector(hop)
					|| HopRewriteUtils.isBinaryMatrixScalarOperation(hop)
					|| HopRewriteUtils.isBinaryMatrixMatrixOperation(hop)
					|| TemplateUtils.isBinaryMatrixRowVector(hop)))
				|| (HopRewriteUtils.isTransposeOperation(hop) && input instanceof AggBinaryOp
					 && !HopRewriteUtils.isOuterProductLikeMM(input))
				|| (hop instanceof AggBinaryOp && !HopRewriteUtils.isOuterProductLikeMM(hop)
					 && TemplateUtils.containsOuterProduct(input, HopRewriteUtils.getOtherInput(hop, input)))
				|| (hop instanceof AggUnaryOp && ((AggUnaryOp)hop).getDirection()==Direction.RowCol));
	}

	@Override
	public boolean merge(Hop hop, Hop input) {
		return !isClosed() && 
			(TemplateUtils.isBinaryMatrixRowVector(hop)
			|| HopRewriteUtils.isBinaryMatrixScalarOperation(hop)
			|| (HopRewriteUtils.isBinary(hop, OpOp2.MULT) 
				&& HopRewriteUtils.isBinarySparseSafe(input)
				&& !TemplateUtils.containsOuterProduct(input)));
	}

	@Override
	public CloseType close(Hop hop) {
		// close on second matrix multiply (after open) or unary aggregate
		if( (hop instanceof AggUnaryOp && (HopRewriteUtils.isOuterProductLikeMM(hop.getInput().get(0))
				|| !HopRewriteUtils.isBinarySparseSafe(hop.getInput().get(0))))
			|| (hop instanceof AggBinaryOp && (HopRewriteUtils.isOuterProductLikeMM(hop.getInput().get(0))
				|| HopRewriteUtils.isOuterProductLikeMM(hop.getInput().get(1))
				|| (!HopRewriteUtils.isOuterProductLikeMM(hop)
					&& !HopRewriteUtils.isBinarySparseSafe(HopRewriteUtils.getLargestInput(hop))))) )
 			return CloseType.CLOSED_INVALID;
		else if( (hop instanceof AggUnaryOp) 
			|| (hop instanceof AggBinaryOp && !HopRewriteUtils.isOuterProductLikeMM(hop) 
					&& !HopRewriteUtils.isTransposeOperation(hop.getParent().get(0)))
			|| (HopRewriteUtils.isTransposeOperation(hop) && hop.getInput().get(0) instanceof AggBinaryOp
					&& !HopRewriteUtils.isOuterProductLikeMM(hop.getInput().get(0)) ))
			return CloseType.CLOSED_VALID;
		else if( HopRewriteUtils.isBinaryMatrixMatrixOperation(hop)
			&& HopRewriteUtils.isBinary(hop, OpOp2.MULT, OpOp2.DIV) )
			return CloseType.OPEN_VALID;
		else
			return CloseType.OPEN_INVALID;
	}

	@Override
	public Pair<Hop[], CNodeTpl> constructCplan(Hop hop, CPlanMemoTable memo, boolean compileLiterals) 
	{
		//recursively process required cplan output
		HashSet<Hop> inHops = new HashSet<>();
		HashMap<String,Hop> inHops2 = new HashMap<>();
		HashMap<Long, CNode> tmp = new HashMap<>();
		hop.resetVisitStatus();
		rConstructCplan(hop, memo, tmp, inHops, inHops2, compileLiterals);
		hop.resetVisitStatus();

		// Remove CNodes that would produce the following unnecessary
		// line of code: "double tmpX = (a != 0) ? 1 : 0"
		// This is unnecessary with SpoofOuterProduct, since code for tmpX==0
		// is not invoked anyway.
		long outputHopID = hop.getHopID();
		if(hop instanceof BinaryOp)
			outputHopID = TemplateUtils.skipConditionalInOuterProduct(hop, tmp, inHops);

		//reorder inputs (ensure matrix is first input)
		Hop X = inHops2.get("_X");
		Hop U = inHops2.get("_U");
		Hop V = inHops2.get("_V");
		LinkedList<Hop> sinHops = new LinkedList<>(inHops);

		// order of adds and removes is important here (all removes before adds)
		sinHops.remove(V);
		sinHops.remove(U);
		sinHops.remove(X);
		sinHops.addFirst(V);
		sinHops.addFirst(U);
		sinHops.addFirst(X);

		//construct template node
		ArrayList<CNode> inputs = new ArrayList<>();
		for( Hop in : sinHops )
			if( in != null )
				inputs.add(tmp.get(in.getHopID()));

		CNode output = tmp.get(outputHopID);
		CNodeOuterProduct tpl = new CNodeOuterProduct(inputs, output, mmtsj);
		tpl.setOutProdType(TemplateUtils.getOuterProductType(X, U, V, hop));
		tpl.setTransposeOutput(!HopRewriteUtils.isTransposeOperation(hop)
			&& tpl.getOutProdType()==OutProdType.LEFT_OUTER_PRODUCT);
		tpl.setBeginLine(hop.getBeginLine());
		
		return new Pair<>(sinHops.toArray(new Hop[0]), tpl);
	}
	
	private void rConstructCplan(Hop hop, CPlanMemoTable memo, HashMap<Long, CNode> tmp, HashSet<Hop> inHops, HashMap<String, Hop> inHops2, boolean compileLiterals) 
	{
		//memoization for common subexpression elimination and to avoid redundant work 
		if( tmp.containsKey(hop.getHopID()) )
			return;
		
		//recursively process required children
		MemoTableEntry me = memo.getBest(hop.getHopID(), TemplateType.OUTER, TemplateType.CELL);
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
			String primitiveOpName = ((UnaryOp)hop).getOp().name();
			out = new CNodeUnary(cdata1, UnaryType.valueOf(primitiveOpName));
		}
		else if(hop instanceof BinaryOp)
		{
			CNode cdata1 = tmp.get(hop.getInput().get(0).getHopID());
			CNode cdata2 = tmp.get(hop.getInput().get(1).getHopID());
			String primitiveOpName = ((BinaryOp)hop).getOp().name();
			
			if( HopRewriteUtils.isBinarySparseSafe(hop) ) {
				if( TemplateUtils.isMatrix(hop.getInput().get(0)) && cdata1 instanceof CNodeData )
					inHops2.put("_X", hop.getInput().get(0));
				if( TemplateUtils.isMatrix(hop.getInput().get(1)) && cdata2 instanceof CNodeData )
					inHops2.put("_X", hop.getInput().get(1));
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
				if( HopRewriteUtils.isTransposeOperation(hop.getInput().get(0)) )
					inHops2.put("_U", hop.getInput().get(0).getInput().get(0));
				else
					inHops2.put("_U", hop.getInput().get(0));

				if( HopRewriteUtils.isTransposeOperation(hop.getInput().get(1)) )
					inHops2.put("_V", hop.getInput().get(1).getInput().get(0));
				else
					inHops2.put("_V", hop.getInput().get(1));
				
				/* TODO find out why we need to propagate this at all
				 * (maybe a more generic treatment will yield the same result?)
				 * At the moment this is needed to decide whether we need to add
				 * a duplicate input (CNodeOuterProduct constructor) and if we
				 * need to fill in a transpose in rConstructModifiedHopDag in SpoofCompiler.
				 */
				mmtsj = ((AggBinaryOp) hop).checkTransposeSelf(); //determine tsmm pattern

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

	public static MemoTableEntry dropAlternativePlan(CPlanMemoTable memo, MemoTableEntry me1, MemoTableEntry me2) {
		//if there are two alternative sub plans with references to disjoint outer product plans
		//drop the one that would render the other invalid
		if( me1.countPlanRefs()==1 && me2.countPlanRefs()==1 
			&& me1.getPlanRefIndex() != me2.getPlanRefIndex() ) 
		{
			Hop c1 = memo._hopRefs.get(me1.input(me1.getPlanRefIndex()));
			Hop c2 = memo._hopRefs.get(me2.input(me2.getPlanRefIndex()));
			
			if( memo.contains(c1.getHopID(), TemplateType.OUTER) 
				&& memo.contains(c2.getHopID(), TemplateType.OUTER) )
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
