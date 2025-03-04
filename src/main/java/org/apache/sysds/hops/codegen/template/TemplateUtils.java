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

import java.util.HashMap;
import java.util.HashSet;
import java.util.LinkedList;
import java.util.List;
import java.util.Map;
import java.util.Set;

import org.apache.commons.lang3.ArrayUtils;
import org.apache.commons.lang3.mutable.MutableInt;
import org.apache.commons.lang3.tuple.Pair;
import org.apache.sysds.hops.AggBinaryOp;
import org.apache.sysds.hops.AggUnaryOp;
import org.apache.sysds.hops.BinaryOp;
import org.apache.sysds.hops.Hop;
import org.apache.sysds.hops.IndexingOp;
import org.apache.sysds.hops.LiteralOp;
import org.apache.sysds.hops.ParameterizedBuiltinOp;
import org.apache.sysds.hops.TernaryOp;
import org.apache.sysds.hops.UnaryOp;
import org.apache.sysds.common.Types.AggOp;
import org.apache.sysds.common.Types.Direction;
import org.apache.sysds.common.Types.OpOp1;
import org.apache.sysds.common.Types.OpOp2;
import org.apache.sysds.common.Types.OpOpDnn;
import org.apache.sysds.common.Types.OpOpN;
import org.apache.sysds.hops.codegen.SpoofCompiler;
import org.apache.sysds.hops.codegen.cplan.CNode;
import org.apache.sysds.hops.codegen.cplan.CNodeBinary;
import org.apache.sysds.hops.codegen.cplan.CNodeData;
import org.apache.sysds.hops.codegen.cplan.CNodeNary;
import org.apache.sysds.hops.codegen.cplan.CNodeRow;
import org.apache.sysds.hops.codegen.cplan.CNodeTernary;
import org.apache.sysds.hops.codegen.cplan.CNodeTpl;
import org.apache.sysds.hops.codegen.cplan.CNodeUnary;
import org.apache.sysds.hops.codegen.cplan.CNodeBinary.BinType;
import org.apache.sysds.hops.codegen.cplan.CNodeTernary.TernaryType;
import org.apache.sysds.hops.codegen.cplan.CNodeUnary.UnaryType;
import org.apache.sysds.hops.codegen.template.CPlanMemoTable.MemoTableEntry;
import org.apache.sysds.hops.codegen.template.TemplateBase.CloseType;
import org.apache.sysds.hops.codegen.template.TemplateBase.TemplateType;
import org.apache.sysds.hops.rewrite.HopRewriteUtils;
import org.apache.sysds.common.Types.DataType;
import org.apache.sysds.runtime.codegen.SpoofCellwise.CellType;
import org.apache.sysds.runtime.codegen.SpoofOuterProduct.OutProdType;
import org.apache.sysds.runtime.codegen.SpoofRowwise.RowType;
import org.apache.sysds.runtime.controlprogram.parfor.util.IDSequence;
import org.apache.sysds.runtime.util.UtilFunctions;

public class TemplateUtils 
{
	public static final TemplateBase[] TEMPLATES = new TemplateBase[]{
		new TemplateRow(), new TemplateCell(), new TemplateOuterProduct()};
	//note: multiagg not included because it's a composite template
	
	public static boolean isVector(Hop hop) {
		return (hop.getDataType() == DataType.MATRIX 
			&& (hop.getDim1() != 1 && hop.getDim2() == 1 
			  || hop.getDim1() == 1 && hop.getDim2() != 1 ) );
	}
	
	public static boolean isColVector(Hop hop) {
		return (hop.getDataType() == DataType.MATRIX 
			&& hop.getDim1() != 1 && hop.getDim2() == 1 );
	}
	
	public static boolean isColVector(CNode hop) {
		return (hop.getDataType() == DataType.MATRIX 
			&& hop.getNumRows() != 1 && hop.getNumCols() == 1);
	}
	
	public static boolean isRowVector(CNode hop) {
		return (hop.getDataType() == DataType.MATRIX 
			&& hop.getNumRows() == 1 && hop.getNumCols() != 1);
	}
	
	public static boolean isMatrix(CNode hop) {
		return (hop.getDataType() == DataType.MATRIX 
			&& hop.getNumRows() != 1 && hop.getNumCols() != 1);
	}
	
	public static CNode wrapLookupIfNecessary(CNode node, Hop hop) {
		return wrapLookupIfNecessary(node, hop, false);
	}
	
	public static CNode wrapLookupIfNecessary(CNode node, Hop hop, boolean rowTpl) {
		CNode ret = node;
		if( isColVector(node) )
			ret = new CNodeUnary(node, UnaryType.LOOKUP_R);
		else if( isRowVector(node) )
			ret = new CNodeUnary(node, UnaryType.LOOKUP_C);
		else if( node instanceof CNodeData && hop.getDataType().isMatrix() )
			ret = rowTpl ? node : new CNodeUnary(node, UnaryType.LOOKUP_RC);
		return ret;
	}
	
	public static boolean isMatrix(Hop hop) {
		return (hop.getDataType() == DataType.MATRIX && hop.getDim1() != 1 && hop.getDim2()!=1);
	}
	
	public static boolean isVectorOrScalar(Hop hop) {
		return hop.dimsKnown() && (hop.getDataType() == DataType.SCALAR || isVector(hop) );
	}
	
	public static boolean isBinaryMatrixRowVector(Hop hop) {
		if( !(hop instanceof BinaryOp) )
			return false;
		Hop left = hop.getInput().get(0);
		Hop right = hop.getInput().get(1);
		return left.dimsKnown() && right.dimsKnown() 
			&& left.getDataType().isMatrix() && right.getDataType().isMatrix()
			&& left.getDim1() > right.getDim1();
	}
	
	public static boolean isBinaryMatrixColVector(Hop hop) {
		if( !(hop instanceof BinaryOp) )
			return false;
		Hop left = hop.getInput().get(0);
		Hop right = hop.getInput().get(1);
		return left.dimsKnown() && right.dimsKnown() 
			&& left.getDataType().isMatrix() && right.getDataType().isMatrix()
			&& left.getDim2() > right.getDim2();
	}
	
	public static boolean hasMatrixInput( Hop hop ) {
		for( Hop c : hop.getInput() )
			if( isMatrix(c) )
				return true;
		return false;
	}

	public static boolean isOperationSupported(Hop h) {
		if(h instanceof  UnaryOp)
			return UnaryType.contains(((UnaryOp)h).getOp().name());
		else if(h instanceof BinaryOp && !((BinaryOp)h).isOuter())
			return BinType.contains(((BinaryOp)h).getOp().name());
		else if(h instanceof TernaryOp)
			return TernaryType.contains(((TernaryOp)h).getOp().name());
		else if(h instanceof ParameterizedBuiltinOp) 
			return TernaryType.contains(((ParameterizedBuiltinOp)h).getOp().name());
		return false;
	}
	
	public static TemplateBase createTemplate(TemplateType type) {
		return createTemplate(type, CloseType.OPEN_VALID);
	}
	
	public static TemplateBase createTemplate(TemplateType type, CloseType ctype) {
		TemplateBase tpl = null;
		switch( type ) {
			case CELL: tpl = new TemplateCell(ctype); break;
			case ROW: tpl = new TemplateRow(ctype); break;
			case MAGG: tpl = new TemplateMultiAgg(ctype); break;
			case OUTER: tpl = new TemplateOuterProduct(ctype); break;
		}
		return tpl;
	}
	
	public static TemplateBase[] createCompatibleTemplates(TemplateType type, CloseType ctype) {
		TemplateBase[] tpl = null;
		switch( type ) {
			case CELL: tpl = new TemplateBase[]{new TemplateCell(ctype), new TemplateRow(ctype)}; break;
			case ROW: tpl = new TemplateBase[]{new TemplateRow(ctype)}; break;
			case MAGG: tpl = new TemplateBase[]{new TemplateMultiAgg(ctype)}; break;
			case OUTER: tpl = new TemplateBase[]{new TemplateOuterProduct(ctype)}; break;
		}
		return tpl;
	}
	
	public static CellType getCellType(Hop hop) {
		if( hop instanceof AggBinaryOp )
			return CellType.FULL_AGG;
		else if( hop instanceof AggUnaryOp )
			switch( ((AggUnaryOp)hop).getDirection() ) {
				case Row: return CellType.ROW_AGG;
				case Col: return CellType.COL_AGG;
				case RowCol: return CellType.FULL_AGG;
			}
		return CellType.NO_AGG;
	}
	
	public static RowType getRowType(Hop output, Hop... inputs) {
		Hop X = inputs[0];
		Hop B1 = (inputs.length>1) ? inputs[1] : null;
		if( (X!=null && HopRewriteUtils.isEqualSize(output, X)) || X==null || !X.dimsKnown() )
			return RowType.NO_AGG;
		else if( ((B1!=null && output.getDim1()==X.getDim1() && output.getDim2()==B1.getDim2())
			|| (output instanceof IndexingOp && HopRewriteUtils.isColumnRangeIndexing((IndexingOp)output)))
			&& !(output instanceof AggBinaryOp && HopRewriteUtils.isTransposeOfItself(output.getInput().get(0),X)) )
			return RowType.NO_AGG_B1;
		else if( output.getDim1()==X.getDim1() && (output.getDim2()==1)
			&& !(output instanceof AggBinaryOp && HopRewriteUtils
				.isTransposeOfItself(output.getInput().get(0),X)))
			return RowType.ROW_AGG;
		else if( output instanceof AggUnaryOp 
			&& ((AggUnaryOp)output).getDirection()==Direction.RowCol )
			return RowType.FULL_AGG;
		else if( output.getDim1()==X.getDim2() && output.getDim2()==1 )
			return RowType.COL_AGG_T;
		else if( output.getDim1()==1 && output.getDim2()==X.getDim2() )
			return RowType.COL_AGG;
		else if( B1 != null && output.getDim1()==X.getDim2() && output.getDim2()==B1.getDim2() )
			return RowType.COL_AGG_B1_T;
		else if( B1 != null && output.getDim1()==B1.getDim2() && output.getDim2()==X.getDim2())
			return RowType.COL_AGG_B1;
		else if( B1 != null && output.getDim1()==1 && B1.getDim2() == output.getDim2() )
			return RowType.COL_AGG_B1R;
		else if( X.getDim1() == output.getDim1() && X.getDim2() != output.getDim2() )
			return RowType.NO_AGG_CONST;
		else if( output.getDim1() == 1 && X.getDim2() != output.getDim2() )
			return RowType.COL_AGG_CONST;
		else
			throw new RuntimeException("Unknown row type for hop "+output.getHopID()+".");
	}
	
	public static AggOp getAggOp(Hop hop) {
		return (hop instanceof AggUnaryOp) ? ((AggUnaryOp)hop).getOp() :
			(hop instanceof AggBinaryOp) ? AggOp.SUM : null;
	}
	
	public static OutProdType getOuterProductType(Hop X, Hop U, Hop V, Hop out) {
		if( out.getDataType() == DataType.SCALAR ) 
			return OutProdType.AGG_OUTER_PRODUCT;
		else if( (out instanceof AggBinaryOp && (out.getInput().get(0) == U 
				|| HopRewriteUtils.isTransposeOperation(out.getInput().get(0))
				&& out.getInput().get(0).getInput().get(0) == U))
				|| HopRewriteUtils.isTransposeOperation(out) ) 
			return OutProdType.LEFT_OUTER_PRODUCT;
		else if( out instanceof AggBinaryOp && (out.getInput().get(1) == V
				|| HopRewriteUtils.isTransposeOperation(out.getInput().get(1)) 
				&& out.getInput().get(1).getInput().get(0) == V ) )
			return OutProdType.RIGHT_OUTER_PRODUCT;
		else if( out instanceof BinaryOp && HopRewriteUtils.isEqualSize(out.getInput().get(0), out.getInput().get(1)) )
			return OutProdType.CELLWISE_OUTER_PRODUCT;
		
		//should never come here
		throw new RuntimeException("Undefined outer product type for hop "+out.getHopID());
	}
	
	public static CNodeData getLiteral(CNode node) {
		return ((CNodeData) node).isLiteral() ? (CNodeData)node :
			createCNodeData(new LiteralOp(node.getVarname()), true);
	}
	
	public static boolean isLiteral(CNode node) {
		return node instanceof CNodeData && ((CNodeData)node).isLiteral();
	}
	
	public static boolean isLiteral(CNode node, String val) {
		return isLiteral(node) && ((CNodeData)node).getVarname().equals(val);
	}
	
	public static boolean isLookup(CNode node, boolean includeRC1) {
		return isUnary(node, UnaryType.LOOKUP_C, UnaryType.LOOKUP_RC)
			|| (includeRC1 && isUnary(node, UnaryType.LOOKUP_R))
			|| (includeRC1 && isTernary(node, TernaryType.LOOKUP_RC1));
	}
	
	public static boolean isUnary(CNode node, UnaryType...types) {
		return node instanceof CNodeUnary
			&& ArrayUtils.contains(types, ((CNodeUnary)node).getType());
	}

	public static boolean isUnaryRowAgg(CNode node) {
		return isUnary(node, UnaryType.ROW_MAXS, UnaryType.ROW_SUMS);
	}

	public static boolean isBinary(CNode node, BinType...types) {
		return node instanceof CNodeBinary
			&& ArrayUtils.contains(types, ((CNodeBinary)node).getType());
	}
	
	public static boolean rIsSparseSafeOnly(CNode node, BinType...types) {
		if( !(isBinary(node, types) || node instanceof CNodeData 
			|| (node instanceof CNodeUnary && ((((CNodeUnary)node).getType().isScalarLookup())
				|| ((CNodeUnary)node).getType().isSparseSafeScalar()
				|| ((CNodeUnary)node).getType()==UnaryType.POW2
				|| ((CNodeUnary)node).getType()==UnaryType.MULT2)) ))
			return false;
		boolean ret = true;
		for( CNode c : node.getInput() )
			ret &= rIsSparseSafeOnly(c, types);
		return ret;
	}
	
	public static boolean rContainsInput(CNode node, long hopID) {
		boolean ret = false;
		for( CNode c : node.getInput() )
			ret |= rContainsInput(c, hopID);
		if( node instanceof CNodeData )
			ret |= (((CNodeData)node).getHopID()==hopID);
		return ret;
	}
	
	public static boolean isTernary(CNode node, TernaryType...types) {
		return node instanceof CNodeTernary
			&& ArrayUtils.contains(types, ((CNodeTernary)node).getType());
	}

	public static CNodeData createCNodeData(Hop hop, boolean compileLiterals) {
		CNodeData cdata = new CNodeData(hop);
		cdata.setLiteral(hop instanceof LiteralOp && (compileLiterals 
				|| UtilFunctions.isIntegerNumber(((LiteralOp)hop).getStringValue())));
		return cdata;
	}

	public static CNode skipTranspose(CNode cdataOrig, Hop hop, HashMap<Long, CNode> tmp, boolean compileLiterals) {
		if( HopRewriteUtils.isTransposeOperation(hop) ) {
			CNode cdata = tmp.get(hop.getInput().get(0).getHopID());
			if( cdata == null ) { //never accessed
				cdata = TemplateUtils.createCNodeData(hop.getInput().get(0), compileLiterals);
				tmp.put(hop.getInput().get(0).getHopID(), cdata);
			}
			tmp.put(hop.getHopID(), cdata);
			return cdata;
		}
		else {
			return cdataOrig;
		}
	}

	public static LinkedList<Long> findRemovableConditionalPatternInOuterProduct(Hop hop) {
		LinkedList<Long> removableHopIDs = new LinkedList<>();
		if(((BinaryOp) hop).getOp() == OpOp2.MULT) {
			if (hop.getInput().get(0) instanceof BinaryOp &&
					((BinaryOp) hop.getInput().get(0)).getOp() == OpOp2.NOTEQUAL) {
				removableHopIDs.add(hop.getHopID());
				removableHopIDs.add(hop.getInput().get(0).getHopID());
				removableHopIDs.add(hop.getInput().get(0).getInput().get(0).getHopID());
				removableHopIDs.add(hop.getInput().get(0).getInput().get(1).getHopID());
			}
			else if (hop.getInput().get(1) instanceof BinaryOp &&
					((BinaryOp) hop.getInput().get(1)).getOp() == OpOp2.NOTEQUAL) {
				removableHopIDs.add(hop.getHopID());
				removableHopIDs.add(hop.getInput().get(1).getHopID());
				removableHopIDs.add(hop.getInput().get(1).getInput().get(0).getHopID());
				removableHopIDs.add(hop.getInput().get(1).getInput().get(1).getHopID());
			}
		}
		return removableHopIDs;
	}

	public static long skipConditionalInOuterProduct(Hop hop, HashMap<Long, CNode> tmp, HashSet<Hop> inHops) {
		LinkedList<Long> ll = findRemovableConditionalPatternInOuterProduct(hop);
		if(!ll.isEmpty()) {
			for (long hopid : ll) {
				boolean is_input = false;
				for (Hop in : inHops) {
					is_input = in.getHopID() == hopid;
					if (is_input)
						break;
				}
				if(!is_input)
					tmp.remove(hopid);
			}
			if(tmp.containsKey(hop.getInput().get(0).getHopID()))
				return hop.getInput().get(0).getHopID();
			else
				return hop.getInput().get(1).getHopID();
		}
		else return hop.getHopID();
	}

	public static boolean hasTransposeParentUnderOuterProduct(Hop hop) {
		for( Hop p : hop.getParent() )
			if( HopRewriteUtils.isTransposeOperation(p) )
				for( Hop p2 : p.getParent() )
					if( HopRewriteUtils.isOuterProductLikeMM(p2) )
						return true;
		return false;
	}

	public static boolean hasSingleOperation(CNodeTpl tpl) {
		CNode output = tpl.getOutput();
		return ((output instanceof CNodeUnary 
				&& !TemplateUtils.isUnary(output, 
					UnaryType.EXP, UnaryType.LOG, UnaryType.ROW_COUNTNNZS)) 
			|| (output instanceof CNodeBinary
				&& (!(TemplateUtils.isBinary(output, BinType.VECT_OUTERMULT_ADD) ||
					!TemplateUtils.isBinary(output, BinType.ROWMAXS_VECTMULT))))
			|| output instanceof CNodeTernary 
				&& ((CNodeTernary)output).getType() == TernaryType.IFELSE)
			&& hasOnlyDataNodeOrLookupInputs(output);
	}
	
	public static boolean isValidSingleOperation(Hop hop) {
		return HopRewriteUtils.isNary(hop, OpOpN.MIN, OpOpN.MAX, OpOpN.PLUS)
			|| HopRewriteUtils.isUnary(hop, OpOp1.EXP, OpOp1.LOG)
			|| HopRewriteUtils.isDnn(hop, OpOpDnn.BIASADD, OpOpDnn.BIASMULT);
	}
	
	public static boolean hasNoOperation(CNodeTpl tpl) {
		return tpl.getOutput() instanceof CNodeData 
			|| isLookup(tpl.getOutput(), true);
	}
	
	public static boolean hasOnlyDataNodeOrLookupInputs(CNode node) {
		boolean ret = true;
		for( CNode c : node.getInput() )
			ret &= (c instanceof CNodeData || (c instanceof CNodeUnary 
				&& (((CNodeUnary)c).getType()==UnaryType.LOOKUP0 
					|| ((CNodeUnary)c).getType()==UnaryType.LOOKUP_R 
					|| ((CNodeUnary)c).getType()==UnaryType.LOOKUP_RC)));
		return ret;
	}
	
	public static int determineMinVectorIntermediates(CNode node, CNode main) {
		node.resetVisitStatus();
		int count = -1;
		switch( SpoofCompiler.REG_ALLOC_POLICY ) {
			case HEURISTIC: {
				boolean unaryPipe = isUnaryOperatorPipeline(node);
				node.resetVisitStatus();
				count = unaryPipe ? getMaxVectorIntermediates(node) :
					countVectorIntermediates(node);
				break;
			}
			case EXACT_DYNAMIC_BUFF: {
				Map<Long, Set<Long>> parents = getAllParents(node);
				node.resetVisitStatus();
				count = getMaxLiveVectorIntermediates(
					node, main, parents, new HashSet<>());
				break;
			}
			case EXACT_STATIC_BUFF: {
				//init with basic heuristic
				boolean unaryPipe = isUnaryOperatorPipeline(node);
				node.resetVisitStatus();
				count = unaryPipe ? getMaxVectorIntermediates(node) :
					countVectorIntermediates(node);
				//reduce count and proof validity
				Map<Long, Set<Long>> parents = getAllParents(node);
				Map<Long, Pair<Long, MutableInt>> inUse = new HashMap<>(); //node ID, vector ID, num Refs
				Set<Long> inUse2 = new HashSet<>(); //for fast probes
				while( count > 0 && isValidNumVectorIntermediates(node, main, parents, inUse, inUse2, count-1) )
					count--;
				break;
			}
		}
		node.resetVisitStatus();
		return count;
	}
	
	public static boolean isUnaryOperatorPipeline(CNode node) {
		if( node.isVisited() ) {
			//second reference to vector intermediate invalidates a unary pipeline
			return !((node instanceof CNodeBinary && ((CNodeBinary)node).getType().isVectorPrimitive())
				|| (node instanceof CNodeTernary && ((CNodeTernary)node).getType().isVectorPrimitive())
				|| (node instanceof CNodeNary && ((CNodeNary)node).getType().isVectorPrimitive()));
		}
		boolean ret = true;
		for( CNode input : node.getInput() )
			ret &= isUnaryOperatorPipeline(input);
		node.setVisited();
		return ret;
	}
	
	public static int getMaxVectorIntermediates(CNode node) {
		if( node.isVisited() )
			return 0;
		int max = 0;
		for( CNode input : node.getInput() )
			max = Math.max(max, getMaxVectorIntermediates(input));
		max = Math.max(max, (node instanceof CNodeTernary
			&& ((CNodeTernary)node).getType().isVectorPrimitive()) ? 1 : 0);
		max = Math.max(max, (node instanceof CNodeBinary)? 
			(((CNodeBinary)node).getType().isVectorVectorPrimitive() ? 3 :
			((CNodeBinary)node).getType().isVectorScalarPrimitive() ? 2 :
			((CNodeBinary)node).getType().isVectorMatrixPrimitive() ? 1 : 0) : 0);
		max = Math.max(max, (node instanceof CNodeUnary 
			&& ((CNodeUnary)node).getType().isVectorScalarPrimitive()) ? 2 : 0);
		node.setVisited();
		return max;
	}
	
	public static int countVectorIntermediates(CNode node) {
		if( node.isVisited() )
			return 0;
		node.setVisited();
		//compute vector requirements over all inputs
		int ret = 0;
		for( CNode c : node.getInput() )
			ret += countVectorIntermediates(c);
		//compute vector requirements of current node
		int cntBin = (node instanceof CNodeBinary 
			&& ((CNodeBinary)node).getType().isVectorPrimitive()
			&& !((CNodeBinary)node).getType().name().endsWith("_ADD")) ? 1 : 0;
		int cntUn = (node instanceof CNodeUnary
				&& ((CNodeUnary)node).getType().isVectorScalarPrimitive()) ? 1 : 0;
		int cntTn = (node instanceof CNodeTernary
				&& ((CNodeTernary)node).getType().isVectorPrimitive()) ? 1 : 0;
		int cntNn = (node instanceof CNodeNary 
				&& ((CNodeNary)node).getType().isVectorPrimitive()) ? 1 : 0;
		return ret + cntBin + cntUn + cntTn + cntNn;
	}
	
	public static int getMaxLiveVectorIntermediates(CNode node, CNode main, Map<Long, Set<Long>> parents, Set<Pair<Long, Long>> stack) {
		if( node.isVisited() )
			return -1;
		//recursively process inputs
		int max = -1;
		for( CNode c : node.getInput() )
			max = Math.max(max, getMaxLiveVectorIntermediates(c, main, parents, stack));
		// add current node consumers
		if( !node.getDataType().isScalar() && parents.containsKey(node.getID())
			&& node != main ) {
			for( Long pID : parents.get(node.getID()) )
				stack.add(Pair.of(pID, node.getID()));
		}
		//get current maximum (distinct dep targets)
		max = Math.max(max, (int)stack.stream()
			.map(p -> p.getValue()).distinct().count());
		//remove input dependencies
		for( CNode c : node.getInput() )
			stack.remove(Pair.of(node.getID(), c.getID()));
		node.setVisited();
		return max;
	}
	
	public static boolean isValidNumVectorIntermediates(CNode node, CNode main, Map<Long, Set<Long>> parents, Map<Long, Pair<Long, MutableInt>> inUse, Set<Long> inUse2, int count) {
		if( count <= 1 ) return false;
		IDSequence buff = new IDSequence(true, count-2); //-1 based
		inUse.clear(); inUse2.clear();
		node.resetVisitStatus();
		return rIsValidNumVectorIntermediates(node, main, parents, inUse, inUse2, buff);
	}
	
	public static boolean rIsValidNumVectorIntermediates(CNode node, CNode main, Map<Long, Set<Long>> parents,
			Map<Long, Pair<Long, MutableInt>> inUse, Set<Long> inUse2, IDSequence buff) {
		if( node.isVisited() )
			return true;
		//recursively process inputs
		for( CNode c : node.getInput() )
			if( !rIsValidNumVectorIntermediates(c, main, parents, inUse, inUse2, buff) )
				return false;
		// add current node consumers for vectors
		if( !node.getDataType().isScalar() && parents.containsKey(node.getID()) && node != main ) {
			long vectID = buff.getNextID();
			if( inUse2.contains(vectID) )
				return false; //CONFLICT detected
			inUse.put(node.getID(), Pair.of(vectID,
				new MutableInt(parents.get(node.getID()).size())));
			inUse2.add(vectID);
		}
		//remove input dependencies
		for( CNode c : node.getInput() ) {
			Pair<Long, MutableInt> tmp = inUse.get(c.getID());
			if( tmp != null ) {
				tmp.getValue().decrement();
				if( tmp.getValue().intValue() <= 0 ) {
					inUse.remove(c.getID());
					inUse2.remove(tmp.getKey());
				}
			}
		}
		node.setVisited();
		return true;
	}
	
	public static Map<Long, Set<Long>> getAllParents(CNode node) {
		Map<Long, Set<Long>> ret = new HashMap<>();
		getAllParents(node, ret);
		return ret;
	}
	
	public static void getAllParents(CNode node, Map<Long, Set<Long>> parents) {
		for( CNode c : node.getInput() ) {
			if( !parents.containsKey(c.getID()) )
				parents.put(c.getID(), new HashSet<>());
			parents.get(c.getID()).add(node.getID());
			getAllParents(c, parents);
		}
	}

	public static boolean isType(TemplateType type, TemplateType... validTypes) {
		return ArrayUtils.contains(validTypes, type);
	}
	
	public static boolean hasCommonRowTemplateMatrixInput(Hop input1, Hop input2, CPlanMemoTable memo) {
		//if second input has no row template, it's always true
		if( !memo.contains(input2.getHopID(), TemplateType.ROW) )
			return true;
		//check for common row template input
		long tmp1 = getRowTemplateMatrixInput(input1, memo);
		long tmp2 = getRowTemplateMatrixInput(input2, memo);
		return (tmp1 == tmp2);
	}
	
	public static long getRowTemplateMatrixInput(Hop current, CPlanMemoTable memo) {
		MemoTableEntry me = memo.getBest(current.getHopID(), TemplateType.ROW);
		long ret = -1;
		for( int i=0; ret<0 && i<current.getInput().size(); i++ ) {
			Hop input = current.getInput().get(i);
			if( me.isPlanRef(i) && memo.contains(input.getHopID(), TemplateType.ROW) )
				ret = getRowTemplateMatrixInput(input, memo);
			else if( !me.isPlanRef(i) && isMatrix(input) )
				ret = input.getHopID();
		}
		return ret;
	}
	
	public static boolean containsBinary(CNode node, BinType type) {
		node.resetVisitStatus();
		boolean ret = rContainsBinary(node, type);
		node.resetVisitStatus();
		return ret;
	}
	
	public static boolean rContainsBinary(CNode node, BinType type) {
		if( node.isVisited() )
			return false;
		boolean ret = false;
		for( CNode input : node.getInput() )
			ret |= rContainsBinary(input, type);
		ret |= isBinary(node, type);
		node.setVisited();
		return ret;
	}
	
	public static boolean containsOuterProduct(Hop hop) {
		hop.resetVisitStatus();
		boolean ret = rContainsOuterProduct(hop);
		hop.resetVisitStatus();
		return ret;
	}
	
	public static boolean containsOuterProduct(Hop hop, Hop probe) {
		hop.resetVisitStatus();
		boolean ret = rContainsOuterProduct(hop, probe);
		hop.resetVisitStatus();
		return ret;
	}
	
	private static boolean rContainsOuterProduct(Hop current) {
		if( current.isVisited() )
			return false;
		boolean ret = false;
		ret |= HopRewriteUtils.isOuterProductLikeMM(current);
		for( int i=0; i<current.getInput().size() && !ret; i++ )
			ret |= rContainsOuterProduct(current.getInput().get(i));
		current.setVisited();
		return ret;
	}
	
	private static boolean rContainsOuterProduct(Hop current, Hop probe) {
		if( current.isVisited() )
			return false;
		boolean ret = false;
		ret |= HopRewriteUtils.isOuterProductLikeMM(current)
			&& checkContainment(current.getInput(), probe, true);
		for( int i=0; i<current.getInput().size() && !ret; i++ )
			ret |= rContainsOuterProduct(current.getInput().get(i), probe);
		current.setVisited();
		return ret;
	}
	
	private static boolean checkContainment(List<Hop> inputs, Hop probe, boolean inclTranspose) {
		if( !inclTranspose )
			return inputs.contains(probe);
		for( Hop hop : inputs )
			if( HopRewriteUtils.isTransposeOfItself(hop, probe) )
				return true;
		return false;
	}
	
	public static void rFlipVectorLookups(CNode current) {
		//flip vector lookups if necessary
		if( isUnary(current, UnaryType.LOOKUP_C) )
			((CNodeUnary)current).setType(UnaryType.LOOKUP_R);
		else if( isUnary(current, UnaryType.LOOKUP_R) )
			((CNodeUnary)current).setType(UnaryType.LOOKUP_C);
		//recursively process children
		for( CNode input : current.getInput() )
			rFlipVectorLookups(input);
	}

	public static boolean containsFusedRowVecAgg(CNodeTpl tpl) {
		if(!(tpl instanceof CNodeRow))
			return false;

		if(TemplateUtils.isBinary(tpl.getOutput(), BinType.ROWMAXS_VECTMULT))
			return true;

		for (CNode n : tpl.getOutput().getInput()) {
			if(TemplateUtils.isBinary(n, BinType.ROWMAXS_VECTMULT))
				return true;
		}
		return false;
	}
}
