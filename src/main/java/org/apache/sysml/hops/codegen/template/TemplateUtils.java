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
import java.util.Iterator;
import java.util.LinkedHashSet;

import org.apache.sysml.hops.AggBinaryOp;
import org.apache.sysml.hops.AggUnaryOp;
import org.apache.sysml.hops.BinaryOp;
import org.apache.sysml.hops.Hop;
import org.apache.sysml.hops.LiteralOp;
import org.apache.sysml.hops.TernaryOp;
import org.apache.sysml.hops.Hop.AggOp;
import org.apache.sysml.hops.Hop.Direction;
import org.apache.sysml.hops.UnaryOp;
import org.apache.sysml.hops.codegen.cplan.CNode;
import org.apache.sysml.hops.codegen.cplan.CNodeBinary;
import org.apache.sysml.hops.codegen.cplan.CNodeBinary.BinType;
import org.apache.sysml.hops.codegen.cplan.CNodeData;
import org.apache.sysml.hops.codegen.cplan.CNodeTernary;
import org.apache.sysml.hops.codegen.cplan.CNodeUnary;
import org.apache.sysml.hops.codegen.cplan.CNodeUnary.UnaryType;
import org.apache.sysml.hops.codegen.template.TemplateBase.TemplateType;
import org.apache.sysml.hops.rewrite.HopRewriteUtils;
import org.apache.sysml.hops.codegen.cplan.CNodeTernary.TernaryType;
import org.apache.sysml.hops.codegen.cplan.CNodeTpl;
import org.apache.sysml.parser.Expression.DataType;
import org.apache.sysml.runtime.codegen.SpoofCellwise.CellType;
import org.apache.sysml.runtime.codegen.SpoofOuterProduct.OutProdType;
import org.apache.sysml.runtime.util.UtilFunctions;

public class TemplateUtils 
{
	public static final TemplateBase[] TEMPLATES = new TemplateBase[]{new TemplateRowAgg(), new TemplateCell(), new TemplateOuterProduct()};
	
	public static boolean isVector(Hop hop) {
		return (hop.getDataType() == DataType.MATRIX 
			&& (hop.getDim1() != 1 && hop.getDim2() == 1 
			  || hop.getDim1() == 1 && hop.getDim2() != 1 ) );
	}
	
	public static boolean isColVector(CNode hop) {
		return (hop.getDataType() == DataType.MATRIX 
			&& hop.getNumRows() != 1 && hop.getNumCols() == 1);
	}
	
	public static boolean isRowVector(CNode hop) {
		return (hop.getDataType() == DataType.MATRIX 
			&& hop.getNumRows() == 1 && hop.getNumCols() != 1);
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

	public static boolean isOperationSupported(Hop h) {
		if(h instanceof  UnaryOp)
			return UnaryType.contains(((UnaryOp)h).getOp().name());
		else if(h instanceof BinaryOp)
			return BinType.contains(((BinaryOp)h).getOp().name());
		else if(h instanceof TernaryOp)
			return TernaryType.contains(((TernaryOp)h).getOp().name());
		return false;
	}

	private static void rfindChildren(Hop hop, HashSet<Hop> children ) {		
		if( hop instanceof UnaryOp || (hop instanceof BinaryOp && hop.getInput().get(0).getDataType() == DataType.MATRIX  &&  TemplateUtils.isVectorOrScalar( hop.getInput().get(1))) || (hop instanceof BinaryOp && TemplateUtils.isVectorOrScalar( hop.getInput().get(0))  &&  hop.getInput().get(1).getDataType() == DataType.MATRIX)    //unary operation or binary operaiton with one matrix and a scalar
					&& 	hop.getDataType() == DataType.MATRIX )
		{	
			if(!children.contains(hop))
				children.add(hop);
			Hop matrix = TemplateUtils.isMatrix(hop.getInput().get(0)) ? hop.getInput().get(0) : hop.getInput().get(1);
			rfindChildren(matrix,children);
		}
		else 
			children.add(hop);
	}
	
	private static Hop findCommonChild(Hop hop1, Hop hop2) {
		//this method assumes that each two nodes have at most one common child 
		LinkedHashSet<Hop> children1 = new LinkedHashSet<Hop>();
		LinkedHashSet<Hop> children2 = new LinkedHashSet<Hop>();
		
		rfindChildren(hop1, children1 );
		rfindChildren(hop2, children2 );
		
		//iterate on one set and find the first common child in the other set
		Iterator<Hop> iter = children1.iterator();
		while (iter.hasNext()) {
			Hop candidate = iter.next();
			if(children2.contains(candidate))
				return candidate;
		}
		return null;
	}
	
	public static Hop commonChild(ArrayList<Hop> _adddedMatrices, Hop input) {
		Hop currentChild = null;
		//loop on every added matrix and find its common child with the input, if all of them have the same common child then return it, otherwise null 
		for(Hop addedMatrix : _adddedMatrices)
		{
			Hop child = findCommonChild(addedMatrix,input);
			if(child == null)  // did not find a common child
				return null;
			if(currentChild == null) // first common child to be seen
				currentChild = child;
			else if(child.getHopID() != currentChild.getHopID())
				return null;
		}
		return currentChild;
	}

	public static HashSet<Long> rGetInputHopIDs( CNode node, HashSet<Long> ids ) {
		if( node instanceof CNodeData && !node.isLiteral() )
			ids.add(((CNodeData)node).getHopID());
		
		for( CNode c : node.getInput() )
			rGetInputHopIDs(c, ids);
			
		return ids;
	}
	
	public static Hop[] mergeDistinct(HashSet<Long> ids, Hop[] input1, Hop[] input2) {
		Hop[] ret = new Hop[ids.size()];
		int pos = 0;
		for( Hop[] input : new Hop[][]{input1, input2} )
			for( Hop c : input )
				if( ids.contains(c.getHopID()) )
					ret[pos++] = c; 
		return ret;
	}

	public static TemplateBase createTemplate(TemplateType type) {
		return createTemplate(type, false);
	}
	
	public static TemplateBase createTemplate(TemplateType type, boolean closed) {
		TemplateBase tpl = null;
		switch( type ) {
			case CellTpl: tpl = new TemplateCell(); break;
			case RowAggTpl: tpl = new TemplateRowAgg(); break;
			case OuterProdTpl: tpl = new TemplateOuterProduct(); break;
		}
		tpl._closed = closed;
		return tpl;
	}
	
	public static CellType getCellType(Hop hop) {
		return (hop instanceof AggBinaryOp) ? CellType.FULL_AGG :
			(hop instanceof AggUnaryOp) ? ((((AggUnaryOp) hop).getDirection() == Direction.RowCol) ? 
			CellType.FULL_AGG : CellType.ROW_AGG) : CellType.NO_AGG;
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
		throw new RuntimeException("Undefined outer product type");
	}
	
	public static boolean isLookup(CNode node) {
		return (node instanceof CNodeUnary 
				&& (((CNodeUnary)node).getType()==UnaryType.LOOKUP_R 
				|| ((CNodeUnary)node).getType()==UnaryType.LOOKUP_RC));
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
		return (output instanceof CNodeUnary || output instanceof CNodeBinary
				|| output instanceof CNodeTernary) && hasOnlyDataNodeOrLookupInputs(output);
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
}
