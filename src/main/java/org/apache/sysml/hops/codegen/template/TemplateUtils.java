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
import java.util.Map.Entry;

import org.apache.sysml.hops.AggUnaryOp;
import org.apache.sysml.hops.BinaryOp;
import org.apache.sysml.hops.DataGenOp;
import org.apache.sysml.hops.DataOp;
import org.apache.sysml.hops.Hop;
import org.apache.sysml.hops.LiteralOp;
import org.apache.sysml.hops.ReorgOp;
import org.apache.sysml.hops.TernaryOp;
import org.apache.sysml.hops.Hop.AggOp;
import org.apache.sysml.hops.Hop.Direction;
import org.apache.sysml.hops.Hop.ReOrgOp;
import org.apache.sysml.hops.UnaryOp;
import org.apache.sysml.hops.codegen.cplan.CNode;
import org.apache.sysml.hops.codegen.cplan.CNodeBinary.BinType;
import org.apache.sysml.hops.codegen.cplan.CNodeCell;
import org.apache.sysml.hops.codegen.cplan.CNodeData;
import org.apache.sysml.hops.codegen.cplan.CNodeOuterProduct;
import org.apache.sysml.hops.codegen.cplan.CNodeTpl;
import org.apache.sysml.hops.codegen.cplan.CNodeUnary;
import org.apache.sysml.hops.codegen.cplan.CNodeUnary.UnaryType;
import org.apache.sysml.hops.codegen.cplan.CNodeTernary.TernaryType;
import org.apache.sysml.parser.Expression.DataType;
import org.apache.sysml.runtime.codegen.SpoofCellwise.CellType;
import org.apache.sysml.runtime.matrix.data.Pair;
import org.apache.sysml.runtime.util.UtilFunctions;

public class TemplateUtils 
{
	public static boolean inputsAreGenerated(Hop parent, ArrayList<Hop> inputs, HashMap<Long, Pair<Hop[],CNodeTpl>> cpplans)
	{		
		if( parent instanceof DataOp || parent instanceof DataGenOp || parent instanceof LiteralOp || inputs.contains(parent) )
			return false;
	
		for(Hop hop : parent.getInput() )
			if(!inputs.contains(hop) && !(hop instanceof DataOp) && !(hop instanceof DataGenOp) && !(hop.getDataType()==DataType.SCALAR) && !isVector(hop) && !(cpplans.containsKey(hop.getHopID())) && !( hop instanceof ReorgOp && ((ReorgOp)hop).getOp() == ReOrgOp.TRANSPOSE && inputsAreGenerated(hop,inputs, cpplans) ))
				return false;
		return true;
	}
	
	public static ArrayList<CNode> fetchOperands(Hop hop,  HashMap<Long, Pair<Hop[],CNodeTpl>> cpplans, ArrayList<CNode> addedCNodes, ArrayList<Hop> addedHops, ArrayList<CNodeData> initialCNodes, boolean compileLiterals)
	{
		ArrayList<CNode> cnodeData = new ArrayList<CNode>();
		for (Hop h: hop.getInput())
		{
			CNode cdata = null;
			
			//CNodeData already in template inputs
			for(CNodeData c : initialCNodes) {
				if( c.getHopID() == h.getHopID() ) {
					cdata = c;
					break;
				}
			}
			
			if(cdata != null)
			{
				cnodeData.add(cdata);
				continue;
			}
			//hop already in the cplan
			else if(cpplans.containsKey(h.getHopID()))
			{
				cdata = cpplans.get(h.getHopID()).getValue().getOutput();
			}
			else if(h instanceof ReorgOp && ((ReorgOp)h).getOp()==ReOrgOp.TRANSPOSE )
			{
				//fetch what is under the transpose
				Hop in = h.getInput().get(0);
				cdata = new CNodeData(in);
				if(in instanceof DataOp || in instanceof DataGenOp ) {
					addedCNodes.add(cdata);
					addedHops.add(in);
				}
			}
			else
			{
				//note: only compile literals if forced or integer literals (likely constants) 
				//to increase reuse potential on literal replacement during recompilation
				cdata = new CNodeData(h);
				cdata.setLiteral(h instanceof LiteralOp && (compileLiterals 
					|| UtilFunctions.isIntegerNumber(((LiteralOp)h).getStringValue())));
				if( !cdata.isLiteral() ) {
					addedCNodes.add(cdata);
					addedHops.add(h);
				}
			}
			
			cnodeData.add(cdata);
		}
		return cnodeData;
	}
	
	public static void setOutputToExistingTemplate(Hop hop, CNode out,  HashMap<Long, Pair<Hop[],CNodeTpl>> cpplans, ArrayList<CNode> addedCNodes, ArrayList<Hop> addedHops)
	{
		//get the toplevel rowTemp
		Entry<Long, Pair<Hop[],CNodeTpl>> cplan = null;
		Iterator<Entry<Long, Pair<Hop[],CNodeTpl>>> iterator = cpplans.entrySet().iterator();
		while (iterator.hasNext()) 
			cplan = iterator.next();
		
		CNodeTpl tmpl = cplan.getValue().getValue().clone();
		tmpl.setDataType(hop.getDataType());
		
		if(tmpl instanceof CNodeOuterProduct) {
			((CNodeOuterProduct) tmpl).setOutProdType( ((CNodeOuterProduct)cplan.getValue().getValue()).getOutProdType());
			((CNodeOuterProduct) tmpl).setTransposeOutput(((CNodeOuterProduct)cplan.getValue().getValue()).isTransposeOutput() );
		}
		else if( tmpl instanceof CNodeCell ) {
			((CNodeCell)tmpl).setCellType(getCellType(hop));
			((CNodeCell)tmpl).setMultipleConsumers(hop.getParent().size()>1);
		}
		
		//add extra inputs
		for(CNode c : addedCNodes)
			tmpl.addInput(c);
		
		//modify addedHops if they exist
		
		Hop[] currentInputHops = cplan.getValue().getKey();
		for (Hop h : currentInputHops)
			if (addedHops.contains(h))
				addedHops.remove(h);
		
		Hop[] extendedHopInputs = new Hop[cplan.getValue().getKey().length + addedHops.size()];
		System.arraycopy(cplan.getValue().getKey(), 0, extendedHopInputs, 0, cplan.getValue().getKey().length);
		for(int j=addedHops.size(); j > 0; j--)	
			extendedHopInputs[extendedHopInputs.length-j] = addedHops.get(addedHops.size() - j);  //append the added hops to the end of the array
	
		//set the template output and add it to the cpplans
		Pair<Hop[],CNodeTpl> pair = new Pair<Hop[],CNodeTpl>(extendedHopInputs,tmpl);
		pair.getValue().setOutput(out);
		cpplans.put(hop.getHopID(), pair);
		
	}

	public static boolean isOperandsIndependent(ArrayList<CNode> cnodeData, ArrayList<Hop> addedHops, String[] varNames)
	{
		for(CNode c : cnodeData) {
			// it is some variable inside the cplan // TODO needs to be modified because sometimes the varname is not null but the variable is in the cplan
			if(c.getVarname() == null)
				return false;
			//if one of the operands is is any of the varnames // if one of the operands is T(X) this condition will apply as well because during fetch operands we fetch what is inside transpose 
			for(String varName : varNames)
				if(c.getVarname().equals(varName))
					return false;
		}
		return true;
	}
	
	public static Entry<Long, Pair<Hop[],CNodeTpl>> getTopLevelCpplan(HashMap<Long, Pair<Hop[],CNodeTpl>> cplans)
	{
		Entry<Long, Pair<Hop[],CNodeTpl>> ret = null;
		
		//get last entry (most fused operators) or special handling
		boolean hasExp = false;
		for( Entry<Long, Pair<Hop[],CNodeTpl>> e : cplans.entrySet() ) 
		{ 
			ret = e; //keep last seen entry
			
			//special handling overlapping fused operators with exp
			hasExp |= (ret.getValue().getValue().getOutput() instanceof CNodeUnary
					&& ((CNodeUnary)ret.getValue().getValue().getOutput()).getType()==UnaryType.EXP);
			
			if( hasExp && ret.getValue().getValue() instanceof CNodeCell
				&& ((CNodeCell)ret.getValue().getValue()).hasMultipleConsumers() )
				break;
		}
		
		return ret;
	}
	
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
	
	private static CellType getCellType(Hop hop) {
		return (hop instanceof AggUnaryOp && ((AggUnaryOp)hop).getOp() == AggOp.SUM) ?
			((((AggUnaryOp) hop).getDirection() == Direction.RowCol) ? 
			CellType.FULL_AGG : CellType.ROW_AGG) : CellType.NO_AGG;
	}
	
	public static boolean isLookup(CNode node) {
		return (node instanceof CNodeUnary 
				&& (((CNodeUnary)node).getType()==UnaryType.LOOKUP_R 
				|| ((CNodeUnary)node).getType()==UnaryType.LOOKUP_RC));
	}
}
