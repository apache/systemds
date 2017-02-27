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
import java.util.HashSet;
import java.util.LinkedHashMap;

import org.apache.sysml.api.DMLException;
import org.apache.sysml.hops.AggBinaryOp;
import org.apache.sysml.hops.AggUnaryOp;
import org.apache.sysml.hops.BinaryOp;
import org.apache.sysml.hops.DataGenOp;
import org.apache.sysml.hops.DataOp;
import org.apache.sysml.hops.Hop;
import org.apache.sysml.hops.LiteralOp;
import org.apache.sysml.hops.ReorgOp;
import org.apache.sysml.hops.UnaryOp;
import org.apache.sysml.hops.codegen.cplan.CNode;
import org.apache.sysml.hops.codegen.cplan.CNodeBinary;
import org.apache.sysml.hops.codegen.cplan.CNodeBinary.BinType;
import org.apache.sysml.hops.codegen.cplan.CNodeData;
import org.apache.sysml.hops.codegen.cplan.CNodeRowAggVector;
import org.apache.sysml.hops.codegen.cplan.CNodeTpl;
import org.apache.sysml.hops.codegen.cplan.CNodeUnary;
import org.apache.sysml.hops.codegen.cplan.CNodeUnary.UnaryType;
import org.apache.sysml.hops.Hop.AggOp;
import org.apache.sysml.hops.Hop.Direction;
import org.apache.sysml.hops.Hop.OpOp2;
import org.apache.sysml.hops.Hop.ReOrgOp;
import org.apache.sysml.parser.Expression.DataType;
import org.apache.sysml.runtime.matrix.data.Pair;

public class RowAggTpl extends BaseTpl {

	public RowAggTpl() {
		super(TemplateType.RowAggTpl);
	}
	
	@Override
	public boolean openTpl(Hop hop) {
		if ( (hop instanceof AggBinaryOp || hop instanceof AggUnaryOp) // An aggregate operation  			  
			&& ( (hop.getDim1()==1 && hop.getDim2()!=1) || (hop.getDim1()!=1 && hop.getDim2()==1) )  )// the output is a vector  
			return true;
		return false;
	}

	@Override
	public boolean findTplBoundaries(Hop initialHop, CplanRegister cplanRegister) {
		_initialHop = initialHop;
		if(initialHop instanceof AggBinaryOp) {
			// for simplicity we assume that the first operand should be t(X) however, it could be later on W.T(X)
			if(initialHop.getInput().get(0) instanceof ReorgOp && ((ReorgOp)initialHop.getInput().get(0)).getOp()== ReOrgOp.TRANSPOSE  )
				_matrixInputs.add(initialHop.getInput().get(0).getInput().get(0)); //add what is under the transpose
			else
				return false; 
		}
		rFindRowAggPattern(initialHop, new HashSet<Long>());
		
		if(cplanRegister.containsHop(TemplateType.RowAggTpl, initialHop.getHopID()))
			return false;
		
		return (_endHop != null);
	}
	
	
	private void rFindRowAggPattern(Hop h, HashSet<Long> memo)
	{
		if(memo.contains(h.getHopID()) || h.getDataType() == DataType.SCALAR 
			|| h instanceof DataOp || h instanceof DataGenOp || h instanceof LiteralOp) {
			return;
		}
		
		boolean continueTraversing = false;
		if (h instanceof AggBinaryOp)
		{
			if(h != _initialHop) {
				//T(X) % ..... X %*% v ,check that X is the same as what we saw previously under transpose
				if( h.getInput().get(0).equals(_matrixInputs.get(0)) && TemplateUtils.isVector(h.getInput().get(1)) ) {
					_endHop = h;
				}
			}
			else {
				continueTraversing = true;
			}
		}
		// if initial hop is colSums continue
		else if(h instanceof AggUnaryOp && (((AggUnaryOp)_initialHop).getDirection() == Direction.Col && ((AggUnaryOp)_initialHop).getOp() == AggOp.SUM ) && h == _initialHop)
		{
			continueTraversing=true;
		}
		//rowSums(X)
		else if(h instanceof AggUnaryOp && ((AggUnaryOp)h).getDirection() == Direction.Row && ((AggUnaryOp)h).getOp() == AggOp.SUM )
		{
			// check if root pattern is colsums
			if((((AggUnaryOp)_initialHop).getDirection() == Direction.Col && ((AggUnaryOp)_initialHop).getOp() == AggOp.SUM ))
			{
				
				//TODO Now the pattern is limited to finding rowSums 
				_matrixInputs.add(h.getInput().get(0));
				_endHop = h;
			}
		}
		// unary operation || binary operation with first input as a matrix || binary operation with second input as a matrix 
		else if( ( h instanceof UnaryOp || (h instanceof BinaryOp && h.getInput().get(0).getDataType() == DataType.MATRIX  &&  TemplateUtils.isVectorOrScalar(h.getInput().get(1))) || (h instanceof BinaryOp && TemplateUtils.isVectorOrScalar(h.getInput().get(0))  &&  h.getInput().get(1).getDataType() == DataType.MATRIX)  )  //unary operation or binary operaiton with one matrix and a scalar
				&& 	h.getDataType() == DataType.MATRIX		 // Output is a matrix
				&&  TemplateUtils.isOperationSupported(h) )	 //Operation is supported in codegen
		{
			continueTraversing = true;
		}
		
		//check if we should continue traversing
		if(!continueTraversing)
		{
			return; // stop traversing if conditions does not apply 
		}
		else
		{
			//process childs recursively
			for( Hop in : h.getInput() )
				rFindRowAggPattern(in,memo);
		}
	    memo.add(h.getHopID());
	}
	
	@Override
	public LinkedHashMap<Long, Pair<Hop[],CNodeTpl>> constructTplCplan(boolean compileLiterals)
		throws DMLException {
		
		//re-assign the dimensions of inputs to match the generated code dimensions
		_initialCnodes.add(new CNodeData(_matrixInputs.get(0)));
		
		rConstructRowAggCplan(_initialHop,_initialHop,new HashSet<Long>(), compileLiterals);
		return _cpplans;
	}
	
	private void rConstructRowAggCplan(Hop root, Hop hop, HashSet<Long> memo, boolean compileLiterals) throws DMLException
	{
		if( memo.contains(hop.getHopID()) )
			return;
		//process childs recursively
		for( Hop c : hop.getInput() )
			rConstructRowAggCplan(root, c, memo, compileLiterals);
		if(hop == _endHop)
			_endHopReached = true;
		
		 // first hop to enter here should be _endHop
		if(TemplateUtils.inputsAreGenerated(hop,_matrixInputs,_cpplans) && _endHopReached)  // if direct children are DataGenOps, literals, or already in the cpplans then we are ready to generate code
		{
			CNodeRowAggVector rowTmpl = null;
			
			//Fetch operands
			CNode out = null;
			ArrayList<CNode> addedCNodes = new ArrayList<CNode>();
			ArrayList<Hop> addedHops = new ArrayList<Hop>();
			ArrayList<CNode> cnodeData = TemplateUtils.fetchOperands(hop, _cpplans, addedCNodes, addedHops, _initialCnodes, compileLiterals);
			
			//if operands are scalar or independent from X 
			boolean independentOperands = hop.getDataType() == DataType.SCALAR 
					|| TemplateUtils.isOperandsIndependent(cnodeData, addedHops, new String[] {_matrixInputs.get(0).getName()});
			
			if(!independentOperands)
			{
			
				if(hop instanceof AggUnaryOp)
				{
					CNode cdata1 = cnodeData.get(0);
					//set the out cnode based on the operation
					if(  ((AggUnaryOp)hop).getDirection() == Direction.Row && ((AggUnaryOp)hop).getOp() == AggOp.SUM  ) //RowSums 
					{
						if(hop.getInput().get(0).getDim2()==1)
							out = (cdata1.getDataType()==DataType.SCALAR) ? cdata1 : new CNodeUnary(cdata1,UnaryType.LOOKUP);
						else
							out = new CNodeUnary(cdata1, UnaryType.ROW_SUMS);
					}
					// if colsums is the root hop, wire the input to the out because colsums it is done automatically by the template
					else  if (((AggUnaryOp)hop).getDirection() == Direction.Col && ((AggUnaryOp)hop).getOp() == AggOp.SUM && root == hop)
					{
						//vector div add without temporary copy
						if(cdata1 instanceof CNodeBinary && ((CNodeBinary)cdata1).getType()==BinType.VECT_DIV_SCALAR)
							out = new CNodeBinary(cdata1.getInput().get(0), cdata1.getInput().get(1), BinType.VECT_DIV_ADD);
						else	
							out = cdata1;
					}
				}
				else if(hop instanceof AggBinaryOp)
				{
					//Fetch operands specific to the operation
					CNode cdata1 = cnodeData.get(0);
					CNode cdata2 = cnodeData.get(1);
					
					//choose the operation based on the transpose
					if( hop.getInput().get(0) instanceof ReorgOp && ((ReorgOp)hop.getInput().get(0)).getOp()==ReOrgOp.TRANSPOSE )
					{
						//fetch the data inside the transpose
						//cdata1 = new CNodeData(hop.getInput().get(0).getInput().get(0).getName(), (int)hop.getInput().get(0).getInput().get(0).getDim1(), (int)hop.getInput().get(0).getInput().get(0).getDim2());
						out = new CNodeBinary(cdata2, cdata1, BinType.VECT_MULT_ADD);
					}
					else
					{
						if(hop.getInput().get(0).getDim2()==1 && hop.getInput().get(1).getDim2()==1)
							out = new CNodeBinary((cdata1.getDataType()==DataType.SCALAR)? cdata1 : new CNodeUnary(cdata1, UnaryType.LOOKUP0),
								(cdata2.getDataType()==DataType.SCALAR)? cdata2 : new CNodeUnary(cdata2, UnaryType.LOOKUP0), BinType.MULT);
						else	
							out = new CNodeBinary(cdata1, cdata2, BinType.DOT_PRODUCT);
					}
				}
				else if(hop instanceof BinaryOp)
				{
					CNode cdata1 = cnodeData.get(0);
					CNode cdata2 = cnodeData.get(1);
					
					// if one input is a matrix then we need to do vector by scalar operations
					if(hop.getInput().get(0).getDim1() > 1 && hop.getInput().get(0).getDim2() > 1 )
					{
						if (((BinaryOp)hop).getOp()== OpOp2.DIV)
							//CNode generatedScalar = new CNodeData("1", 0, 0); // generate literal in order to rewrite the div to x * 1/y
							//CNode outScalar = new CNodeBinary(generatedScalar, cdata2, BinType.SCALAR_DIVIDE);
							//out = new CNodeBinary(outScalar, cdata1, BinType.VECT_MULT_ADD);
							out = new CNodeBinary(cdata1, cdata2, BinType.VECT_DIV_SCALAR);
						
					}
					else //one input is a vector/scalar other is a scalar
					{
						//Primitive Operation has the same name as Hop Type OpOp2
						String primitiveOpName = ((BinaryOp)hop).getOp().toString();
						
						if( (cdata1.getNumRows() > 1 && cdata1.getNumCols() == 1) || (cdata1.getNumRows() == 1 && cdata1.getNumCols() > 1) )
						{
							//second argument is always the vector
							cdata1 = new CNodeUnary(cdata1, UnaryType.LOOKUP);
							//out = new CNodeBinary(tmp, cdata2, BinType.valueOf(primitiveOpName));
						}
						//cdata2 is vector
						//else if( cdata2 instanceof CNodeData && (((CNodeData)cdata2).getNumRows() > 1 && ((CNodeData)cdata2).getNumCols() == 1) || ( ((CNodeData)cdata2).getNumRows() == 1 && ((CNodeData)cdata2).getNumCols() > 1  ))
						if( (cdata2.getNumRows() > 1 && cdata2.getNumCols() == 1) || (cdata2.getNumRows() == 1 && cdata2.getNumCols() > 1) )
						{
							cdata2 = new CNodeUnary(cdata2, UnaryType.LOOKUP);
							//out = new CNodeBinary(cdata1, tmp, BinType.valueOf(primitiveOpName));
						}
						out = new CNodeBinary(cdata1, cdata2, BinType.valueOf(primitiveOpName));	
					}
					
				}
				
				if( out.getDataType().isMatrix() ) {
					out.setNumRows(hop.getDim1());
					out.setNumCols(hop.getDim2());
				}
			}
			// wire output to the template
			if(out != null || independentOperands)
			{
				if(_cpplans.isEmpty())
				{
					//first initialization has to have the first variable as input
					ArrayList<CNode> initialInputs = new ArrayList<CNode>();
										
					if(independentOperands) // pass the hop itself as an input instead of its children
					{
						CNode c =  new CNodeData(hop);
						initialInputs.addAll(_initialCnodes);
						initialInputs.add(c);
						rowTmpl =  new CNodeRowAggVector(initialInputs, c); 
						_cpplans.put(hop.getHopID(), new Pair<Hop[],CNodeTpl>(new Hop[] {_matrixInputs.get(0),hop} ,rowTmpl));
					}
					else
					{
						initialInputs.addAll(_initialCnodes);
						initialInputs.addAll(cnodeData);
						rowTmpl =  new CNodeRowAggVector(initialInputs, out); 
						
						//Hop[] hopArray = new Hop[hop.getInput().size()+1];
						Hop[] hopArray = new Hop[addedHops.size()+1];
						hopArray[0] = _matrixInputs.get(0);
						
						//System.arraycopy( hop.getInput().toArray(), 0, hopArray, 1, hop.getInput().size());
						System.arraycopy( addedHops.toArray(), 0, hopArray, 1, addedHops.size());
						
						_cpplans.put(hop.getHopID(), new Pair<Hop[],CNodeTpl>(hopArray,rowTmpl));
					}
				}
				else
				{
					if(independentOperands)
					{
						CNode c =  new CNodeData(hop);
						//clear Operands
						addedCNodes.clear();
						addedHops.clear();
						
						//added the current hop as the input
						addedCNodes.add(c);
						addedHops.add(hop);
						out = c;
					}
					//wire the output to existing or new template	
					TemplateUtils.setOutputToExistingTemplate(hop, out, _cpplans, addedCNodes, addedHops);
				}
			}
			memo.add(hop.getHopID());
		}	
	}
}
