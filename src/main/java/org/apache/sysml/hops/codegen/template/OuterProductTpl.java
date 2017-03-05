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
import java.util.Arrays;
import java.util.HashMap;
import java.util.HashSet;
import java.util.LinkedHashMap;
import java.util.List;

import org.apache.sysml.api.DMLException;
import org.apache.sysml.hops.AggBinaryOp;
import org.apache.sysml.hops.AggUnaryOp;
import org.apache.sysml.hops.BinaryOp;
import org.apache.sysml.hops.Hop;
import org.apache.sysml.hops.Hop.OpOp2;
import org.apache.sysml.hops.ReorgOp;
import org.apache.sysml.hops.UnaryOp;
import org.apache.sysml.hops.Hop.AggOp;
import org.apache.sysml.hops.Hop.Direction;
import org.apache.sysml.hops.Hop.ReOrgOp;
import org.apache.sysml.hops.codegen.cplan.CNode;
import org.apache.sysml.hops.codegen.cplan.CNodeBinary;
import org.apache.sysml.hops.codegen.cplan.CNodeBinary.BinType;
import org.apache.sysml.hops.codegen.cplan.CNodeData;
import org.apache.sysml.hops.codegen.cplan.CNodeOuterProduct;
import org.apache.sysml.hops.codegen.cplan.CNodeTpl;
import org.apache.sysml.hops.codegen.cplan.CNodeUnary;
import org.apache.sysml.hops.codegen.cplan.CNodeUnary.UnaryType;
import org.apache.sysml.hops.rewrite.HopRewriteUtils;
import org.apache.sysml.parser.Expression.DataType;
import org.apache.sysml.runtime.codegen.SpoofOuterProduct.OutProdType;
import org.apache.sysml.runtime.matrix.data.Pair;

public class OuterProductTpl extends BaseTpl {
	
	public OuterProductTpl() {
		super(TemplateType.OuterProductTpl);
	}
	
	private List<OpOp2> sparseDrivers = new ArrayList<OpOp2>(Arrays.asList(OpOp2.MULT, OpOp2.DIV));	
	private OutProdType _outerProductType = null;
	private boolean _transposeOutput = false;
	private boolean _transposeInput = false;
	
	@Override
	public boolean openTpl(Hop hop) {
		// outerproduct ( output dimensions is greater than the common dimension) 
		return ( hop instanceof AggBinaryOp && ((AggBinaryOp)hop).isMatrixMultiply() && hop.dimsKnown()
				&& hop.getInput().get(0).dimsKnown() && hop.getInput().get(1).dimsKnown()
				&& (hop.getDim1() > hop.getInput().get(0).getDim2() && hop.getDim2() > hop.getInput().get(1).getDim1()) );
	}

	@Override
	public boolean findTplBoundaries(Hop h, CplanRegister cplanRegister) { 
		_endHop = h;//outerProduct tpl starts with endHop
		HashMap<String,Hop> uniqueMatrixInputs = new HashMap<String,Hop>();
		uniqueMatrixInputs.put("U",  h.getInput().get(0));
		if( h.getInput().get(1) instanceof ReorgOp && ((ReorgOp)h.getInput().get(1)).getOp() == ReOrgOp.TRANSPOSE )
			uniqueMatrixInputs.put("V",  h.getInput().get(1).getInput().get(0));
		else
		{
			_transposeInput = true; // we need to transpose V to be tall and skinny
			uniqueMatrixInputs.put("V",  h.getInput().get(1));
		}
		rfindOuterProduct(_endHop, _endHop, uniqueMatrixInputs, h.getDim1(), h.getDim2(), new HashSet<Long>());
		
		if(uniqueMatrixInputs.size() == 3 && _initialHop != null && _initialHop != _endHop )	//sanity check
		{
			//check if added matrices can be inferred from input matrices for example (X!=0) or abs(X) are not different from X
			Hop commonChild = null;
			if(! _adddedMatrices.isEmpty() ) {
				//if addedMatrices does not have a common child with input X then do not compile
				commonChild = TemplateUtils.commonChild(_adddedMatrices,uniqueMatrixInputs.get("X"));
				if(commonChild == null ) // there are multiple matrices involved other than X
						return false;
			}
			if(commonChild != null) {
				_matrixInputs.add(commonChild); //add common child as the major input matrix
				_adddedMatrices.add(uniqueMatrixInputs.get("X")); // put unique matrix as one of the additional matrices that is a chain of cell wise operations for the input matrix
			}
			else {
				_matrixInputs.add(uniqueMatrixInputs.get("X")); //major matrix is the sparse driver
			}
			_matrixInputs.add(uniqueMatrixInputs.get("U"));
			
			if(_transposeInput) {
				ReorgOp transposeV = HopRewriteUtils.createTranspose(uniqueMatrixInputs.get("V"));
				//ReorgOp transposeV = new ReorgOp("", uniqueMatrixInputs.get("V").getDataType(), uniqueMatrixInputs.get("V").getValueType(), ReOrgOp.TRANSPOSE, uniqueMatrixInputs.get("V"));
				_matrixInputs.add(transposeV);
			}
			else {
				_matrixInputs.add(uniqueMatrixInputs.get("V"));
			}
			
			
			//add also added matrices so that they can be interpreted as inputs
			for(Hop addedMatrix : _adddedMatrices)
				if(!_matrixInputs.contains(addedMatrix))
					_matrixInputs.add(addedMatrix);
		
			//add the children of _endHop ( this will handle the case for wdivmm right when I add the both t(V) and V as inputs
			for (Hop hop: _endHop.getInput())
				_matrixInputs.add(hop);
			
			return true;
		}
		else
			return false;
		
	}	
	private void rfindOuterProduct(Hop child, Hop h, HashMap<String,Hop> uniqueMatrixInputs, long outerProductDim1, long outerProductDim2, HashSet<Long> memo)
	{
		if(memo.contains(h.getHopID()))
			return;
		
		if( ( h instanceof UnaryOp || h instanceof BinaryOp  )  //unary operation or binary operation
				&& 	h.getDataType() == DataType.MATRIX			 // Output is a matrix
				&& h.getDim1() == outerProductDim1 && h.getDim2() == outerProductDim2 // output is the same size as the matrix
				&& TemplateUtils.isOperationSupported(h))  // operation is supported in codegen
		{
			if(h instanceof BinaryOp)
			{
				
				// find the other child rather than the one that called the parent
				Hop otherChild = h.getInput().get(0) !=  child ? h.getInput().get(0) : h.getInput().get(1);
				
				//if scalar or vector then we fuse it similar to the way we fuse celltpl,
				if(TemplateUtils.isVectorOrScalar(otherChild))
				{
					_initialHop = h;
					_outerProductType = OutProdType.CELLWISE_OUTER_PRODUCT;

				}
				// other child is a  matrix
				else
				{
					//if the binary operation is sparse safe (mult, div)
					if(sparseDrivers.contains(((BinaryOp)h).getOp()) ) 
					{
						if(!uniqueMatrixInputs.containsKey("X"))
						{
							//extra sanity check
							if(otherChild.getDim1() == outerProductDim1 && otherChild.getDim2() == outerProductDim2) {
								uniqueMatrixInputs.put("X", otherChild);
								_initialHop = h;
							}
							else { //matrix size does not match what is expected for X
								return; 
							}
						}
					}
					else {
						_adddedMatrices.add(otherChild);
					}
				}
			}
		}
		
		if(  h instanceof AggBinaryOp && ((AggBinaryOp) h).isMatrixMultiply() && h != child) //make sure that the AggBinaryOp is not the same as the outerproduct that triggered this method
		{
			if(memo.contains(h.getInput().get(0).getHopID())) { // if current node is the parent for the left child then it is right matrix multiply
			
				if (h.getInput().get(1) == uniqueMatrixInputs.get("V") )//right operand is V
				{
					_initialHop = h;
					_outerProductType = OutProdType.RIGHT_OUTER_PRODUCT;
					return;
				}
				//right operand is t(V)
				else if(h.getInput().get(1) instanceof ReorgOp && ((ReorgOp)h.getInput().get(1)).getOp() == ReOrgOp.TRANSPOSE && h.getInput().get(1).getInput().get(0) == uniqueMatrixInputs.get("V") )
				{
					//replace V with T(V)
					uniqueMatrixInputs.put("V", h.getInput().get(1));
					_transposeInput = false; //no need to transpose Input
					_initialHop = h;
					_outerProductType = OutProdType.RIGHT_OUTER_PRODUCT;
					return;
				}
				else
				{
					_initialHop = h.getInput().get(0); // set the child that was processed
					return;	
				}
			}
			else {//left matrix multiply
				
				//left is T(U) 
				if (h.getInput().get(0) instanceof ReorgOp && ((ReorgOp)h.getInput().get(0)).getOp() == ReOrgOp.TRANSPOSE && h.getInput().get(0).getInput().get(0) == uniqueMatrixInputs.get("U") ) 
				{
					_initialHop = h;
					_outerProductType = OutProdType.LEFT_OUTER_PRODUCT;
					//T(T(U) %*% ..)
					for(Hop hParent : h.getParent())
						if(hParent instanceof ReorgOp && ((ReorgOp)hParent).getOp() == ReOrgOp.TRANSPOSE) {
							_initialHop = hParent; // set the transpose hop
							return;
						}	
					_transposeOutput = true;
					return;
				}
				else {
					_initialHop = h.getInput().get(1); // set the child that was processed
					return;	
				}
			}
		}
		
		if( h instanceof AggUnaryOp && ((AggUnaryOp) h).getOp() == AggOp.SUM 
			&& ((AggUnaryOp) h).getDirection() == Direction.RowCol)
		{
			_initialHop = h;
			_outerProductType = OutProdType.AGG_OUTER_PRODUCT;
			return;
		}
		
		memo.add(h.getHopID());
		//process parents recursively
		for( Hop parent : h.getParent())
			rfindOuterProduct(h, parent,uniqueMatrixInputs, outerProductDim1,outerProductDim2, memo);
	}
	
	////////////////Helper methods for finding boundaries 
	private OutProdType getOuterProductType(Hop X, Hop U, Hop V, Hop out)
	{
		if (_outerProductType != null)
			return _outerProductType;
				
		
		//extra checks to infer type
		if (out.getDataType() == DataType.SCALAR) // sum
		{
			_outerProductType = OutProdType.AGG_OUTER_PRODUCT;
		}
		else if( isDimsEqual(out,V) && out instanceof ReorgOp) // the second condition is added because sometimes V and U might be same dimensions if the dims of X are equal
		{
			_outerProductType = OutProdType.LEFT_OUTER_PRODUCT;
		}
		else if( isDimsEqual(out,U))
		{
			_outerProductType = OutProdType.RIGHT_OUTER_PRODUCT;
		}
		else if ( isDimsEqual(out,X) )
		{
			_outerProductType = OutProdType.CELLWISE_OUTER_PRODUCT;
		}
		
		return _outerProductType;
	}
	
	private static boolean isDimsEqual(Hop hop1, Hop hop2)
	{
		if(hop1.getDim1() == hop2.getDim1() && hop1.getDim2() == hop2.getDim2())
			return true;
		return false;
	}
	
	@Override
	public LinkedHashMap<Long, Pair<Hop[],CNodeTpl>> constructTplCplan(boolean compileLiterals) throws DMLException {

		//re-assign the dimensions of inputs to match the generated code dimensions

		//matrix X is a scalar in the generated code
		_initialCnodes.add(new CNodeData(_matrixInputs.get(0), 1,1,DataType.SCALAR));
		
		//matrix V
		_initialCnodes.add(new CNodeData(_matrixInputs.get(1), 1,(int)_matrixInputs.get(1).getDim2(), DataType.MATRIX));
		
		//matrix V
		_initialCnodes.add(new CNodeData(_matrixInputs.get(2), 1,(int)_matrixInputs.get(2).getDim2(),DataType.MATRIX));
		
		rConstructOuterProdCplan(_initialHop, _initialHop, new HashSet<Long>(), compileLiterals);
		return _cpplans;		
	}
	
	private void rConstructOuterProdCplan(Hop root, Hop hop, HashSet<Long> memo, boolean compileLiterals) throws DMLException
	{
		if( memo.contains(hop.getHopID()) )
			return;
		//process childs recursively
		for( Hop c : hop.getInput() )
			rConstructOuterProdCplan(root, c, memo, compileLiterals);
		
		//organize the main inputs
		Hop X, U, V;
		X = _matrixInputs.get(0);
		U = _matrixInputs.get(1);
		V = _matrixInputs.get(2);
		if(hop==_endHop)
			_endHopReached = true;
		
		 // first hop to enter here should be _endHop
		if(TemplateUtils.inputsAreGenerated(hop,_matrixInputs,_cpplans) && _endHopReached)  // if direct children are DataGenOps, literals, or already in the cpplans then we are ready to generate code
		{
			CNodeOuterProduct outerProdTmpl = null;
			
			//Fetch operands
			CNode out = null;
			ArrayList<CNode> addedCNodes = new ArrayList<CNode>();
			ArrayList<Hop> addedHops = new ArrayList<Hop>();
			ArrayList<CNode> cnodeData = TemplateUtils.fetchOperands(hop, _cpplans, addedCNodes, addedHops, _initialCnodes, compileLiterals);
			
			//if operands are scalar or independent from X 
			boolean independentOperands = hop != root && (hop.getDataType() == DataType.SCALAR || TemplateUtils.isOperandsIndependent(cnodeData, addedHops, new String[]{_matrixInputs.get(0).getName(),_matrixInputs.get(1).getName(),_matrixInputs.get(2).getName()}));
			if(!independentOperands)
			{
				if(hop instanceof UnaryOp)
				{
					CNode cdata1 = cnodeData.get(0);
					
					//Primitive Operation has the same name as Hop Type OpOp1
					String primitiveOpName = ((UnaryOp)hop).getOp().toString();
					out = new CNodeUnary(cdata1, UnaryType.valueOf(primitiveOpName));
				}
				else if(hop instanceof BinaryOp)
				{
					CNode cdata1 = cnodeData.get(0);
					CNode cdata2 = cnodeData.get(1);
					
					//Primitive Operation has the same name as Hop Type OpOp2
					String primitiveOpName = ((BinaryOp)hop).getOp().toString();
					
					if( (cdata1.getNumRows() > 1 && cdata1.getNumCols() == 1) || (cdata1.getNumRows() == 1 && cdata1.getNumCols() > 1) )
					{
						//second argument is always the vector
						cdata1 = new CNodeUnary(cdata1, UnaryType.LOOKUP_R);
						//out = new CNodeBinary(tmp, cdata2, BinType.valueOf(primitiveOpName));
					}
					//cdata1 is a matrix 
					else if ( (cdata1.getNumRows() > 1 && cdata1.getNumCols() > 1) )
					{
						CellTpl cellTpl = new CellTpl();
						cdata1 = cellTpl.fuseCellWise(hop.getInput().get(0), _matrixInputs.get(0), compileLiterals); // second argument is always matrix X
						if (cdata1 == null)
							return;
					}
					//cdata2 is vector
					//else if( cdata2 instanceof CNodeData && (((CNodeData)cdata2).getNumRows() > 1 && ((CNodeData)cdata2).getNumCols() == 1) || ( ((CNodeData)cdata2).getNumRows() == 1 && ((CNodeData)cdata2).getNumCols() > 1  ))
					if( (cdata2.getNumRows() > 1 && cdata2.getNumCols() == 1) || (cdata2.getNumRows() == 1 && cdata2.getNumCols() > 1) )
					{
						cdata2 = new CNodeUnary(cdata2, UnaryType.LOOKUP_R);
						//out = new CNodeBinary(cdata1, tmp, BinType.valueOf(primitiveOpName));
					}
					//cdata2 is a matrix 
					else if ( (cdata2.getNumRows() > 1 && cdata2.getNumCols() > 1) )
					{
						CellTpl cellTpl = new CellTpl();
						cdata2 = cellTpl.fuseCellWise(hop.getInput().get(1), _matrixInputs.get(0), compileLiterals); // second argument is always matrix X
						if (cdata2 == null)
							return;
					}
					out = new CNodeBinary(cdata1, cdata2, BinType.valueOf(primitiveOpName));
				}
				else if(hop instanceof AggBinaryOp)
				{
					CNode cdata1 = cnodeData.get(0);
					CNode cdata2 = cnodeData.get(1); // remember that we already fetched what is under transpose
					
					//outerproduct U%*%t(V) then we should have passsed in V as the input
					if(hop.getInput().get(0) == U && hop.getInput().get(1) instanceof ReorgOp && hop.getInput().get(1).getInput().get(0)  == V)
					{
						//re-assign cdata2 to read V instead of t(V)
						cdata2 = _initialCnodes.get(2); // the initialCNodes holds V
						out = new CNodeBinary(cdata1, cdata2, BinType.DOT_PRODUCT);
					}
					
					//outerproduct U%*%V then we should have passsed in trnasposeV as the input
					else if(hop.getInput().get(0) == U &&  V instanceof ReorgOp && V.getInput().get(0)== hop.getInput().get(1))
					{
						//re-assign cdata2 to read t(V) instead of V
						cdata2 = _initialCnodes.get(2); // the initialCNodes holds transpose of V
						out = new CNodeBinary(cdata1, cdata2, BinType.DOT_PRODUCT);
					}
					//outerproduct U%*%V  but not right wdivmm so we did not pass T(V)
					else if(hop.getInput().get(0) == U &&  hop.getInput().get(1) == V )
					{
						//re-assign cdata2 to read t(V) instead of V
						cdata2 = _initialCnodes.get(2); // the initialCNodes holds transpose of V
						out = new CNodeBinary(cdata1, cdata2, BinType.DOT_PRODUCT);
					}
					
					//left outerproduct (i.e., left operand is T(U) )
					else if(hop.getInput().get(0) instanceof ReorgOp && hop.getInput().get(0).getInput().get(0)  == U)
					{
						//scalar is cdata2
						out = new CNodeBinary(cdata2, cdata1, BinType.VECT_MULT_ADD);
					}
					
					//right outerproduct (i.e., right operand is V )
					else if(hop.getInput().get(1) != U && hop.getInput().get(1) == V)
					{
						cdata2 = _initialCnodes.get(2);
						out = new CNodeBinary(cdata1, cdata2, BinType.VECT_MULT_ADD);
					}
					
					//right outerproduct (i.e., right operand is t(V) )
					else if(hop.getInput().get(1) instanceof ReorgOp && hop.getInput().get(1).getInput().get(0)  == V)
					{
						cdata2 = _initialCnodes.get(2);
						out = new CNodeBinary(cdata1, cdata2, BinType.VECT_MULT_ADD);
					}
				}
				else if ( hop instanceof ReorgOp && ((ReorgOp)hop).getOp() == ReOrgOp.TRANSPOSE && root == hop) // if transpose wire the oinput in T( T(U ...)
				{
					out =  cnodeData.get(0);
				}
				else if (hop instanceof AggUnaryOp && ((AggUnaryOp)hop).getOp() == AggOp.SUM && root == hop
					&& ((AggUnaryOp)hop).getDirection() == Direction.RowCol )
				{
					out =  cnodeData.get(0);
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
						outerProdTmpl =  new CNodeOuterProduct(initialInputs, c); 
						outerProdTmpl.setOutProdType(getOuterProductType(X, U, V, root));
						outerProdTmpl.setTransposeOutput(_transposeOutput);
						_cpplans.put(hop.getHopID(), new Pair<Hop[],CNodeTpl>(new Hop[] {X,U,V,hop} ,outerProdTmpl));
					}
					else
					{
						initialInputs.addAll(_initialCnodes);
						initialInputs.addAll(cnodeData);
						outerProdTmpl =  new CNodeOuterProduct(initialInputs, out); 
						outerProdTmpl.setOutProdType(getOuterProductType(X, U, V, root));
						outerProdTmpl.setTransposeOutput(_transposeOutput);
								
						Hop[] hopArray = new Hop[addedHops.size()+3];
						hopArray[0] = X;
						hopArray[1] = U;
						hopArray[2] = V;
						
						System.arraycopy( addedHops.toArray(), 0, hopArray, 3, addedHops.size());
						
						_cpplans.put(hop.getHopID(), new Pair<Hop[],CNodeTpl>(hopArray,outerProdTmpl));
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
