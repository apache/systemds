/**
 * (C) Copyright IBM Corp. 2010, 2015
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * 
 */

package com.ibm.bi.dml.hops.rewrite;

import java.util.ArrayList;
import java.util.HashMap;

import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;

import com.ibm.bi.dml.hops.AggBinaryOp;
import com.ibm.bi.dml.hops.AggUnaryOp;
import com.ibm.bi.dml.hops.BinaryOp;
import com.ibm.bi.dml.hops.DataGenOp;
import com.ibm.bi.dml.hops.Hop;
import com.ibm.bi.dml.hops.Hop.OpOp1;
import com.ibm.bi.dml.hops.Hop.OpOp4;
import com.ibm.bi.dml.hops.IndexingOp;
import com.ibm.bi.dml.hops.QuaternaryOp;
import com.ibm.bi.dml.hops.TernaryOp;
import com.ibm.bi.dml.hops.UnaryOp;
import com.ibm.bi.dml.hops.Hop.AggOp;
import com.ibm.bi.dml.hops.Hop.DataGenMethod;
import com.ibm.bi.dml.hops.Hop.Direction;
import com.ibm.bi.dml.hops.Hop.ParamBuiltinOp;
import com.ibm.bi.dml.hops.Hop.ReOrgOp;
import com.ibm.bi.dml.hops.HopsException;
import com.ibm.bi.dml.hops.LiteralOp;
import com.ibm.bi.dml.hops.Hop.OpOp2;
import com.ibm.bi.dml.hops.ParameterizedBuiltinOp;
import com.ibm.bi.dml.hops.ReorgOp;
import com.ibm.bi.dml.lops.MapMultChain.ChainType;
import com.ibm.bi.dml.parser.DataExpression;
import com.ibm.bi.dml.parser.Statement;
import com.ibm.bi.dml.parser.Expression.DataType;
import com.ibm.bi.dml.parser.Expression.ValueType;

/**
 * Rule: Algebraic Simplifications. Simplifies binary expressions
 * in terms of two major purposes: (1) rewrite binary operations
 * to unary operations when possible (in CP this reduces the memory
 * estimate, in MR this allows map-only operations and hence prevents 
 * unnecessary shuffle and sort) and (2) remove binary operations that
 * are in itself are unnecessary (e.g., *1 and /1).
 * 
 */
public class RewriteAlgebraicSimplificationStatic extends HopRewriteRule
{
	
	private static final Log LOG = LogFactory.getLog(RewriteAlgebraicSimplificationDynamic.class.getName());
	
	private static OpOp2[] LOOKUP_VALID_DISTRIBUTIVE_BINARY = new OpOp2[]{OpOp2.PLUS, OpOp2.MINUS}; 
	private static OpOp2[] LOOKUP_VALID_ASSOCIATIVE_BINARY = new OpOp2[]{OpOp2.PLUS, OpOp2.MULT}; 
	private static OpOp2[] LOOKUP_VALID_WDIVMM_BINARY = new OpOp2[]{OpOp2.MULT, OpOp2.DIV}; 
	
	@Override
	public ArrayList<Hop> rewriteHopDAGs(ArrayList<Hop> roots, ProgramRewriteStatus state) 
		throws HopsException
	{
		if( roots == null )
			return roots;

		//one pass rewrite-descend (rewrite created pattern)
		for( Hop h : roots )
			rule_AlgebraicSimplification( h, false );

		Hop.resetVisitStatus(roots);
		
		//one pass descend-rewrite (for rollup) 
		for( Hop h : roots )
			rule_AlgebraicSimplification( h, true );
		
		return roots;
	}

	@Override
	public Hop rewriteHopDAG(Hop root, ProgramRewriteStatus state) 
		throws HopsException
	{
		if( root == null )
			return root;
		
		//one pass rewrite-descend (rewrite created pattern)
		rule_AlgebraicSimplification( root, false );

		root.resetVisitStatus();
		
		//one pass descend-rewrite (for rollup) 
		rule_AlgebraicSimplification( root, true );
		
		return root;
	}


	/**
	 * Note: X/y -> X * 1/y would be useful because * cheaper than / and sparsesafe; however,
	 * (1) the results would be not exactly the same (2 rounds instead of 1) and (2) it should 
	 * come before constant folding while the other simplifications should come after constant
	 * folding. Hence, not applied yet.
	 * 
	 * @throws HopsException
	 */
	private void rule_AlgebraicSimplification(Hop hop, boolean descendFirst) 
		throws HopsException 
	{
		if(hop.getVisited() == Hop.VisitStatus.DONE)
			return;
		
		//recursively process children
		for( int i=0; i<hop.getInput().size(); i++)
		{
			Hop hi = hop.getInput().get(i);
			
			//process childs recursively first (to allow roll-up)
			if( descendFirst )
				rule_AlgebraicSimplification(hi, descendFirst); //see below
			
			//apply actual simplification rewrites (of childs incl checks)
			hi = removeUnnecessaryVectorizeOperation(hi);        //e.g., matrix(1,nrow(X),ncol(X))/X -> 1/X
			hi = removeUnnecessaryBinaryOperation(hop, hi, i);   //e.g., X*1 -> X (dep: should come after rm unnecessary vectorize)
			hi = fuseDatagenAndBinaryOperation(hop, hi, i);      //e.g., rand(min=-1,max=1)*7 -> rand(min=-7,max=7)
			hi = fuseDatagenAndMinusOperation(hop, hi, i);       //e.g., -(rand(min=-2,max=1)) -> rand(min=-1,max=2)
 			hi = simplifyBinaryToUnaryOperation(hi);             //e.g., X*X -> X^2 (pow2)
 			hi = simplifyMultiBinaryToBinaryOperation(hi);       //e.g., 1-X*Y -> X 1-* Y
 			hi = simplifyDistributiveBinaryOperation(hop, hi, i);//e.g., (X-Y*X) -> (1-Y)*X
 			hi = simplifyBushyBinaryOperation(hop, hi, i);       //e.g., (X*(Y*(Z%*%v))) -> (X*Y)*(Z%*%v)
 			hi = simplifyUnaryAggReorgOperation(hop, hi, i);     //e.g., sum(t(X)) -> sum(X)
			hi = fuseBinarySubDAGToUnaryOperation(hop, hi, i);   //e.g., X*(1-X)-> sprop(X) || 1/(1+exp(-X)) -> sigmoid(X) || X*(X>0) -> selp(X)
			hi = simplifyTraceMatrixMult(hop, hi, i);            //e.g., trace(X%*%Y)->sum(X*t(Y));  
			hi = simplifySlicedMatrixMult(hop, hi, i);           //e.g., (X%*%Y)[1,1] -> X[1,] %*% Y[,1];
			hi = simplifyConstantSort(hop, hi, i);               //e.g., order(matrix())->matrix/seq; 
			hi = simplifyOrderedSort(hop, hi, i);                //e.g., order(matrix())->seq; 
			hi = removeUnnecessaryTranspose(hop, hi, i);         //e.g., t(t(X))->X; potentially introduced by diag/trace_MM
			hi = removeUnnecessaryMinus(hop, hi, i);             //e.g., -(-X)->X; potentially introduced by simplfiy binary or dyn rewrites
			hi = simplifyGroupedAggregate(hi);          	     //e.g., aggregate(target=X,groups=y,fn="count") -> aggregate(target=y,groups=y,fn="count")
			hi = simplifyWeightedSquaredLoss(hop, hi, i);        //e.g., sum(W * (X - U %*% t(V)) ^ 2) -> wsl(X, U, t(V), W, true)
			hi = simplifyWeightedSigmoidMMChains(hop, hi, i);    //e.g., W * sigmoid(Y%*%t(X)) -> wsigmoid(W, Y, t(X), type)
			hi = simplifyWeightedDivMM(hop, hi, i);              //e.g., t(U) %*% (X/(U%*%t(V))) -> wdivmm(X, U, t(V), left)
			hi = simplifyWeightedCrossEntropy(hop, hi, i);       //e.g., sum(X*log(U%*%t(V))) -> wcemm(X, U, t(V))
			hi = fuseMinusNzBinaryOperation(hop, hi, i);         //e.g., X-mean*ppred(X,0,!=) -> X -nz mean
			hi = fuseLogNzBinaryOperation(hop, hi, i);           //e.g., ppred(X,0,"!=")*log(X,0.5) -> log_nz(X,0.5)
			hi = simplifyOuterSeqExpand(hop, hi, i);             //e.g., outer(v, seq(1,m), "==") -> rexpand(v, max=m, dir=row, ignore=true, cast=false)
			hi = simplifyTableSeqExpand(hop, hi, i);             //e.g., table(seq(1,nrow(v)), v, nrow(v), m) -> rexpand(v, max=m, dir=row, ignore=false, cast=true)
			//hi = removeUnecessaryPPred(hop, hi, i);            //e.g., ppred(X,X,"==")->matrix(1,rows=nrow(X),cols=ncol(X))
			
			//process childs recursively after rewrites (to investigate pattern newly created by rewrites)
			if( !descendFirst )
				rule_AlgebraicSimplification(hi, descendFirst);
		}

		hop.setVisited(Hop.VisitStatus.DONE);
	}
	
	
	/**
	 * 
	 * @param hi
	 */
	private Hop removeUnnecessaryVectorizeOperation(Hop hi)
	{
		//applies to all binary matrix operations, if one input is unnecessarily vectorized 
		if(    hi instanceof BinaryOp && hi.getDataType()==DataType.MATRIX 
			&& ((BinaryOp)hi).supportsMatrixScalarOperations()   )
		{
			BinaryOp bop = (BinaryOp)hi;
			Hop left = bop.getInput().get(0);
			Hop right = bop.getInput().get(1);
			
			//NOTE: these rewrites of binary cell operations need to be aware that right is 
			//potentially a vector but the result is of the size of left
			//TODO move to dynamic rewrites (since size dependent to account for mv binary cell and outer operations)
			
			if( !(left.getDim1()>1 && left.getDim2()==1 && right.getDim1()==1 && right.getDim2()>1) ) // no outer
			{
				//check and remove right vectorized scalar
				if( left.getDataType() == DataType.MATRIX && right instanceof DataGenOp )
				{
					DataGenOp dright = (DataGenOp) right;
					if( dright.getOp()==DataGenMethod.RAND && dright.hasConstantValue() )
					{
						Hop drightIn = dright.getInput().get(dright.getParamIndex(DataExpression.RAND_MIN));
						HopRewriteUtils.removeChildReference(bop, dright);
						HopRewriteUtils.addChildReference(bop, drightIn, 1);
						//cleanup if only consumer of intermediate
						if( dright.getParent().isEmpty() ) 
							HopRewriteUtils.removeAllChildReferences( dright );
						
						LOG.debug("Applied removeUnnecessaryVectorizeOperation1");
					}
				}
				//check and remove left vectorized scalar
				else if( right.getDataType() == DataType.MATRIX && left instanceof DataGenOp )
				{
					DataGenOp dleft = (DataGenOp) left;
					if( dleft.getOp()==DataGenMethod.RAND && dleft.hasConstantValue()
						&& (left.getDim2()==1 || right.getDim2()>1) 
						&& (left.getDim1()==1 || right.getDim1()>1))
					{
						Hop dleftIn = dleft.getInput().get(dleft.getParamIndex(DataExpression.RAND_MIN));
						HopRewriteUtils.removeChildReference(bop, dleft);
						HopRewriteUtils.addChildReference(bop, dleftIn, 0);
						//cleanup if only consumer of intermediate
						if( dleft.getParent().isEmpty() ) 
							HopRewriteUtils.removeAllChildReferences( dleft );
						
						LOG.debug("Applied removeUnnecessaryVectorizeOperation2");
					}
				}

				//Note: we applied this rewrite to at most one side in order to keep the
				//output semantically equivalent. However, future extensions might consider
				//to remove vectors from both side, compute the binary op on scalars and 
				//finally feed it into a datagenop of the original dimensions.
			}
		}
		
		return hi;
	}
	
	
	/**
	 * handle removal of unnecessary binary operations
	 * 
	 * X/1 or X*1 or 1*X or X-0 -> X
	 * -1*X or X*-1-> -X		
	 * 
	 * @param parent
	 * @param hi
	 * @param pos
	 * @throws HopsException
	 */
	private Hop removeUnnecessaryBinaryOperation( Hop parent, Hop hi, int pos ) 
		throws HopsException
	{
		if( hi instanceof BinaryOp )
		{
			BinaryOp bop = (BinaryOp)hi;
			Hop left = bop.getInput().get(0);
			Hop right = bop.getInput().get(1);
			//X/1 or X*1 -> X 
			if(    left.getDataType()==DataType.MATRIX 
				&& right instanceof LiteralOp && ((LiteralOp)right).getDoubleValue()==1.0 )
			{
				if( bop.getOp()==OpOp2.DIV || bop.getOp()==OpOp2.MULT )
				{
					HopRewriteUtils.removeChildReference(parent, bop);
					HopRewriteUtils.addChildReference(parent, left, pos);
					hi = left;

					LOG.debug("Applied removeUnnecessaryBinaryOperation1 (line "+bop.getBeginLine()+")");
				}
			}
			//X-0 -> X 
			else if(    left.getDataType()==DataType.MATRIX 
					&& right instanceof LiteralOp && ((LiteralOp)right).getDoubleValue()==0.0 )
			{
				if( bop.getOp()==OpOp2.MINUS )
				{
					HopRewriteUtils.removeChildReference(parent, bop);
					HopRewriteUtils.addChildReference(parent, left, pos);
					hi = left;

					LOG.debug("Applied removeUnnecessaryBinaryOperation2 (line "+bop.getBeginLine()+")");
				}
			}
			//1*X -> X
			else if(   right.getDataType()==DataType.MATRIX 
					&& left instanceof LiteralOp && ((LiteralOp)left).getDoubleValue()==1.0 )
			{
				if( bop.getOp()==OpOp2.MULT )
				{
					HopRewriteUtils.removeChildReference(parent, bop);
					HopRewriteUtils.addChildReference(parent, right, pos);
					hi = right;

					LOG.debug("Applied removeUnnecessaryBinaryOperation3 (line "+bop.getBeginLine()+")");
				}
			}
			//-1*X -> -X
			//note: this rewrite is necessary since the new antlr parser always converts 
			//-X to -1*X due to mechanical reasons
			else if(   right.getDataType()==DataType.MATRIX 
					&& left instanceof LiteralOp && ((LiteralOp)left).getDoubleValue()==-1.0 )
			{
				if( bop.getOp()==OpOp2.MULT )
				{
					bop.setOp(OpOp2.MINUS);
					HopRewriteUtils.removeChildReferenceByPos(bop, left, 0);
					HopRewriteUtils.addChildReference(bop, new LiteralOp("0",0), 0);
					hi = bop;

					LOG.debug("Applied removeUnnecessaryBinaryOperation4 (line "+bop.getBeginLine()+")");
				}
			}
			//X*-1 -> -X (see comment above)
			else if(   left.getDataType()==DataType.MATRIX 
					&& right instanceof LiteralOp && ((LiteralOp)right).getDoubleValue()==-1.0 )
			{
				if( bop.getOp()==OpOp2.MULT )
				{
					bop.setOp(OpOp2.MINUS);
					HopRewriteUtils.removeChildReferenceByPos(bop, right, 1);
					HopRewriteUtils.addChildReference(bop, new LiteralOp("0",0), 0);
					hi = bop;
					
					LOG.debug("Applied removeUnnecessaryBinaryOperation5 (line "+bop.getBeginLine()+")");
				}
			}
		}
		
		return hi;
	}
	
	/**
	 * handle removal of unnecessary binary operations over rand data
	 * 
	 * rand*7 -> rand(min*7,max*7); rand+7 -> rand(min+7,max+7);
	 * 
	 * @param parent
	 * @param hi
	 * @param pos
	 * @return
	 * @throws HopsException
	 */
	private Hop fuseDatagenAndBinaryOperation( Hop parent, Hop hi, int pos ) 
		throws HopsException
	{
		if( hi instanceof BinaryOp )
		{
			BinaryOp bop = (BinaryOp)hi;
			Hop left = bop.getInput().get(0);
			Hop right = bop.getInput().get(1);
			
			//left input rand and hence output matrix double, right scalar literal
			if( left instanceof DataGenOp && ((DataGenOp)left).getOp()==DataGenMethod.RAND &&
				right instanceof LiteralOp )
			{
				DataGenOp inputGen = (DataGenOp)left;
				HashMap<String,Integer> params = inputGen.getParamIndexMap();
				Hop min = left.getInput().get(params.get(DataExpression.RAND_MIN));
				Hop max = left.getInput().get(params.get(DataExpression.RAND_MAX));
				double sval = ((LiteralOp)right).getDoubleValue();
				
				if( (bop.getOp()==OpOp2.MULT || bop.getOp()==OpOp2.PLUS)
					&& min instanceof LiteralOp && max instanceof LiteralOp )
				{
					//create fused data gen operator
					DataGenOp gen = null;
					if( bop.getOp()==OpOp2.MULT )
						gen = HopRewriteUtils.copyDataGenOp(inputGen, sval, 0);
					else //if( bop.getOp()==OpOp2.PLUS )		
						gen = HopRewriteUtils.copyDataGenOp(inputGen, 1, sval);
						
					//rewire parents
					HopRewriteUtils.removeChildReference(parent, bop);
					HopRewriteUtils.addChildReference(parent, gen, pos);
					
					//propagate potentially updated nnz=0
					parent.refreshSizeInformation(); 
					
					hi = gen;
					
					LOG.debug("Applied fuseDatagenAndBinaryOperation1");
				}
			}
			//right input rand and hence output matrix double, left scalar literal
			else if( right instanceof DataGenOp && ((DataGenOp)right).getOp()==DataGenMethod.RAND &&
				left instanceof LiteralOp )
			{
				DataGenOp inputGen = (DataGenOp)right;
				HashMap<String,Integer> params = inputGen.getParamIndexMap();
				Hop min = right.getInput().get(params.get(DataExpression.RAND_MIN));
				Hop max = right.getInput().get(params.get(DataExpression.RAND_MAX));
				double sval = ((LiteralOp)left).getDoubleValue();
				
				if( (bop.getOp()==OpOp2.MULT || bop.getOp()==OpOp2.PLUS)
					&& min instanceof LiteralOp && max instanceof LiteralOp )
				{
					//create fused data gen operator
					DataGenOp gen = null;
					if( bop.getOp()==OpOp2.MULT )
						gen = HopRewriteUtils.copyDataGenOp(inputGen, sval, 0);
					else //if( bop.getOp()==OpOp2.PLUS )		
						gen = HopRewriteUtils.copyDataGenOp(inputGen, 1, sval);
						
					//rewire parents
					HopRewriteUtils.removeChildReference(parent, bop);
					HopRewriteUtils.addChildReference(parent, gen, pos);

					//propagate potentially updated nnz=0
					parent.refreshSizeInformation(); 
					
					hi = gen;
					
					LOG.debug("Applied fuseDatagenAndBinaryOperation2");
				}
			}
		}
		
		return hi;
	}
	
	/**
	 * 
	 * @param parent
	 * @param hi
	 * @param pos
	 * @return
	 * @throws HopsException
	 */
	private Hop fuseDatagenAndMinusOperation( Hop parent, Hop hi, int pos ) 
		throws HopsException
	{
		if( hi instanceof BinaryOp )
		{
			BinaryOp bop = (BinaryOp)hi;
			Hop left = bop.getInput().get(0);
			Hop right = bop.getInput().get(1);
			
			if( right instanceof DataGenOp && ((DataGenOp)right).getOp()==DataGenMethod.RAND &&
				left instanceof LiteralOp && ((LiteralOp)left).getDoubleValue()==0.0 )
			{
				DataGenOp inputGen = (DataGenOp)right;
				HashMap<String,Integer> params = inputGen.getParamIndexMap();
				int ixMin = params.get(DataExpression.RAND_MIN);
				int ixMax = params.get(DataExpression.RAND_MAX);
				Hop min = right.getInput().get(ixMin);
				Hop max = right.getInput().get(ixMax);
				
				//apply rewrite under additional conditions (for simplicity)
				if( inputGen.getParent().size()==1 
					&& min instanceof LiteralOp && max instanceof LiteralOp )
				{
					//exchange and *-1 (special case 0 stays 0 instead of -0 for consistency)
					double newMinVal = (((LiteralOp)max).getDoubleValue()==0)?0:(-1 * ((LiteralOp)max).getDoubleValue());
					double newMaxVal = (((LiteralOp)min).getDoubleValue()==0)?0:(-1 * ((LiteralOp)min).getDoubleValue());
					Hop newMin = new LiteralOp(String.valueOf(newMinVal), newMinVal);
					Hop newMax = new LiteralOp(String.valueOf(newMaxVal), newMaxVal);
					
					HopRewriteUtils.removeChildReferenceByPos(inputGen, min, ixMin);
					HopRewriteUtils.addChildReference(inputGen, newMin, ixMin);
					HopRewriteUtils.removeChildReferenceByPos(inputGen, max, ixMax);
					HopRewriteUtils.addChildReference(inputGen, newMax, ixMax);
					
					HopRewriteUtils.removeChildReference(parent, bop);
					HopRewriteUtils.addChildReference(parent, inputGen, pos);
					hi = inputGen;

					LOG.debug("Applied fuseDatagenAndMinusOperation");		
				}
			}
		}
		
		return hi;
	}
	
	/**
	 * handle simplification of binary operations
	 * (relies on previous common subexpression elimination)
	 * 
	 * X+X -> X*2 or X*X -> X^2
	 */
	private Hop simplifyBinaryToUnaryOperation( Hop hi )
	{
		if( hi instanceof BinaryOp )
		{
			BinaryOp bop = (BinaryOp)hi;
			Hop left = hi.getInput().get(0);
			Hop right = hi.getInput().get(1);
			if( left == right && left.getDataType()==DataType.MATRIX )
			{
				//note: we simplify this to unary operations first (less mem and better MR plan),
				//however, we later compile specific LOPS for X*2 and X^2
				if( bop.getOp()==OpOp2.PLUS ) //X+X -> X*2
				{
					bop.setOp(OpOp2.MULT);
					LiteralOp tmp = new LiteralOp("2", 2);
					bop.getInput().remove(1);
					right.getParent().remove(bop);
					HopRewriteUtils.addChildReference(hi, tmp, 1);

					LOG.debug("Applied simplifyBinaryToUnaryOperation1");
				}
				else if ( bop.getOp()==OpOp2.MULT ) //X*X -> X^2
				{
					bop.setOp(OpOp2.POW);
					LiteralOp tmp = new LiteralOp("2", 2);
					bop.getInput().remove(1);
					right.getParent().remove(bop);
					HopRewriteUtils.addChildReference(hi, tmp, 1);
					
					LOG.debug("Applied simplifyBinaryToUnaryOperation2");
				}
			}
		}
		
		return hi;
	}
	
	/**
	 * 
	 * @param hi
	 * @return
	 */
	private Hop simplifyMultiBinaryToBinaryOperation( Hop hi )
	{
		//pattern: 1-(X*Y) --> X 1-* Y (avoid intermediate)
		if( hi instanceof BinaryOp && ((BinaryOp)hi).getOp()==OpOp2.MINUS
			&& hi.getDataType() == DataType.MATRIX	
			&& hi.getInput().get(0) instanceof LiteralOp
			&& HopRewriteUtils.getDoubleValueSafe((LiteralOp)hi.getInput().get(0))==1
			&& hi.getInput().get(1) instanceof BinaryOp
			&& ((BinaryOp)hi.getInput().get(1)).getOp()==OpOp2.MULT
			&& hi.getInput().get(1).getParent().size() == 1 ) //single consumer
		{
			BinaryOp bop = (BinaryOp)hi;
			Hop left = hi.getInput().get(1).getInput().get(0);
			Hop right = hi.getInput().get(1).getInput().get(1);
			
			//set new binaryop type and rewire inputs
			bop.setOp(OpOp2.MINUS1_MULT);
			HopRewriteUtils.removeAllChildReferences(hi);
			HopRewriteUtils.addChildReference(bop, left);
			HopRewriteUtils.addChildReference(bop, right);
			
			LOG.debug("Applied simplifyMultiBinaryToBinaryOperation.");
		}
		
		return hi;
	}
	
	/**
	 * (X-Y*X) -> (1-Y)*X,    (Y*X-X) -> (Y-1)*X
	 * (X+Y*X) -> (1+Y)*X,    (Y*X+X) -> (Y+1)*X
	 * 
	 * 
	 * @param parent
	 * @param hi
	 * @param pos
	 * @return
	 */
	private Hop simplifyDistributiveBinaryOperation( Hop parent, Hop hi, int pos )
	{
		
		if( hi instanceof BinaryOp )
		{
			BinaryOp bop = (BinaryOp)hi;
			Hop left = bop.getInput().get(0);
			Hop right = bop.getInput().get(1);
			
			//(X+Y*X) -> (1+Y)*X,    (Y*X+X) -> (Y+1)*X
			//(X-Y*X) -> (1-Y)*X,    (Y*X-X) -> (Y-1)*X
			boolean applied = false;
			if( left.getDataType()==DataType.MATRIX && right.getDataType()==DataType.MATRIX 
				&& HopRewriteUtils.isValidOp(bop.getOp(), LOOKUP_VALID_DISTRIBUTIVE_BINARY) )
			{
				Hop X = null; Hop Y = null;
				if( left instanceof BinaryOp && ((BinaryOp)left).getOp()==OpOp2.MULT ) //(Y*X-X) -> (Y-1)*X
				{
					Hop leftC1 = left.getInput().get(0);
					Hop leftC2 = left.getInput().get(1);
					//System.out.println("aOp2:"+((BinaryOp)left).getOp()+": "+leftC1.getName()+" "+leftC2.getName());
						
					if( leftC1.getDataType()==DataType.MATRIX && leftC2.getDataType()==DataType.MATRIX &&
						(right == leftC1 || right == leftC2) && leftC1 !=leftC2 ){ //any mult order
						X = right;
						Y = ( right == leftC1 ) ? leftC2 : leftC1;
					}
					if( X != null ){ //rewrite 'binary +/-' 
						HopRewriteUtils.removeChildReference(parent, hi);
						LiteralOp literal = new LiteralOp("1",1);
						BinaryOp plus = new BinaryOp(right.getName(), right.getDataType(), right.getValueType(), bop.getOp(), Y, literal);
						HopRewriteUtils.refreshOutputParameters(plus, right);						
						BinaryOp mult = new BinaryOp(left.getName(), left.getDataType(), left.getValueType(), OpOp2.MULT, plus, X);
						HopRewriteUtils.refreshOutputParameters(mult, left);
						
						HopRewriteUtils.addChildReference(parent, mult, pos);							
						hi = mult;
						applied = true;
						
						LOG.debug("Applied simplifyDistributiveBinaryOperation1");
					}					
				}	
				
				if( !applied && right instanceof BinaryOp && ((BinaryOp)right).getOp()==OpOp2.MULT ) //(X-Y*X) -> (1-Y)*X
				{
					Hop rightC1 = right.getInput().get(0);
					Hop rightC2 = right.getInput().get(1);
					if( rightC1.getDataType()==DataType.MATRIX && rightC2.getDataType()==DataType.MATRIX &&
						(left == rightC1 || left == rightC2) && rightC1 !=rightC2 ){ //any mult order
						X = left;
						Y = ( left == rightC1 ) ? rightC2 : rightC1;
					}
					if( X != null ){ //rewrite '+/- binary'
						HopRewriteUtils.removeChildReference(parent, hi);
						LiteralOp literal = new LiteralOp("1",1);
						BinaryOp plus = new BinaryOp(left.getName(), left.getDataType(), left.getValueType(), bop.getOp(), literal, Y);
						HopRewriteUtils.refreshOutputParameters(plus, left);						
						BinaryOp mult = new BinaryOp(right.getName(), right.getDataType(), right.getValueType(), OpOp2.MULT, plus, X);
						HopRewriteUtils.refreshOutputParameters(mult, right);
						
						HopRewriteUtils.addChildReference(parent, mult, pos);	
						hi = mult;

						LOG.debug("Applied simplifyDistributiveBinaryOperation2");
					}
				}	
			}
		}
		
		return hi;
	}
	
	/**
	 * (X*(Y*(Z%*%v))) -> (X*Y)*(Z%*%v)
	 * (X+(Y+(Z%*%v))) -> (X+Y)+(Z%*%v)
	 * 
	 * Note: Restriction ba() at leaf and root instead of data at leaf to not reorganize too
	 * eagerly, which would loose additional rewrite potential. This rewrite has two goals
	 * (1) enable XtwXv, and increase piggybacking potential by creating bushy trees.
	 * 
	 * @param parent
	 * @param hi
	 * @param pos
	 * @return
	 */
	private Hop simplifyBushyBinaryOperation( Hop parent, Hop hi, int pos )
	{
		if( hi instanceof BinaryOp && parent instanceof AggBinaryOp )
		{
			BinaryOp bop = (BinaryOp)hi;
			Hop left = bop.getInput().get(0);
			Hop right = bop.getInput().get(1);
			OpOp2 op = bop.getOp();
			
			if( left.getDataType()==DataType.MATRIX && right.getDataType()==DataType.MATRIX &&
				HopRewriteUtils.isValidOp(op, LOOKUP_VALID_ASSOCIATIVE_BINARY) )
			{
				boolean applied = false;
				
				if( right instanceof BinaryOp )
				{
					BinaryOp bop2 = (BinaryOp)right;
					Hop left2 = bop2.getInput().get(0);
					Hop right2 = bop2.getInput().get(1);
					OpOp2 op2 = bop2.getOp();
					
					if( op==op2 && right2.getDataType()==DataType.MATRIX 
						&& (right2 instanceof AggBinaryOp) )
					{
						//(X*(Y*op()) -> (X*Y)*op()
						HopRewriteUtils.removeChildReference(parent, bop);
						
						BinaryOp bop3 = new BinaryOp("tmp1", DataType.MATRIX, ValueType.DOUBLE, op, left, left2);
						HopRewriteUtils.refreshOutputParameters(bop3, bop);
						BinaryOp bop4 = new BinaryOp("tmp2", DataType.MATRIX, ValueType.DOUBLE, op, bop3, right2);
						HopRewriteUtils.refreshOutputParameters(bop4, bop2);
						
						HopRewriteUtils.addChildReference(parent, bop4, pos);	
						hi = bop4;
						
						applied = true;
						
						LOG.debug("Applied simplifyBushyBinaryOperation1");
					}
				}
				
				if( !applied && left instanceof BinaryOp )
				{
					BinaryOp bop2 = (BinaryOp)left;
					Hop left2 = bop2.getInput().get(0);
					Hop right2 = bop2.getInput().get(1);
					OpOp2 op2 = bop2.getOp();
					
					if( op==op2 && left2.getDataType()==DataType.MATRIX 
						&& (left2 instanceof AggBinaryOp) 
						&& (right2.getDim2() > 1 || right.getDim2() == 1)   //X not vector, or Y vector
						&& (right2.getDim1() > 1 || right.getDim1() == 1) ) //X not vector, or Y vector
					{
						//((op()*X)*Y) -> op()*(X*Y)
						HopRewriteUtils.removeChildReference(parent, bop);
						
						BinaryOp bop3 = new BinaryOp("tmp1", DataType.MATRIX, ValueType.DOUBLE, op, right2, right);
						HopRewriteUtils.refreshOutputParameters(bop3, bop2);
						BinaryOp bop4 = new BinaryOp("tmp2", DataType.MATRIX, ValueType.DOUBLE, op, left2, bop3);
						HopRewriteUtils.refreshOutputParameters(bop4, bop);
						
						HopRewriteUtils.addChildReference(parent, bop4, pos);	
						hi = bop4;
						
						LOG.debug("Applied simplifyBushyBinaryOperation2");
					}
				}
			}
			
		}
		
		return hi;
	}
	
	/**
	 * 
	 * @param parent
	 * @param hi
	 * @param pos
	 * @return
	 */
	private Hop simplifyUnaryAggReorgOperation( Hop parent, Hop hi, int pos )
	{
		if(   hi instanceof AggUnaryOp && ((AggUnaryOp)hi).getDirection()==Direction.RowCol  //full uagg
		   && hi.getInput().get(0) instanceof ReorgOp  ) //reorg operation
		{
			ReorgOp rop = (ReorgOp)hi.getInput().get(0);
			if(   (rop.getOp()==ReOrgOp.TRANSPOSE || rop.getOp()==ReOrgOp.RESHAPE)         //valid reorg
				&& rop.getParent().size()==1 )                                //uagg only reorg consumer
			{
				Hop input = rop.getInput().get(0);
				HopRewriteUtils.removeAllChildReferences(hi);
				HopRewriteUtils.removeAllChildReferences(rop);
				HopRewriteUtils.addChildReference(hi, input);
				
				LOG.debug("Applied simplifyUnaryAggReorgOperation");
			}
		}
		
		return hi;
	}
	
	/**
	 * handle simplification of more complex sub DAG to unary operation.
	 * 
	 * X*(1-X) -> sprop(X)
	 * (1-X)*X -> sprop(X)
	 * 1/(1+exp(-X)) -> sigmoid(X)
	 * 
	 * @param hi
	 * @throws HopsException 
	 */
	private Hop fuseBinarySubDAGToUnaryOperation( Hop parent, Hop hi, int pos ) 
		throws HopsException
	{
		if( hi instanceof BinaryOp )
		{
			BinaryOp bop = (BinaryOp)hi;
			Hop left = hi.getInput().get(0);
			Hop right = hi.getInput().get(1);
			
			//sample proportion (sprop) operator
			if( bop.getOp() == OpOp2.MULT && left.getDataType()==DataType.MATRIX && right.getDataType()==DataType.MATRIX )
			{
				//by definition, either left or right or none applies. 
				//note: if there are multiple consumers on the intermediate,
				//we follow the heuristic that redundant computation is more beneficial, 
				//i.e., we still fuse but leave the intermediate for the other consumers  
				
				if( left instanceof BinaryOp ) //(1-X)*X
				{
					BinaryOp bleft = (BinaryOp)left;
					Hop left1 = bleft.getInput().get(0);
					Hop left2 = bleft.getInput().get(1);		
				
					if( left1 instanceof LiteralOp &&
						HopRewriteUtils.getDoubleValue((LiteralOp)left1)==1 &&	
						left2 == right && bleft.getOp() == OpOp2.MINUS  ) 
					{
						UnaryOp unary = HopRewriteUtils.createUnary(right, OpOp1.SPROP);
						HopRewriteUtils.removeChildReferenceByPos(parent, bop, pos);
						HopRewriteUtils.addChildReference(parent, unary, pos);
						
						//cleanup if only consumer of intermediate
						if( bop.getParent().isEmpty() )
							HopRewriteUtils.removeAllChildReferences(bop);					
						if( left.getParent().isEmpty() ) 
							HopRewriteUtils.removeAllChildReferences(left);
						
						hi = unary;
						
						LOG.debug("Applied fuseBinarySubDAGToUnaryOperation-sprop1");
					}
				}				
				if( right instanceof BinaryOp ) //X*(1-X)
				{
					BinaryOp bright = (BinaryOp)right;
					Hop right1 = bright.getInput().get(0);
					Hop right2 = bright.getInput().get(1);		
				
					if( right1 instanceof LiteralOp &&
						HopRewriteUtils.getDoubleValue((LiteralOp)right1)==1 &&	
						right2 == left && bright.getOp() == OpOp2.MINUS )
					{
						UnaryOp unary = HopRewriteUtils.createUnary(left, OpOp1.SPROP);
						HopRewriteUtils.removeChildReferenceByPos(parent, bop, pos);
						HopRewriteUtils.addChildReference(parent, unary, pos);
						
						//cleanup if only consumer of intermediate
						if( bop.getParent().isEmpty() )
							HopRewriteUtils.removeAllChildReferences(bop);					
						if( left.getParent().isEmpty() ) 
							HopRewriteUtils.removeAllChildReferences(right);
						
						hi = unary;
						
						LOG.debug("Applied fuseBinarySubDAGToUnaryOperation-sprop2");
					}
				}
			}
			//sigmoid operator
			else if( bop.getOp() == OpOp2.DIV && left.getDataType()==DataType.SCALAR && right.getDataType()==DataType.MATRIX
					 && left instanceof LiteralOp && HopRewriteUtils.getDoubleValue((LiteralOp)left)==1 && right instanceof BinaryOp)
			{
				//note: if there are multiple consumers on the intermediate,
				//we follow the heuristic that redundant computation is more beneficial, 
				//i.e., we still fuse but leave the intermediate for the other consumers  
				
				BinaryOp bop2 = (BinaryOp)right;
				Hop left2 = bop2.getInput().get(0);
				Hop right2 = bop2.getInput().get(1);
				
				if(    bop2.getOp() == OpOp2.PLUS && left2.getDataType()==DataType.SCALAR && right2.getDataType()==DataType.MATRIX
				    && left2 instanceof LiteralOp && HopRewriteUtils.getDoubleValue((LiteralOp)left2)==1 && right2 instanceof UnaryOp)
				{
					UnaryOp uop = (UnaryOp) right2;
					Hop uopin = uop.getInput().get(0);
					
					if( uop.getOp()==OpOp1.EXP ) 
					{
						UnaryOp unary = null;
						
						//Pattern 1: (1/(1 + exp(-X)) 
						if( uopin instanceof BinaryOp && ((BinaryOp)uopin).getOp()==OpOp2.MINUS )
						{
							BinaryOp bop3 = (BinaryOp) uopin;
							Hop left3 = bop3.getInput().get(0);
							Hop right3 = bop3.getInput().get(1);
							
							if( left3 instanceof LiteralOp && HopRewriteUtils.getDoubleValue((LiteralOp)left3)==0 ) {
								unary = HopRewriteUtils.createUnary(right3, OpOp1.SIGMOID);
							}	
						}						
						//Pattern 2: (1/(1 + exp(X)), e.g., where -(-X) has been removed by 
						//the 'remove unnecessary minus' rewrite --> reintroduce the minus
						else
						{
							BinaryOp minus = HopRewriteUtils.createMinus(uopin);
							unary = HopRewriteUtils.createUnary(minus, OpOp1.SIGMOID);
						}	
					
						if( unary != null )
						{
							HopRewriteUtils.removeChildReferenceByPos(parent, bop, pos);
							HopRewriteUtils.addChildReference(parent, unary, pos);
							
							//cleanup if only consumer of intermediate
							if( bop.getParent().isEmpty() )
								HopRewriteUtils.removeAllChildReferences(bop);	
							if( bop2.getParent().isEmpty() )
								HopRewriteUtils.removeAllChildReferences(bop2);	
							if( uop.getParent().isEmpty() )
								HopRewriteUtils.removeAllChildReferences(uop);	
							
							hi = unary;
							
							LOG.debug("Applied fuseBinarySubDAGToUnaryOperation-sigmoid1");
						}				
					}
				}		
			}
			//select positive (selp) operator
			if( bop.getOp() == OpOp2.MULT && left.getDataType()==DataType.MATRIX && right.getDataType()==DataType.MATRIX )
			{
				//by definition, either left or right or none applies. 
				//note: if there are multiple consumers on the intermediate tmp=(X>0), it's still beneficial
				//to replace the X*tmp with selp(X) due to lower memory requirements and simply sparsity propagation 
				
				if( left instanceof BinaryOp ) //(X>0)*X
				{
					BinaryOp bleft = (BinaryOp)left;
					Hop left1 = bleft.getInput().get(0);
					Hop left2 = bleft.getInput().get(1);		
				
					if( left2 instanceof LiteralOp &&
						HopRewriteUtils.getDoubleValue((LiteralOp)left2)==0 &&	
						left1 == right && bleft.getOp() == OpOp2.GREATER  ) 
					{
						UnaryOp unary = HopRewriteUtils.createUnary(right, OpOp1.SELP);
						HopRewriteUtils.removeChildReferenceByPos(parent, bop, pos);
						HopRewriteUtils.addChildReference(parent, unary, pos);
						
						//cleanup if only consumer of intermediate
						if( bop.getParent().isEmpty() )
							HopRewriteUtils.removeAllChildReferences(bop);					
						if( left.getParent().isEmpty() ) 
							HopRewriteUtils.removeAllChildReferences(left);
						
						hi = unary;
						
						LOG.debug("Applied fuseBinarySubDAGToUnaryOperation-selp1");
					}
				}				
				if( right instanceof BinaryOp ) //X*(X>0)
				{
					BinaryOp bright = (BinaryOp)right;
					Hop right1 = bright.getInput().get(0);
					Hop right2 = bright.getInput().get(1);		
				
					if( right2 instanceof LiteralOp &&
						HopRewriteUtils.getDoubleValue((LiteralOp)right2)==0 &&	
						right1 == left && bright.getOp() == OpOp2.GREATER )
					{
						UnaryOp unary = HopRewriteUtils.createUnary(left, OpOp1.SELP);
						HopRewriteUtils.removeChildReferenceByPos(parent, bop, pos);
						HopRewriteUtils.addChildReference(parent, unary, pos);
						
						//cleanup if only consumer of intermediate
						if( bop.getParent().isEmpty() )
							HopRewriteUtils.removeAllChildReferences(bop);					
						if( left.getParent().isEmpty() ) 
							HopRewriteUtils.removeAllChildReferences(right);
						
						hi = unary;
						
						LOG.debug("Applied fuseBinarySubDAGToUnaryOperation-selp2");
					}
				}
			}
		}
		
		return hi;
	}
	
	
	/**
	 * 
	 * @param parent
	 * @param hi
	 * @param pos
	 * @return
	 */
	private Hop simplifyTraceMatrixMult(Hop parent, Hop hi, int pos)
	{
		if( hi instanceof AggUnaryOp && ((AggUnaryOp)hi).getOp()==AggOp.TRACE ) //trace()
		{
			Hop hi2 = hi.getInput().get(0);
			if( hi2 instanceof AggBinaryOp && ((AggBinaryOp)hi2).isMatrixMultiply() ) //X%*%Y
			{
				Hop left = hi2.getInput().get(0);
				Hop right = hi2.getInput().get(1);
				
				//remove link from parent to diag
				HopRewriteUtils.removeChildReference(parent, hi);
				
				//remove links to inputs to matrix mult
				//removeChildReference(hi2, left);
				//removeChildReference(hi2, right);
				
				//create new operators (incl refresh size inside for transpose)
				ReorgOp trans = HopRewriteUtils.createTranspose(right);
				BinaryOp mult = new BinaryOp(right.getName(), right.getDataType(), right.getValueType(), OpOp2.MULT, left, trans);
				mult.setRowsInBlock(right.getRowsInBlock());
				mult.setColsInBlock(right.getColsInBlock());
				mult.refreshSizeInformation();
				AggUnaryOp sum = new AggUnaryOp(right.getName(), DataType.SCALAR, right.getValueType(), AggOp.SUM, Direction.RowCol, mult);
				sum.refreshSizeInformation();
				
				//rehang new subdag under parent node
				HopRewriteUtils.addChildReference(parent, sum, pos);
				
				//cleanup if only consumer of intermediate
				if( hi.getParent().isEmpty() ) 
					HopRewriteUtils.removeAllChildReferences( hi );
				if( hi2.getParent().isEmpty() ) 
					HopRewriteUtils.removeAllChildReferences( hi2 );
				
				hi = sum;
				
				LOG.debug("Applied simplifyTraceMatrixMult");
			}	
		}
		
		return hi;
	}
	
	/**
	 * 
	 * @param parent
	 * @param hi
	 * @param pos
	 * @return
	 * @throws HopsException 
	 */
	private Hop simplifySlicedMatrixMult(Hop parent, Hop hi, int pos) 
		throws HopsException
	{
		//e.g., (X%*%Y)[1,1] -> X[1,] %*% Y[,1] 
		if( hi instanceof IndexingOp 
			&& ((IndexingOp)hi).getRowLowerEqualsUpper()
			&& ((IndexingOp)hi).getColLowerEqualsUpper()  
			&& hi.getInput().get(0).getParent().size()==1 //rix is single mm consumer
			&& hi.getInput().get(0) instanceof AggBinaryOp 
			&& ((AggBinaryOp)hi.getInput().get(0)).isMatrixMultiply() )
		{
			Hop mm = hi.getInput().get(0);
			Hop X = mm.getInput().get(0);
			Hop Y = mm.getInput().get(1);
			Hop rowExpr = hi.getInput().get(1); //rl==ru
			Hop colExpr = hi.getInput().get(3); //cl==cu
			
			HopRewriteUtils.removeAllChildReferences(mm);
			
			//create new indexing operations
			IndexingOp ix1 = new IndexingOp("tmp1", DataType.MATRIX, ValueType.DOUBLE, X, 
					rowExpr, rowExpr, new LiteralOp("1",1), HopRewriteUtils.createValueHop(X, false), true, false);
			HopRewriteUtils.setOutputBlocksizes(ix1, X.getRowsInBlock(), X.getColsInBlock());
			ix1.refreshSizeInformation();
			IndexingOp ix2 = new IndexingOp("tmp2", DataType.MATRIX, ValueType.DOUBLE, Y, 
					new LiteralOp("1",1), HopRewriteUtils.createValueHop(Y, true), colExpr, colExpr, false, true);
			HopRewriteUtils.setOutputBlocksizes(ix2, Y.getRowsInBlock(), Y.getColsInBlock());
			ix2.refreshSizeInformation();
			
			//rewire matrix mult over ix1 and ix2
			HopRewriteUtils.addChildReference(mm, ix1, 0);
			HopRewriteUtils.addChildReference(mm, ix2, 1);
			mm.refreshSizeInformation();
			
			hi = mm;
				
			LOG.debug("Applied simplifySlicedMatrixMult");	
		}
		
		return hi;
	}
	
	/**
	 * 
	 * @param parent
	 * @param hi
	 * @param pos
	 * @return
	 * @throws HopsException
	 */
	private Hop simplifyConstantSort(Hop parent, Hop hi, int pos) 
		throws HopsException
	{
		//order(matrix(7), indexreturn=FALSE) -> matrix(7)
		//order(matrix(7), indexreturn=TRUE) -> seq(1,nrow(X),1)
		if( hi instanceof ReorgOp && ((ReorgOp)hi).getOp()==ReOrgOp.SORT )  //order
		{
			Hop hi2 = hi.getInput().get(0);
			
			if( hi2 instanceof DataGenOp && ((DataGenOp)hi2).getOp()==DataGenMethod.RAND
				&& ((DataGenOp)hi2).hasConstantValue() 
				&& hi.getInput().get(3) instanceof LiteralOp ) //known indexreturn
			{
				if( HopRewriteUtils.getBooleanValue((LiteralOp)hi.getInput().get(3)) )
				{
					//order(matrix(7), indexreturn=TRUE) -> seq(1,nrow(X),1)
					HopRewriteUtils.removeChildReferenceByPos(parent, hi, pos);
					Hop seq = HopRewriteUtils.createSeqDataGenOp(hi2);
					seq.refreshSizeInformation();
					HopRewriteUtils.addChildReference(parent, seq, pos);
					if( hi.getParent().isEmpty() )
						HopRewriteUtils.removeChildReference(hi, hi2);
					hi = seq;
					
					LOG.debug("Applied simplifyConstantSort1.");
				}
				else
				{
					//order(matrix(7), indexreturn=FALSE) -> matrix(7)
					HopRewriteUtils.removeChildReferenceByPos(parent, hi, pos);
					HopRewriteUtils.addChildReference(parent, hi2, pos);
					if( hi.getParent().isEmpty() )
						HopRewriteUtils.removeChildReference(hi, hi2);
					hi = hi2;
					
					LOG.debug("Applied simplifyConstantSort2.");
				}
			}	
		}
		
		return hi;
	}
	
	private Hop simplifyOrderedSort(Hop parent, Hop hi, int pos) 
		throws HopsException
	{
		//order(seq(2,N+1,1), indexreturn=FALSE) -> matrix(7)
		//order(seq(2,N+1,1), indexreturn=TRUE) -> seq(1,N,1)/seq(N,1,-1)
		if( hi instanceof ReorgOp && ((ReorgOp)hi).getOp()==ReOrgOp.SORT )  //order
		{
			Hop hi2 = hi.getInput().get(0);
			
			if( hi2 instanceof DataGenOp && ((DataGenOp)hi2).getOp()==DataGenMethod.SEQ )
			{
				Hop incr = hi2.getInput().get(((DataGenOp)hi2).getParamIndex(Statement.SEQ_INCR));
				//check for known ascending ordering and known indexreturn
				if( incr instanceof LiteralOp && HopRewriteUtils.getDoubleValue((LiteralOp)incr)==1
					&& hi.getInput().get(2) instanceof LiteralOp      //decreasing
					&& hi.getInput().get(3) instanceof LiteralOp )    //indexreturn
				{
					if( HopRewriteUtils.getBooleanValue((LiteralOp)hi.getInput().get(3)) ) //IXRET, ASC/DESC
					{
						//order(seq(2,N+1,1), indexreturn=TRUE) -> seq(1,N,1)/seq(N,1,-1)
						boolean desc = HopRewriteUtils.getBooleanValue((LiteralOp)hi.getInput().get(2));
						HopRewriteUtils.removeChildReferenceByPos(parent, hi, pos);
						Hop seq = HopRewriteUtils.createSeqDataGenOp(hi2, !desc);
						seq.refreshSizeInformation();
						HopRewriteUtils.addChildReference(parent, seq, pos);
						if( hi.getParent().isEmpty() )
							HopRewriteUtils.removeChildReference(hi, hi2);
						hi = seq;
						
						LOG.debug("Applied simplifyOrderedSort1.");
					}
					else if( !HopRewriteUtils.getBooleanValue((LiteralOp)hi.getInput().get(2)) ) //DATA, ASC
					{
						//order(seq(2,N+1,1), indexreturn=FALSE) -> seq(2,N+1,1)
						HopRewriteUtils.removeChildReferenceByPos(parent, hi, pos);
						HopRewriteUtils.addChildReference(parent, hi2, pos);
						if( hi.getParent().isEmpty() )
							HopRewriteUtils.removeChildReference(hi, hi2);
						hi = hi2;
						
						LOG.debug("Applied simplifyOrderedSort2.");
					}
				}
			}	   
		}
		
		return hi;
	}
	
	/**
	 * 
	 * @param parent
	 * @param hi
	 * @param pos
	 */
	private Hop removeUnnecessaryTranspose(Hop parent, Hop hi, int pos)
	{
		if( hi instanceof ReorgOp && ((ReorgOp)hi).getOp()==ReOrgOp.TRANSPOSE  ) //first transpose
		{
			Hop hi2 = hi.getInput().get(0);
			if( hi2 instanceof ReorgOp && ((ReorgOp)hi2).getOp()==ReOrgOp.TRANSPOSE ) //second transpose
			{
				Hop hi3 = hi2.getInput().get(0);
				//remove unnecessary chain of t(t())
				HopRewriteUtils.removeChildReference(parent, hi);
				HopRewriteUtils.addChildReference(parent, hi3, pos);
				hi = hi3;
				
				//cleanup if only consumer of intermediate
				if( hi.getParent().isEmpty() ) 
					HopRewriteUtils.removeAllChildReferences( hi );
				if( hi2.getParent().isEmpty() ) 
					HopRewriteUtils.removeAllChildReferences( hi2 );
				
				LOG.debug("Applied removeUnecessaryTranspose");
			}
		}
		
		return hi;
	}
	
	/**
	 * 
	 * @param parent
	 * @param hi
	 * @param pos
	 * @return
	 * @throws HopsException 
	 */
	private Hop removeUnnecessaryMinus(Hop parent, Hop hi, int pos) 
		throws HopsException
	{
		if( hi.getDataType() == DataType.MATRIX && hi instanceof BinaryOp 
			&& ((BinaryOp)hi).getOp()==OpOp2.MINUS  						//first minus
			&& hi.getInput().get(0) instanceof LiteralOp && ((LiteralOp)hi.getInput().get(0)).getDoubleValue()==0 )
		{
			Hop hi2 = hi.getInput().get(1);
			if( hi2.getDataType() == DataType.MATRIX && hi2 instanceof BinaryOp 
				&& ((BinaryOp)hi2).getOp()==OpOp2.MINUS  						//second minus
				&& hi2.getInput().get(0) instanceof LiteralOp && ((LiteralOp)hi2.getInput().get(0)).getDoubleValue()==0 )
				
			{
				Hop hi3 = hi2.getInput().get(1);
				//remove unnecessary chain of -(-())
				HopRewriteUtils.removeChildReference(parent, hi);
				HopRewriteUtils.addChildReference(parent, hi3, pos);
				hi = hi3;
				
				//cleanup if only consumer of intermediate
				if( hi.getParent().isEmpty() ) 
					HopRewriteUtils.removeAllChildReferences( hi );
				if( hi2.getParent().isEmpty() ) 
					HopRewriteUtils.removeAllChildReferences( hi2 );
				
				LOG.debug("Applied removeUnecessaryMinus");
			}
		}
		
		return hi;
	}
	
	/**
	 * 
	 * @param hi
	 * @return
	 */
	private Hop simplifyGroupedAggregate(Hop hi)
	{
		if( hi instanceof ParameterizedBuiltinOp && ((ParameterizedBuiltinOp)hi).getOp()==ParamBuiltinOp.GROUPEDAGG  ) //aggregate
		{
			ParameterizedBuiltinOp phi = (ParameterizedBuiltinOp)hi;
			
			if( phi.isCountFunction() ) //aggregate(fn="count")
			{
				HashMap<String, Integer> params = phi.getParamIndexMap();
				int ix1 = params.get(Statement.GAGG_TARGET);
				int ix2 = params.get(Statement.GAGG_GROUPS);
				
				//check for unnecessary memory consumption for "count"
				if( ix1 != ix2 && phi.getInput().get(ix1)!=phi.getInput().get(ix2) ) 
				{
					Hop th = phi.getInput().get(ix1);
					Hop gh = phi.getInput().get(ix2);
					
					HopRewriteUtils.removeChildReference(hi, th);
					HopRewriteUtils.addChildReference(hi, gh, ix1);
					
					LOG.debug("Applied simplifyGroupedAggregateCount");	
				}
			}
		}
		
		return hi;
	}
	
	/**
	 * Searches for weighted squared loss expressions and replaces them with a quaternary operator. 
	 * Currently, this search includes the following three patterns:
	 * 1) sum (W * (X - U %*% t(V)) ^ 2) (post weighting)
	 * 2) sum ((X - W * (U %*% t(V))) ^ 2) (pre weighting)
	 * 3) sum ((X - (U %*% t(V))) ^ 2) (no weighting)
	 * 
	 * NOTE: We include transpose into the pattern because during runtime we need to compute
	 * U%*% t(V) pointwise; having V and not t(V) at hand allows for a cache-friendly implementation
	 * without additional memory requirements for internal transpose.
	 * 
	 * This rewrite is conceptually a static rewrite; however, the current MR runtime only supports
	 * U/V factors of rank up to the blocksize (1000). We enforce this contraint here during the general
	 * rewrite because this is an uncommon case. Also, the intention is to remove this constaint as soon
	 * as we generalized the runtime or hop/lop compilation. 
	 * 
	 * @param parent
	 * @param hi
	 * @param pos
	 * @return
	 * @throws HopsException 
	 */
	private Hop simplifyWeightedSquaredLoss(Hop parent, Hop hi, int pos) 
		throws HopsException
	{
		//NOTE: there might be also a general simplification without custom operator
		//via (X-UVt)^2 -> X^2 - 2X*UVt + UVt^2
		Hop hnew = null;
		
		if( hi instanceof AggUnaryOp && ((AggUnaryOp)hi).getDirection()==Direction.RowCol
			&& ((AggUnaryOp)hi).getOp() == AggOp.SUM      //all patterns rooted by sum()
			&& hi.getInput().get(0) instanceof BinaryOp ) //all patterns subrooted by binary op
		{
			BinaryOp bop = (BinaryOp) hi.getInput().get(0);
			boolean appliedPattern = false;
			
			//Pattern 1) sum (W * (X - U %*% t(V)) ^ 2) (post weighting)
			//alternative pattern: sum (W * (U %*% t(V) - X) ^ 2)
			if( bop.getOp()==OpOp2.MULT && bop.getInput().get(1) instanceof BinaryOp	
				&& bop.getInput().get(0).getDataType()==DataType.MATRIX	
				&& HopRewriteUtils.isEqualSize(bop.getInput().get(0), bop.getInput().get(1)) //prevent mv
				&& ((BinaryOp)bop.getInput().get(1)).getOp()==OpOp2.POW 
				&& bop.getInput().get(1).getInput().get(1) instanceof LiteralOp
				&& HopRewriteUtils.getIntValue((LiteralOp)bop.getInput().get(1).getInput().get(1))==2)
			{
				Hop W = bop.getInput().get(0);
				Hop tmp = bop.getInput().get(1).getInput().get(0); //(X - U %*% t(V))
				
				if( tmp instanceof BinaryOp && ((BinaryOp)tmp).getOp()==OpOp2.MINUS
					&& HopRewriteUtils.isEqualSize(tmp.getInput().get(0), tmp.getInput().get(1)) //prevent mv	
					&& tmp.getInput().get(0).getDataType() == DataType.MATRIX )
				{
					//a) sum (W * (X - U %*% t(V)) ^ 2)
					int uvIndex = -1;
					if( tmp.getInput().get(1) instanceof AggBinaryOp  //ba gurantees matrices
							&& HopRewriteUtils.isSingleBlock(tmp.getInput().get(1).getInput().get(0),true)) //BLOCKSIZE CONSTRAINT
					{
						uvIndex = 1;   
					}
					//b) sum (W * (U %*% t(V) - X) ^ 2)
					else if(tmp.getInput().get(0) instanceof AggBinaryOp  //ba gurantees matrices
						&& HopRewriteUtils.isSingleBlock(tmp.getInput().get(0).getInput().get(0),true)) //BLOCKSIZE CONSTRAINT
					{
						uvIndex = 0;
					}   
				 
					if( uvIndex >= 0 ) //rewrite match
					{
						Hop X = tmp.getInput().get((uvIndex==0)?1:0); 
						Hop U = tmp.getInput().get(uvIndex).getInput().get(0);
						Hop V = tmp.getInput().get(uvIndex).getInput().get(1);
	                    
						if( !HopRewriteUtils.isTransposeOperation(V) ) {
							V = HopRewriteUtils.createTranspose(V);
						}
						else{
							V = V.getInput().get(0);
						}
	                    
						//handle special case of post_nz
						if( HopRewriteUtils.isNonZeroIndicator(W, X) ){
							W = new LiteralOp("1", 1);
						}
						
						//construct quaternary hop
						hnew = new QuaternaryOp(hi.getName(), DataType.SCALAR, ValueType.DOUBLE, 
								OpOp4.WSLOSS, X, U, V, W, true);
						HopRewriteUtils.setOutputParametersForScalar(hnew);
	
						appliedPattern = true;
						LOG.debug("Applied simplifyWeightedSquaredLoss1"+uvIndex+" (line "+hi.getBeginLine()+")");  
					}
				}
			}
			
			//Pattern 2) sum ((X - W * (U %*% t(V))) ^ 2) (pre weighting)
			//alternative pattern: sum ((W * (U %*% t(V)) - X) ^ 2)
			if( !appliedPattern
				&& bop.getOp()==OpOp2.POW && bop.getInput().get(1) instanceof LiteralOp
				&& HopRewriteUtils.getIntValue((LiteralOp)bop.getInput().get(1))==2
				&& bop.getInput().get(0) instanceof BinaryOp	
				&& bop.getInput().get(0).getDataType()==DataType.MATRIX	
				&& ((BinaryOp)bop.getInput().get(0)).getOp()==OpOp2.MINUS
				&& HopRewriteUtils.isEqualSize(bop.getInput().get(0).getInput().get(0), bop.getInput().get(0).getInput().get(1)) //prevent mv
				&& bop.getInput().get(0).getInput().get(0).getDataType()==DataType.MATRIX)
			{
			    Hop lleft = bop.getInput().get(0).getInput().get(0); 
			    Hop lright = bop.getInput().get(0).getInput().get(1); 
                
			    //a) sum ((X - W * (U %*% t(V))) ^ 2)
			    int wuvIndex = -1;
			    if( lright instanceof BinaryOp && lright.getInput().get(1) instanceof AggBinaryOp ){
			    	wuvIndex = 1;
			    }
			    //b) sum ((W * (U %*% t(V)) - X) ^ 2)
			    else if( lleft instanceof BinaryOp && lleft.getInput().get(1) instanceof AggBinaryOp ){
			    	wuvIndex = 0;
			    }
			    
			    if( wuvIndex >= 0 ) //rewrite match
			    {
			    	Hop X = bop.getInput().get(0).getInput().get((wuvIndex==0)?1:0);
			    	Hop tmp = bop.getInput().get(0).getInput().get(wuvIndex); //(W * (U %*% t(V)))
    				
    				if( ((BinaryOp)tmp).getOp()==OpOp2.MULT
    					&& tmp.getInput().get(0).getDataType() == DataType.MATRIX	
    					&& HopRewriteUtils.isEqualSize(tmp.getInput().get(0), tmp.getInput().get(1)) //prevent mv
    					&& HopRewriteUtils.isSingleBlock(tmp.getInput().get(1).getInput().get(0),true)) //BLOCKSIZE CONSTRAINT
    				{
    					Hop W = tmp.getInput().get(0); 
    					Hop U = tmp.getInput().get(1).getInput().get(0);
    					Hop V = tmp.getInput().get(1).getInput().get(1);
    					
    					if( !HopRewriteUtils.isTransposeOperation(V) ) { 
    						V = HopRewriteUtils.createTranspose(V);
    					}
    					else {
    						V = V.getInput().get(0);
    					}
    					
    					hnew = new QuaternaryOp(hi.getName(), DataType.SCALAR, ValueType.DOUBLE, 
    							  OpOp4.WSLOSS, X, U, V, W, false);
    					HopRewriteUtils.setOutputParametersForScalar(hnew);
    
    					appliedPattern = true;
    					LOG.debug("Applied simplifyWeightedSquaredLoss2"+wuvIndex+" (line "+hi.getBeginLine()+")");	
    				}
			    }
			}
			
			//Pattern 3) sum ((X - (U %*% t(V))) ^ 2) (no weighting)
			//alternative pattern: sum (((U %*% t(V)) - X) ^ 2)
			if( !appliedPattern
				&& bop.getOp()==OpOp2.POW && bop.getInput().get(1) instanceof LiteralOp
				&& HopRewriteUtils.getIntValue((LiteralOp)bop.getInput().get(1))==2
				&& bop.getInput().get(0) instanceof BinaryOp	
				&& bop.getInput().get(0).getDataType()==DataType.MATRIX	
				&& ((BinaryOp)bop.getInput().get(0)).getOp()==OpOp2.MINUS
				&& HopRewriteUtils.isEqualSize(bop.getInput().get(0).getInput().get(0), bop.getInput().get(0).getInput().get(1)) //prevent mv
				&& bop.getInput().get(0).getInput().get(0).getDataType()==DataType.MATRIX)
			{
				Hop lleft = bop.getInput().get(0).getInput().get(0);
				Hop lright = bop.getInput().get(0).getInput().get(1);
                
				//a) sum ((X - (U %*% t(V))) ^ 2)
				int uvIndex = -1;
				if( lright instanceof AggBinaryOp //ba gurantees matrices
					&& HopRewriteUtils.isSingleBlock(lright.getInput().get(0),true) )  //BLOCKSIZE CONSTRAINT
				{
					uvIndex = 1;
				}
				//b) sum (((U %*% t(V)) - X) ^ 2)
				else if( lleft instanceof AggBinaryOp //ba gurantees matrices
						&& HopRewriteUtils.isSingleBlock(lleft.getInput().get(0),true) )  //BLOCKSIZE CONSTRAINT
				{
					uvIndex = 0;
				}
			    
				if( uvIndex >= 0 ) //rewrite match
				{
					Hop X = bop.getInput().get(0).getInput().get((uvIndex==0)?1:0);
					Hop tmp = bop.getInput().get(0).getInput().get(uvIndex); //(U %*% t(V))
					Hop W = new LiteralOp("1", 1); //no weighting 
					Hop U = tmp.getInput().get(0);
					Hop V = tmp.getInput().get(1);
	
					if( !HopRewriteUtils.isTransposeOperation(V) ) { 
						V = HopRewriteUtils.createTranspose(V);
					}
					else {
						V = V.getInput().get(0);
					}
					
					hnew = new QuaternaryOp(hi.getName(), DataType.SCALAR, ValueType.DOUBLE, 
							  OpOp4.WSLOSS, X, U, V, W, false);
					HopRewriteUtils.setOutputParametersForScalar(hnew);

					appliedPattern = true;
					LOG.debug("Applied simplifyWeightedSquaredLoss3"+uvIndex+" (line "+hi.getBeginLine()+")");	
				}
			}			
		}
		
		//relink new hop into original position
		if( hnew != null ) {
			HopRewriteUtils.removeChildReferenceByPos(parent, hi, pos);
			HopRewriteUtils.addChildReference(parent, hnew, pos);
			hi = hnew;
		}
		
		return hi;
	}
	
	/**
	 * 
	 * @param parent
	 * @param hi
	 * @param pos
	 * @return
	 * @throws HopsException
	 */
	private Hop simplifyWeightedSigmoidMMChains(Hop parent, Hop hi, int pos) 
		throws HopsException
	{
		Hop hnew = null;
		
		if(    hi instanceof BinaryOp //all patterns subrooted by W *
			&& ((BinaryOp) hi).getOp()==OpOp2.MULT
			&& HopRewriteUtils.isEqualSize(hi.getInput().get(0), hi.getInput().get(1)) //prevent mv
			&& hi.getInput().get(0).getDataType()==DataType.MATRIX 
			&& hi.getInput().get(1) instanceof UnaryOp ) //sigmoid/log
		{
			UnaryOp uop = (UnaryOp) hi.getInput().get(1);
			boolean appliedPattern = false;
			
			//Pattern 1) W * sigmoid(Y%*%t(X)) (basic)
			if(    uop.getOp() == OpOp1.SIGMOID 
				&& uop.getInput().get(0) instanceof AggBinaryOp
				&& HopRewriteUtils.isSingleBlock(uop.getInput().get(0).getInput().get(0),true) )
			{
				Hop W = hi.getInput().get(0); 
				Hop Y = uop.getInput().get(0).getInput().get(0);
				Hop tX = uop.getInput().get(0).getInput().get(1);
				
				if( !HopRewriteUtils.isTransposeOperation(tX) ) { 
					tX = HopRewriteUtils.createTranspose(tX);
				}
				else 
					tX = tX.getInput().get(0);
				
				hnew = new QuaternaryOp(hi.getName(), DataType.MATRIX, ValueType.DOUBLE, 
						  OpOp4.WSIGMOID, W, Y, tX, false, false);
				HopRewriteUtils.setOutputBlocksizes(hnew, W.getRowsInBlock(), W.getColsInBlock());

				appliedPattern = true;
				LOG.debug("Applied simplifyWeightedSigmoid1 (line "+hi.getBeginLine()+")");	
			}
			
			//Pattern 2) W * sigmoid(-(Y%*%t(X))) (minus)
			if(    !appliedPattern 
				&& uop.getOp() == OpOp1.SIGMOID 
				&& uop.getInput().get(0) instanceof BinaryOp
				&& ((BinaryOp)uop.getInput().get(0)).getOp()==OpOp2.MINUS
				&& uop.getInput().get(0).getInput().get(0) instanceof LiteralOp
				&& HopRewriteUtils.getDoubleValueSafe(
				   (LiteralOp)uop.getInput().get(0).getInput().get(0))==0
				&& uop.getInput().get(0).getInput().get(1) instanceof AggBinaryOp
				&& HopRewriteUtils.isSingleBlock(uop.getInput().get(0).getInput().get(1).getInput().get(0),true))
			{
				Hop W = hi.getInput().get(0); 
				Hop Y = uop.getInput().get(0).getInput().get(1).getInput().get(0);
				Hop tX = uop.getInput().get(0).getInput().get(1).getInput().get(1);
				
				if( !HopRewriteUtils.isTransposeOperation(tX) ) { 
					tX = HopRewriteUtils.createTranspose(tX);
				}
				else 
					tX = tX.getInput().get(0);
				
				hnew = new QuaternaryOp(hi.getName(), DataType.MATRIX, ValueType.DOUBLE, 
						  OpOp4.WSIGMOID, W, Y, tX, false, true);
				HopRewriteUtils.setOutputBlocksizes(hnew, W.getRowsInBlock(), W.getColsInBlock());

				appliedPattern = true;
				LOG.debug("Applied simplifyWeightedSigmoid2 (line "+hi.getBeginLine()+")");	
			}
			
			//Pattern 3) W * log(sigmoid(Y%*%t(X))) (log)			
			if(    !appliedPattern 
				&& uop.getOp() == OpOp1.LOG
				&& uop.getInput().get(0) instanceof UnaryOp
				&& ((UnaryOp)uop.getInput().get(0)).getOp() == OpOp1.SIGMOID 
				&& uop.getInput().get(0).getInput().get(0) instanceof AggBinaryOp
				&& HopRewriteUtils.isSingleBlock(uop.getInput().get(0).getInput().get(0).getInput().get(0),true) )
			{
				Hop W = hi.getInput().get(0); 
				Hop Y = uop.getInput().get(0).getInput().get(0).getInput().get(0);
				Hop tX = uop.getInput().get(0).getInput().get(0).getInput().get(1);
				
				if( !HopRewriteUtils.isTransposeOperation(tX) ) { 
					tX = HopRewriteUtils.createTranspose(tX);
				}
				else 
					tX = tX.getInput().get(0);
				
				hnew = new QuaternaryOp(hi.getName(), DataType.MATRIX, ValueType.DOUBLE, 
						  OpOp4.WSIGMOID, W, Y, tX, true, false);
				HopRewriteUtils.setOutputBlocksizes(hnew, W.getRowsInBlock(), W.getColsInBlock());

				appliedPattern = true;
				LOG.debug("Applied simplifyWeightedSigmoid3 (line "+hi.getBeginLine()+")");	
			}			
			
			//Pattern 4) W * log(sigmoid(-(Y%*%t(X)))) (log_minus)
			if(    !appliedPattern 
				&& uop.getOp() == OpOp1.LOG
				&& uop.getInput().get(0) instanceof UnaryOp
				&& ((UnaryOp)uop.getInput().get(0)).getOp() == OpOp1.SIGMOID 
				&& uop.getInput().get(0).getInput().get(0) instanceof BinaryOp )
			{
				BinaryOp bop = (BinaryOp) uop.getInput().get(0).getInput().get(0);
				
				if(    bop.getOp() == OpOp2.MINUS 
					&& bop.getInput().get(0) instanceof LiteralOp
					&& HopRewriteUtils.getDoubleValueSafe((LiteralOp)bop.getInput().get(0))==0
					&& bop.getInput().get(1) instanceof AggBinaryOp
					&& HopRewriteUtils.isSingleBlock(bop.getInput().get(1).getInput().get(0),true))
				{
					Hop W = hi.getInput().get(0); 
					Hop Y = bop.getInput().get(1).getInput().get(0);
					Hop tX = bop.getInput().get(1).getInput().get(1);
					
					if( !HopRewriteUtils.isTransposeOperation(tX) ) { 
						tX = HopRewriteUtils.createTranspose(tX);
					}
					else 
						tX = tX.getInput().get(0);
					
					hnew = new QuaternaryOp(hi.getName(), DataType.MATRIX, ValueType.DOUBLE, 
							  OpOp4.WSIGMOID, W, Y, tX, true, true);
					HopRewriteUtils.setOutputBlocksizes(hnew, W.getRowsInBlock(), W.getColsInBlock());
	
					appliedPattern = true;
					LOG.debug("Applied simplifyWeightedSigmoid4 (line "+hi.getBeginLine()+")");	
				}
			}
		}
		
		//relink new hop into original position
		if( hnew != null ) {
			HopRewriteUtils.removeChildReferenceByPos(parent, hi, pos);
			HopRewriteUtils.addChildReference(parent, hnew, pos);
			hi = hnew;
		}
		
		return hi;
	}

	/**
	 * 
	 * @param parent
	 * @param hi
	 * @param pos
	 * @return
	 * @throws HopsException
	 */
	private Hop simplifyWeightedDivMM(Hop parent, Hop hi, int pos) 
		throws HopsException
	{
		Hop hnew = null;
		boolean appliedPattern = false;
		
		//left/right patterns rooted by 'ab - b(div)' or 'ab - b(mult)'
		//note: we do not rewrite t(X)%*%(w*(X%*%v)) where w and v are vectors (see mmchain ops) 
		if( hi instanceof AggBinaryOp && ((AggBinaryOp)hi).isMatrixMultiply()  
			&& (hi.getInput().get(0) instanceof BinaryOp
			&& HopRewriteUtils.isValidOp(((BinaryOp)hi.getInput().get(0)).getOp(), LOOKUP_VALID_WDIVMM_BINARY)
			||  hi.getInput().get(1) instanceof BinaryOp 
			&& (((AggBinaryOp) hi).checkMapMultChain() == ChainType.NONE || hi.getInput().get(1).getDim2() > 1) //no mmchain
			&& HopRewriteUtils.isValidOp(((BinaryOp)hi.getInput().get(1)).getOp(), LOOKUP_VALID_WDIVMM_BINARY)) ) 
		{
			Hop left = hi.getInput().get(0);
			Hop right = hi.getInput().get(1);
			
			//Pattern 1) t(U) %*% (W/(U%*%t(V)))
			//alternative pattern: t(U) %*% (W*(U%*%t(V)))
			if( right instanceof BinaryOp && HopRewriteUtils.isValidOp(((BinaryOp)right).getOp(),LOOKUP_VALID_WDIVMM_BINARY)	
				&& HopRewriteUtils.isEqualSize(right.getInput().get(0), right.getInput().get(1)) //prevent mv
				&& right.getInput().get(1) instanceof AggBinaryOp
				&& HopRewriteUtils.isSingleBlock(right.getInput().get(1).getInput().get(0),true) ) //BLOCKSIZE CONSTRAINT
			{
				Hop W = right.getInput().get(0); 
				Hop U = right.getInput().get(1).getInput().get(0);
				Hop V = right.getInput().get(1).getInput().get(1);
				
				if( HopRewriteUtils.isTransposeOfItself(left, U) ) 
				{
					if( !HopRewriteUtils.isTransposeOperation(V) )
						V = HopRewriteUtils.createTranspose(V);
					else 
						V = V.getInput().get(0);
					
					boolean mult = ((BinaryOp)right).getOp() == OpOp2.MULT;
					hnew = new QuaternaryOp(hi.getName(), DataType.MATRIX, ValueType.DOUBLE, 
							  OpOp4.WDIVMM, W, U, V, 1, mult, false);
					HopRewriteUtils.setOutputBlocksizes(hnew, W.getRowsInBlock(), W.getColsInBlock());
					
					//add output transpose for efficient target indexing (redundant t() removed by other rewrites)
					hnew = HopRewriteUtils.createTranspose(hnew);
					
					appliedPattern = true;
					LOG.debug("Applied simplifyWeightedDivMM1 (line "+hi.getBeginLine()+")");					
				}
			}	
			
			//Pattern 2) (W/(U%*%t(V))) %*% V
			//alternative pattern: (W*(U%*%t(V))) %*% V
			if( !appliedPattern
				&& left instanceof BinaryOp && HopRewriteUtils.isValidOp(((BinaryOp)left).getOp(), LOOKUP_VALID_WDIVMM_BINARY)	
				&& HopRewriteUtils.isEqualSize(left.getInput().get(0), left.getInput().get(1)) //prevent mv
				&& left.getInput().get(1) instanceof AggBinaryOp
				&& HopRewriteUtils.isSingleBlock(left.getInput().get(1).getInput().get(0),true) ) //BLOCKSIZE CONSTRAINT
			{
				Hop W = left.getInput().get(0); 
				Hop U = left.getInput().get(1).getInput().get(0);
				Hop V = left.getInput().get(1).getInput().get(1);
				
				if( HopRewriteUtils.isTransposeOfItself(right, V) ) 
				{
					if( !HopRewriteUtils.isTransposeOperation(V) )
						V = right;
					else 
						V = V.getInput().get(0);
					
					boolean mult = ((BinaryOp)left).getOp() == OpOp2.MULT;
					hnew = new QuaternaryOp(hi.getName(), DataType.MATRIX, ValueType.DOUBLE, 
							  OpOp4.WDIVMM, W, U, V, 2, mult, false);
					HopRewriteUtils.setOutputBlocksizes(hnew, W.getRowsInBlock(), W.getColsInBlock());

					appliedPattern = true;
					LOG.debug("Applied simplifyWeightedDivMM2 (line "+hi.getBeginLine()+")");	
				}
			}
			
			//Pattern 3) t(U) %*% ((X!=0)*(U%*%t(V)-X))
			if( right instanceof BinaryOp && ((BinaryOp)right).getOp()==LOOKUP_VALID_WDIVMM_BINARY[0] //MULT
				&& right.getInput().get(1) instanceof BinaryOp && ((BinaryOp)right.getInput().get(1)).getOp()==OpOp2.MINUS	
				&& right.getInput().get(1).getInput().get(0) instanceof AggBinaryOp
                && right.getInput().get(1).getInput().get(1).getDataType() == DataType.MATRIX
				&& HopRewriteUtils.isSingleBlock(right.getInput().get(1).getInput().get(0).getInput().get(0),true) ) //BLOCKSIZE CONSTRAINT
			{
				Hop W = right.getInput().get(0); 
				Hop U = right.getInput().get(1).getInput().get(0).getInput().get(0);
				Hop V = right.getInput().get(1).getInput().get(0).getInput().get(1);
				Hop X = right.getInput().get(1).getInput().get(1);
				
				if(    HopRewriteUtils.isNonZeroIndicator(W, X)        //W-X constraint
				    && HopRewriteUtils.isTransposeOfItself(left, U) )  //t(U)-U constraint
				{
					if( !HopRewriteUtils.isTransposeOperation(V) )
						V = HopRewriteUtils.createTranspose(V);
					else 
						V = V.getInput().get(0);
					
					hnew = new QuaternaryOp(hi.getName(), DataType.MATRIX, ValueType.DOUBLE, 
							  OpOp4.WDIVMM, X, U, V, 1, true, true);
					HopRewriteUtils.setOutputBlocksizes(hnew, W.getRowsInBlock(), W.getColsInBlock());
					
					//add output transpose for efficient target indexing (redundant t() removed by other rewrites)
					hnew = HopRewriteUtils.createTranspose(hnew);
					
					appliedPattern = true;
					LOG.debug("Applied simplifyWeightedDivMM3 (line "+hi.getBeginLine()+")");					
				}
			}	
			
			//Pattern 4) ((X!=0)*(U%*%t(V)-X)) %*% V
			if( !appliedPattern
				&& left instanceof BinaryOp && ((BinaryOp)left).getOp()==LOOKUP_VALID_WDIVMM_BINARY[0] //MULT	
				&& left.getInput().get(1) instanceof BinaryOp && ((BinaryOp)left.getInput().get(1)).getOp()==OpOp2.MINUS	
				&& left.getInput().get(1).getInput().get(0) instanceof AggBinaryOp
                && left.getInput().get(1).getInput().get(1).getDataType() == DataType.MATRIX
				&& HopRewriteUtils.isSingleBlock(left.getInput().get(1).getInput().get(0).getInput().get(0),true) ) //BLOCKSIZE CONSTRAINT
			{
				Hop W = left.getInput().get(0); 
				Hop U = left.getInput().get(1).getInput().get(0).getInput().get(0);
				Hop V = left.getInput().get(1).getInput().get(0).getInput().get(1);
				Hop X = left.getInput().get(1).getInput().get(1);
				
				if(    HopRewriteUtils.isNonZeroIndicator(W, X)        //W-X constraint
					&& HopRewriteUtils.isTransposeOfItself(right, V) )  //V-t(V) constraint
				{
					if( !HopRewriteUtils.isTransposeOperation(V) )
						V = right;
					else 
						V = V.getInput().get(0);
					
					hnew = new QuaternaryOp(hi.getName(), DataType.MATRIX, ValueType.DOUBLE, 
							  OpOp4.WDIVMM, X, U, V, 2, true, true);
					HopRewriteUtils.setOutputBlocksizes(hnew, W.getRowsInBlock(), W.getColsInBlock());

					appliedPattern = true;
					LOG.debug("Applied simplifyWeightedDivMM4 (line "+hi.getBeginLine()+")");	
				}
			}
		}
		
		//Pattern 5) (W*(U%*%t(V)))
		if( !appliedPattern
			&& hi instanceof BinaryOp && ((BinaryOp)hi).getOp()==LOOKUP_VALID_WDIVMM_BINARY[0] //MULT	
			&& HopRewriteUtils.isEqualSize(hi.getInput().get(0), hi.getInput().get(1)) //prevent mv
			&& hi.getInput().get(0).getDataType() == DataType.MATRIX 
			&& hi.getInput().get(0).getDim2() > hi.getInput().get(0).getColsInBlock()
			&& hi.getInput().get(1) instanceof AggBinaryOp
			&& (((AggBinaryOp) hi.getInput().get(1)).checkMapMultChain() == ChainType.NONE || hi.getInput().get(1).getInput().get(1).getDim2() > 1) //no mmchain
			&& HopRewriteUtils.isSingleBlock(hi.getInput().get(1).getInput().get(0),true) ) //BLOCKSIZE CONSTRAINT
		{
			Hop W = hi.getInput().get(0); 
			Hop U = hi.getInput().get(1).getInput().get(0);
			Hop V = hi.getInput().get(1).getInput().get(1);
			
			if( !HopRewriteUtils.isTransposeOperation(V) )
				V = HopRewriteUtils.createTranspose(V);
			else 
				V = V.getInput().get(0);
				
			hnew = new QuaternaryOp(hi.getName(), DataType.MATRIX, ValueType.DOUBLE, 
					  OpOp4.WDIVMM, W, U, V, 0, true, false);
			HopRewriteUtils.setOutputBlocksizes(hnew, W.getRowsInBlock(), W.getColsInBlock());

			appliedPattern = true;
			LOG.debug("Applied simplifyWeightedDivMM5 (line "+hi.getBeginLine()+")");	
		}
		
		//relink new hop into original position
		if( hnew != null ) {
			HopRewriteUtils.removeChildReferenceByPos(parent, hi, pos);
			HopRewriteUtils.addChildReference(parent, hnew, pos);
			hi = hnew;
		}
		
		return hi;
	}

	/**
	 * 
	 * @param parent
	 * @param hi
	 * @param pos
	 * @return
	 * @throws HopsException
	 */
	private Hop simplifyWeightedCrossEntropy(Hop parent, Hop hi, int pos) 
		throws HopsException
	{
		Hop hnew = null;
		
		//Pattern 1) sum( X * log(U %*% t(V)))
		if( hi instanceof AggUnaryOp && ((AggUnaryOp)hi).getDirection()==Direction.RowCol
			&& ((AggUnaryOp)hi).getOp() == AggOp.SUM      //pattern rooted by sum()
			&& hi.getInput().get(0) instanceof BinaryOp ) //pattern subrooted by binary op
		{
			BinaryOp bop = (BinaryOp) hi.getInput().get(0);
			Hop left = bop.getInput().get(0);
			Hop right = bop.getInput().get(1);
			
			if( bop.getOp()==OpOp2.MULT && left.getDataType()==DataType.MATRIX		
				&& HopRewriteUtils.isEqualSize(left, right)  //prevent mb
				&& right instanceof UnaryOp	&& ((UnaryOp)right).getOp()==OpOp1.LOG
				&& right.getInput().get(0) instanceof AggBinaryOp  //ba gurantees matrices
				&& HopRewriteUtils.isSingleBlock(right.getInput().get(0).getInput().get(0),true)) //BLOCKSIZE CONSTRAINT
			{
				Hop X = left; 
				Hop U = right.getInput().get(0).getInput().get(0);
				Hop V = right.getInput().get(0).getInput().get(1);
				
				if( !HopRewriteUtils.isTransposeOperation(V) )
					V = HopRewriteUtils.createTranspose(V);
				else 
					V = V.getInput().get(0);
					
				hnew = new QuaternaryOp(hi.getName(), DataType.SCALAR, ValueType.DOUBLE, OpOp4.WCEMM, X, U, V);
				HopRewriteUtils.setOutputBlocksizes(hnew, X.getRowsInBlock(), X.getColsInBlock());
					
				LOG.debug("Applied simplifyWeightedCEMM (line "+hi.getBeginLine()+")");					
			}
		}
		
		//relink new hop into original position
		if( hnew != null ) {
			HopRewriteUtils.removeChildReferenceByPos(parent, hi, pos);
			HopRewriteUtils.addChildReference(parent, hnew, pos);
			hi = hnew;
		}
		
		return hi;
	}
	
	/**
	 * 
	 * @param parent
	 * @param hi
	 * @param pos
	 * @return
	 * @throws HopsException
	 */
	private Hop fuseMinusNzBinaryOperation(Hop parent, Hop hi, int pos) 
		throws HopsException
	{
		//pattern X - (s * ppred(X,0,!=)) -> X -nz s
		//note: this is done as a hop rewrite in order to significantly reduce the 
		//memory estimate for X - tmp if X is sparse 
		if( hi instanceof BinaryOp && ((BinaryOp)hi).getOp()==OpOp2.MINUS
			&& hi.getInput().get(0).getDataType()==DataType.MATRIX
			&& hi.getInput().get(1).getDataType()==DataType.MATRIX
			&& hi.getInput().get(1) instanceof BinaryOp 
			&& ((BinaryOp)hi.getInput().get(1)).getOp()==OpOp2.MULT )
		{
			Hop X = hi.getInput().get(0);
			Hop s = hi.getInput().get(1).getInput().get(0);
			Hop pred = hi.getInput().get(1).getInput().get(1);
			
			if( s.getDataType()==DataType.SCALAR && pred.getDataType()==DataType.MATRIX
				&& pred instanceof BinaryOp && ((BinaryOp)pred).getOp()==OpOp2.NOTEQUAL
				&& pred.getInput().get(0) == X //depend on common subexpression elimination
				&& pred.getInput().get(1) instanceof LiteralOp
				&& HopRewriteUtils.getDoubleValueSafe((LiteralOp)pred.getInput().get(1))==0 )
			{
				Hop hnew = new BinaryOp("tmp", DataType.MATRIX, ValueType.DOUBLE, OpOp2.MINUS_NZ, X, s);
				HopRewriteUtils.setOutputBlocksizes(hnew, hi.getRowsInBlock(), hi.getColsInBlock());
				hnew.refreshSizeInformation();
		
				//relink new hop into original position
				HopRewriteUtils.removeChildReferenceByPos(parent, hi, pos);
				HopRewriteUtils.addChildReference(parent, hnew, pos);
				hi = hnew;
				
				LOG.debug("Applied fuseMinusNzBinaryOperation (line "+hi.getBeginLine()+")");	
			}		
		}
		
		return hi;
	}
	
	
	/**
	 * 
	 * @param parent
	 * @param hi
	 * @param pos
	 * @return
	 * @throws HopsException
	 */
	private Hop fuseLogNzBinaryOperation(Hop parent, Hop hi, int pos) 
		throws HopsException
	{
		//pattern ppred(X,0,"!=")*log(X,0.5) -> log_nz(X,0.5)
		//note: this is done as a hop rewrite in order to significantly reduce the 
		//memory estimate and to prevent dense intermediates if X is ultra sparse  
		if( hi instanceof BinaryOp && ((BinaryOp)hi).getOp()==OpOp2.MULT
			&& hi.getInput().get(0).getDataType()==DataType.MATRIX
			&& hi.getInput().get(1).getDataType()==DataType.MATRIX
			&& hi.getInput().get(1) instanceof BinaryOp 
			&& ((BinaryOp)hi.getInput().get(1)).getOp()==OpOp2.LOG )
		{
			Hop pred = hi.getInput().get(0);
			Hop X = hi.getInput().get(1).getInput().get(0);
			Hop log = hi.getInput().get(1).getInput().get(1);
			
			if(    pred instanceof BinaryOp && ((BinaryOp)pred).getOp()==OpOp2.NOTEQUAL
				&& pred.getInput().get(0) == X //depend on common subexpression elimination
				&& pred.getInput().get(1) instanceof LiteralOp
				&& HopRewriteUtils.getDoubleValueSafe((LiteralOp)pred.getInput().get(1))==0 )
			{
				Hop hnew = new BinaryOp("tmp", DataType.MATRIX, ValueType.DOUBLE, OpOp2.LOG_NZ, X, log);
				HopRewriteUtils.setOutputBlocksizes(hnew, hi.getRowsInBlock(), hi.getColsInBlock());
				hnew.refreshSizeInformation();
		
				//relink new hop into original position
				HopRewriteUtils.removeChildReferenceByPos(parent, hi, pos);
				HopRewriteUtils.addChildReference(parent, hnew, pos);
				hi = hnew;
				
				LOG.debug("Applied fuseLogNzBinaryOperation (line "+hi.getBeginLine()+")");	
			}		
		}
		
		return hi;
	}

	/**
	 * 
	 * @param parent
	 * @param hi
	 * @param pos
	 * @return
	 * @throws HopsException
	 */
	private Hop simplifyOuterSeqExpand(Hop parent, Hop hi, int pos) 
		throws HopsException
	{
		//pattern: outer(v, t(seq(1,m)), "==") -> rexpand(v, max=m, dir=row, ignore=true, cast=false)
		//note: this rewrite supports both left/right sequence 
		
		if( hi instanceof BinaryOp && ((BinaryOp)hi).isOuterVectorOperator()
			&& ((BinaryOp)hi).getOp()==OpOp2.EQUAL )
		{
			if(   ( hi.getInput().get(1) instanceof ReorgOp                 //pattern a: outer(v, t(seq(1,m)), "==")
				    && ((ReorgOp) hi.getInput().get(1)).getOp()==ReOrgOp.TRANSPOSE
				    && HopRewriteUtils.isBasic1NSequence(hi.getInput().get(1).getInput().get(0))) 
				|| HopRewriteUtils.isBasic1NSequence(hi.getInput().get(0))) //pattern b: outer(seq(1,m), t(v) "==")
			{
				//determine variable parameters for pattern a/b
				boolean isPatternB = HopRewriteUtils.isBasic1NSequence(hi.getInput().get(0));
				boolean isTransposeRight = (hi.getInput().get(1) instanceof ReorgOp 
						&& ((ReorgOp) hi.getInput().get(1)).getOp()==ReOrgOp.TRANSPOSE);				
				Hop trgt = isPatternB ? (isTransposeRight ? 
						hi.getInput().get(1).getInput().get(0) :                  //get v from t(v)
						HopRewriteUtils.createTranspose(hi.getInput().get(1)) ) : //create v via t(v')
						hi.getInput().get(0);                                     //get v directly 
				Hop seq = isPatternB ?
						hi.getInput().get(0) : hi.getInput().get(1).getInput().get(0);					
				String direction = HopRewriteUtils.isBasic1NSequence(hi.getInput().get(0)) ? "rows" : "cols";
				
				//setup input parameter hops
				HashMap<String,Hop> inputargs = new HashMap<String,Hop>();
				inputargs.put("target", trgt);
				inputargs.put("max", HopRewriteUtils.getBasic1NSequenceMaxLiteral(seq));
				inputargs.put("dir", new LiteralOp(direction, direction));
				inputargs.put("ignore", new LiteralOp("true", true));
				inputargs.put("cast", new LiteralOp("false", false));
			
				//create new hop
				ParameterizedBuiltinOp pbop = new ParameterizedBuiltinOp("tmp", DataType.MATRIX, ValueType.DOUBLE, 
						ParamBuiltinOp.REXPAND, inputargs);
				HopRewriteUtils.setOutputBlocksizes(pbop, hi.getRowsInBlock(), hi.getColsInBlock());
				pbop.refreshSizeInformation();
		
				//relink new hop into original position
				HopRewriteUtils.removeChildReferenceByPos(parent, hi, pos);
				HopRewriteUtils.addChildReference(parent, pbop, pos);
				hi = pbop;
				
				LOG.debug("Applied simplifyOuterSeqExpand (line "+hi.getBeginLine()+")");	
			}
		}
	
		return hi;
	}
	
	/**
	 * 
	 * @param parent
	 * @param hi
	 * @param pos
	 * @return
	 * @throws HopsException
	 */
	private Hop simplifyTableSeqExpand(Hop parent, Hop hi, int pos) 
		throws HopsException
	{
		//pattern: table(seq(1,nrow(v)), v, nrow(v), m) -> rexpand(v, max=m, dir=row, ignore=false, cast=true)
		//note: this rewrite supports both left/right sequence 
		
		if(    hi instanceof TernaryOp && hi.getInput().size()==5 
			&& hi.getInput().get(3) instanceof LiteralOp && hi.getInput().get(4) instanceof LiteralOp )
		{
			if(  (HopRewriteUtils.isBasic1NSequence(hi.getInput().get(0)) &&
				   hi.getInput().get(3) instanceof LiteralOp)   //pattern a: table(seq(1,nrow(v)), v, nrow(v), m)
			   ||(HopRewriteUtils.isBasic1NSequence(hi.getInput().get(1)) &&
				   hi.getInput().get(2) instanceof LiteralOp) ) //pattern b: table(v, seq(1,nrow(v)), m, nrow(v))
			{
				//determine variable parameters for pattern a/b
				int ixTgt = HopRewriteUtils.isBasic1NSequence(hi.getInput().get(0)) ? 1 : 0;
				int ixMax = HopRewriteUtils.isBasic1NSequence(hi.getInput().get(0)) ? 4 : 3;
				String direction = HopRewriteUtils.isBasic1NSequence(hi.getInput().get(0)) ? "cols" : "rows";
				
				//setup input parameter hops
				HashMap<String,Hop> inputargs = new HashMap<String,Hop>();
				inputargs.put("target", hi.getInput().get(ixTgt));
				inputargs.put("max", hi.getInput().get(ixMax));
				inputargs.put("dir", new LiteralOp(direction, direction));
				inputargs.put("ignore", new LiteralOp("true", false));
				inputargs.put("cast", new LiteralOp("false", true));
			
				//create new hop
				ParameterizedBuiltinOp pbop = new ParameterizedBuiltinOp("tmp", DataType.MATRIX, ValueType.DOUBLE, 
						ParamBuiltinOp.REXPAND, inputargs);
				HopRewriteUtils.setOutputBlocksizes(pbop, hi.getRowsInBlock(), hi.getColsInBlock());
				pbop.refreshSizeInformation();
		
				//relink new hop into original position
				HopRewriteUtils.removeChildReferenceByPos(parent, hi, pos);
				HopRewriteUtils.addChildReference(parent, pbop, pos);
				hi = pbop;
				
				LOG.debug("Applied simplifyTableSeqExpand (line "+hi.getBeginLine()+")");	
			}
		}
	
		return hi;
	}

	/**
	 * NOTE: currently disabled since this rewrite is INVALID in the
	 * presence of NaNs (because (NaN!=NaN) is true). 
	 * 
	 * @param parent
	 * @param hi
	 * @param pos
	 * @return
	 * @throws HopsException
	 */
	@SuppressWarnings("unused")
	private Hop removeUnecessaryPPred(Hop parent, Hop hi, int pos) 
		throws HopsException
	{
		if( hi instanceof BinaryOp )
		{
			BinaryOp bop = (BinaryOp)hi;
			Hop left = bop.getInput().get(0);
			Hop right = bop.getInput().get(1);
			
			Hop datagen = null;
			
			//ppred(X,X,"==") -> matrix(1, rows=nrow(X),cols=nrow(Y))
			if( left==right && bop.getOp()==OpOp2.EQUAL || bop.getOp()==OpOp2.GREATEREQUAL || bop.getOp()==OpOp2.LESSEQUAL )
				datagen = HopRewriteUtils.createDataGenOp(left, 1);
			
			//ppred(X,X,"!=") -> matrix(0, rows=nrow(X),cols=nrow(Y))
			if( left==right && bop.getOp()==OpOp2.NOTEQUAL || bop.getOp()==OpOp2.GREATER || bop.getOp()==OpOp2.LESS )
				datagen = HopRewriteUtils.createDataGenOp(left, 0);
					
			if( datagen != null )
			{
				HopRewriteUtils.removeChildReference(parent, hi);
				HopRewriteUtils.addChildReference(parent, datagen, pos);
				hi = datagen;
			}
		}
		
		return hi;
	}
	
}
