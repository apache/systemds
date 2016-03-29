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

package org.apache.sysml.hops.rewrite;

import java.util.ArrayList;
import java.util.HashMap;

import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;
import org.apache.sysml.hops.AggBinaryOp;
import org.apache.sysml.hops.AggUnaryOp;
import org.apache.sysml.hops.BinaryOp;
import org.apache.sysml.hops.DataGenOp;
import org.apache.sysml.hops.Hop;
import org.apache.sysml.hops.Hop.OpOp1;
import org.apache.sysml.hops.IndexingOp;
import org.apache.sysml.hops.TernaryOp;
import org.apache.sysml.hops.UnaryOp;
import org.apache.sysml.hops.Hop.AggOp;
import org.apache.sysml.hops.Hop.DataGenMethod;
import org.apache.sysml.hops.Hop.Direction;
import org.apache.sysml.hops.Hop.OpOp3;
import org.apache.sysml.hops.Hop.ParamBuiltinOp;
import org.apache.sysml.hops.Hop.ReOrgOp;
import org.apache.sysml.hops.HopsException;
import org.apache.sysml.hops.LiteralOp;
import org.apache.sysml.hops.Hop.OpOp2;
import org.apache.sysml.hops.ParameterizedBuiltinOp;
import org.apache.sysml.hops.ReorgOp;
import org.apache.sysml.parser.DataExpression;
import org.apache.sysml.parser.Statement;
import org.apache.sysml.parser.Expression.DataType;
import org.apache.sysml.parser.Expression.ValueType;

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
	private static final Log LOG = LogFactory.getLog(RewriteAlgebraicSimplificationStatic.class.getName());
	
	//valid aggregation operation types for rowOp to colOp conversions and vice versa
	private static AggOp[] LOOKUP_VALID_ROW_COL_AGGREGATE = new AggOp[]{AggOp.SUM, AggOp.SUM_SQ, AggOp.MIN, AggOp.MAX, AggOp.MEAN, AggOp.VAR};
	
	//valid binary operations for distributive and associate reorderings
	private static OpOp2[] LOOKUP_VALID_DISTRIBUTIVE_BINARY = new OpOp2[]{OpOp2.PLUS, OpOp2.MINUS}; 
	private static OpOp2[] LOOKUP_VALID_ASSOCIATIVE_BINARY = new OpOp2[]{OpOp2.PLUS, OpOp2.MULT}; 
		
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
 			hi = simplifyBinaryToUnaryOperation(hop, hi, i);     //e.g., X*X -> X^2 (pow2), X+X -> X*2, (X>0)-(X<0) -> sign(X)
 			hi = canonicalizeMatrixMultScalarAdd(hi);            //e.g., eps+U%*%t(V) -> U%*%t(V)+eps, U%*%t(V)-eps -> U%*%t(V)+(-eps) 
 			hi = simplifyReverseOperation(hop, hi, i);           //e.g., table(seq(1,nrow(X),1),seq(nrow(X),1,-1)) %*% X -> rev(X)
			hi = simplifyMultiBinaryToBinaryOperation(hi);       //e.g., 1-X*Y -> X 1-* Y
 			hi = simplifyDistributiveBinaryOperation(hop, hi, i);//e.g., (X-Y*X) -> (1-Y)*X
 			hi = simplifyBushyBinaryOperation(hop, hi, i);       //e.g., (X*(Y*(Z%*%v))) -> (X*Y)*(Z%*%v)
 			hi = simplifyUnaryAggReorgOperation(hop, hi, i);     //e.g., sum(t(X)) -> sum(X)
 			hi = pushdownUnaryAggTransposeOperation(hop, hi, i); //e.g., colSums(t(X)) -> t(rowSums(X))
			hi = simplifyUnaryPPredOperation(hop, hi, i);        //e.g., abs(ppred()) -> ppred(), others: round, ceil, floor
 			hi = simplifyTransposedAppend(hop, hi, i);           //e.g., t(cbind(t(A),t(B))) -> rbind(A,B);
 			hi = fuseBinarySubDAGToUnaryOperation(hop, hi, i);   //e.g., X*(1-X)-> sprop(X) || 1/(1+exp(-X)) -> sigmoid(X) || X*(X>0) -> selp(X)
			hi = simplifyTraceMatrixMult(hop, hi, i);            //e.g., trace(X%*%Y)->sum(X*t(Y));  
			hi = simplifySlicedMatrixMult(hop, hi, i);           //e.g., (X%*%Y)[1,1] -> X[1,] %*% Y[,1];
			hi = simplifyConstantSort(hop, hi, i);               //e.g., order(matrix())->matrix/seq; 
			hi = simplifyOrderedSort(hop, hi, i);                //e.g., order(matrix())->seq; 
			hi = removeUnnecessaryReorgOperation(hop, hi, i);    //e.g., t(t(X))->X; rev(rev(X))->X potentially introduced by other rewrites
			hi = simplifyTransposeAggBinBinaryChains(hop, hi, i);//e.g., t(t(A)%*%t(B)+C) -> B%*%A+t(C)
			hi = removeUnnecessaryMinus(hop, hi, i);             //e.g., -(-X)->X; potentially introduced by simplfiy binary or dyn rewrites
			hi = simplifyGroupedAggregate(hi);          	     //e.g., aggregate(target=X,groups=y,fn="count") -> aggregate(target=y,groups=y,fn="count")
			hi = fuseMinusNzBinaryOperation(hop, hi, i);         //e.g., X-mean*ppred(X,0,!=) -> X -nz mean
			hi = fuseLogNzUnaryOperation(hop, hi, i);            //e.g., ppred(X,0,"!=")*log(X) -> log_nz(X)
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
					HopRewriteUtils.addChildReference(bop, new LiteralOp(0), 0);
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
					HopRewriteUtils.addChildReference(bop, new LiteralOp(0), 0);
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
					Hop newMin = new LiteralOp(newMinVal);
					Hop newMax = new LiteralOp(newMaxVal);
					
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
	 * Handle simplification of binary operations (relies on previous common subexpression elimination).
	 * At the same time this servers as a canonicalization for more complex rewrites. 
	 * 
	 * X+X -> X*2, X*X -> X^2, (X>0)-(X<0) -> sign(X)
	 * @throws HopsException 
	 */
	private Hop simplifyBinaryToUnaryOperation( Hop parent, Hop hi, int pos ) 
		throws HopsException
	{
		if( hi instanceof BinaryOp )
		{
			BinaryOp bop = (BinaryOp)hi;
			Hop left = hi.getInput().get(0);
			Hop right = hi.getInput().get(1);
			
			//patterns: X+X -> X*2, X*X -> X^2,
			if( left == right && left.getDataType()==DataType.MATRIX )
			{
				//note: we simplify this to unary operations first (less mem and better MR plan),
				//however, we later compile specific LOPS for X*2 and X^2
				if( bop.getOp()==OpOp2.PLUS ) //X+X -> X*2
				{
					bop.setOp(OpOp2.MULT);
					LiteralOp tmp = new LiteralOp(2);
					bop.getInput().remove(1);
					right.getParent().remove(bop);
					HopRewriteUtils.addChildReference(hi, tmp, 1);

					LOG.debug("Applied simplifyBinaryToUnaryOperation1");
				}
				else if ( bop.getOp()==OpOp2.MULT ) //X*X -> X^2
				{
					bop.setOp(OpOp2.POW);
					LiteralOp tmp = new LiteralOp(2);
					bop.getInput().remove(1);
					right.getParent().remove(bop);
					HopRewriteUtils.addChildReference(hi, tmp, 1);
					
					LOG.debug("Applied simplifyBinaryToUnaryOperation2");
				}
			}
			//patterns: (X>0)-(X<0) -> sign(X)
			else if( bop.getOp() == OpOp2.MINUS 
				&& left instanceof BinaryOp && right instanceof BinaryOp
				&& ((BinaryOp)left).getOp()==OpOp2.GREATER && ((BinaryOp)right).getOp()==OpOp2.LESS 
				&& left.getInput().get(0) == right.getInput().get(0) 
				&& left.getInput().get(1) instanceof LiteralOp
				&& HopRewriteUtils.getDoubleValue((LiteralOp)left.getInput().get(1))==0
				&& right.getInput().get(1) instanceof LiteralOp
				&& HopRewriteUtils.getDoubleValue((LiteralOp)right.getInput().get(1))==0 )
			{
				UnaryOp uop = HopRewriteUtils.createUnary(left.getInput().get(0), OpOp1.SIGN);
				
				HopRewriteUtils.removeChildReferenceByPos(parent, hi, pos);
				HopRewriteUtils.removeAllChildReferences(hi);
				HopRewriteUtils.addChildReference(parent, uop, pos);
				if( left.getParent().isEmpty() )
					HopRewriteUtils.removeAllChildReferences(left);
				if( right.getParent().isEmpty() )
					HopRewriteUtils.removeAllChildReferences(right);
				
				hi = uop;
				
				LOG.debug("Applied simplifyBinaryToUnaryOperation3");
			}
		}
		
		return hi;
	}
	
	/**
	 * Rewrite to canonicalize all patterns like U%*%V+eps, eps+U%*%V, and
	 * U%*%V-eps into the common representation U%*%V+s which simplifies 
	 * subsequent rewrites (e.g., wdivmm or wcemm with epsilon).   
	 * 
	 * @param parent
	 * @param hi
	 * @param pos
	 * @return
	 * @throws HopsException
	 */
	private Hop canonicalizeMatrixMultScalarAdd( Hop hi ) 
		throws HopsException
	{
		//pattern: binary operation (+ or -) of matrix mult and scalar 		
		if( hi instanceof BinaryOp )
		{
			BinaryOp bop = (BinaryOp)hi;
			Hop left = hi.getInput().get(0);
			Hop right = hi.getInput().get(1);
			
			//pattern: (eps + U%*%V) -> (U%*%V+eps)
			if( left.getDataType().isScalar() && right instanceof AggBinaryOp
				&& bop.getOp()==OpOp2.PLUS )
			{
				HopRewriteUtils.removeAllChildReferences(bop);
				HopRewriteUtils.addChildReference(bop, right, 0);
				HopRewriteUtils.addChildReference(bop, left, 1);
				LOG.debug("Applied canonicalizeMatrixMultScalarAdd1 (line "+hi.getBeginLine()+").");
			}
			//pattern: (U%*%V - eps) -> (U%*%V + (-eps))
			else if( right.getDataType().isScalar() && left instanceof AggBinaryOp
					&& bop.getOp() == OpOp2.MINUS )
			{
				bop.setOp(OpOp2.PLUS);
				HopRewriteUtils.removeChildReferenceByPos(bop, right, 1);
				HopRewriteUtils.addChildReference(bop, 
						HopRewriteUtils.createBinary(new LiteralOp(0), right, OpOp2.MINUS), 1);				
				LOG.debug("Applied canonicalizeMatrixMultScalarAdd2 (line "+hi.getBeginLine()+").");
			}
		}
		
		return hi;
	}

	/**
	 * NOTE: this would be by definition a dynamic rewrite; however, we apply it as a static
	 * rewrite in order to apply it before splitting dags which would hide the table information
	 * if dimensions are not specified.
	 * 
	 * 
	 * @param parent
	 * @param hi
	 * @param pos
	 * @return
	 * @throws HopsException
	 */
	private Hop simplifyReverseOperation( Hop parent, Hop hi, int pos ) 
		throws HopsException
	{
		if(    hi instanceof AggBinaryOp 
			&& hi.getInput().get(0) instanceof TernaryOp )
		{
			TernaryOp top = (TernaryOp) hi.getInput().get(0);
			
			if( top.getOp()==OpOp3.CTABLE
				&& HopRewriteUtils.isBasic1NSequence(top.getInput().get(0))
				&& HopRewriteUtils.isBasicN1Sequence(top.getInput().get(1)) 
				&& top.getInput().get(0).getDim1()==top.getInput().get(1).getDim1())
			{
				ReorgOp rop = HopRewriteUtils.createReorg(hi.getInput().get(1), ReOrgOp.REV);
				
				HopRewriteUtils.removeChildReferenceByPos(parent, hi, pos);
				HopRewriteUtils.addChildReference(parent, rop, pos);
				if( hi.getParent().isEmpty() ) 
					HopRewriteUtils.removeAllChildReferences(hi);
				if( top.getParent().isEmpty() )
					HopRewriteUtils.removeAllChildReferences(top);
				
				hi = rop;
				
				LOG.debug("Applied simplifyReverseOperation.");
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
						LiteralOp literal = new LiteralOp(1);
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
						LiteralOp literal = new LiteralOp(1);
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
			if(   (rop.getOp()==ReOrgOp.TRANSPOSE || rop.getOp()==ReOrgOp.RESHAPE
					|| rop.getOp() == ReOrgOp.REV )         //valid reorg
				&& rop.getParent().size()==1 )              //uagg only reorg consumer
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
	 * 
	 * @param parent
	 * @param hi
	 * @param pos
	 * @return
	 */
	private Hop pushdownUnaryAggTransposeOperation( Hop parent, Hop hi, int pos )
	{
		if( hi instanceof AggUnaryOp && hi.getParent().size()==1 
			&& (((AggUnaryOp) hi).getDirection()==Direction.Row || ((AggUnaryOp) hi).getDirection()==Direction.Col)	
			&& hi.getInput().get(0) instanceof ReorgOp && hi.getInput().get(0).getParent().size()==1
			&& ((ReorgOp)hi.getInput().get(0)).getOp()==ReOrgOp.TRANSPOSE
			&& HopRewriteUtils.isValidOp(((AggUnaryOp) hi).getOp(), LOOKUP_VALID_ROW_COL_AGGREGATE) )
		{
			AggUnaryOp uagg = (AggUnaryOp) hi;
			
			//get input rewire existing operators (remove inner transpose)
			Hop input = uagg.getInput().get(0).getInput().get(0);
			HopRewriteUtils.removeAllChildReferences(hi.getInput().get(0));
			HopRewriteUtils.removeAllChildReferences(hi);
			HopRewriteUtils.removeChildReferenceByPos(parent, hi, pos);
			
			//pattern 1: row-aggregate to col aggregate, e.g., rowSums(t(X))->t(colSums(X))
			if( uagg.getDirection()==Direction.Row ) {
				uagg.setDirection(Direction.Col); 
				LOG.debug("Applied pushdownUnaryAggTransposeOperation1 (line "+hi.getBeginLine()+").");						
			}
			//pattern 2: col-aggregate to row aggregate, e.g., colSums(t(X))->t(rowSums(X))
			else if( uagg.getDirection()==Direction.Col ) {
				uagg.setDirection(Direction.Row); 
				LOG.debug("Applied pushdownUnaryAggTransposeOperation2 (line "+hi.getBeginLine()+").");
			}
			
			//create outer transpose operation and rewire operators
			HopRewriteUtils.addChildReference(uagg, input); uagg.refreshSizeInformation();
			Hop trans = HopRewriteUtils.createTranspose(uagg); //incl refresh size
			HopRewriteUtils.addChildReference(parent, trans, pos); //by def, same size
			
			hi = trans;	
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
	private Hop simplifyUnaryPPredOperation( Hop parent, Hop hi, int pos )
	{
		if( hi instanceof UnaryOp && hi.getDataType()==DataType.MATRIX  //unaryop
			&& hi.getInput().get(0) instanceof BinaryOp                 //binaryop - ppred
			&& ((BinaryOp)hi.getInput().get(0)).isPPredOperation() )
		{
			UnaryOp uop = (UnaryOp) hi; //valid unary op
			if( uop.getOp()==OpOp1.ABS || uop.getOp()==OpOp1.SIGN
				|| uop.getOp()==OpOp1.SELP || uop.getOp()==OpOp1.CEIL
				|| uop.getOp()==OpOp1.FLOOR || uop.getOp()==OpOp1.ROUND )
			{
				//clear link unary-binary
				Hop input = uop.getInput().get(0);
				HopRewriteUtils.removeAllChildReferences(hi);
				
				HopRewriteUtils.removeChildReferenceByPos(parent, hi, pos);
				HopRewriteUtils.addChildReference(parent, input, pos);
				hi = input;
				
				LOG.debug("Applied simplifyUnaryPPredOperation.");	
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
	private Hop simplifyTransposedAppend( Hop parent, Hop hi, int pos )
	{
		//e.g., t(cbind(t(A),t(B))) --> rbind(A,B), t(rbind(t(A),t(B))) --> cbind(A,B)		
		if(   hi instanceof ReorgOp && ((ReorgOp)hi).getOp()==ReOrgOp.TRANSPOSE  //t() rooted
		   && hi.getInput().get(0) instanceof BinaryOp
		   && (((BinaryOp)hi.getInput().get(0)).getOp()==OpOp2.CBIND    //append (cbind/rbind)
		    || ((BinaryOp)hi.getInput().get(0)).getOp()==OpOp2.RBIND) 
		   && hi.getInput().get(0).getParent().size() == 1 ) //single consumer of append
		{
			BinaryOp bop = (BinaryOp)hi.getInput().get(0);
			if( bop.getInput().get(0) instanceof ReorgOp  //both inputs transpose ops
				&& ((ReorgOp)bop.getInput().get(0)).getOp()==ReOrgOp.TRANSPOSE
				&& bop.getInput().get(0).getParent().size() == 1 //single consumer of transpose
				&& bop.getInput().get(1) instanceof ReorgOp 
				&& ((ReorgOp)bop.getInput().get(1)).getOp()==ReOrgOp.TRANSPOSE
				&& bop.getInput().get(1).getParent().size() == 1 ) //single consumer of transpose
			{
				Hop left = bop.getInput().get(0).getInput().get(0);
				Hop right = bop.getInput().get(1).getInput().get(0);
				
				//create new subdag (no in-place dag update to prevent anomalies with
				//multiple consumers during rewrite process)
				HopRewriteUtils.removeChildReferenceByPos(parent, hi, pos);
				OpOp2 binop = (bop.getOp()==OpOp2.CBIND) ? OpOp2.RBIND : OpOp2.CBIND;
				BinaryOp bopnew = HopRewriteUtils.createBinary(left, right, binop);
				HopRewriteUtils.addChildReference(parent, bopnew, pos);
				
				hi = bopnew;
				LOG.debug("Applied simplifyTransposedAppend (line "+hi.getBeginLine()+").");				
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
			
			//select positive (selp) operator; pattern: max(X,0) -> selp+
			if( bop.getOp() == OpOp2.MAX && left.getDataType()==DataType.MATRIX 
					&& right instanceof LiteralOp && HopRewriteUtils.getDoubleValue((LiteralOp)right)==0 )
			{
				UnaryOp unary = HopRewriteUtils.createUnary(left, OpOp1.SELP);
				HopRewriteUtils.removeChildReferenceByPos(parent, bop, pos);
				HopRewriteUtils.addChildReference(parent, unary, pos);
				
				//cleanup if only consumer of intermediate
				if( bop.getParent().isEmpty() )
					HopRewriteUtils.removeAllChildReferences(bop);					
				hi = unary;
				
				LOG.debug("Applied fuseBinarySubDAGToUnaryOperation-selp3");
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
					rowExpr, rowExpr, new LiteralOp(1), HopRewriteUtils.createValueHop(X, false), true, false);
			HopRewriteUtils.setOutputBlocksizes(ix1, X.getRowsInBlock(), X.getColsInBlock());
			ix1.refreshSizeInformation();
			IndexingOp ix2 = new IndexingOp("tmp2", DataType.MATRIX, ValueType.DOUBLE, Y, 
					new LiteralOp(1), HopRewriteUtils.createValueHop(Y, true), colExpr, colExpr, false, true);
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
	 * Patterns: t(t(A)%*%t(B)+C) -> B%*%A+t(C)
	 * 
	 * @param parent
	 * @param hi
	 * @param pos
	 * @return
	 * @throws HopsException
	 */
	private Hop simplifyTransposeAggBinBinaryChains(Hop parent, Hop hi, int pos) 
		throws HopsException
	{
		if( hi instanceof ReorgOp && ((ReorgOp)hi).getOp()==ReOrgOp.TRANSPOSE //transpose
			&& hi.getInput().get(0) instanceof BinaryOp                       //basic binary
			&& ((BinaryOp)hi.getInput().get(0)).supportsMatrixScalarOperations()) 
		{
			Hop left = hi.getInput().get(0).getInput().get(0);
			Hop C = hi.getInput().get(0).getInput().get(1);
			
			//check matrix mult and both inputs transposes w/ single consumer
			if( left instanceof AggBinaryOp && C.getDataType().isMatrix()
				&& left.getInput().get(0).getParent().size()==1 && left.getInput().get(0) instanceof ReorgOp
				&& ((ReorgOp)left.getInput().get(0)).getOp()==ReOrgOp.TRANSPOSE     
				&& left.getInput().get(1).getParent().size()==1 && left.getInput().get(1) instanceof ReorgOp
				&& ((ReorgOp)left.getInput().get(1)).getOp()==ReOrgOp.TRANSPOSE )
			{
				Hop A = left.getInput().get(0).getInput().get(0);
				Hop B = left.getInput().get(1).getInput().get(0);
				
				AggBinaryOp abop = HopRewriteUtils.createMatrixMultiply(B, A);
				ReorgOp rop = HopRewriteUtils.createTranspose(C);
				BinaryOp bop = HopRewriteUtils.createBinary(abop, rop, OpOp2.PLUS);
				
				HopRewriteUtils.removeChildReferenceByPos(parent, hi, pos);
				HopRewriteUtils.addChildReference(parent, bop, pos);
				
				hi = bop;
				LOG.debug("Applied simplifyTransposeAggBinBinaryChains (line "+hi.getBeginLine()+").");						
			}  
		}
		
		return hi;
	}
	
	/**
	 * Pattners: t(t(X)) -> X, rev(rev(X)) -> X
	 * 
	 * @param parent
	 * @param hi
	 * @param pos
	 */
	private Hop removeUnnecessaryReorgOperation(Hop parent, Hop hi, int pos)
	{
		ReOrgOp[] lookup = new ReOrgOp[]{ReOrgOp.TRANSPOSE, ReOrgOp.REV};
		
		if( hi instanceof ReorgOp && HopRewriteUtils.isValidOp(((ReorgOp)hi).getOp(), lookup)  ) //first reorg
		{
			ReOrgOp firstOp = ((ReorgOp)hi).getOp();
			Hop hi2 = hi.getInput().get(0);
			if( hi2 instanceof ReorgOp && ((ReorgOp)hi2).getOp()==firstOp ) //second reorg w/ same type
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
				
				LOG.debug("Applied removeUnecessaryReorgOperation.");
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
			
			if( phi.isCountFunction() //aggregate(fn="count")
				&& phi.getTargetHop().getDim2()==1 ) //only for vector
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
	private Hop fuseLogNzUnaryOperation(Hop parent, Hop hi, int pos) 
		throws HopsException
	{
		//pattern ppred(X,0,"!=")*log(X) -> log_nz(X)
		//note: this is done as a hop rewrite in order to significantly reduce the 
		//memory estimate and to prevent dense intermediates if X is ultra sparse  
		if( hi instanceof BinaryOp && ((BinaryOp)hi).getOp()==OpOp2.MULT
			&& hi.getInput().get(0).getDataType()==DataType.MATRIX
			&& hi.getInput().get(1).getDataType()==DataType.MATRIX
			&& hi.getInput().get(1) instanceof UnaryOp 
			&& ((UnaryOp)hi.getInput().get(1)).getOp()==OpOp1.LOG )
		{
			Hop pred = hi.getInput().get(0);
			Hop X = hi.getInput().get(1).getInput().get(0);
			
			if(    pred instanceof BinaryOp && ((BinaryOp)pred).getOp()==OpOp2.NOTEQUAL
				&& pred.getInput().get(0) == X //depend on common subexpression elimination
				&& pred.getInput().get(1) instanceof LiteralOp
				&& HopRewriteUtils.getDoubleValueSafe((LiteralOp)pred.getInput().get(1))==0 )
			{
				Hop hnew = new UnaryOp("tmp", DataType.MATRIX, ValueType.DOUBLE, OpOp1.LOG_NZ, X);
				HopRewriteUtils.setOutputBlocksizes(hnew, hi.getRowsInBlock(), hi.getColsInBlock());
				hnew.refreshSizeInformation();
		
				//relink new hop into original position
				HopRewriteUtils.removeChildReferenceByPos(parent, hi, pos);
				HopRewriteUtils.addChildReference(parent, hnew, pos);
				hi = hnew;
				
				LOG.debug("Applied fuseLogNzUnaryOperation (line "+hi.getBeginLine()+").");	
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
				inputargs.put("dir", new LiteralOp(direction));
				inputargs.put("ignore", new LiteralOp(true));
				inputargs.put("cast", new LiteralOp(false));
			
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
		
		if(    hi instanceof TernaryOp && hi.getInput().size()==5 //table without weights 
			&& hi.getInput().get(2) instanceof LiteralOp
			&& HopRewriteUtils.getDoubleValue((LiteralOp)hi.getInput().get(2))==1	
			&& hi.getInput().get(3) instanceof LiteralOp && hi.getInput().get(4) instanceof LiteralOp)
		{
			if(  (HopRewriteUtils.isBasic1NSequence(hi.getInput().get(0)) &&
				   hi.getInput().get(4) instanceof LiteralOp)   //pattern a: table(seq(1,nrow(v)), v, nrow(v), m)
			   ||(HopRewriteUtils.isBasic1NSequence(hi.getInput().get(1)) &&
				   hi.getInput().get(3) instanceof LiteralOp) ) //pattern b: table(v, seq(1,nrow(v)), m, nrow(v))
			{
				//determine variable parameters for pattern a/b
				int ixTgt = HopRewriteUtils.isBasic1NSequence(hi.getInput().get(0)) ? 1 : 0;
				int ixMax = HopRewriteUtils.isBasic1NSequence(hi.getInput().get(0)) ? 4 : 3;
				String direction = HopRewriteUtils.isBasic1NSequence(hi.getInput().get(0)) ? "cols" : "rows";
				
				//setup input parameter hops
				HashMap<String,Hop> inputargs = new HashMap<String,Hop>();
				inputargs.put("target", hi.getInput().get(ixTgt));
				inputargs.put("max", hi.getInput().get(ixMax));
				inputargs.put("dir", new LiteralOp(direction));
				inputargs.put("ignore", new LiteralOp(false));
				inputargs.put("cast", new LiteralOp(true));
			
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
