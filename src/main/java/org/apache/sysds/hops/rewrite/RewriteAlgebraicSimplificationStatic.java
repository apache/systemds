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

package org.apache.sysds.hops.rewrite;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Iterator;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Set;

import org.apache.sysds.hops.AggBinaryOp;
import org.apache.sysds.hops.AggUnaryOp;
import org.apache.sysds.hops.BinaryOp;
import org.apache.sysds.hops.DataGenOp;
import org.apache.sysds.hops.Hop;
import org.apache.sysds.hops.IndexingOp;
import org.apache.sysds.hops.LiteralOp;
import org.apache.sysds.hops.NaryOp;
import org.apache.sysds.hops.OptimizerUtils;
import org.apache.sysds.hops.ParameterizedBuiltinOp;
import org.apache.sysds.hops.ReorgOp;
import org.apache.sysds.hops.TernaryOp;
import org.apache.sysds.hops.UnaryOp;
import org.apache.sysds.common.Types.AggOp;
import org.apache.sysds.common.Types.Direction;
import org.apache.sysds.common.Types.OpOp1;
import org.apache.sysds.common.Types.OpOp2;
import org.apache.sysds.common.Types.OpOp3;
import org.apache.sysds.common.Types.OpOpDG;
import org.apache.sysds.common.Types.OpOpData;
import org.apache.sysds.common.Types.OpOpN;
import org.apache.sysds.common.Types.ParamBuiltinOp;
import org.apache.sysds.common.Types.ReOrgOp;
import org.apache.sysds.parser.DataExpression;
import org.apache.sysds.parser.Statement;
import org.apache.sysds.common.Types.DataType;
import org.apache.sysds.common.Types.ValueType;

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
	//valid aggregation operation types for rowOp to colOp conversions and vice versa
	private static final AggOp[] LOOKUP_VALID_ROW_COL_AGGREGATE = new AggOp[] {
			AggOp.SUM, AggOp.SUM_SQ, AggOp.MIN, AggOp.MAX, AggOp.MEAN, AggOp.VAR};

	//valid binary operations for distributive and associate reorderings
	private static final OpOp2[] LOOKUP_VALID_DISTRIBUTIVE_BINARY = new OpOp2[] {OpOp2.PLUS, OpOp2.MINUS};
	private static final OpOp2[] LOOKUP_VALID_ASSOCIATIVE_BINARY = new OpOp2[] {OpOp2.PLUS, OpOp2.MULT};

	//valid binary operations for scalar operations
	private static final OpOp2[] LOOKUP_VALID_SCALAR_BINARY = new OpOp2[] {OpOp2.AND, OpOp2.DIV,
			OpOp2.EQUAL, OpOp2.GREATER, OpOp2.GREATEREQUAL, OpOp2.INTDIV, OpOp2.LESS, OpOp2.LESSEQUAL,
			OpOp2.LOG, OpOp2.MAX, OpOp2.MIN, OpOp2.MINUS, OpOp2.MODULUS, OpOp2.MULT, OpOp2.NOTEQUAL,
			OpOp2.OR, OpOp2.PLUS, OpOp2.POW};

	@Override
	public ArrayList<Hop> rewriteHopDAGs(ArrayList<Hop> roots, ProgramRewriteStatus state)
	{
		if( roots == null )
			return roots;

		//one pass rewrite-descend (rewrite created pattern)
		for( Hop h : roots )
			rule_AlgebraicSimplification( h, false );
		Hop.resetVisitStatus(roots, true);

		//one pass descend-rewrite (for rollup) 
		for( Hop h : roots )
			rule_AlgebraicSimplification( h, true );
		Hop.resetVisitStatus(roots, true);

		//cleanup remove (twrite <- tread) pairs (unless checkpointing)
		removeTWriteTReadPairs(roots);

		return roots;
	}

	@Override
	public Hop rewriteHopDAG(Hop root, ProgramRewriteStatus state)
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
	 * @param hop high-level operator
	 * @param descendFirst if process children recursively first
	 */
	private void rule_AlgebraicSimplification(Hop hop, boolean descendFirst)
	{
		if(hop.isVisited())
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
			hi = fuseDatagenAndBinaryOperation(hi);              //e.g., rand(min=-1,max=1)*7 -> rand(min=-7,max=7)
			hi = fuseDatagenAndMinusOperation(hi);               //e.g., -(rand(min=-2,max=1)) -> rand(min=-1,max=2)
			hi = foldMultipleAppendOperations(hi);               //e.g., cbind(X,cbind(Y,Z)) -> cbind(X,Y,Z)
			hi = simplifyBinaryToUnaryOperation(hop, hi, i);     //e.g., X*X -> X^2 (pow2), X+X -> X*2, (X>0)-(X<0) -> sign(X)
			hi = canonicalizeMatrixMultScalarAdd(hi);            //e.g., eps+U%*%t(V) -> U%*%t(V)+eps, U%*%t(V)-eps -> U%*%t(V)+(-eps) 
			hi = simplifyCTableWithConstMatrixInputs(hi);        //e.g., table(X, matrix(1,...)) -> table(X, 1)
			hi = removeUnnecessaryCTable(hop, hi, i);            //e.g., sum(table(X, 1)) -> nrow(X) and sum(table(1, Y)) -> nrow(Y) and sum(table(X, Y)) -> nrow(X)
			hi = simplifyConstantConjunction(hop, hi, i);        //e.g., a & !a -> FALSE 
			hi = simplifyReverseOperation(hop, hi, i);           //e.g., table(seq(1,nrow(X),1),seq(nrow(X),1,-1)) %*% X -> rev(X)
			hi = simplifyReverseSequence(hop, hi, i);            //e.g., rev(seq(1,n)) -> seq(n,1)
			hi = simplifyReverseSequenceStep(hop, hi, i);        //e.g., rev(seq(1,n,2)) -> rev(n,1,-2)
			if(OptimizerUtils.ALLOW_OPERATOR_FUSION)
				hi = simplifyMultiBinaryToBinaryOperation(hi);       //e.g., 1-X*Y -> X 1-* Y
			hi = simplifyDistributiveBinaryOperation(hop, hi, i);//e.g., (X-Y*X) -> (1-Y)*X
			hi = simplifyTransposeInDetOperation(hop, hi, i);    //e.g., det(t(X)) -> det(X)
			hi = simplifyBushyBinaryOperation(hop, hi, i);       //e.g., (X*(Y*(Z%*%v))) -> (X*Y)*(Z%*%v)
			hi = simplifyUnaryAggReorgOperation(hop, hi, i);     //e.g., sum(t(X)) -> sum(X)
			hi = removeUnnecessaryAggregates(hi);                //e.g., sum(rowSums(X)) -> sum(X)
			hi = simplifyBinaryMatrixScalarOperation(hop, hi, i);//e.g., as.scalar(X*s) -> as.scalar(X)*s;
			hi = pushdownUnaryAggTransposeOperation(hop, hi, i); //e.g., colSums(t(X)) -> t(rowSums(X))
			hi = pushdownCSETransposeScalarOperation(hop, hi, i);//e.g., a=t(X), b=t(X^2) -> a=t(X), b=t(X)^2 for CSE t(X)
			hi = pushdownDetMultOperation(hop, hi, i);           //e.g., det(X%*%Y) -> det(X)*det(Y)
			hi = pushdownDetScalarMatrixMultOperation(hop, hi, i);  //e.g., det(lambda*X) -> lambda^nrow(X)*det(X)
			hi = pushdownSumBinaryMult(hop, hi, i);              //e.g., sum(lambda*X) -> lambda*sum(X)
			hi = pullupAbs(hop, hi, i);                          //e.g., abs(X)*abs(Y) --> abs(X*Y)
			hi = simplifyUnaryPPredOperation(hop, hi, i);        //e.g., abs(ppred()) -> ppred(), others: round, ceil, floor
			hi = simplifyTransposedAppend(hop, hi, i);           //e.g., t(cbind(t(A),t(B))) -> rbind(A,B);
			if(OptimizerUtils.ALLOW_OPERATOR_FUSION)
				hi = fuseBinarySubDAGToUnaryOperation(hop, hi, i);   //e.g., X*(1-X)-> sprop(X) || 1/(1+exp(-X)) -> sigmoid(X) || X*(X>0) -> selp(X)
			hi = simplifyTraceMatrixMult(hop, hi, i);            //e.g., trace(X%*%Y)->sum(X*t(Y));
			hi = simplifyTraceSum(hop, hi, i);                   //e.g. , trace(A+B)->trace(A)+trace(B);
			hi = simplifyTraceTranspose(hop, hi, i);             //e.g. , trace(t(A))->trace(A)
			hi = simplifySlicedMatrixMult(hop, hi, i);           //e.g., (X%*%Y)[1,1] -> X[1,] %*% Y[,1];
			hi = simplifyListIndexing(hi);                       //e.g., L[i:i, 1:ncol(L)] -> L[i:i, 1:1]
			hi = simplifyScalarIndexing(hop, hi, i);             //e.g., as.scalar(X[i,1])->X[i,1] w/ scalar output
			hi = simplifyConstantSort(hop, hi, i);               //e.g., order(matrix())->matrix/seq;
			hi = simplifyOrderedSort(hop, hi, i);                //e.g., order(matrix())->seq;
			hi = fuseOrderOperationChain(hi);                    //e.g., order(order(X,2),1) -> order(X,(12))
			hi = removeUnnecessaryReorgOperation(hop, hi, i);    //e.g., t(t(X))->X; rev(rev(X))->X potentially introduced by other rewrites
			hi = removeUnnecessaryRemoveEmpty(hop, hi, i);       //e.g., nrow(removeEmpty(A)) -> nnz(A) iff col vector
			hi = simplifyTransposeAggBinBinaryChains(hop, hi, i);//e.g., t(t(A)%*%t(B)+C) -> B%*%A+t(C)
			hi = simplifyReplaceZeroOperation(hop, hi, i);       //e.g., X + (X==0) * s -> replace(X, 0, s)
			hi = removeUnnecessaryMinus(hop, hi, i);             //e.g., -(-X)->X; potentially introduced by simplify binary or dyn rewrites
			hi = simplifyGroupedAggregate(hi);                   //e.g., aggregate(target=X,groups=y,fn="count") -> aggregate(target=y,groups=y,fn="count")
			if(OptimizerUtils.ALLOW_OPERATOR_FUSION) {
				hi = fuseMinusNzBinaryOperation(hop, hi, i);         //e.g., X-mean*ppred(X,0,!=) -> X -nz mean
				hi = fuseLogNzUnaryOperation(hop, hi, i);            //e.g., ppred(X,0,"!=")*log(X) -> log_nz(X)
				hi = fuseLogNzBinaryOperation(hop, hi, i);           //e.g., ppred(X,0,"!=")*log(X,0.5) -> log_nz(X,0.5)
			}
			hi = simplifyOuterSeqExpand(hop, hi, i);             //e.g., outer(v, seq(1,m), "==") -> rexpand(v, max=m, dir=row, ignore=true, cast=false)
			hi = simplifyBinaryComparisonChain(hop, hi, i);      //e.g., outer(v1,v2,"==")==1 -> outer(v1,v2,"=="), outer(v1,v2,"==")==0 -> outer(v1,v2,"!="),
			hi = simplifyCumsumColOrFullAggregates(hi);          //e.g., colSums(cumsum(X)) -> cumSums(X*seq(nrow(X),1))
			hi = simplifyCumsumReverse(hop, hi, i);              //e.g., rev(cumsum(rev(X))) -> X + colSums(X) - cumsum(X)
			hi = simplifyNegatedSubtraction(hop, hi, i);         //e.g., -(B-A)->A-B
			hi = simplifyTransposeAddition(hop, hi, i);          //e.g., t(A+s1)+s2 -> t(A)+(s1+s2) + potential constant folding
			hi = simplifyNotOverComparisons(hop, hi, i);         //e.g., !(A>B) -> (A<=B)
			hi = simplifyMatrixScalarPMOperation(hop, hi, i);             //e.g., a-A-b -> (a-b)-A; a+A-b -> (a-b)+A
			//hi = removeUnecessaryPPred(hop, hi, i);            //e.g., ppred(X,X,"==")->matrix(1,rows=nrow(X),cols=ncol(X))

			//process childs recursively after rewrites (to investigate pattern newly created by rewrites)
			if( !descendFirst )
				rule_AlgebraicSimplification(hi, descendFirst);
		}

		hop.setVisited();
	}

	private Hop simplifyMatrixScalarPMOperation(Hop parent, Hop hi, int pos) {
		if (!(hi instanceof BinaryOp))
			return hi;

		BinaryOp outer = (BinaryOp) hi;
		Hop left = outer.getInput().get(0);
		Hop right = outer.getInput().get(1);
		OpOp2 outerOp = outer.getOp();

		if ((outerOp != OpOp2.PLUS && outerOp != OpOp2.MINUS) || !(left instanceof BinaryOp))
			return hi;

		BinaryOp inner = (BinaryOp) left;
		Hop a = inner.getInput().get(0);
		Hop A = inner.getInput().get(1);
		Hop b = right;
		OpOp2 innerOp = inner.getOp();

		// Check for valid types: a and b must be scalar, A must be matrix
		java.util.function.Predicate<Hop> isScalar = h -> h.getDataType().isScalar();
		if (!isScalar.test(a) || !isScalar.test(b) || A.getDataType() != DataType.MATRIX)
			return hi;

		// Determine the scalarOp (between a and b) and matrixOp (with A)
		OpOp2 scalarOp = null;
		OpOp2 matrixOp = null;

		if (innerOp == OpOp2.MINUS && outerOp == OpOp2.MINUS) {
			scalarOp = OpOp2.MINUS;
			matrixOp = OpOp2.MINUS;
		}
		else if (innerOp == OpOp2.PLUS && outerOp == OpOp2.MINUS) {
			scalarOp = OpOp2.MINUS;
			matrixOp = OpOp2.PLUS;
		}
		else if (innerOp == OpOp2.MINUS && outerOp == OpOp2.PLUS) {
			scalarOp = OpOp2.PLUS;
			matrixOp = OpOp2.MINUS;
		}
		else if (innerOp == OpOp2.PLUS && outerOp == OpOp2.PLUS) {
			scalarOp = OpOp2.PLUS;
			matrixOp = OpOp2.PLUS;
		}
		else {
			// No valid pattern
			return hi;
		}

		// Create and replace the rewritten expression
		Hop scalarCombined = HopRewriteUtils.createBinary(a, b, scalarOp);
		Hop result = HopRewriteUtils.createBinary(scalarCombined, A, matrixOp);

		HopRewriteUtils.replaceChildReference(parent, hi, result, pos);
		LOG.debug("Applied simplifyMatrixScalarPMOperation");
		return result;
	}


	private static Hop simplifyTransposeAddition(Hop parent, Hop hi, int pos) {
		//pattern: t(A+s1)+s2 -> t(A)+(s1+s2), and subsequent constant folding
		if (HopRewriteUtils.isBinary(hi, OpOp2.PLUS) 
			&& hi.isMatrix() && hi.getInput(1).isScalar()
			&& HopRewriteUtils.isReorg(hi.getInput(0), ReOrgOp.TRANS)
			&& hi.getInput(0).getParent().size() == 1
			&& HopRewriteUtils.isBinary(hi.getInput(0).getInput(0), OpOp2.PLUS)
			&& hi.getInput(0).getInput(0).getParent().size() == 1
			&& (hi.getInput(0).getInput(0).getInput(0).isScalar()
				|| hi.getInput(0).getInput(0).getInput(1).isScalar()))
		{
			int six = hi.getInput(0).getInput(0).getInput(0).isScalar() ? 0 : 1;
			Hop A = hi.getInput(0).getInput(0).getInput(six==0 ? 1 : 0);
			Hop s1 = hi.getInput(0).getInput(0).getInput(six);
			Hop s2 = hi.getInput(1);
			
			Hop tA = HopRewriteUtils.createTranspose(A);
			Hop s12 = HopRewriteUtils.createBinary(s1, s2, OpOp2.PLUS);
			Hop newHop = HopRewriteUtils.createBinary(tA, s12, OpOp2.PLUS);
			
			HopRewriteUtils.replaceChildReference(parent, hi, newHop, pos);
			HopRewriteUtils.cleanupUnreferenced(hi);
			hi = newHop;
			
			LOG.debug("Applied simplifyTransposeAddition (line " + hi.getBeginLine() + ").");
		}
		
		return hi;
	}

	private static Hop simplifyNegatedSubtraction(Hop parent, Hop hi, int pos) {
		//pattern: -(B-A)->A-B, but only of (B-A) consumed once
		if (HopRewriteUtils.isBinary(hi, OpOp2.MINUS) 
			&& HopRewriteUtils.isLiteralOfValue(hi.getInput(0), 0)
			&& HopRewriteUtils.isBinary(hi.getInput(1), OpOp2.MINUS)
			&& hi.getInput().get(1).getParent().size() == 1)
		{
			Hop B = hi.getInput(1).getInput(0);
			Hop A = hi.getInput(1).getInput(1);

			BinaryOp newHop = HopRewriteUtils.createBinary(A, B, OpOp2.MINUS);
			HopRewriteUtils.replaceChildReference(parent, hi, newHop, pos);
			HopRewriteUtils.cleanupUnreferenced(hi);
			hi = newHop;

			LOG.debug("Applied simplifyNegatedSubtraction (line " + hi.getBeginLine() + ").");
		}
		return hi;
	}


	private static Hop removeUnnecessaryVectorizeOperation(Hop hi)
	{
		//applies to all binary matrix operations, if one input is unnecessarily vectorized
		if(    hi instanceof BinaryOp && hi.getDataType()==DataType.MATRIX
				&& ((BinaryOp)hi).supportsMatrixScalarOperations()   )
		{
			BinaryOp bop = (BinaryOp)hi;
			Hop left = bop.getInput(0);
			Hop right = bop.getInput(1);

			//NOTE: these rewrites of binary cell operations need to be aware that right is
			//potentially a vector but the result is of the size of left
			//TODO move to dynamic rewrites (since size dependent to account for mv binary cell and outer operations)

			if( !(left.getDim1()>1 && left.getDim2()==1 && right.getDim1()==1 && right.getDim2()>1) ) // no outer
			{
				//check and remove right vectorized scalar
				if( left.getDataType() == DataType.MATRIX && right instanceof DataGenOp )
				{
					DataGenOp dright = (DataGenOp) right;
					if( dright.getOp()==OpOpDG.RAND && dright.hasConstantValue() )
					{
						Hop drightIn = dright.getInput().get(dright.getParamIndex(DataExpression.RAND_MIN));
						HopRewriteUtils.replaceChildReference(bop, dright, drightIn, 1);
						HopRewriteUtils.cleanupUnreferenced(dright);

						LOG.debug("Applied removeUnnecessaryVectorizeOperation1");
					}
				}
				//check and remove left vectorized scalar
				else if( right.getDataType() == DataType.MATRIX && left instanceof DataGenOp )
				{
					DataGenOp dleft = (DataGenOp) left;
					if( dleft.getOp()==OpOpDG.RAND && dleft.hasConstantValue()
							&& (left.getDim2()==1 || right.getDim2()>1)
							&& (left.getDim1()==1 || right.getDim1()>1))
					{
						Hop dleftIn = dleft.getInput().get(dleft.getParamIndex(DataExpression.RAND_MIN));
						HopRewriteUtils.replaceChildReference(bop, dleft, dleftIn, 0);
						HopRewriteUtils.cleanupUnreferenced(dleft);

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
	 * @param parent parent high-level operator
	 * @param hi high-level operator
	 * @param pos position
	 * @return high-level operator
	 */
	private static Hop removeUnnecessaryBinaryOperation( Hop parent, Hop hi, int pos )
	{
		if( hi instanceof BinaryOp )
		{
			BinaryOp bop = (BinaryOp)hi;
			Hop left = bop.getInput(0);
			Hop right = bop.getInput(1);
			//X/1 or X*1 -> X
			if(    left.getDataType()==DataType.MATRIX
					&& right instanceof LiteralOp && right.getValueType().isNumeric()
					&& ((LiteralOp)right).getDoubleValue()==1.0 )
			{
				if( bop.getOp()==OpOp2.DIV || bop.getOp()==OpOp2.MULT )
				{
					HopRewriteUtils.replaceChildReference(parent, bop, left, pos);
					hi = left;

					LOG.debug("Applied removeUnnecessaryBinaryOperation1 (line "+bop.getBeginLine()+")");
				}
			}
			//X-0 -> X
			else if(    left.getDataType()==DataType.MATRIX
					&& right instanceof LiteralOp && right.getValueType().isNumeric()
					&& ((LiteralOp)right).getDoubleValue()==0.0 )
			{
				if( bop.getOp()==OpOp2.MINUS )
				{
					HopRewriteUtils.replaceChildReference(parent, bop, left, pos);
					hi = left;

					LOG.debug("Applied removeUnnecessaryBinaryOperation2 (line "+bop.getBeginLine()+")");
				}
			}
			//1*X -> X
			else if(   right.getDataType()==DataType.MATRIX
					&& left instanceof LiteralOp && left.getValueType().isNumeric()
					&& ((LiteralOp)left).getDoubleValue()==1.0 )
			{
				if( bop.getOp()==OpOp2.MULT )
				{
					HopRewriteUtils.replaceChildReference(parent, bop, right, pos);
					hi = right;

					LOG.debug("Applied removeUnnecessaryBinaryOperation3 (line "+bop.getBeginLine()+")");
				}
			}
			//-1*X -> -X
			//note: this rewrite is necessary since the new antlr parser always converts
			//-X to -1*X due to mechanical reasons
			else if(   right.getDataType()==DataType.MATRIX
					&& left instanceof LiteralOp && left.getValueType().isNumeric()
					&& ((LiteralOp)left).getDoubleValue()==-1.0 )
			{
				if( bop.getOp()==OpOp2.MULT )
				{
					bop.setOp(OpOp2.MINUS);
					HopRewriteUtils.replaceChildReference(bop, left, new LiteralOp(0), 0);
					hi = bop;

					LOG.debug("Applied removeUnnecessaryBinaryOperation4 (line "+bop.getBeginLine()+")");
				}
			}
			//X*-1 -> -X (see comment above)
			else if(   left.getDataType()==DataType.MATRIX
					&& right instanceof LiteralOp && right.getValueType().isNumeric()
					&& ((LiteralOp)right).getDoubleValue()==-1.0 )
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

	public static Hop simplifyConstantConjunction(Hop parent, Hop hi, int pos) {
		if (hi instanceof BinaryOp) {
			BinaryOp bop = (BinaryOp) hi;
			Hop left = hi.getInput(0);
			Hop right = hi.getInput(1);

			// Patterns: a & !a --> FALSE / !a & a --> FALSE
			if (bop.getOp() == OpOp2.AND
				&& ((HopRewriteUtils.isUnary(right, OpOp1.NOT) && left == right.getInput(0)) 
				|| (HopRewriteUtils.isUnary(left, OpOp1.NOT) && left.getInput(0) == right)))
			{
				LiteralOp falseOp = new LiteralOp(false);

				// Ensure parent has the input before attempting replacement
				if (parent != null && parent.getInput().size() > pos) {
					HopRewriteUtils.replaceChildReference(parent, hi, falseOp, pos);
					HopRewriteUtils.cleanupUnreferenced(hi, left, right);
					hi = falseOp;
				}

				LOG.debug("Applied simplifyBooleanRewrite1 (line " + hi.getBeginLine() + ").");
			}
			// Pattern: a | !a --> TRUE
			else if (bop.getOp() == OpOp2.OR
				&& ((HopRewriteUtils.isUnary(right, OpOp1.NOT) && left == right.getInput(0)) 
				|| (HopRewriteUtils.isUnary(left, OpOp1.NOT) && left.getInput(0) == right)))
			{
				LiteralOp trueOp = new LiteralOp(true);

				// Ensure parent has the input before attempting replacement
				if (parent != null && parent.getInput().size() > pos) {
					HopRewriteUtils.replaceChildReference(parent, hi, trueOp, pos);
					HopRewriteUtils.cleanupUnreferenced(hi, left, right);
					hi = trueOp;
				}

				LOG.debug("Applied simplifyBooleanRewrite2 (line " + hi.getBeginLine() + ").");
			}
		}

		return hi;
	}


	/**
	 * Handle removal of unnecessary binary operations over rand data
	 *
	 * rand*7 -> rand(min*7,max*7); rand+7 -> rand(min+7,max+7); rand-7 -> rand(min+(-7),max+(-7))
	 * 7*rand -> rand(min*7,max*7); 7+rand -> rand(min+7,max+7); 
	 *
	 * @param hi high-order operation
	 * @return high-level operator
	 */
	@SuppressWarnings("incomplete-switch")
	private static Hop fuseDatagenAndBinaryOperation( Hop hi )
	{
		if( hi instanceof BinaryOp )
		{
			BinaryOp bop = (BinaryOp)hi;
			Hop left = bop.getInput(0);
			Hop right = bop.getInput(1);

			//NOTE: rewrite not applied if more than one datagen consumer because this would lead to 
			//the creation of multiple datagen ops and thus potentially different results if seed not specified)

			//left input rand and hence output matrix double, right scalar literal
			if( HopRewriteUtils.isDataGenOp(left, OpOpDG.RAND) &&
					right instanceof LiteralOp && left.getParent().size()==1 )
			{
				DataGenOp inputGen = (DataGenOp)left;
				Hop pdf = inputGen.getInput(DataExpression.RAND_PDF);
				Hop min = inputGen.getInput(DataExpression.RAND_MIN);
				Hop max = inputGen.getInput(DataExpression.RAND_MAX);
				double sval = ((LiteralOp)right).getDoubleValue();
				boolean pdfUniform = pdf instanceof LiteralOp
						&& DataExpression.RAND_PDF_UNIFORM.equals(((LiteralOp)pdf).getStringValue());

				if( HopRewriteUtils.isBinary(bop, OpOp2.MULT, OpOp2.PLUS, OpOp2.MINUS, OpOp2.DIV)
						&& min instanceof LiteralOp && max instanceof LiteralOp && pdfUniform )
				{
					//create fused data gen operator
					DataGenOp gen = null;
					switch( bop.getOp() ) { //fuse via scale and shift
						case MULT:  gen = HopRewriteUtils.copyDataGenOp(inputGen, sval, 0); break;
						case PLUS:
						case MINUS: gen = HopRewriteUtils.copyDataGenOp(inputGen,
								1, sval * ((bop.getOp()==OpOp2.MINUS)?-1:1)); break;
						case DIV:   gen = HopRewriteUtils.copyDataGenOp(inputGen, 1/sval, 0); break;
					}

					//rewire all parents (avoid anomalies with replicated datagen)
					List<Hop> parents = new ArrayList<>(bop.getParent());
					for( Hop p : parents )
						HopRewriteUtils.replaceChildReference(p, bop, gen);

					hi = gen;
					LOG.debug("Applied fuseDatagenAndBinaryOperation1 "
							+ "("+bop.getFilename()+", line "+bop.getBeginLine()+").");
				}
			}
			//right input rand and hence output matrix double, left scalar literal
			else if( right instanceof DataGenOp && ((DataGenOp)right).getOp()==OpOpDG.RAND &&
					left instanceof LiteralOp && right.getParent().size()==1 )
			{
				DataGenOp inputGen = (DataGenOp)right;
				Hop pdf = inputGen.getInput(DataExpression.RAND_PDF);
				Hop min = inputGen.getInput(DataExpression.RAND_MIN);
				Hop max = inputGen.getInput(DataExpression.RAND_MAX);
				double sval = ((LiteralOp)left).getDoubleValue();
				boolean pdfUniform = pdf instanceof LiteralOp
						&& DataExpression.RAND_PDF_UNIFORM.equals(((LiteralOp)pdf).getStringValue());

				if( (bop.getOp()==OpOp2.MULT || bop.getOp()==OpOp2.PLUS)
						&& min instanceof LiteralOp && max instanceof LiteralOp && pdfUniform )
				{
					//create fused data gen operator
					DataGenOp gen = null;
					if( bop.getOp()==OpOp2.MULT )
						gen = HopRewriteUtils.copyDataGenOp(inputGen, sval, 0);
					else { //OpOp2.PLUS 
						gen = HopRewriteUtils.copyDataGenOp(inputGen, 1, sval);
					}

					//rewire all parents (avoid anomalies with replicated datagen)
					List<Hop> parents = new ArrayList<>(bop.getParent());
					for( Hop p : parents )
						HopRewriteUtils.replaceChildReference(p, bop, gen);

					hi = gen;
					LOG.debug("Applied fuseDatagenAndBinaryOperation2 "
							+ "("+bop.getFilename()+", line "+bop.getBeginLine()+").");
				}
			}
			//left input rand and hence output matrix double, right scalar variable
			else if( HopRewriteUtils.isDataGenOp(left, OpOpDG.RAND)
					&& right.getDataType().isScalar() && left.getParent().size()==1 )
			{
				DataGenOp gen = (DataGenOp)left;
				Hop min = gen.getInput(DataExpression.RAND_MIN);
				Hop max = gen.getInput(DataExpression.RAND_MAX);
				Hop pdf = gen.getInput(DataExpression.RAND_PDF);
				boolean pdfUniform = pdf instanceof LiteralOp
						&& DataExpression.RAND_PDF_UNIFORM.equals(((LiteralOp)pdf).getStringValue());


				if( HopRewriteUtils.isBinary(bop, OpOp2.PLUS)
						&& HopRewriteUtils.isLiteralOfValue(min, 0)
						&& HopRewriteUtils.isLiteralOfValue(max, 0) )
				{
					gen.setInput(DataExpression.RAND_MIN, right, true);
					gen.setInput(DataExpression.RAND_MAX, right, true);
					//rewire all parents (avoid anomalies with replicated datagen)
					List<Hop> parents = new ArrayList<>(bop.getParent());
					for( Hop p : parents )
						HopRewriteUtils.replaceChildReference(p, bop, gen);
					hi = gen;
					LOG.debug("Applied fuseDatagenAndBinaryOperation3a "
							+ "("+bop.getFilename()+", line "+bop.getBeginLine()+").");
				}
				else if( HopRewriteUtils.isBinary(bop, OpOp2.MULT)
						&& ((HopRewriteUtils.isLiteralOfValue(min, 0) && pdfUniform)
						|| HopRewriteUtils.isLiteralOfValue(min, 1))
						&& HopRewriteUtils.isLiteralOfValue(max, 1) )
				{
					if( HopRewriteUtils.isLiteralOfValue(min, 1) )
						gen.setInput(DataExpression.RAND_MIN, right, true);
					gen.setInput(DataExpression.RAND_MAX, right, true);
					//rewire all parents (avoid anomalies with replicated datagen)
					List<Hop> parents = new ArrayList<>(bop.getParent());
					for( Hop p : parents )
						HopRewriteUtils.replaceChildReference(p, bop, gen);
					hi = gen;
					LOG.debug("Applied fuseDatagenAndBinaryOperation3b "
							+ "("+bop.getFilename()+", line "+bop.getBeginLine()+").");
				}
			}
		}

		return hi;
	}

	private static Hop fuseDatagenAndMinusOperation( Hop hi )
	{
		if( hi instanceof BinaryOp )
		{
			BinaryOp bop = (BinaryOp)hi;
			Hop left = bop.getInput(0);
			Hop right = bop.getInput(1);

			if( right instanceof DataGenOp && ((DataGenOp)right).getOp()==OpOpDG.RAND &&
					left instanceof LiteralOp && ((LiteralOp)left).getDoubleValue()==0.0 )
			{
				DataGenOp inputGen = (DataGenOp)right;
				HashMap<String,Integer> params = inputGen.getParamIndexMap();
				Hop pdf = right.getInput().get(params.get(DataExpression.RAND_PDF));
				int ixMin = params.get(DataExpression.RAND_MIN);
				int ixMax = params.get(DataExpression.RAND_MAX);
				Hop min = right.getInput().get(ixMin);
				Hop max = right.getInput().get(ixMax);

				//apply rewrite under additional conditions (for simplicity)
				if( inputGen.getParent().size()==1
						&& min instanceof LiteralOp && max instanceof LiteralOp && pdf instanceof LiteralOp
						&& DataExpression.RAND_PDF_UNIFORM.equals(((LiteralOp)pdf).getStringValue()) )
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

					//rewire all parents (avoid anomalies with replicated datagen)
					List<Hop> parents = new ArrayList<>(bop.getParent());
					for( Hop p : parents )
						HopRewriteUtils.replaceChildReference(p, bop, inputGen);

					hi = inputGen;
					LOG.debug("Applied fuseDatagenAndMinusOperation (line "+bop.getBeginLine()+").");
				}
			}
		}

		return hi;
	}

	private static Hop foldMultipleAppendOperations(Hop hi)
	{
		if( hi.getDataType().isMatrix() //no string appends or frames
				&& (HopRewriteUtils.isBinary(hi, OpOp2.CBIND, OpOp2.RBIND)
				|| HopRewriteUtils.isNary(hi, OpOpN.CBIND, OpOpN.RBIND)) )
		{
			OpOp2 bop = (hi instanceof BinaryOp) ? ((BinaryOp)hi).getOp() :
					OpOp2.valueOf(((NaryOp)hi).getOp().name());
			OpOpN nop = (hi instanceof NaryOp) ? ((NaryOp)hi).getOp() :
					OpOpN.valueOf(((BinaryOp)hi).getOp().name());

			boolean converged = false;
			while( !converged ) {
				//get first matching cbind or rbind
				Hop first = hi.getInput().stream()
						.filter(h -> HopRewriteUtils.isBinary(h, bop) || HopRewriteUtils.isNary(h, nop))
						.findFirst().orElse(null);

				//replace current op with new nary cbind/rbind
				if( first != null && first.getParent().size()==1 ) {
					//construct new list of inputs (in original order)
					ArrayList<Hop> linputs = new ArrayList<>();
					for(Hop in : hi.getInput())
						if( in == first )
							linputs.addAll(first.getInput());
						else
							linputs.add(in);
					Hop hnew = HopRewriteUtils.createNary(nop, linputs.toArray(new Hop[0]));
					//clear dangling references
					HopRewriteUtils.removeAllChildReferences(hi);
					HopRewriteUtils.removeAllChildReferences(first);
					//rewire all parents (avoid anomalies with refs to hi)
					List<Hop> parents = new ArrayList<>(hi.getParent());
					for( Hop p : parents )
						HopRewriteUtils.replaceChildReference(p, hi, hnew);
					hi = hnew;
					LOG.debug("Applied foldMultipleAppendOperations (line "+hi.getBeginLine()+").");
				}
				else {
					converged = true;
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
	 *
	 * @param parent parent high-level operator
	 * @param hi high-level operator
	 * @param pos position
	 * @return high-level operator
	 */
	private static Hop simplifyBinaryToUnaryOperation( Hop parent, Hop hi, int pos )
	{
		if( hi instanceof BinaryOp )
		{
			BinaryOp bop = (BinaryOp)hi;
			Hop left = hi.getInput(0);
			Hop right = hi.getInput(1);

			//patterns: X+X -> X*2, X*X -> X^2,
			if( left == right && left.getDataType()==DataType.MATRIX )
			{
				//note: we simplify this to unary operations first (less mem and better MR plan),
				//however, we later compile specific LOPS for X*2 and X^2
				if( bop.getOp()==OpOp2.PLUS ) //X+X -> X*2
				{
					bop.setOp(OpOp2.MULT);
					HopRewriteUtils.replaceChildReference(hi, right, new LiteralOp(2), 1);

					LOG.debug("Applied simplifyBinaryToUnaryOperation1 (line "+hi.getBeginLine()+").");
				}
				else if ( bop.getOp()==OpOp2.MULT ) //X*X -> X^2
				{
					bop.setOp(OpOp2.POW);
					HopRewriteUtils.replaceChildReference(hi, right, new LiteralOp(2), 1);

					LOG.debug("Applied simplifyBinaryToUnaryOperation2 (line "+hi.getBeginLine()+").");
				}
			}
			//patterns: (X>0)-(X<0) -> sign(X)
			else if( bop.getOp() == OpOp2.MINUS
					&& HopRewriteUtils.isBinary(left, OpOp2.GREATER)
					&& HopRewriteUtils.isBinary(right, OpOp2.LESS)
					&& left.getInput(0) == right.getInput(0)
					&& left.getInput(1) instanceof LiteralOp
					&& HopRewriteUtils.getDoubleValue((LiteralOp)left.getInput(1))==0
					&& right.getInput(1) instanceof LiteralOp
					&& HopRewriteUtils.getDoubleValue((LiteralOp)right.getInput(1))==0 )
			{
				UnaryOp uop = HopRewriteUtils.createUnary(left.getInput(0), OpOp1.SIGN);
				HopRewriteUtils.replaceChildReference(parent, hi, uop, pos);
				HopRewriteUtils.cleanupUnreferenced(hi, left, right);
				hi = uop;

				LOG.debug("Applied simplifyBinaryToUnaryOperation3 (line "+hi.getBeginLine()+").");
			}
		}

		return hi;
	}

	/**
	 * Rewrite to canonicalize all patterns like U%*%V+eps, eps+U%*%V, and
	 * U%*%V-eps into the common representation U%*%V+s which simplifies 
	 * subsequent rewrites (e.g., wdivmm or wcemm with epsilon).   
	 *
	 * @param hi high-level operator
	 * @return high-level operator
	 */
	private static Hop canonicalizeMatrixMultScalarAdd( Hop hi )
	{
		//pattern: binary operation (+ or -) of matrix mult and scalar 
		if( hi instanceof BinaryOp )
		{
			BinaryOp bop = (BinaryOp)hi;
			Hop left = hi.getInput(0);
			Hop right = hi.getInput(1);

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
				HopRewriteUtils.replaceChildReference(bop,  right,
						HopRewriteUtils.createBinaryMinus(right), 1);
				LOG.debug("Applied canonicalizeMatrixMultScalarAdd2 (line "+hi.getBeginLine()+").");
			}
		}

		return hi;
	}

	private static Hop simplifyCTableWithConstMatrixInputs( Hop hi )
	{
		//pattern: table(X, matrix(1,...), matrix(7, ...)) -> table(X, 1, 7)
		if( HopRewriteUtils.isTernary(hi, OpOp3.CTABLE) ) {
			//note: the first input always expected to be a matrix
			for( int i=1; i<hi.getInput().size(); i++ ) {
				Hop inCurr = hi.getInput().get(i);
				if( HopRewriteUtils.isDataGenOpWithConstantValue(inCurr) ) {
					Hop inNew = ((DataGenOp)inCurr).getInput(DataExpression.RAND_MIN);
					HopRewriteUtils.replaceChildReference(hi, inCurr, inNew, i);
					LOG.debug("Applied simplifyCTableWithConstMatrixInputs"
							+ i + " (line "+hi.getBeginLine()+").");
				}
			}
		}
		return hi;
	}

	private static Hop removeUnnecessaryCTable( Hop parent, Hop hi, int pos ) {
		if ( HopRewriteUtils.isAggUnaryOp(hi, AggOp.SUM, Direction.RowCol)
				&& HopRewriteUtils.isTernary(hi.getInput(0), OpOp3.CTABLE)
				&& HopRewriteUtils.isLiteralOfValue(hi.getInput(0).getInput(2), 1.0))
		{
			Hop matrixInput = hi.getInput(0).getInput(0);
			OpOp1 opcode = matrixInput.getDim2() == 1 ? OpOp1.NROW : OpOp1.LENGTH;
			Hop newOpLength = new UnaryOp("tmp", DataType.SCALAR, ValueType.INT64, opcode, matrixInput);
			HopRewriteUtils.replaceChildReference(parent, hi, newOpLength, pos);
			HopRewriteUtils.cleanupUnreferenced(hi, hi.getInput(0));
			hi = newOpLength;
		}
		return hi;
	}

	/**
	 * NOTE: this would be by definition a dynamic rewrite; however, we apply it as a static
	 * rewrite in order to apply it before splitting dags which would hide the table information
	 * if dimensions are not specified.
	 *
	 * @param parent parent high-level operator
	 * @param hi high-level operator
	 * @param pos position
	 * @return high-level operator
	 */
	private static Hop simplifyReverseOperation( Hop parent, Hop hi, int pos )
	{
		if(    hi instanceof AggBinaryOp
				&& hi.getInput(0) instanceof TernaryOp )
		{
			TernaryOp top = (TernaryOp) hi.getInput(0);

			if( top.getOp()==OpOp3.CTABLE
					&& HopRewriteUtils.isBasic1NSequence(top.getInput(0))
					&& HopRewriteUtils.isBasicN1Sequence(top.getInput(1))
					&& top.getInput(0).getDim1()==top.getInput(1).getDim1())
			{
				ReorgOp rop = HopRewriteUtils.createReorg(hi.getInput(1), ReOrgOp.REV);
				HopRewriteUtils.replaceChildReference(parent, hi, rop, pos);
				HopRewriteUtils.cleanupUnreferenced(hi, top);
				hi = rop;

				LOG.debug("Applied simplifyReverseOperation.");
			}
		}

		return hi;
	}
	
	private static Hop simplifyReverseSequence( Hop parent, Hop hi, int pos )
	{
		if( HopRewriteUtils.isReorg(hi, ReOrgOp.REV) 
			&& HopRewriteUtils.isBasic1NSequence(hi.getInput(0))
			&& hi.getInput(0).getParent().size() == 1) //only consumer
		{
			DataGenOp seq = (DataGenOp) hi.getInput(0);
			Hop from = seq.getInput().get(seq.getParamIndex(Statement.SEQ_FROM));
			Hop to = seq.getInput().get(seq.getParamIndex(Statement.SEQ_TO));
			seq.getInput().set(seq.getParamIndex(Statement.SEQ_FROM), to);
			seq.getInput().set(seq.getParamIndex(Statement.SEQ_TO), from);
			seq.getInput().set(seq.getParamIndex(Statement.SEQ_INCR), new LiteralOp(-1));
			
			HopRewriteUtils.replaceChildReference(parent, hi, seq, pos);
			HopRewriteUtils.cleanupUnreferenced(hi, seq);
			hi = seq;
			LOG.debug("Applied simplifyReverseSequence (line "+hi.getBeginLine()+").");
		}

		return hi;
	}
	
	private static Hop simplifyReverseSequenceStep(Hop parent, Hop hi, int pos) {
		if (HopRewriteUtils.isReorg(hi, ReOrgOp.REV)
				&& hi.getInput(0) instanceof DataGenOp
				&& ((DataGenOp) hi.getInput(0)).getOp() == OpOpDG.SEQ
				&& hi.getInput(0).getParent().size() == 1) // only one consumer
		{
			DataGenOp seq = (DataGenOp) hi.getInput(0);
			Hop from = seq.getInput().get(seq.getParamIndex(Statement.SEQ_FROM));
			Hop to = seq.getInput().get(seq.getParamIndex(Statement.SEQ_TO));
			Hop incr = seq.getInput().get(seq.getParamIndex(Statement.SEQ_INCR));

			if (from instanceof LiteralOp && to instanceof LiteralOp && incr instanceof LiteralOp) {
				double fromVal = ((LiteralOp) from).getDoubleValue();
				double toVal = ((LiteralOp) to).getDoubleValue();
				double incrVal = ((LiteralOp) incr).getDoubleValue();

				// Skip if increment is zero (invalid sequence)
				if (Math.abs(incrVal) < 1e-10)
					return hi;

				boolean isValidDirection = false;

				// Checking direction compatibility
				if ((incrVal > 0 && fromVal <= toVal) || (incrVal < 0 && fromVal >= toVal)) {
					isValidDirection = true;
				}

				if (isValidDirection) {
					// Calculate the number of elements and the last element
					int numValues = (int)Math.floor(Math.abs((toVal - fromVal) / incrVal)) + 1;
					double lastVal = fromVal + (numValues - 1) * incrVal;

					// Create a new sequence based on actual last value
					LiteralOp newFrom = new LiteralOp(lastVal);
					LiteralOp newTo = new LiteralOp(fromVal);
					LiteralOp newIncr = new LiteralOp(-incrVal);

					// Replace the parameters
					seq.getInput().set(seq.getParamIndex(Statement.SEQ_FROM), newFrom);
					seq.getInput().set(seq.getParamIndex(Statement.SEQ_TO), newTo);
					seq.getInput().set(seq.getParamIndex(Statement.SEQ_INCR), newIncr);

					// Replace the old sequence with the new one
					HopRewriteUtils.replaceChildReference(parent, hi, seq, pos);
					HopRewriteUtils.cleanupUnreferenced(hi, seq);
					hi = seq;
					LOG.debug("Applied simplifyReverseSequenceStep (line " + hi.getBeginLine() + ").");
				}
			}
		}
		return hi;
	}

	private static Hop simplifyMultiBinaryToBinaryOperation( Hop hi )
	{
		//pattern: 1-(X*Y) --> X 1-* Y (avoid intermediate)
		if( HopRewriteUtils.isBinary(hi, OpOp2.MINUS)
				&& hi.getDataType() == DataType.MATRIX
				&& hi.getInput(0) instanceof LiteralOp
				&& HopRewriteUtils.getDoubleValueSafe((LiteralOp)hi.getInput(0))==1
				&& HopRewriteUtils.isBinary(hi.getInput(1), OpOp2.MULT)
				&& hi.getInput(1).getParent().size() == 1 ) //single consumer
		{
			BinaryOp bop = (BinaryOp)hi;
			Hop left = hi.getInput(1).getInput(0);
			Hop right = hi.getInput(1).getInput(1);

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
	 * @param parent parent high-level operator
	 * @param hi high-level operator
	 * @param pos position
	 * @return high-level operator
	 */
	private static Hop simplifyDistributiveBinaryOperation( Hop parent, Hop hi, int pos )
	{
		if( hi instanceof BinaryOp )
		{
			BinaryOp bop = (BinaryOp)hi;
			Hop left = bop.getInput(0);
			Hop right = bop.getInput(1);

			//(X+Y*X) -> (1+Y)*X,    (Y*X+X) -> (Y+1)*X
			//(X-Y*X) -> (1-Y)*X,    (Y*X-X) -> (Y-1)*X
			boolean applied = false;
			if( left.getDataType()==DataType.MATRIX && right.getDataType()==DataType.MATRIX
					&& HopRewriteUtils.isValidOp(bop.getOp(), LOOKUP_VALID_DISTRIBUTIVE_BINARY) )
			{
				Hop X = null; Hop Y = null;
				if( HopRewriteUtils.isBinary(left, OpOp2.MULT) ) //(Y*X-X) -> (Y-1)*X
				{
					Hop leftC1 = left.getInput(0);
					Hop leftC2 = left.getInput(1);

					if( leftC1.getDataType()==DataType.MATRIX && leftC2.getDataType()==DataType.MATRIX &&
							(right == leftC1 || right == leftC2) && leftC1 !=leftC2 ){ //any mult order
						X = right;
						Y = ( right == leftC1 ) ? leftC2 : leftC1;
					}
					if( X != null && Y.dimsKnown() ){ //rewrite 'binary +/-' 
						LiteralOp literal = new LiteralOp(1);
						BinaryOp plus = HopRewriteUtils.createBinary(Y, literal, bop.getOp());
						
						BinaryOp mult = (plus.getDim1()==1 || plus.getDim2() == 1)
								&& (X.getDim1()>1 && X.getDim2()>1) ?
							HopRewriteUtils.createBinary(X, plus, OpOp2.MULT) :
							HopRewriteUtils.createBinary(plus, X, OpOp2.MULT);
						HopRewriteUtils.replaceChildReference(parent, hi, mult, pos);
						HopRewriteUtils.cleanupUnreferenced(hi, left);
						hi = mult;
						applied = true;

						LOG.debug("Applied simplifyDistributiveBinaryOperation1 (line "+hi.getBeginLine()+").");
					}
				}

				if( !applied && HopRewriteUtils.isBinary(right, OpOp2.MULT) ) //(X-Y*X) -> (1-Y)*X
				{
					Hop rightC1 = right.getInput(0);
					Hop rightC2 = right.getInput(1);
					if( rightC1.getDataType()==DataType.MATRIX && rightC2.getDataType()==DataType.MATRIX &&
							(left == rightC1 || left == rightC2) && rightC1 !=rightC2 ){ //any mult order
						X = left;
						Y = ( left == rightC1 ) ? rightC2 : rightC1;
					}
					if( X != null && Y.dimsKnown() ){ //rewrite '+/- binary'
						LiteralOp literal = new LiteralOp(1);
						BinaryOp plus = HopRewriteUtils.createBinary(literal, Y, bop.getOp());
						BinaryOp mult = (plus.getDim1()==1 || plus.getDim2() == 1) 
								&& (X.getDim1()>1 && X.getDim2()>1) ?
							HopRewriteUtils.createBinary(X, plus, OpOp2.MULT) :
							HopRewriteUtils.createBinary(plus, X, OpOp2.MULT);
						HopRewriteUtils.replaceChildReference(parent, hi, mult, pos);
						HopRewriteUtils.cleanupUnreferenced(hi, right);
						hi = mult;

						LOG.debug("Applied simplifyDistributiveBinaryOperation2 (line "+hi.getBeginLine()+").");
					}
				}
			}
		}

		return hi;
	}

	/**
	 * det(t(X)) -> det(X)
	 *
	 * @param parent parent high-level operator
	 * @param hi high-level operator
	 * @param pos position
	 * @return high-level operator
	 */
	private static Hop simplifyTransposeInDetOperation(Hop parent, Hop hi, int pos)
	{
		if(HopRewriteUtils.isUnary(hi, OpOp1.DET)
				&& HopRewriteUtils.isReorg(hi.getInput(0), ReOrgOp.TRANS))
		{
			Hop operand = hi.getInput(0).getInput(0);
			Hop uop = HopRewriteUtils.createUnary(operand, OpOp1.DET);
			HopRewriteUtils.replaceChildReference(parent, hi, uop, pos);

			LOG.debug("Applied simplifyTransposeInDetOperation.");
			return uop;
		}
		return hi;
	}

	/**
	 * t(Z)%*%(X*(Y*(Z%*%v))) -> t(Z)%*%(X*Y)*(Z%*%v)
	 * t(Z)%*%(X+(Y+(Z%*%v))) -> t(Z)%*%((X+Y)+(Z%*%v))
	 *
	 * Note: Restriction ba() at leaf and root instead of data at leaf to not reorganize too
	 * eagerly, which would loose additional rewrite potential. This rewrite has two goals
	 * (1) enable XtwXv, and increase piggybacking potential by creating bushy trees.
	 *
	 * @param parent parent high-level operator
	 * @param hi high-level operator
	 * @param pos position
	 * @return high-level operator
	 */
	private static Hop simplifyBushyBinaryOperation( Hop parent, Hop hi, int pos )
	{
		if( hi instanceof BinaryOp && parent instanceof AggBinaryOp )
		{
			BinaryOp bop = (BinaryOp)hi;
			Hop left = bop.getInput(0);
			Hop right = bop.getInput(1);
			OpOp2 op = bop.getOp();

			if( left.getDataType()==DataType.MATRIX && right.getDataType()==DataType.MATRIX &&
					HopRewriteUtils.isValidOp(op, LOOKUP_VALID_ASSOCIATIVE_BINARY) )
			{
				boolean applied = false;

				if( right instanceof BinaryOp )
				{
					BinaryOp bop2 = (BinaryOp)right;
					Hop left2 = bop2.getInput(0);
					Hop right2 = bop2.getInput(1);
					OpOp2 op2 = bop2.getOp();

					if( op==op2 && right2.getDataType()==DataType.MATRIX
							&& (right2 instanceof AggBinaryOp) )
					{
						//(X*(Y*op()) -> (X*Y)*op()
						BinaryOp bop3 = HopRewriteUtils.createBinary(left, left2, op);
						BinaryOp bop4 = HopRewriteUtils.createBinary(bop3, right2, op);
						HopRewriteUtils.replaceChildReference(parent, bop, bop4, pos);
						HopRewriteUtils.cleanupUnreferenced(bop, bop2);
						hi = bop4;

						applied = true;

						LOG.debug("Applied simplifyBushyBinaryOperation1");
					}
				}

				if( !applied && left instanceof BinaryOp )
				{
					BinaryOp bop2 = (BinaryOp)left;
					Hop left2 = bop2.getInput(0);
					Hop right2 = bop2.getInput(1);
					OpOp2 op2 = bop2.getOp();

					if( op==op2 && left2.getDataType()==DataType.MATRIX
							&& (left2 instanceof AggBinaryOp)
							&& (right2.getDim2() > 1 || right.getDim2() == 1)   //X not vector, or Y vector
							&& (right2.getDim1() > 1 || right.getDim1() == 1) ) //X not vector, or Y vector
					{
						//((op()*X)*Y) -> op()*(X*Y)
						BinaryOp bop3 = HopRewriteUtils.createBinary(right2, right, op);
						BinaryOp bop4 = HopRewriteUtils.createBinary(left2, bop3, op);
						HopRewriteUtils.replaceChildReference(parent, bop, bop4, pos);
						HopRewriteUtils.cleanupUnreferenced(bop, bop2);
						hi = bop4;

						LOG.debug("Applied simplifyBushyBinaryOperation2");
					}
				}
			}

		}

		return hi;
	}

	private static Hop simplifyUnaryAggReorgOperation( Hop parent, Hop hi, int pos )
	{
		if( hi instanceof AggUnaryOp && ((AggUnaryOp)hi).getDirection()==Direction.RowCol 
			&& ((AggUnaryOp)hi).getOp() != AggOp.TRACE    //full uagg
			&& hi.getInput(0) instanceof ReorgOp  ) //reorg operation
		{
			ReorgOp rop = (ReorgOp)hi.getInput(0);
			if( rop.getOp().preservesValues()       //valid reorg
				&& rop.getParent().size()==1 )      //uagg only reorg consumer
			{
				Hop input = rop.getInput(0);
				HopRewriteUtils.removeAllChildReferences(hi);
				HopRewriteUtils.removeAllChildReferences(rop);
				HopRewriteUtils.addChildReference(hi, input);
				LOG.debug("Applied simplifyUnaryAggReorgOperation");
			}
		}
		return hi;
	}

	private static Hop removeUnnecessaryAggregates(Hop hi)
	{
		//sum(rowSums(X)) -> sum(X), sum(colSums(X)) -> sum(X)
		//min(rowMins(X)) -> min(X), min(colMins(X)) -> min(X)
		//max(rowMaxs(X)) -> max(X), max(colMaxs(X)) -> max(X)
		//sum(rowSums(X^2)) -> sum(X), sum(colSums(X^2)) -> sum(X)
		if( hi instanceof AggUnaryOp && hi.getInput(0) instanceof AggUnaryOp
				&& ((AggUnaryOp)hi).getDirection()==Direction.RowCol
				&& hi.getInput(0).getParent().size()==1 )
		{
			AggUnaryOp au1 = (AggUnaryOp) hi;
			AggUnaryOp au2 = (AggUnaryOp) hi.getInput(0);
			if( (au1.getOp()==AggOp.SUM && (au2.getOp()==AggOp.SUM || au2.getOp()==AggOp.SUM_SQ))
					|| (au1.getOp()==AggOp.MIN && au2.getOp()==AggOp.MIN)
					|| (au1.getOp()==AggOp.MAX && au2.getOp()==AggOp.MAX) )
			{
				Hop input = au2.getInput(0);
				HopRewriteUtils.removeAllChildReferences(au2);
				HopRewriteUtils.replaceChildReference(au1, au2, input);
				if( au2.getOp() == AggOp.SUM_SQ )
					au1.setOp(AggOp.SUM_SQ);

				LOG.debug("Applied removeUnnecessaryAggregates (line "+hi.getBeginLine()+").");
			}
		}

		return hi;
	}

	private static Hop simplifyBinaryMatrixScalarOperation( Hop parent, Hop hi, int pos )
	{
		// Note: This rewrite is not applicable for all binary operations because some of them 
		// are undefined over scalars. We explicitly exclude potential conflicting matrix-scalar binary
		// operations; other operations like cbind/rbind will never occur as matrix-scalar operations.

		if( HopRewriteUtils.isUnary(hi, OpOp1.CAST_AS_SCALAR)
				&& hi.getInput(0) instanceof BinaryOp
				&& HopRewriteUtils.isBinary(hi.getInput(0), LOOKUP_VALID_SCALAR_BINARY))
		{
			BinaryOp bin = (BinaryOp) hi.getInput(0);
			BinaryOp bout = null;

			//as.scalar(X*Y) -> as.scalar(X) * as.scalar(Y)
			if( bin.getInput(0).getDataType()==DataType.MATRIX
					&& bin.getInput(1).getDataType()==DataType.MATRIX ) {
				UnaryOp cast1 = HopRewriteUtils.createUnary(bin.getInput(0), OpOp1.CAST_AS_SCALAR);
				UnaryOp cast2 = HopRewriteUtils.createUnary(bin.getInput(1), OpOp1.CAST_AS_SCALAR);
				bout = HopRewriteUtils.createBinary(cast1, cast2, bin.getOp());
			}
			//as.scalar(X*s) -> as.scalar(X) * s
			else if( bin.getInput(0).getDataType()==DataType.MATRIX ) {
				UnaryOp cast = HopRewriteUtils.createUnary(bin.getInput(0), OpOp1.CAST_AS_SCALAR);
				bout = HopRewriteUtils.createBinary(cast, bin.getInput(1), bin.getOp());
			}
			//as.scalar(s*X) -> s * as.scalar(X)
			else if ( bin.getInput(1).getDataType()==DataType.MATRIX ) {
				UnaryOp cast = HopRewriteUtils.createUnary(bin.getInput(1), OpOp1.CAST_AS_SCALAR);
				bout = HopRewriteUtils.createBinary(bin.getInput(0), cast, bin.getOp());
			}

			if( bout != null ) {
				HopRewriteUtils.replaceChildReference(parent, hi, bout, pos);

				LOG.debug("Applied simplifyBinaryMatrixScalarOperation.");
			}
		}

		return hi;
	}

	private static Hop pushdownUnaryAggTransposeOperation( Hop parent, Hop hi, int pos )
	{
		if( hi instanceof AggUnaryOp && hi.getParent().size()==1
				&& (((AggUnaryOp) hi).getDirection()==Direction.Row || ((AggUnaryOp) hi).getDirection()==Direction.Col)
				&& HopRewriteUtils.isTransposeOperation(hi.getInput(0), 1)
				&& HopRewriteUtils.isValidOp(((AggUnaryOp) hi).getOp(), LOOKUP_VALID_ROW_COL_AGGREGATE) )
		{
			AggUnaryOp uagg = (AggUnaryOp) hi;

			//get input rewire existing operators (remove inner transpose)
			Hop input = uagg.getInput(0).getInput(0);
			HopRewriteUtils.removeAllChildReferences(hi.getInput(0));
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

	private static Hop pushdownCSETransposeScalarOperation( Hop parent, Hop hi, int pos )
	{
		// a=t(X), b=t(X^2) -> a=t(X), b=t(X)^2 for CSE t(X)
		// probed at root node of b in above example
		// (with support for left or right scalar operations)
		if( HopRewriteUtils.isTransposeOperation(hi, 1)
				&& HopRewriteUtils.isBinaryMatrixScalarOperation(hi.getInput(0))
				&& hi.getInput(0).getParent().size()==1)
		{
			int Xpos = hi.getInput(0).getInput(0).getDataType().isMatrix() ? 0 : 1;
			Hop X = hi.getInput(0).getInput().get(Xpos);
			BinaryOp binary = (BinaryOp) hi.getInput(0);

			if( HopRewriteUtils.containsTransposeOperation(X.getParent())
					&& !HopRewriteUtils.isValidOp(binary.getOp(), new OpOp2[]{OpOp2.MOMENT, OpOp2.QUANTILE}))
			{
				//clear existing wiring
				HopRewriteUtils.removeChildReferenceByPos(parent, hi, pos);
				HopRewriteUtils.removeChildReference(hi, binary);
				HopRewriteUtils.removeChildReference(binary, X);

				//re-wire operators
				HopRewriteUtils.addChildReference(parent, binary, pos);
				HopRewriteUtils.addChildReference(binary, hi, Xpos);
				HopRewriteUtils.addChildReference(hi, X);
				//note: common subexpression later eliminated by dedicated rewrite

				hi = binary;
				LOG.debug("Applied pushdownCSETransposeScalarOperation (line "+hi.getBeginLine()+").");
			}
		}

		return hi;
	}

	/**
	 * det(X%*%Y) -> det(X)*det(Y)
	 *
	 * @param parent parent high-level operator
	 * @param hi high-level operator
	 * @param pos position
	 * @return high-level operator
	 */
	private static Hop pushdownDetMultOperation(Hop parent, Hop hi, int pos) {
		if( HopRewriteUtils.isUnary(hi, OpOp1.DET)
				&& HopRewriteUtils.isMatrixMultiply(hi.getInput(0))
				&& hi.getInput(0).getInput(0).isMatrix()
				&& hi.getInput(0).getInput(1).isMatrix())
		{
			Hop operand1 = hi.getInput(0).getInput(0);
			Hop operand2 = hi.getInput(0).getInput(1);
			Hop uop1 = HopRewriteUtils.createUnary(operand1, OpOp1.DET);
			Hop uop2 = HopRewriteUtils.createUnary(operand2, OpOp1.DET);
			Hop bop = HopRewriteUtils.createBinary(uop1, uop2, OpOp2.MULT);
			HopRewriteUtils.replaceChildReference(parent, hi, bop, pos);

			LOG.debug("Applied pushdownDetMultOperation.");
			return bop;
		}
		return hi;
	}

	/**
	 * det(lambda*X) -> lambda^nrow*det(X)
	 *
	 * @param parent parent high-level operator
	 * @param hi high-level operator
	 * @param pos position
	 * @return high-level operator
	 */
	private static Hop pushdownDetScalarMatrixMultOperation(Hop parent, Hop hi, int pos) {
		if( HopRewriteUtils.isUnary(hi, OpOp1.DET)
				&& HopRewriteUtils.isBinary(hi.getInput(0), OpOp2.MULT)
				&& ((hi.getInput(0).getInput(0).isMatrix() && hi.getInput(0).getInput(1).isScalar())
					|| (hi.getInput(0).getInput(0).isScalar() && hi.getInput(0).getInput(1).isMatrix())))
		{
			Hop operand1 = hi.getInput(0).getInput(0);
			Hop operand2 = hi.getInput(0).getInput(1);

			Hop lambda = (operand1.isScalar()) ? operand1 : operand2;
			Hop matrix = (operand1.isMatrix()) ? operand1 : operand2;

			Hop uopDet = HopRewriteUtils.createUnary(matrix, OpOp1.DET);
			Hop uopNrow = HopRewriteUtils.createUnary(matrix, OpOp1.NROW);
			Hop bopPow = HopRewriteUtils.createBinary(lambda, uopNrow, OpOp2.POW);
			Hop bopMult = HopRewriteUtils.createBinary(bopPow, uopDet, OpOp2.MULT);
			HopRewriteUtils.replaceChildReference(parent, hi, bopMult, pos);

			LOG.debug("Applied pushdownDetScalarMatrixMultOperation.");
			return bopMult;
		}
		return hi;
	}

	private static Hop pushdownSumBinaryMult(Hop parent, Hop hi, int pos ) {
		//pattern:  sum(lamda*X) -> lamda*sum(X)
		if( hi instanceof AggUnaryOp && ((AggUnaryOp)hi).getDirection()==Direction.RowCol
				&& ((AggUnaryOp)hi).getOp()==AggOp.SUM // only one parent which is the sum
				&& HopRewriteUtils.isBinary(hi.getInput(0), OpOp2.MULT, 1)
				&& ((hi.getInput(0).getInput(0).getDataType()==DataType.SCALAR && hi.getInput(0).getInput(1).getDataType()==DataType.MATRIX)
				||(hi.getInput(0).getInput(0).getDataType()==DataType.MATRIX && hi.getInput(0).getInput(1).getDataType()==DataType.SCALAR)))
		{
			Hop operand1 = hi.getInput(0).getInput(0);
			Hop operand2 = hi.getInput(0).getInput(1);

			//check which operand is the Scalar and which is the matrix
			Hop lamda = (operand1.getDataType()==DataType.SCALAR) ? operand1 : operand2;
			Hop matrix = (operand1.getDataType()==DataType.MATRIX) ? operand1 : operand2;

			AggUnaryOp aggOp=HopRewriteUtils.createAggUnaryOp(matrix, AggOp.SUM, Direction.RowCol);
			Hop bop = HopRewriteUtils.createBinary(lamda, aggOp, OpOp2.MULT);

			HopRewriteUtils.replaceChildReference(parent, hi, bop, pos);

			LOG.debug("Applied pushdownSumBinaryMult (line "+hi.getBeginLine()+").");
			return bop;
		}
		return hi;
	}

	private static Hop pullupAbs(Hop parent, Hop hi, int pos ) {
		if( HopRewriteUtils.isBinary(hi, OpOp2.MULT)
				&& HopRewriteUtils.isUnary(hi.getInput(0), OpOp1.ABS)
				&& hi.getInput(0).getParent().size()==1
				&& HopRewriteUtils.isUnary(hi.getInput(1), OpOp1.ABS)
				&& hi.getInput(1).getParent().size()==1)
		{
			Hop operand1 = hi.getInput(0).getInput(0);
			Hop operand2 = hi.getInput(1).getInput(0);
			Hop bop = HopRewriteUtils.createBinary(operand1, operand2, OpOp2.MULT);
			Hop uop = HopRewriteUtils.createUnary(bop, OpOp1.ABS);
			HopRewriteUtils.replaceChildReference(parent, hi, uop, pos);

			LOG.debug("Applied pullupAbs (line "+hi.getBeginLine()+").");
			return uop;
		}
		return hi;
	}

	private static Hop simplifyUnaryPPredOperation( Hop parent, Hop hi, int pos )
	{
		if( hi instanceof UnaryOp && hi.getDataType()==DataType.MATRIX  //unaryop
				&& hi.getInput(0) instanceof BinaryOp                 //binaryop - ppred
				&& ((BinaryOp)hi.getInput(0)).isPPredOperation() )
		{
			UnaryOp uop = (UnaryOp) hi; //valid unary op
			if( uop.getOp()==OpOp1.ABS || uop.getOp()==OpOp1.SIGN
					|| uop.getOp()==OpOp1.CEIL || uop.getOp()==OpOp1.FLOOR || uop.getOp()==OpOp1.ROUND )
			{
				//clear link unary-binary
				Hop input = uop.getInput(0);
				HopRewriteUtils.replaceChildReference(parent, hi, input, pos);
				HopRewriteUtils.cleanupUnreferenced(hi);
				hi = input;

				LOG.debug("Applied simplifyUnaryPPredOperation.");
			}
		}

		return hi;
	}

	private static Hop simplifyTransposedAppend( Hop parent, Hop hi, int pos )
	{
		//e.g., t(cbind(t(A),t(B))) --> rbind(A,B), t(rbind(t(A),t(B))) --> cbind(A,B)
		if(   HopRewriteUtils.isTransposeOperation(hi)  //t() rooted
				&& hi.getInput(0) instanceof BinaryOp
				&& (((BinaryOp)hi.getInput(0)).getOp()==OpOp2.CBIND    //append (cbind/rbind)
				|| ((BinaryOp)hi.getInput(0)).getOp()==OpOp2.RBIND)
				&& hi.getInput(0).getParent().size() == 1 ) //single consumer of append
		{
			BinaryOp bop = (BinaryOp)hi.getInput(0);
			//both inputs transpose ops, where transpose is single consumer
			if( HopRewriteUtils.isTransposeOperation(bop.getInput(0), 1)
					&& HopRewriteUtils.isTransposeOperation(bop.getInput(1), 1) )
			{
				Hop left = bop.getInput(0).getInput(0);
				Hop right = bop.getInput(1).getInput(0);

				//create new subdag (no in-place dag update to prevent anomalies with
				//multiple consumers during rewrite process)
				OpOp2 binop = (bop.getOp()==OpOp2.CBIND) ? OpOp2.RBIND : OpOp2.CBIND;
				BinaryOp bopnew = HopRewriteUtils.createBinary(left, right, binop);
				HopRewriteUtils.replaceChildReference(parent, hi, bopnew, pos);

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
	 * @param parent parent high-level operator
	 * @param hi high-level operator
	 * @param pos position
	 */
	private static Hop fuseBinarySubDAGToUnaryOperation( Hop parent, Hop hi, int pos )
	{
		if( hi instanceof BinaryOp )
		{
			BinaryOp bop = (BinaryOp)hi;
			Hop left = hi.getInput(0);
			Hop right = hi.getInput(1);
			boolean applied = false;

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
					Hop left1 = bleft.getInput(0);
					Hop left2 = bleft.getInput(1);

					if( left1 instanceof LiteralOp &&
							HopRewriteUtils.getDoubleValue((LiteralOp)left1)==1 &&
							left2 == right && bleft.getOp() == OpOp2.MINUS  )
					{
						UnaryOp unary = HopRewriteUtils.createUnary(right, OpOp1.SPROP);
						HopRewriteUtils.replaceChildReference(parent, bop, unary, pos);
						HopRewriteUtils.cleanupUnreferenced(bop, left);
						hi = unary;
						applied = true;

						LOG.debug("Applied fuseBinarySubDAGToUnaryOperation-sprop1");
					}
				}
				if( !applied && right instanceof BinaryOp ) //X*(1-X)
				{
					BinaryOp bright = (BinaryOp)right;
					Hop right1 = bright.getInput(0);
					Hop right2 = bright.getInput(1);

					if( right1 instanceof LiteralOp &&
							HopRewriteUtils.getDoubleValue((LiteralOp)right1)==1 &&
							right2 == left && bright.getOp() == OpOp2.MINUS )
					{
						UnaryOp unary = HopRewriteUtils.createUnary(left, OpOp1.SPROP);
						HopRewriteUtils.replaceChildReference(parent, bop, unary, pos);
						HopRewriteUtils.cleanupUnreferenced(bop, left);
						hi = unary;
						applied = true;

						LOG.debug("Applied fuseBinarySubDAGToUnaryOperation-sprop2");
					}
				}
			}

			//sigmoid operator
			if( !applied && bop.getOp() == OpOp2.DIV && left.getDataType()==DataType.SCALAR && right.getDataType()==DataType.MATRIX
					&& left instanceof LiteralOp && HopRewriteUtils.getDoubleValue((LiteralOp)left)==1 && right instanceof BinaryOp)
			{
				//note: if there are multiple consumers on the intermediate,
				//we follow the heuristic that redundant computation is more beneficial, 
				//i.e., we still fuse but leave the intermediate for the other consumers  

				BinaryOp bop2 = (BinaryOp)right;
				Hop left2 = bop2.getInput(0);
				Hop right2 = bop2.getInput(1);

				if(    bop2.getOp() == OpOp2.PLUS && left2.getDataType()==DataType.SCALAR && right2.getDataType()==DataType.MATRIX
						&& left2 instanceof LiteralOp && HopRewriteUtils.getDoubleValue((LiteralOp)left2)==1 && right2 instanceof UnaryOp)
				{
					UnaryOp uop = (UnaryOp) right2;
					Hop uopin = uop.getInput(0);

					if( uop.getOp()==OpOp1.EXP )
					{
						UnaryOp unary = null;

						//Pattern 1: (1/(1 + exp(-X)) 
						if( HopRewriteUtils.isBinary(uopin, OpOp2.MINUS) ) {
							BinaryOp bop3 = (BinaryOp) uopin;
							Hop left3 = bop3.getInput(0);
							Hop right3 = bop3.getInput(1);

							if( left3 instanceof LiteralOp && HopRewriteUtils.getDoubleValue((LiteralOp)left3)==0 )
								unary = HopRewriteUtils.createUnary(right3, OpOp1.SIGMOID);
						}
						//Pattern 2: (1/(1 + exp(X)), e.g., where -(-X) has been removed by 
						//the 'remove unnecessary minus' rewrite --> reintroduce the minus
						else {
							BinaryOp minus = HopRewriteUtils.createBinaryMinus(uopin);
							unary = HopRewriteUtils.createUnary(minus, OpOp1.SIGMOID);
						}

						if( unary != null ) {
							HopRewriteUtils.replaceChildReference(parent, bop, unary, pos);
							HopRewriteUtils.cleanupUnreferenced(bop, bop2, uop);
							hi = unary;
							applied = true;

							LOG.debug("Applied fuseBinarySubDAGToUnaryOperation-sigmoid1");
						}
					}
				}
			}

			//select positive (selp) operator (note: same initial pattern as sprop)
			if( !applied && bop.getOp() == OpOp2.MULT && left.getDataType()==DataType.MATRIX && right.getDataType()==DataType.MATRIX )
			{
				//by definition, either left or right or none applies. 
				//note: if there are multiple consumers on the intermediate tmp=(X>0), it's still beneficial
				//to replace the X*tmp with selp(X) due to lower memory requirements and simply sparsity propagation 
				if( left instanceof BinaryOp ) //(X>0)*X
				{
					BinaryOp bleft = (BinaryOp)left;
					Hop left1 = bleft.getInput(0);
					Hop left2 = bleft.getInput(1);

					if( left2 instanceof LiteralOp &&
							HopRewriteUtils.getDoubleValue((LiteralOp)left2)==0 &&
							left1 == right && (bleft.getOp() == OpOp2.GREATER ) )
					{
						BinaryOp binary = HopRewriteUtils.createBinary(right, new LiteralOp(0), OpOp2.MAX);
						HopRewriteUtils.replaceChildReference(parent, bop, binary, pos);
						HopRewriteUtils.cleanupUnreferenced(bop, left);
						hi = binary;
						applied = true;

						LOG.debug("Applied fuseBinarySubDAGToUnaryOperation-max0a");
					}
				}
				if( !applied && right instanceof BinaryOp ) //X*(X>0)
				{
					BinaryOp bright = (BinaryOp)right;
					Hop right1 = bright.getInput(0);
					Hop right2 = bright.getInput(1);

					if( right2 instanceof LiteralOp &&
							HopRewriteUtils.getDoubleValue((LiteralOp)right2)==0 &&
							right1 == left && bright.getOp() == OpOp2.GREATER )
					{
						BinaryOp binary = HopRewriteUtils.createBinary(left, new LiteralOp(0), OpOp2.MAX);
						HopRewriteUtils.replaceChildReference(parent, bop, binary, pos);
						HopRewriteUtils.cleanupUnreferenced(bop, left);
						hi = binary;
						applied= true;

						LOG.debug("Applied fuseBinarySubDAGToUnaryOperation-max0b");
					}
				}
			}
		}

		return hi;
	}

	private static Hop simplifyTraceMatrixMult(Hop parent, Hop hi, int pos)
	{
		if( hi instanceof AggUnaryOp && ((AggUnaryOp)hi).getOp()==AggOp.TRACE ) //trace()
		{
			Hop hi2 = hi.getInput(0);
			if( HopRewriteUtils.isMatrixMultiply(hi2) ) //X%*%Y
			{
				Hop left = hi2.getInput(0);
				Hop right = hi2.getInput(1);

				//create new operators (incl refresh size inside for transpose)
				ReorgOp trans = HopRewriteUtils.createTranspose(right);
				BinaryOp mult = HopRewriteUtils.createBinary(left, trans, OpOp2.MULT);
				AggUnaryOp sum = HopRewriteUtils.createSum(mult);

				//rehang new subdag under parent node
				HopRewriteUtils.replaceChildReference(parent, hi, sum, pos);
				HopRewriteUtils.cleanupUnreferenced(hi, hi2);
				hi = sum;

				LOG.debug("Applied simplifyTraceMatrixMult");
			}
		}

		return hi;
	}

	private static Hop simplifyTraceSum(Hop parent, Hop hi, int pos) {
		if (hi instanceof AggUnaryOp && ((AggUnaryOp) hi).getOp() == AggOp.TRACE) {
			Hop hi2 = hi.getInput().get(0);
			if (HopRewriteUtils.isBinary(hi2, OpOp2.PLUS) && hi2.getParent().size() == 1) {
				Hop left = hi2.getInput().get(0);
				Hop right = hi2.getInput().get(1);

				// Create trace nodes
				AggUnaryOp traceLeft = HopRewriteUtils.createAggUnaryOp(left, AggOp.TRACE, Direction.RowCol);
				AggUnaryOp traceRight = HopRewriteUtils.createAggUnaryOp(right, AggOp.TRACE, Direction.RowCol);

				// Add them
				BinaryOp sum = HopRewriteUtils.createBinary(traceLeft, traceRight, OpOp2.PLUS);

				// Replace in DAG
				HopRewriteUtils.replaceChildReference(parent, hi, sum, pos);
				HopRewriteUtils.cleanupUnreferenced(hi, hi2);

				LOG.debug("Applied simplifyTraceSum rewrite");
				return sum;
			}
		}
		return hi;
	}

	private static Hop simplifyTraceTranspose(Hop parent, Hop hi, int pos) {
		// Check if the current Hop is a trace operation
		if ( HopRewriteUtils.isAggUnaryOp(hi, AggOp.TRACE) ) {
			Hop input = hi.getInput().get(0);

			// Check if input is a transpose and it is only consumer
			if (HopRewriteUtils.isReorg(input, ReOrgOp.TRANS) && input.getParent().size() == 1) {
				HopRewriteUtils.replaceChildReference(hi, input, input.getInput(0));
				LOG.debug("Applied simplifyTraceTranspose rewrite");
			}
		}
		return hi;
	}

	private static Hop simplifySlicedMatrixMult(Hop parent, Hop hi, int pos)
	{
		//e.g., (X%*%Y)[1,1] -> X[1,] %*% Y[,1] 
		if( hi instanceof IndexingOp
				&& ((IndexingOp)hi).isRowLowerEqualsUpper()
				&& ((IndexingOp)hi).isColLowerEqualsUpper()
				&& hi.getInput(0).getParent().size()==1 //rix is single mm consumer
				&& HopRewriteUtils.isMatrixMultiply(hi.getInput(0)) )
		{
			Hop mm = hi.getInput(0);
			Hop X = mm.getInput(0);
			Hop Y = mm.getInput(1);
			Hop rowExpr = hi.getInput(1); //rl==ru
			Hop colExpr = hi.getInput(3); //cl==cu

			HopRewriteUtils.removeAllChildReferences(mm);

			//create new indexing operations
			IndexingOp ix1 = new IndexingOp("tmp1", DataType.MATRIX, ValueType.FP64, X,
					rowExpr, rowExpr, new LiteralOp(1), HopRewriteUtils.createValueHop(X, false), true, false);
			ix1.setBlocksize(X.getBlocksize());
			ix1.refreshSizeInformation();
			IndexingOp ix2 = new IndexingOp("tmp2", DataType.MATRIX, ValueType.FP64, Y,
					new LiteralOp(1), HopRewriteUtils.createValueHop(Y, true), colExpr, colExpr, false, true);
			ix2.setBlocksize(Y.getBlocksize());
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

	private static Hop simplifyListIndexing(Hop hi) {
		//e.g., L[i:i, 1:ncol(L)] -> L[i:i, 1:1]
		if( hi instanceof IndexingOp && hi.getDataType().isList()
				&& !(hi.getInput(4) instanceof LiteralOp) )
		{
			HopRewriteUtils.replaceChildReference(hi, hi.getInput(4), new LiteralOp(1));
			LOG.debug("Applied simplifyListIndexing (line "+hi.getBeginLine()+").");
		}
		return hi;
	}

	private static Hop simplifyScalarIndexing(Hop parent, Hop hi, int pos)
	{
		//as.scalar(X[i,1]) -> X[i,1] w/ scalar output
		if( HopRewriteUtils.isUnary(hi, OpOp1.CAST_AS_SCALAR) 
			&& hi.getInput(0).getParent().size() == 1 // only consumer
			&& hi.getParent().size() == 1 //avoid temp inconsistency
			&& hi.getInput(0) instanceof IndexingOp 
			&& ((IndexingOp)hi.getInput(0)).isScalarOutput() 
			&& hi.getInput(0).isMatrix() //no frame support yet 
			&& !HopRewriteUtils.isData(parent, OpOpData.TRANSIENTWRITE)) 
		{
			Hop hi2 = hi.getInput(0);
			hi2.setDataType(DataType.SCALAR); 
			hi2.setDim1(0); hi2.setDim2(0);
			HopRewriteUtils.replaceChildReference(parent, hi, hi2, pos);
			HopRewriteUtils.cleanupUnreferenced(hi);
			hi = hi2;
			LOG.debug("Applied simplifyScalarIndexing (line "+hi.getBeginLine()+").");
		}
		return hi;
	}
	
	private static Hop simplifyConstantSort(Hop parent, Hop hi, int pos)
	{
		//order(matrix(7), indexreturn=FALSE) -> matrix(7)
		//order(matrix(7), indexreturn=TRUE) -> seq(1,nrow(X),1)
		if( hi instanceof ReorgOp && ((ReorgOp)hi).getOp()==ReOrgOp.SORT )  //order
		{
			Hop hi2 = hi.getInput(0);

			if( hi2 instanceof DataGenOp && ((DataGenOp)hi2).getOp()==OpOpDG.RAND
					&& ((DataGenOp)hi2).hasConstantValue()
					&& hi.getInput(3) instanceof LiteralOp ) //known indexreturn
			{
				if( HopRewriteUtils.getBooleanValue((LiteralOp)hi.getInput(3)) )
				{
					//order(matrix(7), indexreturn=TRUE) -> seq(1,nrow(X),1)
					Hop seq = HopRewriteUtils.createSeqDataGenOp(hi2);
					seq.refreshSizeInformation();
					HopRewriteUtils.replaceChildReference(parent, hi, seq, pos);
					HopRewriteUtils.cleanupUnreferenced(hi);
					hi = seq;

					LOG.debug("Applied simplifyConstantSort1.");
				}
				else
				{
					//order(matrix(7), indexreturn=FALSE) -> matrix(7)
					HopRewriteUtils.replaceChildReference(parent, hi, hi2, pos);
					HopRewriteUtils.cleanupUnreferenced(hi);
					hi = hi2;

					LOG.debug("Applied simplifyConstantSort2.");
				}
			}
		}

		return hi;
	}

	private static Hop simplifyOrderedSort(Hop parent, Hop hi, int pos)
	{
		//order(seq(2,N+1,1), indexreturn=FALSE) -> matrix(7)
		//order(seq(2,N+1,1), indexreturn=TRUE) -> seq(1,N,1)/seq(N,1,-1)
		if( hi instanceof ReorgOp && ((ReorgOp)hi).getOp()==ReOrgOp.SORT )  //order
		{
			Hop hi2 = hi.getInput(0);

			if( hi2 instanceof DataGenOp && ((DataGenOp)hi2).getOp()==OpOpDG.SEQ )
			{
				Hop incr = hi2.getInput().get(((DataGenOp)hi2).getParamIndex(Statement.SEQ_INCR));
				//check for known ascending ordering and known indexreturn
				if( incr instanceof LiteralOp && HopRewriteUtils.getDoubleValue((LiteralOp)incr)==1
						&& hi.getInput(2) instanceof LiteralOp      //decreasing
						&& hi.getInput(3) instanceof LiteralOp )    //indexreturn
				{
					if( HopRewriteUtils.getBooleanValue((LiteralOp)hi.getInput(3)) ) //IXRET, ASC/DESC
					{
						//order(seq(2,N+1,1), indexreturn=TRUE) -> seq(1,N,1)/seq(N,1,-1)
						boolean desc = HopRewriteUtils.getBooleanValue((LiteralOp)hi.getInput(2));
						Hop seq = HopRewriteUtils.createSeqDataGenOp(hi2, !desc);
						seq.refreshSizeInformation();
						HopRewriteUtils.replaceChildReference(parent, hi, seq, pos);
						HopRewriteUtils.cleanupUnreferenced(hi);
						hi = seq;

						LOG.debug("Applied simplifyOrderedSort1.");
					}
					else if( !HopRewriteUtils.getBooleanValue((LiteralOp)hi.getInput(2)) ) //DATA, ASC
					{
						//order(seq(2,N+1,1), indexreturn=FALSE) -> seq(2,N+1,1)
						HopRewriteUtils.replaceChildReference(parent, hi, hi2, pos);
						HopRewriteUtils.cleanupUnreferenced(hi);
						hi = hi2;

						LOG.debug("Applied simplifyOrderedSort2.");
					}
				}
			}
		}

		return hi;
	}

	private static Hop fuseOrderOperationChain(Hop hi)
	{
		//order(order(X,2),1) -> order(X, (12)), 
		if( HopRewriteUtils.isReorg(hi, ReOrgOp.SORT)
				&& hi.getInput(1) instanceof LiteralOp //scalar by
				&& hi.getInput(2) instanceof LiteralOp //scalar desc
				&& HopRewriteUtils.isLiteralOfValue(hi.getInput(3), false) ) //not ixret
		{
			LiteralOp by = (LiteralOp) hi.getInput(1);
			boolean desc = HopRewriteUtils.getBooleanValue((LiteralOp)hi.getInput(2));

			//find chain of order operations with same desc/ixret configuration and single consumers
			Set<String> probe = new HashSet<>();
			ArrayList<LiteralOp> byList = new ArrayList<>();
			byList.add(by); probe.add(by.getStringValue());
			Hop input = hi.getInput(0);
			while( HopRewriteUtils.isReorg(input, ReOrgOp.SORT)
					&& input.getInput(1) instanceof LiteralOp //scalar by
					&& !probe.contains(input.getInput(1).getName())
					&& HopRewriteUtils.isLiteralOfValue(input.getInput(2), desc)
					&& HopRewriteUtils.isLiteralOfValue(hi.getInput(3), false)
					&& input.getParent().size() == 1 )
			{
				byList.add((LiteralOp)input.getInput(1));
				probe.add(input.getInput(1).getName());
				input = input.getInput(0);
			}

			//merge order chain if at least two instances
			if( byList.size() >= 2 ) {
				//create new order operations
				ArrayList<Hop> inputs = new ArrayList<>();
				inputs.add(input);
				inputs.add(HopRewriteUtils.createDataGenOpByVal(byList, 1, byList.size()));
				inputs.add(new LiteralOp(desc));
				inputs.add(new LiteralOp(false));
				Hop hnew = HopRewriteUtils.createReorg(inputs, ReOrgOp.SORT);

				//cleanup references recursively
				Hop current = hi;
				while(current != input ) {
					Hop tmp = current.getInput(0);
					HopRewriteUtils.removeAllChildReferences(current);
					current = tmp;
				}

				//rewire all parents (avoid anomalies with replicated datagen)
				List<Hop> parents = new ArrayList<>(hi.getParent());
				for( Hop p : parents )
					HopRewriteUtils.replaceChildReference(p, hi, hnew);

				hi = hnew;
				LOG.debug("Applied fuseOrderOperationChain (line "+hi.getBeginLine()+").");
			}
		}

		return hi;
	}

	/**
	 * Patterns: t(t(A)%*%t(B)+C) -> B%*%A+t(C)
	 *
	 * @param parent parent high-level operator
	 * @param hi high-level operator
	 * @param pos position
	 * @return high-level operator
	 */
	private static Hop simplifyTransposeAggBinBinaryChains(Hop parent, Hop hi, int pos)
	{
		if( HopRewriteUtils.isTransposeOperation(hi)
				&& hi.getInput(0) instanceof BinaryOp                       //basic binary
				&& ((BinaryOp)hi.getInput(0)).supportsMatrixScalarOperations())
		{
			Hop left = hi.getInput(0).getInput(0);
			Hop C = hi.getInput(0).getInput(1);

			//check matrix mult and both inputs transposes w/ single consumer
			if( left instanceof AggBinaryOp && C.getDataType().isMatrix()
					&& HopRewriteUtils.isTransposeOperation(left.getInput(0))
					&& left.getInput(0).getParent().size()==1
					&& HopRewriteUtils.isTransposeOperation(left.getInput(1))
					&& left.getInput(1).getParent().size()==1 )
			{
				Hop A = left.getInput(0).getInput(0);
				Hop B = left.getInput(1).getInput(0);

				AggBinaryOp abop = HopRewriteUtils.createMatrixMultiply(B, A);
				ReorgOp rop = HopRewriteUtils.createTranspose(C);
				BinaryOp bop = HopRewriteUtils.createBinary(abop, rop, OpOp2.PLUS);

				HopRewriteUtils.replaceChildReference(parent, hi, bop, pos);

				hi = bop;
				LOG.debug("Applied simplifyTransposeAggBinBinaryChains (line "+hi.getBeginLine()+").");
			}
		}

		return hi;
	}

	// Patterns: X + (X==0) * s -> replace(X, 0, s)
	private static Hop simplifyReplaceZeroOperation(Hop parent, Hop hi, int pos)
	{
		if( HopRewriteUtils.isBinary(hi, OpOp2.PLUS) && hi.getInput(0).isMatrix()
				&& HopRewriteUtils.isBinary(hi.getInput(1), OpOp2.MULT)
				&& hi.getInput(1).getInput(1).isScalar()
				&& HopRewriteUtils.isBinaryMatrixScalar(hi.getInput(1).getInput(0), OpOp2.EQUAL, 0)
				&& hi.getInput(1).getInput(0).getInput().contains(hi.getInput(0)) )
		{
			LinkedHashMap<String, Hop> args = new LinkedHashMap<>();
			args.put("target", hi.getInput(0));
			args.put("pattern", new LiteralOp(0));
			args.put("replacement", hi.getInput(1).getInput(1));
			Hop replace = HopRewriteUtils.createParameterizedBuiltinOp(
					hi.getInput(0), args, ParamBuiltinOp.REPLACE);
			HopRewriteUtils.replaceChildReference(parent, hi, replace, pos);
			hi = replace;
			LOG.debug("Applied simplifyReplaceZeroOperation (line "+hi.getBeginLine()+").");
		}
		return hi;
	}

	/**
	 * Pattners: t(t(X)) -> X, rev(rev(X)) -> X
	 *
	 * @param parent parent high-level operator
	 * @param hi high-level operator
	 * @param pos position
	 * @return high-level operator
	 */
	private static Hop removeUnnecessaryReorgOperation(Hop parent, Hop hi, int pos)
	{
		ReOrgOp[] lookup = new ReOrgOp[]{ReOrgOp.TRANS, ReOrgOp.REV};

		if( hi instanceof ReorgOp && HopRewriteUtils.isValidOp(((ReorgOp)hi).getOp(), lookup)  ) //first reorg
		{
			ReOrgOp firstOp = ((ReorgOp)hi).getOp();
			Hop hi2 = hi.getInput(0);
			if( hi2 instanceof ReorgOp && ((ReorgOp)hi2).getOp()==firstOp ) //second reorg w/ same type
			{
				Hop hi3 = hi2.getInput(0);
				//remove unnecessary chain of t(t())
				HopRewriteUtils.replaceChildReference(parent, hi, hi3, pos);
				HopRewriteUtils.cleanupUnreferenced(hi, hi2);
				hi = hi3;

				LOG.debug("Applied removeUnecessaryReorgOperation.");
			}
		}

		return hi;
	}

	/*
	 * Eliminate RemoveEmpty for SUM, SUM_SQ, and NNZ (number of non-zeros)
	 */
	private static Hop removeUnnecessaryRemoveEmpty(Hop parent, Hop hi, int pos)
	{
		//check if SUM or SUM_SQ is computed with input rmEmpty without select vector
		//rewrite pattern:
		//sum(removeEmpty(target=X)) -> sum(X)
		//rowSums(removeEmpty(target=X,margin="cols")) -> rowSums(X)
		//colSums(removeEmpty(target=X,margin="rows")) -> colSums(X)
		if( (HopRewriteUtils.isSum(hi) || HopRewriteUtils.isSumSq(hi))
				&& HopRewriteUtils.isRemoveEmpty(hi.getInput(0))
				&& hi.getInput(0).getParent().size() == 1 )
		{
			AggUnaryOp agg = (AggUnaryOp)hi;
			ParameterizedBuiltinOp rmEmpty = (ParameterizedBuiltinOp) hi.getInput(0);
			boolean needRmEmpty = (agg.getDirection() == Direction.Row && HopRewriteUtils.isRemoveEmpty(rmEmpty, true))
					|| (agg.getDirection() == Direction.Col && HopRewriteUtils.isRemoveEmpty(rmEmpty, false));

			if (rmEmpty.getParameterHop("select") == null && !needRmEmpty) {
				Hop input = rmEmpty.getTargetHop();
				if( input != null )  {
					HopRewriteUtils.replaceChildReference(hi, rmEmpty, input);
					return hi; //eliminate rmEmpty
				}
			}
		}

		//check if nrow is called on the output of removeEmpty
		if( HopRewriteUtils.isUnary(hi, OpOp1.NROW)
				&& HopRewriteUtils.isRemoveEmpty(hi.getInput(0), true)
				&& hi.getInput(0).getParent().size() == 1 )
		{
			ParameterizedBuiltinOp rm = (ParameterizedBuiltinOp) hi.getInput(0);
			//obtain optional select vector or input if col vector
			//(nnz will be the same as the select vector if 
			// the select vector is provided and it will be the same
			// as the input if the select vector is not provided)
			//NOTE: part of static rewrites despite size dependence for phase 
			//ordering before rewrite for DAG splits after table/removeEmpty
			Hop input = (rm.getParameterHop("select") != null) ?
					rm.getParameterHop("select") :
					(rm.getDim2() == 1) ? rm.getTargetHop() : null;

			//create new expression w/o rmEmpty if applicable
			if( input != null ) {
				HopRewriteUtils.removeAllChildReferences(rm);
				Hop hnew = HopRewriteUtils.createComputeNnz(input);

				//modify dag if nnz is called on the output of removeEmpty
				if( hnew != null ){
					HopRewriteUtils.replaceChildReference(parent, hi, hnew, pos);
					hi = hnew;
					LOG.debug("Applied removeUnnecessaryRemoveEmpty (line " + hi.getBeginLine() + ")");
				}
			}
		}

		return hi;
	}

	private static Hop removeUnnecessaryMinus(Hop parent, Hop hi, int pos)
	{
		if( hi.getDataType() == DataType.MATRIX && hi instanceof BinaryOp
				&& ((BinaryOp)hi).getOp()==OpOp2.MINUS  						//first minus
				&& hi.getInput(0) instanceof LiteralOp && ((LiteralOp)hi.getInput(0)).getDoubleValue()==0 )
		{
			Hop hi2 = hi.getInput(1);
			if( hi2.getDataType() == DataType.MATRIX && hi2 instanceof BinaryOp
					&& ((BinaryOp)hi2).getOp()==OpOp2.MINUS  						//second minus
					&& hi2.getInput(0) instanceof LiteralOp && ((LiteralOp)hi2.getInput(0)).getDoubleValue()==0 )

			{
				Hop hi3 = hi2.getInput(1);
				//remove unnecessary chain of -(-())
				HopRewriteUtils.replaceChildReference(parent, hi, hi3, pos);
				HopRewriteUtils.cleanupUnreferenced(hi, hi2);
				hi = hi3;

				LOG.debug("Applied removeUnecessaryMinus");
			}
		}

		return hi;
	}

	private static Hop simplifyGroupedAggregate(Hop hi)
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

					HopRewriteUtils.replaceChildReference(hi, th, gh, ix1);

					LOG.debug("Applied simplifyGroupedAggregateCount");
				}
			}
		}

		return hi;
	}

	private static Hop fuseMinusNzBinaryOperation(Hop parent, Hop hi, int pos)
	{
		//pattern X - (s * ppred(X,0,!=)) -> X -nz s
		//note: this is done as a hop rewrite in order to significantly reduce the 
		//memory estimate for X - tmp if X is sparse 
		if( HopRewriteUtils.isBinary(hi, OpOp2.MINUS)
				&& hi.getInput(0).getDataType()==DataType.MATRIX
				&& hi.getInput(1).getDataType()==DataType.MATRIX
				&& HopRewriteUtils.isBinary(hi.getInput(1), OpOp2.MULT) )
		{
			Hop X = hi.getInput(0);
			Hop s = hi.getInput(1).getInput(0);
			Hop pred = hi.getInput(1).getInput(1);

			if( s.getDataType()==DataType.SCALAR && pred.getDataType()==DataType.MATRIX
					&& HopRewriteUtils.isBinary(pred, OpOp2.NOTEQUAL)
					&& pred.getInput(0) == X //depend on common subexpression elimination
					&& pred.getInput(1) instanceof LiteralOp
					&& HopRewriteUtils.getDoubleValueSafe((LiteralOp)pred.getInput(1))==0 )
			{
				Hop hnew = HopRewriteUtils.createBinary(X, s, OpOp2.MINUS_NZ);

				//relink new hop into original position
				HopRewriteUtils.replaceChildReference(parent, hi, hnew, pos);
				hi = hnew;

				LOG.debug("Applied fuseMinusNzBinaryOperation (line "+hi.getBeginLine()+")");
			}
		}

		return hi;
	}

	private static Hop fuseLogNzUnaryOperation(Hop parent, Hop hi, int pos)
	{
		//pattern ppred(X,0,"!=")*log(X) -> log_nz(X)
		//note: this is done as a hop rewrite in order to significantly reduce the 
		//memory estimate and to prevent dense intermediates if X is ultra sparse  
		if( HopRewriteUtils.isBinary(hi, OpOp2.MULT)
				&& hi.getInput(0).getDataType()==DataType.MATRIX
				&& hi.getInput(1).getDataType()==DataType.MATRIX
				&& HopRewriteUtils.isUnary(hi.getInput(1), OpOp1.LOG) )
		{
			Hop pred = hi.getInput(0);
			Hop X = hi.getInput(1).getInput(0);

			if( HopRewriteUtils.isBinary(pred, OpOp2.NOTEQUAL)
					&& pred.getInput(0) == X //depend on common subexpression elimination
					&& pred.getInput(1) instanceof LiteralOp
					&& HopRewriteUtils.getDoubleValueSafe((LiteralOp)pred.getInput(1))==0 )
			{
				Hop hnew = HopRewriteUtils.createUnary(X, OpOp1.LOG_NZ);

				//relink new hop into original position
				HopRewriteUtils.replaceChildReference(parent, hi, hnew, pos);
				hi = hnew;

				LOG.debug("Applied fuseLogNzUnaryOperation (line "+hi.getBeginLine()+").");
			}
		}

		return hi;
	}

	private static Hop fuseLogNzBinaryOperation(Hop parent, Hop hi, int pos)
	{
		//pattern ppred(X,0,"!=")*log(X,0.5) -> log_nz(X,0.5)
		//note: this is done as a hop rewrite in order to significantly reduce the 
		//memory estimate and to prevent dense intermediates if X is ultra sparse  
		if( HopRewriteUtils.isBinary(hi, OpOp2.MULT)
				&& hi.getInput(0).getDataType()==DataType.MATRIX
				&& hi.getInput(1).getDataType()==DataType.MATRIX
				&& HopRewriteUtils.isBinary(hi.getInput(1), OpOp2.LOG) )
		{
			Hop pred = hi.getInput(0);
			Hop X = hi.getInput(1).getInput(0);
			Hop log = hi.getInput(1).getInput(1);

			if( HopRewriteUtils.isBinary(pred, OpOp2.NOTEQUAL)
					&& pred.getInput(0) == X //depend on common subexpression elimination
					&& pred.getInput(1) instanceof LiteralOp
					&& HopRewriteUtils.getDoubleValueSafe((LiteralOp)pred.getInput(1))==0 )
			{
				Hop hnew = HopRewriteUtils.createBinary(X, log, OpOp2.LOG_NZ);

				//relink new hop into original position
				HopRewriteUtils.replaceChildReference(parent, hi, hnew, pos);
				hi = hnew;

				LOG.debug("Applied fuseLogNzBinaryOperation (line "+hi.getBeginLine()+")");
			}
		}

		return hi;
	}

	private static Hop simplifyOuterSeqExpand(Hop parent, Hop hi, int pos)
	{
		//pattern: outer(v, t(seq(1,m)), "==") -> rexpand(v, max=m, dir=row, ignore=true, cast=false)
		//note: this rewrite supports both left/right sequence 

		if( HopRewriteUtils.isBinary(hi, OpOp2.EQUAL) && ((BinaryOp)hi).isOuter() )
		{
			if(   ( HopRewriteUtils.isTransposeOperation(hi.getInput(1)) //pattern a: outer(v, t(seq(1,m)), "==")
					&& HopRewriteUtils.isBasic1NSequence(hi.getInput(1).getInput(0)))
					|| HopRewriteUtils.isBasic1NSequence(hi.getInput(0))) //pattern b: outer(seq(1,m), t(v) "==")
			{
				//determine variable parameters for pattern a/b
				boolean isPatternB = HopRewriteUtils.isBasic1NSequence(hi.getInput(0));
				boolean isTransposeRight = HopRewriteUtils.isTransposeOperation(hi.getInput(1));
				Hop trgt = isPatternB ? (isTransposeRight ?
						hi.getInput(1).getInput(0) :                  //get v from t(v)
						HopRewriteUtils.createTranspose(hi.getInput(1)) ) : //create v via t(v')
						hi.getInput(0);                                     //get v directly 
				Hop seq = isPatternB ?
						hi.getInput(0) : hi.getInput(1).getInput(0);
				String direction = HopRewriteUtils.isBasic1NSequence(hi.getInput(0)) ? "rows" : "cols";

				//setup input parameter hops
				LinkedHashMap<String,Hop> inputargs = new LinkedHashMap<>();
				inputargs.put("target", trgt);
				inputargs.put("max", HopRewriteUtils.getBasic1NSequenceMax(seq));
				inputargs.put("dir", new LiteralOp(direction));
				inputargs.put("ignore", new LiteralOp(true));
				inputargs.put("cast", new LiteralOp(false));

				//create new hop
				ParameterizedBuiltinOp pbop = HopRewriteUtils
						.createParameterizedBuiltinOp(trgt, inputargs, ParamBuiltinOp.REXPAND);

				//relink new hop into original position
				HopRewriteUtils.replaceChildReference(parent, hi, pbop, pos);
				hi = pbop;

				LOG.debug("Applied simplifyOuterSeqExpand (line "+hi.getBeginLine()+")");
			}
		}

		return hi;
	}

	private static Hop simplifyBinaryComparisonChain(Hop parent, Hop hi, int pos) {
		if( HopRewriteUtils.isBinaryPPred(hi)
				&& HopRewriteUtils.isLiteralOfValue(hi.getInput(1), 0d, 1d)
				&& HopRewriteUtils.isBinaryPPred(hi.getInput(0)) )
		{
			BinaryOp bop = (BinaryOp) hi;
			BinaryOp bop2 = (BinaryOp) hi.getInput(0);
			boolean one = HopRewriteUtils.isLiteralOfValue(hi.getInput(1), 1);

			//pattern: outer(v1,v2,"!=") == 1 -> outer(v1,v2,"!=")
			if( (one && bop.getOp() == OpOp2.EQUAL)
					|| (!one && bop.getOp() == OpOp2.NOTEQUAL) )
			{
				HopRewriteUtils.replaceChildReference(parent, bop, bop2, pos);
				HopRewriteUtils.cleanupUnreferenced(bop);
				hi = bop2;
				LOG.debug("Applied simplifyBinaryComparisonChain1 (line "+hi.getBeginLine()+")");
			}
			//pattern: outer(v1,v2,"!=") == 0 -> outer(v1,v2,"==")
			else if( !one && bop.getOp() == OpOp2.EQUAL ) {
				OpOp2 optr = bop2.getComplementPPredOperation();
				BinaryOp tmp = HopRewriteUtils.createBinary(bop2.getInput(0),
						bop2.getInput(1), optr, bop2.isOuter());
				HopRewriteUtils.replaceChildReference(parent, bop, tmp, pos);
				HopRewriteUtils.cleanupUnreferenced(bop, bop2);
				hi = tmp;
				LOG.debug("Applied simplifyBinaryComparisonChain0 (line "+hi.getBeginLine()+")");
			}
		}

		return hi;
	}

	private static Hop simplifyCumsumColOrFullAggregates(Hop hi) {
		//pattern: colSums(cumsum(X)) -> cumSums(X*seq(nrow(X),1))
		if( (HopRewriteUtils.isAggUnaryOp(hi, AggOp.SUM, Direction.Col)
				|| HopRewriteUtils.isAggUnaryOp(hi, AggOp.SUM, Direction.RowCol))
				&& HopRewriteUtils.isUnary(hi.getInput(0), OpOp1.CUMSUM)
				&& hi.getInput(0).getParent().size()==1)
		{
			Hop cumsumX = hi.getInput(0);
			Hop X = cumsumX.getInput(0);
			Hop mult = HopRewriteUtils.createBinary(X,
					HopRewriteUtils.createSeqDataGenOp(X, false), OpOp2.MULT);
			HopRewriteUtils.replaceChildReference(hi, cumsumX, mult);
			HopRewriteUtils.removeAllChildReferences(cumsumX);
			LOG.debug("Applied simplifyCumsumColOrFullAggregates (line "+hi.getBeginLine()+")");
		}
		return hi;
	}

	private static Hop simplifyCumsumReverse(Hop parent, Hop hi, int pos) {
		//pattern: rev(cumsum(rev(X))) -> X + colSums(X) - cumsum(X)
		if( HopRewriteUtils.isReorg(hi, ReOrgOp.REV)
				&& HopRewriteUtils.isUnary(hi.getInput(0), OpOp1.CUMSUM)
				&& hi.getInput(0).getParent().size()==1
				&& HopRewriteUtils.isReorg(hi.getInput(0).getInput(0), ReOrgOp.REV)
				&& hi.getInput(0).getInput(0).getParent().size()==1)
		{
			Hop cumsumX = hi.getInput(0);
			Hop revX = cumsumX.getInput(0);
			Hop X = revX.getInput(0);
			Hop plus = HopRewriteUtils.createBinary(X, HopRewriteUtils
					.createAggUnaryOp(X, AggOp.SUM, Direction.Col), OpOp2.PLUS);
			Hop minus = HopRewriteUtils.createBinary(plus,
					HopRewriteUtils.createUnary(X, OpOp1.CUMSUM), OpOp2.MINUS);
			HopRewriteUtils.replaceChildReference(parent, hi, minus, pos);
			HopRewriteUtils.cleanupUnreferenced(hi, cumsumX, revX);

			hi = minus;
			LOG.debug("Applied simplifyCumsumReverse (line "+hi.getBeginLine()+")");
		}
		return hi;
	}

	private static Hop simplifyNotOverComparisons(Hop parent, Hop hi, int pos){
		if(HopRewriteUtils.isUnary(hi, OpOp1.NOT) && hi.getInput(0) instanceof BinaryOp
				&& hi.getInput(0).getParent().size() == 1) //NOT is only consumer
		{
			Hop binaryOperator = hi.getInput(0);
			Hop A = binaryOperator.getInput(0);
			Hop B = binaryOperator.getInput(1);
			Hop newHop = null;

			// !(A>B) -> A<=B
			if(HopRewriteUtils.isBinary(binaryOperator, OpOp2.GREATER)) {
				newHop = HopRewriteUtils.createBinary(A, B, OpOp2.LESSEQUAL);
			}
			// !(A<B) -> A>=B
			else if(HopRewriteUtils.isBinary(binaryOperator, OpOp2.LESS)) {
				newHop = HopRewriteUtils.createBinary(A, B, OpOp2.GREATEREQUAL);
			}
			// !(A==B) -> A!=B, including !(A==0) -> A!=0
			else if(HopRewriteUtils.isBinary(binaryOperator, OpOp2.EQUAL)) {
				newHop = HopRewriteUtils.createBinary(A, B, OpOp2.NOTEQUAL);
			}
			//TODO add remaining cases of comparison operators

			if(parent != null && newHop != null) {
				HopRewriteUtils.replaceChildReference(parent, hi, newHop, pos);
				HopRewriteUtils.cleanupUnreferenced(hi);
				hi = newHop;
				LOG.debug("Applied simplifyNotOverComparisons (line " + hi.getBeginLine() + ")");
			}
		}

		return hi;
	}

	/**
	 * NOTE: currently disabled since this rewrite is INVALID in the
	 * presence of NaNs (because (NaN!=NaN) is true). 
	 *
	 * @param parent parent high-level operator
	 * @param hi high-level operator
	 * @param pos position
	 * @return high-level operator
	 */
	@SuppressWarnings("unused")
	private static Hop removeUnecessaryPPred(Hop parent, Hop hi, int pos)
	{
		if( hi instanceof BinaryOp )
		{
			BinaryOp bop = (BinaryOp)hi;
			Hop left = bop.getInput(0);
			Hop right = bop.getInput(1);

			Hop datagen = null;

			//ppred(X,X,"==") -> matrix(1, rows=nrow(X),cols=nrow(Y))
			if( left==right && bop.getOp()==OpOp2.EQUAL || bop.getOp()==OpOp2.GREATEREQUAL || bop.getOp()==OpOp2.LESSEQUAL )
				datagen = HopRewriteUtils.createDataGenOp(left, 1);

			//ppred(X,X,"!=") -> matrix(0, rows=nrow(X),cols=nrow(Y))
			if( left==right && bop.getOp()==OpOp2.NOTEQUAL || bop.getOp()==OpOp2.GREATER || bop.getOp()==OpOp2.LESS )
				datagen = HopRewriteUtils.createDataGenOp(left, 0);

			if( datagen != null ) {
				HopRewriteUtils.replaceChildReference(parent, hi, datagen, pos);
				hi = datagen;
			}
		}

		return hi;
	}

	private static void removeTWriteTReadPairs(ArrayList<Hop> roots) {
		Iterator<Hop> iter = roots.iterator();
		while(iter.hasNext()) {
			Hop root = iter.next();
			if( HopRewriteUtils.isData(root, OpOpData.TRANSIENTWRITE)
					&& HopRewriteUtils.isData(root.getInput(0), OpOpData.TRANSIENTREAD)
					&& root.getName().equals(root.getInput(0).getName())
					&& !root.getInput(0).requiresCheckpoint())
			{
				iter.remove();
			}
		}
	}
}
