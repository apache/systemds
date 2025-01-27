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

import org.apache.sysds.common.Types;
import static org.apache.sysds.hops.OptimizerUtils.ALLOW_SUM_PRODUCT_REWRITES;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.LinkedHashMap;
import java.util.List;

import org.apache.sysds.common.Types.AggOp;
import org.apache.sysds.common.Types.DataType;
import org.apache.sysds.common.Types.Direction;
import org.apache.sysds.common.Types.ExecType;
import org.apache.sysds.common.Types.OpOp1;
import org.apache.sysds.common.Types.OpOp2;
import org.apache.sysds.common.Types.OpOp3;
import org.apache.sysds.common.Types.OpOp4;
import org.apache.sysds.common.Types.OpOpDG;
import org.apache.sysds.common.Types.OpOpN;
import org.apache.sysds.common.Types.ParamBuiltinOp;
import org.apache.sysds.common.Types.ReOrgOp;
import org.apache.sysds.common.Types.ValueType;
import org.apache.sysds.conf.ConfigurationManager;
import org.apache.sysds.conf.DMLConfig;
import org.apache.sysds.hops.AggBinaryOp;
import org.apache.sysds.hops.AggUnaryOp;
import org.apache.sysds.hops.BinaryOp;
import org.apache.sysds.hops.DataGenOp;
import org.apache.sysds.hops.Hop;
import org.apache.sysds.hops.IndexingOp;
import org.apache.sysds.hops.LeftIndexingOp;
import org.apache.sysds.hops.LiteralOp;
import org.apache.sysds.hops.NaryOp;
import org.apache.sysds.hops.OptimizerUtils;
import org.apache.sysds.hops.ParameterizedBuiltinOp;
import org.apache.sysds.hops.QuaternaryOp;
import org.apache.sysds.hops.ReorgOp;
import org.apache.sysds.hops.TernaryOp;
import org.apache.sysds.hops.UnaryOp;
import org.apache.sysds.lops.MapMultChain.ChainType;
import org.apache.sysds.parser.DataExpression;

/**
 * Rule: Algebraic Simplifications. Simplifies binary expressions
 * in terms of two major purposes: (1) rewrite binary operations
 * to unary operations when possible (in CP this reduces the memory
 * estimate, in MR this allows map-only operations and hence prevents 
 * unnecessary shuffle and sort) and (2) remove binary operations that
 * are in itself are unnecessary (e.g., *1 and /1).
 * 
 */
public class RewriteAlgebraicSimplificationDynamic extends HopRewriteRule
{
	//valid aggregation operation types for rowOp to Op conversions (not all operations apply)
	private static AggOp[] LOOKUP_VALID_ROW_COL_AGGREGATE = new AggOp[]{AggOp.SUM, AggOp.SUM_SQ, AggOp.MIN, AggOp.MAX, AggOp.MEAN, AggOp.VAR};
	
	//valid aggregation operation types for empty (sparse-safe) operations (not all operations apply)
	//AggOp.MEAN currently not due to missing count/corrections
	private static AggOp[] LOOKUP_VALID_EMPTY_AGGREGATE = new AggOp[]{AggOp.SUM, AggOp.SUM_SQ, AggOp.MIN, AggOp.MAX, AggOp.PROD, AggOp.TRACE};
	private static AggOp[] LOOKUP_VALID_UNNECESSARY_AGGREGATE = new AggOp[]{AggOp.SUM, AggOp.MIN, AggOp.MAX, AggOp.PROD, AggOp.TRACE};
	
	//valid unary operation types for empty (sparse-safe) operations (not all operations apply)
	private static OpOp1[] LOOKUP_VALID_EMPTY_UNARY = new OpOp1[]{OpOp1.ABS, OpOp1.SIN, OpOp1.TAN, OpOp1.SQRT, OpOp1.ROUND, OpOp1.CUMSUM}; 
	
	//valid pseudo-sparse-safe binary operators for wdivmm 
	private static OpOp2[] LOOKUP_VALID_WDIVMM_BINARY = new OpOp2[]{OpOp2.MULT, OpOp2.DIV}; 
	
	//valid unary and binary operators for wumm
	private static OpOp1[] LOOKUP_VALID_WUMM_UNARY = new OpOp1[]{
		OpOp1.ABS, OpOp1.ROUND, OpOp1.CEIL, OpOp1.FLOOR, OpOp1.EXP, OpOp1.LOG,
		OpOp1.SQRT, OpOp1.SIN, OpOp1.COS, OpOp1.SIGMOID, OpOp1.SPROP}; 
	private static OpOp2[] LOOKUP_VALID_WUMM_BINARY = new OpOp2[]{
		OpOp2.MULT, OpOp2.POW}; 
	
	@Override
	public ArrayList<Hop> rewriteHopDAGs(ArrayList<Hop> roots, ProgramRewriteStatus state) {
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
		
		return roots;
	}

	@Override
	public Hop rewriteHopDAG(Hop root, ProgramRewriteStatus state) {
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
	 * @param descendFirst true if recursively process children first
	 */
	private void rule_AlgebraicSimplification(Hop hop, boolean descendFirst) 
	{
		if(hop.isVisited())
			return;
		
		//recursively process children
		for( int i=0; i<hop.getInput().size(); i++)
		{
			Hop hi = hop.getInput(i);
			
			//process childs recursively first (to allow roll-up)
			if( descendFirst )
				rule_AlgebraicSimplification(hi, descendFirst); //see below
			
			//apply actual simplification rewrites (of childs incl checks)
			hi = removeEmptyRightIndexing(hop, hi, i);        //e.g., X[,1] -> matrix(0,ru-rl+1,cu-cl+1), if nnz(X)==0 and known indices
			hi = removeUnnecessaryRightIndexing(hop, hi, i);  //e.g., X[,1] -> X, if output == input size 
			hi = removeEmptyLeftIndexing(hop, hi, i);         //e.g., X[,1]=Y -> matrix(0,nrow(X),ncol(X)), if nnz(X)==0 and nnz(Y)==0 
			hi = removeUnnecessaryLeftIndexing(hop, hi, i);   //e.g., X[,1]=Y -> Y, if output == input dims 
			if(OptimizerUtils.ALLOW_OPERATOR_FUSION)
				hi = fuseLeftIndexingChainToAppend(hop, hi, i);   //e.g., X[,1]=A; X[,2]=B -> X=cbind(A,B), iff ncol(X)==2 and col1/2 lix
			hi = removeUnnecessaryCumulativeOp(hop, hi, i);   //e.g., cumsum(X) -> X, if nrow(X)==1;
			hi = removeUnnecessaryReorgOperation(hop, hi, i); //e.g., matrix(X) -> X, if dims(in)==dims(out); r(X)->X, if 1x1 dims
			hi = removeUnnecessaryOuterProduct(hop, hi, i);   //e.g., X*(Y%*%matrix(1,...) -> X*Y, if Y col vector
			hi = removeUnnecessaryIfElseOperation(hop, hi, i);//e.g., ifelse(E, A, B) -> A, if E==TRUE or nnz(E)==length(E)
			if(ConfigurationManager.getDMLConfig().getBooleanValue(DMLConfig.COMPILERASSISTED_RW))
				hi = removeUnnecessaryAppendTSMM(hop, hi, i);     //e.g., X = t(rbind(A,B,C)) %*% rbind(A,B,C) -> t(A)%*%A + t(B)%*%B + t(C)%*%C
			if(OptimizerUtils.ALLOW_OPERATOR_FUSION)
				hi = fuseDatagenAndReorgOperation(hop, hi, i);    //e.g., t(rand(rows=10,cols=1)) -> rand(rows=1,cols=10), if one dim=1
			hi = simplifyColwiseAggregate(hop, hi, i);        //e.g., colSums(X) -> sum(X) or X, if col/row vector
			hi = simplifyRowwiseAggregate(hop, hi, i);        //e.g., rowSums(X) -> sum(X) or X, if row/col vector
			hi = simplifyMeanAggregation(hop, hi, i);         //e.g., colSums(X)/N -> colMeans(X) if N = nrow(X)
			hi = simplifyColSumsMVMult(hop, hi, i);           //e.g., colSums(X*Y) -> t(Y) %*% X, if Y col vector
			hi = simplifyRowSumsMVMult(hop, hi, i);           //e.g., rowSums(X*Y) -> X %*% t(Y), if Y row vector
			hi = simplifyUnnecessaryAggregate(hop, hi, i);    //e.g., sum(X) -> as.scalar(X), if 1x1 dims
			hi = simplifyEmptyAggregate(hop, hi, i);          //e.g., sum(X) -> 0, if nnz(X)==0
			hi = simplifyEmptyColMeans(hop, hi, i);           //e.g., colMeans(X-colMeans(X)) if none or scaling by scalars/col-vectors
			hi = simplifyEmptyUnaryOperation(hop, hi, i);     //e.g., round(X) -> matrix(0,nrow(X),ncol(X)), if nnz(X)==0
			hi = simplifyEmptyReorgOperation(hop, hi, i);     //e.g., t(X) -> matrix(0, ncol(X), nrow(X)) 
			hi = simplifyEmptySortOperation(hop, hi, i);      //e.g., order(X) -> seq(1, nrow(X)), if nnz(X)==0 
			hi = simplifyEmptyMatrixMult(hop, hi, i);         //e.g., X%*%Y -> matrix(0,...), if nnz(Y)==0 | X if Y==matrix(1,1,1)
			hi = simplifyIdentityRepMatrixMult(hop, hi, i);   //e.g., X%*%y -> X if y matrix(1,1,1);
			hi = simplifyScalarMatrixMult(hop, hi, i);        //e.g., X%*%y -> X*as.scalar(y), if y is a 1-1 matrix
			hi = simplifyMatrixMultDiag(hop, hi, i);          //e.g., diag(X)%*%Y -> X*Y, if ncol(Y)==1 / -> Y*X if ncol(Y)>1 
			hi = simplifyDiagMatrixMult(hop, hi, i);          //e.g., diag(X%*%Y)->rowSums(X*t(Y)); if col vector
			hi = simplifyDistributiveMatrixMult(hop, hi, i);  //e.g., (A%*%B)+(A%*%C) -> A%*%(B+C)
			hi = simplifySumDiagToTrace(hi);                  //e.g., sum(diag(X)) -> trace(X); if col vector
			hi = simplifyLowerTriExtraction(hop, hi, i);      //e.g., X * cumsum(diag(matrix(1,nrow(X),1))) -> lower.tri
			hi = simplifyConstantCumsum(hop, hi, i);          //e.g., cumsum(matrix(1/n,n,1)) -> seq(1/n, 1, 1/n)
			hi = pushdownBinaryOperationOnDiag(hop, hi, i);   //e.g., diag(X)*7 -> diag(X*7); if col vector
			hi = pushdownSumOnAdditiveBinary(hop, hi, i);     //e.g., sum(A+B) -> sum(A)+sum(B); if dims(A)==dims(B)
			if(OptimizerUtils.ALLOW_OPERATOR_FUSION) {
				hi = simplifyWeightedSquaredLoss(hop, hi, i);     //e.g., sum(W * (X - U %*% t(V)) ^ 2) -> wsl(X, U, t(V), W, true), 
				hi = simplifyWeightedSigmoidMMChains(hop, hi, i); //e.g., W * sigmoid(Y%*%t(X)) -> wsigmoid(W, Y, t(X), type)
				hi = simplifyWeightedDivMM(hop, hi, i);           //e.g., t(U) %*% (X/(U%*%t(V))) -> wdivmm(X, U, t(V), left)
				hi = simplifyWeightedCrossEntropy(hop, hi, i);    //e.g., sum(X*log(U%*%t(V))) -> wcemm(X, U, t(V))
				hi = simplifyWeightedUnaryMM(hop, hi, i);         //e.g., X*exp(U%*%t(V)) -> wumm(X, U, t(V), exp)
				hi = simplifyDotProductSum(hop, hi, i);           //e.g., sum(v^2) -> t(v)%*%v if ncol(v)==1 
				hi = fuseSumSquared(hop, hi, i);                  //e.g., sum(X^2) -> sumSq(X), if ncol(X)>1
				hi = fuseAxpyBinaryOperationChain(hop, hi, i);    //e.g., (X+s*Y) -> (X+*s Y), (X-s*Y) -> (X-*s Y)
			}
			hi = reorderMinusMatrixMult(hop, hi, i);          //e.g., (-t(X))%*%y->-(t(X)%*%y), TODO size
			hi = simplifySumMatrixMult(hop, hi, i);           //e.g., sum(A%*%B) -> sum(t(colSums(A))*rowSums(B)), if not dot product / wsloss
			hi = simplifyEmptyBinaryOperation(hop, hi, i);    //e.g., X*Y -> matrix(0,nrow(X), ncol(X)) / X+Y->X / X-Y -> X
			hi = simplifyScalarMVBinaryOperation(hi);         //e.g., X*y -> X*as.scalar(y), if y is a 1-1 matrix
			hi = simplifyNnzComputation(hop, hi, i);          //e.g., sum(ppred(X,0,"!=")) -> literal(nnz(X)), if nnz known
			hi = simplifyNrowNcolComputation(hop, hi, i);     //e.g., nrow(X) -> literal(nrow(X)), if nrow known to remove data dependency
			hi = simplifyTableSeqExpand(hop, hi, i);          //e.g., table(seq(1,nrow(v)), v, nrow(v), m) -> rexpand(v, max=m, dir=row, ignore=false, cast=true)
			hi = simplyfyMMCBindZeroVector(hop, hi, i);       //e.g.. cbind((X %*% Y), matrix (0, nrow(X), 1)) -> X %*% (cbind(Y, matrix(0, nrow(Y), 1))) if nRows of x is larger than nCols of y
			if( OptimizerUtils.ALLOW_OPERATOR_FUSION )
				foldMultipleMinMaxOperations(hi);             //e.g., min(X,min(min(3,7),Y)) -> min(X,3,7,Y)
			
			//process childs recursively after rewrites (to investigate pattern newly created by rewrites)
			if( !descendFirst )
				rule_AlgebraicSimplification(hi, descendFirst);

			hi = fuseSeqAndTableExpand(hi);
		}

		hop.setVisited();
	}
	
	private static Hop removeEmptyRightIndexing(Hop parent, Hop hi, int pos) 
	{
		if( hi instanceof IndexingOp && hi.getDataType()==DataType.MATRIX  ) //indexing op
		{
			Hop input = hi.getInput(0);
			if( input.getNnz()==0 //nnz input known and empty
				&& HopRewriteUtils.isDimsKnown(hi) //output dims known
				//we also check for known indices to ensure correct error handling of out-of-bounds indexing
				&& hi.getInput(1) instanceof LiteralOp && hi.getInput(2) instanceof LiteralOp
				&& hi.getInput(3) instanceof LiteralOp && hi.getInput(4) instanceof LiteralOp)
			{
				//remove unnecessary right indexing
				Hop hnew = HopRewriteUtils.createDataGenOpByVal( new LiteralOp(hi.getDim1()),
						new LiteralOp(hi.getDim2()), null, DataType.MATRIX, ValueType.FP64, 0);
				HopRewriteUtils.replaceChildReference(parent, hi, hnew, pos);
				HopRewriteUtils.cleanupUnreferenced(hi, input);
				hi = hnew;
				
				LOG.debug("Applied removeEmptyRightIndexing (line "+hi.getBeginLine()+")");
			}
		}
		
		return hi;
	}
	
	private static Hop removeUnnecessaryRightIndexing(Hop parent, Hop hi, int pos)
	{
		if( HopRewriteUtils.isUnnecessaryRightIndexing(hi) && !hi.isScalar() ) {
			//remove unnecessary right indexing
			Hop input = hi.getInput(0);
			HopRewriteUtils.replaceChildReference(parent, hi, input, pos);
			HopRewriteUtils.cleanupUnreferenced(hi);
			hi = input;
			
			LOG.debug("Applied removeUnnecessaryRightIndexing  (line "+hi.getBeginLine()+")");
		}
		
		return hi;
	}
	
	private static Hop removeEmptyLeftIndexing(Hop parent, Hop hi, int pos) 
	{
		if( hi instanceof LeftIndexingOp && hi.getDataType() == DataType.MATRIX  ) //left indexing op
		{
			Hop input1 = hi.getInput(0); //lhs matrix
			Hop input2 = hi.getInput(1); //rhs matrix
			
			if(   input1.getNnz()==0 //nnz original known and empty
			   && input2.getNnz()==0  ) //nnz input known and empty
			{
				//remove unnecessary right indexing		
				Hop hnew = HopRewriteUtils.createDataGenOp( input1, 0);
				HopRewriteUtils.replaceChildReference(parent, hi, hnew, pos);
				HopRewriteUtils.cleanupUnreferenced(hi, input2);
				hi = hnew;
				
				LOG.debug("Applied removeEmptyLeftIndexing");
			}
		}
		
		return hi;
	}
	
	private static Hop removeUnnecessaryLeftIndexing(Hop parent, Hop hi, int pos)
	{
		if( hi instanceof LeftIndexingOp  ) //left indexing op
		{
			Hop input = hi.getInput(1); //rhs matrix/frame
			
			if( HopRewriteUtils.isEqualSize(hi, input) ) //equal dims
			{
				//equal dims of left indexing input and output -> no need for indexing
				
				//remove unnecessary right indexing
				HopRewriteUtils.replaceChildReference(parent, hi, input, pos);
				HopRewriteUtils.cleanupUnreferenced(hi);
				hi = input;
				
				LOG.debug("Applied removeUnnecessaryLeftIndexing");
			}
		}
		
		return hi;
	}
	
	private static Hop fuseLeftIndexingChainToAppend(Hop parent, Hop hi, int pos)
	{
		boolean applied = false;
		
		//pattern1: X[,1]=A; X[,2]=B -> X=cbind(A,B); matrix / frame
		if( hi instanceof LeftIndexingOp                      //first lix 
			&& HopRewriteUtils.isFullColumnIndexing((LeftIndexingOp)hi)
			&& hi.getInput(0) instanceof LeftIndexingOp //second lix	
			&& HopRewriteUtils.isFullColumnIndexing((LeftIndexingOp)hi.getInput(0))
			&& hi.getInput(0).getParent().size()==1     //first lix is single consumer
			&& hi.getInput(0).getInput(0).getDim2() == 2 ) //two column matrix
		{
			Hop input2 = hi.getInput(1); //rhs matrix
			Hop pred2 = hi.getInput(4); //cl=cu
			Hop input1 = hi.getInput(0).getInput(1); //lhs matrix
			Hop pred1 = hi.getInput(0).getInput(4); //cl=cu
			
			if( pred1 instanceof LiteralOp && HopRewriteUtils.getDoubleValueSafe((LiteralOp)pred1)==1
				&& pred2 instanceof LiteralOp && HopRewriteUtils.getDoubleValueSafe((LiteralOp)pred2)==2
				&& input1.getDataType()!=DataType.SCALAR && input2.getDataType()!=DataType.SCALAR )
			{
				//create new cbind operation and rewrite inputs
				BinaryOp bop = HopRewriteUtils.createBinary(input1, input2, OpOp2.CBIND);
				HopRewriteUtils.replaceChildReference(parent, hi, bop, pos);
				
				hi = bop;
				applied = true;
			}
		}
		
		//pattern1: X[1,]=A; X[2,]=B -> X=rbind(A,B)
		if( !applied && hi instanceof LeftIndexingOp          //first lix 
			&& HopRewriteUtils.isFullRowIndexing((LeftIndexingOp)hi)
			&& hi.getInput(0) instanceof LeftIndexingOp //second lix	
			&& HopRewriteUtils.isFullRowIndexing((LeftIndexingOp)hi.getInput(0))
			&& hi.getInput(0).getParent().size()==1     //first lix is single consumer
			&& hi.getInput(0).getInput(0).getDim1() == 2 ) //two column matrix
		{
			Hop input2 = hi.getInput(1); //rhs matrix
			Hop pred2 = hi.getInput(2); //rl=ru
			Hop input1 = hi.getInput(0).getInput(1); //lhs matrix
			Hop pred1 = hi.getInput(0).getInput(2); //rl=ru
			
			if( pred1 instanceof LiteralOp && HopRewriteUtils.getDoubleValueSafe((LiteralOp)pred1)==1
				&& pred2 instanceof LiteralOp && HopRewriteUtils.getDoubleValueSafe((LiteralOp)pred2)==2
				&& input1.getDataType()!=DataType.SCALAR && input2.getDataType()!=DataType.SCALAR )
			{
				//create new cbind operation and rewrite inputs
				BinaryOp bop = HopRewriteUtils.createBinary(input1, input2, OpOp2.RBIND);
				HopRewriteUtils.replaceChildReference(parent, hi, bop, pos);
				
				hi = bop;
				applied = true;
				
				LOG.debug("Applied fuseLeftIndexingChainToAppend2 (line "+hi.getBeginLine()+")");
			}
		}
		
		return hi;
	}
	
	private static Hop removeUnnecessaryCumulativeOp(Hop parent, Hop hi, int pos)
	{
		if( hi instanceof UnaryOp && ((UnaryOp)hi).isCumulativeUnaryOperation()  )
		{
			Hop input = hi.getInput(0); //input matrix
			
			if(   HopRewriteUtils.isDimsKnown(input)  //dims input known
		       && input.getDim1()==1 ) //1 row
			{
				OpOp1 op = ((UnaryOp)hi).getOp();
				
				//remove unnecessary unary cumsum operator
				HopRewriteUtils.replaceChildReference(parent, hi, input, pos);
				hi = input;
				
				LOG.debug("Applied removeUnnecessaryCumulativeOp: "+op);
			}
		}
		
		return hi;
	}

	private static Hop removeUnnecessaryReorgOperation(Hop parent, Hop hi, int pos) {
		if( hi instanceof ReorgOp ) {
			ReorgOp rop = (ReorgOp) hi;
			Hop input = hi.getInput(0);
			boolean apply = false;

			//equal dims of reshape input and output -> no need for reshape because
			//byrow always refers to both input/output and hence gives the same result
			apply |= (rop.getOp()==ReOrgOp.RESHAPE && HopRewriteUtils.isEqualSize(hi, input));

			//1x1 dimensions of transpose/reshape/roll -> no need for reorg
			apply |= ((rop.getOp()==ReOrgOp.TRANS || rop.getOp()==ReOrgOp.RESHAPE
					|| rop.getOp()==ReOrgOp.ROLL) && rop.getDim1()==1 && rop.getDim2()==1);

			if( apply ) {
				HopRewriteUtils.replaceChildReference(parent, hi, input, pos);
				hi = input;
				LOG.debug("Applied removeUnnecessaryReorg.");
			}
		}

		return hi;
	}
	
	private static Hop removeUnnecessaryOuterProduct(Hop parent, Hop hi, int pos)
	{
		if( hi instanceof BinaryOp  ) //binary cell operation 
		{
			OpOp2 bop = ((BinaryOp)hi).getOp();
			Hop left = hi.getInput(0);
			Hop right = hi.getInput(1);
			
			//check for matrix-vector column replication: (A + b %*% ones) -> (A + b)
			if(    HopRewriteUtils.isMatrixMultiply(right) //matrix mult with datagen
				&& HopRewriteUtils.isDataGenOpWithConstantValue(right.getInput(1), 1)
				&& right.getInput(0).getDim2() == 1 ) //column vector for mv binary
			{
				//remove unnecessary outer product
				HopRewriteUtils.replaceChildReference(hi, right, right.getInput(0), 1 );
				HopRewriteUtils.cleanupUnreferenced(right);
				
				LOG.debug("Applied removeUnnecessaryOuterProduct1 (line "+right.getBeginLine()+")");
			}
			//check for matrix-vector row replication: (A + ones %*% b) -> (A + b)
			else if( HopRewriteUtils.isMatrixMultiply(right) //matrix mult with datagen
				&& HopRewriteUtils.isDataGenOpWithConstantValue(right.getInput(0), 1)
				&& right.getInput(1).getDim1() == 1 ) //row vector for mv binary
			{
				//remove unnecessary outer product
				HopRewriteUtils.replaceChildReference(hi, right, right.getInput(1), 1 );
				HopRewriteUtils.cleanupUnreferenced(right);
				
				LOG.debug("Applied removeUnnecessaryOuterProduct2 (line "+right.getBeginLine()+")");
			}
			//check for vector-vector column replication: (a %*% ones) == b) -> outer(a, b, "==")
			else if(HopRewriteUtils.isValidOuterBinaryOp(bop) 
				&& HopRewriteUtils.isMatrixMultiply(left)
				&& HopRewriteUtils.isDataGenOpWithConstantValue(left.getInput(1), 1)
				&& (left.getInput(0).getDim2() == 1 //outer product
					|| left.getInput(1).getDim1() == 1)
				&& left.getDim1() != 1 && right.getDim1() == 1 ) //outer vector binary 
			{
				Hop hnew = HopRewriteUtils.createBinary(left.getInput(0), right, bop, true);
				HopRewriteUtils.replaceChildReference(parent, hi, hnew, pos);
				HopRewriteUtils.cleanupUnreferenced(hi);
				
				hi = hnew;
				LOG.debug("Applied removeUnnecessaryOuterProduct3 (line "+right.getBeginLine()+")");
			}
		}
		
		return hi;
	}
	
	private static Hop removeUnnecessaryIfElseOperation(Hop parent, Hop hi, int pos)
	{
		if( !HopRewriteUtils.isTernary(hi, OpOp3.IFELSE) )
			return hi;
		
		Hop expr = hi.getInput(0);
		Hop first = hi.getInput(1);
		Hop second = hi.getInput(2);
		boolean applied = false;
		
		//pattern 1: ifelse(TRUE/FALSE, A, B) -> A/B (constant scalar predicate)
		if( expr instanceof LiteralOp ) {
			Hop hnew = ((LiteralOp)expr).getBooleanValue() ? first : second;
			if( HopRewriteUtils.isEqualSize(hnew, hi) ) {
				HopRewriteUtils.replaceChildReference(parent, hi, hnew, pos );
				HopRewriteUtils.cleanupUnreferenced(hi);
				LOG.debug("Applied removeUnnecessaryIfElse1 (line "+hi.getBeginLine()+")");
				hi = hnew; applied = true;
			}
		}
		//pattern 2: ifelse(E, A, B) -> A/B if nnz(E)==length(E) or nnz(E)==0 (constant matrix predicate)
		if( !applied && expr.getNnz()==expr.getLength() || expr.getNnz()==0 ) {
			Hop hnew = expr.getNnz()==0 ? second : first;
			if( HopRewriteUtils.isEqualSize(hnew, hi) ) {
				HopRewriteUtils.replaceChildReference(parent, hi, hnew, pos );
				HopRewriteUtils.cleanupUnreferenced(hi);
				LOG.debug("Applied removeUnnecessaryIfElse2 (line "+hi.getBeginLine()+")");
				hi = hnew; applied = true;
			}
		}
		//pattern 3: ifelse(E, A, A) -> A (same input)
		if( !applied && first == second  //dep CSE
			&& HopRewriteUtils.isEqualSize(first, hi) ){
			HopRewriteUtils.replaceChildReference(parent, hi, first, pos );
			HopRewriteUtils.cleanupUnreferenced(hi);
			LOG.debug("Applied removeUnnecessaryIfElse3 (line "+hi.getBeginLine()+")");
			hi = first;
		}
		
		return hi;
	}
	
	private static Hop removeUnnecessaryAppendTSMM(Hop parent, Hop hi, int pos)
	{
		Hop hnew = null;
		//pattern 1: X = t(rbind(A,B,C)) %*% rbind(A,B,C) -> t(A)%*%A + t(B)%*%B + t(C)%*%C
		int branch = -1;
		if( HopRewriteUtils.isTsmm(hi) 
			&& HopRewriteUtils.isTransposeOperation(hi.getInput(0))
			&& HopRewriteUtils.isNary(hi.getInput(1), OpOpN.RBIND) )
		{
			List<Hop> inputs = hi.getInput(1).getInput();
			if( HopRewriteUtils.checkAvgRowsGteCols(inputs) ) {
				Hop[] tsmms = inputs.stream()
					.map(h -> HopRewriteUtils.createTsmm(h, true)).toArray(Hop[]::new);
				hnew = HopRewriteUtils.createNary(OpOpN.PLUS, tsmms);
				//cleanup parent references from rbind
				//HopRewriteUtils.removeAllChildReferences(hi.getInput(1));
				branch = 1;
			}
		}
		//pattern 2: X = t(rbind(A,B,C)) %*% rbind(D,E,F)  -> t(A)%*%D + t(B)%*%E + t(C)%*%F
		else if( HopRewriteUtils.isMatrixMultiply(hi) 
			&& HopRewriteUtils.isTransposeOperation(hi.getInput(0))
			&& HopRewriteUtils.isNary(hi.getInput(0).getInput(0), OpOpN.RBIND)
			&& HopRewriteUtils.isNary(hi.getInput(1), OpOpN.RBIND) )
		{
			List<Hop> inputs1 = hi.getInput(0).getInput(0).getInput();
			List<Hop> inputs2 = hi.getInput(1).getInput();
			if( HopRewriteUtils.checkAvgRowsGteCols(inputs1) 
				&& HopRewriteUtils.checkAvgRowsGteCols(inputs2) 
				&& HopRewriteUtils.checkConsistentRows(inputs1, inputs2) ) 
			{
				Hop[] mms = new Hop[inputs1.size()];
				for( int i=0; i<inputs1.size(); i++ )
					mms[i] = HopRewriteUtils.createMatrixMultiply(
						HopRewriteUtils.createTranspose(inputs1.get(i)), inputs2.get(i));
				hnew = HopRewriteUtils.createNary(OpOpN.PLUS, mms);
				//cleanup parent references from rbind left/right
				//HopRewriteUtils.removeAllChildReferences(hi.getInput(0).getInput(0));
				//HopRewriteUtils.removeAllChildReferences(hi.getInput(1));
				branch = 2;
			}
		}
		//pattern 3: X = t(cbind(A, B)) %*% cbind(A, B), w/ one cbind consumer (twice in tsmm)
		else if( HopRewriteUtils.isTsmm(hi) && hi.getInput(1).getParent().size()==2
			&& HopRewriteUtils.isTransposeOperation(hi.getInput(0))
			&& HopRewriteUtils.isBinary(hi.getInput(1), OpOp2.CBIND) )
		{
			Hop input1 = hi.getInput(1).getInput(0);
			Hop input2 = hi.getInput(1).getInput(1);
			if( input1.getDim1() > input1.getDim2() && input2.getDim2() == 1 ) {
				hnew = HopRewriteUtils.createPartialTsmmCbind(
					input1, input2, HopRewriteUtils.createTsmm(input1, true));
				branch = 3;
			}
		}
		
		//modify dag if one of the above rules applied
		if( hnew != null ){ 
			HopRewriteUtils.replaceChildReference(parent, hi, hnew, pos);
			HopRewriteUtils.removeAllChildReferences(hi);
			hi = hnew;
			LOG.debug("Applied removeUnnecessaryAppendTSMM"
				+ branch + " (line " + hi.getBeginLine() + ")");
		}
		
		return hi;
	}
	
	private static Hop fuseDatagenAndReorgOperation(Hop parent, Hop hi, int pos)
	{
		if( HopRewriteUtils.isTransposeOperation(hi)
			&& hi.getInput(0) instanceof DataGenOp     //datagen
			&& hi.getInput(0).getParent().size()==1 )  //transpose only consumer
		{
			DataGenOp dop = (DataGenOp)hi.getInput(0);
			if(    (dop.getOp() == OpOpDG.RAND || dop.getOp() == OpOpDG.SINIT)
				&& (dop.getDim1()==1 || dop.getDim2()==1 )) 
			{
				//relink all parents and dataop (remove transpose)
				HopRewriteUtils.removeAllChildReferences(hi);
				List<Hop> parents = new ArrayList<>(hi.getParent());
				for( int i=0; i<parents.size(); i++ ) {
					Hop lparent = parents.get(i);
					int ppos = HopRewriteUtils.getChildReferencePos(lparent, hi);
					HopRewriteUtils.removeChildReferenceByPos(lparent, hi, ppos);
					HopRewriteUtils.addChildReference(lparent, dop, pos);
				}
				
				//flip rows/cols attributes in datagen
				HashMap<String, Integer> rparams = dop.getParamIndexMap();
				int pos1 = rparams.get(DataExpression.RAND_ROWS);
				int pos2 = rparams.get(DataExpression.RAND_COLS);
				rparams.put(DataExpression.RAND_ROWS, pos2);
				rparams.put(DataExpression.RAND_COLS, pos1);
				dop.refreshSizeInformation();
				
				hi = dop;
				
				LOG.debug("Applied fuseDatagenReorgOperation.");
			}
		}
		
		return hi;
	}

	private static Hop simplifyColwiseAggregate( Hop parent, Hop hi, int pos ) {
		if( hi instanceof AggUnaryOp  ) 
		{
			AggUnaryOp uhi = (AggUnaryOp)hi;
			Hop input = uhi.getInput(0);
			
			if( HopRewriteUtils.isValidOp(uhi.getOp(), LOOKUP_VALID_ROW_COL_AGGREGATE) ) {
				if( uhi.getDirection() == Direction.Col  )
				{
					if( input.getDim1() == 1 )
					{
						if (uhi.getOp() == AggOp.VAR) {
							// For the column variance aggregation, if the input is a row vector,
							// the column variances will each be zero.
							// Therefore, perform a rewrite from COLVAR(X) to a row vector of zeros.
							Hop emptyRow = HopRewriteUtils.createDataGenOp(uhi, input, 0);
							HopRewriteUtils.replaceChildReference(parent, hi, emptyRow, pos);
							HopRewriteUtils.cleanupUnreferenced(hi, input);
							hi = emptyRow;

							LOG.debug("Applied simplifyColwiseAggregate for colVars");
						} else {
							// All other valid column aggregations over a row vector will result
							// in the row vector itself.
							// Therefore, remove unnecessary col aggregation for 1 row.
							HopRewriteUtils.replaceChildReference(parent, hi, input, pos);
							HopRewriteUtils.cleanupUnreferenced(hi);
							hi = input;

							LOG.debug("Applied simplifyColwiseAggregate1");
						}
					}
					else if( input.getDim2() == 1 )
					{
						//get old parents (before creating cast over aggregate)
						List<Hop> parents = new ArrayList<>(hi.getParent());

						//simplify col-aggregate to full aggregate
						uhi.setDirection(Direction.RowCol);
						uhi.setDataType(DataType.SCALAR);
						
						//create cast to keep same output datatype
						UnaryOp cast = HopRewriteUtils.createUnary(uhi, OpOp1.CAST_AS_MATRIX);
						
						//rehang cast under all parents
						for( Hop p : parents ) {
							int ix = HopRewriteUtils.getChildReferencePos(p, hi);
							HopRewriteUtils.replaceChildReference(p, hi, cast, ix);
						}
						
						hi = cast;
						
						LOG.debug("Applied simplifyColwiseAggregate2");
					}
				}
			}
		}
		
		return hi;
	}

	private static Hop simplifyRowwiseAggregate( Hop parent, Hop hi, int pos ) {
		if( hi instanceof AggUnaryOp  ) 
		{
			AggUnaryOp uhi = (AggUnaryOp)hi;
			Hop input = uhi.getInput(0);
			
			if( HopRewriteUtils.isValidOp(uhi.getOp(), LOOKUP_VALID_ROW_COL_AGGREGATE) ) {
				if( uhi.getDirection() == Direction.Row  )
				{
					if( input.getDim2() == 1 )
					{
						if (uhi.getOp() == AggOp.VAR) {
							// For the row variance aggregation, if the input is a column vector,
							// the row variances will each be zero.
							// Therefore, perform a rewrite from ROWVAR(X) to a column vector of
							// zeros.
							Hop emptyCol = HopRewriteUtils.createDataGenOp(input, uhi, 0);
							HopRewriteUtils.replaceChildReference(parent, hi, emptyCol, pos);
							HopRewriteUtils.cleanupUnreferenced(hi, input);

							// replace current HOP with new empty column HOP
							hi = emptyCol;

							LOG.debug("Applied simplifyRowwiseAggregate for rowVars");
						} else {
							// All other valid row aggregations over a column vector will result
							// in the column vector itself.
							// Therefore, remove unnecessary row aggregation for 1 col
							HopRewriteUtils.replaceChildReference(parent, hi, input, pos);
							HopRewriteUtils.cleanupUnreferenced(hi);
							hi = input;

							LOG.debug("Applied simplifyRowwiseAggregate1 (line "+hi.getBeginLine()+")");
						}
					}
					else if( input.getDim1() == 1 )
					{
						//get old parents (before creating cast over aggregate)
						List<Hop> parents = new ArrayList<>(hi.getParent());

						//simplify row-aggregate to full aggregate
						uhi.setDirection(Direction.RowCol);
						uhi.setDataType(DataType.SCALAR);
						
						//create cast to keep same output datatype
						UnaryOp cast = HopRewriteUtils.createUnary(uhi, OpOp1.CAST_AS_MATRIX);
						
						//rehang cast under all parents
						for( Hop p : parents ) {
							int ix = HopRewriteUtils.getChildReferencePos(p, hi);
							HopRewriteUtils.replaceChildReference(p, hi, cast, ix);
						}
						
						hi = cast;
						
						LOG.debug("Applied simplifyRowwiseAggregate2");
					}
				}
			}
		}
		
		return hi;
	}
	
	private static Hop simplifyMeanAggregation( Hop parent, Hop hi, int pos ) {
		// colSums(X)/N -> colMeans(X), if N = nrow(X), all directions but different vals
		if( HopRewriteUtils.isBinary(hi, OpOp2.DIV)
			&& HopRewriteUtils.isAggUnaryOp(hi.getInput(0), AggOp.SUM)
			&& hi.getInput(0).getParent().size()==1 //prevent repeated scans
			&& hi.getInput(1).getDataType().isScalar())
		{
			AggUnaryOp agg = (AggUnaryOp)hi.getInput(0);
			Hop in = agg.getInput(0);
			Hop N = hi.getInput(1);
			if( (agg.getDirection().isRow() && HopRewriteUtils.isSizeExpressionOf(N, in, false))
				|| (agg.getDirection().isCol() && HopRewriteUtils.isSizeExpressionOf(N, in, true)) )
			{
				HopRewriteUtils.replaceChildReference(parent, hi, agg, pos);
				HopRewriteUtils.cleanupUnreferenced(hi, N);
				agg.setOp(AggOp.MEAN);
				hi = agg;
				LOG.debug("Applied simplifyMeanAggregation");
			}
		}
		
		return hi;
	}
	
	private static Hop simplifyColSumsMVMult( Hop parent, Hop hi, int pos ) 
	{
		//colSums(X*Y) -> t(Y) %*% X, if Y col vector; additional transpose later
		//removed by other rewrite if unnecessary, i.e., if Y==t(Z)
		if( hi instanceof AggUnaryOp  ) 
		{
			AggUnaryOp uhi = (AggUnaryOp)hi;
			Hop input = uhi.getInput(0);
			
			if( uhi.getOp() == AggOp.SUM && uhi.getDirection() == Direction.Col //colsums
			    && HopRewriteUtils.isBinary(input, OpOp2.MULT) ) //b(*) 
			{
				Hop left = input.getInput(0);
				Hop right = input.getInput(1);
				
				if(    left.getDim1()>1 && left.getDim2()>1 
					&& right.getDim1()>1 && right.getDim2()==1 ) // MV (col vector)
				{
					//create new operators 
					ReorgOp trans = HopRewriteUtils.createTranspose(right);
					AggBinaryOp mmult = HopRewriteUtils.createMatrixMultiply(trans, left);
					
					//relink new child
					HopRewriteUtils.replaceChildReference(parent, hi, mmult, pos);
					HopRewriteUtils.cleanupUnreferenced(uhi, input);
					hi = mmult;
					
					LOG.debug("Applied simplifyColSumsMVMult");
				}
			}
		}
		
		return hi;
	}
	
	private static Hop simplifyRowSumsMVMult( Hop parent, Hop hi, int pos ) 
	{
		//rowSums(X * Y) -> X %*% t(Y), if Y row vector; additional transpose later
		//removed by other rewrite if unnecessary, i.e., if Y==t(Z)
		if( hi instanceof AggUnaryOp  ) 
		{
			AggUnaryOp uhi = (AggUnaryOp)hi;
			Hop input = uhi.getInput(0);
			
			if( uhi.getOp() == AggOp.SUM && uhi.getDirection() == Direction.Row //rowsums
				&& HopRewriteUtils.isBinary(input, OpOp2.MULT) ) //b(*) 
			{
				Hop left = input.getInput(0);
				Hop right = input.getInput(1);
				
				if(    left.getDim1()>1 && left.getDim2()>1      
					&& right.getDim1()==1 && right.getDim2()>1 ) // MV (row vector)
				{
					//create new operators 
					ReorgOp trans = HopRewriteUtils.createTranspose(right);
					AggBinaryOp mmult = HopRewriteUtils.createMatrixMultiply(left, trans);
					
					//relink new child
					HopRewriteUtils.replaceChildReference(parent, hi, mmult, pos);
					HopRewriteUtils.cleanupUnreferenced(hi, input);
					hi = mmult;
					
					LOG.debug("Applied simplifyRowSumsMVMult");
				}
			}	
		}
		
		return hi;
	}
	
	private static Hop simplifyUnnecessaryAggregate(Hop parent, Hop hi, int pos) 
	{
		// TODO implement for tensor
		//e.g., sum(X) -> as.scalar(X) if 1x1 (applies to sum, min, max, prod, trace)
		if( hi instanceof AggUnaryOp && ((AggUnaryOp)hi).getDirection()==Direction.RowCol  ) 
		{
			AggUnaryOp uhi = (AggUnaryOp)hi;
			Hop input = uhi.getInput(0);
			
			if( HopRewriteUtils.isValidOp(uhi.getOp(), LOOKUP_VALID_UNNECESSARY_AGGREGATE) ){		
				
				if( input.getDim1()==1 && input.getDim2()==1 && input.getDataType()==DataType.MATRIX)
				{
					UnaryOp cast = HopRewriteUtils.createUnary(input, OpOp1.CAST_AS_SCALAR);
					
					//remove unnecessary aggregation 
					HopRewriteUtils.replaceChildReference(parent, hi, cast, pos);
					hi = cast;
					
					LOG.debug("Applied simplifyUnncessaryAggregate");
				}
			}			
		}
		
		return hi;
	}
	
	private static Hop simplifyEmptyAggregate(Hop parent, Hop hi, int pos) 
	{
		if( hi instanceof AggUnaryOp )
		{
			AggUnaryOp uhi = (AggUnaryOp)hi;
			Hop input = uhi.getInput(0);
			
			//check for valid empty aggregates, except for matrices with zero rows/cols
			if( HopRewriteUtils.isValidOp(uhi.getOp(), LOOKUP_VALID_EMPTY_AGGREGATE) 
				&& HopRewriteUtils.isEmpty(input)
				&& input.getDim1()>=1 && input.getDim2() >= 1 )
			{
				Hop hnew = null;
				if( uhi.getDirection() == Direction.RowCol ) 
					hnew = new LiteralOp(0.0);
				else if( uhi.getDirection() == Direction.Col ) 
					hnew = HopRewriteUtils.createDataGenOp(uhi, input, 0); //nrow(uhi)=1
				else //if( uhi.getDirection() == Direction.Row ) 
					hnew = HopRewriteUtils.createDataGenOp(input, uhi, 0); //ncol(uhi)=1
				
				//add new child to parent input
				HopRewriteUtils.replaceChildReference(parent, hi, hnew, pos);
				hi = hnew;
				LOG.debug("Applied simplifyEmptyAggregate");
			}
		}
		return hi;
	}
	
	private static Hop simplifyEmptyColMeans(Hop parent, Hop hi, int pos) 
	{
		if( hi.dimsKnown() && HopRewriteUtils.isAggUnaryOp(hi, AggOp.MEAN, Direction.Col) ) {
			Hop in = hi.getInput(0);
			//colMeans(X-colMeans(X)) without scaling
			boolean apply = HopRewriteUtils.isBinary(in, OpOp2.MINUS)
				&& HopRewriteUtils.isAggUnaryOp(in.getInput(1), AggOp.MEAN, Direction.Col)
				&& in.getInput(0) == in.getInput(1).getInput(0); //requires CSE
			//colMeans((X-colMeans(X))/colSds(X)) if scaling by scalars/col-vectors
			apply = apply || (HopRewriteUtils.isBinary(in, OpOp2.DIV, OpOp2.MULT)
				&& in.getInput(1).getDim1()==1 //row vector
				&& HopRewriteUtils.isBinary(in.getInput(0), OpOp2.MINUS)
				&& HopRewriteUtils.isAggUnaryOp(in.getInput(0).getInput(1), AggOp.MEAN, Direction.Col)
				&& in.getInput(0).getInput(0) == in.getInput(0).getInput(1).getInput(0));
			if( apply ) {
				Hop hnew = HopRewriteUtils.createDataGenOp(hi, hi, 0); //empty
				HopRewriteUtils.replaceChildReference(parent, hi, hnew, pos);
				hi = hnew;
				LOG.debug("Applied simplifyEmptyColMeans");
			}
		}
		return hi;
	}
	
	private static Hop simplifyEmptyUnaryOperation(Hop parent, Hop hi, int pos) 
	{
		if( hi instanceof UnaryOp  ) 
		{
			UnaryOp uhi = (UnaryOp)hi;
			Hop input = uhi.getInput(0);
			
			if( HopRewriteUtils.isValidOp(uhi.getOp(), LOOKUP_VALID_EMPTY_UNARY) ){		
				
				if( HopRewriteUtils.isEmpty(input) )
				{
					//create literal add it to parent
					Hop hnew = HopRewriteUtils.createDataGenOp(input, 0);
					HopRewriteUtils.replaceChildReference(parent, hi, hnew, pos);
					hi = hnew;
					
					LOG.debug("Applied simplifyEmptyUnaryOperation");
				}
			}
		}
		
		return hi;
	}
	
	private static Hop simplifyEmptyReorgOperation(Hop parent, Hop hi, int pos) 
	{
		if( hi instanceof ReorgOp  ) 
		{
			ReorgOp rhi = (ReorgOp)hi;
			Hop input = rhi.getInput(0);
			
			if( HopRewriteUtils.isEmpty(input) ) //empty input
			{
				//reorg-operation-specific rewrite  
				Hop hnew = null;
				if( rhi.getOp() == ReOrgOp.TRANS )
					hnew = HopRewriteUtils.createDataGenOp(input, true, input, true, 0);
				else if( rhi.getOp() == ReOrgOp.REV )
					hnew = HopRewriteUtils.createDataGenOp(input, 0);
				else if( rhi.getOp() == ReOrgOp.DIAG ) {
					if( HopRewriteUtils.isDimsKnown(input) ) {
						if( input.getDim2()==1 ) //diagv2m
							hnew = HopRewriteUtils.createDataGenOp(input, false, input, true, 0);
						else //diagm2v TODO support tensor operation
							hnew = HopRewriteUtils.createDataGenOpByVal(
									HopRewriteUtils.createValueHop(input,true), new LiteralOp(1),
									null, DataType.MATRIX, ValueType.FP64, 0);
					}
				}
				else if( rhi.getOp() == ReOrgOp.RESHAPE )
					hnew = HopRewriteUtils.createDataGenOpByVal(rhi.getInput(1), rhi.getInput(2),
							rhi.getInput(3), rhi.getDataType(), rhi.getValueType(), 0);
			
				//modify dag if one of the above rules applied
				if( hnew != null ){ 
					HopRewriteUtils.replaceChildReference(parent, hi, hnew, pos);
					hi = hnew;
					
					LOG.debug("Applied simplifyEmptyReorgOperation");
				}
			}
			
		}
		
		return hi;
	}
	
	private static Hop simplifyEmptySortOperation(Hop parent, Hop hi, int pos) 
	{
		//order(X, indexreturn=FALSE) -> matrix(0,nrow(X),1)
		//order(X, indexreturn=TRUE) -> seq(1,nrow(X),1)
		if( hi instanceof ReorgOp && ((ReorgOp)hi).getOp()==ReOrgOp.SORT  ) 
		{
			ReorgOp rhi = (ReorgOp)hi;
			Hop input = rhi.getInput(0);
			
			if( HopRewriteUtils.isEmpty(input) ) //empty input
			{
				//reorg-operation-specific rewrite  
				Hop hnew = null;
				boolean ixret = false;
				
				if( rhi.getInput(3) instanceof LiteralOp ) //index return known
				{
					ixret = HopRewriteUtils.getBooleanValue((LiteralOp)rhi.getInput(3));
					if( ixret )
						hnew = HopRewriteUtils.createSeqDataGenOp(input);
					else
						hnew = HopRewriteUtils.createDataGenOp(input, 0);
				}
								
				//modify dag if one of the above rules applied
				if( hnew != null ){ 
					HopRewriteUtils.replaceChildReference(parent, hi, hnew, pos);
					hi = hnew;
					
					LOG.debug("Applied simplifyEmptySortOperation (indexreturn="+ixret+").");
				}
			}
			
		}
		
		return hi;
	}
	
	private static Hop simplifyEmptyMatrixMult(Hop parent, Hop hi, int pos) {
		if( HopRewriteUtils.isMatrixMultiply(hi) ) //X%*%Y -> matrix(0, )
		{
			Hop left = hi.getInput(0);
			Hop right = hi.getInput(1);
		
			if(    HopRewriteUtils.isEmpty(left)  //one input empty
				|| HopRewriteUtils.isEmpty(right) )
			{
				//create datagen and add it to parent
				Hop hnew = HopRewriteUtils.createDataGenOp(left, right, 0);
				HopRewriteUtils.replaceChildReference(parent, hi, hnew, pos);
				hi = hnew;
				
				LOG.debug("Applied simplifyEmptyMatrixMult");
			}
		}
		
		return hi;
	}
	
	private static Hop simplifyIdentityRepMatrixMult(Hop parent, Hop hi, int pos) 
	{
		if( HopRewriteUtils.isMatrixMultiply(hi) ) //X%*%Y -> X, if y is matrix(1,1,1)
		{
			Hop left = hi.getInput(0);
			Hop right = hi.getInput(1);
			
			// X %*% y -> X
			if( HopRewriteUtils.isDimsKnown(right) && right.getDim1()==1 && right.getDim2()==1 && //scalar right
				right instanceof DataGenOp && ((DataGenOp)right).getOp()==OpOpDG.RAND
				&& ((DataGenOp)right).hasConstantValue(1.0)) //matrix(1,)
			{
				HopRewriteUtils.replaceChildReference(parent, hi, left, pos);
				hi = left;
				
				LOG.debug("Applied simplifyIdentiyMatrixMult");
			}
		}
		
		return hi;
	}
	
	private static Hop simplifyScalarMatrixMult(Hop parent, Hop hi, int pos)
	{
		if( HopRewriteUtils.isMatrixMultiply(hi) ) //X%*%Y
		{
			Hop left = hi.getInput(0);
			Hop right = hi.getInput(1);
			
			// y %*% X -> as.scalar(y) * X
			if( HopRewriteUtils.isDimsKnown(left) && left.getDim1()==1 && left.getDim2()==1 ) //scalar left
			{
				UnaryOp cast = HopRewriteUtils.createUnary(left, OpOp1.CAST_AS_SCALAR);
				BinaryOp mult = HopRewriteUtils.createBinary(cast, right, OpOp2.MULT);
				
				//add mult to parent
				HopRewriteUtils.replaceChildReference(parent, hi, mult, pos);
				HopRewriteUtils.cleanupUnreferenced(hi);
				
				hi = mult;
				LOG.debug("Applied simplifyScalarMatrixMult1");
			}
			// X %*% y -> X * as.scalar(y)
			else if( HopRewriteUtils.isDimsKnown(right) && right.getDim1()==1 && right.getDim2()==1 ) //scalar right
			{
				UnaryOp cast = HopRewriteUtils.createUnary(right, OpOp1.CAST_AS_SCALAR);
				BinaryOp mult = HopRewriteUtils.createBinary(cast, left, OpOp2.MULT);
				
				//add mult to parent
				HopRewriteUtils.replaceChildReference(parent, hi, mult, pos);
				HopRewriteUtils.cleanupUnreferenced(hi);
				
				hi = mult;
				LOG.debug("Applied simplifyScalarMatrixMult2");
			}
		}
		
		return hi;
	}
	
	private static Hop simplifyMatrixMultDiag(Hop parent, Hop hi, int pos) 
	{
		Hop hnew = null;
		
		if( HopRewriteUtils.isMatrixMultiply(hi) ) //X%*%Y
		{
			
			Hop left = hi.getInput(0);
			Hop right = hi.getInput(1);
		
			// diag(X) %*% Y -> X * Y / diag(X) %*% Y -> Y * X 
			// previously rep required for the second case: diag(X) %*% Y -> (X%*%ones) * Y
			if( left instanceof ReorgOp && ((ReorgOp)left).getOp()==ReOrgOp.DIAG //left diag
				&& HopRewriteUtils.isDimsKnown(left) && left.getDim2()>1 ) //diagV2M
			{
				if( right.getDim2()==1 ) //right column vector
				{
					//create binary operation over input and right
					Hop input = left.getInput(0); //diag input
					hnew = HopRewriteUtils.createBinary(input, right, OpOp2.MULT);
					
					LOG.debug("Applied simplifyMatrixMultDiag1");
				}
				else if( right.getDim2()>1 ) //multi column vector
				{
					//create binary operation over input and right; in contrast to above rewrite,
					//we need to switch the order because MV binary cell operations require vector on the right
					Hop input = left.getInput(0); //diag input
					hnew = HopRewriteUtils.createBinary(right, input, OpOp2.MULT);
					
					//NOTE: previously to MV binary cell operations we replicated the left
					//(if moderate number of columns: 2), but this is no longer required
					
					LOG.debug("Applied simplifyMatrixMultDiag2");
				}
			}
			
			//notes: similar rewrites would be possible for the right side as well, just transposed into the right alignment
		}
		
		//if one of the above rewrites applied
		if( hnew !=null ){
			//add mult to parent
			HopRewriteUtils.replaceChildReference(parent, hi, hnew, pos);
			HopRewriteUtils.cleanupUnreferenced(hi);
			
			hi = hnew;
		}
		
		return hi;
	}

	private static Hop simplifyDiagMatrixMult(Hop parent, Hop hi, int pos) {
		if(hi instanceof ReorgOp && ((ReorgOp) hi).getOp() == ReOrgOp.DIAG && hi.getDim2() == 1) //diagM2V
		{
			Hop hi2 = hi.getInput(0);
			if(HopRewriteUtils.isMatrixMultiply(hi2)) //X%*%Y
			{
				Hop left = hi2.getInput(0);
				Hop right = hi2.getInput(1);

				//create new operators (incl refresh size inside for transpose)
				ReorgOp trans = HopRewriteUtils.createTranspose(right);
				BinaryOp mult = HopRewriteUtils.createBinary(left, trans, OpOp2.MULT);
				AggUnaryOp rowSum = HopRewriteUtils.createAggUnaryOp(mult, AggOp.SUM, Direction.Row);

				//rehang new subdag under parent node
				HopRewriteUtils.replaceChildReference(parent, hi, rowSum, pos);
				HopRewriteUtils.cleanupUnreferenced(hi, hi2);

				hi = rowSum;
				LOG.debug("Applied simplifyDiagMatrixMult");
			}
		}

		return hi;
	}

	private static Hop simplifyDistributiveMatrixMult(Hop parent, Hop hi, int pos) {
		// A%*%B + A%*%C -> A%*%(B+C)
		if(HopRewriteUtils.isBinary(hi, OpOp2.PLUS) 
			&& HopRewriteUtils.isMatrixMultiply(hi.getInput(0))
			&& HopRewriteUtils.isMatrixMultiply(hi.getInput(1))
			&& hi.getInput(0).getParent().size() == 1 //single consumer
			&& hi.getInput(1).getParent().size() == 1 //single consumer
			&& hi.getInput(0).getInput(0) == hi.getInput(1).getInput(0)) //common A
		{
			Hop A = hi.getInput(0).getInput(0);
			Hop B = hi.getInput(0).getInput(1);
			Hop C = hi.getInput(1).getInput(1);
			boolean dense = HopRewriteUtils.isDense(A) 
				&& HopRewriteUtils.isDense(B) && HopRewriteUtils.isDense(C);
			//compute floating point and mem bandwidth requirements and 
			//according for special cases where C might be a column vector
			long m = A.getDim1(), n = A.getDim2(), l = B.getDim2(), o = C.getDim2();
			long costOriginal = m * n * l + m * n * o + m * l //FLOP
						+ m*n + n*l + n*o + m*l + m*o + m*l;  //I/O ABC+intermediates
			long costRewrite = n * l + m * n * l              //FLOP
						+ m*n + n*l + n*o + n*l + m*l;        //I/O ABC+intermediates
			//Check that rewrite reduces FLOPs
			if(dense && costRewrite < costOriginal) {
				Hop BplusC = HopRewriteUtils.createBinary(B, C, OpOp2.PLUS);
				Hop newHop = HopRewriteUtils.createMatrixMultiply(A, BplusC);
				if(parent != null) {
					HopRewriteUtils.replaceChildReference(parent, hi, newHop, pos);
					HopRewriteUtils.cleanupUnreferenced(hi);
					hi = newHop;
					LOG.debug("Applied simplifyDistributiveMatrixMult (line " + hi.getBeginLine() + ")");
				}
			}
		}
		return hi;
	}

	private static Hop simplifySumDiagToTrace(Hop hi) {
		if(hi instanceof AggUnaryOp) {
			AggUnaryOp au = (AggUnaryOp) hi;
			if(au.getOp() == AggOp.SUM && au.getDirection() == Direction.RowCol)    //sum
			{
				Hop hi2 = au.getInput(0);
				if(hi2 instanceof ReorgOp && ((ReorgOp) hi2).getOp() == ReOrgOp.DIAG && hi2.getDim2() == 1) //diagM2V
				{
					Hop hi3 = hi2.getInput(0);

					//remove diag operator
					HopRewriteUtils.replaceChildReference(au, hi2, hi3, 0);
					HopRewriteUtils.cleanupUnreferenced(hi2);
					
					//change sum to trace
					au.setOp( AggOp.TRACE );
					
					LOG.debug("Applied simplifySumDiagToTrace");
				}
			}
		}
		
		return hi;
	}
	
	private static Hop simplifyLowerTriExtraction(Hop parent, Hop hi, int pos) {
		//pattern: X * cumsum(diag(matrix(1,nrow(X),1))) -> lower.tri (only right)
		if( HopRewriteUtils.isBinary(hi, OpOp2.MULT) 
			&& hi.getDim1() == hi.getDim2() && hi.getDim1() > 1 ) {
			Hop left = hi.getInput(0);
			Hop right = hi.getInput(1);
			
			if( HopRewriteUtils.isUnary(right, OpOp1.CUMSUM) && right.getParent().size()==1
				&& HopRewriteUtils.isReorg(right.getInput(0), ReOrgOp.DIAG)
				&& HopRewriteUtils.isDataGenOpWithConstantValue(right.getInput(0).getInput(0), 1d))
			{
				LinkedHashMap<String,Hop> args = new LinkedHashMap<>();
				args.put("target", left);
				args.put("diag", new LiteralOp(true));
				args.put("values", new LiteralOp(true));
				Hop hnew = HopRewriteUtils.createParameterizedBuiltinOp(
					left, args, ParamBuiltinOp.LOWER_TRI);
				HopRewriteUtils.replaceChildReference(parent, hi, hnew);
				HopRewriteUtils.removeAllChildReferences(right);
				
				hi = hnew;
				LOG.debug("Applied simplifyLowerTriExtraction");
			}
		}
		return hi;
	}
	
	private static Hop simplifyConstantCumsum(Hop parent, Hop hi, int pos) {
		//pattern: cumsum(matrix(1/n,n,1)) -> seq(1/n, 1, 1/n)
		if( HopRewriteUtils.isUnary(hi, OpOp1.CUMSUM)
			&& HopRewriteUtils.isDataGenOpWithConstantValue(hi.getInput(0))
			&& hi.getInput(0).getParent().size() == 1 //cumsum only consumer
			&& hi.dimsKnown() && hi.getDim2() == 1 )
		{
			Hop constVal = ((DataGenOp) hi.getInput(0)).getConstantValue();
			Hop to = HopRewriteUtils.createBinary(new LiteralOp(hi.getDim1()), constVal, OpOp2.MULT);
			Hop hnew = HopRewriteUtils.createSeqDataGenOp(hi, constVal, to, constVal);
			HopRewriteUtils.replaceChildReference(parent, hi, hnew, pos);
			hi = hnew;
			LOG.debug("Applied simplifyConstantCumsum (line "+hi.getBeginLine()+").");
		}
		return hi;
	}
	
	private static Hop pushdownBinaryOperationOnDiag(Hop parent, Hop hi, int pos) 
	{
		//diag(X)*7 --> diag(X*7) in order to (1) reduce required memory for b(*) and
		//(2) in order to make the binary operation more efficient (dense vector vs sparse matrix)
		if( HopRewriteUtils.isBinary(hi, OpOp2.MULT) )
		{
			Hop left = hi.getInput(0);
			Hop right = hi.getInput(1);
			
			boolean applyLeft = false;
			boolean applyRight = false;
			
			//left input is diag
			if( left instanceof ReorgOp && ((ReorgOp)left).getOp()==ReOrgOp.DIAG
				&& left.getParent().size()==1 //binary op only parent
				&& left.getInput(0).getDim2()==1 //col vector
				&& right.getDataType() == DataType.SCALAR )
			{
				applyLeft = true;
			}
			else if( right instanceof ReorgOp && ((ReorgOp)right).getOp()==ReOrgOp.DIAG
					&& right.getParent().size()==1 //binary op only parent
					&& right.getInput(0).getDim2()==1 //col vector
					&& left.getDataType() == DataType.SCALAR )
			{
				applyRight = true;
			}
			
			//perform actual rewrite
			if( applyLeft || applyRight )
			{
				//remove all parent links to binary op (since we want to reorder
				//we cannot just look at the current parent)
				List<Hop> parents = new ArrayList<>(hi.getParent());
				List<Integer> parentspos = new ArrayList<>(); 
				for(Hop lparent : parents) {
					int lpos = HopRewriteUtils.getChildReferencePos(lparent, hi);
					HopRewriteUtils.removeChildReferenceByPos(lparent, hi, lpos);
					parentspos.add(lpos);
				}
				
				//rewire binop-diag-input into diag-binop-input
				if( applyLeft ) {
					Hop input = left.getInput(0);
					HopRewriteUtils.removeChildReferenceByPos(hi, left, 0);
					HopRewriteUtils.removeChildReferenceByPos(left, input, 0);
					HopRewriteUtils.addChildReference(left, hi, 0);
					HopRewriteUtils.addChildReference(hi, input, 0);
					hi.refreshSizeInformation();
					hi = left;
				}
				else if ( applyRight ) {
					Hop input = right.getInput(0);
					HopRewriteUtils.removeChildReferenceByPos(hi, right, 1);
					HopRewriteUtils.removeChildReferenceByPos(right, input, 0);
					HopRewriteUtils.addChildReference(right, hi, 0);
					HopRewriteUtils.addChildReference(hi, input, 1);	
					hi.refreshSizeInformation();
					hi = right;
				}
				
				//relink all parents to the diag operation
				for( int i=0; i<parents.size(); i++ ) {
					Hop lparent = parents.get(i);
					int lpos = parentspos.get(i);
					HopRewriteUtils.addChildReference(lparent, hi, lpos);
				}	
				
				LOG.debug("Applied pushdownBinaryOperationOnDiag.");
			}
		}
		
		return hi;
	}
	
	/**
	 * patterns: sum(A+B)->sum(A)+sum(B); sum(A-B)->sum(A)-sum(B)
	 * 
	 * @param parent the parent high-level operator
	 * @param hi high-level operator
	 * @param pos position
	 * @return high-level operator
	 */
	private static Hop pushdownSumOnAdditiveBinary(Hop parent, Hop hi, int pos)
	{
		//all patterns headed by full sum over binary operation
		if(    hi instanceof AggUnaryOp //full sum root over binaryop
				&& ((AggUnaryOp)hi).getDirection()==Direction.RowCol
				&& ((AggUnaryOp)hi).getOp() == AggOp.SUM
				&& hi.getInput(0) instanceof BinaryOp
				&& hi.getInput(0).getParent().size()==1 ) //single parent
		{
			BinaryOp bop = (BinaryOp) hi.getInput(0);
			Hop left = bop.getInput(0);
			Hop right = bop.getInput(1);

			if( left.getDataType() == DataType.MATRIX
					&& right.getDataType() == DataType.MATRIX )
			{
				OpOp2 applyOp = ( bop.getOp() == OpOp2.PLUS //pattern a: sum(A+B)->sum(A)+sum(B)
						|| bop.getOp() == OpOp2.MINUS )     //pattern b: sum(A-B)->sum(A)-sum(B)
						? bop.getOp() : null;

				if( applyOp != null ) {
					if (HopRewriteUtils.isEqualSize(left, right)) {
						//create new subdag sum(A) bop sum(B) for equal-sized matrices
						AggUnaryOp sum1 = HopRewriteUtils.createSum(left);
						AggUnaryOp sum2 = HopRewriteUtils.createSum(right);
						BinaryOp newBin = HopRewriteUtils.createBinary(sum1, sum2, applyOp);
						//rewire new subdag
						HopRewriteUtils.replaceChildReference(parent, hi, newBin, pos);
						HopRewriteUtils.cleanupUnreferenced(hi, bop);

						hi = newBin;

						LOG.debug("Applied pushdownSumOnAdditiveBinary (line "+hi.getBeginLine()+").");
					}
					// Check if right operand is a vector (has dimension of 1 in either rows or columns)
					else if (right.getDim1() == 1 || right.getDim2() == 1) {
						AggUnaryOp sum1 = HopRewriteUtils.createSum(left);
						AggUnaryOp sum2 = HopRewriteUtils.createSum(right);

						// Row vector case (1 x n)
						if (right.getDim1() == 1) {
							// Create nrow(A) operation using dimensions
							UnaryOp nRows = HopRewriteUtils.createUnary(left, OpOp1.NROW);
							BinaryOp scaledSum = HopRewriteUtils.createBinary(nRows, sum2, OpOp2.MULT);
							BinaryOp newBin = HopRewriteUtils.createBinary(sum1, scaledSum, applyOp);
							//rewire new subdag
							HopRewriteUtils.replaceChildReference(parent, hi, newBin, pos);
							HopRewriteUtils.cleanupUnreferenced(hi, bop);

							hi = newBin;

							LOG.debug("Applied pushdownSumOnAdditiveBinary with row vector (line "+hi.getBeginLine()+").");
						}
						// Column vector case (n x 1)
						else if (right.getDim2() == 1) {
							// Create ncol(A) operation using dimensions
							UnaryOp nCols = HopRewriteUtils.createUnary(left, OpOp1.NCOL);
							BinaryOp scaledSum = HopRewriteUtils.createBinary(nCols, sum2, OpOp2.MULT);
							BinaryOp newBin = HopRewriteUtils.createBinary(sum1, scaledSum, applyOp);
							//rewire new subdag
							HopRewriteUtils.replaceChildReference(parent, hi, newBin, pos);
							HopRewriteUtils.cleanupUnreferenced(hi, bop);

							hi = newBin;

							LOG.debug("Applied pushdownSumOnAdditiveBinary with column vector (line "+hi.getBeginLine()+").");
						}
					}
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
	 * 4) sumSq (X - U %*% t(V)) (no weighting sumSq)
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
	 * @param parent parent high-level operator
	 * @param hi high-level operator
	 * @param pos position
	 * @return high-level operator
	 */
	private static Hop simplifyWeightedSquaredLoss(Hop parent, Hop hi, int pos) 
	{
		//NOTE: there might be also a general simplification without custom operator
		//via (X-UVt)^2 -> X^2 - 2X*UVt + UVt^2
		Hop hnew = null;
		boolean appliedPattern = false;
		
		if( HopRewriteUtils.isAggUnaryOp(hi, AggOp.SUM, Direction.RowCol) //all patterns rooted by sum()
			&& hi.getInput(0) instanceof BinaryOp  //all patterns subrooted by binary op
			&& hi.getInput(0).getDim2() > 1  )     //not applied for vector-vector mult
		{
			BinaryOp bop = (BinaryOp) hi.getInput(0);
			
			//Pattern 1) sum (W * (X - U %*% t(V)) ^ 2) (post weighting)
			//alternative pattern: sum (W * (U %*% t(V) - X) ^ 2)
			if( bop.getOp()==OpOp2.MULT && HopRewriteUtils.isBinary(bop.getInput(1), OpOp2.POW)
				&& bop.getInput(0).getDataType()==DataType.MATRIX
				&& HopRewriteUtils.isEqualSize(bop.getInput(0), bop.getInput(1)) //prevent mv
				&& HopRewriteUtils.isLiteralOfValue(bop.getInput(1).getInput(1), 2) )
			{
				Hop W = bop.getInput(0);
				Hop tmp = bop.getInput(1).getInput(0); //(X - U %*% t(V))
				
				if( HopRewriteUtils.isBinary(tmp, OpOp2.MINUS)
					&& HopRewriteUtils.isEqualSize(tmp.getInput(0), tmp.getInput(1)) //prevent mv
					&& tmp.getInput(0).getDataType() == DataType.MATRIX )
				{
					//a) sum (W * (X - U %*% t(V)) ^ 2)
					int uvIndex = -1;
					if( tmp.getInput(1) instanceof AggBinaryOp  //ba gurantees matrices
						&& HopRewriteUtils.isSingleBlock(tmp.getInput(1).getInput(0),true)) { //BLOCKSIZE CONSTRAINT
						uvIndex = 1;
					}
					//b) sum (W * (U %*% t(V) - X) ^ 2)
					else if(tmp.getInput(0) instanceof AggBinaryOp  //ba gurantees matrices
						&& HopRewriteUtils.isSingleBlock(tmp.getInput(0).getInput(0),true)) { //BLOCKSIZE CONSTRAINT
						uvIndex = 0;
					}

					if( uvIndex >= 0 ) { //rewrite match
						Hop X = tmp.getInput().get((uvIndex==0)?1:0); 
						Hop U = tmp.getInput().get(uvIndex).getInput(0);
						Hop V = tmp.getInput().get(uvIndex).getInput(1);
						V = !HopRewriteUtils.isTransposeOperation(V) ?
							HopRewriteUtils.createTranspose(V) : V.getInput(0);

						//handle special case of post_nz
						if( HopRewriteUtils.isNonZeroIndicator(W, X) ){
							W = new LiteralOp(1);
						}

						//construct quaternary hop
						hnew = new QuaternaryOp(hi.getName(), DataType.SCALAR,
							ValueType.FP64, OpOp4.WSLOSS, X, U, V, W, true);
						HopRewriteUtils.setOutputParametersForScalar(hnew);

						appliedPattern = true;
						LOG.debug("Applied simplifyWeightedSquaredLoss1"+uvIndex+" (line "+hi.getBeginLine()+")");
					}
				}
			}
			
			//Pattern 2) sum ((X - W * (U %*% t(V))) ^ 2) (pre weighting)
			//alternative pattern: sum ((W * (U %*% t(V)) - X) ^ 2)
			if( !appliedPattern
				&& bop.getOp()==OpOp2.POW && HopRewriteUtils.isLiteralOfValue(bop.getInput(1), 2)
				&& HopRewriteUtils.isBinary(bop.getInput(0), OpOp2.MINUS)
				&& HopRewriteUtils.isEqualMatrixSize((BinaryOp)bop.getInput(0)))
			{
				Hop lleft = bop.getInput(0).getInput(0); 
				Hop lright = bop.getInput(0).getInput(1); 

				//a) sum ((X - W * (U %*% t(V))) ^ 2)
				int wuvIndex = -1;
				if( lright instanceof BinaryOp && lright.getInput(1) instanceof AggBinaryOp ){
					wuvIndex = 1;
				}
				//b) sum ((W * (U %*% t(V)) - X) ^ 2)
				else if( lleft instanceof BinaryOp && lleft.getInput(1) instanceof AggBinaryOp ){
					wuvIndex = 0;
				}

				if( wuvIndex >= 0 ) //rewrite match
				{
					Hop X = bop.getInput(0).getInput().get((wuvIndex==0)?1:0);
					Hop tmp = bop.getInput(0).getInput().get(wuvIndex); //(W * (U %*% t(V)))
					
					if( ((BinaryOp)tmp).getOp()==OpOp2.MULT
						&& tmp.getInput(0).getDataType() == DataType.MATRIX
						&& HopRewriteUtils.isEqualSize(tmp.getInput(0), tmp.getInput(1)) //prevent mv
						&& HopRewriteUtils.isSingleBlock(tmp.getInput(1).getInput(0),true)) //BLOCKSIZE CONSTRAINT
					{
						Hop W = tmp.getInput(0);
						Hop U = tmp.getInput(1).getInput(0);
						Hop V = tmp.getInput(1).getInput(1);
						V = !HopRewriteUtils.isTransposeOperation(V) ?
							HopRewriteUtils.createTranspose(V) : V.getInput(0);
						hnew = new QuaternaryOp(hi.getName(), DataType.SCALAR,
							ValueType.FP64, OpOp4.WSLOSS, X, U, V, W, false);
						HopRewriteUtils.setOutputParametersForScalar(hnew);
						appliedPattern = true;
						LOG.debug("Applied simplifyWeightedSquaredLoss2"+wuvIndex+" (line "+hi.getBeginLine()+")");	
					}
				}
			}

			//Pattern 3) sum ((X - (U %*% t(V))) ^ 2) (no weighting)
			//alternative pattern: sum (((U %*% t(V)) - X) ^ 2)
			if( !appliedPattern
				&& bop.getOp()==OpOp2.POW && HopRewriteUtils.isLiteralOfValue(bop.getInput(1), 2)
				&& HopRewriteUtils.isBinary(bop.getInput(0), OpOp2.MINUS) 
				&& HopRewriteUtils.isEqualMatrixSize((BinaryOp)bop.getInput(0))) //prevent mv
			{
				Hop lleft = bop.getInput(0).getInput(0);
				Hop lright = bop.getInput(0).getInput(1);

				//a) sum ((X - (U %*% t(V))) ^ 2)
				int uvIndex = -1;
				if( lright instanceof AggBinaryOp //ba guarantees matrices
					&& HopRewriteUtils.isSingleBlock(lright.getInput(0),true) ) { //BLOCKSIZE CONSTRAINT
					uvIndex = 1;
				}
				//b) sum (((U %*% t(V)) - X) ^ 2)
				else if( lleft instanceof AggBinaryOp //ba guarantees matrices
						&& HopRewriteUtils.isSingleBlock(lleft.getInput(0),true) ) { //BLOCKSIZE CONSTRAINT
					uvIndex = 0;
				}

				if( uvIndex >= 0 ) { //rewrite match
					Hop X = bop.getInput(0).getInput().get((uvIndex==0)?1:0);
					Hop tmp = bop.getInput(0).getInput().get(uvIndex); //(U %*% t(V))
					Hop W = new LiteralOp(1); //no weighting 
					Hop U = tmp.getInput(0);
					Hop V = tmp.getInput(1);
					V = !HopRewriteUtils.isTransposeOperation(V) ?
						HopRewriteUtils.createTranspose(V) : V.getInput(0);
					hnew = new QuaternaryOp(hi.getName(), DataType.SCALAR,
						ValueType.FP64, OpOp4.WSLOSS, X, U, V, W, false);
					HopRewriteUtils.setOutputParametersForScalar(hnew);
					appliedPattern = true;
					
					LOG.debug("Applied simplifyWeightedSquaredLoss3"+uvIndex+" (line "+hi.getBeginLine()+")");
				}
			}
		}
		
		//Pattern 4) sumSq (X - U %*% t(V)) (no weighting)
		//alternative pattern: sumSq (U %*% t(V) - X)
		if( !appliedPattern
			&& HopRewriteUtils.isAggUnaryOp(hi, AggOp.SUM_SQ, Direction.RowCol)
			&& HopRewriteUtils.isBinary(hi.getInput(0), OpOp2.MINUS) 
			&& HopRewriteUtils.isEqualMatrixSize((BinaryOp)hi.getInput(0))) //prevent mv
		{
			Hop lleft = hi.getInput(0).getInput(0);
			Hop lright = hi.getInput(0).getInput(1);

			//a) sumSq (X - U %*% t(V))
			int uvIndex = -1;
			if( lright instanceof AggBinaryOp //ba guarantees matrices
				&& HopRewriteUtils.isSingleBlock(lright.getInput(0),true) ) { //BLOCKSIZE CONSTRAINT
				uvIndex = 1;
			}
			//b) sumSq (U %*% t(V) - X)
			else if( lleft instanceof AggBinaryOp //ba guarantees matrices
					&& HopRewriteUtils.isSingleBlock(lleft.getInput(0),true) ) { //BLOCKSIZE CONSTRAINT
				uvIndex = 0;
			}

			if( uvIndex >= 0 ) { //rewrite match
				Hop X = hi.getInput(0).getInput().get((uvIndex==0)?1:0);
				Hop tmp = hi.getInput(0).getInput().get(uvIndex); //(U %*% t(V))
				Hop W = new LiteralOp(1); //no weighting 
				Hop U = tmp.getInput(0);
				Hop V = tmp.getInput(1);
				V = !HopRewriteUtils.isTransposeOperation(V) ?
					HopRewriteUtils.createTranspose(V) : V.getInput(0);
				hnew = new QuaternaryOp(hi.getName(), DataType.SCALAR,
					ValueType.FP64, OpOp4.WSLOSS, X, U, V, W, false);
				HopRewriteUtils.setOutputParametersForScalar(hnew);
				appliedPattern = true;
				
				LOG.debug("Applied simplifyWeightedSquaredLoss4"+uvIndex+" (line "+hi.getBeginLine()+")");
			}
		}
		
		//relink new hop into original position
		if( hnew != null ) {
			HopRewriteUtils.replaceChildReference(parent, hi, hnew, pos);
			hi = hnew;
		}
		
		return hi;
	}
	
	private static Hop simplifyWeightedSigmoidMMChains(Hop parent, Hop hi, int pos) 
	{
		Hop hnew = null;
		
		if( HopRewriteUtils.isBinary(hi, OpOp2.MULT) //all patterns subrooted by W *
			&& hi.getDim2() > 1       //not applied for vector-vector mult
			&& HopRewriteUtils.isEqualSize(hi.getInput(0), hi.getInput(1)) //prevent mv
			&& hi.getInput(0).getDataType()==DataType.MATRIX 
			&& hi.getInput(1) instanceof UnaryOp ) //sigmoid/log
		{
			UnaryOp uop = (UnaryOp) hi.getInput(1);
			boolean appliedPattern = false;
			
			//Pattern 1) W * sigmoid(Y%*%t(X)) (basic)
			if(    uop.getOp() == OpOp1.SIGMOID 
				&& uop.getInput(0) instanceof AggBinaryOp
				&& HopRewriteUtils.isSingleBlock(uop.getInput(0).getInput(0),true) )
			{
				Hop W = hi.getInput(0); 
				Hop Y = uop.getInput(0).getInput(0);
				Hop tX = uop.getInput(0).getInput(1);
				
				if( !HopRewriteUtils.isTransposeOperation(tX) ) { 
					tX = HopRewriteUtils.createTranspose(tX);
				}
				else
					tX = tX.getInput(0);
				
				hnew = new QuaternaryOp(hi.getName(), DataType.MATRIX, ValueType.FP64, 
						  OpOp4.WSIGMOID, W, Y, tX, false, false);
				hnew.setBlocksize(W.getBlocksize());
				hnew.refreshSizeInformation();
				
				appliedPattern = true;
				LOG.debug("Applied simplifyWeightedSigmoid1 (line "+hi.getBeginLine()+")");	
			}
			
			//Pattern 2) W * sigmoid(-(Y%*%t(X))) (minus)
			if(    !appliedPattern 
				&& uop.getOp() == OpOp1.SIGMOID 
				&& HopRewriteUtils.isBinary(uop.getInput(0), OpOp2.MINUS)
				&& uop.getInput(0).getInput(0) instanceof LiteralOp
				&& HopRewriteUtils.getDoubleValueSafe(
				   (LiteralOp)uop.getInput(0).getInput(0))==0
				&& uop.getInput(0).getInput(1) instanceof AggBinaryOp
				&& HopRewriteUtils.isSingleBlock(uop.getInput(0).getInput(1).getInput(0),true))
			{
				Hop W = hi.getInput(0); 
				Hop Y = uop.getInput(0).getInput(1).getInput(0);
				Hop tX = uop.getInput(0).getInput(1).getInput(1);
				
				if( !HopRewriteUtils.isTransposeOperation(tX) ) { 
					tX = HopRewriteUtils.createTranspose(tX);
				}
				else
					tX = tX.getInput(0);
				
				hnew = new QuaternaryOp(hi.getName(), DataType.MATRIX, ValueType.FP64, 
						  OpOp4.WSIGMOID, W, Y, tX, false, true);
				hnew.setBlocksize(W.getBlocksize());
				hnew.refreshSizeInformation();
				
				appliedPattern = true;
				LOG.debug("Applied simplifyWeightedSigmoid2 (line "+hi.getBeginLine()+")");	
			}
			
			//Pattern 3) W * log(sigmoid(Y%*%t(X))) (log)			
			if(    !appliedPattern 
				&& uop.getOp() == OpOp1.LOG
				&& HopRewriteUtils.isUnary(uop.getInput(0), OpOp1.SIGMOID) 
				&& uop.getInput(0).getInput(0) instanceof AggBinaryOp
				&& HopRewriteUtils.isSingleBlock(uop.getInput(0).getInput(0).getInput(0),true) )
			{
				Hop W = hi.getInput(0); 
				Hop Y = uop.getInput(0).getInput(0).getInput(0);
				Hop tX = uop.getInput(0).getInput(0).getInput(1);
				
				if( !HopRewriteUtils.isTransposeOperation(tX) ) { 
					tX = HopRewriteUtils.createTranspose(tX);
				}
				else
					tX = tX.getInput(0);
				
				hnew = new QuaternaryOp(hi.getName(), DataType.MATRIX, ValueType.FP64, 
						  OpOp4.WSIGMOID, W, Y, tX, true, false);
				hnew.setBlocksize(W.getBlocksize());
				hnew.refreshSizeInformation();
				
				appliedPattern = true;
				LOG.debug("Applied simplifyWeightedSigmoid3 (line "+hi.getBeginLine()+")");	
			}			
			
			//Pattern 4) W * log(sigmoid(-(Y%*%t(X)))) (log_minus)
			if(    !appliedPattern 
				&& uop.getOp() == OpOp1.LOG
				&& HopRewriteUtils.isUnary(uop.getInput(0), OpOp1.SIGMOID) 
				&& HopRewriteUtils.isBinary(uop.getInput(0).getInput(0), OpOp2.MINUS) )
			{
				BinaryOp bop = (BinaryOp) uop.getInput(0).getInput(0);
				
				if(    bop.getInput(0) instanceof LiteralOp
					&& HopRewriteUtils.getDoubleValueSafe((LiteralOp)bop.getInput(0))==0
					&& bop.getInput(1) instanceof AggBinaryOp
					&& HopRewriteUtils.isSingleBlock(bop.getInput(1).getInput(0),true))
				{
					Hop W = hi.getInput(0); 
					Hop Y = bop.getInput(1).getInput(0);
					Hop tX = bop.getInput(1).getInput(1);
					
					if( !HopRewriteUtils.isTransposeOperation(tX) ) { 
						tX = HopRewriteUtils.createTranspose(tX);
					}
					else
						tX = tX.getInput(0);
					
					hnew = new QuaternaryOp(hi.getName(), DataType.MATRIX, ValueType.FP64, 
							  OpOp4.WSIGMOID, W, Y, tX, true, true);
					hnew.setBlocksize(W.getBlocksize());
					hnew.refreshSizeInformation();
					
					appliedPattern = true;
					LOG.debug("Applied simplifyWeightedSigmoid4 (line "+hi.getBeginLine()+")");	
				}
			}
		}
		
		//relink new hop into original position
		if( hnew != null ) {
			HopRewriteUtils.replaceChildReference(parent, hi, hnew, pos);
			hi = hnew;
		}
		
		return hi;
	}

	private static Hop simplifyWeightedDivMM(Hop parent, Hop hi, int pos) {
		Hop hnew = null;
		boolean appliedPattern = false;
		
		//left/right patterns rooted by 'ab - b(div)' or 'ab - b(mult)'
		//note: we do not rewrite t(X)%*%(w*(X%*%v)) where w and v are vectors (see mmchain ops) 
		if( HopRewriteUtils.isMatrixMultiply(hi)  
			&& (hi.getInput(0) instanceof BinaryOp
			&& HopRewriteUtils.isValidOp(((BinaryOp)hi.getInput(0)).getOp(), LOOKUP_VALID_WDIVMM_BINARY)
			|| hi.getInput(1) instanceof BinaryOp 
			&& hi.getDim2() > 1 //not applied for vector-vector mult
			&& HopRewriteUtils.isValidOp(((BinaryOp)hi.getInput(1)).getOp(), LOOKUP_VALID_WDIVMM_BINARY)) ) 
		{
			Hop left = hi.getInput(0);
			Hop right = hi.getInput(1);
			
			//Pattern 1) t(U) %*% (W/(U%*%t(V)))
			//alternative pattern: t(U) %*% (W*(U%*%t(V)))
			if( right instanceof BinaryOp && HopRewriteUtils.isValidOp(((BinaryOp)right).getOp(),LOOKUP_VALID_WDIVMM_BINARY)	
				&& HopRewriteUtils.isEqualSize(right.getInput(0), right.getInput(1)) //prevent mv
				&& HopRewriteUtils.isOuterProductLikeMM(right.getInput(1))
				&& HopRewriteUtils.isSingleBlock(right.getInput(1).getInput(0),true) ) //BLOCKSIZE CONSTRAINT
			{
				Hop W = right.getInput(0); 
				Hop U = right.getInput(1).getInput(0);
				Hop V = right.getInput(1).getInput(1);
				
				if( HopRewriteUtils.isTransposeOfItself(left, U) ) 
				{
					if( !HopRewriteUtils.isTransposeOperation(V) )
						V = HopRewriteUtils.createTranspose(V);
					else 
						V = V.getInput(0);
					
					boolean mult = ((BinaryOp)right).getOp() == OpOp2.MULT;
					hnew = new QuaternaryOp(hi.getName(), DataType.MATRIX, ValueType.FP64, 
							  OpOp4.WDIVMM, W, U, V, new LiteralOp(-1), 1, mult, false);
					hnew.setBlocksize(W.getBlocksize());
					hnew.refreshSizeInformation();
					
					//add output transpose for efficient target indexing (redundant t() removed by other rewrites)
					hnew = HopRewriteUtils.createTranspose(hnew);
					
					appliedPattern = true;
					LOG.debug("Applied simplifyWeightedDivMM1 (line "+hi.getBeginLine()+")");
				}
			}	
			
			//Pattern 1e) t(U) %*% (W/(U%*%t(V) + x))
			if( !appliedPattern
				&& HopRewriteUtils.isBinary(right, LOOKUP_VALID_WDIVMM_BINARY[1]) //DIV
				&& HopRewriteUtils.isEqualSize(right.getInput(0), right.getInput(1)) //prevent mv
				&& HopRewriteUtils.isBinary(right.getInput(1), OpOp2.PLUS)
				&& right.getInput(1).getInput(1).getDataType() == DataType.SCALAR
				&& HopRewriteUtils.isOuterProductLikeMM(right.getInput(1).getInput(0))
				&& HopRewriteUtils.isSingleBlock(right.getInput(1).getInput(0).getInput(0),true) ) //BLOCKSIZE CONSTRAINT
			{
				Hop W = right.getInput(0); 
				Hop U = right.getInput(1).getInput(0).getInput(0);
				Hop V = right.getInput(1).getInput(0).getInput(1);
				Hop X = right.getInput(1).getInput(1);
				
				if( HopRewriteUtils.isTransposeOfItself(left, U) ) 
				{
					if( !HopRewriteUtils.isTransposeOperation(V) )
						V = HopRewriteUtils.createTranspose(V);
					else 
						V = V.getInput(0);
					
					hnew = new QuaternaryOp(hi.getName(), DataType.MATRIX, ValueType.FP64, 
							  OpOp4.WDIVMM, W, U, V, X, 3, false, false); // 3=>DIV_LEFT_EPS
					hnew.setBlocksize(W.getBlocksize());
					hnew.refreshSizeInformation();
					
					//add output transpose for efficient target indexing (redundant t() removed by other rewrites)
					hnew = HopRewriteUtils.createTranspose(hnew);
					
					appliedPattern = true;
					LOG.debug("Applied simplifyWeightedDivMM1e (line "+hi.getBeginLine()+")");
				}
			}	
			
			//Pattern 2) (W/(U%*%t(V))) %*% V
			//alternative pattern: (W*(U%*%t(V))) %*% V
			if( !appliedPattern
				&& left instanceof BinaryOp && HopRewriteUtils.isValidOp(((BinaryOp)left).getOp(), LOOKUP_VALID_WDIVMM_BINARY)	
				&& HopRewriteUtils.isEqualSize(left.getInput(0), left.getInput(1)) //prevent mv
				&& HopRewriteUtils.isOuterProductLikeMM(left.getInput(1))
				&& HopRewriteUtils.isSingleBlock(left.getInput(1).getInput(0),true) ) //BLOCKSIZE CONSTRAINT
			{
				Hop W = left.getInput(0); 
				Hop U = left.getInput(1).getInput(0);
				Hop V = left.getInput(1).getInput(1);
				
				if( HopRewriteUtils.isTransposeOfItself(right, V) ) 
				{
					if( !HopRewriteUtils.isTransposeOperation(V) )
						V = right;
					else 
						V = V.getInput(0);
					
					boolean mult = ((BinaryOp)left).getOp() == OpOp2.MULT;
					hnew = new QuaternaryOp(hi.getName(), DataType.MATRIX, ValueType.FP64, 
							  OpOp4.WDIVMM, W, U, V, new LiteralOp(-1), 2, mult, false);
					hnew.setBlocksize(W.getBlocksize());
					hnew.refreshSizeInformation();
					
					appliedPattern = true;
					LOG.debug("Applied simplifyWeightedDivMM2 (line "+hi.getBeginLine()+")");
				}
			}
			
			//Pattern 2e) (W/(U%*%t(V) + x)) %*% V
			if( !appliedPattern
				&& HopRewriteUtils.isBinary(left, LOOKUP_VALID_WDIVMM_BINARY[1]) //DIV
				&& HopRewriteUtils.isEqualSize(left.getInput(0), left.getInput(1)) //prevent mv
				&& HopRewriteUtils.isBinary(left.getInput(1), OpOp2.PLUS)
				&& left.getInput(1).getInput(1).getDataType() == DataType.SCALAR
				&& HopRewriteUtils.isOuterProductLikeMM(left.getInput(1).getInput(0))
				&& HopRewriteUtils.isSingleBlock(left.getInput(1).getInput(0).getInput(0),true) ) //BLOCKSIZE CONSTRAINT
			{
				Hop W = left.getInput(0); 
				Hop U = left.getInput(1).getInput(0).getInput(0);
				Hop V = left.getInput(1).getInput(0).getInput(1);
				Hop X = left.getInput(1).getInput(1);
				
				if( HopRewriteUtils.isTransposeOfItself(right, V) ) 
				{
					if( !HopRewriteUtils.isTransposeOperation(V) )
						V = right;
					else 
						V = V.getInput(0);
					
					hnew = new QuaternaryOp(hi.getName(), DataType.MATRIX, ValueType.FP64, 
							  OpOp4.WDIVMM, W, U, V, X, 4, false, false); // 4=>DIV_RIGHT_EPS
					hnew.setBlocksize(W.getBlocksize());
					hnew.refreshSizeInformation();
					
					appliedPattern = true;
					LOG.debug("Applied simplifyWeightedDivMM2e (line "+hi.getBeginLine()+")");	
				}
			}
			
			//Pattern 3) t(U) %*% ((X!=0)*(U%*%t(V)-X))
			if( !appliedPattern
				&& HopRewriteUtils.isBinary(right, LOOKUP_VALID_WDIVMM_BINARY[0]) //MULT
				&& HopRewriteUtils.isBinary(right.getInput(1), OpOp2.MINUS)	
				&& HopRewriteUtils.isOuterProductLikeMM(right.getInput(1).getInput(0))
				&& right.getInput(1).getInput(1).getDataType() == DataType.MATRIX
				&& HopRewriteUtils.isSingleBlock(right.getInput(1).getInput(0).getInput(0),true) ) //BLOCKSIZE CONSTRAINT
			{
				Hop W = right.getInput(0); 
				Hop U = right.getInput(1).getInput(0).getInput(0);
				Hop V = right.getInput(1).getInput(0).getInput(1);
				Hop X = right.getInput(1).getInput(1);
				
				if(    HopRewriteUtils.isNonZeroIndicator(W, X)        //W-X constraint
				    && HopRewriteUtils.isTransposeOfItself(left, U) )  //t(U)-U constraint
				{
					if( !HopRewriteUtils.isTransposeOperation(V) )
						V = HopRewriteUtils.createTranspose(V);
					else 
						V = V.getInput(0);
					
					hnew = new QuaternaryOp(hi.getName(), DataType.MATRIX, ValueType.FP64, 
							  OpOp4.WDIVMM, X, U, V, new LiteralOp(-1), 1, true, true);
					hnew.setBlocksize(W.getBlocksize());
					hnew.refreshSizeInformation();
					
					//add output transpose for efficient target indexing (redundant t() removed by other rewrites)
					hnew = HopRewriteUtils.createTranspose(hnew);
					
					appliedPattern = true;
					LOG.debug("Applied simplifyWeightedDivMM3 (line "+hi.getBeginLine()+")");
				}
			}	
			
			//Pattern 4) ((X!=0)*(U%*%t(V)-X)) %*% V
			if( !appliedPattern
				&& HopRewriteUtils.isBinary(left, LOOKUP_VALID_WDIVMM_BINARY[0]) //MULT
				&& HopRewriteUtils.isBinary(left.getInput(1), OpOp2.MINUS)
				&& HopRewriteUtils.isOuterProductLikeMM(left.getInput(1).getInput(0))
				&& left.getInput(1).getInput(1).getDataType() == DataType.MATRIX
				&& HopRewriteUtils.isSingleBlock(left.getInput(1).getInput(0).getInput(0),true) ) //BLOCKSIZE CONSTRAINT
			{
				Hop W = left.getInput(0); 
				Hop U = left.getInput(1).getInput(0).getInput(0);
				Hop V = left.getInput(1).getInput(0).getInput(1);
				Hop X = left.getInput(1).getInput(1);
				
				if(    HopRewriteUtils.isNonZeroIndicator(W, X)        //W-X constraint
					&& HopRewriteUtils.isTransposeOfItself(right, V) )  //V-t(V) constraint
				{
					if( !HopRewriteUtils.isTransposeOperation(V) )
						V = right;
					else 
						V = V.getInput(0);
					
					hnew = new QuaternaryOp(hi.getName(), DataType.MATRIX, ValueType.FP64, 
							  OpOp4.WDIVMM, X, U, V, new LiteralOp(-1), 2, true, true);
					hnew.setBlocksize(W.getBlocksize());
					hnew.refreshSizeInformation();
					
					appliedPattern = true;
					LOG.debug("Applied simplifyWeightedDivMM4 (line "+hi.getBeginLine()+")");
				}
			}
			
			//Pattern 5) t(U) %*% (W*(U%*%t(V)-X))
			if( !appliedPattern
				&& HopRewriteUtils.isBinary(right, LOOKUP_VALID_WDIVMM_BINARY[0]) //MULT
				&& HopRewriteUtils.isBinary(right.getInput(1), OpOp2.MINUS)	
				&& HopRewriteUtils.isOuterProductLikeMM(right.getInput(1).getInput(0))
				&& right.getInput(1).getInput(1).getDataType() == DataType.MATRIX
				&& HopRewriteUtils.isSingleBlock(right.getInput(1).getInput(0).getInput(0),true) ) //BLOCKSIZE CONSTRAINT
			{
				Hop W = right.getInput(0); 
				Hop U = right.getInput(1).getInput(0).getInput(0);
				Hop V = right.getInput(1).getInput(0).getInput(1);
				Hop X = right.getInput(1).getInput(1);
				
				if( HopRewriteUtils.isTransposeOfItself(left, U) )  //t(U)-U constraint
				{
					if( !HopRewriteUtils.isTransposeOperation(V) )
						V = HopRewriteUtils.createTranspose(V);
					else 
						V = V.getInput(0);
					
					//note: x and w exchanged compared to patterns 1-4, 7
					hnew = new QuaternaryOp(hi.getName(), DataType.MATRIX, ValueType.FP64, 
							  OpOp4.WDIVMM, W, U, V, X, 1, true, true);
					hnew.setBlocksize(W.getBlocksize());
					hnew.refreshSizeInformation();
					
					//add output transpose for efficient target indexing (redundant t() removed by other rewrites)
					hnew = HopRewriteUtils.createTranspose(hnew);
					
					appliedPattern = true;
					LOG.debug("Applied simplifyWeightedDivMM5 (line "+hi.getBeginLine()+")");
				}
			}	
			
			//Pattern 6) (W*(U%*%t(V)-X)) %*% V
			if( !appliedPattern
				&& HopRewriteUtils.isBinary(left, LOOKUP_VALID_WDIVMM_BINARY[0]) //MULT	
				&& HopRewriteUtils.isBinary(left.getInput(1), OpOp2.MINUS)	
				&& HopRewriteUtils.isOuterProductLikeMM(left.getInput(1).getInput(0))
				&& left.getInput(1).getInput(1).getDataType() == DataType.MATRIX
				&& HopRewriteUtils.isSingleBlock(left.getInput(1).getInput(0).getInput(0),true) ) //BLOCKSIZE CONSTRAINT
			{
				Hop W = left.getInput(0); 
				Hop U = left.getInput(1).getInput(0).getInput(0);
				Hop V = left.getInput(1).getInput(0).getInput(1);
				Hop X = left.getInput(1).getInput(1);
				
				if( HopRewriteUtils.isTransposeOfItself(right, V) )  //V-t(V) constraint
				{
					if( !HopRewriteUtils.isTransposeOperation(V) )
						V = right;
					else 
						V = V.getInput(0);
					
					//note: x and w exchanged compared to patterns 1-4, 7
					hnew = new QuaternaryOp(hi.getName(), DataType.MATRIX, ValueType.FP64, 
							  OpOp4.WDIVMM, W, U, V, X, 2, true, true);
					hnew.setBlocksize(W.getBlocksize());
					hnew.refreshSizeInformation();
					
					appliedPattern = true;
					LOG.debug("Applied simplifyWeightedDivMM6 (line "+hi.getBeginLine()+")");
				}
			}
		}
		
		//Pattern 7) (W*(U%*%t(V)))
		if( !appliedPattern
			&& HopRewriteUtils.isBinary(hi, LOOKUP_VALID_WDIVMM_BINARY[0]) //MULT	
			&& HopRewriteUtils.isEqualSize(hi.getInput(0), hi.getInput(1)) //prevent mv
			&& hi.getDim2() > 1 //not applied for vector-vector mult
			&& hi.getInput(0).getDataType() == DataType.MATRIX 
			&& hi.getInput(0).getDim2() > hi.getInput(0).getBlocksize()
			&& HopRewriteUtils.isOuterProductLikeMM(hi.getInput(1))
			&& (((AggBinaryOp) hi.getInput(1)).checkMapMultChain() == ChainType.NONE || hi.getInput(1).getInput(1).getDim2() > 1) //no mmchain
			&& HopRewriteUtils.isSingleBlock(hi.getInput(1).getInput(0),true) ) //BLOCKSIZE CONSTRAINT
		{
			Hop W = hi.getInput(0); 
			Hop U = hi.getInput(1).getInput(0);
			Hop V = hi.getInput(1).getInput(1);
			
			//for this basic pattern, we're more conservative and only apply wdivmm if
			//W is sparse and U/V unknown or dense; or if U/V are dense
			if( (HopRewriteUtils.isSparse(W) && !HopRewriteUtils.isSparse(U) && !HopRewriteUtils.isSparse(V))
				|| (HopRewriteUtils.isDense(U) && HopRewriteUtils.isDense(V)) ) {
				V = !HopRewriteUtils.isTransposeOperation(V) ?
					HopRewriteUtils.createTranspose(V) : V.getInput(0);
				hnew = new QuaternaryOp(hi.getName(), DataType.MATRIX, ValueType.FP64, 
					OpOp4.WDIVMM, W, U, V, new LiteralOp(-1), 0, true, false);
				hnew.setBlocksize(W.getBlocksize());
				hnew.refreshSizeInformation();
				appliedPattern = true;
				LOG.debug("Applied simplifyWeightedDivMM7 (line "+hi.getBeginLine()+")");
			}
		}
		
		//relink new hop into original position
		if( hnew != null ) {
			HopRewriteUtils.replaceChildReference(parent, hi, hnew, pos);
			hi = hnew;
		}
		
		return hi;
	}

	private static Hop simplifyWeightedCrossEntropy(Hop parent, Hop hi, int pos) 
	{
		Hop hnew = null;
		boolean appliedPattern = false;
		
		if( hi instanceof AggUnaryOp && ((AggUnaryOp)hi).getDirection()==Direction.RowCol
			&& ((AggUnaryOp)hi).getOp() == AggOp.SUM     //pattern rooted by sum()
			&& hi.getInput(0) instanceof BinaryOp  //pattern subrooted by binary op
			&& hi.getInput(0).getDim2() > 1   )    //not applied for vector-vector mult
		{
			BinaryOp bop = (BinaryOp) hi.getInput(0);
			Hop left = bop.getInput(0);
			Hop right = bop.getInput(1);
			
			//Pattern 1) sum( X * log(U %*% t(V)))
			if( bop.getOp()==OpOp2.MULT && left.getDataType()==DataType.MATRIX
				&& HopRewriteUtils.isEqualSize(left, right)  //prevent mb
				&& HopRewriteUtils.isUnary(right, OpOp1.LOG)
				&& right.getInput(0) instanceof AggBinaryOp  //ba gurantees matrices
				&& HopRewriteUtils.isSingleBlock(right.getInput(0).getInput(0),true)) //BLOCKSIZE CONSTRAINT
			{
				Hop X = left; 
				Hop U = right.getInput(0).getInput(0);
				Hop V = right.getInput(0).getInput(1);
				
				if( !HopRewriteUtils.isTransposeOperation(V) )
					V = HopRewriteUtils.createTranspose(V);
				else 
					V = V.getInput(0);
					
				hnew = new QuaternaryOp(hi.getName(), DataType.SCALAR, ValueType.FP64, OpOp4.WCEMM, X, U, V,
						new LiteralOp(0.0), 0, false, false);
				hnew.setBlocksize(X.getBlocksize());
				appliedPattern = true;
				
				LOG.debug("Applied simplifyWeightedCEMM (line "+hi.getBeginLine()+")");
			}
			
			//Pattern 2) sum( X * log(U %*% t(V) + eps))
			if( !appliedPattern
				&& bop.getOp()==OpOp2.MULT && left.getDataType()==DataType.MATRIX
				&& HopRewriteUtils.isEqualSize(left, right)
				&& HopRewriteUtils.isUnary(right, OpOp1.LOG)
				&& HopRewriteUtils.isBinary(right.getInput(0), OpOp2.PLUS)
				&& right.getInput(0).getInput(0) instanceof AggBinaryOp
				&& right.getInput(0).getInput(1) instanceof LiteralOp
				&& right.getInput(0).getInput(1).getDataType() == DataType.SCALAR
				&& HopRewriteUtils.isSingleBlock(right.getInput(0).getInput(0).getInput(0),true))
			{
				Hop X = left; 
				Hop U = right.getInput(0).getInput(0).getInput(0);
				Hop V = right.getInput(0).getInput(0).getInput(1);
				Hop eps = right.getInput(0).getInput(1);
				
				if( !HopRewriteUtils.isTransposeOperation(V) )
					V = HopRewriteUtils.createTranspose(V);
				else 
					V = V.getInput(0);
					
				hnew = new QuaternaryOp(hi.getName(), DataType.SCALAR, ValueType.FP64, 
						OpOp4.WCEMM, X, U, V, eps, 1, false, false); // 1 => BASIC_EPS
				hnew.setBlocksize(X.getBlocksize());
					
				LOG.debug("Applied simplifyWeightedCEMMEps (line "+hi.getBeginLine()+")");
			}
		}
		
		//relink new hop into original position
		if( hnew != null ) {
			HopRewriteUtils.replaceChildReference(parent, hi, hnew, pos);
			hi = hnew;
		}
		
		return hi;
	}
	
	private static Hop simplifyWeightedUnaryMM(Hop parent, Hop hi, int pos) {
		Hop hnew = null;
		boolean appliedPattern = false;
		
		//Pattern 1) (W*uop(U%*%t(V)))
		if( hi instanceof BinaryOp && HopRewriteUtils.isValidOp(((BinaryOp)hi).getOp(),LOOKUP_VALID_WDIVMM_BINARY)	
			&& HopRewriteUtils.isEqualSize(hi.getInput(0), hi.getInput(1)) //prevent mv
			&& hi.getDim2() > 1 //not applied for vector-vector mult
			&& hi.getInput(0).getDataType() == DataType.MATRIX 
			&& hi.getInput(0).getDim2() > hi.getInput(0).getBlocksize()
			&& hi.getInput(1) instanceof UnaryOp
			&& HopRewriteUtils.isValidOp(((UnaryOp)hi.getInput(1)).getOp(), LOOKUP_VALID_WUMM_UNARY) 
			&& hi.getInput(1).getInput(0) instanceof AggBinaryOp
			&& HopRewriteUtils.isSingleBlock(hi.getInput(1).getInput(0).getInput(0),true) ) //BLOCKSIZE CONSTRAINT			
		{
			Hop W = hi.getInput(0); 
			Hop U = hi.getInput(1).getInput(0).getInput(0);
			Hop V = hi.getInput(1).getInput(0).getInput(1);
			boolean mult = ((BinaryOp)hi).getOp()==OpOp2.MULT;
			OpOp1 op = ((UnaryOp)hi.getInput(1)).getOp();
			
			if( !HopRewriteUtils.isTransposeOperation(V) )
				V = HopRewriteUtils.createTranspose(V);
			else
				V = V.getInput(0);
				
			hnew = new QuaternaryOp(hi.getName(), DataType.MATRIX, ValueType.FP64, 
				OpOp4.WUMM, W, U, V, mult, op, null);
			hnew.setBlocksize(W.getBlocksize());
			hnew.refreshSizeInformation();
			
			appliedPattern = true;
			LOG.debug("Applied simplifyWeightedUnaryMM1 (line "+hi.getBeginLine()+")");
		}

		//Pattern 2.7) (W*(U%*%t(V))*2 or 2*(W*(U%*%t(V))
		if( !appliedPattern
				&& hi instanceof BinaryOp && HopRewriteUtils.isValidOp(((BinaryOp)hi).getOp(), OpOp2.MULT)
				&& (HopRewriteUtils.isLiteralOfValue(hi.getInput(0), 2)
					|| HopRewriteUtils.isLiteralOfValue(hi.getInput(1), 2)))
		{
			final Hop nl; // non-literal
			if( hi.getInput(0) instanceof LiteralOp ) {
				nl = hi.getInput(1);
			} else {
				nl = hi.getInput(0);
			}

			if (       HopRewriteUtils.isBinary(nl, OpOp2.MULT)
					&& nl.getParent().size()==1 // ensure no foreign parents
					&& HopRewriteUtils.isEqualSize(nl.getInput(0), nl.getInput(1)) //prevent mv
					&& nl.getDim2() > 1 //not applied for vector-vector mult
					&& nl.getInput(0).getDataType() == DataType.MATRIX
					&& nl.getInput(0).getDim2() > nl.getInput(0).getBlocksize()
					&& HopRewriteUtils.isOuterProductLikeMM(nl.getInput(1))
					&& (((AggBinaryOp) nl.getInput(1)).checkMapMultChain() == ChainType.NONE || nl.getInput(1).getInput(1).getDim2() > 1) //no mmchain
					&& HopRewriteUtils.isSingleBlock(nl.getInput(1).getInput(0),true) )
			{
				final Hop W = nl.getInput(0);
				final Hop U = nl.getInput(1).getInput(0);
				Hop V = nl.getInput(1).getInput(1);
				if( !HopRewriteUtils.isTransposeOperation(V) )
					V = HopRewriteUtils.createTranspose(V);
				else
					V = V.getInput(0);

				hnew = new QuaternaryOp(hi.getName(), DataType.MATRIX, ValueType.FP64,
						OpOp4.WUMM, W, U, V, true, null, OpOp2.MULT);
				hnew.setBlocksize(W.getBlocksize());
				hnew.refreshSizeInformation();

				appliedPattern = true;
				LOG.debug("Applied simplifyWeightedUnaryMM2.7 (line "+hi.getBeginLine()+")");
			}
		}
		
		//Pattern 2) (W*sop(U%*%t(V),c)) for known sop translating to unary ops
		if( !appliedPattern
			&& hi instanceof BinaryOp && HopRewriteUtils.isValidOp(((BinaryOp)hi).getOp(),LOOKUP_VALID_WDIVMM_BINARY)
			&& HopRewriteUtils.isEqualSize(hi.getInput(0), hi.getInput(1)) //prevent mv
			&& hi.getDim2() > 1 //not applied for vector-vector mult
			&& hi.getInput(0).getDataType() == DataType.MATRIX
			&& hi.getInput(0).getDim2() > hi.getInput(0).getBlocksize()
			&& hi.getInput(1) instanceof BinaryOp
			&& HopRewriteUtils.isValidOp(((BinaryOp)hi.getInput(1)).getOp(), LOOKUP_VALID_WUMM_BINARY) )
		{
			Hop left = hi.getInput(1).getInput(0);
			Hop right = hi.getInput(1).getInput(1);
			Hop abop = null;
			
			//pattern 2a) matrix-scalar operations
			if( right.getDataType()==DataType.SCALAR && right instanceof LiteralOp
				&& HopRewriteUtils.getDoubleValue((LiteralOp)right)==2 //pow2, mult2
				&& left instanceof AggBinaryOp
				&& HopRewriteUtils.isSingleBlock(left.getInput(0),true) ) //BLOCKSIZE CONSTRAINT
			{
				abop = left;
			}
			//pattern 2b) scalar-matrix operations
			else if( left.getDataType()==DataType.SCALAR && left instanceof LiteralOp 
				&& HopRewriteUtils.getDoubleValue((LiteralOp)left)==2 //mult2
				&& ((BinaryOp)hi.getInput(1)).getOp() == OpOp2.MULT
				&& right instanceof AggBinaryOp
				&& HopRewriteUtils.isSingleBlock(right.getInput(0),true) ) //BLOCKSIZE CONSTRAINT
			{
				abop = right;
			}
			
			if( abop != null ) {
				Hop W = hi.getInput(0); 
				Hop U = abop.getInput(0);
				Hop V = abop.getInput(1);
				boolean mult = ((BinaryOp)hi).getOp()==OpOp2.MULT;
				OpOp2 op = ((BinaryOp)hi.getInput(1)).getOp();
				
				if( !HopRewriteUtils.isTransposeOperation(V) )
					V = HopRewriteUtils.createTranspose(V);
				else
					V = V.getInput(0);
					
				hnew = new QuaternaryOp(hi.getName(), DataType.MATRIX, ValueType.FP64, 
						  OpOp4.WUMM, W, U, V, mult, null, op);
				hnew.setBlocksize(W.getBlocksize());
				hnew.refreshSizeInformation();
				
				appliedPattern = true;
				LOG.debug("Applied simplifyWeightedUnaryMM2 (line "+hi.getBeginLine()+")");	
			}
		}
		
		
		//relink new hop into original position
		if( hnew != null ) {
			HopRewriteUtils.replaceChildReference(parent, hi, hnew, pos);
			hi = hnew;
		}
		
		return hi;
	}
	
	/**
	 * NOTE: dot-product-sum could be also applied to sum(a*b). However, we 
	 * restrict ourselfs to sum(a^2) and transitively sum(a*a) since a general mm
	 * a%*%b on MR can be also counter-productive (e.g., MMCJ) while tsmm is always 
	 * beneficial. 
	 * 
	 * @param parent parent high-level operator
	 * @param hi high-level operator
	 * @param pos position
	 * @return high-level operator
	 */
	private static Hop simplifyDotProductSum(Hop parent, Hop hi, int pos) {
		//sum(v^2)/sum(v1*v2) --> as.scalar(t(v)%*%v) in order to exploit tsmm vector dotproduct 
		//w/o materialization of intermediates
		if( hi instanceof AggUnaryOp && ((AggUnaryOp)hi).getOp()==AggOp.SUM //sum
			&& ((AggUnaryOp)hi).getDirection()==Direction.RowCol //full aggregate
			&& hi.getInput(0).getDim2() == 1 ) //vector (for correctness)
		{
			Hop baLeft = null;
			Hop baRight = null;
			
			Hop hi2 = hi.getInput(0); //check for ^2 w/o multiple consumers
			//check for sum(v^2), might have been rewritten from sum(v*v)
			if( HopRewriteUtils.isBinary(hi2, OpOp2.POW)
				&& hi2.getInput(1) instanceof LiteralOp 
				&& HopRewriteUtils.getDoubleValue((LiteralOp)hi2.getInput(1))==2
				&& hi2.getParent().size() == 1 ) //no other consumer than sum
			{
				Hop input = hi2.getInput(0);
				baLeft = input;
				baRight = input;
			}
			//check for sum(v1*v2), but prevent to rewrite sum(v1*v2*v3) which is later compiled into a ta+* lop
			else if( HopRewriteUtils.isBinary(hi2, OpOp2.MULT, 1) //no other consumer than sum
					&& hi2.getInput(0).getDim2()==1 && hi2.getInput(1).getDim2()==1
					&& hi2.getInput(0).isMatrix() && hi2.getInput(1).isMatrix()
					&& !HopRewriteUtils.isBinary(hi2.getInput(0), OpOp2.MULT)
					&& !HopRewriteUtils.isBinary(hi2.getInput(1), OpOp2.MULT)
					&& ( !ALLOW_SUM_PRODUCT_REWRITES
						|| !(  HopRewriteUtils.isBinary(hi2.getInput(0), OpOp2.POW)     // do not rewrite (A^2)*B
							&& hi2.getInput(0).getInput(1) instanceof LiteralOp   // let tak+* handle it
							&& ((LiteralOp)hi2.getInput(0).getInput(1)).getLongValue() == 2 ))
					&& ( !ALLOW_SUM_PRODUCT_REWRITES
						|| !( HopRewriteUtils.isBinary(hi2.getInput(1), OpOp2.POW)      // do not rewrite B*(A^2)
							&& hi2.getInput(1).getInput(1) instanceof LiteralOp   // let tak+* handle it
							&& ((LiteralOp)hi2.getInput(1).getInput(1)).getLongValue() == 2 ))
					)
			{
				baLeft = hi2.getInput(0);
				baRight = hi2.getInput(1);
			}
			
			//perform actual rewrite (if necessary)
			if( baLeft != null && baRight != null  )
			{
				//create new operator chain
				ReorgOp trans = HopRewriteUtils.createTranspose(baLeft);
				AggBinaryOp mmult = HopRewriteUtils.createMatrixMultiply(trans, baRight);
				UnaryOp cast = HopRewriteUtils.createUnary(mmult, OpOp1.CAST_AS_SCALAR);
				
				//rehang new subdag under parent node
				HopRewriteUtils.replaceChildReference(parent, hi, cast, pos);
				HopRewriteUtils.cleanupUnreferenced(hi, hi2);
				
				hi = cast;
				
				LOG.debug("Applied simplifyDotProductSum (line "+hi.getBeginLine()+").");
			}
		}
		
		return hi;
	}

	/**
	 * Replace SUM(X^2) with a fused SUM_SQ(X) HOP.
	 *
	 * @param parent Parent HOP for which hi is an input.
	 * @param hi Current HOP for potential rewrite.
	 * @param pos Position of hi in parent's list of inputs.
	 *
	 * @return Either hi or the rewritten HOP replacing it.
	 */
	private static Hop fuseSumSquared(Hop parent, Hop hi, int pos) {
		// if SUM
		if (hi instanceof AggUnaryOp && ((AggUnaryOp) hi).getOp() == AggOp.SUM) {
			Hop sumInput = hi.getInput(0);

			// if input to SUM is POW(X,2), and no other consumers of the POW(X,2) HOP
			if( HopRewriteUtils.isBinary(sumInput, OpOp2.POW)
					&& sumInput.getInput(1) instanceof LiteralOp
					&& HopRewriteUtils.getDoubleValue((LiteralOp) sumInput.getInput(1)) == 2
					&& sumInput.getParent().size() == 1) {
				Hop x = sumInput.getInput(0);

				// if X is NOT a column vector
				if (x.getDim2() > 1) {
					// perform rewrite from SUM(POW(X,2)) to SUM_SQ(X)
					Direction dir = ((AggUnaryOp) hi).getDirection();
					AggUnaryOp sumSq = HopRewriteUtils.createAggUnaryOp(x, AggOp.SUM_SQ, dir);
					HopRewriteUtils.replaceChildReference(parent, hi, sumSq, pos);
					HopRewriteUtils.cleanupUnreferenced(hi, sumInput);
					hi = sumSq;
					
					LOG.debug("Applied fuseSumSquared (line " +hi.getBeginLine()+").");
				}
			}
		}
		return hi;
	}
	
	private static Hop fuseAxpyBinaryOperationChain(Hop parent, Hop hi, int pos) 
	{
		//patterns: (a) X + s*Y -> X +* sY, (b) s*Y+X -> X +* sY, (c) X - s*Y -> X -* sY
		if( hi instanceof BinaryOp && !((BinaryOp) hi).isOuter()
			&& (((BinaryOp)hi).getOp()==OpOp2.PLUS || ((BinaryOp)hi).getOp()==OpOp2.MINUS) )
		{
			BinaryOp bop = (BinaryOp) hi;
			Hop left = bop.getInput(0);
			Hop right = bop.getInput(1);
			Hop ternop = null;
			
			//pattern (a) X + s*Y -> X +* sY
			if( bop.getOp() == OpOp2.PLUS && left.getDataType()==DataType.MATRIX 
				&& HopRewriteUtils.isScalarMatrixBinaryMult(right)
				&& HopRewriteUtils.isEqualSize(left, right)
				&& right.getParent().size() == 1 )           //single consumer s*Y
			{
				Hop smid = right.getInput().get( (right.getInput(0).getDataType()==DataType.SCALAR) ? 0 : 1); 
				Hop mright = right.getInput().get( (right.getInput(0).getDataType()==DataType.SCALAR) ? 1 : 0);
				ternop = (smid instanceof LiteralOp && HopRewriteUtils.getDoubleValueSafe((LiteralOp)smid)==0) ? 
						left : HopRewriteUtils.createTernary(left, smid, mright, OpOp3.PLUS_MULT);
				LOG.debug("Applied fuseAxpyBinaryOperationChain1. (line " +hi.getBeginLine()+")");
			}
			//pattern (b) s*Y + X -> X +* sY
			else if( bop.getOp() == OpOp2.PLUS && right.getDataType()==DataType.MATRIX 
				&& HopRewriteUtils.isScalarMatrixBinaryMult(left)
				&& HopRewriteUtils.isEqualSize(left, right)
				&& left.getParent().size() == 1 )            //single consumer s*Y
			{
				Hop smid = left.getInput().get( (left.getInput(0).getDataType()==DataType.SCALAR) ? 0 : 1); 
				Hop mright = left.getInput().get( (left.getInput(0).getDataType()==DataType.SCALAR) ? 1 : 0);
				ternop = (smid instanceof LiteralOp && HopRewriteUtils.getDoubleValueSafe((LiteralOp)smid)==0) ? 
						right : HopRewriteUtils.createTernary(right, smid, mright, OpOp3.PLUS_MULT);
				LOG.debug("Applied fuseAxpyBinaryOperationChain2. (line " +hi.getBeginLine()+")");
			}
			//pattern (c) X - s*Y -> X -* sY
			else if( bop.getOp() == OpOp2.MINUS && left.getDataType()==DataType.MATRIX 
				&& HopRewriteUtils.isScalarMatrixBinaryMult(right)
				&& HopRewriteUtils.isEqualSize(left, right)
				&& right.getParent().size() == 1 )           //single consumer s*Y
			{
				Hop smid = right.getInput().get( (right.getInput(0).getDataType()==DataType.SCALAR) ? 0 : 1); 
				Hop mright = right.getInput().get( (right.getInput(0).getDataType()==DataType.SCALAR) ? 1 : 0);
				ternop = (smid instanceof LiteralOp && HopRewriteUtils.getDoubleValueSafe((LiteralOp)smid)==0) ? 
						left : HopRewriteUtils.createTernary(left, smid, mright, OpOp3.MINUS_MULT);
				LOG.debug("Applied fuseAxpyBinaryOperationChain3. (line " +hi.getBeginLine()+")");
			}
			
			//rewire parent-child operators if rewrite applied
			if( ternop != null ) {
				if (right.getForcedExecType() == Types.ExecType.FED)
					ternop.setForcedExecType(Types.ExecType.FED);
				HopRewriteUtils.replaceChildReference(parent, hi, ternop, pos);
				hi = ternop;
			}
		}
		
		return hi;
	}
	
	private static Hop simplifyEmptyBinaryOperation(Hop parent, Hop hi, int pos) 
	{
		if( hi instanceof BinaryOp ) //b(?) X Y
		{
			BinaryOp bop = (BinaryOp) hi;
			Hop left = hi.getInput(0);
			Hop right = hi.getInput(1);
		
			if( left.getDataType()==DataType.MATRIX && right.getDataType()==DataType.MATRIX )
			{
				Hop hnew = null;
				
				//NOTE: these rewrites of binary cell operations need to be aware that right is 
				//potentially a vector but the result is of the size of left
				
				boolean notBinaryMV = HopRewriteUtils.isNotMatrixVectorBinaryOperation(bop);
				
				switch( bop.getOp() ){
					//X * Y -> matrix(0,nrow(X),ncol(X));
					case MULT: {
						if( HopRewriteUtils.isEmpty(left) ) //empty left and size known
							hnew = HopRewriteUtils.createDataGenOp(left, left, 0);
						else if( HopRewriteUtils.isEmpty(right) //empty right and right not a vector
								&& right.getDim1()>1 && right.getDim2()>1  ) {
							hnew = HopRewriteUtils.createDataGenOp(right, right, 0);
						}
						else if( HopRewriteUtils.isEmpty(right) )//empty right and right potentially a vector
							hnew = HopRewriteUtils.createDataGenOp(left, left, 0);
						break;
					}
					case PLUS: {
						if( HopRewriteUtils.isEmpty(left) && HopRewriteUtils.isEmpty(right) ) //empty left/right and size known
							hnew = HopRewriteUtils.createDataGenOp(left, left, 0);
						else if( HopRewriteUtils.isEmpty(left) && notBinaryMV ) //empty left
							hnew = right;
						else if( HopRewriteUtils.isEmpty(right) ) //empty right
							hnew = left;
						break;
					}
					case MINUS: {
						if( HopRewriteUtils.isEmpty(left) && notBinaryMV ) { //empty left
							HopRewriteUtils.removeChildReference(hi, left);
							HopRewriteUtils.addChildReference(hi, new LiteralOp(0), 0);
							hnew = hi;
						}
						else if( HopRewriteUtils.isEmpty(right) ) //empty and size known
							hnew = left;
						break;
					}
					default:
						//do nothing (hnew = null)
				}
				
				if( hnew != null ) {
					//create datagen and add it to parent
					HopRewriteUtils.replaceChildReference(parent, hi, hnew, pos);
					hi = hnew;
					
					LOG.debug("Applied simplifyEmptyBinaryOperation (line "+hi.getBeginLine()+").");
				}
			}
		}
		
		return hi;
	}
	
	/**
	 * This is rewrite tries to reorder minus operators from inputs of matrix
	 * multiply to its output because the output is (except for outer products)
	 * usually significantly smaller. Furthermore, this rewrite is a precondition
	 * for the important hops-lops rewrite of transpose-matrixmult if the transpose
	 * is hidden under the minus. 
	 * 
	 * NOTE: in this rewrite we need to modify the links to all parents because we 
	 * remove existing links of subdags and hence affect all consumers.
	 * 
	 * @param parent the parent high-level operator
	 * @param hi high-level operator
	 * @param pos position
	 * @return high-level operator
	 */
	private static Hop reorderMinusMatrixMult(Hop parent, Hop hi, int pos) 
	{
		if( HopRewriteUtils.isMatrixMultiply(hi) ) //X%*%Y
		{
			Hop hileft = hi.getInput(0);
			Hop hiright = hi.getInput(1);
			
			if( HopRewriteUtils.isBinary(hileft, OpOp2.MINUS)  //X=-Z
				&& hileft.getInput(0) instanceof LiteralOp 
				&& HopRewriteUtils.getDoubleValue((LiteralOp)hileft.getInput(0))==0.0 
				&& hi.dimsKnown() && hileft.getInput(1).dimsKnown()   //size comparison
				&& HopRewriteUtils.compareSize(hi, hileft.getInput(1)) < 0 ) 
			{
				Hop hi2 = hileft.getInput(1);
				
				//remove link from matrixmult to minus
				HopRewriteUtils.removeChildReference(hi, hileft);
				
				//get old parents (before creating minus over matrix mult)
				List<Hop> parents = new ArrayList<>(hi.getParent());
				
				//create new operators 
				BinaryOp minus = HopRewriteUtils.createBinary(new LiteralOp(0), hi, OpOp2.MINUS);
				
				//rehang minus under all parents
				for( Hop p : parents ) {
					int ix = HopRewriteUtils.getChildReferencePos(p, hi);
					HopRewriteUtils.removeChildReference(p, hi);
					HopRewriteUtils.addChildReference(p, minus, ix);
				}
				
				//rehang child of minus under matrix mult
				HopRewriteUtils.addChildReference(hi, hi2, 0);
				
				//cleanup if only consumer of minus
				HopRewriteUtils.cleanupUnreferenced(hileft);
				
				hi = minus;
				
				LOG.debug("Applied reorderMinusMatrixMult (line "+hi.getBeginLine()+").");
			}
			else if( HopRewriteUtils.isBinary(hiright, OpOp2.MINUS)  //X=-Z
					&& hiright.getInput(0) instanceof LiteralOp 
					&& HopRewriteUtils.getDoubleValue((LiteralOp)hiright.getInput(0))==0.0
					&& hi.dimsKnown() && hiright.getInput(1).dimsKnown()     //size comparison
					&& HopRewriteUtils.compareSize(hi, hiright.getInput(1)) < 0 ) 
			{
				Hop hi2 = hiright.getInput(1);
				
				//remove link from matrixmult to minus
				HopRewriteUtils.removeChildReference(hi, hiright);
				
				//get old parents (before creating minus over matrix mult)
				List<Hop> parents = new ArrayList<>(hi.getParent());
				
				//create new operators 
				BinaryOp minus = HopRewriteUtils.createBinary(new LiteralOp(0), hi, OpOp2.MINUS);
				
				//rehang minus under all parents
				for( Hop p : parents ) {
					int ix = HopRewriteUtils.getChildReferencePos(p, hi);
					HopRewriteUtils.removeChildReference(p, hi);
					HopRewriteUtils.addChildReference(p, minus, ix);
				}
				
				//rehang child of minus under matrix mult
				HopRewriteUtils.addChildReference(hi, hi2, 1);
				
				//cleanup if only consumer of minus
				HopRewriteUtils.cleanupUnreferenced(hiright);
				
				hi = minus;
				
				LOG.debug("Applied reorderMinusMatrixMult (line "+hi.getBeginLine()+").");
			}
		}
		
		return hi;
	}


	private static Hop simplifySumMatrixMult(Hop parent, Hop hi, int pos)
	{
		//sum(A%*%B) -> sum(t(colSums(A))*rowSums(B)), later rewritten to dot-product
		//colSums(A%*%B) -> colSums(A)%*%B
		//rowSums(A%*%B) -> A%*%rowSums(B)
		//-- if not dot product, not applied since aggregate removed
		//-- if sum not the only consumer, not applied to prevent redundancy 
		if( hi instanceof AggUnaryOp && ((AggUnaryOp)hi).getOp()==AggOp.SUM  //sum
			&& hi.getInput(0) instanceof AggBinaryOp                   //A%*%B
			&& (hi.getInput(0).getDim1()>1 || hi.getInput(0).getDim2()>1) //not dot product
			&& hi.getInput(0).getParent().size()==1 )     //not multiple consumers of matrix mult
		{
			Hop hi2 = hi.getInput(0);
			Hop left = hi2.getInput(0);
			Hop right = hi2.getInput(1);
				
			//remove link from parent to matrix mult
			HopRewriteUtils.removeChildReference(hi, hi2);
				
			//create new operators
			Hop root = null;
			//pattern: sum(A%*%B) -> sum(t(colSums(A))*rowSums(B)), later rewritten to dot-product
			if( ((AggUnaryOp)hi).getDirection() == Direction.RowCol ) {
				AggUnaryOp colSum = HopRewriteUtils.createAggUnaryOp(left, AggOp.SUM, Direction.Col);
				ReorgOp trans = HopRewriteUtils.createTranspose(colSum);
				AggUnaryOp rowSum = HopRewriteUtils.createAggUnaryOp(right, AggOp.SUM, Direction.Row);
				root = HopRewriteUtils.createBinary(trans, rowSum, OpOp2.MULT);
				LOG.debug("Applied simplifySumMatrixMult RC.");
			}
			//colSums(A%*%B) -> colSums(A)%*%B
			else if( ((AggUnaryOp)hi).getDirection() == Direction.Col ) {
				AggUnaryOp colSum = HopRewriteUtils.createAggUnaryOp(left, AggOp.SUM, Direction.Col);
				root = HopRewriteUtils.createMatrixMultiply(colSum, right);
				LOG.debug("Applied simplifySumMatrixMult C.");
			}
			//rowSums(A%*%B) -> A%*%rowSums(B)
			else if( ((AggUnaryOp)hi).getDirection() == Direction.Row ) {
				AggUnaryOp rowSum = HopRewriteUtils.createAggUnaryOp(right, AggOp.SUM, Direction.Row);
				root = HopRewriteUtils.createMatrixMultiply(left, rowSum);
				LOG.debug("Applied simplifySumMatrixMult R.");
			}
			
			//rehang new subdag under current node (keep hi intact)
			HopRewriteUtils.addChildReference(hi, root, 0);				
			hi.refreshSizeInformation();
			
			//cleanup if only consumer of intermediate
			HopRewriteUtils.cleanupUnreferenced(hi2);
		}
		
		return hi;
	}
	
	private static Hop simplifyScalarMVBinaryOperation(Hop hi) 
	{
		if( hi instanceof BinaryOp && ((BinaryOp)hi).supportsMatrixScalarOperations() //e.g., X * s
			&& hi.getInput(0).getDataType()==DataType.MATRIX 
			&& hi.getInput(1).getDataType()==DataType.MATRIX )	
		{
			Hop right = hi.getInput(1);
			
			//X * s -> X * as.scalar(s)
			if( HopRewriteUtils.isDimsKnown(right) && right.getDim1()==1 && right.getDim2()==1 ) //scalar right
			{
				//remove link to right child and introduce cast
				UnaryOp cast = HopRewriteUtils.createUnary(right, OpOp1.CAST_AS_SCALAR);
				HopRewriteUtils.replaceChildReference(hi, right, cast, 1);			
				
				LOG.debug("Applied simplifyScalarMVBinaryOperation.");
			}
		}
		
		return hi;
	}
	
	private static Hop simplifyNnzComputation(Hop parent, Hop hi, int pos) 
	{
		//sum(ppred(X,0,"!=")) -> literal(nnz(X)), if nnz known		
		if(    hi instanceof AggUnaryOp && ((AggUnaryOp)hi).getOp()==AggOp.SUM  //sum
			&& ((AggUnaryOp)hi).getDirection() == Direction.RowCol	            //full aggregate
			&& HopRewriteUtils.isBinary(hi.getInput(0), OpOp2.NOTEQUAL) )
		{
			Hop ppred = hi.getInput(0);
			Hop X = null;
			if(    ppred.getInput(0) instanceof LiteralOp 
				&& HopRewriteUtils.getDoubleValue((LiteralOp)ppred.getInput(0))==0 )
			{
				X = ppred.getInput(1);
			}
			else if(   ppred.getInput(1) instanceof LiteralOp 
					&& HopRewriteUtils.getDoubleValue((LiteralOp)ppred.getInput(1))==0 )
			{
				X = ppred.getInput(0);
			}
		
			//apply rewrite if known nnz 
			if( X != null && X.getNnz() > 0 ){
				Hop hnew = new LiteralOp(X.getNnz());
				HopRewriteUtils.replaceChildReference(parent, hi, hnew, pos);
				HopRewriteUtils.cleanupUnreferenced(hi);
				hi = hnew;
				
				LOG.debug("Applied simplifyNnzComputation.");	
			}
		}
		
		return hi;
	}
	
	private static Hop simplifyNrowNcolComputation(Hop parent, Hop hi, int pos) 
	{
		//nrow(X) -> literal(nrow(X)), ncol(X) -> literal(ncol(X)), if respective dims known
		//(this rewrite aims to remove unnecessary data dependencies to X which trigger computation
		//even if the intermediate is otherwise not required, e.g., when part of a fused operator)
		if( hi instanceof UnaryOp ) 
		{
			if( ((UnaryOp)hi).getOp()==OpOp1.NROW && hi.getInput(0).rowsKnown() ) {
				Hop hnew = new LiteralOp(hi.getInput(0).getDim1());
				HopRewriteUtils.replaceChildReference(parent, hi, hnew, pos, false);
				HopRewriteUtils.cleanupUnreferenced(hi);
				LOG.debug("Applied simplifyNrowComputation nrow("+hi.getHopID()+") -> "
					+ hnew.getName()+" (line "+hi.getBeginLine()+").");
				hi = hnew;
			}
			else if( ((UnaryOp)hi).getOp()==OpOp1.NCOL && hi.getInput(0).colsKnown() ) {
				Hop hnew = new LiteralOp(hi.getInput(0).getDim2());
				HopRewriteUtils.replaceChildReference(parent, hi, hnew, pos, false);
				HopRewriteUtils.cleanupUnreferenced(hi);
				LOG.debug("Applied simplifyNcolComputation ncol("+hi.getHopID()+") -> "
					+ hnew.getName()+" (line "+hi.getBeginLine()+").");
				hi = hnew;
			}
		}
		
		return hi;
	}
	
	private static Hop simplifyTableSeqExpand(Hop parent, Hop hi, int pos) 
	{
		//pattern: table(seq(1,nrow(v)), v, nrow(v), m) -> rexpand(v, max=m, dir=row, ignore=false, cast=true)
		//note: this rewrite supports both left/right sequence 
		if(    hi instanceof TernaryOp && hi.getInput().size()==6 //table without weights 
			&& HopRewriteUtils.isLiteralOfValue(hi.getInput(2), 1) ) //i.e., weight of 1
		{
			Hop first = hi.getInput(0);
			Hop second = hi.getInput(1);
			
			//pattern a: table(seq(1,nrow(v)), v, nrow(v), m, 1)
			if( HopRewriteUtils.isBasic1NSequence(first, second, true) 
				&& HopRewriteUtils.isSizeExpressionOf(hi.getInput(3), second, true) )
			{
				//setup input parameter hops
				LinkedHashMap<String,Hop> args = new LinkedHashMap<>();
				args.put("target", second);
				args.put("max", hi.getInput().get(4));
				args.put("dir", new LiteralOp("cols"));
				args.put("ignore", new LiteralOp(false));
				args.put("cast", new LiteralOp(true));
			
				//create new hop
				ParameterizedBuiltinOp pbop = HopRewriteUtils
					.createParameterizedBuiltinOp(second, args, ParamBuiltinOp.REXPAND);
				HopRewriteUtils.replaceChildReference(parent, hi, pbop, pos);
				HopRewriteUtils.cleanupUnreferenced(hi);
				hi = pbop;
				
				LOG.debug("Applied simplifyTableSeqExpand1 (line "+hi.getBeginLine()+")");	
			}
			//pattern b: table(v, seq(1,nrow(v)), m, nrow(v))
			else if( HopRewriteUtils.isBasic1NSequence(second, first, true)
				&& HopRewriteUtils.isSizeExpressionOf(hi.getInput().get(4), first, true) )
			{
				//setup input parameter hops
				LinkedHashMap<String,Hop> args = new LinkedHashMap<>();
				args.put("target", first);
				args.put("max", hi.getInput(3));
				args.put("dir", new LiteralOp("rows"));
				args.put("ignore", new LiteralOp(false));
				args.put("cast", new LiteralOp(true));
			
				//create new hop
				ParameterizedBuiltinOp pbop = HopRewriteUtils
					.createParameterizedBuiltinOp(first, args, ParamBuiltinOp.REXPAND);
				HopRewriteUtils.replaceChildReference(parent, hi, pbop, pos);
				HopRewriteUtils.cleanupUnreferenced(hi);
				hi = pbop;
				
				LOG.debug("Applied simplifyTableSeqExpand2 (line "+hi.getBeginLine()+")");	
			}
		}
	
		return hi;
	}
	
	private static Hop foldMultipleMinMaxOperations(Hop hi) 
	{
		if( (HopRewriteUtils.isBinary(hi, OpOp2.MIN, OpOp2.MAX, OpOp2.PLUS, OpOp2.MULT)
			|| HopRewriteUtils.isNary(hi, OpOpN.MIN, OpOpN.MAX, OpOpN.PLUS, OpOpN.MULT))
			&& hi.getValueType() != ValueType.STRING //exclude string concat
			&& HopRewriteUtils.isNotMatrixVectorBinaryOperation(hi))
		{
			OpOp2 bop = (hi instanceof BinaryOp) ? ((BinaryOp)hi).getOp() :
				OpOp2.valueOf(((NaryOp)hi).getOp().name());
			OpOpN nop = (hi instanceof NaryOp) ? ((NaryOp)hi).getOp() :
				OpOpN.valueOf(((BinaryOp)hi).getOp().name());
			
			boolean converged = false;
			while( !converged ) {
				//get first matching min/max
				Hop first = hi.getInput().stream()
					.filter(h -> HopRewriteUtils.isBinary(h, bop) || HopRewriteUtils.isNary(h, nop))
					.findFirst().orElse(null);
				
				//replace current op with new nary min/max
				final Hop lhi = hi;
				if( first != null && first.getParent().size()==1
					&& first.getInput().stream().allMatch(c -> c.getDataType()==DataType.SCALAR 
						|| HopRewriteUtils.isEqualSize(lhi, c))) {
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
					LOG.debug("Applied foldMultipleMinMaxPlusMultOperations (line "+hi.getBeginLine()+").");
				}
				else {
					converged = true;
				}
			}
		}
		
		return hi;
	}

	private static Hop simplyfyMMCBindZeroVector(Hop parent, Hop hi, int pos) {

		// cbind((X %*% Y), matrix(0, nrow(X), 1)) ->
		// X %*% (cbind(Y, matrix(0, nrow(Y), 1)))
		// if nRows of x is larger than nRows of y
		// rewrite used in MLogReg first level loop.

		if(HopRewriteUtils.isBinary(hi, OpOp2.CBIND) && HopRewriteUtils.isMatrixMultiply(hi.getInput(0)) &&
			HopRewriteUtils.isDataGenOpWithConstantValue(hi.getInput(1), 0) && hi.getInput(0).getInput(0).dimsKnown() &&
			hi.getInput(0).getInput(1).dimsKnown()) {
			final Hop y = hi.getInput(0).getInput(1);
			final Hop x = hi.getInput(0).getInput(0);
			final long m = x.getDim1(); // number of rows in output or X
			final long n = y.getDim1(); // number of rows in Y or common dimension
			if(m > n * 2) {
				final Hop oldGen = hi.getInput(1);
				final Hop newGen = HopRewriteUtils.createDataGenOp(y, oldGen, 0);
				final Hop newCBind = HopRewriteUtils.createBinary(y, newGen, OpOp2.CBIND);
				final Hop newMM = HopRewriteUtils.createMatrixMultiply(x, newCBind);

				HopRewriteUtils.replaceChildReference(parent, hi, newMM, pos);
				LOG.debug("Applied MMCBind Zero algebraic simplification (line " + hi.getBeginLine() + ").");
				return newMM;
			}
		}
		return hi;
	}


	private static Hop fuseSeqAndTableExpand(Hop hi) {

		if(TernaryOp.ALLOW_CTABLE_SEQUENCE_REWRITES && hi instanceof TernaryOp ) {
			TernaryOp thop = (TernaryOp) hi;
			thop.getOp();

			if(thop.isSequenceRewriteApplicable(true) && thop.findExecTypeTernaryOp() == ExecType.CP && 
				thop.getInput(1).getForcedExecType() != Types.ExecType.FED) {
				Hop input1 = thop.getInput(0);
				if(input1 instanceof DataGenOp){
					Hop literal = new LiteralOp("seq(1, "+input1.getDim1() +")");
					HopRewriteUtils.replaceChildReference(hi, input1, literal);
				}
			}

		}
		return hi;
	}
}
