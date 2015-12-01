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
import org.apache.sysml.hops.QuaternaryOp;
import org.apache.sysml.hops.Hop.AggOp;
import org.apache.sysml.hops.Hop.DataGenMethod;
import org.apache.sysml.hops.Hop.Direction;
import org.apache.sysml.hops.Hop.OpOp1;
import org.apache.sysml.hops.Hop.OpOp4;
import org.apache.sysml.hops.Hop.ReOrgOp;
import org.apache.sysml.hops.HopsException;
import org.apache.sysml.hops.IndexingOp;
import org.apache.sysml.hops.LeftIndexingOp;
import org.apache.sysml.hops.LiteralOp;
import org.apache.sysml.hops.Hop.OpOp2;
import org.apache.sysml.hops.ReorgOp;
import org.apache.sysml.hops.UnaryOp;
import org.apache.sysml.lops.MapMultChain.ChainType;
import org.apache.sysml.parser.DMLTranslator;
import org.apache.sysml.parser.DataExpression;
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
public class RewriteAlgebraicSimplificationDynamic extends HopRewriteRule
{
	private static final Log LOG = LogFactory.getLog(RewriteAlgebraicSimplificationDynamic.class.getName());
	
	//valid aggregation operation types for rowOp to Op conversions (not all operations apply)
	private static AggOp[] LOOKUP_VALID_ROW_COL_AGGREGATE = new AggOp[]{AggOp.SUM, AggOp.SUM_SQ, AggOp.MIN, AggOp.MAX, AggOp.MEAN};
	
	//valid aggregation operation types for empty (sparse-safe) operations (not all operations apply)
	//AggOp.MEAN currently not due to missing count/corrections
	private static AggOp[] LOOKUP_VALID_EMPTY_AGGREGATE = new AggOp[]{AggOp.SUM, AggOp.SUM_SQ, AggOp.MIN, AggOp.MAX, AggOp.PROD, AggOp.TRACE};
	
	//valid unary operation types for empty (sparse-safe) operations (not all operations apply)
	private static OpOp1[] LOOKUP_VALID_EMPTY_UNARY = new OpOp1[]{OpOp1.ABS, OpOp1.SIN, OpOp1.TAN, OpOp1.SQRT, OpOp1.ROUND, OpOp1.CUMSUM}; 
	
	//valid pseudo-sparse-safe binary operators for wdivmm 
	private static OpOp2[] LOOKUP_VALID_WDIVMM_BINARY = new OpOp2[]{OpOp2.MULT, OpOp2.DIV}; 
	
	//valid unary and binary operators for wumm
	private static OpOp1[] LOOKUP_VALID_WUMM_UNARY = new OpOp1[]{OpOp1.ABS, OpOp1.ROUND, OpOp1.CEIL, OpOp1.FLOOR, OpOp1.EXP, OpOp1.LOG, OpOp1.SQRT,  OpOp1.SIGMOID, OpOp1.SPROP}; 
	private static OpOp2[] LOOKUP_VALID_WUMM_BINARY = new OpOp2[]{OpOp2.MULT, OpOp2.POW}; 
	
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
			hi = removeEmptyRightIndexing(hop, hi, i);        //e.g., X[,1] -> matrix(0,ru-rl+1,cu-cl+1), if nnz(X)==0 
			hi = removeUnnecessaryRightIndexing(hop, hi, i);  //e.g., X[,1] -> X, if output == input size 
			hi = removeEmptyLeftIndexing(hop, hi, i);         //e.g., X[,1]=Y -> matrix(0,nrow(X),ncol(X)), if nnz(X)==0 and nnz(Y)==0 
			hi = removeUnnecessaryLeftIndexing(hop, hi, i);   //e.g., X[,1]=Y -> Y, if output == input dims 
			hi = fuseLeftIndexingChainToAppend(hop, hi, i);   //e.g., X[,1]=A; X[,2]=B -> X=cbind(A,B), iff ncol(X)==2 and col1/2 lix
			hi = removeUnnecessaryCumulativeOp(hop, hi, i);   //e.g., cumsum(X) -> X, if nrow(X)==1;
			hi = removeUnnecessaryReorgOperation(hop, hi, i); //e.g., matrix(X) -> X, if output == input dims
			hi = removeUnnecessaryOuterProduct(hop, hi, i);   //e.g., X*(Y%*%matrix(1,...) -> X*Y, if Y col vector
			hi = fuseDatagenAndReorgOperation(hop, hi, i);    //e.g., t(rand(rows=10,cols=1)) -> rand(rows=1,cols=10), if one dim=1
			hi = simplifyColwiseAggregate(hop, hi, i);        //e.g., colsums(X) -> sum(X) or X, if col/row vector
			hi = simplifyRowwiseAggregate(hop, hi, i);        //e.g., rowsums(X) -> sum(X) or X, if row/col vector
			hi = simplifyColSumsMVMult(hop, hi, i);           //e.g., colSums(X*Y) -> t(Y) %*% X, if Y col vector
			hi = simplifyRowSumsMVMult(hop, hi, i);           //e.g., rowSums(X*Y) -> X %*% t(Y), if Y row vector
			hi = simplifyEmptyAggregate(hop, hi, i);          //e.g., sum(X) -> 0, if nnz(X)==0
			hi = simplifyEmptyUnaryOperation(hop, hi, i);     //e.g., round(X) -> matrix(0,nrow(X),ncol(X)), if nnz(X)==0			
			hi = simplifyEmptyReorgOperation(hop, hi, i);     //e.g., t(X) -> matrix(0, ncol(X), nrow(X)) 
			hi = simplifyEmptySortOperation(hop, hi, i);      //e.g., order(X) -> seq(1, nrow(X)), if nnz(X)==0 
			hi = simplifyEmptyMatrixMult(hop, hi, i);         //e.g., X%*%Y -> matrix(0,...), if nnz(Y)==0 | X if Y==matrix(1,1,1)
			hi = simplifyIdentityRepMatrixMult(hop, hi, i);   //e.g., X%*%y -> X if y matrix(1,1,1);
			hi = simplifyScalarMatrixMult(hop, hi, i);        //e.g., X%*%y -> X*as.scalar(y), if y is a 1-1 matrix
			hi = simplifyMatrixMultDiag(hop, hi, i);          //e.g., diag(X)%*%Y -> X*Y, if ncol(Y)==1 / -> Y*X if ncol(Y)>1 
			hi = simplifyDiagMatrixMult(hop, hi, i);          //e.g., diag(X%*%Y)->rowSums(X*t(Y)); if col vector
			hi = simplifySumDiagToTrace(hi);                  //e.g., sum(diag(X)) -> trace(X); if col vector
			hi = pushdownBinaryOperationOnDiag(hop, hi, i);   //e.g., diag(X)*7 -> diag(X*7); if col vector
			hi = simplifyWeightedSquaredLoss(hop, hi, i);     //e.g., sum(W * (X - U %*% t(V)) ^ 2) -> wsl(X, U, t(V), W, true), 
			hi = simplifyWeightedSigmoidMMChains(hop, hi, i); //e.g., W * sigmoid(Y%*%t(X)) -> wsigmoid(W, Y, t(X), type)
			hi = simplifyWeightedDivMM(hop, hi, i);           //e.g., t(U) %*% (X/(U%*%t(V))) -> wdivmm(X, U, t(V), left)
			hi = simplifyWeightedCrossEntropy(hop, hi, i);    //e.g., sum(X*log(U%*%t(V))) -> wcemm(X, U, t(V))
			hi = simplifyWeightedUnaryMM(hop, hi, i);         //e.g., X*exp(U%*%t(V)) -> wumm(X, U, t(V), exp)
			hi = simplifyDotProductSum(hop, hi, i);           //e.g., sum(v^2) -> t(v)%*%v if ncol(v)==1 
			hi = fuseSumSquared(hop, hi, i);                  //e.g., sum(X^2) -> sumSq(X), if ncol(X)>1
			hi = reorderMinusMatrixMult(hop, hi, i);          //e.g., (-t(X))%*%y->-(t(X)%*%y), TODO size
			hi = simplifySumMatrixMult(hop, hi, i);           //e.g., sum(A%*%B) -> sum(t(colSums(A))*rowSums(B)), if not dot product / wsloss
			hi = simplifyEmptyBinaryOperation(hop, hi, i);    //e.g., X*Y -> matrix(0,nrow(X), ncol(X)) / X+Y->X / X-Y -> X
			hi = simplifyScalarMVBinaryOperation(hi); 		  //e.g., X*y -> X*as.scalar(y), if y is a 1-1 matrix
			hi = simplifyNnzComputation(hop, hi, i);          //e.g., sum(ppred(X,0,"!=")) -> literal(nnz(X)), if nnz known
			
			//process childs recursively after rewrites (to investigate pattern newly created by rewrites)
			if( !descendFirst )
				rule_AlgebraicSimplification(hi, descendFirst);
		}

		hop.setVisited(Hop.VisitStatus.DONE);
	}
	
	/**
	 * 
	 * @param parent
	 * @param hi
	 * @param pos
	 * @return
	 * @throws HopsException
	 */
	private Hop removeEmptyRightIndexing(Hop parent, Hop hi, int pos) 
		throws HopsException
	{
		if( hi instanceof IndexingOp  ) //indexing op
		{	
			Hop input = hi.getInput().get(0);
			if( input.getNnz()==0 && //nnz input known and empty
			    HopRewriteUtils.isDimsKnown(hi)) //output dims known
			{
				//remove unnecessary right indexing
				HopRewriteUtils.removeChildReference(parent, hi);
				
				Hop hnew = HopRewriteUtils.createDataGenOpByVal( new LiteralOp(hi.getDim1()), 
						                                         new LiteralOp(hi.getDim2()), 0);
				HopRewriteUtils.addChildReference(parent, hnew, pos);
				parent.refreshSizeInformation();
				hi = hnew;
				
				LOG.debug("Applied removeEmptyRightIndexing");
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
	private Hop removeUnnecessaryRightIndexing(Hop parent, Hop hi, int pos)
	{
		if( hi instanceof IndexingOp  ) //indexing op
		{
			Hop input = hi.getInput().get(0);
			if( HopRewriteUtils.isEqualSize(hi, input) ) //equal dims
			{
				//equal dims of right indexing input and output -> no need for indexing
				
				//remove unnecessary right indexing
				HopRewriteUtils.removeChildReference(parent, hi);
				HopRewriteUtils.addChildReference(parent, input, pos);
				parent.refreshSizeInformation();
				hi = input;
				
				LOG.debug("Applied removeUnnecessaryRightIndexing");
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
	private Hop removeEmptyLeftIndexing(Hop parent, Hop hi, int pos) 
		throws HopsException
	{
		if( hi instanceof LeftIndexingOp  ) //left indexing op
		{
			Hop input1 = hi.getInput().get(0); //lhs matrix
			Hop input2 = hi.getInput().get(1); //rhs matrix
			
			if(   input1.getNnz()==0 //nnz original known and empty
			   && input2.getNnz()==0  ) //nnz input known and empty
			{
				//remove unnecessary right indexing
				HopRewriteUtils.removeChildReference(parent, hi);		
				Hop hnew = HopRewriteUtils.createDataGenOp( input1, 0);
				HopRewriteUtils.addChildReference(parent, hnew, pos);
				parent.refreshSizeInformation();
				hi = hnew;
				
				LOG.debug("Applied removeEmptyLeftIndexing");
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
	private Hop removeUnnecessaryLeftIndexing(Hop parent, Hop hi, int pos)
	{
		if( hi instanceof LeftIndexingOp  ) //left indexing op
		{
			Hop input = hi.getInput().get(1); //rhs matrix
			
			if( HopRewriteUtils.isEqualSize(hi, input) ) //equal dims
			{
				//equal dims of left indexing input and output -> no need for indexing
				
				//remove unnecessary right indexing
				HopRewriteUtils.removeChildReference(parent, hi);				
				HopRewriteUtils.addChildReference(parent, input, pos);
				parent.refreshSizeInformation();
				hi = input;
				
				LOG.debug("Applied removeUnnecessaryLeftIndexing");
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
	private Hop fuseLeftIndexingChainToAppend(Hop parent, Hop hi, int pos)
	{
		boolean applied = false;
		
		//pattern1: X[,1]=A; X[,2]=B -> X=cbind(A,B)
		if( hi instanceof LeftIndexingOp                      //first lix 
			&& HopRewriteUtils.isFullColumnIndexing((LeftIndexingOp)hi)
			&& hi.getInput().get(0) instanceof LeftIndexingOp //second lix	
			&& HopRewriteUtils.isFullColumnIndexing((LeftIndexingOp)hi.getInput().get(0))
			&& hi.getInput().get(0).getParent().size()==1     //first lix is single consumer
			&& hi.getInput().get(0).getInput().get(0).getDim2() == 2 ) //two column matrix
		{
			Hop input2 = hi.getInput().get(1); //rhs matrix
			Hop pred2 = hi.getInput().get(4); //cl=cu
			Hop input1 = hi.getInput().get(0).getInput().get(1); //lhs matrix
			Hop pred1 = hi.getInput().get(0).getInput().get(4); //cl=cu
			
			if( pred1 instanceof LiteralOp && HopRewriteUtils.getDoubleValueSafe((LiteralOp)pred1)==1
				&& pred2 instanceof LiteralOp && HopRewriteUtils.getDoubleValueSafe((LiteralOp)pred2)==2
				&& input1.getDataType()==DataType.MATRIX && input2.getDataType()==DataType.MATRIX )
			{
				//create new cbind operation and rewrite inputs
				HopRewriteUtils.removeChildReference(parent, hi);		
				BinaryOp bop = HopRewriteUtils.createBinary(input1, input2, OpOp2.CBIND);
				HopRewriteUtils.addChildReference(parent, bop, pos);
				
				hi = bop;
				applied = true;
			}
		}
		
		//pattern1: X[1,]=A; X[2,]=B -> X=rbind(A,B)
		if( !applied && hi instanceof LeftIndexingOp          //first lix 
			&& HopRewriteUtils.isFullRowIndexing((LeftIndexingOp)hi)
			&& hi.getInput().get(0) instanceof LeftIndexingOp //second lix	
			&& HopRewriteUtils.isFullRowIndexing((LeftIndexingOp)hi.getInput().get(0))
			&& hi.getInput().get(0).getParent().size()==1     //first lix is single consumer
			&& hi.getInput().get(0).getInput().get(0).getDim1() == 2 ) //two column matrix
		{
			Hop input2 = hi.getInput().get(1); //rhs matrix
			Hop pred2 = hi.getInput().get(2); //rl=ru
			Hop input1 = hi.getInput().get(0).getInput().get(1); //lhs matrix
			Hop pred1 = hi.getInput().get(0).getInput().get(2); //rl=ru
			
			if( pred1 instanceof LiteralOp && HopRewriteUtils.getDoubleValueSafe((LiteralOp)pred1)==1
				&& pred2 instanceof LiteralOp && HopRewriteUtils.getDoubleValueSafe((LiteralOp)pred2)==2
				&& input1.getDataType()==DataType.MATRIX && input2.getDataType()==DataType.MATRIX )
			{
				//create new cbind operation and rewrite inputs
				HopRewriteUtils.removeChildReference(parent, hi);		
				BinaryOp bop = HopRewriteUtils.createBinary(input1, input2, OpOp2.RBIND);
				HopRewriteUtils.addChildReference(parent, bop, pos);
				
				hi = bop;
				applied = true;
				
				LOG.debug("Applied fuseLeftIndexingChainToAppend2 (line "+hi.getBeginLine()+")");
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
	private Hop removeUnnecessaryCumulativeOp(Hop parent, Hop hi, int pos)
	{
		if( hi instanceof UnaryOp && ((UnaryOp)hi).isCumulativeUnaryOperation()  )
		{
			Hop input = hi.getInput().get(0); //input matrix
			
			if(   HopRewriteUtils.isDimsKnown(input)  //dims input known
		       && input.getDim1()==1 ) //1 row
			{
				OpOp1 op = ((UnaryOp)hi).getOp();
				
				//remove unnecessary unary cumsum operator
				HopRewriteUtils.removeChildReference(parent, hi);				
				HopRewriteUtils.addChildReference(parent, input, pos);
				parent.refreshSizeInformation();
				hi = input;
				
				LOG.debug("Applied removeUnnecessaryCumulativeOp: "+op);
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
	private Hop removeUnnecessaryReorgOperation(Hop parent, Hop hi, int pos)
	{
		if( hi instanceof ReorgOp && ((ReorgOp)hi).getOp() == ReOrgOp.RESHAPE ) //reshape operation
		{
			Hop input = hi.getInput().get(0); 

			if( HopRewriteUtils.isEqualSize(hi, input) ) //equal dims
			{
				//equal dims of reshape input and output -> no need for reshape because 
				//byrow always refers to both input/output and hence gives the same result
				
				//remove unnecessary right indexing
				HopRewriteUtils.removeChildReference(parent, hi);				
				HopRewriteUtils.addChildReference(parent, input, pos);
				parent.refreshSizeInformation();
				hi = input;
				
				LOG.debug("Applied removeUnnecessaryReshape");
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
	private Hop removeUnnecessaryOuterProduct(Hop parent, Hop hi, int pos)
	{
		if( hi instanceof BinaryOp  ) //binary cell operation 
		{
			Hop right = hi.getInput().get(1);
			
			//check for column replication
			if(    right instanceof AggBinaryOp //matrix mult with datagen
				&& right.getInput().get(1) instanceof DataGenOp 
				&& ((DataGenOp)right.getInput().get(1)).hasConstantValue(1d)
				&& right.getInput().get(1).getDim1() == 1 //row vector for replication
				&& right.getInput().get(0).getDim2() == 1 ) //column vector for mv binary
			{
				//remove unnecessary outer product
				HopRewriteUtils.removeChildReference(hi, right);				
				HopRewriteUtils.addChildReference(hi, right.getInput().get(0) );
				hi.refreshSizeInformation();
				
				//cleanup refs to matrix mult if no remaining consumers
				if( right.getParent().isEmpty() ) 
					HopRewriteUtils.removeAllChildReferences( right );
				
				LOG.debug("Applied removeUnnecessaryOuterProduct1 (line "+right.getBeginLine()+")");
			}
			//check for row replication
			else if(    right instanceof AggBinaryOp //matrix mult with datagen
				&& right.getInput().get(0) instanceof DataGenOp 
				&& ((DataGenOp)right.getInput().get(0)).hasConstantValue(1d)
				&& right.getInput().get(0).getDim2() == 1 //colunm vector for replication
				&& right.getInput().get(1).getDim1() == 1 ) //row vector for mv binary
			{
				//remove unnecessary outer product
				HopRewriteUtils.removeChildReference(hi, right);				
				HopRewriteUtils.addChildReference(hi, right.getInput().get(1) );
				hi.refreshSizeInformation();
				
				//cleanup refs to matrix mult if no remaining consumers
				if( right.getParent().isEmpty() ) 
					HopRewriteUtils.removeAllChildReferences( right );
				
				LOG.debug("Applied removeUnnecessaryOuterProduct2 (line "+right.getBeginLine()+")");
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
	@SuppressWarnings("unchecked")
	private Hop fuseDatagenAndReorgOperation(Hop parent, Hop hi, int pos)
	{
		if( hi instanceof ReorgOp && ((ReorgOp)hi).getOp()==ReOrgOp.TRANSPOSE  //transpose
			&& hi.getInput().get(0) instanceof DataGenOp                       //datagen
			&& hi.getInput().get(0).getParent().size()==1 )                    //transpose only consumer
		{
			DataGenOp dop = (DataGenOp)hi.getInput().get(0);
			if(    (dop.getOp() == DataGenMethod.RAND || dop.getOp() == DataGenMethod.SINIT) 
				&& (dop.getDim1()==1 || dop.getDim2()==1 )) 
			{
				//relink all parents and dataop (remove transpose)
				HopRewriteUtils.removeAllChildReferences(hi);
				ArrayList<Hop> parents = (ArrayList<Hop>) hi.getParent().clone();
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
	
	/**
	 * 
	 * @param parent
	 * @param hi
	 * @param pos
	 * @return
	 * @throws HopsException 
	 */
	@SuppressWarnings("unchecked")
	private Hop simplifyColwiseAggregate( Hop parent, Hop hi, int pos ) 
		throws HopsException
	{
		if( hi instanceof AggUnaryOp  ) 
		{
			AggUnaryOp uhi = (AggUnaryOp)hi;
			Hop input = uhi.getInput().get(0);
			
			if( HopRewriteUtils.isValidOp(uhi.getOp(), LOOKUP_VALID_ROW_COL_AGGREGATE) ) {
				if( uhi.getDirection() == Direction.Col  )
				{
					if( input.getDim1() == 1 )
					{
						//remove unnecessary col aggregation for 1 row
						HopRewriteUtils.removeChildReference(parent, hi);
						HopRewriteUtils.addChildReference(parent, input, pos);
						parent.refreshSizeInformation();
						hi = input;
						
						LOG.debug("Applied simplifyColwiseAggregate1");
					}
					else if( input.getDim2() == 1 )
					{
						//get old parents (before creating cast over aggregate)
						ArrayList<Hop> parents = (ArrayList<Hop>) hi.getParent().clone();

						//simplify col-aggregate to full aggregate
						uhi.setDirection(Direction.RowCol);
						uhi.setDataType(DataType.SCALAR);
						
						//create cast to keep same output datatype
						UnaryOp cast = new UnaryOp(uhi.getName(), DataType.MATRIX, ValueType.DOUBLE, 
				                   OpOp1.CAST_AS_MATRIX, uhi);
						HopRewriteUtils.setOutputParameters(cast, 1, 1, DMLTranslator.DMLBlockSize, DMLTranslator.DMLBlockSize, -1);
						
						//rehang cast under all parents
						for( Hop p : parents ) {
							int ix = HopRewriteUtils.getChildReferencePos(p, hi);
							HopRewriteUtils.removeChildReference(p, hi);
							HopRewriteUtils.addChildReference(p, cast, ix);
						}
						
						hi = cast;
						
						LOG.debug("Applied simplifyColwiseAggregate2");
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
	 * @throws HopsException
	 */
	@SuppressWarnings("unchecked")
	private Hop simplifyRowwiseAggregate( Hop parent, Hop hi, int pos ) 
		throws HopsException
	{
		if( hi instanceof AggUnaryOp  ) 
		{
			AggUnaryOp uhi = (AggUnaryOp)hi;
			Hop input = uhi.getInput().get(0);
			
			if( HopRewriteUtils.isValidOp(uhi.getOp(), LOOKUP_VALID_ROW_COL_AGGREGATE) ) {
				if( uhi.getDirection() == Direction.Row  )
				{
					if( input.getDim2() == 1 )
					{
						//remove unnecessary row aggregation for 1 col
						HopRewriteUtils.removeChildReference(parent, hi);
						HopRewriteUtils.addChildReference(parent, input, pos);
						parent.refreshSizeInformation();
						hi = input;
						
						LOG.debug("Applied simplifyRowwiseAggregate1");
					}
					else if( input.getDim1() == 1 )
					{
						//get old parents (before creating cast over aggregate)
						ArrayList<Hop> parents = (ArrayList<Hop>) hi.getParent().clone();

						//simplify row-aggregate to full aggregate
						uhi.setDirection(Direction.RowCol);
						uhi.setDataType(DataType.SCALAR);
						
						//create cast to keep same output datatype
						UnaryOp cast = new UnaryOp(uhi.getName(), DataType.MATRIX, ValueType.DOUBLE, 
				                   OpOp1.CAST_AS_MATRIX, uhi);
						HopRewriteUtils.setOutputParameters(cast, 1, 1, DMLTranslator.DMLBlockSize, DMLTranslator.DMLBlockSize, -1);
						
						//rehang cast under all parents
						for( Hop p : parents ) {
							int ix = HopRewriteUtils.getChildReferencePos(p, hi);
							HopRewriteUtils.removeChildReference(p, hi);
							HopRewriteUtils.addChildReference(p, cast, ix);
						}
						
						hi = cast;
						
						LOG.debug("Applied simplifyRowwiseAggregate2");
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
	 * @throws HopsException
	 */
	private Hop simplifyColSumsMVMult( Hop parent, Hop hi, int pos ) 
		throws HopsException
	{
		//colSums(X*Y) -> t(Y) %*% X, if Y col vector; additional transpose later
		//removed by other rewrite if unnecessary, i.e., if Y==t(Z)
		if( hi instanceof AggUnaryOp  ) 
		{
			AggUnaryOp uhi = (AggUnaryOp)hi;
			Hop input = uhi.getInput().get(0);
			
			if( uhi.getOp() == AggOp.SUM && uhi.getDirection() == Direction.Col  ) //colsums
			{
				if( input instanceof BinaryOp && ((BinaryOp)input).getOp()==OpOp2.MULT ) //b(*) 
				{
					Hop left = input.getInput().get(0);
					Hop right = input.getInput().get(1);
					
					if(    left.getDim1()>1 && left.getDim2()>1 
						&& right.getDim1()>1 && right.getDim2()==1 ) // MV (col vector)
					{
						//remove link parent to rowsums
						HopRewriteUtils.removeChildReference(parent, hi);
						
						//create new operators 
						ReorgOp trans = HopRewriteUtils.createTranspose(right);
						AggBinaryOp mmult = HopRewriteUtils.createMatrixMultiply(trans, left);
						
						//relink new child
						HopRewriteUtils.addChildReference(parent, mmult, pos);
						hi = mmult;
						
						//cleanup old dag
						if( uhi.getParent().isEmpty() )
							HopRewriteUtils.removeAllChildReferences(uhi);
						if( input.getParent().isEmpty() )
							HopRewriteUtils.removeAllChildReferences(input);
						
						LOG.debug("Applied simplifyColSumsMVMult");
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
	 * @throws HopsException
	 */
	private Hop simplifyRowSumsMVMult( Hop parent, Hop hi, int pos ) 
		throws HopsException
	{
		//rowSums(X * Y) -> X %*% t(Y), if Y row vector; additional transpose later
		//removed by other rewrite if unnecessary, i.e., if Y==t(Z)
		if( hi instanceof AggUnaryOp  ) 
		{
			AggUnaryOp uhi = (AggUnaryOp)hi;
			Hop input = uhi.getInput().get(0);
			
			if( uhi.getOp() == AggOp.SUM && uhi.getDirection() == Direction.Row  ) //rowsums
			{
				if( input instanceof BinaryOp && ((BinaryOp)input).getOp()==OpOp2.MULT ) //b(*) 
				{
					Hop left = input.getInput().get(0);
					Hop right = input.getInput().get(1);
					
					if(    left.getDim1()>1 && left.getDim2()>1      
						&& right.getDim1()==1 && right.getDim2()>1 ) // MV (row vector)
					{
						//remove link parent to rowsums
						HopRewriteUtils.removeChildReference(parent, hi);
						
						//create new operators 
						ReorgOp trans = HopRewriteUtils.createTranspose(right);
						AggBinaryOp mmult = HopRewriteUtils.createMatrixMultiply(left, trans);
						
						//relink new child
						HopRewriteUtils.addChildReference(parent, mmult, pos);
						hi = mmult;
						
						//cleanup old dag
						if( uhi.getParent().isEmpty() )
							HopRewriteUtils.removeAllChildReferences(uhi);
						if( input.getParent().isEmpty() )
							HopRewriteUtils.removeAllChildReferences(input);
						
						LOG.debug("Applied simplifyRowSumsMVMult");
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
	 * @throws HopsException
	 */
	private Hop simplifyEmptyAggregate(Hop parent, Hop hi, int pos) 
		throws HopsException
	{
		if( hi instanceof AggUnaryOp  ) 
		{
			AggUnaryOp uhi = (AggUnaryOp)hi;
			Hop input = uhi.getInput().get(0);
			
			if( HopRewriteUtils.isValidOp(uhi.getOp(), LOOKUP_VALID_EMPTY_AGGREGATE) ){		
				
				if( HopRewriteUtils.isEmpty(input) )
				{
					//remove unnecessary aggregation 
					HopRewriteUtils.removeChildReference(parent, hi);
				
					Hop hnew = null;
					if( uhi.getDirection() == Direction.RowCol ) 
						hnew = new LiteralOp(0.0);
					else if( uhi.getDirection() == Direction.Col ) 
						hnew = HopRewriteUtils.createDataGenOp(uhi, input, 0); //nrow(uhi)=1
					else //if( uhi.getDirection() == Direction.Row ) 
						hnew = HopRewriteUtils.createDataGenOp(input, uhi, 0); //ncol(uhi)=1
					
					//add new child to parent input
					HopRewriteUtils.addChildReference(parent, hnew, pos);
					parent.refreshSizeInformation();
					hi = hnew;
					
					LOG.debug("Applied simplifyEmptyAggregate");
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
	private Hop simplifyEmptyUnaryOperation(Hop parent, Hop hi, int pos) 
		throws HopsException
	{
		if( hi instanceof UnaryOp  ) 
		{
			UnaryOp uhi = (UnaryOp)hi;
			Hop input = uhi.getInput().get(0);
			
			if( HopRewriteUtils.isValidOp(uhi.getOp(), LOOKUP_VALID_EMPTY_UNARY) ){		
				
				if( HopRewriteUtils.isEmpty(input) )
				{
					//remove unnecessary aggregation 
					HopRewriteUtils.removeChildReference(parent, hi);
					
					//create literal add it to parent
					Hop hnew = HopRewriteUtils.createDataGenOp(input, 0);
					HopRewriteUtils.addChildReference(parent, hnew, pos);
					parent.refreshSizeInformation();
					
					hi = hnew;
					
					LOG.debug("Applied simplifyEmptyUnaryOperation");
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
	private Hop simplifyEmptyReorgOperation(Hop parent, Hop hi, int pos) 
		throws HopsException
	{
		if( hi instanceof ReorgOp  ) 
		{
			ReorgOp rhi = (ReorgOp)hi;
			Hop input = rhi.getInput().get(0);
			
			if( HopRewriteUtils.isEmpty(input) ) //empty input
			{
				//reorg-operation-specific rewrite  
				Hop hnew = null;
				if( rhi.getOp() == ReOrgOp.TRANSPOSE )
					hnew = HopRewriteUtils.createDataGenOp(input, true, input, true, 0);
				else if( rhi.getOp() == ReOrgOp.DIAG ){
					if( HopRewriteUtils.isDimsKnown(input) ){
						if( input.getDim1()==1 ) //diagv2m
							hnew = HopRewriteUtils.createDataGenOp(input, false, input, true, 0);
						else //diagm2v
							hnew = HopRewriteUtils.createDataGenOpByVal(
									HopRewriteUtils.createValueHop(input,true), new LiteralOp(1), 0);
					}
				}
				else if( rhi.getOp() == ReOrgOp.RESHAPE )
					hnew = HopRewriteUtils.createDataGenOpByVal(rhi.getInput().get(1), rhi.getInput().get(2), 0);
			
				//modify dag if one of the above rules applied
				if( hnew != null ){ 
					HopRewriteUtils.removeChildReference(parent, hi);
					HopRewriteUtils.addChildReference(parent, hnew, pos);
					parent.refreshSizeInformation();
					hi = hnew;
					
					LOG.debug("Applied simplifyEmptyReorgOperation");
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
	private Hop simplifyEmptySortOperation(Hop parent, Hop hi, int pos) 
		throws HopsException
	{
		//order(X, indexreturn=FALSE) -> matrix(0,nrow(X),1)
		//order(X, indexreturn=TRUE) -> seq(1,nrow(X),1)
		if( hi instanceof ReorgOp && ((ReorgOp)hi).getOp()==ReOrgOp.SORT  ) 
		{
			ReorgOp rhi = (ReorgOp)hi;
			Hop input = rhi.getInput().get(0);
			
			if( HopRewriteUtils.isEmpty(input) ) //empty input
			{
				//reorg-operation-specific rewrite  
				Hop hnew = null;
				boolean ixret = false;
				
				if( rhi.getInput().get(3) instanceof LiteralOp ) //index return known
				{
					ixret = HopRewriteUtils.getBooleanValue((LiteralOp)rhi.getInput().get(3));
					if( ixret )
						hnew = HopRewriteUtils.createSeqDataGenOp(input);
					else
						hnew = HopRewriteUtils.createDataGenOp(input, 0);
				}
								
				//modify dag if one of the above rules applied
				if( hnew != null ){ 
					HopRewriteUtils.removeChildReference(parent, hi);
					HopRewriteUtils.addChildReference(parent, hnew, pos);
					parent.refreshSizeInformation();
					hi = hnew;
					
					LOG.debug("Applied simplifyEmptySortOperation (indexreturn="+ixret+").");
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
	private Hop simplifyEmptyMatrixMult(Hop parent, Hop hi, int pos) 
		throws HopsException
	{
		if( hi instanceof AggBinaryOp && ((AggBinaryOp)hi).isMatrixMultiply() ) //X%*%Y -> matrix(0, )
		{
			Hop left = hi.getInput().get(0);
			Hop right = hi.getInput().get(1);
		
			if(    HopRewriteUtils.isEmpty(left)  //one input empty
				|| HopRewriteUtils.isEmpty(right) )
			{
				//remove unnecessary matrix mult 
				HopRewriteUtils.removeChildReference(parent, hi);
				
				//create datagen and add it to parent
				Hop hnew = HopRewriteUtils.createDataGenOp(left, right, 0);
				HopRewriteUtils.addChildReference(parent, hnew, pos);
				parent.refreshSizeInformation();
				
				hi = hnew;	
				
				LOG.debug("Applied simplifyEmptyMatrixMult");
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
	private Hop simplifyIdentityRepMatrixMult(Hop parent, Hop hi, int pos) 
		throws HopsException
	{
		if( hi instanceof AggBinaryOp && ((AggBinaryOp)hi).isMatrixMultiply() ) //X%*%Y -> X, if y is matrix(1,1,1)
		{
			Hop left = hi.getInput().get(0);
			Hop right = hi.getInput().get(1);
			
			// X %*% y -> X
			if( HopRewriteUtils.isDimsKnown(right) && right.getDim1()==1 && right.getDim2()==1 && //scalar right
				right instanceof DataGenOp && ((DataGenOp)right).hasConstantValue(1.0)) //matrix(1,)
			{
				HopRewriteUtils.removeChildReference(parent, hi);			
				HopRewriteUtils.addChildReference(parent, left, pos);			
				hi = left;
				
				LOG.debug("Applied simplifyIdentiyMatrixMult");
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
	private Hop simplifyScalarMatrixMult(Hop parent, Hop hi, int pos) 
		throws HopsException
	{
		if( hi instanceof AggBinaryOp && ((AggBinaryOp)hi).isMatrixMultiply() ) //X%*%Y
		{
			Hop left = hi.getInput().get(0);
			Hop right = hi.getInput().get(1);
			
			// y %*% X -> as.scalar(y) * X
			if( HopRewriteUtils.isDimsKnown(left) && left.getDim1()==1 && left.getDim2()==1 ) //scalar left
			{
				//remove link from parent to matrix mult
				HopRewriteUtils.removeChildReference(parent, hi);
			
				UnaryOp cast = new UnaryOp(left.getName(), DataType.SCALAR, ValueType.DOUBLE, 
						                   OpOp1.CAST_AS_SCALAR, left);
				HopRewriteUtils.setOutputParameters(cast, 0, 0, 0, 0, 0);
				BinaryOp mult = new BinaryOp(cast.getName(), DataType.MATRIX, ValueType.DOUBLE, OpOp2.MULT, cast, right);
				HopRewriteUtils.setOutputParameters(mult, right.getDim1(), right.getDim2(), right.getRowsInBlock(), right.getColsInBlock(), -1);
				
				//cleanup if only consumer of intermediate
				if( hi.getParent().isEmpty() ) 
					HopRewriteUtils.removeAllChildReferences( hi );
				
				//add mult to parent
				HopRewriteUtils.addChildReference(parent, mult, pos);			
				parent.refreshSizeInformation();
				
				hi = mult;
				
				LOG.debug("Applied simplifyScalarMatrixMult1");
			}
			// X %*% y -> X * as.scalar(y)
			else if( HopRewriteUtils.isDimsKnown(right) && right.getDim1()==1 && right.getDim2()==1 ) //scalar right
			{
				//remove link from parent to matrix mult
				HopRewriteUtils.removeChildReference(parent, hi);
			
				UnaryOp cast = new UnaryOp(right.getName(), DataType.SCALAR, ValueType.DOUBLE, 
						                   OpOp1.CAST_AS_SCALAR, right);
				HopRewriteUtils.setOutputParameters(cast, 0, 0, 0, 0, 0);
				BinaryOp mult = new BinaryOp(cast.getName(), DataType.MATRIX, ValueType.DOUBLE, OpOp2.MULT, cast, left);
				HopRewriteUtils.setOutputParameters(mult, left.getDim1(), left.getDim2(), left.getRowsInBlock(), left.getColsInBlock(), -1);
				
				//cleanup if only consumer of intermediate
				if( hi.getParent().isEmpty() ) 
					HopRewriteUtils.removeAllChildReferences( hi );
				
				//add mult to parent
				HopRewriteUtils.addChildReference(parent, mult, pos);			
				parent.refreshSizeInformation();
				
				hi = mult;
				
				LOG.debug("Applied simplifyScalarMatrixMult2");
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
	private Hop simplifyMatrixMultDiag(Hop parent, Hop hi, int pos) 
		throws HopsException
	{
		Hop hnew = null;
		
		if( hi instanceof AggBinaryOp && ((AggBinaryOp)hi).isMatrixMultiply() ) //X%*%Y
		{
			
			Hop left = hi.getInput().get(0);
			Hop right = hi.getInput().get(1);
		
			// diag(X) %*% Y -> X * Y / diag(X) %*% Y -> Y * X 
			// previously rep required for the second case: diag(X) %*% Y -> (X%*%ones) * Y
			if( left instanceof ReorgOp && ((ReorgOp)left).getOp()==ReOrgOp.DIAG //left diag
				&& HopRewriteUtils.isDimsKnown(left) && left.getDim2()>1 ) //diagV2M
			{
				//System.out.println("diag mm rewrite: dim2(right)="+right.getDim2());
				
				if( right.getDim2()==1 ) //right column vector
				{
					//remove link from parent to matrix mult
					HopRewriteUtils.removeChildReference(parent, hi);
					
					//create binary operation over input and right
					Hop input = left.getInput().get(0); //diag input
					hnew = new BinaryOp(input.getName(), DataType.MATRIX, ValueType.DOUBLE, OpOp2.MULT, input, right);
					HopRewriteUtils.setOutputParameters(hnew, left.getDim1(), right.getDim2(), left.getRowsInBlock(), left.getColsInBlock(), -1);
				
					LOG.debug("Applied simplifyMatrixMultDiag1");
				}
				else if( right.getDim2()>1 ) //multi column vector 
				{
					//remove link from parent to matrix mult
					HopRewriteUtils.removeChildReference(parent, hi);
					
					//create binary operation over input and right; in contrast to above rewrite,
					//we need to switch the order because MV binary cell operations require vector on the right
					Hop input = left.getInput().get(0); //diag input
					hnew = new BinaryOp(input.getName(), DataType.MATRIX, ValueType.DOUBLE, OpOp2.MULT, right, input);
					HopRewriteUtils.setOutputParameters(hnew, left.getDim1(), right.getDim2(), left.getRowsInBlock(), left.getColsInBlock(), -1);
					
					//NOTE: previously to MV binary cell operations we replicated the left (if moderate number of columns: 2)
					//create binary operation over input and right
					//Hop input = left.getInput().get(0);
					//Hop ones = HopRewriteUtils.createDataGenOpByVal(new LiteralOp("1",1), new LiteralOp(String.valueOf(right.getDim2()),right.getDim2()), 1);
					//Hop repmat = new AggBinaryOp( input.getName(), DataType.MATRIX, ValueType.DOUBLE, OpOp2.MULT, AggOp.SUM, input, ones );
					//HopRewriteUtils.setOutputParameters(repmat, input.getDim1(), ones.getDim2(), input.getRowsInBlock(), input.getColsInBlock(), -1);
					//hnew = new BinaryOp(input.getName(), DataType.MATRIX, ValueType.DOUBLE, OpOp2.MULT, repmat, right);
					//HopRewriteUtils.setOutputParameters(hnew, right.getDim1(), right.getDim2(), right.getRowsInBlock(), right.getColsInBlock(), -1);
				
					LOG.debug("Applied simplifyMatrixMultDiag2");
				}
			}
			
			//notes: similar rewrites would be possible for the right side as well, just transposed into the right alignment
		}
		
		//if one of the above rewrites applied
		if( hnew !=null ){
			//cleanup if only consumer of intermediate
			if( hi.getParent().isEmpty() ) 
				HopRewriteUtils.removeAllChildReferences( hi );
			
			//add mult to parent
			HopRewriteUtils.addChildReference(parent, hnew, pos);			
			parent.refreshSizeInformation();
			
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
	 */
	private Hop simplifyDiagMatrixMult(Hop parent, Hop hi, int pos)
	{
		if( hi instanceof ReorgOp && ((ReorgOp)hi).getOp()==ReOrgOp.DIAG && hi.getDim2()==1 ) //diagM2V
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
				AggUnaryOp rowSum = new AggUnaryOp(right.getName(), right.getDataType(), right.getValueType(), AggOp.SUM, Direction.Row, mult);
				rowSum.setRowsInBlock(right.getRowsInBlock());
				rowSum.setColsInBlock(right.getColsInBlock());
				rowSum.refreshSizeInformation();
				
				//rehang new subdag under parent node
				HopRewriteUtils.addChildReference(parent, rowSum, pos);				
				parent.refreshSizeInformation();
				
				//cleanup if only consumer of intermediate
				if( hi.getParent().isEmpty() ) 
					HopRewriteUtils.removeAllChildReferences( hi );
				if( hi2.getParent().isEmpty() ) 
					HopRewriteUtils.removeAllChildReferences( hi2 );
				
				hi = rowSum;
				
				LOG.debug("Applied simplifyDiagMatrixMult");
			}	
		}
		
		return hi;
	}
	

	/**
	 * 
	 * @param hi
	 */
	private Hop simplifySumDiagToTrace(Hop hi)
	{
		if( hi instanceof AggUnaryOp ) 
		{
			AggUnaryOp au = (AggUnaryOp) hi;
			if( au.getOp()==AggOp.SUM && au.getDirection()==Direction.RowCol )	//sum	
			{
				Hop hi2 = au.getInput().get(0);
				if( hi2 instanceof ReorgOp && ((ReorgOp)hi2).getOp()==ReOrgOp.DIAG && hi2.getDim2()==1 ) //diagM2V
				{
					Hop hi3 = hi2.getInput().get(0);
					
					//remove diag operator
					HopRewriteUtils.removeChildReference(au, hi2);
					HopRewriteUtils.addChildReference(au, hi3, 0);	
					
					//change sum to trace
					au.setOp( AggOp.TRACE );
					
					//cleanup if only consumer of intermediate
					if( hi2.getParent().isEmpty() ) 
						HopRewriteUtils.removeAllChildReferences( hi2 );
					
					LOG.debug("Applied simplifySumDiagToTrace");
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
	@SuppressWarnings("unchecked")
	private Hop pushdownBinaryOperationOnDiag(Hop parent, Hop hi, int pos) 
	{
		//diag(X)*7 --> diag(X*7) in order to (1) reduce required memory for b(*) and
		//(2) in order to make the binary operation more efficient (dense vector vs sparse matrix)
		if( hi instanceof BinaryOp && ((BinaryOp)hi).getOp()==OpOp2.MULT )
		{
			Hop left = hi.getInput().get(0);
			Hop right = hi.getInput().get(1);
			
			boolean applyLeft = false;
			boolean applyRight = false;
			
			//left input is diag
			if( left instanceof ReorgOp && ((ReorgOp)left).getOp()==ReOrgOp.DIAG
				&& left.getParent().size()==1 //binary op only parent
				&& left.getInput().get(0).getDim2()==1 //col vector
				&& right.getDataType() == DataType.SCALAR )
			{
				applyLeft = true;
			}
			else if( right instanceof ReorgOp && ((ReorgOp)right).getOp()==ReOrgOp.DIAG
					&& right.getParent().size()==1 //binary op only parent
					&& right.getInput().get(0).getDim2()==1 //col vector
					&& left.getDataType() == DataType.SCALAR )
			{
				applyRight = true;
			}
			
			//perform actual rewrite
			if( applyLeft || applyRight )
			{
				//remove all parent links to binary op (since we want to reorder
				//we cannot just look at the current parent)
				ArrayList<Hop> parents = (ArrayList<Hop>) hi.getParent().clone();
				ArrayList<Integer> parentspos = new ArrayList<Integer>(); 
				for(Hop lparent : parents) {
					int lpos = HopRewriteUtils.getChildReferencePos(lparent, hi);
					HopRewriteUtils.removeChildReferenceByPos(lparent, hi, lpos);
					parentspos.add(lpos);
				}
				
				//rewire binop-diag-input into diag-binop-input
				if( applyLeft ) {
					Hop input = left.getInput().get(0);
					HopRewriteUtils.removeChildReferenceByPos(hi, left, 0);
					HopRewriteUtils.removeChildReferenceByPos(left, input, 0);
					HopRewriteUtils.addChildReference(left, hi, 0);
					HopRewriteUtils.addChildReference(hi, input, 0);
					hi.refreshSizeInformation();
					hi = left;
				}
				else if ( applyRight ) {
					Hop input = right.getInput().get(0);
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
			&& ((AggUnaryOp)hi).getOp() == AggOp.SUM     //all patterns rooted by sum()
			&& hi.getInput().get(0) instanceof BinaryOp  //all patterns subrooted by binary op
			&& hi.getInput().get(0).getDim2() > 1  )     //not applied for vector-vector mult
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
				&& HopRewriteUtils.getDoubleValue((LiteralOp)bop.getInput().get(1).getInput().get(1))==2)
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
							W = new LiteralOp(1);
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
				&& HopRewriteUtils.getDoubleValue((LiteralOp)bop.getInput().get(1))==2
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
				&& HopRewriteUtils.getDoubleValue((LiteralOp)bop.getInput().get(1))==2
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
					Hop W = new LiteralOp(1); //no weighting 
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
			&& hi.getDim2() > 1       //not applied for vector-vector mult
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
			|| hi.getInput().get(1) instanceof BinaryOp 
			&& hi.getDim2() > 1 //not applied for vector-vector mult
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
			&& hi.getDim2() > 1 //not applied for vector-vector mult
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
			&& ((AggUnaryOp)hi).getOp() == AggOp.SUM     //pattern rooted by sum()
			&& hi.getInput().get(0) instanceof BinaryOp  //pattern subrooted by binary op
			&& hi.getInput().get(0).getDim2() > 1   )    //not applied for vector-vector mult
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
	private Hop simplifyWeightedUnaryMM(Hop parent, Hop hi, int pos) 
		throws HopsException
	{
		Hop hnew = null;
		boolean appliedPattern = false;
		
		//Pattern 1) (W*uop(U%*%t(V)))
		if( hi instanceof BinaryOp && HopRewriteUtils.isValidOp(((BinaryOp)hi).getOp(),LOOKUP_VALID_WDIVMM_BINARY)	
			&& HopRewriteUtils.isEqualSize(hi.getInput().get(0), hi.getInput().get(1)) //prevent mv
			&& hi.getDim2() > 1 //not applied for vector-vector mult
			&& hi.getInput().get(0).getDataType() == DataType.MATRIX 
			&& hi.getInput().get(0).getDim2() > hi.getInput().get(0).getColsInBlock()
			&& hi.getInput().get(1) instanceof UnaryOp
			&& HopRewriteUtils.isValidOp(((UnaryOp)hi.getInput().get(1)).getOp(), LOOKUP_VALID_WUMM_UNARY) 
			&& hi.getInput().get(1).getInput().get(0) instanceof AggBinaryOp
			&& HopRewriteUtils.isSingleBlock(hi.getInput().get(1).getInput().get(0).getInput().get(0),true) ) //BLOCKSIZE CONSTRAINT			
		{
			Hop W = hi.getInput().get(0); 
			Hop U = hi.getInput().get(1).getInput().get(0).getInput().get(0);
			Hop V = hi.getInput().get(1).getInput().get(0).getInput().get(1);
			boolean mult = ((BinaryOp)hi).getOp()==OpOp2.MULT;
			OpOp1 op = ((UnaryOp)hi.getInput().get(1)).getOp();
			
			if( !HopRewriteUtils.isTransposeOperation(V) )
				V = HopRewriteUtils.createTranspose(V);
			else 
				V = V.getInput().get(0);
				
			hnew = new QuaternaryOp(hi.getName(), DataType.MATRIX, ValueType.DOUBLE, 
					  OpOp4.WUMM, W, U, V, mult, op, null);
			HopRewriteUtils.setOutputBlocksizes(hnew, W.getRowsInBlock(), W.getColsInBlock());

			appliedPattern = true;
			LOG.debug("Applied simplifyWeightedUnaryMM1 (line "+hi.getBeginLine()+")");	
		}
		
		//Pattern 2) (W*sop(U%*%t(V),c)) for known sop translating to unary ops
		if( !appliedPattern
			&& hi instanceof BinaryOp && HopRewriteUtils.isValidOp(((BinaryOp)hi).getOp(),LOOKUP_VALID_WDIVMM_BINARY)
			&& HopRewriteUtils.isEqualSize(hi.getInput().get(0), hi.getInput().get(1)) //prevent mv
			&& hi.getDim2() > 1 //not applied for vector-vector mult
			&& hi.getInput().get(0).getDataType() == DataType.MATRIX
			&& hi.getInput().get(0).getDim2() > hi.getInput().get(0).getColsInBlock()
			&& hi.getInput().get(1) instanceof BinaryOp
			&& HopRewriteUtils.isValidOp(((BinaryOp)hi.getInput().get(1)).getOp(), LOOKUP_VALID_WUMM_BINARY) )
		{
			Hop left = hi.getInput().get(1).getInput().get(0);
			Hop right = hi.getInput().get(1).getInput().get(1);
			Hop abop = null;
			
			//pattern 2a) matrix-scalar operations
			if( right.getDataType()==DataType.SCALAR && right instanceof LiteralOp
				&& HopRewriteUtils.getDoubleValue((LiteralOp)right)==2 //pow2, mult2
				&& left instanceof AggBinaryOp
				&& HopRewriteUtils.isSingleBlock(left.getInput().get(0),true) ) //BLOCKSIZE CONSTRAINT			
			{
				abop = left;
			}
			//pattern 2b) scalar-matrix operations
			else if( left.getDataType()==DataType.SCALAR && left instanceof LiteralOp 
				&& HopRewriteUtils.getDoubleValue((LiteralOp)left)==2 //mult2
				&& ((BinaryOp)hi.getInput().get(1)).getOp() == OpOp2.MULT
				&& right instanceof AggBinaryOp
				&& HopRewriteUtils.isSingleBlock(right.getInput().get(0),true) ) //BLOCKSIZE CONSTRAINT			
			{
				abop = right;
			}
			
			if( abop != null ) {
				Hop W = hi.getInput().get(0); 
				Hop U = abop.getInput().get(0);
				Hop V = abop.getInput().get(1);
				boolean mult = ((BinaryOp)hi).getOp()==OpOp2.MULT;
				OpOp2 op = ((BinaryOp)hi.getInput().get(1)).getOp();
				
				if( !HopRewriteUtils.isTransposeOperation(V) )
					V = HopRewriteUtils.createTranspose(V);
				else 
					V = V.getInput().get(0);
					
				hnew = new QuaternaryOp(hi.getName(), DataType.MATRIX, ValueType.DOUBLE, 
						  OpOp4.WUMM, W, U, V, mult, null, op);
				HopRewriteUtils.setOutputBlocksizes(hnew, W.getRowsInBlock(), W.getColsInBlock());
	
				appliedPattern = true;
				LOG.debug("Applied simplifyWeightedUnaryMM2 (line "+hi.getBeginLine()+")");	
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
	 * NOTE: dot-product-sum could be also applied to sum(a*b). However, we 
	 * restrict ourselfs to sum(a^2) and transitively sum(a*a) since a general mm
	 * a%*%b on MR can be also counter-productive (e.g., MMCJ) while tsmm is always 
	 * beneficial. 
	 * 
	 * @param parent
	 * @param hi
	 * @param pos
	 * @return
	 * @throws HopsException 
	 */
	private Hop simplifyDotProductSum(Hop parent, Hop hi, int pos) 
		throws HopsException
	{
		//sum(v^2)/sum(v1*v2) --> as.scalar(t(v)%*%v) in order to exploit tsmm vector dotproduct 
		//w/o materialization of intermediates
		if( hi instanceof AggUnaryOp && ((AggUnaryOp)hi).getOp()==AggOp.SUM //sum
			&& ((AggUnaryOp)hi).getDirection()==Direction.RowCol //full aggregate	
			&& hi.getInput().get(0).getDim2() == 1 ) //vector (for correctness)
		{
			Hop baLeft = null;
			Hop baRight = null;
			
			Hop hi2 = hi.getInput().get(0); //check for ^2 w/o multiple consumers
			//check for sum(v^2), might have been rewritten from sum(v*v)
			if( hi2 instanceof BinaryOp && ((BinaryOp)hi2).getOp()==OpOp2.POW
				&& hi2.getInput().get(1) instanceof LiteralOp 
				&& HopRewriteUtils.getDoubleValue((LiteralOp)hi2.getInput().get(1))==2
				&& hi2.getParent().size() == 1 ) //no other consumer than sum
			{
				Hop input = hi2.getInput().get(0);
				baLeft = input;
				baRight = input;
			}
			//check for sum(v1*v2), but prevent to rewrite sum(v1*v2*v3) which is later compiled into a ta+* lop
			else if(   hi2 instanceof BinaryOp && ((BinaryOp)hi2).getOp()==OpOp2.MULT
					&& hi2.getInput().get(0).getDim2()==1 && hi2.getInput().get(1).getDim2()==1
					&& hi2.getParent().size() == 1  //no other consumer than sum
					&& !(hi2.getInput().get(0) instanceof BinaryOp && ((BinaryOp)hi2.getInput().get(0)).getOp()==OpOp2.MULT)
					&& !(hi2.getInput().get(1) instanceof BinaryOp && ((BinaryOp)hi2.getInput().get(1)).getOp()==OpOp2.MULT))
			{
				baLeft = hi2.getInput().get(0);
				baRight = hi2.getInput().get(1);
			}
			
			//perform actual rewrite (if necessary)
			if( baLeft != null && baRight != null  )
			{
				//remove link from parent to diag
				HopRewriteUtils.removeChildReference(parent, hi);
				
				//create new operator chain
				ReorgOp trans = HopRewriteUtils.createTranspose(baLeft);
				AggBinaryOp mmult = HopRewriteUtils.createMatrixMultiply(trans, baRight);
				
				UnaryOp cast = new UnaryOp(baLeft.getName(), DataType.SCALAR, ValueType.DOUBLE, OpOp1.CAST_AS_SCALAR, mmult);
				HopRewriteUtils.setOutputParameters(cast, 0, 0, 0, 0, -1);
				
				//rehang new subdag under parent node
				HopRewriteUtils.addChildReference(parent, cast, pos);				
				parent.refreshSizeInformation();
				
				//cleanup if only consumer of intermediate
				if( hi.getParent().isEmpty() ) 
					HopRewriteUtils.removeAllChildReferences( hi );
				if( hi2.getParent().isEmpty() ) 
					HopRewriteUtils.removeAllChildReferences( hi2 );
				
				hi = cast;
				
				LOG.debug("Applied simplifyDotProductSum.");
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
	 *
	 * @throws HopsException
	 */
	private Hop fuseSumSquared(Hop parent, Hop hi, int pos)
			throws HopsException {
		// if SUM
		if (hi instanceof AggUnaryOp && ((AggUnaryOp) hi).getOp() == AggOp.SUM) {
			Hop sumInput = hi.getInput().get(0);

			// if input to SUM is POW(X,2), and no other consumers of the POW(X,2) HOP
			if (sumInput instanceof BinaryOp && ((BinaryOp) sumInput).getOp() == OpOp2.POW
					&& sumInput.getInput().get(1) instanceof LiteralOp
					&& HopRewriteUtils.getDoubleValue((LiteralOp) sumInput.getInput().get(1)) == 2
					&& sumInput.getParent().size() == 1) {
				Hop x = sumInput.getInput().get(0);

				// if X is NOT a column vector
				if (x.getDim2() > 1) {
					// perform rewrite from SUM(POW(X,2)) to SUM_SQ(X)
					DataType dt = hi.getDataType();
					ValueType vt = hi.getValueType();
					Direction dir = ((AggUnaryOp) hi).getDirection();
					long brlen = hi.getRowsInBlock();
					long bclen = hi.getColsInBlock();
					AggUnaryOp sumSq = new AggUnaryOp("sumSq", dt, vt, AggOp.SUM_SQ, dir, x);
					HopRewriteUtils.setOutputBlocksizes(sumSq, brlen, bclen);
					HopRewriteUtils.removeChildReferenceByPos(parent, hi, pos);
					HopRewriteUtils.addChildReference(parent, sumSq, pos);

					// cleanup
					if (hi.getParent().isEmpty())
						HopRewriteUtils.removeAllChildReferences(hi);
					if(sumInput.getParent().isEmpty())
						HopRewriteUtils.removeAllChildReferences(sumInput);

					// replace current HOP with new SUM_SQ HOP
					hi = sumSq;
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
	private Hop simplifyEmptyBinaryOperation(Hop parent, Hop hi, int pos) 
		throws HopsException
	{
		if( hi instanceof BinaryOp ) //b(?) X Y
		{
			BinaryOp bop = (BinaryOp) hi;
			Hop left = hi.getInput().get(0);
			Hop right = hi.getInput().get(1);
		
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
						hnew = null;
				}
				
				if( hnew != null )
				{
					//remove unnecessary matrix mult 
					HopRewriteUtils.removeChildReference(parent, hi);
					
					//create datagen and add it to parent
					HopRewriteUtils.addChildReference(parent, hnew, pos);
					parent.refreshSizeInformation();
					
					hi = hnew;
					
					LOG.debug("Applied simplifyEmptyBinaryOperation");
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
	 * TODO select up or down based on size
	 * 
	 * @param parent
	 * @param hi
	 * @param pos
	 * @return
	 * @throws HopsException 
	 */
	@SuppressWarnings("unchecked")
	private Hop reorderMinusMatrixMult(Hop parent, Hop hi, int pos) 
		throws HopsException
	{
		if( hi instanceof AggBinaryOp && ((AggBinaryOp)hi).isMatrixMultiply() ) //X%*%Y
		{
			Hop hileft = hi.getInput().get(0);
			Hop hiright = hi.getInput().get(1);
			
			if( hileft instanceof BinaryOp && ((BinaryOp)hileft).getOp()==OpOp2.MINUS  //X=-Z
				&& hileft.getInput().get(0) instanceof LiteralOp 
				&& HopRewriteUtils.getDoubleValue((LiteralOp)hileft.getInput().get(0))==0.0 ) 
			{
				Hop hi2 = hileft.getInput().get(1);
				
				//remove link from matrixmult to minus
				HopRewriteUtils.removeChildReference(hi, hileft);
				
				//get old parents (before creating minus over matrix mult)
				ArrayList<Hop> parents = (ArrayList<Hop>) hi.getParent().clone();
				
				//create new operators 
				BinaryOp minus = new BinaryOp(hi.getName(), hi.getDataType(), hi.getValueType(), OpOp2.MINUS, new LiteralOp(0), hi);			
				minus.setRowsInBlock(hi.getRowsInBlock());
				minus.setColsInBlock(hi.getColsInBlock());
				
				//rehang minus under all parents
				for( Hop p : parents ) {
					int ix = HopRewriteUtils.getChildReferencePos(p, hi);
					HopRewriteUtils.removeChildReference(p, hi);
					HopRewriteUtils.addChildReference(p, minus, ix);
				}
				
				//rehang child of minus under matrix mult
				HopRewriteUtils.addChildReference(hi, hi2, 0);
				
				//cleanup if only consumer of minus
				if( hileft.getParent().isEmpty() ) 
					HopRewriteUtils.removeAllChildReferences( hileft );
				
				hi = minus;
				
				LOG.debug("Applied reorderMinusMatrixMult");
			}
			else if( hiright instanceof BinaryOp && ((BinaryOp)hiright).getOp()==OpOp2.MINUS  //X=-Z
					&& hiright.getInput().get(0) instanceof LiteralOp 
					&& HopRewriteUtils.getDoubleValue((LiteralOp)hiright.getInput().get(0))==0.0 ) 
			{
				Hop hi2 = hiright.getInput().get(1);
				
				//remove link from matrixmult to minus
				HopRewriteUtils.removeChildReference(hi, hiright);
				
				//get old parents (before creating minus over matrix mult)
				ArrayList<Hop> parents = (ArrayList<Hop>) hi.getParent().clone();
				
				//create new operators 
				BinaryOp minus = new BinaryOp(hi.getName(), hi.getDataType(), hi.getValueType(), OpOp2.MINUS, new LiteralOp(0), hi);			
				minus.setRowsInBlock(hi.getRowsInBlock());
				minus.setColsInBlock(hi.getColsInBlock());
				
				//rehang minus under all parents
				for( Hop p : parents ) {
					int ix = HopRewriteUtils.getChildReferencePos(p, hi);
					HopRewriteUtils.removeChildReference(p, hi);
					HopRewriteUtils.addChildReference(p, minus, ix);
				}
				
				//rehang child of minus under matrix mult
				HopRewriteUtils.addChildReference(hi, hi2, 1);
				
				//cleanup if only consumer of minus
				if( hiright.getParent().isEmpty() ) 
					HopRewriteUtils.removeAllChildReferences( hiright );
				
				hi = minus;
				
				LOG.debug("Applied reorderMinusMatrixMult");
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
	private Hop simplifySumMatrixMult(Hop parent, Hop hi, int pos)
	{
		//sum(A%*%B) -> sum(t(colSums(A))*rowSums(B))
		//if not dot product, not applied since aggregate removed
		//if sum not the only consumer, not applied to prevent redundancy 
		if( hi instanceof AggUnaryOp && ((AggUnaryOp)hi).getOp()==AggOp.SUM  //sum
			&& ((AggUnaryOp)hi).getDirection() == Direction.RowCol	         //full aggregate
			&& hi.getInput().get(0) instanceof AggBinaryOp                   //A%*%B
			&& (hi.getInput().get(0).getDim1()>1 || hi.getInput().get(0).getDim2()>1) //not dot product
			&& hi.getInput().get(0).getParent().size()==1 )     //not multiple consumers of matrix mult
		{
			Hop hi2 = hi.getInput().get(0);
			Hop left = hi2.getInput().get(0);
			Hop right = hi2.getInput().get(1);
				
			//remove link from parent to diag
			HopRewriteUtils.removeChildReference(hi, hi2);
				
			//create new operators
			AggUnaryOp colSum = new AggUnaryOp(left.getName(), left.getDataType(), left.getValueType(), AggOp.SUM, Direction.Col, left);
			colSum.setRowsInBlock(left.getRowsInBlock());
			colSum.setColsInBlock(left.getColsInBlock());
			colSum.refreshSizeInformation();
			ReorgOp trans = HopRewriteUtils.createTranspose(colSum);
			AggUnaryOp rowSum = new AggUnaryOp(right.getName(), right.getDataType(), right.getValueType(), AggOp.SUM, Direction.Row, right);
			rowSum.setRowsInBlock(right.getRowsInBlock());
			rowSum.setColsInBlock(right.getColsInBlock());
			rowSum.refreshSizeInformation();
			BinaryOp mult = new BinaryOp(right.getName(), right.getDataType(), right.getValueType(), OpOp2.MULT, trans, rowSum);
			mult.setRowsInBlock(right.getRowsInBlock());
			mult.setColsInBlock(right.getColsInBlock());
			mult.refreshSizeInformation();
				
			
			//rehang new subdag under current node (keep hi intact)
			HopRewriteUtils.addChildReference(hi, mult, 0);				
			hi.refreshSizeInformation();
				
			//cleanup if only consumer of intermediate
			if( hi2.getParent().isEmpty() ) 
				HopRewriteUtils.removeAllChildReferences( hi2 );
			
			LOG.debug("Applied simplifySumMatrixMult.");	
		}
		
		return hi;
	}
	
	/**
	 * 
	 * @param hi
	 * @return
	 * @throws HopsException
	 */
	private Hop simplifyScalarMVBinaryOperation(Hop hi) 
		throws HopsException
	{
		if( hi instanceof BinaryOp && ((BinaryOp)hi).supportsMatrixScalarOperations() //e.g., X * s
			&& hi.getInput().get(0).getDataType()==DataType.MATRIX 
			&& hi.getInput().get(1).getDataType()==DataType.MATRIX )	
		{
			Hop right = hi.getInput().get(1);
			
			//X * s -> X * as.scalar(s)
			if( HopRewriteUtils.isDimsKnown(right) && right.getDim1()==1 && right.getDim2()==1 ) //scalar right
			{
				//remove link to right child and introduce cast
				HopRewriteUtils.removeChildReference(hi, right);
				UnaryOp cast = new UnaryOp(right.getName(), DataType.SCALAR, ValueType.DOUBLE, 
						                   OpOp1.CAST_AS_SCALAR, right);
				HopRewriteUtils.setOutputParameters(cast, 0, 0, 0, 0, 0);
				HopRewriteUtils.addChildReference(hi, cast, 1);			
				
				LOG.debug("Applied simplifyScalarMVBinaryOperation.");
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
	private Hop simplifyNnzComputation(Hop parent, Hop hi, int pos) 
		throws HopsException
	{
		//sum(ppred(X,0,"!=")) -> literal(nnz(X)), if nnz known		
		if(    hi instanceof AggUnaryOp && ((AggUnaryOp)hi).getOp()==AggOp.SUM  //sum
			&& ((AggUnaryOp)hi).getDirection() == Direction.RowCol	            //full aggregate
			&& hi.getInput().get(0) instanceof BinaryOp 
			&& ((BinaryOp)hi.getInput().get(0)).getOp()==OpOp2.NOTEQUAL )
		{
			Hop ppred = hi.getInput().get(0);
			Hop X = null;
			if(    ppred.getInput().get(0) instanceof LiteralOp 
				&& HopRewriteUtils.getDoubleValue((LiteralOp)ppred.getInput().get(0))==0 )
			{
				X = ppred.getInput().get(1);
			}
			else if(   ppred.getInput().get(1) instanceof LiteralOp 
					&& HopRewriteUtils.getDoubleValue((LiteralOp)ppred.getInput().get(1))==0 )
			{
				X = ppred.getInput().get(0);
			}
		
			//apply rewrite if known nnz 
			if( X != null && X.getNnz() > 0 ){
				Hop hnew = new LiteralOp(X.getNnz());
				HopRewriteUtils.removeChildReferenceByPos(parent, hi, pos);
				HopRewriteUtils.addChildReference(parent, hnew, pos);
				
				if( hi.getParent().isEmpty() )
					HopRewriteUtils.removeAllChildReferences( hi );
				
				hi = hnew;
				LOG.debug("Applied simplifyNnzComputation.");	
			}
		}
		
		return hi;
	}
}
