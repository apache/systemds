/**
 * IBM Confidential
 * OCO Source Materials
 * (C) Copyright IBM Corp. 2010, 2015
 * The source code for this program is not published or otherwise divested of its trade secrets, irrespective of what has been deposited with the U.S. Copyright Office.
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
import com.ibm.bi.dml.hops.Hop.AggOp;
import com.ibm.bi.dml.hops.Hop.DataGenMethod;
import com.ibm.bi.dml.hops.Hop.Direction;
import com.ibm.bi.dml.hops.Hop.OpOp1;
import com.ibm.bi.dml.hops.Hop.ReOrgOp;
import com.ibm.bi.dml.hops.HopsException;
import com.ibm.bi.dml.hops.IndexingOp;
import com.ibm.bi.dml.hops.LeftIndexingOp;
import com.ibm.bi.dml.hops.LiteralOp;
import com.ibm.bi.dml.hops.Hop.OpOp2;
import com.ibm.bi.dml.hops.ReorgOp;
import com.ibm.bi.dml.hops.UnaryOp;
import com.ibm.bi.dml.parser.DMLTranslator;
import com.ibm.bi.dml.parser.DataExpression;
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
public class RewriteAlgebraicSimplificationDynamic extends HopRewriteRule
{
	@SuppressWarnings("unused")
	private static final String _COPYRIGHT = "Licensed Materials - Property of IBM\n(C) Copyright IBM Corp. 2010, 2015\n" +
                                             "US Government Users Restricted Rights - Use, duplication  disclosure restricted by GSA ADP Schedule Contract with IBM Corp.";
	
	private static final Log LOG = LogFactory.getLog(RewriteAlgebraicSimplificationDynamic.class.getName());
	
	
	//valid aggregation operation types for rowOp to Op conversions (not all operations apply)
	private static AggOp[] LOOKUP_VALID_ROW_COL_AGGREGATE = new AggOp[]{AggOp.SUM, AggOp.MIN, AggOp.MAX, AggOp.MEAN};	
	
	//valid aggregation operation types for empty (sparse-safe) operations (not all operations apply)
	//AggOp.MEAN currently not due to missing count/corrections
	private static AggOp[] LOOKUP_VALID_EMPTY_AGGREGATE = new AggOp[]{AggOp.SUM, AggOp.MIN, AggOp.MAX, AggOp.PROD, AggOp.TRACE}; 
	
	//valid unary operation types for empty (sparse-safe) operations (not all operations apply)
	private static OpOp1[] LOOKUP_VALID_EMPTY_UNARY = new OpOp1[]{OpOp1.ABS, OpOp1.SIN, OpOp1.TAN, OpOp1.SQRT, OpOp1.ROUND, OpOp1.CUMSUM}; 
	
	
	
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
			hi = removeUnnecessaryCumulativeOp(hop, hi, i);   //e.g., cumsum(X) -> X, if nrow(X)==1;
			hi = removeUnnecessaryReorgOperation(hop, hi, i); //e.g., matrix(X) -> X, if output == input dims
			hi = removeUnnecessaryOuterProduct(hop, hi, i);   //e.g., X*(Y%*%matrix(1,...) -> X*Y, if Y col vector
			hi = fuseDatagenAndReorgOperation(hop, hi, i);    //e.g., t(rand(rows=10,max=1)) -> rand(rows=1,max=10), if one dim=1
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
			hi = simplifyDotProductSum(hop, hi, i);           //e.g., sum(v^2) -> t(v)%*%v if ncol(v)==1 
			hi = reorderMinusMatrixMult(hop, hi, i);          //e.g., (-t(X))%*%y->-(t(X)%*%y), TODO size 
			hi = simplifySumMatrixMult(hop, hi, i);           //e.g., sum(A%*%B) -> sum(t(colSums(A))*rowSums(B)), if not dot product
			hi = simplifyEmptyBinaryOperation(hop, hi, i);    //e.g., X*Y -> matrix(0,nrow(X), ncol(X)) / X+Y->X / X-Y -> X
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
				
				Hop hnew = HopRewriteUtils.createDataGenOpByVal( new LiteralOp(String.valueOf(hi.getDim1()), hi.getDim1()), 
						                                         new LiteralOp(String.valueOf(hi.getDim2()), hi.getDim2()), 0);
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
			if(   HopRewriteUtils.isDimsKnown(hi)  //dims output known
			   && HopRewriteUtils.isDimsKnown(input)  //dims input known
		       && HopRewriteUtils.isEqualSize(hi, input)) //equal dims
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
			
			if(   HopRewriteUtils.isDimsKnown(hi)  //dims output known
			   && HopRewriteUtils.isDimsKnown(input)  //dims input known
		       && HopRewriteUtils.isEqualSize(hi, input)) //equal dims
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

			if(   HopRewriteUtils.isDimsKnown(hi)  //dims output known
			   && HopRewriteUtils.isDimsKnown(input)  //dims input known
		       && HopRewriteUtils.isEqualSize(hi, input)) //equal dims
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
				
				LOG.debug("Applied removeUnnecessaryOuterProduct1");
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
				
				LOG.debug("Applied removeUnnecessaryOuterProduct2");
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
						hnew = new LiteralOp("0", 0.0);
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
									HopRewriteUtils.createValueHop(input,true), new LiteralOp("1",1), 0);
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
			}
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
				&& HopRewriteUtils.getIntValue((LiteralOp)hi2.getInput().get(1))==2
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
							HopRewriteUtils.addChildReference(hi, new LiteralOp("0",0), 0);
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
				BinaryOp minus = new BinaryOp(hi.getName(), hi.getDataType(), hi.getValueType(), OpOp2.MINUS, new LiteralOp("0",0), hi);			
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
				BinaryOp minus = new BinaryOp(hi.getName(), hi.getDataType(), hi.getValueType(), OpOp2.MINUS, new LiteralOp("0",0), hi);			
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
			if( X.getNnz() > 0 ){
				Hop hnew = new LiteralOp(String.valueOf(X.getNnz()), X.getNnz());
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
