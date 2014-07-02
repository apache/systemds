/**
 * IBM Confidential
 * OCO Source Materials
 * (C) Copyright IBM Corp. 2010, 2014
 * The source code for this program is not published or otherwise divested of its trade secrets, irrespective of what has been deposited with the U.S. Copyright Office.
 */

package com.ibm.bi.dml.hops.rewrite;

import java.util.ArrayList;

import com.ibm.bi.dml.hops.AggBinaryOp;
import com.ibm.bi.dml.hops.AggUnaryOp;
import com.ibm.bi.dml.hops.BinaryOp;
import com.ibm.bi.dml.hops.DataGenOp;
import com.ibm.bi.dml.hops.Hop;
import com.ibm.bi.dml.hops.Hop.AggOp;
import com.ibm.bi.dml.hops.Hop.DataGenMethod;
import com.ibm.bi.dml.hops.Hop.Direction;
import com.ibm.bi.dml.hops.Hop.ReOrgOp;
import com.ibm.bi.dml.hops.HopsException;
import com.ibm.bi.dml.hops.LiteralOp;
import com.ibm.bi.dml.hops.Hop.OpOp2;
import com.ibm.bi.dml.hops.ReorgOp;
import com.ibm.bi.dml.parser.DataExpression;
import com.ibm.bi.dml.parser.Expression.DataType;

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
	@SuppressWarnings("unused")
	private static final String _COPYRIGHT = "Licensed Materials - Property of IBM\n(C) Copyright IBM Corp. 2010, 2013\n" +
                                             "US Government Users Restricted Rights - Use, duplication  disclosure restricted by GSA ADP Schedule Contract with IBM Corp.";
	
	@Override
	public ArrayList<Hop> rewriteHopDAGs(ArrayList<Hop> roots) 
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
	public Hop rewriteHopDAG(Hop root) 
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
		if(hop.get_visited() == Hop.VISIT_STATUS.DONE)
			return;
		
		//recursively process children
		for( int i=0; i<hop.getInput().size(); i++)
		{
			Hop hi = hop.getInput().get(i);
			
			//process childs recursively first (to allow roll-up)
			if( descendFirst )
				rule_AlgebraicSimplification(hi, descendFirst); //see below
			
			//apply actual simplification rewrites (of childs incl checks)
			hi = removeUnnecessaryVectorizeOperation(hi);       //e.g., matrix(1,nrow(X),ncol(X))/X -> 1/X
			hi = removeUnnecessaryBinaryOperation(hop, hi, i);  //e.g., X*1 -> X (dep: should come after rm unnecessary vectorize)
			hi = simplifyBinaryToUnaryOperation(hi);            //e.g., X*X -> X^2 (pow2)
			hi = fuseBinarySubDAGToUnaryOperation(hi);          //e.g., X*(1-X)-> pow2mc(1)
			hi = simplifySumDiagToTrace(hi);                    //e.g., sum(diag(X)) -> trace(X)
			hi = simplifyTraceMatrixMult(hop, hi, i);           //e.g., trace(X%*%Y)->sum(X*t(Y));    
			hi = removeUnecessaryTranspose(hop, hi, i);         //e.g., t(t(X))->X; potentially introduced by diag/trace_MM
			//hi = removeUnecessaryPPred(hop, hi, i);             //e.g., ppred(X,X,"==")->matrix(1,rows=nrow(X),cols=ncol(X))
			
			//process childs recursively after rewrites (to investigate pattern newly created by rewrites)
			if( !descendFirst )
				rule_AlgebraicSimplification(hi, descendFirst);
		}

		hop.set_visited(Hop.VISIT_STATUS.DONE);
	}
	
	
	/**
	 * 
	 * @param hi
	 */
	private Hop removeUnnecessaryVectorizeOperation(Hop hi)
	{
		//applies to all binary matrix operations, if one input is unnecessarily vectorized 
		if(    hi instanceof BinaryOp && hi.get_dataType()==DataType.MATRIX 
			&& ((BinaryOp)hi).supportsMatrixScalarOperations()               )
		{
			BinaryOp bop = (BinaryOp)hi;
			Hop left = bop.getInput().get(0);
			Hop right = bop.getInput().get(1);
			
			//check and remove right vectorized scalar
			if( left.get_dataType() == DataType.MATRIX && right instanceof DataGenOp )
			{
				DataGenOp dright = (DataGenOp) right;
				if( dright.getDataGenMethod()==DataGenMethod.RAND && dright.hasConstantValue() )
				{
					Hop drightIn = dright.getInput().get(dright.getParamIndex(DataExpression.RAND_MIN));
					HopRewriteUtils.removeChildReference(bop, dright);
					HopRewriteUtils.addChildReference(bop, drightIn, 1);
					//cleanup if only consumer of intermediate
					if( dright.getParent().size()<1 ) 
						HopRewriteUtils.removeAllChildReferences( dright );
				}
			}
			//check and remove left vectorized scalar
			else if( right.get_dataType() == DataType.MATRIX && left instanceof DataGenOp )
			{
				DataGenOp dleft = (DataGenOp) left;
				if( dleft.getDataGenMethod()==DataGenMethod.RAND && dleft.hasConstantValue() )
				{
					Hop dleftIn = dleft.getInput().get(dleft.getParamIndex(DataExpression.RAND_MIN));
					HopRewriteUtils.removeChildReference(bop, dleft);
					HopRewriteUtils.addChildReference(bop, dleftIn, 0);
					//cleanup if only consumer of intermediate
					if( dleft.getParent().size()<1 ) 
						HopRewriteUtils.removeAllChildReferences( dleft );
				}
			}
			

			//Note: we applied this rewrite to at most one side in order to keep the
			//output semantically equivalent. However, future extensions might consider
			//to remove vectors from both side, compute the binary op on scalars and 
			//finally feed it into a datagenop of the original dimensions.
			
		}
		
		return hi;
	}
	
	
	/**
	 * handle removal of unnecessary binary operations
	 * 
	 * X/1 or X*1 or 1*X or X-0 -> X
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
			if(    left.get_dataType()==DataType.MATRIX 
				&& right instanceof LiteralOp && ((LiteralOp)right).getDoubleValue()==1.0 )
			{
				if( bop.getOp()==OpOp2.DIV || bop.getOp()==OpOp2.MULT )
				{
					HopRewriteUtils.removeChildReference(parent, bop);
					HopRewriteUtils.addChildReference(parent, left, pos);
					hi = left;
				}
			}
			//X-0 -> X 
			else if(    left.get_dataType()==DataType.MATRIX 
					&& right instanceof LiteralOp && ((LiteralOp)right).getDoubleValue()==0.0 )
			{
				if( bop.getOp()==OpOp2.MINUS )
				{
					HopRewriteUtils.removeChildReference(parent, bop);
					HopRewriteUtils.addChildReference(parent, left, pos);
					hi = left;
				}
			}
			//1*X -> X
			else if(   right.get_dataType()==DataType.MATRIX 
					&& left instanceof LiteralOp && ((LiteralOp)left).getDoubleValue()==1.0 )
			{
				if( bop.getOp()==OpOp2.MULT )
				{
					HopRewriteUtils.removeChildReference(parent, bop);
					HopRewriteUtils.addChildReference(parent, right, pos);
					hi = right;
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
			if( left == right && left.get_dataType()==DataType.MATRIX )
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
				}
				else if ( bop.getOp()==OpOp2.MULT ) //X*X -> X^2
				{
					bop.setOp(OpOp2.POW);
					LiteralOp tmp = new LiteralOp("2", 2);
					bop.getInput().remove(1);
					right.getParent().remove(bop);
					HopRewriteUtils.addChildReference(hi, tmp, 1);
				}
			}
		}
		
		return hi;
	}
	
	/**
	 * handle simplification of more complex sub DAG to unary operation.
	 * 
	 * X*(1-X) -> pow2mc(1), X*(2-X) -> pow2mc(2)
	 * (1-X)*X -> pow2mc(1), (2-X)*X -> pow2mc(2)
	 * 
	 * @param hi
	 */
	private Hop fuseBinarySubDAGToUnaryOperation( Hop hi )
	{
		if( hi instanceof BinaryOp )
		{
			BinaryOp bop = (BinaryOp)hi;
			Hop left = hi.getInput().get(0);
			Hop right = hi.getInput().get(1);
			
			if( bop.getOp() == OpOp2.MULT && left.get_dataType()==DataType.MATRIX && right.get_dataType()==DataType.MATRIX )
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
				
					if( left1.get_dataType() == DataType.SCALAR &&
						left2 == right &&
						bleft.getOp() == OpOp2.MINUS  ) 
					{
						bop.setOp(OpOp2.POW2CM);
						HopRewriteUtils.removeChildReference(bop, left);
						HopRewriteUtils.addChildReference(bop, left1);
						//cleanup if only consumer of intermediate
						if( left.getParent().size()<1 ) {
							HopRewriteUtils.removeChildReference(left, left1);
							HopRewriteUtils.removeChildReference(left, left2);
						}
					}
				}				
				if( right instanceof BinaryOp ) //X*(1-X)
				{
					BinaryOp bright = (BinaryOp)right;
					Hop right1 = bright.getInput().get(0);
					Hop right2 = bright.getInput().get(1);		
				
					if( right1.get_dataType() == DataType.SCALAR &&
						right2 == left &&
						bright.getOp() == OpOp2.MINUS )
					{
						bop.setOp(OpOp2.POW2CM);
						HopRewriteUtils.removeChildReference(bop, right);
						HopRewriteUtils.addChildReference(bop, right1);
						//cleanup if only consumer of intermediate
						if( right.getParent().size()<1 ) {
							HopRewriteUtils.removeChildReference(right, right1);
							HopRewriteUtils.removeChildReference(right, right2);
						}
					}
				}
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
				if( hi2 instanceof ReorgOp && ((ReorgOp)hi2).getOp()==ReOrgOp.DIAG && hi2.get_dim2()==1 ) //diagM2V
				{
					Hop hi3 = hi2.getInput().get(0);
					
					//remove diag operator
					HopRewriteUtils.removeChildReference(au, hi2);
					HopRewriteUtils.addChildReference(au, hi3, 0);	
					
					//change sum to trace
					au.setOp( AggOp.TRACE );
					
					//cleanup if only consumer of intermediate
					if( hi2.getParent().size()<1 ) 
						HopRewriteUtils.removeAllChildReferences( hi2 );
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
				ReorgOp trans = new ReorgOp(right.get_name(), right.get_dataType(), right.get_valueType(), ReOrgOp.TRANSPOSE, right);
				trans.set_rows_in_block(right.get_rows_in_block());
				trans.set_cols_in_block(right.get_cols_in_block());
				BinaryOp mult = new BinaryOp(right.get_name(), right.get_dataType(), right.get_valueType(), OpOp2.MULT, left, trans);
				mult.set_rows_in_block(right.get_rows_in_block());
				mult.set_cols_in_block(right.get_cols_in_block());
				mult.refreshSizeInformation();
				AggUnaryOp sum = new AggUnaryOp(right.get_name(), DataType.SCALAR, right.get_valueType(), AggOp.SUM, Direction.RowCol, mult);
				sum.refreshSizeInformation();
				
				//rehang new subdag under parent node
				HopRewriteUtils.addChildReference(parent, sum, pos);
				
				//cleanup if only consumer of intermediate
				if( hi.getParent().size()<1 ) 
					HopRewriteUtils.removeAllChildReferences( hi );
				if( hi2.getParent().size()<1 ) 
					HopRewriteUtils.removeAllChildReferences( hi2 );
				
				hi = sum;
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
	private Hop removeUnecessaryTranspose(Hop parent, Hop hi, int pos)
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
				if( hi.getParent().size()<1 ) 
					HopRewriteUtils.removeAllChildReferences( hi );
				if( hi2.getParent().size()<1 ) 
					HopRewriteUtils.removeAllChildReferences( hi2 );
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
