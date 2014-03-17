/**
 * IBM Confidential
 * OCO Source Materials
 * (C) Copyright IBM Corp. 2010, 2014
 * The source code for this program is not published or otherwise divested of its trade secrets, irrespective of what has been deposited with the U.S. Copyright Office.
 */

package com.ibm.bi.dml.hops.rewrite;

import java.util.ArrayList;

import com.ibm.bi.dml.hops.BinaryOp;
import com.ibm.bi.dml.hops.DataGenOp;
import com.ibm.bi.dml.hops.Hop;
import com.ibm.bi.dml.hops.Hop.DataGenMethod;
import com.ibm.bi.dml.hops.HopsException;
import com.ibm.bi.dml.hops.LiteralOp;
import com.ibm.bi.dml.hops.Hop.OpOp2;
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
public class RewriteAlgebraicSimplification extends HopRewriteRule
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

		for( Hop h : roots )
			rule_AlgebraicSimplification( h );
		
		return roots;
	}

	@Override
	public Hop rewriteHopDAG(Hop root) 
		throws HopsException
	{
		if( root == null )
			return root;
		
		rule_AlgebraicSimplification( root );
		
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
	private void rule_AlgebraicSimplification(Hop hop) 
		throws HopsException 
	{
		if(hop.get_visited() == Hop.VISIT_STATUS.DONE)
			return;
		
		//recursively process children
		for( int i=0; i<hop.getInput().size(); i++)
		{
			Hop hi = hop.getInput().get(i);
			
			//process childs recursively first (to allow roll-up)
			rule_AlgebraicSimplification(hi);
			
			//apply actual simplification rewrites (of childs incl checks)
			removeUnnecessaryVectorizeOperation(hi);       //e.g., matrix(1,nrow(X),ncol(X))/X -> 1/X
			removeUnnecessaryBinaryOperation(hop, hi, i);  //e.g., X*1 -> X (dep: should come after rm unnecessary vectorize)
			simplifyBinaryToUnaryOperation(hi);            //e.g., X*X -> X^2 (pow2)
			fuseBinarySubDAGToUnaryOperation(hi);          //e.g., X*(1-X)-> pow2mc(1)
		}

		hop.set_visited(Hop.VISIT_STATUS.DONE);
	}
	
	/**
	 * 
	 * @param hi
	 */
	private void removeUnnecessaryVectorizeOperation(Hop hi)
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
					removeChildReference(bop, dright);
					addChildReference(bop, drightIn, 1);
					//cleanup if only consumer of intermediate
					if( dright.getParent().size()<1 ) 
						removeAllChildReferences( dright );
				}
			}
			//check and remove left vectorized scalar
			else if( right.get_dataType() == DataType.MATRIX && left instanceof DataGenOp )
			{
				DataGenOp dleft = (DataGenOp) left;
				if( dleft.getDataGenMethod()==DataGenMethod.RAND && dleft.hasConstantValue() )
				{
					Hop dleftIn = dleft.getInput().get(dleft.getParamIndex(DataExpression.RAND_MIN));
					removeChildReference(bop, dleft);
					addChildReference(bop, dleftIn, 0);
					//cleanup if only consumer of intermediate
					if( dleft.getParent().size()<1 ) 
						removeAllChildReferences( dleft );
				}
			}
			

			//Note: we applied this rewrite to at most one side in order to keep the
			//output semantically equivalent. However, future extensions might consider
			//to remove vectors from both side, compute the binary op on scalars and 
			//finally feed it into a datagenop of the original dimensions.
			
		}
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
	private void removeUnnecessaryBinaryOperation( Hop parent, Hop hi, int pos ) 
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
					parent.getInput().remove(pos);
					parent.getInput().add(pos, left);
					left.getParent().remove(bop);
					left.getParent().add(parent);
				}
			}
			//X-0 -> X 
			else if(    left.get_dataType()==DataType.MATRIX 
					&& right instanceof LiteralOp && ((LiteralOp)right).getDoubleValue()==0.0 )
			{
				if( bop.getOp()==OpOp2.MINUS )
				{
					parent.getInput().remove(pos);
					parent.getInput().add(pos, left);
					left.getParent().remove(bop);
					left.getParent().add(parent);
				}
			}
			//1*X -> X
			else if(   right.get_dataType()==DataType.MATRIX 
					&& left instanceof LiteralOp && ((LiteralOp)left).getDoubleValue()==1.0 )
			{
				if( bop.getOp()==OpOp2.MULT )
				{
					parent.getInput().remove(pos);
					parent.getInput().add(pos, right);
					right.getParent().remove(bop);
					right.getParent().add(parent);
				}
			}
			
		}
	}
	
	/**
	 * handle simplification of binary operations
	 * (relies on previous common subexpression elimination)
	 * 
	 * X+X -> X*2 or X*X -> X^2
	 */
	private void simplifyBinaryToUnaryOperation( Hop hi )
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
					tmp.getParent().add(bop);
					hi.getInput().remove(1);
					hi.getInput().add(1, tmp);
				}
				else if ( bop.getOp()==OpOp2.MULT ) //X*X -> X^2
				{
					bop.setOp(OpOp2.POW);
					LiteralOp tmp = new LiteralOp("2", 2);
					tmp.getParent().add(bop);
					hi.getInput().remove(1);
					hi.getInput().add(1, tmp);
				}
			}
		}
	}
	
	/**
	 * handle simplification of more complex sub DAG to unary operation.
	 * 
	 * X*(1-X) -> pow2mc(1), X*(2-X) -> pow2mc(2)
	 * (1-X)*X -> pow2mc(1), (2-X)*X -> pow2mc(2)
	 * 
	 * @param hi
	 */
	private void fuseBinarySubDAGToUnaryOperation( Hop hi )
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
						removeChildReference(bop, left);
						addChildReference(bop, left1);
						//cleanup if only consumer of intermediate
						if( left.getParent().size()<1 ) {
							removeChildReference(left, left1);
							removeChildReference(left, left2);
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
						removeChildReference(bop, right);
						addChildReference(bop, right1);
						//cleanup if only consumer of intermediate
						if( right.getParent().size()<1 ) {
							removeChildReference(right, right1);
							removeChildReference(right, right2);
						}
					}
				}
			}
			
		}
	}
	
	
	///////////////////////
	// Util functions
	

	private int getChildReferencePos( Hop parent, Hop child )
	{
		ArrayList<Hop> childs = parent.getInput();
		for(int i=0; i<childs.size(); i++)
			if( childs.get( i ) == child ) 
				return i;				
		
		return -1;
	}
	
	private void removeChildReference( Hop parent, Hop child )
	{
		//remove child reference
		parent.getInput().remove( child );
		child.getParent().remove( parent );
	}
	
	private void removeAllChildReferences( Hop parent )
	{
		for( int i=0; i<parent.getInput().size(); i++)
		{
			Hop child = parent.getInput().get(i);
			removeChildReference(parent, child);
		}
	}
	
	private void addChildReference( Hop parent, Hop child )
	{
		parent.getInput().add( child );
		child.getParent().add( parent );
	}
	
	private void addChildReference( Hop parent, Hop child, int pos )
	{
		parent.getInput().add( pos, child );
		child.getParent().add( parent );
	}
}
