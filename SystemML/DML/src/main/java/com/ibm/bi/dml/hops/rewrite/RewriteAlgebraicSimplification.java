/**
 * IBM Confidential
 * OCO Source Materials
 * (C) Copyright IBM Corp. 2010, 2014
 * The source code for this program is not published or otherwise divested of its trade secrets, irrespective of what has been deposited with the U.S. Copyright Office.
 */

package com.ibm.bi.dml.hops.rewrite;

import java.util.ArrayList;

import com.ibm.bi.dml.hops.BinaryOp;
import com.ibm.bi.dml.hops.Hop;
import com.ibm.bi.dml.hops.HopsException;
import com.ibm.bi.dml.hops.LiteralOp;
import com.ibm.bi.dml.hops.Hop.OpOp2;
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
			
			//handle removal of unnecessary binary operations
			// X/1 or X*1 or 1*X or X-0 -> X
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
						hop.getInput().remove(i);
						hop.getInput().add(i, left);
						left.getParent().remove(bop);
						left.getParent().add(hop);
					}
				}
				//X-0 -> X 
				else if(    left.get_dataType()==DataType.MATRIX 
						&& right instanceof LiteralOp && ((LiteralOp)right).getDoubleValue()==0.0 )
				{
					if( bop.getOp()==OpOp2.MINUS )
					{
						hop.getInput().remove(i);
						hop.getInput().add(i, left);
						left.getParent().remove(bop);
						left.getParent().add(hop);
					}
				}
				//1*X -> X
				else if(   right.get_dataType()==DataType.MATRIX 
						&& left instanceof LiteralOp && ((LiteralOp)left).getDoubleValue()==1.0 )
				{
					if( bop.getOp()==OpOp2.MULT )
					{
						hop.getInput().remove(i);
						hop.getInput().add(i, right);
						right.getParent().remove(bop);
						right.getParent().add(hop);
					}
				}
				
			}		
		}
		
		//handle simplification of binary operations
		//(relies on previous common subexpression elimination)
		if( hop instanceof BinaryOp )
		{
			BinaryOp bop = (BinaryOp)hop;
			Hop left = hop.getInput().get(0);
			Hop right = hop.getInput().get(1);
			if( left == right && left.get_dataType()==DataType.MATRIX )
			{
				//note: we simplify this to unary operations first (less mem and better MR plan),
				//however, we later compile specific LOPS for X*2 and X^2
				if( bop.getOp()==OpOp2.PLUS ) //X+X -> X*2
				{
					bop.setOp(OpOp2.MULT);
					LiteralOp tmp = new LiteralOp("2", 2);
					tmp.getParent().add(bop);
					hop.getInput().remove(1);
					hop.getInput().add(1, tmp);
				}
				else if ( bop.getOp()==OpOp2.MULT ) //X*X -> X^2
				{
					bop.setOp(OpOp2.POW);
					LiteralOp tmp = new LiteralOp("2", 2);
					tmp.getParent().add(bop);
					hop.getInput().remove(1);
					hop.getInput().add(1, tmp);
				}
			}
		}

		hop.set_visited(Hop.VISIT_STATUS.DONE);
	}
	
}
