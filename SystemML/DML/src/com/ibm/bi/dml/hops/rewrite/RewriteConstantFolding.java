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
import com.ibm.bi.dml.hops.Hop.VISIT_STATUS;
import com.ibm.bi.dml.parser.Expression.ValueType;

/**
 * Rule: Constant Folding. For all statement blocks, 
 * eliminate simple binary expressions of literals within dags by 
 * computing them and replacing them with a new Literal op once.
 * For the moment, this only applies within a dag, later this should be 
 * extended across statements block (global, inter-procedure). 
 */
public class RewriteConstantFolding extends HopRewriteRule
{
	@SuppressWarnings("unused")
	private static final String _COPYRIGHT = "Licensed Materials - Property of IBM\n(C) Copyright IBM Corp. 2010, 2013\n" +
                                             "US Government Users Restricted Rights - Use, duplication  disclosure restricted by GSA ADP Schedule Contract with IBM Corp.";
	
	@Override
	public ArrayList<Hop> rewriteHopDAGs(ArrayList<Hop> roots) 
		throws HopsException 
	{
		if( roots == null )
			return null;

		for (Hop h : roots) 
			rule_ConstantFolding(h);
		
		return roots;
	}

	@Override
	public Hop rewriteHopDAG(Hop root) 
		throws HopsException 
	{
		if( root == null )
			return null;

		rule_ConstantFolding(root);
		
		return root;
	}
	

	/**
	 * 
	 * @param hop
	 * @throws HopsException
	 */
	private void rule_ConstantFolding( Hop hop ) 
		throws HopsException 
	{
		rConstantFoldingBinaryExpression(hop);
	}
	
	/**
	 * 
	 * @param root
	 * @throws HopsException
	 */
	private void rConstantFoldingBinaryExpression( Hop root ) 
		throws HopsException
	{
		if( root.get_visited() == VISIT_STATUS.DONE )
			return;
		
		//recursively process childs (before replacement to allow bottom-recursion)
		//no iterator in order to prevent concurrent modification
		for( int i=0; i<root.getInput().size(); i++ )
		{
			Hop h = root.getInput().get(i);
			rConstantFoldingBinaryExpression(h);
		}
		
		//fold binary op if both are literals
		if( root instanceof BinaryOp 
			&& root.getInput().get(0) instanceof LiteralOp && root.getInput().get(1) instanceof LiteralOp )
		{ 
			BinaryOp broot = (BinaryOp) root;
			LiteralOp lit1 = (LiteralOp) root.getInput().get(0);	
			LiteralOp lit2 = (LiteralOp) root.getInput().get(1);
			double ret = Double.MAX_VALUE;
			
			if(   (lit1.get_valueType()==ValueType.DOUBLE || lit1.get_valueType()==ValueType.INT)
			   && (lit2.get_valueType()==ValueType.DOUBLE || lit2.get_valueType()==ValueType.INT) )
			{
				double lret = lit1.getDoubleValue();
				double rret = lit2.getDoubleValue();
				switch( broot.getOp() )
				{
					case PLUS:	ret = lret + rret; break;
					case MINUS:	ret = lret - rret; break;
					case MULT:  ret = lret * rret; break;
					case DIV:   ret = lret / rret; break;
					case MIN:   ret = Math.min(lret, rret); break;
					case MAX:   ret = Math.max(lret, rret); break;
				}
			}
			
			if( ret!=Double.MAX_VALUE )
			{
				LiteralOp literal = null;
				if( broot.get_valueType()==ValueType.DOUBLE )
					literal = new LiteralOp(String.valueOf(ret), ret);
				else if( broot.get_valueType()==ValueType.INT )
					literal = new LiteralOp(String.valueOf((long)ret), (long)ret);
				
				//reverse replacement in order to keep common subexpression elimination
				for( int i=0; i<broot.getParent().size(); i++ ) //for all parents
				{
					Hop parent = broot.getParent().get(i);
					for( int j=0; j<parent.getInput().size(); j++ )
					{
						Hop child = parent.getInput().get(j);
						if( broot == child )
						{
							//replace operator
							parent.getInput().remove(j);
							parent.getInput().add(j, literal);
						}
					}
				}
				broot.getParent().clear();	
			}		
		}
		
		//mark processed
		root.set_visited( VISIT_STATUS.DONE );
	}

}
