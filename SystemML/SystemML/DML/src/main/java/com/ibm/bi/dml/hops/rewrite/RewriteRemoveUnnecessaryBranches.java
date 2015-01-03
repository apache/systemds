/**
 * IBM Confidential
 * OCO Source Materials
 * (C) Copyright IBM Corp. 2010, 2015
 * The source code for this program is not published or otherwise divested of its trade secrets, irrespective of what has been deposited with the U.S. Copyright Office.
 */

package com.ibm.bi.dml.hops.rewrite;

import java.util.ArrayList;

import com.ibm.bi.dml.hops.Hop;
import com.ibm.bi.dml.hops.HopsException;
import com.ibm.bi.dml.hops.LiteralOp;
import com.ibm.bi.dml.parser.IfStatement;
import com.ibm.bi.dml.parser.IfStatementBlock;
import com.ibm.bi.dml.parser.StatementBlock;

/**
 * Rule: Simplify program structure by pulling if or else statement body out
 * (removing the if statement block ifself) in order to allow intra-procedure
 * analysis to propagate exact statistics.
 * 
 */
public class RewriteRemoveUnnecessaryBranches extends StatementBlockRewriteRule
{
	@SuppressWarnings("unused")
	private static final String _COPYRIGHT = "Licensed Materials - Property of IBM\n(C) Copyright IBM Corp. 2010, 2015\n" +
                                             "US Government Users Restricted Rights - Use, duplication  disclosure restricted by GSA ADP Schedule Contract with IBM Corp.";

	@Override
	public ArrayList<StatementBlock> rewriteStatementBlock(StatementBlock sb, ProgramRewriteStatus state)
		throws HopsException 
	{
		ArrayList<StatementBlock> ret = new ArrayList<StatementBlock>();
		
		if( sb instanceof IfStatementBlock )
		{
			IfStatementBlock isb = (IfStatementBlock) sb;
			Hop pred = isb.getPredicateHops();
			
			//apply rewrite if literal op (constant value)
			if( pred instanceof LiteralOp )
			{
				IfStatement istmt = (IfStatement)isb.getStatement(0);
				LiteralOp litpred = (LiteralOp) pred;
				boolean condition = HopRewriteUtils.getBooleanValue(litpred);
				
				if( condition )
				{
					//pull-out simple if body
					int len = istmt.getIfBody().size();
					if( len >= 1 )
						ret.addAll( istmt.getIfBody() ); //pull if-branch
					//otherwise: add nothing (remove if-else)
				}
				else
				{
					//pull-out simple else body
					int len = istmt.getElseBody().size();
					if( len >= 1 )
						ret.addAll( istmt.getElseBody() ); //pull else-branch
					//otherwise: add nothing (remove if-else)
				}
				
				state.setRemovedBranches();
				LOG.debug("Applied removeUnnecessaryBranches.");
			}
			else //keep original sb (non-constant condition)
				ret.add( sb );
		}
		else //keep original sb (no if)
			ret.add( sb );
		
		return ret;
	}
}
