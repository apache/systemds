/**
 * IBM Confidential
 * OCO Source Materials
 * (C) Copyright IBM Corp. 2010, 2014
 * The source code for this program is not published or otherwise divested of its trade secrets, irrespective of what has been deposited with the U.S. Copyright Office.
 */

package com.ibm.bi.dml.hops.rewrite;

import java.util.ArrayList;

import com.ibm.bi.dml.hops.Hop;
import com.ibm.bi.dml.hops.HopsException;
import com.ibm.bi.dml.hops.LiteralOp;
import com.ibm.bi.dml.lops.Lop;
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
	private static final String _COPYRIGHT = "Licensed Materials - Property of IBM\n(C) Copyright IBM Corp. 2010, 2013\n" +
                                             "US Government Users Restricted Rights - Use, duplication  disclosure restricted by GSA ADP Schedule Contract with IBM Corp.";

	@Override
	public StatementBlock rewriteStatementBlock(StatementBlock sb)
		throws HopsException 
	{
		StatementBlock ret = sb;
		
		if( sb instanceof IfStatementBlock )
		{
			IfStatementBlock isb = (IfStatementBlock) sb;
			Hop pred = isb.getPredicateHops();
			
			//apply rewrite if literal op (constant value)
			if( pred instanceof LiteralOp )
			{
				IfStatement istmt = (IfStatement)isb.getStatement(0);
				LiteralOp litpred = (LiteralOp) pred;
				boolean condition = getBooleanValue(litpred);
				
				if( condition )
				{
					//pull-out simple if body
					if( istmt.getIfBody().size() == 1 )
						ret = istmt.getIfBody().get(0);	
					else
						ret = createEmptyStatementBlock();
				}
				else
				{
					//pull-out simple else body
					if( istmt.getElseBody().size() == 1 )
						ret = istmt.getElseBody().get(0);
					else
						ret = createEmptyStatementBlock();
				}
			}
		}
		
		return ret;
	}

	/**
	 * 
	 * @param op
	 * @return
	 * @throws HopsException
	 */
	private boolean getBooleanValue( LiteralOp op )
		throws HopsException
	{
		switch( op.get_valueType() )
		{
			case DOUBLE:  return op.getDoubleValue() != 0; 
			case INT:	  return op.getLongValue()   != 0;
			case BOOLEAN: return op.getBooleanValue();
			
			default: throw new HopsException("Invalid if predicate value type: "+op.get_valueType());
		}
	}
	
	private StatementBlock createEmptyStatementBlock()
	{
		StatementBlock ret = new StatementBlock(); //empty
		ret.set_hops(new ArrayList<Hop>());
		ret.set_lops(new ArrayList<Lop>());
		return ret;
	}
}
