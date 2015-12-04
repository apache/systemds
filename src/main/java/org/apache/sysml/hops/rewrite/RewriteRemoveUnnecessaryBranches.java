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

package org.apache.sysml.hops.rewrite;

import java.util.ArrayList;

import org.apache.sysml.hops.Hop;
import org.apache.sysml.hops.HopsException;
import org.apache.sysml.hops.LiteralOp;
import org.apache.sysml.parser.IfStatement;
import org.apache.sysml.parser.IfStatementBlock;
import org.apache.sysml.parser.StatementBlock;

/**
 * Rule: Simplify program structure by pulling if or else statement body out
 * (removing the if statement block ifself) in order to allow intra-procedure
 * analysis to propagate exact statistics.
 * 
 */
public class RewriteRemoveUnnecessaryBranches extends StatementBlockRewriteRule
{

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
					if( !istmt.getIfBody().isEmpty() )
						ret.addAll( istmt.getIfBody() ); //pull if-branch
					//otherwise: add nothing (remove if-else)
				}
				else
				{
					//pull-out simple else body
					if( !istmt.getElseBody().isEmpty() )
						ret.addAll( istmt.getElseBody() ); //pull else-branch
					//otherwise: add nothing (remove if-else)
				}
				
				state.setRemovedBranches();
				LOG.debug("Applied removeUnnecessaryBranches (lines "+sb.getBeginLine()+"-"+sb.getEndLine()+").");
			}
			else //keep original sb (non-constant condition)
				ret.add( sb );
		}
		else //keep original sb (no if)
			ret.add( sb );
		
		return ret;
	}
}
