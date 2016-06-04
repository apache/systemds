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

import org.apache.sysml.api.DMLScript;
import org.apache.sysml.api.DMLScript.RUNTIME_PLATFORM;
import org.apache.sysml.hops.DataOp;
import org.apache.sysml.hops.Hop;
import org.apache.sysml.hops.HopsException;
import org.apache.sysml.hops.LeftIndexingOp;
import org.apache.sysml.parser.ForStatement;
import org.apache.sysml.parser.ForStatementBlock;
import org.apache.sysml.parser.IfStatement;
import org.apache.sysml.parser.IfStatementBlock;
import org.apache.sysml.parser.StatementBlock;
import org.apache.sysml.parser.VariableSet;
import org.apache.sysml.parser.WhileStatement;
import org.apache.sysml.parser.WhileStatementBlock;
import org.apache.sysml.parser.Expression.DataType;

/**
 * Rule: Mark loop variables that are only read/updated through cp left indexing
 * for update in-place.
 * 
 */
public class RewriteMarkLoopVariablesUpdateInPlace extends StatementBlockRewriteRule
{
	@Override
	public ArrayList<StatementBlock> rewriteStatementBlock(StatementBlock sb, ProgramRewriteStatus status)
		throws HopsException 
	{
		ArrayList<StatementBlock> ret = new ArrayList<StatementBlock>();
		
		if( DMLScript.rtplatform == RUNTIME_PLATFORM.HADOOP
			|| DMLScript.rtplatform == RUNTIME_PLATFORM.SPARK )
		{
			ret.add(sb); // nothing to do here
			return ret; //return original statement block
		}
		
		if( sb instanceof WhileStatementBlock || sb instanceof ForStatementBlock ) //incl parfor 
		{
			ArrayList<String> candidates = new ArrayList<String>(); 
			VariableSet updated = sb.variablesUpdated();
			
			for( String varname : updated.getVariableNames() ) {
				if( updated.getVariable(varname).getDataType()==DataType.MATRIX) {
					if( sb instanceof WhileStatementBlock ) {
						WhileStatement wstmt = (WhileStatement) sb.getStatement(0);
						if( rIsApplicableForUpdateInPlace(wstmt.getBody(), varname) )
							candidates.add(varname);
					}
					else if( sb instanceof ForStatementBlock ) {
						ForStatement wstmt = (ForStatement) sb.getStatement(0);
						if( rIsApplicableForUpdateInPlace(wstmt.getBody(), varname) )
							candidates.add(varname);
					}
				}
			}	
			
			sb.setUpdateInPlaceVars(candidates);
		}
			
		//return modified statement block
		ret.add(sb);
		return ret;
	}
	
	/**
	 * 
	 * @param sbs
	 * @param varname
	 * @return
	 * @throws HopsException 
	 */
	private boolean rIsApplicableForUpdateInPlace( ArrayList<StatementBlock> sbs, String varname ) 
		throws HopsException
	{
		//NOTE: no function statement blocks / predicates considered because function call would 
		//render variable as not applicable and predicates don't allow assignments; further reuse 
		//of loop candidates as child blocks already processed
		
		//recursive invocation
		boolean ret = true;
		for( StatementBlock sb : sbs ) {
			if (sb instanceof WhileStatementBlock || sb instanceof ForStatementBlock ) 
			{
				ret &= sb.getUpdateInPlaceVars()
						 .contains(varname);
			}
			else if (sb instanceof IfStatementBlock)
			{
				IfStatementBlock isb = (IfStatementBlock) sb;
				IfStatement istmt = (IfStatement)isb.getStatement(0);
				ret &= rIsApplicableForUpdateInPlace(istmt.getIfBody(), varname);
				if( ret && istmt.getElseBody() != null )
					ret &= rIsApplicableForUpdateInPlace(istmt.getElseBody(), varname);	
			}
			else {
				if( sb.get_hops() != null )
					for( Hop hop : sb.get_hops() ) 
						ret &= isApplicableForUpdateInPlace(hop, varname);
			}
			
			//early abort if not applicable
			if( !ret ) break;
		}
		
		return ret;
	}
	
	/**
	 * 
	 * @param hop
	 * @param varname
	 * @return
	 */
	private boolean isApplicableForUpdateInPlace( Hop hop, String varname )
	{
		if( !hop.getName().equals(varname) )
			return true;
	
		//valid if read/updated by leftindexing 
		//CP exec type not evaluated here as no lops generated yet 
		return hop instanceof DataOp 
			&& hop.getInput().get(0) instanceof LeftIndexingOp
			&& hop.getInput().get(0).getInput().get(0) instanceof DataOp
			&& hop.getInput().get(0).getInput().get(0).getName().equals(varname)
			&& hop.getInput().get(0).getInput().get(0).getParent().size()==1;
	}
}
