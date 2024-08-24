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

package org.apache.sysds.hops.ipa;

import org.apache.sysds.conf.ConfigurationManager;
import org.apache.sysds.parser.DMLProgram;
import org.apache.sysds.parser.ForStatementBlock;
import org.apache.sysds.parser.IfStatement;
import org.apache.sysds.parser.IfStatementBlock;
import org.apache.sysds.parser.ParForStatementBlock;
import org.apache.sysds.parser.StatementBlock;
import org.apache.sysds.parser.WhileStatementBlock;

/**
 * This rewrite marks loops in the main program as recompile once
 * in order to reduce recompilation overhead. We mark only top-level
 * loops and thus don't need any reset because these loops are executed
 * just once. All other loops are handled by the function-recompile-once
 * rewrite already.
 */
public class IPAPassFlagLoopsRecompileOnce extends IPAPass
{
	@Override
	public boolean isApplicable(FunctionCallGraph fgraph) {
		return InterProceduralAnalysis.FLAG_LOOP_RECOMPILE_ONCE;
	}
	
	@Override
	public boolean rewriteProgram( DMLProgram prog, FunctionCallGraph fgraph, FunctionCallSizeInfo fcallSizes ) 
	{
		if( !ConfigurationManager.isDynamicRecompilation() )
			return false;
		
		//iterate recursive over all main program
		boolean ret = false;
		for( StatementBlock sb : prog.getStatementBlocks() ) {
			if( rFlagFunctionForRecompileOnce(sb) )
				ret = true;
		}
		return ret;
	}
	
	public boolean rFlagFunctionForRecompileOnce(StatementBlock sb)
	{
		boolean ret = false;
		
		//recompilation information not available at this point
		//hence, mark any top-level loop statement block
		if (sb instanceof WhileStatementBlock) {
			ret = markRecompile(sb);
		}
		else if (sb instanceof ForStatementBlock && !(sb instanceof ParForStatementBlock)) {
			//parfor has its own recompilation already builtin
			ret = markRecompile(sb);
		}
		else if (sb instanceof IfStatementBlock) {
			IfStatementBlock isb = (IfStatementBlock) sb;
			IfStatement istmt = (IfStatement)isb.getStatement(0);
			for( StatementBlock c : istmt.getIfBody() )
				ret |= rFlagFunctionForRecompileOnce( c );
			for( StatementBlock c : istmt.getElseBody() )
				ret |= rFlagFunctionForRecompileOnce( c );
		}
		
		return ret;
	}
	
	private static boolean markRecompile(StatementBlock sb) {
		sb.setRecompileOnce( true );
		if( LOG.isDebugEnabled() )
			LOG.debug("IPA: loop (lines "+sb.getBeginLine()+"-"+sb.getEndLine()+") flagged for recompile-once.");
		return true;
	}
}
