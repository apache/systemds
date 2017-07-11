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

package org.apache.sysml.hops.ipa;

import org.apache.sysml.conf.ConfigurationManager;
import org.apache.sysml.hops.HopsException;
import org.apache.sysml.parser.DMLProgram;
import org.apache.sysml.parser.ForStatementBlock;
import org.apache.sysml.parser.FunctionStatement;
import org.apache.sysml.parser.FunctionStatementBlock;
import org.apache.sysml.parser.IfStatement;
import org.apache.sysml.parser.IfStatementBlock;
import org.apache.sysml.parser.LanguageException;
import org.apache.sysml.parser.StatementBlock;
import org.apache.sysml.parser.WhileStatementBlock;

/**
 * This rewrite marks functions with loops as recompile once
 * in order to reduce recompilation overhead. Such functions
 * are recompiled on function entry with the size information
 * of the function inputs which is often sufficient to decide
 * upon execution types; in case there are still unknowns, the
 * traditional recompilation per atomic block still applies.   
 * 
 * TODO call after lops construction
 */
public class IPAPassFlagFunctionsRecompileOnce extends IPAPass
{
	@Override
	public boolean isApplicable() {
		return InterProceduralAnalysis.FLAG_FUNCTION_RECOMPILE_ONCE;
	}
	
	@Override
	public void rewriteProgram( DMLProgram prog, FunctionCallGraph fgraph, FunctionCallSizeInfo fcallSizes ) 
		throws HopsException
	{
		if( !ConfigurationManager.isDynamicRecompilation() )
			return;
		
		try {
			for (String namespaceKey : prog.getNamespaces().keySet())
				for (String fname : prog.getFunctionStatementBlocks(namespaceKey).keySet())
				{
					FunctionStatementBlock fsblock = prog.getFunctionStatementBlock(namespaceKey,fname);
					if( !fgraph.isRecursiveFunction(namespaceKey, fname) &&
						rFlagFunctionForRecompileOnce( fsblock, false ) ) 
					{
						fsblock.setRecompileOnce( true ); 
						if( LOG.isDebugEnabled() )
							LOG.debug("IPA: FUNC flagged for recompile-once: " + 
								DMLProgram.constructFunctionKey(namespaceKey, fname));
					}
				}
		}
		catch( LanguageException ex ) {
			throw new HopsException(ex);
		}
	}
	
	/**
	 * Returns true if this statementblock requires recompilation inside a 
	 * loop statement block.
	 * 
	 * @param sb statement block
	 * @param inLoop true if in loop
	 * @return true if statement block requires recompilation inside a loop statement block
	 */
	public boolean rFlagFunctionForRecompileOnce( StatementBlock sb, boolean inLoop )
	{
		boolean ret = false;
		
		if (sb instanceof FunctionStatementBlock) {
			FunctionStatementBlock fsb = (FunctionStatementBlock)sb;
			FunctionStatement fstmt = (FunctionStatement)fsb.getStatement(0);
			for( StatementBlock c : fstmt.getBody() )
				ret |= rFlagFunctionForRecompileOnce( c, inLoop );			
		}
		else if (sb instanceof WhileStatementBlock) {
			//recompilation information not available at this point
			//hence, mark any loop statement block
			ret = true;
		}
		else if (sb instanceof IfStatementBlock) {
			IfStatementBlock isb = (IfStatementBlock) sb;
			IfStatement istmt = (IfStatement)isb.getStatement(0);
			ret |= (inLoop && isb.requiresPredicateRecompilation() );
			for( StatementBlock c : istmt.getIfBody() )
				ret |= rFlagFunctionForRecompileOnce( c, inLoop );
			for( StatementBlock c : istmt.getElseBody() )
				ret |= rFlagFunctionForRecompileOnce( c, inLoop );
		}
		else if (sb instanceof ForStatementBlock) {
			//recompilation information not available at this point
			//hence, mark any loop statement block
			ret = true;
		}
		else {
			ret |= ( inLoop && sb.requiresRecompilation() );
		}
		
		return ret;
	}
}
