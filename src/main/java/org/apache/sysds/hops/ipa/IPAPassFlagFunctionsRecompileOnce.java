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

import java.util.Map;
import java.util.Map.Entry;

import org.apache.sysds.conf.ConfigurationManager;
import org.apache.sysds.hops.HopsException;
import org.apache.sysds.parser.DMLProgram;
import org.apache.sysds.parser.ForStatementBlock;
import org.apache.sysds.parser.FunctionDictionary;
import org.apache.sysds.parser.FunctionStatement;
import org.apache.sysds.parser.FunctionStatementBlock;
import org.apache.sysds.parser.IfStatement;
import org.apache.sysds.parser.IfStatementBlock;
import org.apache.sysds.parser.LanguageException;
import org.apache.sysds.parser.StatementBlock;
import org.apache.sysds.parser.WhileStatementBlock;

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
	public boolean isApplicable(FunctionCallGraph fgraph) {
		return InterProceduralAnalysis.FLAG_FUNCTION_RECOMPILE_ONCE;
	}
	
	@Override
	public boolean rewriteProgram( DMLProgram prog, FunctionCallGraph fgraph, FunctionCallSizeInfo fcallSizes ) 
	{
		if( !ConfigurationManager.isDynamicRecompilation() )
			return false;
		
		try {
			// flag applicable functions for recompile-once, note that this IPA pass
			// is applied to both 'optimized' and 'unoptimized' functions because this
			// pass is safe wrt correctness, and crucial for performance of mini-batch
			// algorithms in parameter servers that internally call 'unoptimized' functions
			for( Entry<String,FunctionDictionary<FunctionStatementBlock>> e : prog.getNamespaces().entrySet() ) 
				for( boolean opt : new boolean[]{true, false} ) { //optimized/unoptimized
					Map<String, FunctionStatementBlock> map = e.getValue().getFunctions(opt);
					if( map == null ) continue;
					for(Entry<String,FunctionStatementBlock> ef: map.entrySet() ) {
						FunctionStatementBlock fsblock = ef.getValue();
						if( !fgraph.isRecursiveFunction(e.getKey(), ef.getKey()) &&
							rFlagFunctionForRecompileOnce( fsblock, false ) ) {
							fsblock.setRecompileOnce( true );
							if( LOG.isDebugEnabled() )
								LOG.debug("IPA: FUNC flagged for recompile-once: " +
									DMLProgram.constructFunctionKey(e.getKey(), ef.getKey()));
						}
					}
				}
		}
		catch( LanguageException ex ) {
			throw new HopsException(ex);
		}
		return false;
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
