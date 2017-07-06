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

import java.util.HashMap;
import java.util.Iterator;
import java.util.Set;
import java.util.Map.Entry;

import org.apache.sysml.hops.HopsException;
import org.apache.sysml.parser.DMLProgram;
import org.apache.sysml.parser.FunctionStatementBlock;
import org.apache.sysml.parser.LanguageException;

/**
 * This rewrite identifies and removes unused functions in order
 * to reduce compilation overhead and other overheads such as 
 * parfor worker creation, where we construct function copies.
 * 
 */
public class IPAPassRemoveUnusedFunctions extends IPAPass
{
	@Override
	public boolean isApplicable() {
		return InterProceduralAnalysis.REMOVE_UNUSED_FUNCTIONS;
	}
	
	@Override
	public void rewriteProgram( DMLProgram prog, FunctionCallGraph fgraph, FunctionCallSizeInfo fcallSizes ) 
		throws HopsException
	{
		try {
			Set<String> fnamespaces = prog.getNamespaces().keySet();
			for( String fnspace : fnamespaces  ) {
				HashMap<String, FunctionStatementBlock> fsbs = prog.getFunctionStatementBlocks(fnspace);
				Iterator<Entry<String, FunctionStatementBlock>> iter = fsbs.entrySet().iterator();
				while( iter.hasNext() ) {
					Entry<String, FunctionStatementBlock> e = iter.next();
					if( !fgraph.isReachableFunction(fnspace, e.getKey()) ) {
						iter.remove();
						if( LOG.isDebugEnabled() )
							LOG.debug("IPA: Removed unused function: " + 
								DMLProgram.constructFunctionKey(fnspace, e.getKey()));
					}
				}
			}
		}
		catch(LanguageException ex) {
			throw new HopsException(ex);
		}
	}
}
