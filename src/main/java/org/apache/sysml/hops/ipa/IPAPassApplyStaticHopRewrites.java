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


import org.apache.sysml.hops.HopsException;
import org.apache.sysml.hops.rewrite.ProgramRewriter;
import org.apache.sysml.parser.DMLProgram;
import org.apache.sysml.parser.LanguageException;

/**
 * This rewrite applies static hop dag and statement block
 * rewrites such as constant folding and branch removal
 * in order to simplify statistic propagation.
 * 
 */
public class IPAPassApplyStaticHopRewrites extends IPAPass
{
	@Override
	public boolean isApplicable() {
		return InterProceduralAnalysis.APPLY_STATIC_REWRITES;
	}
	
	@Override
	public void rewriteProgram( DMLProgram prog, FunctionCallGraph fgraph, FunctionCallSizeInfo fcallSizes ) 
		throws HopsException
	{
		try {
			ProgramRewriter rewriter = new ProgramRewriter(true, false);
			rewriter.rewriteProgramHopDAGs(prog);
		} 
		catch (LanguageException ex) {
			throw new HopsException(ex);
		}
	}
}
