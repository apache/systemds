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


import org.apache.sysds.hops.HopsException;
import org.apache.sysds.hops.rewrite.ProgramRewriteStatus;
import org.apache.sysds.hops.rewrite.ProgramRewriter;
import org.apache.sysds.hops.rewrite.RewriteInjectSparkLoopCheckpointing;
import org.apache.sysds.parser.DMLProgram;
import org.apache.sysds.parser.LanguageException;

/**
 * This rewrite applies static hop dag and statement block
 * rewrites such as constant folding and branch removal
 * in order to simplify statistic propagation.
 * 
 */
public class IPAPassApplyStaticAndDynamicHopRewrites extends IPAPass
{
	@Override
	@SuppressWarnings("unused")
	public boolean isApplicable(FunctionCallGraph fgraph) {
		return InterProceduralAnalysis.APPLY_STATIC_REWRITES
			|| InterProceduralAnalysis.APPLY_DYNAMIC_REWRITES;
	}
	
	@Override
	public boolean rewriteProgram( DMLProgram prog, FunctionCallGraph fgraph, FunctionCallSizeInfo fcallSizes ) {
		try {
			// construct rewriter w/o checkpoint injection to avoid redundancy
			ProgramRewriter rewriter = new ProgramRewriter(
				InterProceduralAnalysis.APPLY_STATIC_REWRITES,
				InterProceduralAnalysis.APPLY_DYNAMIC_REWRITES);
			rewriter.removeStatementBlockRewrite(RewriteInjectSparkLoopCheckpointing.class);
			
			// rewrite program hop dags and statement blocks
			ProgramRewriteStatus status = new ProgramRewriteStatus();
			rewriter.rewriteProgramHopDAGs(prog, true, status); //rewrite and split
			// in case of removed branches entire function calls might have been eliminated,
			// accordingly, we should rebuild the function call graph to allow for inlining
			// even large functions, and avoid restrictions of scalar/size propagation
			return status.getRemovedBranches();
		}
		catch (LanguageException ex) {
			throw new HopsException(ex);
		}
	}
}
