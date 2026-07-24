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

import org.apache.sysds.api.DMLScript;
import org.apache.sysds.hops.HopsException;
import org.apache.sysds.hops.rewrite.ProgramRewriteStatus;
import org.apache.sysds.hops.rewrite.ProgramRewriter;
import org.apache.sysds.hops.rewrite.RewriteInjectOOCTee;
import org.apache.sysds.parser.DMLProgram;
import org.apache.sysds.parser.LanguageException;

/**
 * Applies OOC tee injection after static/dynamic rewrites in IPA.
 */
public class IPAPassInjectOOCTee extends IPAPass {
	@Override
	public boolean isApplicable(FunctionCallGraph fgraph) {
		return DMLScript.USE_OOC;
	}

	@Override
	public boolean rewriteProgram(DMLProgram prog, FunctionCallGraph fgraph, FunctionCallSizeInfo fcallSizes) {
		try {
			ProgramRewriter rewriter = new ProgramRewriter(new RewriteInjectOOCTee());
			ProgramRewriteStatus status = new ProgramRewriteStatus();
			rewriter.rewriteProgramHopDAGs(prog, true, status);
			return false;
		}
		catch(LanguageException ex) {
			throw new HopsException(ex);
		}
	}
}
