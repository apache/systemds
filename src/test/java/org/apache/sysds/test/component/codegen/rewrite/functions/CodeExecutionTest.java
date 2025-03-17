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

package org.apache.sysds.test.component.codegen.rewrite.functions;

import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;
import org.apache.sysds.api.DMLScript;
import org.apache.sysds.hops.rewriter.dml.DMLExecutor;
import org.apache.sysds.test.component.codegen.rewrite.RewriterTopologySortTests;
import org.junit.Test;

public class CodeExecutionTest {
	protected static final Log LOG = LogFactory.getLog(CodeExecutionTest.class.getName());

	@Test
	public void test() {
		String str = "X = rand(rows=5000, cols=5000, sparsity=0.1)\n" +
				"Y = rand(rows=5000, cols=5000, sparsity=0.1)\n" +
				"R = X*Y\n" +
				"print(lineage(R))";
		DMLScript.APPLY_GENERATED_REWRITES = true;
		DMLExecutor.executeCode(str, false, "-applyGeneratedRewrites");
	}
}
