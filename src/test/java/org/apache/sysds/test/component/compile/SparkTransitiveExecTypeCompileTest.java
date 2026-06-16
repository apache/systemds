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

package org.apache.sysds.test.component.compile;

import org.apache.sysds.common.Types.ExecMode;
import org.apache.sysds.runtime.controlprogram.Program;
import org.junit.Test;

/**
 * Verifies the transitive Spark exec-type refinement in {@link org.apache.sysds.hops.UnaryOp}: a CP-by-estimate unary on
 * a Spark-resident input is pulled into Spark only when it is the sole consumer ({@code getParent().size() == 1}).
 */
public class SparkTransitiveExecTypeCompileTest extends CompilerTestBase {

	private static final String DML_HEADER =
		"X = rand(rows=20000000, cols=8, seed=1);\n" + // ~1.2GB -> rand and colSums run on Spark
		"v = colSums(X);\n";                           // 1x8 Spark-resident vector (opcode uack+)

	@Test
	public void singleConsumerUnaryPulledIntoSpark() {
		String dml = DML_HEADER +
			"r = round(v);\n" + // sole consumer of the Spark-resident vector -> pulled into Spark
			"print(sum(r));\n";
		Program prog = compile(dml, null, ExecMode.HYBRID, SMALL_MEM_BUDGET);

		assertSpark(prog, "uack+"); // input genuinely has a Spark output
		assertSpark(prog, "round"); // unary pulled into Spark (CP by mem estimate, single consumer)
	}

	@Test
	public void multiConsumerUnaryStaysCP() {
		String dml = DML_HEADER +
			"a = round(v);\n" + // v now has two consumers (round + abs) ...
			"b = abs(v);\n" +
			"print(sum(a) + sum(b));\n";
		Program prog = compile(dml, null, ExecMode.HYBRID, SMALL_MEM_BUDGET);

		assertSpark(prog, "uack+"); // input still has a Spark output ...
		assertCP(prog, "round");    // ... but the multi-parent guard keeps both unaries in CP
		assertCP(prog, "abs");
	}
}
