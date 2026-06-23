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
 * Verifies the transitive Spark exec-type refinement in {@link org.apache.sysds.hops.UnaryOp} and
 * {@link org.apache.sysds.hops.BinaryOp}: a CP-by-estimate unary or matrix-scalar binary on a Spark-resident input is
 * pulled into Spark only when it is the sole consumer ({@code getParent().size() == 1}) and the operation is eligible.
 * Cumulative (and cast) operations are explicitly excluded from the pull and must stay CP.
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

	// A tall, Spark-resident column vector that is still small enough (40 KB) to be CP by memory
	// estimate: rowSums over a very wide matrix runs on Spark, but its 1-column result fits in CP.
	private static final String TALL_VECTOR_HEADER =
		"X = rand(rows=5000, cols=200000, seed=1);\n" + // ~8GB -> rand and rowSums run on Spark
		"c = rowSums(X);\n";                            // 5000x1 Spark-resident vector (opcode uark+)

	@Test
	public void cumulativeUnaryStaysCP() {
		String dml = TALL_VECTOR_HEADER +
			"r = cumsum(c);\n" +            // sole consumer of the Spark-resident vector, CP by estimate ...
			"print(as.scalar(r[2500,1]));\n"; // ... consume via indexing (avoids the sum(cumsum) rewrite)
		Program prog = compile(dml, null, ExecMode.HYBRID, SMALL_MEM_BUDGET);

		assertSpark(prog, "uark+");   // input genuinely has a Spark output
		assertCP(prog, "ucumk+"); // ... but cumulative ops are excluded from the transitive pull
	}

	@Test
	public void singleConsumerBinaryPulledIntoSpark() {
		String dml = TALL_VECTOR_HEADER +
			"r = c + 2.0;\n" + // matrix-scalar on the Spark-resident vector, sole consumer -> pulled into Spark
			"print(as.scalar(r[2500,1]));\n";
		Program prog = compile(dml, null, ExecMode.HYBRID, SMALL_MEM_BUDGET);

		assertSpark(prog, "uark+"); // input genuinely has a Spark output (multi-block column vector)
		assertSpark(prog, "+");     // matrix-scalar binary pulled into Spark (CP by estimate, single consumer)
	}

	@Test
	public void multiConsumerBinaryStaysCP() {
		String dml = TALL_VECTOR_HEADER +
			"a = c + 2.0;\n" + // c now has two consumers (+ and *) ...
			"b = c * 3.0;\n" +
			"print(as.scalar(a[2500,1]) + as.scalar(b[2500,1]));\n";
		Program prog = compile(dml, null, ExecMode.HYBRID, SMALL_MEM_BUDGET);

		assertSpark(prog, "uark+"); // input still has a Spark output ...
		assertCP(prog, "+");        // ... but the multi-parent guard keeps both binaries in CP
		assertCP(prog, "*");
	}
}
