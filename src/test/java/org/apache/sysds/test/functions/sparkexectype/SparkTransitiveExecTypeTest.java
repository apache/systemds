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

package org.apache.sysds.test.functions.sparkexectype;

import java.util.HashMap;

import org.apache.sysds.common.Types.ExecMode;
import org.apache.sysds.hops.OptimizerUtils;
import org.apache.sysds.hops.recompile.Recompiler;
import org.apache.sysds.runtime.matrix.data.MatrixValue.CellIndex;
import org.apache.sysds.test.AutomatedTestBase;
import org.apache.sysds.test.TestConfiguration;
import org.apache.sysds.test.TestUtils;
import org.apache.sysds.utils.Statistics;
import org.apache.sysds.utils.stats.InfrastructureAnalyzer;
import org.junit.Assert;
import org.junit.Test;

/**
 * Exercises the transitive Spark exec-type refinement in {@link org.apache.sysds.hops.UnaryOp} and
 * {@link org.apache.sysds.hops.BinaryOp}: cheap unary / matrix-scalar / matrix-vector operations whose input already
 * has a Spark output are pulled into Spark.
 *
 * <p>
 * Each script is run in HYBRID mode with a constrained memory budget, once with the transitive decision enabled and
 * once disabled. The results must match (correctness regardless of placement), and the transitive run must actually
 * execute Spark instructions.
 * </p>
 */
public class SparkTransitiveExecTypeTest extends AutomatedTestBase {

	private static final String TEST_DIR = "functions/sparkexectype/";
	private static final String TEST_CLASS_DIR = TEST_DIR + SparkTransitiveExecTypeTest.class.getSimpleName() + "/";
	private static final String TEST_UNARY = "SparkExecTypeUnary";
	private static final String TEST_BINARY = "SparkExecTypeBinary";

	@Override
	public void setUp() {
		TestUtils.clearAssertionInformation();
		addTestConfiguration(TEST_UNARY, new TestConfiguration(TEST_CLASS_DIR, TEST_UNARY, new String[] {"R"}));
		addTestConfiguration(TEST_BINARY, new TestConfiguration(TEST_CLASS_DIR, TEST_BINARY, new String[] {"R"}));
	}

	@Test
	public void testUnaryPulledIntoSpark() {
		runTransitiveExecTypeTest(TEST_UNARY);
	}

	@Test
	public void testBinaryPulledIntoSpark() {
		runTransitiveExecTypeTest(TEST_BINARY);
	}

	private void runTransitiveExecTypeTest(String testname) {
		final boolean oldTransitive = OptimizerUtils.ALLOW_TRANSITIVE_SPARK_EXEC_TYPE;
		final ExecMode oldPlatform = setExecMode(ExecMode.HYBRID);
		final long oldMem = InfrastructureAnalyzer.getLocalMaxMemory();
		// Small memory budget so the large operations are placed on Spark.
		InfrastructureAnalyzer.setLocalMaxMemory(1024 * 1024 * 8);

		try {
			getAndLoadTestConfiguration(testname);
			fullDMLScriptName = getScript();
			programArgs = new String[] {"-args", output("R")};

			// Reference run with the transitive Spark decision disabled.
			OptimizerUtils.ALLOW_TRANSITIVE_SPARK_EXEC_TYPE = false;
			runTest(true, false, null, -1);
			HashMap<CellIndex, Double> expected = readDMLScalarFromOutputDir("R");

			// Run with the transitive Spark decision enabled (the path under test).
			OptimizerUtils.ALLOW_TRANSITIVE_SPARK_EXEC_TYPE = true;
			runTest(true, false, null, -1);
			HashMap<CellIndex, Double> actual = readDMLScalarFromOutputDir("R");

			TestUtils.compareScalars(expected.get(new CellIndex(1, 1)), actual.get(new CellIndex(1, 1)), 1e-8);
			Assert.assertTrue("Expected Spark instructions to be executed in the transitive run.",
				Statistics.getNoOfExecutedSPInst() > 0);
		}
		finally {
			OptimizerUtils.ALLOW_TRANSITIVE_SPARK_EXEC_TYPE = oldTransitive;
			resetExecMode(oldPlatform);
			InfrastructureAnalyzer.setLocalMaxMemory(oldMem);
			Recompiler.reinitRecompiler();
		}
	}
}
