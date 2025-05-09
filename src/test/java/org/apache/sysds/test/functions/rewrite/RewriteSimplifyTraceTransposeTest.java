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
package org.apache.sysds.test.functions.rewrite;

import org.apache.sysds.hops.OptimizerUtils;
import org.apache.sysds.runtime.matrix.data.MatrixValue;
import org.apache.sysds.test.AutomatedTestBase;
import org.apache.sysds.test.TestConfiguration;
import org.apache.sysds.test.TestUtils;
import org.junit.Assert;
import org.junit.Test;

import java.util.HashMap;

public class RewriteSimplifyTraceTransposeTest extends AutomatedTestBase {
	private static final String TEST_NAME = "RewriteSimplifyTraceTranspose";
	private static final String TEST_DIR = "functions/rewrite/";
	private static final String TEST_CLASS_DIR = TEST_DIR + RewriteSimplifyTraceTransposeTest.class.getSimpleName() + "/";

	private static final int rows = 100;
	private static final int cols = 100;
	private static final double eps = 1e-6;

	@Override
	public void setUp() {
		TestUtils.clearAssertionInformation();
		addTestConfiguration(TEST_NAME, new TestConfiguration(TEST_CLASS_DIR, TEST_NAME, new String[]{"R"}));
	}

	@Test
	public void testRewriteEnabled() {
		runRewriteTest(true);
	}

	@Test
	public void testRewriteDisabled() {
		runRewriteTest(false);
	}

	private void runRewriteTest(boolean rewriteEnabled) {
		boolean oldFlag = OptimizerUtils.ALLOW_ALGEBRAIC_SIMPLIFICATION;
		try {
			TestConfiguration config = getTestConfiguration(TEST_NAME);
			loadTestConfiguration(config);

			String HOME = SCRIPT_DIR + TEST_DIR;
			fullDMLScriptName = HOME + TEST_NAME + ".dml";
			fullRScriptName = HOME + TEST_NAME + ".R";
			programArgs = new String[]{"-stats", "-args", input("A"), output("R")};
			rCmd = getRCmd(inputDir(), expectedDir());

			OptimizerUtils.ALLOW_ALGEBRAIC_SIMPLIFICATION = rewriteEnabled;
			double[][] A = getRandomMatrix(rows, cols, -1, 1, 0.70d, 7);
			writeInputMatrixWithMTD("A", A, true);
			runTest(true, false, null, -1);
			runRScript(true);

			// Read DML scalar output
			HashMap<MatrixValue.CellIndex, Double> dmlMap = readDMLScalarFromOutputDir("R");
			double dmlTrace = dmlMap.get(new MatrixValue.CellIndex(1, 1));

			// Read R scalar output
			HashMap<MatrixValue.CellIndex, Double> rMap = readRScalarFromExpectedDir("R");
			double rTrace = rMap.get(new MatrixValue.CellIndex(1, 1));

			// Compare the scalar values within the given tolerance
			Assert.assertEquals("Trace result mismatch", rTrace, dmlTrace, eps);
			Assert.assertTrue(heavyHittersContainsString("r'")!=rewriteEnabled);
		} 
		finally {
			OptimizerUtils.ALLOW_ALGEBRAIC_SIMPLIFICATION = oldFlag;
		}
	}
}
