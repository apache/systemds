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
import org.apache.sysds.utils.Statistics;
import org.junit.Assert;
import org.junit.Test;

import java.util.HashMap;

public class RewriteSimplifyTraceSumTest extends AutomatedTestBase {
	private static final String TEST_NAME = "RewriteSimplifyTraceSum";
	private static final String TEST_DIR = "functions/rewrite/";
	private static final String TEST_CLASS_DIR = TEST_DIR + RewriteSimplifyTraceSumTest.class.getSimpleName() + "/";

	private static final int rows = 500;
	private static final int cols = 500;
	private static final double eps = 1e-10;

	@Override
	public void setUp() {
		TestUtils.clearAssertionInformation();
		addTestConfiguration(TEST_NAME, new TestConfiguration(TEST_CLASS_DIR, TEST_NAME, new String[]{"R"}));
	}

	@Test
	public void testSimplifyTraceSumRewrite() {
		runTraceRewriteTest(TEST_NAME, true);
	}

	@Test
	public void testSimplifyTraceSumNoRewrite() {
		runTraceRewriteTest(TEST_NAME, false);
	}

	private void runTraceRewriteTest(String testname, boolean rewrites) {
		boolean oldFlag = OptimizerUtils.ALLOW_ALGEBRAIC_SIMPLIFICATION;
		try {
			TestConfiguration config = getTestConfiguration(testname);
			loadTestConfiguration(config);

			String HOME = SCRIPT_DIR + TEST_DIR;
			fullDMLScriptName = HOME + testname + ".dml";
			fullRScriptName = HOME + testname + ".R";

			programArgs = new String[]{"-explain", "-stats", "-args", input("A"), input("B"), output("R")};
			rCmd = getRCmd(inputDir(), expectedDir());
			OptimizerUtils.ALLOW_ALGEBRAIC_SIMPLIFICATION = rewrites;
			double[][] A = getRandomMatrix(rows, cols, -1, 1, 0.70d, 7);
			double[][] B = getRandomMatrix(cols, rows, -1, 1, 0.70d, 6);
			writeInputMatrixWithMTD("A", A, true);
			writeInputMatrixWithMTD("B", B, true);
			// Run SystemDS and R scripts
			runTest(true, false, null, -1);
			runRScript(true);

			// Compare DML and R outputs
			HashMap<MatrixValue.CellIndex, Double> dmlfile = readDMLScalarFromOutputDir("R");
			HashMap<MatrixValue.CellIndex, Double> rfile = readRScalarFromExpectedDir("R");

			// Ensure they're equal (within tolerance)
			TestUtils.compareMatrices(dmlfile, rfile, eps, "DMLResult", "RResult");
			Assert.assertEquals(rewrites?2:1, Statistics.getCPHeavyHitterCount("uaktrace"));
		} finally {
			OptimizerUtils.ALLOW_ALGEBRAIC_SIMPLIFICATION = oldFlag;
		}
	}
}
