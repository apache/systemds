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

public class RewritePushdownBinaryOperationOnDiagTest extends AutomatedTestBase {
	private static final String TEST_NAME = "RewritePushdownBinaryOperationOnDiag";
	private static final String TEST_DIR = "functions/rewrite/";
	private static final String TEST_CLASS_DIR =
		TEST_DIR + RewritePushdownBinaryOperationOnDiagTest.class.getSimpleName() + "/";

	private static final int rows = 100;
	private static final int cols = 1;
	private static final double eps = Math.pow(10, -10);

	@Override
	public void setUp() {
		TestUtils.clearAssertionInformation();
		addTestConfiguration(TEST_NAME, new TestConfiguration(TEST_CLASS_DIR, TEST_NAME, new String[] {"R"}));
	}

	@Test
	public void testPushdownBinaryOperationOnDiagLeftNoRewrite() {
		testRewritePushdownBinaryOperationOnDiag(1, false);
	}

	@Test
	public void testPushdownBinaryOperationOnDiagLeftRewrite() {    //pattern: diag(X) * 7 -> diag(X*7)
		testRewritePushdownBinaryOperationOnDiag(1, true);
	}

	@Test
	public void testPushdownBinaryOperationOnDiagRightNoRewrite() {
		testRewritePushdownBinaryOperationOnDiag(2, false);
	}

	@Test
	public void testPushdownBinaryOperationOnDiagRightRewrite() {    //pattern: 7 * diag(X) -> diag(X*7)
		testRewritePushdownBinaryOperationOnDiag(2, true);
	}

	private void testRewritePushdownBinaryOperationOnDiag(int ID, boolean rewrites) {
		boolean oldFlag = OptimizerUtils.ALLOW_ALGEBRAIC_SIMPLIFICATION;
		try {
			TestConfiguration config = getTestConfiguration(TEST_NAME);
			loadTestConfiguration(config);

			String HOME = SCRIPT_DIR + TEST_DIR;
			fullDMLScriptName = HOME + TEST_NAME + ".dml";
			programArgs = new String[] {"-stats", "-args", input("X"), String.valueOf(ID), output("R")};
			fullRScriptName = HOME + TEST_NAME + ".R";
			rCmd = getRCmd(inputDir(), String.valueOf(ID), expectedDir());

			OptimizerUtils.ALLOW_ALGEBRAIC_SIMPLIFICATION = rewrites;

			//create matrices
			double[][] X = getRandomMatrix(rows, cols, -1, 1, 0.80d, 3);
			writeInputMatrixWithMTD("X", X, true);

			runTest(true, false, null, -1);
			runRScript(true);

			//compare matrices
			HashMap<MatrixValue.CellIndex, Double> dmlfile = readDMLMatrixFromOutputDir("R");
			HashMap<MatrixValue.CellIndex, Double> rfile = readRMatrixFromExpectedDir("R");
			TestUtils.compareMatrices(dmlfile, rfile, eps, "Stat-DML", "Stat-R");

			/**
			 * In this rewrite, the heavy hitters of the rewritten statement and the non-rewritten one
			 * are identical. We manually checked that the rewrite is correctly executed. To automate
			 * the testing, we added a redundant operation in the dml script (i.e., * 1) to the result matrix.
			 * If rewrites are activated, then a heavy hitter '*' will show up only once instead of twice.
			 */

			String elementwiseMult = "*";
			if(rewrites)
				Assert.assertTrue(Statistics.getCPHeavyHitterCount(elementwiseMult) == 1);
			else
				Assert.assertTrue(Statistics.getCPHeavyHitterCount(elementwiseMult) == 2);

		}
		finally {
			OptimizerUtils.ALLOW_ALGEBRAIC_SIMPLIFICATION = oldFlag;
		}

	}
}
