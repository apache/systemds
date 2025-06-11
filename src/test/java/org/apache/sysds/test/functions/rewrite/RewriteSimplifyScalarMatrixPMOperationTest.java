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

import org.apache.sysds.common.Types;
import org.apache.sysds.hops.OptimizerUtils;
import org.apache.sysds.runtime.matrix.data.MatrixValue;
import org.apache.sysds.test.AutomatedTestBase;
import org.apache.sysds.test.TestConfiguration;
import org.apache.sysds.test.TestUtils;
import org.junit.Assert;
import org.junit.Test;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.HashMap;

public class RewriteSimplifyScalarMatrixPMOperationTest extends AutomatedTestBase {
	private static final String TEST_NAME1 = "RewriteScalarMinusMatrixMinusScalar";
	private static final String TEST_NAME2 = "RewriteScalarPlusMatrixMinusScalar";
	private static final String TEST_DIR = "functions/rewrite/";
	private static final String TEST_CLASS_DIR = TEST_DIR + RewriteSimplifyScalarMatrixPMOperationTest.class.getSimpleName() + "/";
	private static final int rows = 100;
	private static final int cols = 100;
	private static final double eps = 1e-6;

	@Override
	public void setUp() {
		TestUtils.clearAssertionInformation();
		addTestConfiguration(TEST_NAME1, new TestConfiguration(TEST_CLASS_DIR, TEST_NAME1, new String[]{"A", "a", "b", "R"}));
		addTestConfiguration(TEST_NAME2, new TestConfiguration(TEST_CLASS_DIR, TEST_NAME2, new String[]{"A", "a", "b", "R"}));
	}

	@Test
	public void testScalarMinusMatrixMinusScalarRewriteEnabled() {
		runRewriteTest(TEST_NAME1, true);
	}

	@Test
	public void testScalarMinusMatrixMinusScalarRewriteDisabled() {
		runRewriteTest(TEST_NAME1, false);
	}

	@Test
	public void testScalarPlusMatrixMinusScalarRewriteEnabled() {
		runRewriteTest(TEST_NAME2, true);
	}

	@Test
	public void testScalarPlusMatrixMinusScalarRewriteDisabled() {
		runRewriteTest(TEST_NAME2, false);
	}

	private void runRewriteTest(String testName, boolean rewriteEnabled) {
		boolean oldFlag = OptimizerUtils.ALLOW_ALGEBRAIC_SIMPLIFICATION;
		try {
			TestConfiguration config = getTestConfiguration(testName);
			loadTestConfiguration(config);

			String HOME = SCRIPT_DIR + TEST_DIR;
			fullDMLScriptName = HOME + testName + ".dml";
			fullRScriptName = HOME + testName + ".R";
			programArgs = new String[]{"-stats", "-args", input("A"), input("a"), input("b"), output("R")};
			rCmd = getRCmd(inputDir(), expectedDir());

			OptimizerUtils.ALLOW_ALGEBRAIC_SIMPLIFICATION = rewriteEnabled;

			double[][] A = getRandomMatrix(rows, cols, -100, 100, 0.9, 3);
			double[][] a = getRandomMatrix(1, 1, -10, 10, 1.0, 7);
			double[][] b = getRandomMatrix(1, 1, -10, 10, 1.0, 5);

			writeInputMatrixWithMTD("A", A, true);
			writeInputMatrixWithMTD("a", a, true);
			writeInputMatrixWithMTD("b", b, true);

			runTest(true, false, null, -1);
			runRScript(true);

			HashMap<MatrixValue.CellIndex, Double> dml = readDMLMatrixFromOutputDir("R");
			HashMap<MatrixValue.CellIndex, Double> r = readRMatrixFromExpectedDir("R");

			Assert.assertEquals("DML and R outputs do not match", r, dml);
		} finally {
			OptimizerUtils.ALLOW_ALGEBRAIC_SIMPLIFICATION = oldFlag;
		}
	}
}

