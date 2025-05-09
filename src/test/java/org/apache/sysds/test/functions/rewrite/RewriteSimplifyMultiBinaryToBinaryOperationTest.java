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

import org.apache.sysds.common.Opcodes;
import org.apache.sysds.hops.OptimizerUtils;
import org.apache.sysds.runtime.matrix.data.MatrixValue;
import org.apache.sysds.test.AutomatedTestBase;
import org.apache.sysds.test.TestConfiguration;
import org.apache.sysds.test.TestUtils;
import org.junit.Assert;
import org.junit.Test;

import java.util.HashMap;

public class RewriteSimplifyMultiBinaryToBinaryOperationTest extends AutomatedTestBase {

	private static final String TEST_NAME = "RewriteSimplifyMultiBinaryToBinaryOperation";
	private static final String TEST_DIR = "functions/rewrite/";
	private static final String TEST_CLASS_DIR =
		TEST_DIR + RewriteSimplifyMultiBinaryToBinaryOperationTest.class.getSimpleName() + "/";

	private static final int rows = 500;
	private static final int cols = 500;
	private static final double eps = Math.pow(10, -10);

	@Override
	public void setUp() {
		TestUtils.clearAssertionInformation();
		addTestConfiguration(TEST_NAME, new TestConfiguration(TEST_CLASS_DIR, TEST_NAME, new String[] {"R"}));
	}

	@Test
	public void testSimplifyMultiBinaryToBinaryOperationNoRewrite() {
		testRewriteSimplifyMultiBinaryToBinaryOperation(false);
	}

	@Test
	public void testSimplifyMultiBinaryToBinaryOperationRewrite() {
		testRewriteSimplifyMultiBinaryToBinaryOperation(true);
	}

	private void testRewriteSimplifyMultiBinaryToBinaryOperation(boolean rewrites) {
		boolean oldFlag1 = OptimizerUtils.ALLOW_ALGEBRAIC_SIMPLIFICATION;
		boolean oldFlag2 = OptimizerUtils.ALLOW_OPERATOR_FUSION;
		try {
			TestConfiguration config = getTestConfiguration(TEST_NAME);
			loadTestConfiguration(config);

			String HOME = SCRIPT_DIR + TEST_DIR;
			fullDMLScriptName = HOME + TEST_NAME + ".dml";
			programArgs = new String[] {"-stats", "-args", input("X"), input("Y"), output("R")};
			fullRScriptName = HOME + TEST_NAME + ".R";
			rCmd = getRCmd(inputDir(), expectedDir());

			OptimizerUtils.ALLOW_ALGEBRAIC_SIMPLIFICATION = rewrites;
			OptimizerUtils.ALLOW_OPERATOR_FUSION = rewrites;

			// create and write matrices
			double[][] X = getRandomMatrix(rows, cols, -1, 1, 0.70d, 5);
			double[][] Y = getRandomMatrix(rows, cols, -1, 1, 0.60d, 4);
			writeInputMatrixWithMTD("X", X, true);
			writeInputMatrixWithMTD("Y", Y, true);

			runTest(true, false, null, -1);
			runRScript(true);

			//compare matrices
			HashMap<MatrixValue.CellIndex, Double> dmlfile = readDMLMatrixFromOutputDir("R");
			HashMap<MatrixValue.CellIndex, Double> rfile = readRMatrixFromExpectedDir("R");
			TestUtils.compareMatrices(dmlfile, rfile, eps, "Stat-DML", "Stat-R");

			if(rewrites)
				Assert.assertTrue(heavyHittersContainsString(Opcodes.MINUS1_MULT.toString()));
			else
				Assert.assertTrue(heavyHittersContainsAllString(Opcodes.MINUS.toString(), Opcodes.MULT.toString()));
		}
		finally {
			OptimizerUtils.ALLOW_ALGEBRAIC_SIMPLIFICATION = oldFlag1;
			OptimizerUtils.ALLOW_OPERATOR_FUSION = oldFlag2;
		}
	}
}
