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

public class RewriteSimplifyWeightedSquaredLossTest extends AutomatedTestBase {
	private static final String TEST_NAME = "RewriteSimplifyWeightedSquaredLoss";
	private static final String TEST_DIR = "functions/rewrite/";
	private static final String TEST_CLASS_DIR =
		TEST_DIR + RewriteSimplifyWeightedSquaredLossTest.class.getSimpleName() + "/";

	private static final int rows = 100;
	private static final int cols = 100;
	private static final double eps = Math.pow(10, -10);

	@Override
	public void setUp() {
		TestUtils.clearAssertionInformation();
		addTestConfiguration(TEST_NAME, new TestConfiguration(TEST_CLASS_DIR, TEST_NAME, new String[] {"R"}));
	}

	@Test
	public void testSquaredLossPostWeightingANoRewrite() {
		testRewriteSimplifyWeightedSquaredLoss(1, false);
	}

	@Test
	public void testSquaredLossPostWeightingARewrite() {
		testRewriteSimplifyWeightedSquaredLoss(1, true); //pattern: sum(W * (X - U %*% t(V)) ^ 2)
	}

	@Test
	public void testSquaredLossPostWeightingBNoRewrite() {
		testRewriteSimplifyWeightedSquaredLoss(2, false);
	}

	@Test
	public void testSquaredLossPostWeightingBRewrite() {
		testRewriteSimplifyWeightedSquaredLoss(2, true); //pattern: sum(W * (U %*% t(V) - X) ^ 2)
	}

	@Test
	public void testSquaredLossPreWeightingANoRewrite() {
		testRewriteSimplifyWeightedSquaredLoss(3, false);
	}

	@Test
	public void testSquaredLossPreWeightingARewrite() {
		testRewriteSimplifyWeightedSquaredLoss(3, true); //pattern: sum((X - W * (U %*% t(V)))^2)
	}

	@Test
	public void testSquaredLossPreWeightingBNoRewrite() {
		testRewriteSimplifyWeightedSquaredLoss(4, false);
	}

	@Test
	public void testSquaredLossPreWeightingBRewrite() {
		testRewriteSimplifyWeightedSquaredLoss(4, true); //pattern: sum((W * (U %*% t(V)) - X)^2)
	}

	@Test
	public void testSquaredLossNoWeightingANoRewrite() {
		testRewriteSimplifyWeightedSquaredLoss(5, false);
	}

	@Test
	public void testSquaredLossNoWeightingARewrite() {
		testRewriteSimplifyWeightedSquaredLoss(5, true); //pattern:  sum((X - (U %*% t(V)))^2)
	}

	@Test
	public void testSquaredLossNoWeightingBNoRewrite() {
		testRewriteSimplifyWeightedSquaredLoss(6, false);
	}

	@Test
	public void testSquaredLossNoWeightingBRewrite() {
		testRewriteSimplifyWeightedSquaredLoss(6, true); //pattern:  sum(((U %*% t(V)) - X)^2)
	}

	private void testRewriteSimplifyWeightedSquaredLoss(int ID, boolean rewrites) {
		boolean oldFlag1 = OptimizerUtils.ALLOW_ALGEBRAIC_SIMPLIFICATION;
		boolean oldFlag2 = OptimizerUtils.ALLOW_OPERATOR_FUSION;
		try {
			TestConfiguration config = getTestConfiguration(TEST_NAME);
			loadTestConfiguration(config);

			String HOME = SCRIPT_DIR + TEST_DIR;
			fullDMLScriptName = HOME + TEST_NAME + ".dml";
			programArgs = new String[] {"-stats", "-args", input("X"), input("U"), input("V"), input("W"),
				String.valueOf(ID), output("R")};
			fullRScriptName = HOME + TEST_NAME + ".R";
			rCmd = getRCmd(inputDir(), String.valueOf(ID), expectedDir());

			OptimizerUtils.ALLOW_ALGEBRAIC_SIMPLIFICATION = rewrites;
			OptimizerUtils.ALLOW_OPERATOR_FUSION = rewrites;

			//create matrices
			double[][] X = getRandomMatrix(rows, cols, -1, 1, 0.80d, 3);
			double[][] U = getRandomMatrix(rows, cols, -1, 1, 0.70d, 4);
			double[][] V = getRandomMatrix(rows, cols, -1, 1, 0.650d, 6);
			double[][] W = getRandomMatrix(rows, cols, -1, 1, 0.60d, 5);
			writeInputMatrixWithMTD("X", X, true);
			writeInputMatrixWithMTD("U", U, true);
			writeInputMatrixWithMTD("V", V, true);
			writeInputMatrixWithMTD("W", W, true);

			runTest(true, false, null, -1);
			runRScript(true);

			//compare matrices
			HashMap<MatrixValue.CellIndex, Double> dmlfile = readDMLScalarFromOutputDir("R");
			HashMap<MatrixValue.CellIndex, Double> rfile = readRScalarFromExpectedDir("R");
			TestUtils.compareMatrices(dmlfile, rfile, eps, "Stat-DML", "Stat-R");

			if(rewrites)
				Assert.assertTrue(heavyHittersContainsString(Opcodes.WSLOSS.toString()));
			else
				Assert.assertFalse(heavyHittersContainsString(Opcodes.WSLOSS.toString()));

		}
		finally {
			OptimizerUtils.ALLOW_ALGEBRAIC_SIMPLIFICATION = oldFlag1;
			OptimizerUtils.ALLOW_OPERATOR_FUSION = oldFlag2;
		}

	}
}
