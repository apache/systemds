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

public class RewriteSimplifyUnaryPPredOperationTest extends AutomatedTestBase {

	private static final String TEST_NAME = "RewriteSimplifyUnaryPPredOperation";
	private static final String TEST_DIR = "functions/rewrite/";
	private static final String TEST_CLASS_DIR =
		TEST_DIR + RewriteSimplifyUnaryPPredOperationTest.class.getSimpleName() + "/";

	private static final int rows = 500;
	private static final int cols = 500;
	private static final double eps = Math.pow(10, -10);

	@Override
	public void setUp() {
		TestUtils.clearAssertionInformation();
		addTestConfiguration(TEST_NAME, new TestConfiguration(TEST_CLASS_DIR, TEST_NAME, new String[] {"R"}));
	}

	/**
	 * (1) Rewrites for Less + {abs, round, ceil, floor, sign}
	 */
	@Test
	public void testSimplifyUnaryPPredOperationLessAbsNoRewrite() {
		testRewriteSimplifyUnaryPPredOperation(1, false);
	}

	@Test
	public void testSimplifyUnaryPPredOperationLessAbsRewrite() {
		testRewriteSimplifyUnaryPPredOperation(1, true);    // abs(X<Y) -> (X<Y)
	}

	@Test
	public void testSimplifyUnaryPPredOperationLessRoundNoRewrite() {
		testRewriteSimplifyUnaryPPredOperation(2, false);
	}

	@Test
	public void testSimplifyUnaryPPredOperationLessRoundRewrite() {
		testRewriteSimplifyUnaryPPredOperation(2, true);    // round(X<Y) -> (X<Y)
	}

	@Test
	public void testSimplifyUnaryPPredOperationLessCeilNoRewrite() {
		testRewriteSimplifyUnaryPPredOperation(3, false);
	}

	@Test
	public void testSimplifyUnaryPPredOperationLessCeilRewrite() {
		testRewriteSimplifyUnaryPPredOperation(3, true);    // ceil(X<Y) -> (X<Y)
	}

	@Test
	public void testSimplifyUnaryPPredOperationLessFloorNoRewrite() {
		testRewriteSimplifyUnaryPPredOperation(4, false);
	}

	@Test
	public void testSimplifyUnaryPPredOperationLessFloorRewrite() {
		testRewriteSimplifyUnaryPPredOperation(4, true);    // floor(X<Y) -> (X<Y)
	}

	@Test
	public void testSimplifyUnaryPPredOperationLessSignNoRewrite() {
		testRewriteSimplifyUnaryPPredOperation(5, false);
	}

	@Test
	public void testSimplifyUnaryPPredOperationLessSignRewrite() {
		testRewriteSimplifyUnaryPPredOperation(5, true);    // sign(X<Y) -> (X<Y)
	}

	/**
	 * (2) Rewrites for LessEqual + {abs, round, ceil, floor, sign}
	 */
	@Test
	public void testSimplifyUnaryPPredOperationLessEqualAbsNoRewrite() {
		testRewriteSimplifyUnaryPPredOperation(6, false);
	}

	@Test
	public void testSimplifyUnaryPPredOperationLessEqualAbsRewrite() {
		testRewriteSimplifyUnaryPPredOperation(6, true);    // abs(X<=Y) -> (X<=Y)
	}

	@Test
	public void testSimplifyUnaryPPredOperationLessEqualRoundNoRewrite() {
		testRewriteSimplifyUnaryPPredOperation(7, false);
	}

	@Test
	public void testSimplifyUnaryPPredOperationLessEqualRoundRewrite() {
		testRewriteSimplifyUnaryPPredOperation(7, true);    // round(X<=Y) -> (X<=Y)
	}

	@Test
	public void testSimplifyUnaryPPredOperationLessEqualCeilNoRewrite() {
		testRewriteSimplifyUnaryPPredOperation(8, false);
	}

	@Test
	public void testSimplifyUnaryPPredOperationLessEqualCeilRewrite() {
		testRewriteSimplifyUnaryPPredOperation(8, true);    // ceil(X<=Y) -> (X<=Y)
	}

	@Test
	public void testSimplifyUnaryPPredOperationLessEqualFloorNoRewrite() {
		testRewriteSimplifyUnaryPPredOperation(9, false);
	}

	@Test
	public void testSimplifyUnaryPPredOperationLessEqualFloorRewrite() {
		testRewriteSimplifyUnaryPPredOperation(9, true);    // floor(X<=Y) -> (X<=Y)
	}

	@Test
	public void testSimplifyUnaryPPredOperationLessEqualSignNoRewrite() {
		testRewriteSimplifyUnaryPPredOperation(10, false);
	}

	@Test
	public void testSimplifyUnaryPPredOperationLessEqualSignRewrite() {
		testRewriteSimplifyUnaryPPredOperation(10, true);    // sign(X<=Y) -> (X<=Y)
	}

	/**
	 * (3) Rewrites for Greater + {abs, round, ceil, floor, sign}
	 */
	@Test
	public void testSimplifyUnaryPPredOperationGreaterAbsNoRewrite() {
		testRewriteSimplifyUnaryPPredOperation(11, false);
	}

	@Test
	public void testSimplifyUnaryPPredOperationGreaterAbsRewrite() {
		testRewriteSimplifyUnaryPPredOperation(11, true);    // abs(X>Y) -> (X>Y)
	}

	@Test
	public void testSimplifyUnaryPPredOperationGreaterRoundNoRewrite() {
		testRewriteSimplifyUnaryPPredOperation(12, false);
	}

	@Test
	public void testSimplifyUnaryPPredOperationGreaterRoundRewrite() {
		testRewriteSimplifyUnaryPPredOperation(12, true);    // round(X>Y) -> (X>Y)
	}

	@Test
	public void testSimplifyUnaryPPredOperationGreaterCeilNoRewrite() {
		testRewriteSimplifyUnaryPPredOperation(13, false);
	}

	@Test
	public void testSimplifyUnaryPPredOperationGreaterCeilRewrite() {
		testRewriteSimplifyUnaryPPredOperation(13, true);    // ceil(X>Y) -> (X>Y)
	}

	@Test
	public void testSimplifyUnaryPPredOperationGreaterFloorNoRewrite() {
		testRewriteSimplifyUnaryPPredOperation(14, false);
	}

	@Test
	public void testSimplifyUnaryPPredOperationGreaterFloorRewrite() {
		testRewriteSimplifyUnaryPPredOperation(14, true);    // floor(X>Y) -> (X>Y)
	}

	@Test
	public void testSimplifyUnaryPPredOperationGreaterSignNoRewrite() {
		testRewriteSimplifyUnaryPPredOperation(15, false);
	}

	@Test
	public void testSimplifyUnaryPPredOperationGreaterSignRewrite() {
		testRewriteSimplifyUnaryPPredOperation(15, true);    // sign(X>Y) -> (X>Y)
	}

	/**
	 * (4) Rewrites for GreaterEqual + {abs, round, ceil, floor, sign}
	 */
	@Test
	public void testSimplifyUnaryPPredOperationGreaterEqualAbsNoRewrite() {
		testRewriteSimplifyUnaryPPredOperation(16, false);
	}

	@Test
	public void testSimplifyUnaryPPredOperationGreaterEqualAbsRewrite() {
		testRewriteSimplifyUnaryPPredOperation(16, true);    // abs(X>=Y) -> (X>=Y)
	}

	@Test
	public void testSimplifyUnaryPPredOperationGreaterEqualRoundNoRewrite() {
		testRewriteSimplifyUnaryPPredOperation(17, false);
	}

	@Test
	public void testSimplifyUnaryPPredOperationGreaterEqualRoundRewrite() {
		testRewriteSimplifyUnaryPPredOperation(17, true);    // round(X>=Y) -> (X>=Y)
	}

	@Test
	public void testSimplifyUnaryPPredOperationGreaterEqualCeilNoRewrite() {
		testRewriteSimplifyUnaryPPredOperation(18, false);
	}

	@Test
	public void testSimplifyUnaryPPredOperationGreaterEqualCeilRewrite() {
		testRewriteSimplifyUnaryPPredOperation(18, true);    // ceil(X>=Y) -> (X>=Y)
	}

	@Test
	public void testSimplifyUnaryPPredOperationGreaterEqualFloorNoRewrite() {
		testRewriteSimplifyUnaryPPredOperation(19, false);
	}

	@Test
	public void testSimplifyUnaryPPredOperationGreaterEqualFloorRewrite() {
		testRewriteSimplifyUnaryPPredOperation(19, true);    // floor(X>=Y) -> (X>=Y)
	}

	@Test
	public void testSimplifyUnaryPPredOperationGreaterEqualSignNoRewrite() {
		testRewriteSimplifyUnaryPPredOperation(20, false);
	}

	@Test
	public void testSimplifyUnaryPPredOperationGreaterEqualSignRewrite() {
		testRewriteSimplifyUnaryPPredOperation(20, true);    // sign(X>=Y) -> (X>=Y)
	}

	/**
	 * (5) Rewrites for Equal + {abs, round, ceil, floor, sign}
	 */
	@Test
	public void testSimplifyUnaryPPredOperationEqualAbsNoRewrite() {
		testRewriteSimplifyUnaryPPredOperation(21, false);
	}

	@Test
	public void testSimplifyUnaryPPredOperationEqualAbsRewrite() {
		testRewriteSimplifyUnaryPPredOperation(21, true);    // abs(X==Y) -> (X==Y)
	}

	@Test
	public void testSimplifyUnaryPPredOperationEqualRoundNoRewrite() {
		testRewriteSimplifyUnaryPPredOperation(22, false);
	}

	@Test
	public void testSimplifyUnaryPPredOperationEqualRoundRewrite() {
		testRewriteSimplifyUnaryPPredOperation(22, true);    // round(X==Y) -> (X==Y)
	}

	@Test
	public void testSimplifyUnaryPPredOperationEqualCeilNoRewrite() {
		testRewriteSimplifyUnaryPPredOperation(23, false);
	}

	@Test
	public void testSimplifyUnaryPPredOperationEqualCeilRewrite() {
		testRewriteSimplifyUnaryPPredOperation(23, true);    // ceil(X==Y) -> (X==Y)
	}

	@Test
	public void testSimplifyUnaryPPredOperationEqualFloorNoRewrite() {
		testRewriteSimplifyUnaryPPredOperation(24, false);
	}

	@Test
	public void testSimplifyUnaryPPredOperationEqualFloorRewrite() {
		testRewriteSimplifyUnaryPPredOperation(24, true);    // floor(X==Y) -> (X==Y)
	}

	@Test
	public void testSimplifyUnaryPPredOperationEqualSignNoRewrite() {
		testRewriteSimplifyUnaryPPredOperation(25, false);
	}

	@Test
	public void testSimplifyUnaryPPredOperationEqualSignRewrite() {
		testRewriteSimplifyUnaryPPredOperation(25, true);    // sign(X==Y) -> (X==Y)
	}

	/**
	 * (6) Rewrites for NotEqual + {abs, round, ceil, floor, sign}
	 */
	@Test
	public void testSimplifyUnaryPPredOperationNotEqualAbsNoRewrite() {
		testRewriteSimplifyUnaryPPredOperation(26, false);
	}

	@Test
	public void testSimplifyUnaryPPredOperationNotEqualAbsRewrite() {
		testRewriteSimplifyUnaryPPredOperation(26, true);    // abs(X!=Y) -> (X!=Y)
	}

	@Test
	public void testSimplifyUnaryPPredOperationNotEqualRoundNoRewrite() {
		testRewriteSimplifyUnaryPPredOperation(27, false);
	}

	@Test
	public void testSimplifyUnaryPPredOperationNotEqualRoundRewrite() {
		testRewriteSimplifyUnaryPPredOperation(27, true);    // round(X!=Y) -> (X!=Y)
	}

	@Test
	public void testSimplifyUnaryPPredOperationNotEqualCeilNoRewrite() {
		testRewriteSimplifyUnaryPPredOperation(28, false);
	}

	@Test
	public void testSimplifyUnaryPPredOperationNotEqualCeilRewrite() {
		testRewriteSimplifyUnaryPPredOperation(28, true);    // ceil(X!=Y) -> (X!=Y)
	}

	@Test
	public void testSimplifyUnaryPPredOperationNotEqualFloorNoRewrite() {
		testRewriteSimplifyUnaryPPredOperation(29, false);
	}

	@Test
	public void testSimplifyUnaryPPredOperationNotEqualFloorRewrite() {
		testRewriteSimplifyUnaryPPredOperation(29, true);    // floor(X!=Y) -> (X!=Y)
	}

	@Test
	public void testSimplifyUnaryPPredOperationNotEqualSignNoRewrite() {
		testRewriteSimplifyUnaryPPredOperation(30, false);
	}

	@Test
	public void testSimplifyUnaryPPredOperationNotEqualSignRewrite() {
		testRewriteSimplifyUnaryPPredOperation(30, true);    // sign(X!=Y) -> (X!=Y)
	}

	private void testRewriteSimplifyUnaryPPredOperation(int ID, boolean rewrites) {
		boolean oldFlag = OptimizerUtils.ALLOW_ALGEBRAIC_SIMPLIFICATION;
		try {
			TestConfiguration config = getTestConfiguration(TEST_NAME);
			loadTestConfiguration(config);

			String HOME = SCRIPT_DIR + TEST_DIR;
			fullDMLScriptName = HOME + TEST_NAME + ".dml";
			programArgs = new String[] {"-stats", "-args", input("X"), input("Y"), String.valueOf(ID), output("R")};
			fullRScriptName = HOME + TEST_NAME + ".R";
			rCmd = getRCmd(inputDir(), String.valueOf(ID), expectedDir());

			OptimizerUtils.ALLOW_ALGEBRAIC_SIMPLIFICATION = rewrites;

			// create and write matrices
			double[][] X = getRandomMatrix(rows, cols, -1, 1, 0.70d, 5);
			double[][] Y = getRandomMatrix(rows, cols, -1, 1, 0.60d, 3);
			writeInputMatrixWithMTD("X", X, true);
			writeInputMatrixWithMTD("Y", Y, true);

			runTest(true, false, null, -1);
			runRScript(true);

			//compare matrices
			HashMap<MatrixValue.CellIndex, Double> dmlfile = readDMLMatrixFromOutputDir("R");
			HashMap<MatrixValue.CellIndex, Double> rfile = readRMatrixFromExpectedDir("R");
			TestUtils.compareMatrices(dmlfile, rfile, eps, "Stat-DML", "Stat-R");

			if(rewrites)
				Assert.assertFalse(heavyHittersContainsString(Opcodes.ABS.toString(), Opcodes.ROUND.toString(),
					Opcodes.CEIL.toString(), Opcodes.FLOOR.toString(), Opcodes.SIGN.toString()));
			else
				Assert.assertTrue(heavyHittersContainsString(Opcodes.ABS.toString(), Opcodes.ROUND.toString(),
					Opcodes.CEIL.toString(), Opcodes.FLOOR.toString(), Opcodes.SIGN.toString()));

		}
		finally {
			OptimizerUtils.ALLOW_ALGEBRAIC_SIMPLIFICATION = oldFlag;
		}
	}

}
