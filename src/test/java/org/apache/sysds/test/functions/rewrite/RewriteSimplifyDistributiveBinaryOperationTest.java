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
import org.apache.sysds.utils.Statistics;
import org.junit.Assert;
import org.junit.Test;

import java.util.HashMap;

public class RewriteSimplifyDistributiveBinaryOperationTest extends AutomatedTestBase {

	private static final String TEST_NAME = "RewriteSimplifyDistributiveBinaryOperation";
	private static final String TEST_DIR = "functions/rewrite/";
	private static final String TEST_CLASS_DIR =
		TEST_DIR + RewriteSimplifyDistributiveBinaryOperationTest.class.getSimpleName() + "/";

	private static final int rows = 300;
	private static final int cols = 300;
	private static final double eps = Math.pow(10, -10);

	@Override
	public void setUp() {
		TestUtils.clearAssertionInformation();
		addTestConfiguration(TEST_NAME, new TestConfiguration(TEST_CLASS_DIR, TEST_NAME, new String[] {"R"}));
	}

	@Test
	public void testDistrBinaryOpMinusMultNoRewrite() {
		testSimplifyDistributiveBinaryOperation(1, false);
	}

	@Test
	public void testDistrBinaryOpMinusMultRewrite() {
		testSimplifyDistributiveBinaryOperation(1, true);    //pattern: (X-Y*X) -> (1-Y)*X
	}

	@Test
	public void testDistrBinaryOpMultMinusNoRewrite() {
		testSimplifyDistributiveBinaryOperation(2, false);
	}

	@Test
	public void testDistrBinaryOpMultMinusRewrite() {
		testSimplifyDistributiveBinaryOperation(2, true);    //pattern: (Y*X-X) -> (Y-1)*X
	}

	@Test
	public void testDistrBinaryOpAddMultNoRewrite() {
		testSimplifyDistributiveBinaryOperation(3, false);
	}

	@Test
	public void testDistrBinaryOpAddMultRewrite() {
		testSimplifyDistributiveBinaryOperation(3, true);    //pattern: (X+Y*X) -> (1+Y)*X
	}

	@Test
	public void testDistrBinaryOpMultAddNoRewrite() {
		testSimplifyDistributiveBinaryOperation(4, false);
	}

	@Test
	public void testDistrBinaryOpMultAddRewrite() {
		testSimplifyDistributiveBinaryOperation(4, true);    //pattern: (Y*X+X) -> (Y+1)*X
	}
	
	@Test
	public void testDistrBinaryOpMultMinusVectorRewrite() {
		testSimplifyDistributiveBinaryOperation(5, true);    //pattern: (X*Y-X) -> (Y+1)*X, Y vector
	}

	private void testSimplifyDistributiveBinaryOperation(int ID, boolean rewrites) {
		boolean oldFlag1 = OptimizerUtils.ALLOW_ALGEBRAIC_SIMPLIFICATION;
		try {
			TestConfiguration config = getTestConfiguration(TEST_NAME);
			loadTestConfiguration(config);

			String HOME = SCRIPT_DIR + TEST_DIR;
			fullDMLScriptName = HOME + TEST_NAME + ".dml";
			programArgs = new String[] {"-stats", "-args", input("X"), input("Y"), String.valueOf(ID), output("R")};
			fullRScriptName = HOME + TEST_NAME + ".R";
			rCmd = getRCmd(inputDir(), String.valueOf(ID), expectedDir());

			OptimizerUtils.ALLOW_ALGEBRAIC_SIMPLIFICATION = rewrites;
			OptimizerUtils.ALLOW_OPERATOR_FUSION = rewrites;

			//create matrices
			double[][] X = getRandomMatrix(rows, cols, -1, 1, 0.60d, 3);
			double[][] Y = getRandomMatrix(rows, ID==5?1:cols, -1, 1, 0.60d, 5);
			writeInputMatrixWithMTD("X", X, true);
			writeInputMatrixWithMTD("Y", Y, true);

			runTest(true, false, null, -1);
			runRScript(true);

			//compare matrices
			HashMap<MatrixValue.CellIndex, Double> dmlfile = readDMLMatrixFromOutputDir("R");
			HashMap<MatrixValue.CellIndex, Double> rfile = readRMatrixFromExpectedDir("R");
			TestUtils.compareMatrices(dmlfile, rfile, eps, "Stat-DML", "Stat-R");

			/**
			 * We add '*1' to the statemements in the dml file to ensure a difference in the
			 * heavy hitter count for the rewritten statements.
			 */

			if(rewrites)
				Assert.assertTrue(Statistics.getCPHeavyHitterCount(Opcodes.MULT.toString()) == 1);
			else
				Assert.assertTrue(Statistics.getCPHeavyHitterCount(Opcodes.MULT.toString()) == 2);

		}
		finally {
			OptimizerUtils.ALLOW_ALGEBRAIC_SIMPLIFICATION = oldFlag1;
		}

	}
}
