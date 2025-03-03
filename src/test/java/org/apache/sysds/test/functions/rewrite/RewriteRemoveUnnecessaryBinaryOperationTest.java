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

public class RewriteRemoveUnnecessaryBinaryOperationTest extends AutomatedTestBase {

	private static final String TEST_NAME = "RewriteRemoveUnnecessaryBinaryOperation";
	private static final String TEST_DIR = "functions/rewrite/";
	private static final String TEST_CLASS_DIR =
		TEST_DIR + RewriteRemoveUnnecessaryBinaryOperationTest.class.getSimpleName() + "/";

	private static final int rows = 500;
	private static final int cols = 500;
	private static final double eps = Math.pow(10, -10);

	@Override
	public void setUp() {
		TestUtils.clearAssertionInformation();
		addTestConfiguration(TEST_NAME, new TestConfiguration(TEST_CLASS_DIR, TEST_NAME, new String[] {"R"}));
	}

	@Test
	public void testRemoveUnnecessaryBinaryOperationDivNoRewrite() {
		testRewriteRemoveUnnecessaryBinaryOperation(1, false);
	}

	@Test
	public void testRemoveUnnecessaryBinaryOperationDivRewrite() {
		testRewriteRemoveUnnecessaryBinaryOperation(1, true);    // X/1
	}

	@Test
	public void testRemoveUnnecessaryBinaryOperationMultRightNoRewrite() {
		testRewriteRemoveUnnecessaryBinaryOperation(2, false);
	}

	@Test
	public void testRemoveUnnecessaryBinaryOperationMultRightRewrite() {
		testRewriteRemoveUnnecessaryBinaryOperation(2, true);    // X*1
	}

	@Test
	public void testRemoveUnnecessaryBinaryOperationMultLeftNoRewrite() {
		testRewriteRemoveUnnecessaryBinaryOperation(3, false);
	}

	@Test
	public void testRemoveUnnecessaryBinaryOperationMultLeftRewrite() {
		testRewriteRemoveUnnecessaryBinaryOperation(3, true);    // 1*X
	}

	@Test
	public void testRemoveUnnecessaryBinaryOperationMinusNoRewrite() {
		testRewriteRemoveUnnecessaryBinaryOperation(4, false);
	}

	@Test
	public void testRemoveUnnecessaryBinaryOperationMinusRewrite() {
		testRewriteRemoveUnnecessaryBinaryOperation(4, true);    // X-0
	}

	@Test
	public void testRemoveUnnecessaryBinaryOperationNegMultLeftNoRewrite() {
		testRewriteRemoveUnnecessaryBinaryOperation(5, false);
	}

	@Test
	public void testRemoveUnnecessaryBinaryOperationNegMultLeftRewrite() {
		testRewriteRemoveUnnecessaryBinaryOperation(5, true);    // -1*X
	}

	@Test
	public void testRemoveUnnecessaryBinaryOperationNegMultRightNoRewrite() {
		testRewriteRemoveUnnecessaryBinaryOperation(6, false);
	}

	@Test
	public void testRemoveUnnecessaryBinaryOperationNegMultRightRewrite() {
		testRewriteRemoveUnnecessaryBinaryOperation(6, true);    // X*-1
	}

	private void testRewriteRemoveUnnecessaryBinaryOperation(int ID, boolean rewrites) {
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

			// create and write matrix
			double[][] X = getRandomMatrix(rows, cols, -1, 1, 0.70d, 5);
			writeInputMatrixWithMTD("X", X, true);

			runTest(true, false, null, -1);
			runRScript(true);

			//compare matrices
			HashMap<MatrixValue.CellIndex, Double> dmlfile = readDMLMatrixFromOutputDir("R");
			HashMap<MatrixValue.CellIndex, Double> rfile = readRMatrixFromExpectedDir("R");
			TestUtils.compareMatrices(dmlfile, rfile, eps, "Stat-DML", "Stat-R");

			if(ID == 1) {
				if(rewrites)
					Assert.assertFalse(heavyHittersContainsString(Opcodes.DIV.toString()));
				else
					Assert.assertTrue(heavyHittersContainsString(Opcodes.DIV.toString()));
			}
			else if(ID == 2 || ID == 3) {
				if(rewrites)
					Assert.assertFalse(heavyHittersContainsString(Opcodes.MULT.toString()));
				else
					Assert.assertTrue(heavyHittersContainsString(Opcodes.MULT.toString()));
			}
			else if(ID == 4) {
				if(rewrites)
					Assert.assertFalse(heavyHittersContainsString(Opcodes.MINUS.toString()));
				else
					Assert.assertTrue(heavyHittersContainsString(Opcodes.MINUS.toString()));
			}
			else if(ID == 5 || ID == 6) {
				if(rewrites)
					Assert.assertTrue(heavyHittersContainsString(Opcodes.MINUS.toString()) &&
						!heavyHittersContainsString(Opcodes.MULT.toString()));
				else
					Assert.assertTrue(!heavyHittersContainsString(Opcodes.MINUS.toString()) &&
						heavyHittersContainsString(Opcodes.MULT.toString()));
			}

		}
		finally {
			OptimizerUtils.ALLOW_ALGEBRAIC_SIMPLIFICATION = oldFlag;
		}
	}
}
