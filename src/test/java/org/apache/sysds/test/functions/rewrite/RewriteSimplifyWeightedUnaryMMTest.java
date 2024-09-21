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

public class RewriteSimplifyWeightedUnaryMMTest extends AutomatedTestBase {
	private static final String TEST_NAME = "RewriteSimplifyWeightedUnaryMM";
	private static final String TEST_DIR = "functions/rewrite/";
	private static final String TEST_CLASS_DIR =
		TEST_DIR + RewriteSimplifyWeightedUnaryMMTest.class.getSimpleName() + "/";

	private static final int rows = 100;
	private static final int cols = 100;
	private static final double eps = Math.pow(10, -10);

	@Override
	public void setUp() {
		TestUtils.clearAssertionInformation();
		addTestConfiguration(TEST_NAME, new TestConfiguration(TEST_CLASS_DIR, TEST_NAME, new String[] {"R"}));
	}

	/**
	 * These tests cover the case for the first pattern
	 * W*uop(U%*%t(V)), where uop stands for unary operator.
	 * Unary operators are e.g., abs(), exp(), sin(), log(),
	 * ceil() among others
	 * */

	@Test
	public void testWeightedUnaryMMExpNoRewrite(){
		testRewriteSimplifyWeightedUnaryMM(1, false);
	}

	@Test
	public void testWeightedUnaryMMExpRewrite(){
		testRewriteSimplifyWeightedUnaryMM(1, true);	//pattern: W * exp(U%*%t(V))
	}

	@Test
	public void testWeightedUnaryMMAbsNoRewrite(){
		testRewriteSimplifyWeightedUnaryMM(2, false);
	}

	@Test
	public void testWeightedUnaryMMAbsRewrite(){
		testRewriteSimplifyWeightedUnaryMM(2, true);	//pattern: W * abs(U%*%t(V))
	}

	@Test
	public void testWeightedUnaryMMSinNoRewrite(){
		testRewriteSimplifyWeightedUnaryMM(3, false);
	}

	@Test
	public void testWeightedUnaryMMSinRewrite(){
		testRewriteSimplifyWeightedUnaryMM(3, true);	//pattern: W * sin(U%*%t(V))
	}

	/**
	 * This second group of tests covers weighted matrix multiply case with an
	 * additional scalar multiplication with factor 2.
	 * */

	@Test
	public void testWeightedUnaryMMScalarRightNoRewrite(){
		testRewriteSimplifyWeightedUnaryMM(4, false);
	}

	@Test
	public void testWeightedUnaryMMScalarRightRewrite(){
		testRewriteSimplifyWeightedUnaryMM(4, true);	//pattern: (W*(U%*%t(V)))*2
	}

	@Test
	public void testWeightedUnaryMMScalarLeftNoRewrite(){
		testRewriteSimplifyWeightedUnaryMM(5, false);
	}

	@Test
	public void testWeightedUnaryMMScalarLeftRewrite(){
		testRewriteSimplifyWeightedUnaryMM(5, true);	//pattern: 2*(W*(U%*%t(V)))
	}

	/**
	 * These tests cover the case for the third pattern
	 * W * sop(U%*%t(V), c) or W * sop(U%*%t(V), c), where
	 * sop stands for scalar operation (+, -, *, /) and c represents
	 * some constant scalar.
	 * */

	@Test
	public void testWeightedUnaryMMAddLeftNoRewrite(){
		testRewriteSimplifyWeightedUnaryMM(6, false);
	}

	@Test
	public void testWeightedUnaryMMAddLeftRewrite(){
		testRewriteSimplifyWeightedUnaryMM(6, true);	//pattern: W * (c + U%*%t(V))
	}

	@Test
	public void testWeightedUnaryMMMinusLeftNoRewrite(){
		testRewriteSimplifyWeightedUnaryMM(7, false);
	}

	@Test
	public void testWeightedUnaryMMMinusLeftRewrite(){
		testRewriteSimplifyWeightedUnaryMM(7, true);	//pattern: W * (c - U%*%t(V))
	}

	@Test
	public void testWeightedUnaryMMMultLeftNoRewrite(){
		testRewriteSimplifyWeightedUnaryMM(8, false);
	}

	@Test
	public void testWeightedUnaryMMMultLeftRewrite(){
		testRewriteSimplifyWeightedUnaryMM(8, true);	//pattern: W * (c * (U%*%t(V)))
	}

	@Test
	public void testWeightedUnaryMMDivLeftNoRewrite(){
		testRewriteSimplifyWeightedUnaryMM(9, false);
	}

	@Test
	public void testWeightedUnaryMMDivLeftRewrite(){
		testRewriteSimplifyWeightedUnaryMM(9, true);	//pattern: W * (c / (U%*%t(V)))
	}

	// Same pattern but scalar from right instead of left

	@Test
	public void testWeightedUnaryMMAddRightNoRewrite(){
		testRewriteSimplifyWeightedUnaryMM(10, false);
	}

	@Test
	public void testWeightedUnaryMMAddRightRewrite(){
		testRewriteSimplifyWeightedUnaryMM(10, true);	//pattern: W * (U%*%t(V) + c)
	}

	@Test
	public void testWeightedUnaryMMMinusRightNoRewrite(){
		testRewriteSimplifyWeightedUnaryMM(11, false);
	}

	@Test
	public void testWeightedUnaryMMMinusRightRewrite(){
		testRewriteSimplifyWeightedUnaryMM(11, true);	//pattern: W * (U%*%t(V) - c)
	}

	@Test
	public void testWeightedUnaryMMMulRightNoRewrite(){
		testRewriteSimplifyWeightedUnaryMM(12, false);
	}

	@Test
	public void testWeightedUnaryMMMultRightRewrite(){
		testRewriteSimplifyWeightedUnaryMM(12, true);	//pattern: W * ((U%*%t(V)) * c)
	}

	@Test
	public void testWeightedUnaryMMDivRightNoRewrite(){
		testRewriteSimplifyWeightedUnaryMM(13, false);
	}

	@Test
	public void testWeightedUnaryMMDivRightRewrite(){
		testRewriteSimplifyWeightedUnaryMM(13, true);	//pattern: W * ((U%*%t(V)) / c)
	}


	private void testRewriteSimplifyWeightedUnaryMM(int ID, boolean rewrites) {
		boolean oldFlag1 = OptimizerUtils.ALLOW_ALGEBRAIC_SIMPLIFICATION;
		boolean oldFlag2 = OptimizerUtils.ALLOW_OPERATOR_FUSION;
		try {
			TestConfiguration config = getTestConfiguration(TEST_NAME);
			loadTestConfiguration(config);

			String HOME = SCRIPT_DIR + TEST_DIR;
			fullDMLScriptName = HOME + TEST_NAME + ".dml";
			programArgs = new String[] {"-stats", "-args", input("U"), input("V"), input("W"), String.valueOf(ID),
				output("R")};
			fullRScriptName = HOME + TEST_NAME + ".R";
			rCmd = getRCmd(inputDir(), String.valueOf(ID), expectedDir());

			OptimizerUtils.ALLOW_ALGEBRAIC_SIMPLIFICATION = rewrites;
			OptimizerUtils.ALLOW_OPERATOR_FUSION = rewrites;

			//create matrices
			double[][] U = getRandomMatrix(rows, cols, -1, 1, 0.80d, 3);
			double[][] V = getRandomMatrix(rows, cols, -1, 1, 0.70d, 4);
			double[][] W = getRandomMatrix(rows, cols, -1, 1, 0.60d, 5);
			writeInputMatrixWithMTD("U", U, true);
			writeInputMatrixWithMTD("V", V, true);
			writeInputMatrixWithMTD("W", W, true);

			runTest(true, false, null, -1);
			runRScript(true);

			//compare matrices
			HashMap<MatrixValue.CellIndex, Double> dmlfile = readDMLMatrixFromOutputDir("R");
			HashMap<MatrixValue.CellIndex, Double> rfile = readRMatrixFromExpectedDir("R");
			TestUtils.compareMatrices(dmlfile, rfile, eps, "Stat-DML", "Stat-R");

			if(rewrites)
				Assert.assertTrue(heavyHittersContainsString("wumm"));
			else
				Assert.assertFalse(heavyHittersContainsString("wumm"));

		}
		finally {
			OptimizerUtils.ALLOW_ALGEBRAIC_SIMPLIFICATION = oldFlag1;
			OptimizerUtils.ALLOW_OPERATOR_FUSION = oldFlag2;
		}

	}

}
