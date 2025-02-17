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

import java.util.HashMap;

import org.apache.sysds.common.Opcodes;
import org.apache.sysds.hops.OptimizerUtils;
import org.apache.sysds.runtime.matrix.data.MatrixValue;
import org.apache.sysds.test.AutomatedTestBase;
import org.apache.sysds.test.TestConfiguration;
import org.apache.sysds.test.TestUtils;
import org.junit.Assert;
import org.junit.Test;

public class RewriteSimplifyWeightedSigmoidMMChainsTest extends AutomatedTestBase {
	private static final String TEST_NAME = "RewriteSimplifyWeightedSigmoidMMChains";
	private static final String TEST_DIR = "functions/rewrite/";
	private static final String TEST_CLASS_DIR =
		TEST_DIR + RewriteSimplifyWeightedSigmoidMMChainsTest.class.getSimpleName() + "/";

	private static final int rows = 150;
	private static final int cols = 100;

	@Override
	public void setUp() {
		TestUtils.clearAssertionInformation();
		addTestConfiguration(TEST_NAME, new TestConfiguration(TEST_CLASS_DIR, TEST_NAME, new String[] {"R"}));
	}

	@Test
	public void testWeightedSigmoidMMChainsBasicNoRewrite() {
		testRewriteSimplifyWeightedSigmoidMMChains(1, false);
	}

	@Test
	public void testWeightedSigmoidMMChainsBasicRewrite() {
		testRewriteSimplifyWeightedSigmoidMMChains(1, true);    //pattern: W * sigmoid(Y%*%t(X))
	}

	@Test
	public void testWeightedSigmoidMMChainsMinusNoRewrite() {
		testRewriteSimplifyWeightedSigmoidMMChains(2, false);
	}

	@Test
	public void testWeightedSigmoidMMChainsMinusRewrite() {
		testRewriteSimplifyWeightedSigmoidMMChains(2, true);    //pattern: W * sigmoid(-(Y%*%t(X)))
	}

	@Test
	public void testWeightedSigmoidMMChainsLogNoRewrite() {
		testRewriteSimplifyWeightedSigmoidMMChains(3, false);
	}

	@Test
	public void testWeightedSigmoidMMChainsLogRewrite() {
		testRewriteSimplifyWeightedSigmoidMMChains(3, true);    //pattern: W * log(sigmoid(Y%*%t(X)))
	}

	@Test
	public void testWeightedSigmoidMMChainsLogMinusNoRewrite() {
		testRewriteSimplifyWeightedSigmoidMMChains(4, false);
	}

	@Test
	public void testWeightedSigmoidMMChainsLogMinusRewrite() {
		testRewriteSimplifyWeightedSigmoidMMChains(4, true);    //pattern: W * log(sigmoid(-(Y%*%t(X))))
	}

	/**
	 * The following tests try to cover the special cases when the transposition is missing. In that case, the
	 * corresponding rewrite should catch this mistake and manually replace 'X' with 't(X)'.
	 */

	@Test
	public void testWeightedSigmoidNoTransposeBasic() {
		testRewriteSimplifyWeightedSigmoidMMChains(5, true);    //pattern: W * sigmoid(Y%*%X)
	}

	@Test
	public void testWeightedSigmoidNoTransposeMinus() {
		testRewriteSimplifyWeightedSigmoidMMChains(6, true);    //pattern: W * sigmoid(-(Y%*%X))
	}

	@Test
	public void testWeightedSigmoidNoTransposeLog() {
		testRewriteSimplifyWeightedSigmoidMMChains(7, true);    //pattern: W * log(sigmoid(Y%*%X))
	}

	@Test
	public void testWeightedSigmoidNoTransposeLogMinus() {
		testRewriteSimplifyWeightedSigmoidMMChains(8, true);    //pattern: W * log(sigmoid(-(Y%*%X)))
	}

	private void testRewriteSimplifyWeightedSigmoidMMChains(int ID, boolean rewrites) {
		boolean oldFlag1 = OptimizerUtils.ALLOW_ALGEBRAIC_SIMPLIFICATION;
		boolean oldFlag2 = OptimizerUtils.ALLOW_OPERATOR_FUSION;
		try {
			TestConfiguration config = getTestConfiguration(TEST_NAME);
			loadTestConfiguration(config);

			String HOME = SCRIPT_DIR + TEST_DIR;
			fullDMLScriptName = HOME + TEST_NAME + ".dml";
			programArgs = new String[] {"-stats", "-args", input("X"), input("Y"), input("W"), String.valueOf(ID),
				output("R")};
			fullRScriptName = HOME + TEST_NAME + ".R";
			rCmd = getRCmd(inputDir(), String.valueOf(ID), expectedDir());

			OptimizerUtils.ALLOW_ALGEBRAIC_SIMPLIFICATION = rewrites;
			OptimizerUtils.ALLOW_OPERATOR_FUSION = rewrites;

			//create matrices
			int rank = 50;
			double[][] X = getRandomMatrix(cols, rank, -1, 1, 0.80d, 3);
			double[][] Y = getRandomMatrix(rows, rank, -1, 1, 0.70d, 4);
			double[][] W = getRandomMatrix(rows, cols, -1, 1, 0.60d, 5);
			writeInputMatrixWithMTD("X", X, true);
			writeInputMatrixWithMTD("Y", Y, true);
			writeInputMatrixWithMTD("W", W, true);

			runTest(true, false, null, -1);
			runRScript(true);

			//compare matrices
			HashMap<MatrixValue.CellIndex, Double> dmlfile = readDMLMatrixFromOutputDir("R");
			HashMap<MatrixValue.CellIndex, Double> rfile = readRMatrixFromExpectedDir("R");
			TestUtils.compareMatrices(dmlfile, rfile, 1e-8, "Stat-DML", "Stat-R");

			if(rewrites)
				Assert.assertTrue(heavyHittersContainsString(Opcodes.WSIGMOID.toString()));
			else
				Assert.assertFalse(heavyHittersContainsString(Opcodes.WSIGMOID.toString()));
		}
		finally {
			OptimizerUtils.ALLOW_ALGEBRAIC_SIMPLIFICATION = oldFlag1;
			OptimizerUtils.ALLOW_OPERATOR_FUSION = oldFlag2;
		}

	}
}
