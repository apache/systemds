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

import org.junit.Assert;
import org.junit.Test;
import org.apache.sysds.hops.OptimizerUtils;
import org.apache.sysds.runtime.matrix.data.MatrixValue.CellIndex;
import org.apache.sysds.test.AutomatedTestBase;
import org.apache.sysds.test.TestConfiguration;
import org.apache.sysds.test.TestUtils;
import org.apache.sysds.utils.Statistics;

public class RewriteDistributiveMatrixMultTest extends AutomatedTestBase {
	private static final String TEST_NAME1 = "RewriteDistributiveMatrixMult";
	private static final String TEST_DIR = "functions/rewrite/";
	private static final String TEST_CLASS_DIR =
		TEST_DIR + RewriteSimplifyRowColSumMVMultTest.class.getSimpleName() + "/";

	private static final int rows = 500;
	private static final int cols = 500;
	private static final double eps = Math.pow(10, -10);

	@Override
	public void setUp() {
		TestUtils.clearAssertionInformation();
		addTestConfiguration(TEST_NAME1, new TestConfiguration(TEST_CLASS_DIR, TEST_NAME1, new String[] {"R"}));

	}

	@Test
	public void testDistributiveMatrixMultNoRewrite() {
		testRewriteDistributiveMatrixMult(TEST_NAME1, false);
	}

	@Test
	public void testDistributiveMatrixMultRewrite() {
		testRewriteDistributiveMatrixMult(TEST_NAME1, true);
	}

	private void testRewriteDistributiveMatrixMult(String testname, boolean rewrites) {
		boolean oldFlag = OptimizerUtils.ALLOW_ALGEBRAIC_SIMPLIFICATION;
		try {
			TestConfiguration config = getTestConfiguration(testname);
			loadTestConfiguration(config);

			String HOME = SCRIPT_DIR + TEST_DIR;
			fullDMLScriptName = HOME + testname + ".dml";
			programArgs = new String[] {"-stats", "-args", input("A"), input("B"), input("C"), output("R")};

			fullRScriptName = HOME + testname + ".R";
			rCmd = getRCmd(inputDir(), expectedDir());

			OptimizerUtils.ALLOW_ALGEBRAIC_SIMPLIFICATION = rewrites;
			//create dense matrices so that rewrites are possible
			double[][] A = getRandomMatrix(rows, cols, -1, 1, 0.70d, 7);
			double[][] B = getRandomMatrix(rows, cols, -1, 1, 0.70d, 6);
			double[][] C = getRandomMatrix(rows, cols, -1, 1, 0.70d, 3);
			writeInputMatrixWithMTD("A", A, 174522, true);
			writeInputMatrixWithMTD("B", B, 174935, true);
			writeInputMatrixWithMTD("C", C, 174848, true);

			runTest(true, false, null, -1);
			runRScript(true);

			//compare matrices
			HashMap<CellIndex, Double> dmlfile = readDMLMatrixFromOutputDir("R");
			HashMap<CellIndex, Double> rfile = readRMatrixFromExpectedDir("R");
			TestUtils.compareMatrices(dmlfile, rfile, eps, "Stat-DML", "Stat-R");

			//check matrix mult existence
			String ba = "ba+*";
			long numMatMul = Statistics.getCPHeavyHitterCount(ba);

			if(rewrites == true) {
				Assert.assertTrue(numMatMul == 1);
			}
			else {
				Assert.assertTrue(numMatMul == 2);
			}

		}
		finally {
			OptimizerUtils.ALLOW_ALGEBRAIC_SIMPLIFICATION = oldFlag;
		}

	}
}
