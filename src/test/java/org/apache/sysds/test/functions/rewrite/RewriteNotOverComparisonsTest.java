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
import org.junit.Assert;
import org.junit.Test;
import org.apache.sysds.hops.OptimizerUtils;
import org.apache.sysds.runtime.matrix.data.MatrixValue.CellIndex;
import org.apache.sysds.test.AutomatedTestBase;
import org.apache.sysds.test.TestConfiguration;
import org.apache.sysds.test.TestUtils;
import org.apache.sysds.utils.Statistics;

public class RewriteNotOverComparisonsTest extends AutomatedTestBase {

	private static final String TEST_NAME = "RewriteNotOverComparisons";
	private static final String TEST_DIR = "functions/rewrite/";
	private static final String TEST_CLASS_DIR = TEST_DIR + RewriteNotOverComparisonsTest.class.getSimpleName() + "/";

	private static final int rows = 10;
	private static final int cols = 10;
	private static final double eps = Math.pow(10, -10);

	@Override
	public void setUp() {
		TestUtils.clearAssertionInformation();
		addTestConfiguration(TEST_NAME, new TestConfiguration(TEST_CLASS_DIR, TEST_NAME, new String[] {"R"}));
	}

	@Test
	public void testNotOverComparisonsGreaterNoRewrite() {
		testRewriteNotOverComparisons(1, false);
	}

	@Test
	public void testNotOverComparisonsGreaterRewrite() {
		testRewriteNotOverComparisons(1, true);
	}

	@Test
	public void testNotOverComparisonsLessNoRewrite() {
		testRewriteNotOverComparisons(2, false);
	}

	@Test
	public void testNotOverComparisonsLessRewrite() {
		testRewriteNotOverComparisons(2, true);
	}

	@Test
	public void testNotOverComparisonsEqualNoRewrite() {
		testRewriteNotOverComparisons(3, false);
	}

	@Test
	public void testNotOverComparisonsEqualRewrite() {
		testRewriteNotOverComparisons(3, true);
	}

	private void testRewriteNotOverComparisons(int ID, boolean rewrites) {
		boolean oldFlag = OptimizerUtils.ALLOW_ALGEBRAIC_SIMPLIFICATION;
		try {
			TestConfiguration config = getTestConfiguration(TEST_NAME);
			loadTestConfiguration(config);

			String HOME = SCRIPT_DIR + TEST_DIR;
			fullDMLScriptName = HOME + TEST_NAME + ".dml";
			programArgs = new String[] {"-stats", "-args",
				input("A"), input("B"), String.valueOf(ID), output("R")};

			fullRScriptName = HOME + TEST_NAME + ".R";
			rCmd = getRCmd(inputDir(), String.valueOf(ID), expectedDir());

			OptimizerUtils.ALLOW_ALGEBRAIC_SIMPLIFICATION = rewrites;

			double[][] A = getRandomMatrix(rows, cols, -1, 1, 0.70d, 7);
			writeInputMatrixWithMTD("A", A, 65, true);
			double[][] B = getRandomMatrix(rows, cols, -1, 1, 0.70d, 6);
			writeInputMatrixWithMTD("B", B, 74, true);
			
			runTest(true, false, null, -1);
			runRScript(true);

			//compare matrices
			HashMap<CellIndex, Double> dmlfile = readDMLMatrixFromOutputDir("R");
			HashMap<CellIndex, Double> rfile = readRMatrixFromExpectedDir("R");
			TestUtils.compareMatrices(dmlfile, rfile, eps, "Stat-DML", "Stat-R");

			long count = Statistics.getCPHeavyHitterCount(Opcodes.NOT.toString());
			Assert.assertTrue(count == (rewrites ? 0 : 1));
		}
		finally {
			OptimizerUtils.ALLOW_ALGEBRAIC_SIMPLIFICATION = oldFlag;
		}
	}
}
