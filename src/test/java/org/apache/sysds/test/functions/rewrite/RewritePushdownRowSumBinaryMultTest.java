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

import org.junit.Test;
import org.apache.sysds.hops.OptimizerUtils;
import org.apache.sysds.runtime.matrix.data.MatrixValue.CellIndex;
import org.apache.sysds.test.AutomatedTestBase;
import org.apache.sysds.test.TestConfiguration;
import org.apache.sysds.test.TestUtils;
import org.apache.sysds.utils.Statistics;
import org.junit.Assert;

public class RewritePushdownRowSumBinaryMultTest extends AutomatedTestBase
{
	private static final String TEST_NAME1 = "RewritePushdownRowSumBinaryMult";
	private static final String TEST_NAME2 = "RewritePushdownRowSumBinaryMult2";

	private static final String TEST_DIR = "functions/rewrite/";
	private static final String TEST_CLASS_DIR = TEST_DIR + RewritePushdownRowSumBinaryMultTest.class.getSimpleName() + "/";

	@Override
	public void setUp() {
		TestUtils.clearAssertionInformation();
		addTestConfiguration(TEST_NAME1, new TestConfiguration(TEST_CLASS_DIR, TEST_NAME1, new String[] { "R" }));
		addTestConfiguration(TEST_NAME2, new TestConfiguration(TEST_CLASS_DIR, TEST_NAME2, new String[] { "R" }));
	}

	@Test
	public void testPushdownRowSumBinaryMultNoRewrite() {
		testRewritePushdownRowSumBinaryMult(TEST_NAME1, false);
	}

	@Test
	public void testPushdownRowSumBinaryMultRewrite() {
		testRewritePushdownRowSumBinaryMult(TEST_NAME1, true);
	}

	@Test
	public void testPushdownRowSumBinaryMultNoRewrite2() {
		testRewritePushdownRowSumBinaryMult(TEST_NAME2, false);
	}

	@Test
	public void testPushdownRowSumBinaryMultRewrite2() {
		testRewritePushdownRowSumBinaryMult(TEST_NAME2, true);
	}

	private void testRewritePushdownRowSumBinaryMult(String testname, boolean rewrites) {
		boolean oldFlag = OptimizerUtils.ALLOW_ALGEBRAIC_SIMPLIFICATION;

		try {
			TestConfiguration config = getTestConfiguration(testname);
			loadTestConfiguration(config);

			String HOME = SCRIPT_DIR + TEST_DIR;
			fullDMLScriptName = HOME + testname + ".dml";
			programArgs = new String[] { "-stats", "-args", output("R") };

			fullRScriptName = HOME + testname + ".R";
			rCmd = getRCmd(inputDir(), expectedDir());

			OptimizerUtils.ALLOW_ALGEBRAIC_SIMPLIFICATION = rewrites;

			runTest(true, false, null, -1);
			runRScript(true);

			HashMap<CellIndex, Double> dmlfile = readDMLMatrixFromOutputDir("R");
			HashMap<CellIndex, Double> rfile = readRMatrixFromExpectedDir("R");
			TestUtils.compareMatrices(dmlfile, rfile, 1e-10, "DML", "R");

			if(rewrites)
				Assert.assertEquals(1, Statistics.getCPHeavyHitterCount("n*"));
			else
				Assert.assertEquals(2, Statistics.getCPHeavyHitterCount("*"));
		}
		finally {
			OptimizerUtils.ALLOW_ALGEBRAIC_SIMPLIFICATION = oldFlag;
		}
	}
}
