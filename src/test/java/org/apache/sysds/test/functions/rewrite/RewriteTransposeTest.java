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
import org.apache.sysds.utils.Statistics;
import org.junit.Assert;
import org.junit.Test;
import java.util.HashMap;

public class RewriteTransposeTest extends AutomatedTestBase {
	private final static String TEST_NAME1 = "RewriteTransposeCase1"; // t(X)%*%Y
	private final static String TEST_NAME2 = "RewriteTransposeCase2"; // X=t(A); t(X)%*%Y
	private final static String TEST_NAME3 = "RewriteTransposeCase3"; // Y=t(A); t(X)%*%Y

	private final static String TEST_DIR = "functions/rewrite/";
	private static final String TEST_CLASS_DIR = TEST_DIR + RewriteTransposeTest.class.getSimpleName() + "/";

	private static final double eps = 1e-9;

	@Override
	public void setUp() {
		OptimizerUtils.ALLOW_ALGEBRAIC_SIMPLIFICATION=false;

		addTestConfiguration(TEST_NAME1, new TestConfiguration(TEST_CLASS_DIR, TEST_NAME1, new String[]{"R"}));
		addTestConfiguration(TEST_NAME2, new TestConfiguration(TEST_CLASS_DIR, TEST_NAME2, new String[]{"R"}));
		addTestConfiguration(TEST_NAME3, new TestConfiguration(TEST_CLASS_DIR, TEST_NAME3, new String[]{"R"}));
	}

	@Test
	public void testTransposeRewrite1CP() {
		runTransposeRewriteTest(TEST_NAME1, false);
	}

	@Test
	public void testTransposeRewrite2CP() {
		runTransposeRewriteTest(TEST_NAME2, true);
	}

	@Test
	public void testTransposeRewrite3CP() {
		runTransposeRewriteTest(TEST_NAME3, false);
	}

	private void runTransposeRewriteTest(String testname, boolean expectedMerge) {
		TestConfiguration config = getTestConfiguration(testname);
		loadTestConfiguration(config);

		String HOME = SCRIPT_DIR + TEST_DIR;
		fullDMLScriptName = HOME + testname + ".dml";

		programArgs = new String[]{"-explain", "-stats", "-args", output("R")};

		fullRScriptName = HOME + testname + ".R";
		rCmd = getRCmd(expectedDir());

		runTest(true, false, null, -1);
		runRScript(true);

		HashMap<MatrixValue.CellIndex, Double> dmlOutput = readDMLMatrixFromOutputDir("R");
		HashMap<MatrixValue.CellIndex, Double> rOutput = readRMatrixFromExpectedDir("R");
		TestUtils.compareMatrices(dmlOutput, rOutput, eps, "Stat-DML", "Stat-R");
		
		Assert.assertTrue(Statistics.getCPHeavyHitterCount("r'") <= 2);
	}
}
