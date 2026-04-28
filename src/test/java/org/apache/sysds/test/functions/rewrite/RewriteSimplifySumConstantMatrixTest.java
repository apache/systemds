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

import org.junit.Assert;
import org.junit.Test;
import org.apache.sysds.hops.OptimizerUtils;
import org.apache.sysds.runtime.matrix.data.MatrixValue.CellIndex;
import org.apache.sysds.test.AutomatedTestBase;
import org.apache.sysds.test.TestConfiguration;
import org.apache.sysds.test.TestUtils;

public class RewriteSimplifySumConstantMatrixTest extends AutomatedTestBase
{
	private static final String TEST_NAME = "RewriteSimplifySumConstantMatrix";
	private static final String TEST_DIR = "functions/rewrite/";
	private static final String TEST_CLASS_DIR = TEST_DIR + RewriteSimplifySumConstantMatrixTest.class.getSimpleName() + "/";

	@Override
	public void setUp() {
		TestUtils.clearAssertionInformation();
		addTestConfiguration(TEST_NAME, new TestConfiguration(TEST_CLASS_DIR, TEST_NAME, new String[] { "R" }));
	}

	@Test
	public void testSimplifySumConstantMatrixNoRewritePositive() {
		testRewriteSimplifySumConstantMatrix(2.5, 7, 11, false);
	}

	@Test
	public void testSimplifySumConstantMatrixRewritePositive() {
		testRewriteSimplifySumConstantMatrix(2.5, 7, 11, true);
	}


	private void testRewriteSimplifySumConstantMatrix(double value, long rows, long cols, boolean rewrites) {
		boolean oldFlag = OptimizerUtils.ALLOW_ALGEBRAIC_SIMPLIFICATION;

		try {
			TestConfiguration config = getTestConfiguration(TEST_NAME);
			loadTestConfiguration(config);

			String HOME = SCRIPT_DIR + TEST_DIR;
			fullDMLScriptName = HOME + TEST_NAME + ".dml";
			programArgs = new String[] {
				"-stats", "-args",
				String.valueOf(value), String.valueOf(rows), String.valueOf(cols), output("R")
			};

			fullRScriptName = HOME + TEST_NAME + ".R";
			rCmd = getRCmd(String.valueOf(value), String.valueOf(rows), String.valueOf(cols), expectedDir());

			OptimizerUtils.ALLOW_ALGEBRAIC_SIMPLIFICATION = rewrites;

			runTest(true, false, null, -1);
			runRScript(true);

			double actual = readDMLScalarFromOutputDir("R").get(new CellIndex(1, 1));
			double expected = readRScalarFromExpectedDir("R").get(new CellIndex(1, 1));
			Assert.assertEquals(expected, actual, 1e-15);

			if(rewrites)
				Assert.assertFalse(heavyHittersContainsString("rand"));
		}
		finally {
			OptimizerUtils.ALLOW_ALGEBRAIC_SIMPLIFICATION = oldFlag;
		}
	}
}
