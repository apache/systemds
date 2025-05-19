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

public class RewriteSimplifyBinaryMatrixScalarOperationTest extends AutomatedTestBase {

	private static final String TEST_NAME = "RewriteSimplifyBinaryMatrixScalarOperation";
	private static final String TEST_DIR = "functions/rewrite/";
	private static final String TEST_CLASS_DIR =
		TEST_DIR + RewriteSimplifyBinaryMatrixScalarOperationTest.class.getSimpleName() + "/";

	private static final double eps = Math.pow(10, -10);

	@Override
	public void setUp() {
		TestUtils.clearAssertionInformation();
		addTestConfiguration(TEST_NAME, new TestConfiguration(TEST_CLASS_DIR, TEST_NAME, new String[] {"R"}));
	}

	@Test
	public void testSimplifyBinaryMatrixScalarOperationMMNoRewrite() {
		testRewriteSimplifyBinaryMatrixScalarOperation(1, false);
	}

	@Test
	public void testSimplifyBinaryMatrixScalarOperationMMRewrite() {
		testRewriteSimplifyBinaryMatrixScalarOperation(1, true);    //as.scalar(X*Y) -> as.scalar(X) * as.scalar(Y)
	}

	@Test
	public void testSimplifyBinaryMatrixScalarOperationRightNoRewrite() {
		testRewriteSimplifyBinaryMatrixScalarOperation(2, false);
	}

	@Test
	public void testSimplifyBinaryMatrixScalarOperationRightRewrite() {
		testRewriteSimplifyBinaryMatrixScalarOperation(2, true);    // as.scalar(X*s) -> as.scalar(X) * s
	}

	@Test
	public void testSimplifyBinaryMatrixScalarOperationLeftNoRewrite() {
		testRewriteSimplifyBinaryMatrixScalarOperation(3, false);
	}

	@Test
	public void testSimplifyBinaryMatrixScalarOperationLeftRewrite() {
		testRewriteSimplifyBinaryMatrixScalarOperation(3, true);    // as.scalar(s*X) -> s * as.scalar(X)
	}

	private void testRewriteSimplifyBinaryMatrixScalarOperation(int ID, boolean rewrites) {
		boolean oldFlag = OptimizerUtils.ALLOW_ALGEBRAIC_SIMPLIFICATION;
		try {
			TestConfiguration config = getTestConfiguration(TEST_NAME);
			loadTestConfiguration(config);

			String HOME = SCRIPT_DIR + TEST_DIR;
			fullDMLScriptName = HOME + TEST_NAME + ".dml";
			programArgs = new String[] {"-stats", "-args", String.valueOf(ID), output("R")};
			fullRScriptName = HOME + TEST_NAME + ".R";
			rCmd = getRCmd(inputDir(), String.valueOf(ID), expectedDir());

			OptimizerUtils.ALLOW_ALGEBRAIC_SIMPLIFICATION = rewrites;

			runTest(true, false, null, -1);
			runRScript(true);

			//compare matrices
			HashMap<MatrixValue.CellIndex, Double> dmlfile = readDMLScalarFromOutputDir("R");
			HashMap<MatrixValue.CellIndex, Double> rfile = readRScalarFromExpectedDir("R");
			TestUtils.compareMatrices(dmlfile, rfile, eps, "Stat-DML", "Stat-R");

			long numCastDts = Statistics.getCPHeavyHitterCount(Opcodes.CAST_AS_SCALAR.toString());
			if(ID == 1) {
				if(rewrites)
					Assert.assertEquals(2, numCastDts);
				else
					Assert.assertEquals(1, numCastDts);
			}
			else if(ID == 2) {
				if(rewrites)
					Assert.assertTrue(heavyHittersContainsString(Opcodes.MULT.toString()));
				else
					Assert.assertTrue(heavyHittersContainsString(Opcodes.MULT2.toString()));
			}
			else if(ID == 3) {
				if(rewrites)
					Assert.assertTrue(heavyHittersContainsString(Opcodes.NM.toString()));
				else
					Assert.assertTrue(heavyHittersContainsString(Opcodes.MULT.toString()));
			}
		}
		finally {
			OptimizerUtils.ALLOW_ALGEBRAIC_SIMPLIFICATION = oldFlag;
		}
	}
}
