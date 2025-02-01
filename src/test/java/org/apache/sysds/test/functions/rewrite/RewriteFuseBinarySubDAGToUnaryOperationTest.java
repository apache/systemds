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

public class RewriteFuseBinarySubDAGToUnaryOperationTest extends AutomatedTestBase {

	private static final String TEST_NAME = "RewriteFuseBinarySubDAGToUnaryOperation";
	private static final String TEST_DIR = "functions/rewrite/";
	private static final String TEST_CLASS_DIR =
		TEST_DIR + RewriteFuseBinarySubDAGToUnaryOperationTest.class.getSimpleName() + "/";

	private static final int rows = 300;
	private static final int cols = 200;
	private static final double eps = Math.pow(10, -10);

	@Override
	public void setUp() {
		TestUtils.clearAssertionInformation();
		addTestConfiguration(TEST_NAME, new TestConfiguration(TEST_CLASS_DIR, TEST_NAME, new String[] {"R"}));
	}

	@Test
	public void testSampleProportionLeftNoRewrite(){
		testSimplifyDistributiveBinaryOperation(1, false);
	}

	@Test
	public void testSampleProportionLeftRewrite(){
		testSimplifyDistributiveBinaryOperation(1, true);	//pattern: (1-X)*X -> sprop(X)
	}

	@Test
	public void testSampleProportionRightNoRewrite(){
		testSimplifyDistributiveBinaryOperation(2, false);
	}

	@Test
	public void testSampleProportionRightRewrite(){
		testSimplifyDistributiveBinaryOperation(2, true);	//pattern: X*(1-X) -> sprop(X)
	}

	@Test
	public void testFuseBinarySubDAGToUnarySigmoidNoRewrite(){
		testSimplifyDistributiveBinaryOperation(3, false);
	}

	@Test
	public void testFuseBinarySubDAGToUnarySigmoidRewrite(){
		testSimplifyDistributiveBinaryOperation(3, true);	//pattern: 1/(1+exp(-X)) -> sigmoid(X)
	}


	private void testSimplifyDistributiveBinaryOperation(int ID, boolean rewrites) {
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

			//create dense matrix so that rewrites are possible
			double[][] X = getRandomMatrix(rows, cols, -1, 1, 0.80d, 3);
			writeInputMatrixWithMTD("X", X, true);

			runTest(true, false, null, -1);
			runRScript(true);

			//compare matrices
			HashMap<MatrixValue.CellIndex, Double> dmlfile = readDMLMatrixFromOutputDir("R");
			HashMap<MatrixValue.CellIndex, Double> rfile = readRMatrixFromExpectedDir("R");
			TestUtils.compareMatrices(dmlfile, rfile, eps, "Stat-DML", "Stat-R");

			if (rewrites)
				Assert.assertTrue(heavyHittersContainsString(Opcodes.SPROP.toString()) || heavyHittersContainsString(Opcodes.SIGMOID.toString()));
			else
				Assert.assertFalse(heavyHittersContainsString(Opcodes.SPROP.toString()) || heavyHittersContainsString(Opcodes.SIGMOID.toString()));
		}
		finally {
			OptimizerUtils.ALLOW_ALGEBRAIC_SIMPLIFICATION = oldFlag;
		}
	}
}
