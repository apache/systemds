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
import org.apache.sysds.hops.recompile.Recompiler;
import org.apache.sysds.runtime.matrix.data.MatrixValue;
import org.apache.sysds.test.AutomatedTestBase;
import org.apache.sysds.test.TestConfiguration;
import org.apache.sysds.test.TestUtils;
import org.junit.Assert;
import org.junit.Test;

import java.util.HashMap;

public class RewriteSimplifyBushyBinaryOperationTest extends AutomatedTestBase {

	private static final String TEST_NAME = "RewriteSimplifyBushyBinaryOperation";
	private static final String TEST_DIR = "functions/rewrite/";
	private static final String TEST_CLASS_DIR =
		TEST_DIR + RewriteSimplifyBushyBinaryOperationTest.class.getSimpleName() + "/";

	private static final int rows = 500;
	private static final int cols = 100;
	private static final double eps = Math.pow(10, -10);

	@Override
	public void setUp() {
		TestUtils.clearAssertionInformation();
		addTestConfiguration(TEST_NAME, new TestConfiguration(TEST_CLASS_DIR, TEST_NAME, new String[] {"R"}));
	}

	//pattern: t(Z)%*%(X*(Y*(Z%*%v))) -> t(Z)%*%((X*Y)*(Z%*%v))
	@Test
	public void testBushyBinaryOperationMultNoRewrite() {
		testSimplifyBushyBinaryOperation(1, false);
	}

	@Test
	public void testBushyBinaryOperationMultRewrite() { 
		testSimplifyBushyBinaryOperation(1, true);
	}

	//pattern: t(Z)%*%(X+(Y+(Z%*%v))) -> t(Z)%*%((X+Y)+(Z%*%v))
	@Test
	public void testBushyBinaryOperationAddNoRewrite() {
		testSimplifyBushyBinaryOperation(2, false);
	}

	@Test
	public void testBushyBinaryOperationAddtRewrite() { 
		testSimplifyBushyBinaryOperation(2, true);
	}

	private void testSimplifyBushyBinaryOperation(int ID, boolean rewrites) {
		boolean oldFlag = OptimizerUtils.ALLOW_ALGEBRAIC_SIMPLIFICATION;
		try {
			TestConfiguration config = getTestConfiguration(TEST_NAME);
			loadTestConfiguration(config);

			String HOME = SCRIPT_DIR + TEST_DIR;
			fullDMLScriptName = HOME + TEST_NAME + ".dml";
			programArgs = new String[] {"-stats", "-explain", "-args", 
				input("X"), input("Y"), input("Z"), input("v"), String.valueOf(ID), output("R")};
			fullRScriptName = HOME + TEST_NAME + ".R";
			rCmd = getRCmd(inputDir(), String.valueOf(ID), expectedDir());

			OptimizerUtils.ALLOW_ALGEBRAIC_SIMPLIFICATION = rewrites;
			OptimizerUtils.ALLOW_SUM_PRODUCT_REWRITES2 = false; //disable nary mult
			OptimizerUtils.ALLOW_OPERATOR_FUSION = false; //disable emult reordering
			//TODO improved phase ordering 
			
			//create matrices
			double[][] X = getRandomMatrix(rows, 1, -1, 1, 0.60d, 3);
			double[][] Y = getRandomMatrix(rows, 1, -1, 1, 0.60d, 5);
			double[][] Z = getRandomMatrix(rows, cols, -1, 1, 0.60d, 6);
			double[][] v = getRandomMatrix(cols, 1, -1, 1, 0.60d, 8);
			writeInputMatrixWithMTD("X", X, true);
			writeInputMatrixWithMTD("Y", Y, true);
			writeInputMatrixWithMTD("Z", Z, true);
			writeInputMatrixWithMTD("v", v, true);

			runTest(true, false, null, -1);
			runRScript(true);

			//compare matrices
			HashMap<MatrixValue.CellIndex, Double> dmlfile = readDMLMatrixFromOutputDir("R");
			HashMap<MatrixValue.CellIndex, Double> rfile = readRMatrixFromExpectedDir("R");
			TestUtils.compareMatrices(dmlfile, rfile, eps, "Stat-DML", "Stat-R");
		
			if( ID == 1 && rewrites ) //check mmchain, enabled by bushy join 
				Assert.assertTrue(heavyHittersContainsString(Opcodes.MMCHAIN.toString()));
		}
		finally {
			OptimizerUtils.ALLOW_ALGEBRAIC_SIMPLIFICATION = oldFlag;
			OptimizerUtils.ALLOW_OPERATOR_FUSION = true;
			OptimizerUtils.ALLOW_SUM_PRODUCT_REWRITES2 = true;
			Recompiler.reinitRecompiler();
		}
	}
}
