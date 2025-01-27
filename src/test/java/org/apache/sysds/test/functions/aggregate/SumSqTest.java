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

package org.apache.sysds.test.functions.aggregate;

import org.apache.sysds.common.Opcodes;
import org.junit.Assert;
import org.junit.Test;
import org.apache.sysds.api.DMLScript;
import org.apache.sysds.common.Types.ExecMode;
import org.apache.sysds.hops.OptimizerUtils;
import org.apache.sysds.common.Types.ExecType;
import org.apache.sysds.runtime.instructions.Instruction;
import org.apache.sysds.runtime.matrix.data.MatrixValue.CellIndex;
import org.apache.sysds.test.AutomatedTestBase;
import org.apache.sysds.test.TestConfiguration;
import org.apache.sysds.test.TestUtils;
import org.apache.sysds.utils.Statistics;

import java.util.HashMap;

/**
 * Test the sum of squared values function, "sum(X^2)".
 */
public class SumSqTest extends AutomatedTestBase {

	private static final String TEST_NAME = "SumSq";
	private static final String TEST_DIR = "functions/aggregate/";
	private static final String TEST_CLASS_DIR = TEST_DIR + SumSqTest.class.getSimpleName() + "/";
	private static final String INPUT_NAME = "X";
	private static final String OUTPUT_NAME = "sumSq";

	private static final String op = Opcodes.UASQKP.toString();
	private static final int rows = 1234;
	private static final int cols = 567;
	private static final double sparsity1 = 1;
	private static final double sparsity2 = 0.2;
	private static final double eps = Math.pow(10, -10);

	@Override
	public void setUp() {
		TestUtils.clearAssertionInformation();
		TestConfiguration config = new TestConfiguration(TEST_CLASS_DIR, TEST_NAME);
		addTestConfiguration(TEST_NAME, config);
	}

	// Dense matrix w/ rewrites
	@Test
	public void testSumSquaredDenseMatrixRewriteCP() {
		testSumSquared(TEST_NAME, false, false, true, ExecType.CP);
	}

	@Test
	public void testSumSquaredDenseMatrixRewriteSpark() {
		testSumSquared(TEST_NAME, false, false, true, ExecType.SPARK);
	}

	// Dense matrix w/o rewrites
	@Test
	public void testSumSquaredDenseMatrixNoRewriteCP() {
		testSumSquared(TEST_NAME, false, false, false, ExecType.CP);
	}

	@Test
	public void testSumSquaredDenseMatrixNoRewriteSpark() {
		testSumSquared(TEST_NAME, false, false, false, ExecType.SPARK);
	}

	// Dense vector w/ rewrites
	@Test
	public void testSumSquaredDenseVectorRewriteCP() {
		testSumSquared(TEST_NAME, false, true, true, ExecType.CP);
	}

	@Test
	public void testSumSquaredDenseVectorRewriteSpark() {
		testSumSquared(TEST_NAME, false, true, true, ExecType.SPARK);
	}

	// Dense vector w/o rewrites
	@Test
	public void testSumSquaredDenseVectorNoRewriteCP() {
		testSumSquared(TEST_NAME, false, true, false, ExecType.CP);
	}

	@Test
	public void testSumSquaredDenseVectorNoRewriteSpark() {
		testSumSquared(TEST_NAME, false, true, false, ExecType.SPARK);
	}

	// Sparse matrix w/ rewrites
	@Test
	public void testSumSquaredSparseMatrixRewriteCP() {
		testSumSquared(TEST_NAME, true, false, true, ExecType.CP);
	}

	@Test
	public void testSumSquaredSparseMatrixRewriteSpark() {
		testSumSquared(TEST_NAME, true, false, true, ExecType.SPARK);
	}

	// Sparse matrix w/o rewrites
	@Test
	public void testSumSquaredSparseMatrixNoRewriteCP() {
		testSumSquared(TEST_NAME, true, false, false, ExecType.CP);
	}

	@Test
	public void testSumSquaredSparseMatrixNoRewriteSpark() {
		testSumSquared(TEST_NAME, true, false, false, ExecType.SPARK);
	}

	// Sparse vector w/ rewrites
	@Test
	public void testSumSquaredSparseVectorRewriteCP() {
		testSumSquared(TEST_NAME, true, true, true, ExecType.CP);
	}

	@Test
	public void testSumSquaredSparseVectorRewriteSpark() {
		testSumSquared(TEST_NAME, true, true, true, ExecType.SPARK);
	}

	// Sparse vector w/o rewrites
	@Test
	public void testSumSquaredSparseVectorNoRewriteCP() {
		testSumSquared(TEST_NAME, true, true, false, ExecType.CP);
	}

	@Test
	public void testSumSquaredSparseVectorNoRewriteSpark() {
		testSumSquared(TEST_NAME, true, true, false, ExecType.SPARK);
	}

	/**
	 * Test the sum of squared values function, "sum(X^2)", on dense/sparse matrices/vectors with rewrites/no rewrites on
	 * the CP/Spark/MR platforms.
	 *
	 * @param testName The name of this test case.
	 * @param sparse   Whether or not the matrix/vector should be sparse.
	 * @param vector   Boolean value choosing between a vector and a matrix.
	 * @param rewrites Whether or not to employ algebraic rewrites.
	 * @param platform Selection between CP/Spark/MR platforms.
	 */
	private void testSumSquared(String testName, boolean sparse, boolean vector, boolean rewrites, ExecType platform) {
		// Configure settings for this test case
		boolean rewritesOld = OptimizerUtils.ALLOW_ALGEBRAIC_SIMPLIFICATION;
		OptimizerUtils.ALLOW_ALGEBRAIC_SIMPLIFICATION = rewrites;

		ExecMode platformOld = rtplatform;
		switch(platform) {
			case SPARK:
				rtplatform = ExecMode.SPARK;
				break;
			default:
				rtplatform = ExecMode.SINGLE_NODE;
				break;
		}

		boolean sparkConfigOld = DMLScript.USE_LOCAL_SPARK_CONFIG;
		if(rtplatform == ExecMode.SPARK)
			DMLScript.USE_LOCAL_SPARK_CONFIG = true;

		try {
			// Create and load test configuration
			getAndLoadTestConfiguration(testName);
			String HOME = SCRIPT_DIR + TEST_DIR;
			fullDMLScriptName = HOME + testName + ".dml";
			programArgs = new String[] {"-explain", "-stats", "-args", input(INPUT_NAME), output(OUTPUT_NAME)};
			fullRScriptName = HOME + testName + ".R";
			rCmd = "Rscript" + " " + fullRScriptName + " " + inputDir() + " " + expectedDir();

			// Generate data
			double sparsity = sparse ? sparsity2 : sparsity1;
			int columns = vector ? 1 : cols;
			double[][] X = getRandomMatrix(rows, columns, -1, 1, sparsity, 7);
			writeInputMatrixWithMTD(INPUT_NAME, X, true);

			// Run DML and R scripts
			runTest(true, false, null, -1);
			runRScript(true);

			// Compare output matrices
			HashMap<CellIndex, Double> dmlfile = readDMLMatrixFromOutputDir(OUTPUT_NAME);
			HashMap<CellIndex, Double> rfile = readRMatrixFromExpectedDir(OUTPUT_NAME);
			TestUtils.compareMatrices(dmlfile, rfile, eps, "Stat-DML", "Stat-R");

			// On CP and Spark modes, check that the rewrite actually
			// occurred for matrix cases and not for vector cases.
			if(rewrites && (platform == ExecType.SPARK || platform == ExecType.CP)) {
				String prefix = (platform == ExecType.SPARK) ? Instruction.SP_INST_PREFIX : (DMLScript.USE_ACCELERATOR ? "gpu_" : "");
				String opcode = prefix + op;
				boolean rewriteApplied = Statistics.getCPHeavyHitterOpCodes().contains(opcode);
				if(vector)
					Assert.assertFalse("Rewrite applied to vector case.", rewriteApplied);
				else
					Assert.assertTrue("Rewrite not applied to matrix case.", rewriteApplied);
			}
		}
		finally {
			// Reset settings
			OptimizerUtils.ALLOW_ALGEBRAIC_SIMPLIFICATION = rewritesOld;
			rtplatform = platformOld;
			DMLScript.USE_LOCAL_SPARK_CONFIG = sparkConfigOld;
		}
	}
}
