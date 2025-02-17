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
 * Test the column variances function, "colVars(X)".
 */
public class ColVariancesTest extends AutomatedTestBase {

	private static final String TEST_NAME = "ColVariances";
	private static final String TEST_DIR = "functions/aggregate/";
	private static final String TEST_CLASS_DIR = TEST_DIR + ColVariancesTest.class.getSimpleName() + "/";
	private static final String INPUT_NAME = "X";
	private static final String OUTPUT_NAME = "colVariances";

	private static final String colVarOp = Opcodes.UACVAR.toString();
	private static final String varOp = Opcodes.UAVAR.toString();
	private static final int rows = 1234;
	private static final int cols = 1432;
	private static final double sparsitySparse = 0.2;
	private static final double sparsityDense = 0.7;
	private static final double eps = Math.pow(10, -10);

	private enum Sparsity {
		EMPTY, SPARSE, DENSE
	}

	private enum DataType {
		MATRIX, ROWVECTOR, COLUMNVECTOR
	}

	@Override
	public void setUp() {
		TestUtils.clearAssertionInformation();
		TestConfiguration config = new TestConfiguration(TEST_CLASS_DIR, TEST_NAME);
		addTestConfiguration(TEST_NAME, config);
	}

	// Dense matrix w/ rewrites
	@Test
	public void testColVariancesDenseMatrixRewritesCP() {
		testColVariances(TEST_NAME, Sparsity.DENSE, DataType.MATRIX, true, ExecType.CP);
	}

	@Test
	public void testColVariancesDenseMatrixRewritesSpark() {
		testColVariances(TEST_NAME, Sparsity.DENSE, DataType.MATRIX, true, ExecType.SPARK);
	}

	// Dense matrix w/o rewrites
	@Test
	public void testColVariancesDenseMatrixNoRewritesCP() {
		testColVariances(TEST_NAME, Sparsity.DENSE, DataType.MATRIX, false, ExecType.CP);
	}

	@Test
	public void testColVariancesDenseMatrixNoRewritesSpark() {
		testColVariances(TEST_NAME, Sparsity.DENSE, DataType.MATRIX, false, ExecType.SPARK);
	}

	// Dense row vector w/ rewrites
	@Test
	public void testColVariancesDenseRowVectorRewritesCP() {
		testColVariances(TEST_NAME, Sparsity.DENSE, DataType.ROWVECTOR, true, ExecType.CP);
	}

	@Test
	public void testColVariancesDenseRowVectorRewritesSpark() {
		testColVariances(TEST_NAME, Sparsity.DENSE, DataType.ROWVECTOR, true, ExecType.SPARK);
	}

	// Dense row vector w/o rewrites
	@Test
	public void testColVariancesDenseRowVectorNoRewritesCP() {
		testColVariances(TEST_NAME, Sparsity.DENSE, DataType.ROWVECTOR, false, ExecType.CP);
	}

	@Test
	public void testColVariancesDenseRowVectorNoRewritesSpark() {
		testColVariances(TEST_NAME, Sparsity.DENSE, DataType.ROWVECTOR, false, ExecType.SPARK);
	}

	// Dense column vector w/ rewrites
	@Test
	public void testColVariancesDenseColVectorRewritesCP() {
		testColVariances(TEST_NAME, Sparsity.DENSE, DataType.COLUMNVECTOR, true, ExecType.CP);
	}

	@Test
	public void testColVariancesDenseColVectorRewritesSpark() {
		testColVariances(TEST_NAME, Sparsity.DENSE, DataType.COLUMNVECTOR, true, ExecType.SPARK);
	}

	// Dense column vector w/o rewrites
	@Test
	public void testColVariancesDenseColVectorNoRewritesCP() {
		testColVariances(TEST_NAME, Sparsity.DENSE, DataType.COLUMNVECTOR, false, ExecType.CP);
	}

	@Test
	public void testColVariancesDenseColVectorNoRewritesSpark() {
		testColVariances(TEST_NAME, Sparsity.DENSE, DataType.COLUMNVECTOR, false, ExecType.SPARK);
	}

	// Sparse matrix w/ rewrites
	@Test
	public void testColVariancesSparseMatrixRewritesCP() {
		testColVariances(TEST_NAME, Sparsity.SPARSE, DataType.MATRIX, true, ExecType.CP);
	}

	@Test
	public void testColVariancesSparseMatrixRewritesSpark() {
		testColVariances(TEST_NAME, Sparsity.SPARSE, DataType.MATRIX, true, ExecType.SPARK);
	}

	// Sparse matrix w/o rewrites
	@Test
	public void testColVariancesSparseMatrixNoRewritesCP() {
		testColVariances(TEST_NAME, Sparsity.SPARSE, DataType.MATRIX, false, ExecType.CP);
	}

	@Test
	public void testColVariancesSparseMatrixNoRewritesSpark() {
		testColVariances(TEST_NAME, Sparsity.SPARSE, DataType.MATRIX, false, ExecType.SPARK);
	}

	// Sparse row vector w/ rewrites
	@Test
	public void testColVariancesSparseRowVectorRewritesCP() {
		testColVariances(TEST_NAME, Sparsity.SPARSE, DataType.ROWVECTOR, true, ExecType.CP);
	}

	@Test
	public void testColVariancesSparseRowVectorRewritesSpark() {
		testColVariances(TEST_NAME, Sparsity.SPARSE, DataType.ROWVECTOR, true, ExecType.SPARK);
	}

	// Sparse row vector w/o rewrites
	@Test
	public void testColVariancesSparseRowVectorNoRewritesCP() {
		testColVariances(TEST_NAME, Sparsity.SPARSE, DataType.ROWVECTOR, false, ExecType.CP);
	}

	@Test
	public void testColVariancesSparseRowVectorNoRewritesSpark() {
		testColVariances(TEST_NAME, Sparsity.SPARSE, DataType.ROWVECTOR, false, ExecType.SPARK);
	}

	// Sparse column vector w/ rewrites
	@Test
	public void testColVariancesSparseColVectorRewritesCP() {
		testColVariances(TEST_NAME, Sparsity.SPARSE, DataType.COLUMNVECTOR, true, ExecType.CP);
	}

	@Test
	public void testColVariancesSparseColVectorRewritesSpark() {
		testColVariances(TEST_NAME, Sparsity.SPARSE, DataType.COLUMNVECTOR, true, ExecType.SPARK);
	}

	// Sparse column vector w/o rewrites
	@Test
	public void testColVariancesSparseColVectorNoRewritesCP() {
		testColVariances(TEST_NAME, Sparsity.SPARSE, DataType.COLUMNVECTOR, false, ExecType.CP);
	}

	@Test
	public void testColVariancesSparseColVectorNoRewritesSpark() {
		testColVariances(TEST_NAME, Sparsity.SPARSE, DataType.COLUMNVECTOR, false, ExecType.SPARK);
	}

	// Empty matrix w/ rewrites
	@Test
	public void testColVariancesEmptyMatrixRewritesCP() {
		testColVariances(TEST_NAME, Sparsity.EMPTY, DataType.MATRIX, true, ExecType.CP);
	}

	@Test
	public void testColVariancesEmptyMatrixRewritesSpark() {
		testColVariances(TEST_NAME, Sparsity.EMPTY, DataType.MATRIX, true, ExecType.SPARK);
	}

	// Empty matrix w/o rewrites
	@Test
	public void testColVariancesEmptyMatrixNoRewritesCP() {
		testColVariances(TEST_NAME, Sparsity.EMPTY, DataType.MATRIX, false, ExecType.CP);
	}

	@Test
	public void testColVariancesEmptyMatrixNoRewritesSpark() {
		testColVariances(TEST_NAME, Sparsity.EMPTY, DataType.MATRIX, false, ExecType.SPARK);
	}

	// Empty row vector w/ rewrites
	@Test
	public void testColVariancesEmptyRowVectorRewritesCP() {
		testColVariances(TEST_NAME, Sparsity.EMPTY, DataType.ROWVECTOR, true, ExecType.CP);
	}

	@Test
	public void testColVariancesEmptyRowVectorRewritesSpark() {
		testColVariances(TEST_NAME, Sparsity.EMPTY, DataType.ROWVECTOR, true, ExecType.SPARK);
	}

	// Empty row vector w/o rewrites
	@Test
	public void testColVariancesEmptyRowVectorNoRewritesCP() {
		testColVariances(TEST_NAME, Sparsity.EMPTY, DataType.ROWVECTOR, false, ExecType.CP);
	}

	@Test
	public void testColVariancesEmptyRowVectorNoRewritesSpark() {
		testColVariances(TEST_NAME, Sparsity.EMPTY, DataType.ROWVECTOR, false, ExecType.SPARK);
	}

	// Empty column vector w/ rewrites
	@Test
	public void testColVariancesEmptyColVectorRewritesCP() {
		testColVariances(TEST_NAME, Sparsity.EMPTY, DataType.COLUMNVECTOR, true, ExecType.CP);
	}

	@Test
	public void testColVariancesEmptyColVectorRewritesSpark() {
		testColVariances(TEST_NAME, Sparsity.EMPTY, DataType.COLUMNVECTOR, true, ExecType.SPARK);
	}

	// Empty column vector w/o rewrites
	@Test
	public void testColVariancesEmptyColVectorNoRewritesCP() {
		testColVariances(TEST_NAME, Sparsity.EMPTY, DataType.COLUMNVECTOR, false, ExecType.CP);
	}

	@Test
	public void testColVariancesEmptyColVectorNoRewritesSpark() {
		testColVariances(TEST_NAME, Sparsity.EMPTY, DataType.COLUMNVECTOR, false, ExecType.SPARK);
	}

	/**
	 * Test the column variances function, "colVars(X)", on dense/sparse matrices/vectors on the CP/Spark/MR platforms.
	 *
	 * @param testName The name of this test case.
	 * @param sparsity Selection between empty, sparse, and dense data.
	 * @param dataType Selection between a matrix, a row vector, and a column vector.
	 * @param rewrites Whether or not to employ algebraic rewrites.
	 * @param platform Selection between CP/Spark/MR platforms.
	 */
	private void testColVariances(String testName, Sparsity sparsity, DataType dataType, boolean rewrites,
		ExecType platform) {
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
			// - sparsity
			double sparsityVal;
			switch(sparsity) {
				case EMPTY:
					sparsityVal = 0;
					break;
				case SPARSE:
					sparsityVal = sparsitySparse;
					break;
				case DENSE:
				default:
					sparsityVal = sparsityDense;
			}
			// - size
			int r;
			int c;
			switch(dataType) {
				case ROWVECTOR:
					r = 1;
					c = cols;
					break;
				case COLUMNVECTOR:
					r = rows;
					c = 1;
					break;
				case MATRIX:
				default:
					r = rows;
					c = cols;
			}
			// - generation
			double[][] X = getRandomMatrix(r, c, -1, 1, sparsityVal, 7);
			writeInputMatrixWithMTD(INPUT_NAME, X, true);

			// Run DML and R scripts
			runTest(true, false, null, -1);
			runRScript(true);

			// Compare output matrices
			HashMap<CellIndex, Double> dmlfile = readDMLMatrixFromOutputDir(OUTPUT_NAME);
			HashMap<CellIndex, Double> rfile = readRMatrixFromExpectedDir(OUTPUT_NAME);
			TestUtils.compareMatrices(dmlfile, rfile, eps, "Stat-DML", "Stat-R");

			// On CP and Spark modes, check that the following
			// rewrites occurred:
			// - colVars(X), where X is a row vector, should be
			// rewritten to an empty row vector of zeros.
			// - colVars(X), where X is a column vector, should be
			// rewritten to var(X).
			if(rewrites && (platform == ExecType.SPARK || platform == ExecType.CP)) {
				String prefix = (platform == ExecType.SPARK) ? Instruction.SP_INST_PREFIX : "";
				if(dataType == DataType.ROWVECTOR) {
					String opcode = prefix + colVarOp;
					boolean rewriteApplied = !Statistics.getCPHeavyHitterOpCodes().contains(opcode);
					Assert.assertTrue("Rewrite not applied to row vector case.", rewriteApplied);

				}
				else if(dataType == DataType.COLUMNVECTOR) {
					String opcode = prefix + varOp;
					boolean rewriteApplied = Statistics.getCPHeavyHitterOpCodes().contains(opcode);
					Assert.assertTrue("Rewrite not applied to column vector case.", rewriteApplied);
				}
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
