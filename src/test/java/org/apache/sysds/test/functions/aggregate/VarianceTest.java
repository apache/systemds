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

import org.junit.Test;
import org.apache.sysds.api.DMLScript;
import org.apache.sysds.common.Types.ExecMode;
import org.apache.sysds.common.Types.ExecType;
import org.apache.sysds.runtime.matrix.data.MatrixValue.CellIndex;
import org.apache.sysds.test.AutomatedTestBase;
import org.apache.sysds.test.TestConfiguration;
import org.apache.sysds.test.TestUtils;

import java.util.HashMap;

/**
 * Test the variance function, "var(X)".
 */
public class VarianceTest extends AutomatedTestBase {

	private static final String TEST_NAME = "Variance";
	private static final String TEST_DIR = "functions/aggregate/";
	private static final String TEST_CLASS_DIR = TEST_DIR + VarianceTest.class.getSimpleName() + "/";
	private static final String INPUT_NAME = "X";
	private static final String OUTPUT_NAME = "variance";

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

	// Dense matrix
	@Test
	public void testVarianceDenseMatrixCP() {
		testVariance(TEST_NAME, Sparsity.DENSE, DataType.MATRIX, ExecType.CP);
	}

	@Test
	public void testVarianceDenseMatrixSpark() {
		testVariance(TEST_NAME, Sparsity.DENSE, DataType.MATRIX, ExecType.SPARK);
	}

	// Dense row vector
	@Test
	public void testVarianceDenseRowVectorCP() {
		testVariance(TEST_NAME, Sparsity.DENSE, DataType.ROWVECTOR, ExecType.CP);
	}

	@Test
	public void testVarianceDenseRowVectorSpark() {
		testVariance(TEST_NAME, Sparsity.DENSE, DataType.ROWVECTOR, ExecType.SPARK);
	}

	// Dense column vector
	@Test
	public void testVarianceDenseColVectorCP() {
		testVariance(TEST_NAME, Sparsity.DENSE, DataType.COLUMNVECTOR, ExecType.CP);
	}

	@Test
	public void testVarianceDenseColVectorSpark() {
		testVariance(TEST_NAME, Sparsity.DENSE, DataType.COLUMNVECTOR, ExecType.SPARK);
	}

	// Sparse matrix
	@Test
	public void testVarianceSparseMatrixCP() {
		testVariance(TEST_NAME, Sparsity.SPARSE, DataType.MATRIX, ExecType.CP);
	}

	@Test
	public void testVarianceSparseMatrixSpark() {
		testVariance(TEST_NAME, Sparsity.SPARSE, DataType.MATRIX, ExecType.SPARK);
	}

	// Sparse row vector
	@Test
	public void testVarianceSparseRowVectorCP() {
		testVariance(TEST_NAME, Sparsity.SPARSE, DataType.ROWVECTOR, ExecType.CP);
	}

	@Test
	public void testVarianceSparseRowVectorSpark() {
		testVariance(TEST_NAME, Sparsity.SPARSE, DataType.ROWVECTOR, ExecType.SPARK);
	}

	// Sparse column vector
	@Test
	public void testVarianceSparseColVectorCP() {
		testVariance(TEST_NAME, Sparsity.SPARSE, DataType.COLUMNVECTOR, ExecType.CP);
	}

	@Test
	public void testVarianceSparseColVectorSpark() {
		testVariance(TEST_NAME, Sparsity.SPARSE, DataType.COLUMNVECTOR, ExecType.SPARK);
	}

	// Empty matrix
	@Test
	public void testVarianceEmptyMatrixCP() {
		testVariance(TEST_NAME, Sparsity.EMPTY, DataType.MATRIX, ExecType.CP);
	}

	@Test
	public void testVarianceEmptyMatrixSpark() {
		testVariance(TEST_NAME, Sparsity.EMPTY, DataType.MATRIX, ExecType.SPARK);
	}

	// Empty row vector
	@Test
	public void testVarianceEmptyRowVectorCP() {
		testVariance(TEST_NAME, Sparsity.EMPTY, DataType.ROWVECTOR, ExecType.CP);
	}

	@Test
	public void testVarianceEmptyRowVectorSpark() {
		testVariance(TEST_NAME, Sparsity.EMPTY, DataType.ROWVECTOR, ExecType.SPARK);
	}

	// Empty column vector
	@Test
	public void testVarianceEmptyColVectorCP() {
		testVariance(TEST_NAME, Sparsity.EMPTY, DataType.COLUMNVECTOR, ExecType.CP);
	}

	@Test
	public void testVarianceEmptyColVectorSpark() {
		testVariance(TEST_NAME, Sparsity.EMPTY, DataType.COLUMNVECTOR, ExecType.SPARK);
	}

	/**
	 * Test the variance function, "var(X)", on dense/sparse matrices/vectors on the CP/Spark/MR platforms.
	 *
	 * @param testName The name of this test case.
	 * @param sparsity Selection between empty, sparse, and dense data.
	 * @param dataType Selection between a matrix, a row vector, and a column vector.
	 * @param platform Selection between CP/Spark/MR platforms.
	 */
	private void testVariance(String testName, Sparsity sparsity, DataType dataType, ExecType platform) {
		// Configure settings for this test case
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
		}
		finally {
			// Reset settings
			rtplatform = platformOld;
			DMLScript.USE_LOCAL_SPARK_CONFIG = sparkConfigOld;
		}
	}
}
