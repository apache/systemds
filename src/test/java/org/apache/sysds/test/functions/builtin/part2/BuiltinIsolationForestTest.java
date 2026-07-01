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
 * KIND, either express or implied.  See the NOTICE file for
 * the specific language governing permissions and limitations
 * under the License.
 */

package org.apache.sysds.test.functions.builtin.part2;

import org.apache.sysds.common.Types.ExecMode;
import org.apache.sysds.runtime.matrix.data.MatrixValue.CellIndex;
import org.apache.sysds.test.AutomatedTestBase;
import org.apache.sysds.test.TestConfiguration;
import org.apache.sysds.test.TestUtils;
import org.junit.Assert;
import org.junit.Test;

import java.util.HashMap;

public class BuiltinIsolationForestTest extends AutomatedTestBase {
	private final static String TEST_NAME = "outlierByIsolationForestTest";
	private final static String TEST_DIR = "functions/builtin/";
	private static final String TEST_CLASS_DIR = TEST_DIR + BuiltinIsolationForestTest.class.getSimpleName() + "/";

	private final static double eps = 1e-10;

	private final static int rows = 100;
	private final static int cols = 3;
	private final static int n_trees = 10;
	private final static int subsampling_size = 20;
	private final static int seed = 42;

	private final static int TEST_BASIC_MODEL = 1;
	private final static int TEST_ANOMALY_RANKING = 2;
	private final static int TEST_SUBSAMPLING_CLAMP = 3;
	private final static int TEST_SINGLE_ROW_APPLY = 4;
	private final static int TEST_SINGLE_TREE_MODEL = 5;
	private final static int TEST_CONSTANT_DATA = 6;
	private final static int TEST_SEED_REPRODUCIBILITY = 7;

	@Override
	public void setUp() {
		TestUtils.clearAssertionInformation();
		addTestConfiguration(TEST_NAME, new TestConfiguration(TEST_CLASS_DIR, TEST_NAME,
				new String[]{"scores", "model", "subsampling_size"}));
	}

	@Test
	public void testBasicModelSingleNode() {
		runIsolationForestTest(TEST_BASIC_MODEL, ExecMode.SINGLE_NODE,
				normalCluster(rows, cols), n_trees, subsampling_size, seed);
	}

	@Test
	public void testBasicModelHybrid() {
		runIsolationForestTest(TEST_BASIC_MODEL, ExecMode.HYBRID,
				normalCluster(rows, cols), n_trees, subsampling_size, seed);
	}

	@Test
	public void testAnomalyRankingSingleNode() {
		runIsolationForestTest(TEST_ANOMALY_RANKING, ExecMode.SINGLE_NODE,
				normalCluster(30, cols), 50, 10, seed);
	}

	@Test
	public void testAnomalyRankingHybrid() {
		runIsolationForestTest(TEST_ANOMALY_RANKING, ExecMode.HYBRID,
				normalCluster(30, cols), 50, 10, seed);
	}

	@Test
	public void testSubsamplingClampSingleNode() {
		runIsolationForestTest(TEST_SUBSAMPLING_CLAMP, ExecMode.SINGLE_NODE,
				normalCluster(5, cols), 3, 256, seed);
	}

	@Test
	public void testSubsamplingClampHybrid() {
		runIsolationForestTest(TEST_SUBSAMPLING_CLAMP, ExecMode.HYBRID,
				normalCluster(5, cols), 3, 256, seed);
	}

	@Test
	public void testSingleRowApplySingleNode() {
		runIsolationForestTest(TEST_SINGLE_ROW_APPLY, ExecMode.SINGLE_NODE,
				normalCluster(20, cols), 5, 10, seed);
	}

	@Test
	public void testSingleTreeModelSingleNode() {
		runIsolationForestTest(TEST_SINGLE_TREE_MODEL, ExecMode.SINGLE_NODE,
				normalCluster(20, cols), 1, 10, seed);
	}

	@Test
	public void testConstantDataSingleNode() {
		runIsolationForestTest(TEST_CONSTANT_DATA, ExecMode.SINGLE_NODE,
				constantMatrix(5, cols, 7.0), 1, 5, seed);
	}

	@Test
	public void testSeedReproducibilitySingleNode() {
		runIsolationForestTest(TEST_SEED_REPRODUCIBILITY, ExecMode.SINGLE_NODE,
				normalCluster(30, cols), 5, 10, seed);
	}

	private void runIsolationForestTest(int testCase, ExecMode mode, double[][] A,
										int nTrees, int subsamplingSize, int seed) {
		ExecMode platformOld = setExecMode(mode);

		try {
			loadTestConfiguration(getTestConfiguration(TEST_NAME));
			String HOME = SCRIPT_DIR + TEST_DIR;

			fullDMLScriptName = HOME + TEST_NAME + ".dml";
			programArgs = new String[]{"-nvargs",
					"X=" + input("A"),
					"test_case=" + testCase,
					"n_trees=" + nTrees,
					"subsampling_size=" + subsamplingSize,
					"seed=" + seed,
					"output=" + output("scores"),
					"model_output=" + output("model"),
					"subsampling_size_output=" + output("subsampling_size")};

			writeInputMatrixWithMTD("A", A, true);

			runTest(true, false, null, -1);

			HashMap<CellIndex, Double> scores = readDMLMatrixFromOutputDir("scores");
			Assert.assertNotNull("Scores should not be null", scores);
			Assert.assertFalse("Scores should have entries", scores.isEmpty());

			HashMap<CellIndex, Double> model = readDMLMatrixFromOutputDir("model");
			Assert.assertNotNull("Model should not be null", model);
			Assert.assertFalse("Model should have entries", model.isEmpty());

			HashMap<CellIndex, Double> actualSubsamplingSize =
					readDMLScalarFromOutputDir("subsampling_size");
			Assert.assertNotNull("Subsampling size should be written", actualSubsamplingSize);

			double expectedSubsamplingSize = Math.min(subsamplingSize, A.length);
			Assert.assertEquals("Stored subsampling size should match the actual training subsample size",
					expectedSubsamplingSize,
					actualSubsamplingSize.get(new CellIndex(1, 1)),
					eps);

			int maxRow = 0;
			for (CellIndex idx : model.keySet())
				maxRow = Math.max(maxRow, idx.row);

			Assert.assertEquals("Model should have n_trees rows", nTrees, maxRow);
		}
		finally {
			rtplatform = platformOld;
		}
	}

	private double[][] normalCluster(int rows, int cols) {
		double[][] A = new double[rows][cols];

		for (int i = 0; i < rows; i++) {
			for (int j = 0; j < cols; j++) {
				A[i][j] = (i + 1) / 100.0 + j / 1000.0;
			}
		}

		return A;
	}

	private double[][] constantMatrix(int rows, int cols, double value) {
		double[][] A = new double[rows][cols];

		for (int i = 0; i < rows; i++) {
			for (int j = 0; j < cols; j++) {
				A[i][j] = value;
			}
		}

		return A;
	}
}