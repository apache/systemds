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

	@Override
	public void setUp() {
		TestUtils.clearAssertionInformation();
		addTestConfiguration(TEST_NAME, new TestConfiguration(TEST_CLASS_DIR, TEST_NAME,
				new String[]{"scores", "model", "subsampling_size"}));
	}

	@Test
	public void testIsolationForestSingleNode() {
		runIsolationForestTest(false, ExecMode.SINGLE_NODE);
	}

	@Test
	public void testIsolationForestHybrid() {
		runIsolationForestTest(false, ExecMode.HYBRID);
	}

	@Test
	public void testIsolationForestWithOutliersSingleNode() {
		runIsolationForestTest(true, ExecMode.SINGLE_NODE);
	}

	@Test
	public void testIsolationForestWithOutliersHybrid() {
		runIsolationForestTest(true, ExecMode.HYBRID);
	}

	private void runIsolationForestTest(boolean withOutliers, ExecMode mode) {
		ExecMode platformOld = setExecMode(mode);

		try {
			loadTestConfiguration(getTestConfiguration(TEST_NAME));
			String HOME = SCRIPT_DIR + TEST_DIR;

			fullDMLScriptName = HOME + TEST_NAME + ".dml";
			programArgs = new String[]{"-nvargs",
					"X=" + input("A"),
					"n_trees=" + n_trees,
					"subsampling_size=" + subsampling_size,
					"seed=" + seed,
					"output=" + output("scores"),
					"model_output=" + output("model"),
					"subsampling_size_output=" + output("subsampling_size")};


			// Generate data
			double[][] A;
			if (withOutliers) {
				// Generate data with clear outliers
				// Most data is around 0, outliers are far away
				A = new double[rows][cols];
				for (int i = 0; i < rows - 5; i++) {
					for (int j = 0; j < cols; j++) {
						// Normal data: mean=0, range=[-2, 2]
						A[i][j] = (Math.random() - 0.5) * 4;
					}
				}
				// Add outliers: far from normal data
				for (int i = rows - 5; i < rows; i++) {
					for (int j = 0; j < cols; j++) {
						// Outliers: mean=10, range=[8, 12]
						A[i][j] = 8 + Math.random() * 4;
					}
				}
			} else {
				// Generate normal data (no outliers)
				A = getRandomMatrix(rows, cols, -5, 5, 0.7, seed);
			}

			writeInputMatrixWithMTD("A", A, true);

			runTest(true, false, null, -1);

			// Verify model was created
			HashMap<CellIndex, Double> model = readDMLMatrixFromOutputDir("model");
			Assert.assertNotNull("Model should not be null", model);
			Assert.assertFalse("Model should have entries", model.isEmpty());

			// Verify subsampling size was stored correctly
			HashMap<CellIndex, Double> subsamplingSize = readDMLScalarFromOutputDir("subsampling_size");
			Assert.assertEquals("Subsampling size should match",
					subsampling_size,
					subsamplingSize.get(new CellIndex(1, 1)),
					eps);

			// Verify model has n_trees rows
			int maxRow = 0;
			for (CellIndex idx : model.keySet()) {
				maxRow = Math.max(maxRow, idx.row);
			}
			Assert.assertEquals("Model should have n_trees rows", n_trees, maxRow);
		} finally {
			rtplatform = platformOld;
		}
	}
}