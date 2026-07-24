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

import java.util.Arrays;
import java.util.Collection;
import java.util.HashMap;
import java.util.Locale;

import org.apache.sysds.common.Types.ExecMode;
import org.apache.sysds.runtime.matrix.data.MatrixValue.CellIndex;
import org.apache.sysds.runtime.meta.MatrixCharacteristics;
import org.apache.sysds.test.AutomatedTestBase;
import org.apache.sysds.test.TestConfiguration;
import org.apache.sysds.test.TestUtils;
import org.junit.Assert;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.Parameterized;

@RunWith(value = Parameterized.class)

@net.jcip.annotations.NotThreadSafe
public class BuiltinIsolationForestTest extends AutomatedTestBase
{
	private static final String TEST_NAME = "outlierByIsolationForestTest";
	private static final String TEST_DIR = "functions/builtin/";
	private static final String TEST_CLASS_DIR = TEST_DIR + BuiltinIsolationForestTest.class.getSimpleName() + "/";

	private static final double SCORE_EPS = 1e-10;
	private static final double R_SCORE_EPS = 1e-9;
	private static final int DATA_SEED = 314159265;
	private static final int FOREST_SEED = 42;
	private static final int REFERENCE_SUBSAMPLING_SIZE = 4;

	private enum TestCase {
		BASIC,
		ANOMALY_RANKING,
		SUBSAMPLING_CLAMP,
		SINGLE_ROW_APPLY,
		SINGLE_TREE,
		CONSTANT_DATA,
		MINIMUM_DATASET,
		SINGLE_FEATURE,
		HIGH_DIMENSIONAL,
		PARTIALLY_CONSTANT
	}

	private final TestCase testCase;
	private final int numRows;
	private final int numCols;
	private final int numTrees;
	private final int subsamplingSize;

	public BuiltinIsolationForestTest(TestCase testCase, int numRows, int numCols,
		int numTrees, int subsamplingSize)
	{
		this.testCase = testCase;
		this.numRows = numRows;
		this.numCols = numCols;
		this.numTrees = numTrees;
		this.subsamplingSize = subsamplingSize;
	}

	@Override
	public void setUp() {
		TestUtils.clearAssertionInformation();
		addTestConfiguration(TEST_NAME, new TestConfiguration(TEST_CLASS_DIR, TEST_NAME,
			new String[] {"scores", "model", "subsampling_size", "same_seed_model",
				"different_seed_model", "reference_scores", "dml_apply_runtime", "model_for_r"}));
	}

	@Test
	public void testIsolationForestSingleNode() {
		runIsolationForestTest(ExecMode.SINGLE_NODE);
	}

	@Test
	public void testIsolationForestHybrid() {
		runIsolationForestTest(ExecMode.HYBRID);
	}

	private void runIsolationForestTest(ExecMode mode) {
		ExecMode platformOld = setExecMode(mode);

		try {
			loadTestConfiguration(getTestConfiguration(TEST_NAME));

			double[][] X = createTrainingData();
			double[][] X_apply = createApplyData(X);
			boolean runExtendedChecks = testCase == TestCase.BASIC && mode == ExecMode.SINGLE_NODE;

			writeInputMatrixWithMTD("X", X, true);
			writeInputMatrixWithMTD("X_apply", X_apply, true);
			writeInputMatrixWithMTD("ReferenceModel", referenceModel(), true);
			writeInputMatrixWithMTD("ReferenceX", referenceSamples(), true);

			String home = SCRIPT_DIR + TEST_DIR;
			fullDMLScriptName = home + TEST_NAME + ".dml";
			programArgs = new String[] {"-nvargs",
				"X=" + input("X"),
				"X_apply=" + input("X_apply"),
				"reference_model=" + input("ReferenceModel"),
				"reference_X=" + input("ReferenceX"),
				"n_trees=" + numTrees,
				"subsampling_size=" + subsamplingSize,
				"seed=" + FOREST_SEED,
				"check_seed=" + (runExtendedChecks ? 1 : 0),
				"scores=" + output("scores"),
				"model=" + output("model"),
				"subsampling_size_out=" + output("subsampling_size"),
				"same_seed_model=" + output("same_seed_model"),
				"different_seed_model=" + output("different_seed_model"),
				"reference_scores=" + output("reference_scores"),
				"dml_apply_runtime=" + output("dml_apply_runtime"),
				"model_for_r=" + output("model_for_r")};

			runTest(true, EXCEPTION_NOT_EXPECTED, null, -1);

			int actualSubsamplingSize = Math.min(subsamplingSize, numRows);
			HashMap<CellIndex, Double> scores = readDMLMatrixFromOutputDir("scores");
			HashMap<CellIndex, Double> model = readDMLMatrixFromOutputDir("model");

			assertOutputDimensions(X_apply.length, actualSubsamplingSize);
			assertScores(scores, X_apply.length);
			assertModelStructure(model, X, actualSubsamplingSize);
			assertScenarioSpecificResults(scores, X_apply.length);
			assertReferenceScores();

			if (runExtendedChecks) {
				assertSeedBehavior(model);
				assertRReference(scores, actualSubsamplingSize, X_apply.length, home);
			}
		}
		finally {
			resetExecMode(platformOld);
		}
	}

	private void assertOutputDimensions(int numApplyRows, int actualSubsamplingSize) {
		MatrixCharacteristics scoreMeta = readDMLMetaDataFile("scores");
		Assert.assertEquals("One anomaly score is expected per input row", numApplyRows, scoreMeta.getRows());
		Assert.assertEquals("Anomaly scores must form a column vector", 1, scoreMeta.getCols());

		int heightLimit = (int) Math.ceil(Math.log(actualSubsamplingSize) / Math.log(2));
		long expectedModelCols = 2L * ((1L << (heightLimit + 1)) - 1);
		MatrixCharacteristics modelMeta = readDMLMetaDataFile("model");
		Assert.assertEquals("The model must contain exactly numTrees trees", numTrees, modelMeta.getRows());
		Assert.assertEquals("Every tree must be padded to the configured height limit",
			expectedModelCols, modelMeta.getCols());

		HashMap<CellIndex, Double> subsamplingSizeOut = readDMLScalarFromOutputDir("subsampling_size");
		Double actualValue = subsamplingSizeOut.get(new CellIndex(1, 1));
		Assert.assertNotNull("The effective subsampling size must be stored in the model", actualValue);
		Assert.assertEquals("The stored value must reflect clamping to the number of training rows",
			actualSubsamplingSize, actualValue.intValue());
	}

	private void assertScores(HashMap<CellIndex, Double> scores, int expectedRows) {
		for (int row = 1; row <= expectedRows; row++) {
			double score = valueAt(scores, row, 1);
			Assert.assertTrue("Anomaly scores must be finite", Double.isFinite(score));
			Assert.assertTrue("Anomaly scores must be in the interval (0, 1]", score > 0 && score <= 1);
		}
	}

	private void assertModelStructure(HashMap<CellIndex, Double> model, double[][] X,
		int actualSubsamplingSize)
	{
		MatrixCharacteristics modelMeta = readDMLMetaDataFile("model");
		int numNodes = (int) modelMeta.getCols() / 2;
		int heightLimit = (int) Math.ceil(Math.log(actualSubsamplingSize) / Math.log(2));
		double[][] featureRanges = featureRanges(X);

		for (int tree = 1; tree <= numTrees; tree++) {
			int leafSizeSum = 0;

			for (int nodeId = 1; nodeId <= numNodes; nodeId++) {
				int nodeColumn = 2 * nodeId - 1;
				double rawNodeType = valueAt(model, tree, nodeColumn);
				int nodeType = (int) Math.rint(rawNodeType);
				double nodeValue = valueAt(model, tree, nodeColumn + 1);

				Assert.assertEquals("Node types must be integer feature indices or the 0/-1 sentinels",
					nodeType, rawNodeType, SCORE_EPS);

				if (nodeId > 1) {
					int parentId = nodeId / 2;
					int parentType = (int) Math.rint(valueAt(model, tree, 2 * parentId - 1));
					if (parentType > 0)
						Assert.assertNotEquals("Both children of an internal node must be materialized", -1, nodeType);
					else
						Assert.assertEquals("Children below leaves and placeholders must remain placeholders", -1, nodeType);
				}

				if (nodeType == -1) {
					Assert.assertEquals("Both entries of a placeholder node must be -1", -1, nodeValue, SCORE_EPS);
				}
				else if (nodeType == 0) {
					Assert.assertTrue("Every external node must contain at least one sample", nodeValue >= 1);
					Assert.assertEquals("External-node sizes must be integral", Math.rint(nodeValue), nodeValue, SCORE_EPS);
					leafSizeSum += (int) nodeValue;
				}
				else if (nodeType > 0) {
					Assert.assertTrue("Split feature index exceeds the training width", nodeType <= numCols);
					Assert.assertTrue("Split values must be finite", Double.isFinite(nodeValue));
					Assert.assertTrue("Split value is below the observed feature range",
						nodeValue >= featureRanges[nodeType - 1][0]);
					Assert.assertTrue("Split value is above the observed feature range",
						nodeValue <= featureRanges[nodeType - 1][1]);

					int nodeDepth = 31 - Integer.numberOfLeadingZeros(nodeId);
					Assert.assertTrue("Nodes at the height limit must be external", nodeDepth < heightLimit);

					if (testCase == TestCase.PARTIALLY_CONSTANT)
						Assert.assertEquals("Constant columns must never be selected for a split", numCols, nodeType);
				}
				else {
					Assert.fail("Invalid node type " + nodeType + " in tree " + tree);
				}
			}

			Assert.assertEquals("External nodes must partition the complete training subsample",
				actualSubsamplingSize, leafSizeSum);
		}
	}

	private void assertScenarioSpecificResults(HashMap<CellIndex, Double> scores, int numApplyRows) {
		if (testCase == TestCase.CONSTANT_DATA || testCase == TestCase.MINIMUM_DATASET) {
			for (int row = 1; row <= numApplyRows; row++)
				Assert.assertEquals("This scenario has the exact theoretical score 0.5",
					0.5, valueAt(scores, row, 1), SCORE_EPS);
		}
		else if (testCase == TestCase.ANOMALY_RANKING) {
			double normalMean = (valueAt(scores, 1, 1) + valueAt(scores, 2, 1)) / 2;
			double outlierMean = (valueAt(scores, 3, 1) + valueAt(scores, 4, 1)) / 2;
			Assert.assertTrue("Points far outside the training range must rank above in-distribution points",
				outlierMean > normalMean);
		}
	}

	private void assertReferenceScores() {
		HashMap<CellIndex, Double> scores = readDMLMatrixFromOutputDir("reference_scores");
		double normalization = averagePathLength(REFERENCE_SUBSAMPLING_SIZE);

		Assert.assertEquals(isolationScore(1, normalization), valueAt(scores, 1, 1), SCORE_EPS);
		Assert.assertEquals(isolationScore(2, normalization), valueAt(scores, 2, 1), SCORE_EPS);
		Assert.assertEquals(isolationScore(3, normalization), valueAt(scores, 3, 1), SCORE_EPS);
	}

	private void assertSeedBehavior(HashMap<CellIndex, Double> model) {
		HashMap<CellIndex, Double> sameSeedModel = readDMLMatrixFromOutputDir("same_seed_model");
		HashMap<CellIndex, Double> differentSeedModel = readDMLMatrixFromOutputDir("different_seed_model");

		Assert.assertEquals("The same input and seed must reproduce the exact forest", model, sameSeedModel);
		Assert.assertNotEquals("Changing the seed must change at least one tree", model, differentSeedModel);
	}

	// R scores the serialized SystemDS forest instead of training a second random
	// forest whose RNG stream would not be element-wise comparable.
	private void assertRReference(HashMap<CellIndex, Double> dmlScores, int actualSubsamplingSize,
		int numApplyRows, String home)
	{
		fullRScriptName = home + TEST_NAME + ".R";
		rCmd = getRCmd(
			output("model_for_r"),
			input("X_apply.mtx"),
			Integer.toString(actualSubsamplingSize),
			expected("scores_R"),
			expected("runtime_R"));

		runRScript(true);

		HashMap<CellIndex, Double> rScores = readRMatrixFromExpectedDir("scores_R");
		for (int row = 1; row <= numApplyRows; row++)
			Assert.assertEquals("R and SystemDS must agree on the score for row " + row,
				valueAt(dmlScores, row, 1), valueAt(rScores, row, 1), R_SCORE_EPS);

		double dmlRuntime = valueAt(readDMLScalarFromOutputDir("dml_apply_runtime"), 1, 1);
		double rRuntime = valueAt(readRMatrixFromExpectedDir("runtime_R"), 1, 1);
		Assert.assertTrue("SystemDS Apply runtime must be finite and non-negative",
			Double.isFinite(dmlRuntime) && dmlRuntime >= 0);
		Assert.assertTrue("R Apply runtime must be finite and non-negative",
			Double.isFinite(rRuntime) && rRuntime >= 0);

		System.out.printf(Locale.ROOT, "Isolation Forest Apply runtime: SystemDS %.6f s, R %.6f s%n",
			dmlRuntime, rRuntime);
	}

	private double[][] createTrainingData() {
		switch (testCase) {
			case CONSTANT_DATA:
				return constantMatrix(numRows, numCols, 7);
			case MINIMUM_DATASET:
				return new double[][] {{0.01, 0.02}, {0.03, 0.04}};
			case PARTIALLY_CONSTANT:
				return partiallyConstantMatrix(numRows, numCols);
			default:
				return getRandomMatrix(numRows, numCols, -1, 1, 1,
					DATA_SEED + testCase.ordinal());
		}
	}

	private double[][] createApplyData(double[][] X) {
		if (testCase == TestCase.ANOMALY_RANKING) {
			double[][] result = new double[4][numCols];
			System.arraycopy(X[numRows / 3], 0, result[0], 0, numCols);
			System.arraycopy(X[2 * numRows / 3], 0, result[1], 0, numCols);
			Arrays.fill(result[2], -10);
			Arrays.fill(result[3], 10);
			return result;
		}
		else if (testCase == TestCase.SINGLE_ROW_APPLY)
			return copyFirstRows(X, 1);
		else if (testCase == TestCase.HIGH_DIMENSIONAL)
			return copyFirstRows(X, 10);

		return X;
	}

	private static double[][] constantMatrix(int rows, int cols, double value) {
		double[][] result = new double[rows][cols];
		for (double[] row : result)
			Arrays.fill(row, value);
		return result;
	}

	private static double[][] partiallyConstantMatrix(int rows, int cols) {
		double[][] result = constantMatrix(rows, cols, 7);
		for (int row = 0; row < rows; row++)
			result[row][cols - 1] = (double) row / (rows - 1);
		return result;
	}

	private static double[][] copyFirstRows(double[][] X, int rows) {
		double[][] result = new double[rows][X[0].length];
		for (int row = 0; row < rows; row++)
			System.arraycopy(X[row], 0, result[row], 0, X[row].length);
		return result;
	}

	private static double[][] featureRanges(double[][] X) {
		double[][] result = new double[X[0].length][2];
		for (double[] range : result) {
			range[0] = Double.POSITIVE_INFINITY;
			range[1] = Double.NEGATIVE_INFINITY;
		}

		for (double[] row : X) {
			for (int col = 0; col < row.length; col++) {
				result[col][0] = Math.min(result[col][0], row[col]);
				result[col][1] = Math.max(result[col][1], row[col]);
			}
		}
		return result;
	}

	private static double[][] referenceModel() {
		return new double[][] {{
			1, 0.5,
			0, 1, 1, 1.5,
			-1, -1, -1, -1, 0, 1, 0, 2
		}};
	}

	private static double[][] referenceSamples() {
		return new double[][] {{0.25}, {0.5}, {2}};
	}

	private static double averagePathLength(int n) {
		double harmonic = 0;
		for (int i = 1; i < n; i++)
			harmonic += 1.0 / i;
		return 2 * harmonic - 2.0 * (n - 1) / n;
	}

	private static double isolationScore(int pathLength, double normalization) {
		return Math.pow(2, -pathLength / normalization);
	}

	private static double valueAt(HashMap<CellIndex, Double> matrix, int row, int col) {
		return matrix.getOrDefault(new CellIndex(row, col), 0.0);
	}

	@Parameterized.Parameters(name = "{0}: rows={1}, cols={2}, trees={3}, subsample={4}")
	public static Collection<Object[]> data() {
		// SCHEMA: TEST_CASE, #ROWS, #COLS, #TREES, REQUESTED_SUBSAMPLE_SIZE
		return Arrays.asList(new Object[][] {
			{TestCase.BASIC,                100,  3, 10,  20},
			{TestCase.ANOMALY_RANKING,      100,  3, 50,  64},
			{TestCase.SUBSAMPLING_CLAMP,      5,  3,  3, 256},
			{TestCase.SINGLE_ROW_APPLY,       20,  3,  5,  10},
			{TestCase.SINGLE_TREE,            20,  3,  1,  10},
			{TestCase.CONSTANT_DATA,           5,  3,  1,   5},
			{TestCase.MINIMUM_DATASET,         2,  2,  5,   2},
			{TestCase.SINGLE_FEATURE,        100,  1, 20,  50},
			{TestCase.HIGH_DIMENSIONAL,      200, 30, 20, 100},
			{TestCase.PARTIALLY_CONSTANT,     80,  5, 20,  40}
		});
	}
}
