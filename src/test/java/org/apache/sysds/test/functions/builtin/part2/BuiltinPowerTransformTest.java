/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements. See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership. The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License. You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied. See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

package org.apache.sysds.test.functions.builtin.part2;

import java.util.HashMap;

import org.junit.Assert;
import org.junit.Test;

import org.apache.sysds.common.Types.ExecMode;
import org.apache.sysds.common.Types.ExecType;
import org.apache.sysds.runtime.DMLScriptException;
import org.apache.sysds.runtime.matrix.data.MatrixValue.CellIndex;
import org.apache.sysds.test.AutomatedTestBase;
import org.apache.sysds.test.TestConfiguration;
import org.apache.sysds.test.TestUtils;

public class BuiltinPowerTransformTest extends AutomatedTestBase {
	private static final String TRANSFORM_TEST_NAME = "powerTransform";
	private static final String APPLY_TEST_NAME = "powerTransformApply";
	private static final String TEST_DIR = "functions/builtin/";
	private static final String TEST_CLASS_DIR = TEST_DIR + BuiltinPowerTransformTest.class.getSimpleName() + "/";

	private static final double REFERENCE_EPS = 1e-4;
	private static final double APPLY_EPS = 1e-9;

	@Override
	public void setUp() {
		addTestConfiguration(TRANSFORM_TEST_NAME,
			new TestConfiguration(TEST_CLASS_DIR, TRANSFORM_TEST_NAME, new String[] {"Y", "L", "S"}));
		addTestConfiguration(APPLY_TEST_NAME,
			new TestConfiguration(TEST_CLASS_DIR, APPLY_TEST_NAME, new String[] {"Y"}));
	}

	@Test
	public void testPowerTransformYeoJohnsonDefaultDenseCP() {
		runPowerTransformYeoJohnsonDefaultDenseTest(ExecType.CP);
	}

	@Test
	public void testPowerTransformYeoJohnsonDefaultDenseSpark() {
		runPowerTransformYeoJohnsonDefaultDenseTest(ExecType.SPARK);
	}

	private void runPowerTransformYeoJohnsonDefaultDenseTest(ExecType execType) {
		double[][] input = {{-2, 1, 5}, {-1, 1, 5}, {0, 2, 5}, {1, 3, 5}, {2, 6, 5}, {4, 12, 5}};
		runPowerTransformTest(execType, "default", true, input, false);
	}

	@Test
	public void testPowerTransformBoxCoxUnstandardizedDenseCP() {
		double[][] input = {{1.0, 1.0}, {2.0, 1.1}, {3.0, 1.2}, {4.0, 1.3}, {5.0, 1.4}, {6.0, 1.5}, {7.0, 2.0},
			{8.0, 8.0}};
		runPowerTransformTest("box-cox", false, input, false);
	}

	@Test
	public void testPowerTransformYeoJohnsonLambdaAboveInitialInterval() {
		double[][] input = {{0.00}, {0.97}, {0.98}, {0.99}, {1.00}};
		runPowerTransformTest("yeo-johnson", false, input, false);
		assertLambdaOutsideInitialInterval(true);
	}

	@Test
	public void testPowerTransformYeoJohnsonPreservesNaNCP() {
		double[][] input = {{-2, 1}, {-1, Double.NaN}, {Double.NaN, 2}, {1, 4}, {2, 8}};
		runPowerTransformTest("yeo-johnson", true, input, false, true, false);
		assertNaNPositions(input);
	}

	@Test
	public void testPowerTransformBoxCoxLambdaBelowInitialInterval() {
		double[][] input = {{1.00}, {1.01}, {1.02}, {1.03}, {10.0}};
		runPowerTransformTest("box-cox", false, input, false);
		assertLambdaOutsideInitialInterval(false);
	}

	@Test
	public void testPowerTransformBoxCoxFallsBackToFiniteLambdaCP() {
		double[][] input = {{1e-100}, {1e-50}, {1.0}, {1e50}, {1e100}};
		runPowerTransformTest("box-cox", false, input, false, false);
		double lambda = readDMLMatrixFromOutputDir("L").get(new CellIndex(1, 1));
		Assert.assertTrue(Double.isFinite(lambda));
		assertNaNPositions(input);
	}

	@Test
	public void testPowerTransformBoxCoxRejectsNonPositiveInput() {
		double[][] input = {{0, 1}, {1, 2}};
		runPowerTransformTest("box-cox", false, input, true);
	}

	@Test
	public void testPowerTransformApplyYeoJohnsonDenseCP() {
		runPowerTransformApplyYeoJohnsonDenseTest(ExecType.CP);
	}

	@Test
	public void testPowerTransformApplyYeoJohnsonDenseSpark() {
		runPowerTransformApplyYeoJohnsonDenseTest(ExecType.SPARK);
	}

	private void runPowerTransformApplyYeoJohnsonDenseTest(ExecType execType) {
		double[][] input = {{-2, -2, -2}, {-1, -1, -1}, {0, 0, 0}, {1, 1, 1}, {2, 2, 2}};
		double[][] expected = {{-3, -1.5, -1.03944491546724}, {-1.33333333333333, -1, -0.877258872223978},
			{-0.333333333333333, -0.5, -0.6}, {0.128764787039964, 0, 0}, {0.399074859112073, 0.5, 1}};
		runPowerTransformApplyTest(execType, "yeo-johnson", true, input, expected, false);
	}

	@Test
	public void testPowerTransformApplyBoxCoxDenseCP() {
		double[][] input = {{0.5, 0.5, 0.5}, {1.0, 1.0, 1.0}, {2.0, 2.0, 2.0}, {4.0, 4.0, 4.0}, {8.0, 8.0, 8.0}};
		double[][] expected = {{-0.693147180559945, -0.5, -0.375}, {0, 0, 0}, {0.693147180559945, 1, 1.5},
			{1.38629436111989, 3, 7.5}, {2.07944154167984, 7, 31.5}};
		runPowerTransformApplyTest(ExecType.CP, "box-cox", false, input, expected, false);
	}

	@Test
	public void testPowerTransformApplyBoxCoxPreservesNaNCP() {
		double[][] input = {{0.5, Double.NaN, 0.5}, {Double.NaN, 1.0, 1.0}, {2.0, 2.0, Double.NaN}};
		double[][] expected = {{-0.693147180559945, Double.NaN, -0.375}, {Double.NaN, 0, 0},
			{0.693147180559945, 1, Double.NaN}};
		runPowerTransformApplyTest(ExecType.CP, "box-cox", false, input, expected, false);
	}

	@Test
	public void testPowerTransformApplyBoxCoxRejectsNonPositiveInput() {
		double[][] input = {{0, 1, 2}, {1, 2, 3}};
		runPowerTransformApplyTest(ExecType.CP, "box-cox", false, input, null, true);
	}

	private void runPowerTransformTest(String method, boolean standardize, double[][] input, boolean shouldFail) {
		runPowerTransformTest(ExecType.CP, method, standardize, input, shouldFail, true, true);
	}

	private void runPowerTransformTest(ExecType execType, String method, boolean standardize, double[][] input,
		boolean shouldFail) {
		runPowerTransformTest(execType, method, standardize, input, shouldFail, true, true);
	}

	private void runPowerTransformTest(String method, boolean standardize, double[][] input, boolean shouldFail,
		boolean compareReference) {
		runPowerTransformTest(ExecType.CP, method, standardize, input, shouldFail, compareReference, true);
	}

	private void runPowerTransformTest(String method, boolean standardize, double[][] input, boolean shouldFail,
		boolean compareReference, boolean compareTransformed) {
		runPowerTransformTest(ExecType.CP, method, standardize, input, shouldFail, compareReference,
			compareTransformed);
	}

	private void runPowerTransformTest(ExecType execType, String method, boolean standardize, double[][] input,
		boolean shouldFail, boolean compareReference, boolean compareTransformed) {
		ExecMode oldExecMode = setExecMode(execType);

		try {
			loadTestConfiguration(getTestConfiguration(TRANSFORM_TEST_NAME));

			String home = SCRIPT_DIR + TEST_DIR;
			fullDMLScriptName = home + TRANSFORM_TEST_NAME + ".dml";
			fullRScriptName = home + TRANSFORM_TEST_NAME + ".R";
			programArgs = new String[] {"-args", input("X"), output("Y"), output("L"), output("S"), method,
				Boolean.toString(standardize)};

			if(compareReference) {
				String referenceMethod = method.equals("default") ? "yeo-johnson" : method;
				rCmd = getRCmd(inputDir(), expectedDir(), referenceMethod, Boolean.toString(standardize));
			}

			writeInputMatrixWithMTD("X", input, true);
			runTest(true, shouldFail, shouldFail ? DMLScriptException.class : null, -1);
			if(shouldFail)
				return;

			if(compareReference) {
				runRScript(true);
				if(compareTransformed)
					compareOutput("Y", REFERENCE_EPS);
				compareOutput("L", REFERENCE_EPS);
				compareOutput("S", REFERENCE_EPS);
			}
		}
		catch(Exception exception) {
			throw new RuntimeException(exception);
		}
		finally {
			resetExecMode(oldExecMode);
		}
	}

	private void runPowerTransformApplyTest(ExecType execType, String method, boolean standardize, double[][] input,
		double[][] expected, boolean shouldFail) {
		ExecMode oldExecMode = setExecMode(execType);

		try {
			loadTestConfiguration(getTestConfiguration(APPLY_TEST_NAME));

			String home = SCRIPT_DIR + TEST_DIR;
			fullDMLScriptName = home + APPLY_TEST_NAME + ".dml";
			programArgs = new String[] {"-args", input("X"), input("L"), input("M"), input("S"), output("Y"), method,
				Boolean.toString(standardize)};

			double[][] L = {{0, 1, 2}};
			double[][] means = {{0.5, 1.0, 1.5}};
			double[][] scales = {{1.5, 2.0, 2.5}};

			writeInputMatrixWithMTD("X", input, true);
			writeInputMatrixWithMTD("L", L, true);
			writeInputMatrixWithMTD("M", means, true);
			writeInputMatrixWithMTD("S", scales, true);
			if(!shouldFail)
				writeExpectedMatrix("Y", expected);

			runTest(true, shouldFail, shouldFail ? DMLScriptException.class : null, -1);
			if(shouldFail)
				return;

			compareResults(APPLY_EPS);
		}
		catch(Exception exception) {
			throw new RuntimeException(exception);
		}
		finally {
			resetExecMode(oldExecMode);
		}
	}

	private void compareOutput(String name, double tolerance) {
		HashMap<CellIndex, Double> dmlResult = readDMLMatrixFromOutputDir(name);
		HashMap<CellIndex, Double> rResult = readRMatrixFromExpectedDir(name);
		TestUtils.compareMatrices(dmlResult, rResult, tolerance, "DML", "R");
	}

	private void assertLambdaOutsideInitialInterval(boolean above) {
		double lambda = readDMLMatrixFromOutputDir("L").get(new CellIndex(1, 1));
		Assert.assertTrue("Expected lambda outside the initial interval, but was " + lambda,
			above ? lambda > 2.0 : lambda < -2.0);
	}

	private void assertNaNPositions(double[][] input) {
		HashMap<CellIndex, Double> output = readDMLMatrixFromOutputDir("Y");
		for(int i = 0; i < input.length; i++) {
			for(int j = 0; j < input[i].length; j++) {
				double value = output.getOrDefault(new CellIndex(i + 1, j + 1), 0.0);
				Assert.assertEquals(Double.isNaN(input[i][j]), Double.isNaN(value));
			}
		}
	}
}
