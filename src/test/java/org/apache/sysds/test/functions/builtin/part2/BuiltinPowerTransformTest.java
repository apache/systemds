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

	private static final double TRANSFORM_EPS = 1e-6;
	private static final double OUTSIDE_INTERVAL_Y_EPS = 2e-6;
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
		double[][] input = {{-2, 1, 5}, {-1, 1, 5}, {0, 2, 5}, {1, 3, 5}, {2, 6, 5}, {4, 12, 5}};
		runPowerTransformTest("default", true, input, false);
	}

	@Test
	public void testPowerTransformBoxCoxUnstandardizedDenseCP() {
		double[][] input = {{0.5, 1}, {1.0, 2}, {2.0, 3}, {4.0, 5}, {8.0, 9}, {16.0, 17}};
		runPowerTransformTest("box-cox", false, input, false);
	}

	@Test
	public void testPowerTransformYeoJohnsonLambdaAboveInitialInterval() {
		double[][] input = {{0.00}, {0.97}, {0.98}, {0.99}, {1.00}};
		runPowerTransformTest("yeo-johnson", false, input, false, OUTSIDE_INTERVAL_Y_EPS);
		assertLambdaOutsideInitialInterval(true);
	}

	@Test
	public void testPowerTransformBoxCoxLambdaBelowInitialInterval() {
		double[][] input = {{1.00}, {1.01}, {1.02}, {1.03}, {10.0}};
		runPowerTransformTest("box-cox", false, input, false);
		assertLambdaOutsideInitialInterval(false);
	}

	@Test
	public void testPowerTransformBoxCoxRejectsNonPositiveInput() {
		double[][] input = {{0, 1}, {1, 2}};
		runPowerTransformTest("box-cox", false, input, true);
	}

	@Test
	public void testPowerTransformApplyYeoJohnsonDenseCP() {
		double[][] input = {{-2, -2, -2}, {-1, -1, -1}, {0, 0, 0}, {1, 1, 1}, {2, 2, 2}};
		runPowerTransformApplyTest(ExecType.CP, "yeo-johnson", true, input, false);
	}

	@Test
	public void testPowerTransformApplyBoxCoxDenseCP() {
		double[][] input = {{0.5, 0.5, 0.5}, {1.0, 1.0, 1.0}, {2.0, 2.0, 2.0}, {4.0, 4.0, 4.0}, {8.0, 8.0, 8.0}};
		runPowerTransformApplyTest(ExecType.CP, "box-cox", false, input, false);
	}

	@Test
	public void testPowerTransformApplyBoxCoxRejectsNonPositiveInput() {
		double[][] input = {{0, 1, 2}, {1, 2, 3}};
		runPowerTransformApplyTest(ExecType.CP, "box-cox", false, input, true);
	}

	private void runPowerTransformTest(String method, boolean standardize, double[][] input, boolean shouldFail) {
		runPowerTransformTest(method, standardize, input, shouldFail, TRANSFORM_EPS);
	}

	private void runPowerTransformTest(String method, boolean standardize, double[][] input, boolean shouldFail,
		double yEps) {
		ExecMode oldExecMode = setExecMode(ExecType.CP);

		try {
			loadTestConfiguration(getTestConfiguration(TRANSFORM_TEST_NAME));

			String home = SCRIPT_DIR + TEST_DIR;
			fullDMLScriptName = home + TRANSFORM_TEST_NAME + ".dml";
			fullRScriptName = home + TRANSFORM_TEST_NAME + ".R";
			programArgs = new String[] {"-args", input("X"), output("Y"), output("L"), output("S"), method,
				Boolean.toString(standardize)};

			String referenceMethod = method.equals("default") ? "yeo-johnson" : method;
			rCmd = getRCmd(inputDir(), expectedDir(), referenceMethod, Boolean.toString(standardize));

			writeInputMatrixWithMTD("X", input, true);
			runTest(true, shouldFail, shouldFail ? DMLScriptException.class : null, -1);
			if(shouldFail)
				return;

			runRScript(true);
			compareOutput("Y", yEps);
			compareOutput("L", TRANSFORM_EPS);
			compareOutput("S", TRANSFORM_EPS);
		}
		catch(Exception exception) {
			throw new RuntimeException(exception);
		}
		finally {
			resetExecMode(oldExecMode);
		}
	}

	private void runPowerTransformApplyTest(ExecType execType, String method, boolean standardize, double[][] input,
		boolean shouldFail) {
		ExecMode oldExecMode = setExecMode(execType);

		try {
			loadTestConfiguration(getTestConfiguration(APPLY_TEST_NAME));

			String home = SCRIPT_DIR + TEST_DIR;
			fullDMLScriptName = home + APPLY_TEST_NAME + ".dml";
			fullRScriptName = home + APPLY_TEST_NAME + ".R";
			programArgs = new String[] {"-args", input("X"), input("L"), input("M"), input("S"), output("Y"), method,
				Boolean.toString(standardize)};

			rCmd = getRCmd(inputDir(), expectedDir(), method, Boolean.toString(standardize));

			double[][] L = {{0, 1, 2}};
			double[][] means = {{0.5, 1.0, 1.5}};
			double[][] scales = {{1.5, 2.0, 2.5}};

			writeInputMatrixWithMTD("X", input, true);
			writeInputMatrixWithMTD("L", L, true);
			writeInputMatrixWithMTD("M", means, true);
			writeInputMatrixWithMTD("S", scales, true);

			runTest(true, shouldFail, shouldFail ? DMLScriptException.class : null, -1);
			if(shouldFail)
				return;

			runRScript(true);
			compareOutput("Y", APPLY_EPS);
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
}
