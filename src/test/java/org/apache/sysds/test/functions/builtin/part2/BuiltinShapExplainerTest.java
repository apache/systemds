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

import org.junit.Test;

import java.util.HashMap;

import org.apache.sysds.common.Types.ExecMode;
import org.apache.sysds.runtime.matrix.data.MatrixValue.CellIndex;
import org.apache.sysds.test.AutomatedTestBase;
import org.apache.sysds.test.TestConfiguration;
import org.apache.sysds.test.TestUtils;

public class BuiltinShapExplainerTest extends AutomatedTestBase
{
	private static final String TEST_NAME = "shapExplainer";
	private static final String TEST_DIR = "functions/builtin/";
	private static final String TEST_CLASS_DIR = TEST_DIR + BuiltinShapExplainerTest.class.getSimpleName() + "/";

	//FIXME need for padding result with zero
	
	@Override
	public void setUp() {
		addTestConfiguration(TEST_NAME, new TestConfiguration(TEST_CLASS_DIR, TEST_NAME, new String[]{"R"}));
	}

	@Test
	public void testPrepareMaskForPermutation() {
		runShapExplainerUnitTest("prepare_mask_for_permutation");
	}

	@Test
	public void testPrepareMaskForPartialPermutation() {
		runShapExplainerUnitTest("prepare_mask_for_partial_permutation");
	}

	@Test
	public void testPrepareMaskForPartitionedPermutation() {
		runShapExplainerUnitTest("prepare_mask_for_partitioned_permutation");
	}

	@Test
	public void testComputeMeansFromPredictions() {
		runShapExplainerUnitTest("compute_means_from_predictions");
	}

	@Test
	public void testComputePhisFromPredictionMeans() {
		runShapExplainerUnitTest("compute_phis_from_prediction_means");
	}

	@Test
	public void testComputePhisFromPredictionMeansNonVars() {
		runShapExplainerUnitTest("compute_phis_from_prediction_means_non_vars");
	}

	@Test
	public void testPrepareFullMask() {
		runShapExplainerUnitTest("prepare_full_mask");
	}

	@Test
	public void testPrepareMaskedXBg() {
		runShapExplainerUnitTest("prepare_masked_X_bg");
	}

	@Test
	public void testPrepareMaskedXBgIndependentPerms() {
		runShapExplainerUnitTest("prepare_masked_X_bg_independent_perms");
	}

	@Test
	public void testApplyFullMask() {
		runShapExplainerUnitTest("apply_full_mask");
	}

	private void runShapExplainerUnitTest(String testType) {
		ExecMode platformOld = setExecMode(ExecMode.SINGLE_NODE);
		loadTestConfiguration(getTestConfiguration(TEST_NAME));
		String HOME = SCRIPT_DIR + TEST_DIR;

		try {
			loadTestConfiguration(getTestConfiguration(TEST_NAME));

			//execute given unit test
			fullDMLScriptName = HOME + TEST_NAME + "Unit.dml";
			programArgs = new String[]{"-args", testType, output("R"), output("R_expected")};
			runTest(true, false, null, -1);

			//compare to expected result
			HashMap<CellIndex, Double> result = readDMLMatrixFromOutputDir("R");
			HashMap<CellIndex, Double> result_expected = readDMLMatrixFromOutputDir("R_expected");

			TestUtils.compareMatrices(result, result_expected, 1e-3, testType+"_result", testType+"_expected");

		}
		finally {
			rtplatform = platformOld;
		}
	}

	@Test
	public void testShapExplainerDummyData(){
		runShapExplainerComponentTest(false);
	}
	//TODO add test with real data

	private void runShapExplainerComponentTest(Boolean useRealData) {
		ExecMode platformOld = setExecMode(ExecMode.SINGLE_NODE);
		loadTestConfiguration(getTestConfiguration(TEST_NAME));
		String HOME = SCRIPT_DIR + TEST_DIR;

		try {
			loadTestConfiguration(getTestConfiguration(TEST_NAME));

			//execute given unit test
			fullDMLScriptName = HOME + TEST_NAME + "Component.dml";
			programArgs = new String[]{"-args", output("R"), output("R_expected")};
			runTest(true, false, null, -1);

			//compare to expected phis
			HashMap<CellIndex, Double> result = readDMLMatrixFromOutputDir("R_phis");
			HashMap<CellIndex, Double> result_expected = readDMLMatrixFromOutputDir("R_expected_phis");

			TestUtils.compareMatrices(result, result_expected, 1e-3, "explainer_result_phis", "explainer_expected_phis");

			//compare to expected value of model
			HashMap<CellIndex, Double> result_e = readDMLMatrixFromOutputDir("R_e");
			HashMap<CellIndex, Double> result_expected_e = readDMLMatrixFromOutputDir("R_expected_e");

			TestUtils.compareMatrices(result_e, result_expected_e, 1e-3, "explainer_result_e", "explainer_expected_e");
		}
		finally {
			rtplatform = platformOld;
		}
	}
}
