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

import org.apache.sysds.common.Types;
import org.apache.sysds.common.Types.ExecMode;
import org.apache.sysds.runtime.matrix.data.MatrixValue;
import org.apache.sysds.test.AutomatedTestBase;
import org.apache.sysds.test.TestConfiguration;
import org.apache.sysds.test.TestUtils;
import org.junit.Assert;
import org.junit.Test;

import java.util.HashMap;

public class BuiltinXgBoostPredictTest_classification extends AutomatedTestBase {
	private static final String TEST_NAME = "xgboost_predict_classification";
	private static final String TEST_DIR = "functions/builtin/";
	private static final String TEST_CLASS_DIR = TEST_DIR + BuiltinXgBoostPredictTest_regression.class.getSimpleName() + "/";

	double threshold = 0.25;

	@Override
	public void setUp() {
		TestUtils.clearAssertionInformation();
		addTestConfiguration(TEST_NAME, new TestConfiguration(TEST_CLASS_DIR, TEST_NAME, new String[]{"C"}));
	}

	@Test
	public void testXgBoost() {
		executeXgBoost(Types.ExecMode.SINGLE_NODE);
	}

	private void executeXgBoost(ExecMode mode) {
		ExecMode platformOld = setExecMode(mode);
		try {
			loadTestConfiguration(getTestConfiguration(TEST_NAME));

			String HOME = SCRIPT_DIR + TEST_DIR;
			fullDMLScriptName = HOME + TEST_NAME + ".dml";
			programArgs = new String[]{"-args",
				output("P"), output("y"), DATASET_DIR+"games/games.csv"};

			runTest(true, false, null, -1);

			HashMap<MatrixValue.CellIndex, Double> predicted_values = readDMLMatrixFromOutputDir("P");
			HashMap<MatrixValue.CellIndex, Double> actual_values = readDMLMatrixFromOutputDir("y");

			int target_size = predicted_values.size() + 1;
			for(int i = 1; i < target_size; i++)
				actual_values.putIfAbsent(new MatrixValue.CellIndex(i, 1), 0.0);

			int miss_class_counter = 0;
			for(int i = 1; i < target_size; i++) {
				double act = actual_values.get(new MatrixValue.CellIndex(i,1));
				double pred = predicted_values.get(new MatrixValue.CellIndex(i, 1));

				if(pred < 0.5 && act == 1 || pred > 0.5 && act == 0)
					miss_class_counter++;
			}
			float val = (float) miss_class_counter / actual_values.size();
			Assert.assertTrue(val < threshold);
		}
		catch (Exception ex) {
			System.out.println("[ERROR] Xgboost test failed, cause: " + ex);
			throw ex;
		} finally {
			rtplatform = platformOld;
		}
	}
}
