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
import org.apache.sysds.runtime.matrix.data.MatrixValue;
import org.apache.sysds.test.AutomatedTestBase;
import org.apache.sysds.test.TestConfiguration;
import org.apache.sysds.test.TestUtils;
import org.junit.Test;

import java.util.HashMap;

public class BuiltinXgBoostTest_classification extends AutomatedTestBase {
	private final static String TEST_NAME = "xgboost_classification";
	private final static String TEST_DIR = "functions/builtin/";
	private static final String TEST_CLASS_DIR = TEST_DIR + BuiltinXgBoostTest_classification.class.getSimpleName() + "/";
	double eps = 1e-10;

	@Override
	public void setUp() {
		TestUtils.clearAssertionInformation();
		addTestConfiguration(TEST_NAME, new TestConfiguration(TEST_CLASS_DIR, TEST_NAME, new String[]{"C"}));
	}

	@Test
	public void testXgBoost() {
		executeXgBoost(Types.ExecMode.SINGLE_NODE);
	}

	private void executeXgBoost(Types.ExecMode mode) {
		Types.ExecMode platformOld = setExecMode(mode);
		try {
			loadTestConfiguration(getTestConfiguration(TEST_NAME));

			String HOME = SCRIPT_DIR + TEST_DIR;
			fullDMLScriptName = HOME + TEST_NAME + ".dml";
			programArgs = new String[]{"-args", input("X"), input("y"), input("R"), input("sml_type"),
					input("num_trees"), output("M")};

			double[][] y = {
					{1.0},
					{1.0},
					{0.0},
					{1.0},
					{1.0},
					{0.0},
					{1.0},
					{0.0}};

			double[][] X = {
					{12.0, 1.0},
					{15.0, 0.0},
					{24.0, 0.0},
					{20.0, 1.0},
					{25.0, 1.0},
					{17.0, 0.0},
					{16.0, 1.0},
					{32.0, 1.0}};

			double[][] R = {
					{1.0, 2.0}};

			writeInputMatrixWithMTD("X", X, true);
			writeInputMatrixWithMTD("y", y, true);
			writeInputMatrixWithMTD("R", R, true);

			runTest(true, false, null, -1);

			HashMap<MatrixValue.CellIndex, Double> actual_M = readDMLMatrixFromOutputDir("M");

			// root node of first tree
			TestUtils.compareScalars(actual_M.get(new MatrixValue.CellIndex(1,2)), 1.0, eps);
			TestUtils.compareScalars(actual_M.get(new MatrixValue.CellIndex(2,2)), 1.0, eps);
			TestUtils.compareScalars(actual_M.get(new MatrixValue.CellIndex(3,2)), 1.0, eps);
			TestUtils.compareScalars(actual_M.get(new MatrixValue.CellIndex(4,2)), 2.0, eps);
			TestUtils.compareScalars(actual_M.get(new MatrixValue.CellIndex(5,2)), 2.0, eps);
			TestUtils.compareScalars(String.valueOf(actual_M.get(new MatrixValue.CellIndex(6, 2))), "null");

			// random node of first tree
			TestUtils.compareScalars(actual_M.get(new MatrixValue.CellIndex(1,12)), 63.0, eps);
			TestUtils.compareScalars(actual_M.get(new MatrixValue.CellIndex(2,12)), 1.0, eps);
			TestUtils.compareScalars(String.valueOf(actual_M.get(new MatrixValue.CellIndex(3, 12))), "null");

			// random leaf node of first tree
			TestUtils.compareScalars(actual_M.get(new MatrixValue.CellIndex(1,15)), 3.0, eps);
			TestUtils.compareScalars(actual_M.get(new MatrixValue.CellIndex(2,15)), 2.0, eps);
			TestUtils.compareScalars(actual_M.get(new MatrixValue.CellIndex(3,15)), 3.0, eps);
			TestUtils.compareScalars(actual_M.get(new MatrixValue.CellIndex(4,15)), 2.0, eps);
			TestUtils.compareScalars(actual_M.get(new MatrixValue.CellIndex(5,15)), 2.0, eps);
			TestUtils.compareScalars(String.valueOf(actual_M.get(new MatrixValue.CellIndex(6, 15))), "null");

			// root node of second tree
			TestUtils.compareScalars(actual_M.get(new MatrixValue.CellIndex(1,17)), 5.0, eps);
			TestUtils.compareScalars(actual_M.get(new MatrixValue.CellIndex(2,17)), 2.0, eps);
			TestUtils.compareScalars(actual_M.get(new MatrixValue.CellIndex(3,17)), 3.0, eps);
			TestUtils.compareScalars(actual_M.get(new MatrixValue.CellIndex(4,17)), 2.0, eps);
			TestUtils.compareScalars(actual_M.get(new MatrixValue.CellIndex(5,17)), 2.0, eps);
			TestUtils.compareScalars(String.valueOf(actual_M.get(new MatrixValue.CellIndex(6, 17))), "null");

			// random node of second tree
			TestUtils.compareScalars(actual_M.get(new MatrixValue.CellIndex(1,18)), 6.0, eps);
			TestUtils.compareScalars(actual_M.get(new MatrixValue.CellIndex(2,18)), 2.0, eps);
			TestUtils.compareScalars(String.valueOf(actual_M.get(new MatrixValue.CellIndex(3, 18))), "null");

			//random leaf node of second tree
			TestUtils.compareScalars(actual_M.get(new MatrixValue.CellIndex(1,28)), 40.0, eps);
			TestUtils.compareScalars(actual_M.get(new MatrixValue.CellIndex(2,28)), 2.0, eps);
			TestUtils.compareScalars(String.valueOf(actual_M.get(new MatrixValue.CellIndex(3, 28))), "null");
		}
		catch (Exception ex) {
			System.out.println("[ERROR] Xgboost test failed, cause: " + ex);
			throw ex;
		} finally {
			rtplatform = platformOld;
		}
	}
}
