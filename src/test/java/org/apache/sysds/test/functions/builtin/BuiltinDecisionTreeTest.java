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

package org.apache.sysds.test.functions.builtin;

import java.util.HashMap;

import org.apache.sysds.common.Types;
import org.apache.sysds.lops.LopProperties;
import org.apache.sysds.runtime.matrix.data.MatrixValue;
import org.apache.sysds.test.AutomatedTestBase;
import org.apache.sysds.test.TestConfiguration;
import org.apache.sysds.test.TestUtils;
import org.junit.Test;

public class BuiltinDecisionTreeTest extends AutomatedTestBase {
	private final static String TEST_NAME = "decisionTree";
	private final static String TEST_DIR = "functions/builtin/";
	private static final String TEST_CLASS_DIR = TEST_DIR + BuiltinDecisionTreeTest.class.getSimpleName() + "/";

	private final static double eps = 1e-10;

	@Override
	public void setUp() {
		addTestConfiguration(TEST_NAME, new TestConfiguration(TEST_CLASS_DIR, TEST_NAME, new String[] {"C"}));
	}

	@Test
	public void testDecisionTreeDefaultCP() {
		runDecisionTree(true, LopProperties.ExecType.CP);
	}

	@Test
	public void testDecisionTreeSP() {
		runDecisionTree(true, LopProperties.ExecType.SPARK);
	}

	private void runDecisionTree(boolean defaultProb, LopProperties.ExecType instType) {
		Types.ExecMode platformOld = setExecMode(instType);
		try {
			loadTestConfiguration(getTestConfiguration(TEST_NAME));

			String HOME = SCRIPT_DIR + TEST_DIR;
			fullDMLScriptName = HOME + TEST_NAME + ".dml";
			programArgs = new String[] {"-args", input("X"), input("Y"), input("R"), output("M")};

			double[][] Y = {{1.0}, {0.0}, {0.0}, {1.0}, {0.0}};

			double[][] X = {{4.5, 4.0, 3.0, 2.8, 3.5}, {1.9, 2.4, 1.0, 3.4, 2.9}, {2.0, 1.1, 1.0, 4.9, 3.4},
				{2.3, 5.0, 2.0, 1.4, 1.8}, {2.1, 1.1, 3.0, 1.0, 1.9},};
			writeInputMatrixWithMTD("X", X, true);
			writeInputMatrixWithMTD("Y", Y, true);

			double[][] R = {{1.0, 1.0, 3.0, 1.0, 1.0},};
			writeInputMatrixWithMTD("R", R, true);

			runTest(true, false, null, -1);

			HashMap<MatrixValue.CellIndex, Double> actual_M = readDMLMatrixFromOutputDir("M");
			HashMap<MatrixValue.CellIndex, Double> expected_M = new HashMap<>();
			expected_M.put(new MatrixValue.CellIndex(1, 1), 1.0);
			expected_M.put(new MatrixValue.CellIndex(1, 3), 3.0);
			expected_M.put(new MatrixValue.CellIndex(3, 1), 2.0);
			expected_M.put(new MatrixValue.CellIndex(1, 2), 2.0);
			expected_M.put(new MatrixValue.CellIndex(2, 1), 1.0);
			expected_M.put(new MatrixValue.CellIndex(5, 1), 1.0);
			expected_M.put(new MatrixValue.CellIndex(4, 1), 1.0);
			expected_M.put(new MatrixValue.CellIndex(5, 3), 1.0);
			expected_M.put(new MatrixValue.CellIndex(5, 2), 1.0);
			expected_M.put(new MatrixValue.CellIndex(6, 1), 3.2);

			TestUtils.compareMatrices(expected_M, actual_M, eps, "Expected-DML", "Actual-DML");
		}
		finally {
			rtplatform = platformOld;
		}
	}
}
