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

package org.apache.sysds.test.functions.builtin.part1;

import java.util.HashMap;

import org.apache.sysds.common.Types;
import org.apache.sysds.common.Types.ExecType;
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
		runDecisionTree(true, ExecType.CP);
	}

	@Test
	public void testDecisionTreeSP() {
		runDecisionTree(true, ExecType.SPARK);
	}

	private void runDecisionTree(boolean defaultProb, ExecType instType) {
		Types.ExecMode platformOld = setExecMode(instType);
		try {
			loadTestConfiguration(getTestConfiguration(TEST_NAME));

			String HOME = SCRIPT_DIR + TEST_DIR;
			fullDMLScriptName = HOME + TEST_NAME + ".dml";
			programArgs = new String[] {"-args", input("X"), input("Y"), input("R"), output("M")};

			double[][] Y = {{2.0}, {1.0}, {1.0}, {2.0}, {1.0}};
			double[][] X = {
				{3, 1, 2, 1, 5}, 
				{2, 1, 2, 2, 4}, 
				{1, 1, 1, 3, 3},
				{4, 2, 1, 4, 2}, 
				{2, 2, 1, 5, 1},};
			double[][] R = {{1.0, 1.0, 2.0, 1.0, 1.0, 1.0},};
			writeInputMatrixWithMTD("X", X, true);
			writeInputMatrixWithMTD("Y", Y, true);
			writeInputMatrixWithMTD("R", R, true);

			runTest(true, false, null, -1);

			HashMap<MatrixValue.CellIndex, Double> actual_M = readDMLMatrixFromOutputDir("M");
			HashMap<MatrixValue.CellIndex, Double> expected_M = new HashMap<>();
			expected_M.put(new MatrixValue.CellIndex(1, 1), 1.0); //split feature 1
			expected_M.put(new MatrixValue.CellIndex(1, 2), 2.0); // <= 2
			expected_M.put(new MatrixValue.CellIndex(1, 3), 0.0); //left leaf node
			expected_M.put(new MatrixValue.CellIndex(1, 4), 1.0); // class 1
			expected_M.put(new MatrixValue.CellIndex(1, 5), 0.0); //right leaf node
			expected_M.put(new MatrixValue.CellIndex(1, 6), 2.0); // class 2

			TestUtils.compareMatrices(expected_M, actual_M, eps, "Expected-DML", "Actual-DML");
		}
		finally {
			resetExecMode(platformOld);
		}
	}
}
