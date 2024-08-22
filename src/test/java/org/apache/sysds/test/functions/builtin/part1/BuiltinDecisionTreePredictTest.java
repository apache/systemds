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
import org.junit.Ignore;
import org.junit.Test;

public class BuiltinDecisionTreePredictTest extends AutomatedTestBase {
	private final static String TEST_NAME = "decisionTreePredict";
	private final static String TEST_DIR = "functions/builtin/";
	private static final String TEST_CLASS_DIR = TEST_DIR + BuiltinDecisionTreeTest.class.getSimpleName() + "/";

	private final static double eps = 1e-10;

	@Override
	public void setUp() {
		addTestConfiguration(TEST_NAME, new TestConfiguration(TEST_CLASS_DIR, TEST_NAME, new String[] {"C"}));
	}

	@Test
	public void testDecisionTreeTTPredictDefaultCP1() {
		runDecisionTreePredict(true, ExecType.CP, "TT", 1);
	}

	@Test
	public void testDecisionTreeTTPredictDefaultCP2() {
		runDecisionTreePredict(true, ExecType.CP, "TT", 2);
	}

	@Test
	public void testDecisionTreeTTPredictDefaultCP3() {
		runDecisionTreePredict(true, ExecType.CP, "TT", 3);
	}

	@Test
	public void testDecisionTreeTTPredictDefaultCP4() {
		runDecisionTreePredict(true, ExecType.CP, "TT", 4);
	}


	@Test
	public void testDecisionTreeTTPredictSP() {
		runDecisionTreePredict(true, ExecType.SPARK, "TT", 1);
	}
	
	@Test
	public void testDecisionTreeGEMMPredictDefaultCP1() {
		runDecisionTreePredict(true, ExecType.CP, "GEMM", 1);
	}

	@Test
	public void testDecisionTreeGEMMPredictDefaultCP2() {
		runDecisionTreePredict(true, ExecType.CP, "GEMM", 2);
	}

	@Test
	public void testDecisionTreeGEMMPredictDefaultCP3() {
		runDecisionTreePredict(true, ExecType.CP, "GEMM", 3);
	}

	@Test
	public void testDecisionTreeGEMMPredictDefaultCP4() {
		runDecisionTreePredict(true, ExecType.CP, "GEMM", 4);
	}


	@Test
	public void testDecisionTreeGEMMPredictSP() {
		runDecisionTreePredict(true, ExecType.SPARK, "GEMM", 1);
	}

	private void runDecisionTreePredict(boolean defaultProb, ExecType instType, String strategy, int test_case) {
		Types.ExecMode platformOld = setExecMode(instType);
		try {
			loadTestConfiguration(getTestConfiguration(TEST_NAME));

			String HOME = SCRIPT_DIR + TEST_DIR;
			fullDMLScriptName = HOME + TEST_NAME + ".dml";
			programArgs = new String[] {"-args", input("M"), input("X"), strategy, output("Y")};

			//data and model consistent with decision tree test
			double[][] X = null;
			double[][] M = null;

			HashMap<MatrixValue.CellIndex, Double> expected_Y = new HashMap<>();
			switch(test_case){
				case 1:
					double[][] X1 = {
							{3, 1, 2, 1, 5},
							{2, 1, 2, 2, 4},
							{1, 1, 1, 3, 3},
							{4, 2, 1, 4, 2},
							{2, 2, 1, 5, 1},};
					double[][] M1 = {{1.0, 2.0, 0.0, 1.0, 0.0, 2.0}};

					expected_Y.put(new MatrixValue.CellIndex(1, 1), 2.0);
					expected_Y.put(new MatrixValue.CellIndex(2, 1), 1.0);
					expected_Y.put(new MatrixValue.CellIndex(3, 1), 1.0);
					expected_Y.put(new MatrixValue.CellIndex(4, 1), 2.0);
					expected_Y.put(new MatrixValue.CellIndex(5, 1), 1.0);
					X = X1;
					M = M1;
					break;
				case 2:
					double[][] X2 = {
							{3, 1, 2, 1},
							{2, 1, 2, 6},
							{1, 1, 1, 3},
							{9, 2, 1, 7},
							{2, 2, 1, 1},};
					double[][] M2 = {{4, 5, 0, 2, 1, 7, 0, 0, 0, 0, 0, 2, 0, 1}};

					expected_Y.put(new MatrixValue.CellIndex(1, 1), 2.0);
					expected_Y.put(new MatrixValue.CellIndex(2, 1), 2.0);
					expected_Y.put(new MatrixValue.CellIndex(3, 1), 2.0);
					expected_Y.put(new MatrixValue.CellIndex(4, 1), 1.0);
					expected_Y.put(new MatrixValue.CellIndex(5, 1), 2.0);
					X = X2;
					M = M2;
					break;
				case 3:
					double[][] X3 = {
							{1, 1, 1},
							{1, 1, 7,},
							{1, 5, 1},
							{1, 5, 7,},};
					double[][] M3 = {{1, 5, 2, 4, 2, 4, 3, 6, 3, 6, 3, 6, 3, 6, 0, 1, 0, 2, 0, 3, 0, 4, 0, 5, 0, 6, 0, 7, 0, 8}};

					expected_Y.put(new MatrixValue.CellIndex(1, 1), 1.0);
					expected_Y.put(new MatrixValue.CellIndex(2, 1), 2.0);
					expected_Y.put(new MatrixValue.CellIndex(3, 1), 3.0);
					expected_Y.put(new MatrixValue.CellIndex(4, 1), 4.0);
					X = X3;
					M = M3;
					break;
				case 4:
					double[][] X4 = {
							{1, 1, 1, 1},
							{4, 1, 1, 1},
							{1, 1, 7, 1},
							{4, 1, 7, 1},
							{1, 5, 1, 1},
							{4, 5, 1, 1},
							{1, 5, 7, 1},
							{4, 5, 7, 1},
							{1, 1, 1, 6},
							{4, 1, 1, 6},
							{1, 1, 7, 6},
							{4, 1, 7, 6},
							{1, 5, 1, 6},
							{4, 5, 1, 6},
							{1, 5, 7, 6},
							{4, 5, 7, 6},};
					double[][] M4 = {{4, 5, 2, 4, 2, 4, 3, 6, 3, 6, 3, 6, 3, 6, 1, 3, 1, 3, 1, 3, 1, 3, 1, 3, 1, 3, 1,
							3, 1, 3, 0, 1, 0, 2, 0, 3, 0, 4, 0, 5, 0, 6, 0, 7, 0, 8, 0, 9, 0, 10, 0, 11, 0, 12, 0, 13,
							0, 14, 0, 15, 0, 16}};

					expected_Y.put(new MatrixValue.CellIndex(1, 1), 1.0);
					expected_Y.put(new MatrixValue.CellIndex(2, 1), 2.0);
					expected_Y.put(new MatrixValue.CellIndex(3, 1), 3.0);
					expected_Y.put(new MatrixValue.CellIndex(4, 1), 4.0);
					expected_Y.put(new MatrixValue.CellIndex(5, 1), 5.0);
					expected_Y.put(new MatrixValue.CellIndex(6, 1), 6.0);
					expected_Y.put(new MatrixValue.CellIndex(7, 1), 7.0);
					expected_Y.put(new MatrixValue.CellIndex(8, 1), 8.0);
					expected_Y.put(new MatrixValue.CellIndex(9, 1), 9.0);
					expected_Y.put(new MatrixValue.CellIndex(10, 1), 10.0);
					expected_Y.put(new MatrixValue.CellIndex(11, 1), 11.0);
					expected_Y.put(new MatrixValue.CellIndex(12, 1), 12.0);
					expected_Y.put(new MatrixValue.CellIndex(13, 1), 13.0);
					expected_Y.put(new MatrixValue.CellIndex(14, 1), 14.0);
					expected_Y.put(new MatrixValue.CellIndex(15, 1), 15.0);
					expected_Y.put(new MatrixValue.CellIndex(16, 1), 16.0);
					X = X4;
					M = M4;
					break;
			}

			writeInputMatrixWithMTD("M", M, true);
			writeInputMatrixWithMTD("X", X, true);

			runTest(true, false, null, -1);

			HashMap<MatrixValue.CellIndex, Double> actual_Y = readDMLMatrixFromOutputDir("Y");
			TestUtils.compareMatrices(expected_Y, actual_Y, eps, "Expected-DML", "Actual-DML");
		}
		finally {
			rtplatform = platformOld;
		}
	}
}
