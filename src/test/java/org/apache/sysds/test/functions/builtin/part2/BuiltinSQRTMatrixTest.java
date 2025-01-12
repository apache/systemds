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
import org.apache.sysds.common.Types.ExecType;
import org.apache.sysds.runtime.matrix.data.MatrixValue;
import org.apache.sysds.test.AutomatedTestBase;
import org.apache.sysds.test.TestConfiguration;
import org.apache.sysds.test.TestUtils;
import org.junit.Test;

import java.util.HashMap;

public class BuiltinSQRTMatrixTest extends AutomatedTestBase {
	private final static String TEST_NAME = "SQRTMatrix";
	private final static String TEST_DIR = "functions/builtin/";
	private static final String TEST_CLASS_DIR = TEST_DIR + BuiltinSQRTMatrixTest.class.getSimpleName() + "/";

	private final static double eps = 1e-8;

	@Override
	public void setUp() {
		addTestConfiguration(TEST_NAME, new TestConfiguration(TEST_CLASS_DIR, TEST_NAME, new String[] {"C"}));
	}

/*
	// tests for strategy "COMMON"
	@Test
	public void testSQRTMatrixJavaSquareMatrixSize1x1() {
		runSQRTMatrix(true, ExecType.CP, "COMMON", 1);
	}

	@Test
	public void testSQRTMatrixJavaSquareMatrixSize2x2() {
		runSQRTMatrix(true, ExecType.CP, "COMMON", 2);
	}

	@Test
	public void testSQRTMatrixJavaSquareMatrixSize4x4() {
		runSQRTMatrix(true, ExecType.CP, "COMMON", 3);
	}

	@Test
	public void testSQRTMatrixJavaSquareMatrixSize8x8() {
		runSQRTMatrix(true, ExecType.CP, "COMMON", 4);
	}

	@Test
	public void testSQRTMatrixJavaDiagonalMatrixSize2x2() {
		runSQRTMatrix(true, ExecType.CP, "COMMON", 5);
	}

	@Test
	public void testSQRTMatrixJavaDiagonalMatrixSize3x3() {
		runSQRTMatrix(true, ExecType.CP, "COMMON", 6);
	}

	@Test
	public void testSQRTMatrixJavaDiagonalMatrixSize4x4() {
		runSQRTMatrix(true, ExecType.CP, "COMMON", 7);
	}

	@Test
	public void testSQRTMatrixJavaPSDMatrixSize2x2() {
		runSQRTMatrix(true, ExecType.CP, "COMMON", 8);
	}

	@Test
	public void testSQRTMatrixJavaPSDMatrixSize4x4() {
		runSQRTMatrix(true, ExecType.CP, "COMMON", 9);
	}

	@Test
	public void testSQRTMatrixJavaPSDMatrixSize3x3() {
		runSQRTMatrix(true, ExecType.CP, "COMMON", 10);
	}
*/

	// tests for strategy "DML"

	@Test
	public void testSQRTMatrixDMLSquareMatrixSize1x1() {
		runSQRTMatrix(true, ExecType.CP, "DML", 1);
	}

	@Test
	public void testSQRTMatrixDMLSquareMatrixSize2x2() {
		runSQRTMatrix(true, ExecType.CP, "DML", 2);
	}

	@Test
	public void testSQRTMatrixDMLSquareMatrixSize4x4() {
		runSQRTMatrix(true, ExecType.CP, "DML", 3);
	}

	@Test
	public void testSQRTMatrixDMLSquareMatrixSize8x8() {
		runSQRTMatrix(true, ExecType.CP, "DML", 4);
	}

	@Test
	public void testSQRTMatrixDMLDiagonalMatrixSize2x2() {
		runSQRTMatrix(true, ExecType.CP, "DML", 5);
	}

	@Test
	public void testSQRTMatrixDMLDiagonalMatrixSize3x3() {
		runSQRTMatrix(true, ExecType.CP, "DML", 6);
	}

	@Test
	public void testSQRTMatrixDMLDiagonalMatrixSize4x4() {
		runSQRTMatrix(true, ExecType.CP, "DML", 7);
	}

	@Test
	public void testSQRTMatrixDMLPSDMatrixSize2x2() {
		runSQRTMatrix(true, ExecType.CP, "DML", 8);
	}

	@Test
	public void testSQRTMatrixDMLPSDMatrixSize4x4() {
		runSQRTMatrix(true, ExecType.CP, "DML", 9);
	}

	@Test
	public void testSQRTMatrixDMLPSDMatrixSize3x3() {
		runSQRTMatrix(true, ExecType.CP, "DML", 10);
	}


	private void runSQRTMatrix(boolean defaultProb, ExecType instType, String strategy, int test_case) {
		Types.ExecMode platformOld = setExecMode(instType);
		try {
			loadTestConfiguration(getTestConfiguration(TEST_NAME));

			// find path to associated dml script and define parameters
			String HOME = SCRIPT_DIR + TEST_DIR;
			fullDMLScriptName = HOME + TEST_NAME + ".dml";
			programArgs = new String[] {"-args", input("X"), strategy, output("Y")};

			// define input matrix for the matrix sqrt function according to test case
			double[][] X = null;
			switch(test_case) {
				case 1: // arbitrary square matrix of dimension 1x1
					double[][] X1 = {
							{4}
					};
					X = X1;
					break;
				case 2: // arbitrary square matrix of dimension 2x2
					double[][] X2 = {
							{1, 1},
							{0, 1},
					};
					X = X2;
					break;
				case 3: // arbitrary square matrix of dimension 4x4
					double[][] X3 = {
							{1, 2, 3, 4},
							{5.2, 6, 7, 8},
							{9, 10.5, 11, 12.3},
							{13, 14, 15.8, 16}
					};
					X = X3;
					break;
				case 4: // arbitrary square matrix of dimension 8x8
					double[][] X4 = {
							{1, 2, 3, 4, 5, 6, 7, 8},
							{9, 10, 11, 12, 13, 14, 15, 16},
							{17, 18, 19, 20, 21, 22, 23, 24},
							{25, 26, 27, 28, 29, 30, 31, 32},
							{33, 34, 35, 36, 37, 38, 39, 40},
							{41, 42, 43, 44, 45, 46, 47, 48},
							{49, 50, 51, 52, 53, 54, 55, 56},
							{57, 58, 59, 60, 61, 62, 63, 64}
					};
					X = X4;
					break;
				case 5: // arbitrary diagonal matrix of dimension 2x2
					double[][] X5 = {
							{1, 0},
							{0, 1},
					};
					X = X5;
					break;
				case 6: // arbitrary diagonal matrix of dimension 3x3
					double[][] X6 = {
							{-1, 0, 0},
							{0, 2, 0},
							{0, 0, 3}
					};
					X = X6;
					break;
				case 7: // arbitrary diagonal matrix of dimension 4x4
					double[][] X7 = {
							{-4.5, 0, 0, 0},
							{0, -2, 0, 0},
							{0, 0, -3.2, 0},
							{0, 0, 0, 6}
					};
					X = X7;
					break;
				case 8: // arbitrary PSD matrix of dimension 2x2
					// PSD matrix generated by taking (A^T)A of matrix A = [[1, 0], [2, 3]]
					double[][] X8 = {
							{1, 2},
							{2, 13}
					};
					X = X8;
					break;
				case 9: // arbitrary PSD matrix of dimension 4x4
					// PSD matrix generated by taking (A^T)A of matrix A=
					// [[1, 0, 5, 6],
					//  [2, 3, 0, 2],
					//  [5, 0, 1, 1],
					//  [2, 3, 4, 8]]
					double[][] X9 = {
							{62, 14, 16, 70},
							{14, 17, 12, 29},
							{16, 12, 27, 22},
							{70, 29, 22, 93}
					};
					X = X9;
					break;
				case 10: // arbitrary PSD matrix of dimension 3x3
					// PSD matrix generated by taking (A^T)A of matrix A =
					// [[1.5, 0, 1.2],
					// [2.2, 3.8, 4.4],
					// [4.2, 6.1, 0.2]]
					double[][] X10 = {
							{3.69, 8.58, 6.54},
							{8.58, 38.64, 33.30},
							{6.54, 33.3, 54.89}
					};
					X = X10;
					break;
			}

			assert X != null;

			// write the input matrix and strategy for matrix sqrt function to dml script
            writeInputMatrixWithMTD("X", X, true);

			// run the test dml script
			runTest(true, false, null, -1);

			// read the result matrix from the dml script output Y
			HashMap<MatrixValue.CellIndex, Double> actual_Y = readDMLMatrixFromOutputDir("Y");

			//System.out.println("This is the actual Y: " + actual_Y);

			// create a HashMap with Matrix Values from the input matrix X to compare to the received output matrix
			HashMap<MatrixValue.CellIndex, Double> expected_Y = new HashMap<>();
			for (int r = 0; r < X.length; r++) {
				for (int c = 0; c < X[0].length; c++) {
					expected_Y.put(new MatrixValue.CellIndex(r + 1, c + 1), X[r][c]);
				}
			}

			// compare the expected matrix (the input matrix X) with the received output matrix Y, which should be the (SQRT_MATRIX(X))^2 = X again
			TestUtils.compareMatrices(expected_Y, actual_Y, eps, "Expected-DML", "Actual-DML");
		}
		finally {
			resetExecMode(platformOld);
		}
	}
}
