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

package org.apache.sysds.test.functions.unique;

import org.apache.sysds.common.Types;
import org.junit.Test;

public class UniqueRowCol extends UniqueBase {
	private final static String TEST_NAME = "uniqueRowCol";
	private final static String TEST_DIR = "functions/unique/";
	private static final String TEST_CLASS_DIR = TEST_DIR + UniqueRowCol.class.getSimpleName() + "/";


	@Override
	protected String getTestName() {
		return TEST_NAME;
	}

	@Override
	protected String getTestDir() {
		return TEST_DIR;
	}

	@Override
	protected String getTestClassDir() {
		return TEST_CLASS_DIR;
	}

	@Test
	public void testBaseCase1CP() {
		double[][] inputMatrix = {{0}};
		double[][] expectedMatrix = {{0}};
		uniqueTest(inputMatrix, expectedMatrix, Types.ExecType.CP, 0.0);
	}

	@Test
	public void testBaseCase2CP() {
		double[][] inputMatrix = {{1}};
		double[][] expectedMatrix = {{1}};
		uniqueTest(inputMatrix, expectedMatrix, Types.ExecType.CP, 0.0);
	}

	@Test
	public void testSkinnySmallCP() {
		double[][] inputMatrix = {{1},{1},{6},{9},{4},{2},{0},{9},{0},{0},{4},{4}};
		double[][] expectedMatrix = {{1},{6},{9},{4},{2},{0}};
		uniqueTest(inputMatrix, expectedMatrix, Types.ExecType.CP, 0.0);
	}

	@Test
	public void testWideSmallCP() {
		double[][] inputMatrix = {{1,1,6,9,4,2,0,9,0,0,4,4}};
		double[][] expectedMatrix = {{1,6,9,4,2,0}};
		uniqueTest(inputMatrix, expectedMatrix, Types.ExecType.CP, 0.0);
	}

	@Test
	public void testSquareLargeCP() {
		double[][] inputMatrix = new double[1000][1000];
		// Input is a 1000 x 1000 matrix:
		// [1, 1, ..., 1, 2, 2, .., 2]
		// [1, 1, ..., 1, 2, 2, .., 2]
		// ..
		// [1, 1, ..., 1, 2, 2, .., 2]
		// [2, 2, ..., 2, 1, 1, .., 1]
		// [2, 2, ..., 2, 1, 1, .., 1]
		// ..
		// [2, 2, ..., 2, 1, 1, .., 1]
		for (int i=0; i<500; ++i) {
			for (int j=0; j<500; ++j) {
				inputMatrix[i][j] = 1;
				inputMatrix[i+500][j+500] = 1;
			}
		}
		for (int i=500; i<1000; ++i) {
			for (int j=0; j<500; ++j) {
				inputMatrix[i][j] = 2;
				inputMatrix[i-500][j+500] = 2;
			}
		}
		// Expect the output to be a skinny matrix due to the following condition in code:
		// (R >= C)? LibMatrixSketch.MatrixShape.SKINNY : LibMatrixSketch.MatrixShape.WIDE;
		double[][] expectedMatrix = {{1},{2}};
		uniqueTest(inputMatrix, expectedMatrix, Types.ExecType.CP, 0.0);
	}

	@Test
	public void testSkinnyLargeCP() {
		double[][] inputMatrix = new double[2000][2];
		// Input is a 2000 x 2 matrix:
		// [1, 2]
		// [1, 2]
		// ..
		// [1, 2]
		// [2, 1]
		// [2, 1]
		// ..
		// [2, 1]
		for (int i=0; i<1000; ++i) {
			inputMatrix[i][0] = 1;
			inputMatrix[i][1] = 2;
		}
		for (int i=1000; i<2000; ++i) {
			inputMatrix[i][0] = 2;
			inputMatrix[i][1] = 1;
		}
		double[][] expectedMatrix = {{1}, {2}};
		uniqueTest(inputMatrix, expectedMatrix, Types.ExecType.CP, 0.0);
	}

	@Test
	public void testWideLargeCP() {
		double[][] inputMatrix = new double[2][2000];
		// Input is a 2 x 2000 matrix:
		// [1, 1, ..., 1, 2, 2, .., 2]
		// [2, 2, ..., 2, 1, 1, .., 1]
		for (int j=0; j<1000; ++j) {
			inputMatrix[0][j] = 1;
			inputMatrix[1][j+1000] = 1;
		}
		for (int j=1000; j<2000; ++j) {
			inputMatrix[0][j] = 2;
			inputMatrix[1][j-1000] = 2;
		}
		double[][] expectedMatrix = {{1,2}};
		uniqueTest(inputMatrix, expectedMatrix, Types.ExecType.CP, 0.0);
	}
}
