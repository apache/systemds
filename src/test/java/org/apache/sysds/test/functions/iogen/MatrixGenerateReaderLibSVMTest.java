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

package org.apache.sysds.test.functions.iogen;

import org.junit.Test;

public class MatrixGenerateReaderLibSVMTest extends GenerateReaderMatrixTest {

	private final static String TEST_NAME = "MatrixGenerateReaderLibSVMTest";

	@Override
	protected String getTestName() {
		return TEST_NAME;
	}

	private void generateRandomLIBSVM(int firstIndex, int nrows, int ncols, double min, double max, double sparsity,
		String separator, String indexSeparator) {

		double[][] random = getRandomMatrix(nrows, ncols, min, max, sparsity, 714);
		sampleMatrix = new double[2 * nrows][ncols];
		StringBuilder sb = new StringBuilder();
		int indexRow = 0;
		for(int r = 0; r < nrows; r++) {
			StringBuilder row1 = new StringBuilder();
			StringBuilder row2 = new StringBuilder();
			row1.append("+1");

			for(int c = 0; c < ncols - 1; c++) {
				if(random[r][c] > 0) {
					sampleMatrix[indexRow][c] = random[r][c];
					row1.append(separator).append(c + firstIndex).append(indexSeparator).append(random[r][c]);
				}
				else {
					sampleMatrix[indexRow][c] = 0;
				}
			}
			sampleMatrix[indexRow++][ncols - 1] = 1;

			row2.append("-1");
			for(int c = 0; c < ncols - 1; c++) {
				if(random[r][c] < 0) {
					sampleMatrix[indexRow][c] = random[r][c];
					row2.append(separator).append(c + firstIndex).append(indexSeparator).append(random[r][c]);
				}
				else {
					sampleMatrix[indexRow][c] = 0;
				}
			}

			sampleMatrix[indexRow++][ncols - 1] = -1;

			sb.append(row1).append("\n");
			sb.append(row2);
			if(r != nrows - 1)
				sb.append("\n");
		}
		sampleRaw = sb.toString();

	}

	// Index start from 0
	@Test
	public void test0_1() {
		sampleRaw = "+1 2:3 4:5 6:7\n" + "-1 8:-9 10:-11";
		sampleMatrix = new double[][] {{0, 0, 3, 0, 5, 0, 7, 0, 0, 0, 0, +1}, {0, 0, 0, 0, 0, 0, 0, 0, -9, 0, -11, -1}};
		runGenerateReaderTest();
	}

	@Test
	public void test0_10() {
		sampleRaw = "-1 8:-9 10:-11\n" + "+1 2:3 4:5 6:7\n";
		sampleMatrix = new double[][] {{0, 0, 0, 0, 0, 0, 0, 0, -9, 0, -11, -1}, {0, 0, 3, 0, 5, 0, 7, 0, 0, 0, 0, +1}};
		runGenerateReaderTest();
	}

	@Test
	public void test0_2() {
		generateRandomLIBSVM(0, 10, 10, -10, 10, 1, "    ", ":");
		runGenerateReaderTest();
	}

	@Test
	public void test0_3() {
		generateRandomLIBSVM(0, 100, 10, -100, 100, 1, " ", ":");
		runGenerateReaderTest();
	}

	@Test
	public void test0_4() {
		generateRandomLIBSVM(0, 10, 10, -100, 100, 1, " ", ":");
		runGenerateReaderTest();
	}

	@Test
	public void test0_5() {
		generateRandomLIBSVM(0, 10, 10, -100, 100, 1, ",,,,", "::");
		runGenerateReaderTest();
	}

	@Test
	public void test0_6() {
		sampleRaw = "+1 2:3.0 4:5. 6:7\n" + "-1 8:9.0E0 10:11e0";
		sampleMatrix = new double[][] {{0, 0, 3, 0, 5, 0, 7, 0, 0, 0, 0, +1}, {0, 0, 0, 0, 0, 0, 0, 0, 9, 0, 11, -1}};
		runGenerateReaderTest();
	}

	@Test
	public void test0_7() {
		sampleRaw = "+10000e-4     2:3     4:5     6:7\n" + "-1     8:9     10:11";
		sampleMatrix = new double[][] {{0, 0, 3, 0, 5, 0, 7, 0, 0, 0, 0, +1}, {0, 0, 0, 0, 0, 0, 0, 0, 9, 0, 11, -1}};
		runGenerateReaderTest();
	}

	@Test
	public void test0_8() {
		sampleRaw = "+10000e-4     2:3     4:5     6:7\n" + "-0.00001e5     8:9     10:11";
		sampleMatrix = new double[][] {{0, 0, 3, 0, 5, 0, 7, 0, 0, 0, 0, +1}, {0, 0, 0, 0, 0, 0, 0, 0, 9, 0, 11, -1}};
		runGenerateReaderTest();
	}

	// Index start from 1
	@Test
	public void test2() {
		sampleRaw = "+1 2:3 4:5 6:7\n" + "-1 8:9 10:11";
		sampleMatrix = new double[][] {{0, 3, 0, 5, 0, 7, 0, 0, 0, 0, +1}, {0, 0, 0, 0, 0, 0, 0, 9, 0, 11, -1}};
		runGenerateReaderTest();
	}

	@Test
	public void test1_2() {
		generateRandomLIBSVM(1, 10, 10, -10, 10, 1, " ", ":");
		runGenerateReaderTest();
	}

	@Test
	public void test1_3() {
		generateRandomLIBSVM(1, 10, 10, -100, 100, 1, " ", ":");
		runGenerateReaderTest();
	}

	@Test
	public void test1_4() {
		generateRandomLIBSVM(0, 10, 12, -100, 100, 1, ",,,,,,", ":::::");
		runGenerateReaderTest();
	}

	@Test
	public void test1_5() {
		generateRandomLIBSVM(1, 100, 50, -100, 100, 1, ",,,,", "::");
		runGenerateReaderTest();
	}
}
