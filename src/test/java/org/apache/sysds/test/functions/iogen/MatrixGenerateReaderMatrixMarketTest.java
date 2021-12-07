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

public class MatrixGenerateReaderMatrixMarketTest extends GenerateReaderMatrixTest {

	private final static String TEST_NAME = "MatrixGenerateReaderMatrixMarketTest";

	@Override
	protected String getTestName() {
		return TEST_NAME;
	}

	private void generateRandomMM(int firstIndex, int nrows, int ncols, double min, double max, double sparsity,
		String separator) {

		sampleMatrix = getRandomMatrix(nrows, ncols, min, max, sparsity, 714);
		StringBuilder sb = new StringBuilder();
		for(int r = 0; r < nrows; r++) {
			for(int c = 0; c < ncols; c++) {
				if(sampleMatrix[r][c] != 0) {
					String rs = (r + firstIndex) + separator + (c + firstIndex) + separator + sampleMatrix[r][c];
					sb.append(rs);
					if(r != nrows - 1 || c != ncols - 1)
						sb.append("\n");
				}
			}
		}
		sampleRaw = sb.toString();
	}

	private void generateRandomSymmetricMM(int firstIndex, int size, double min, double max, double sparsity,
		String separator, boolean isUpperTriangular, boolean isSkew) {

		generateRandomSymmetric(size, min, max, sparsity, isSkew);

		int start, end;
		StringBuilder sb = new StringBuilder();

		for(int r = 0; r < size; r++) {
			if(isUpperTriangular) {
				start = r;
				end = size;
			}
			else {
				start = 0;
				end = r + 1;
			}
			for(int c = start; c < end; c++) {
				if(sampleMatrix[r][c] != 0) {
					String rs = (r + firstIndex) + separator + (c + firstIndex) + separator + sampleMatrix[r][c];
					sb.append(rs);
					if(r != size - 1 || c != size - 1)
						sb.append("\n");
				}
			}
		}
		sampleRaw = sb.toString();
	}

	// Index from 0
	@Test
	public void test0_1() {
		sampleRaw = "0,1,1\n" + "0,2,4\n" + "1,2,2\n" + "2,3,3";
		sampleMatrix = new double[][] {{0, 1, 4, 0}, {0, 0, 2, 0}, {0, 0, 0, 3}};
		runGenerateReaderTest();
	}

	@Test
	public void test0_2() {
		sampleRaw = "0,0,-1\n" + "0,1,1\n" + "0,2,2\n" + "0,3,3\n" + "1,0,4\n" + "1,1,5\n" + "1,2,6\n" + "1,3,7";
		sampleMatrix = new double[][] {{-1, 1, 2, 3}, {4, 5, 6, 7}};
		runGenerateReaderTest();
	}

	@Test
	public void test0_3() {
		sampleRaw = "0,0,-1\n" + "0,1,1\n" + "0,2,2.0\n" + "0,3,3.\n" + "1,0,4e0\n" + "1,1,5\n" + "1,2,6\n" + "1,3,7";
		sampleMatrix = new double[][] {{-1, 1, 2, 3}, {4, 5, 6, 7}};
		runGenerateReaderTest();
	}

	@Test
	public void test0_4() {
		sampleRaw = "0,0,-1\n" + "0,1,0.00001e5\n" + "0,2,2.\n" + "0,3,3\n" + "1,0,4e0\n" + "1,1,5\n" + "1,2,6\n" + "1,3,7";
		sampleMatrix = new double[][] {{-1, 1, 2, 3}, {4, 5, 6, 7}};
		runGenerateReaderTest();
	}

	@Test
	public void test0_5() {
		generateRandomMM(0, 5, 10, -100, 100, 1, ",");
		runGenerateReaderTest();
	}

	@Test public void test0_6() {
		generateRandomMM(0, 10, 10, -100, 100, 1, ",");
		runGenerateReaderTest();
	}

	@Test
	public void test0_7() {
		generateRandomMM(0, 10, 10, -100, 100, 1, "   ,");
		runGenerateReaderTest();
	}

	@Test
	public void test0_8() {
		generateRandomMM(0, 10, 10, -100, 100, 0.5, ",");
		runGenerateReaderTest();
	}

	// Index from 1
	@Test
	public void test1() {
		sampleRaw = "1,1,1\n" + "1,2,4\n" + "2,2,2\n" + "3,3,3";
		sampleMatrix = new double[][] {{1, 4, 0}, {0, 2, 0}, {0, 0, 3}};
		runGenerateReaderTest();
	}

	@Test
	public void test1_2() {
		generateRandomMM(1, 5, 100, -100, 100, 1, ",,,,,");
		runGenerateReaderTest();
	}

	// Symmetric Tests:
	// Symmetric Index from 0
	@Test
	public void SymmetricTest0_1() {
		sampleRaw = "0,0,1\n" + "1,0,2\n" + "1,1,3\n" + "2,0,4\n" + "2,1,5\n" + "2,2,6\n" + "3,0,7\n" + "3,1,8\n" + "3,2,9\n" + "3,3,10\n";
		sampleMatrix = new double[][] {{1, 0, 0, 0}, {2, 3, 0, 0}, {4, 5, 6, 0}, {7, 8, 9, 10}};
		runGenerateReaderTest();
	}

	@Test
	public void SymmetricTest0_2() {
		sampleRaw = "0,0,1\n" + "0,1,2\n" + "0,2,3\n" + "0,0,1\n" + "1,0,2\n" + "1,1,3\n" + "2,0,4\n" + "2,1,5\n" + "2,2,6\n" + "3,0,7\n" + "3,1,8\n" + "3,2,9\n" + "3,3,10\n";
		sampleMatrix = new double[][] {{1, 0, 0, 0}, {2, 3, 0, 0}, {4, 5, 6, 0}, {7, 8, 9, 10}};
		runGenerateReaderTest();
	}

	@Test
	public void SymmetricTest0_3() {
		generateRandomSymmetricMM(0, 5, -5, 5, 1, ",", true, false);
		runGenerateReaderTest();
	}

	@Test
	public void SymmetricTest0_4() {
		generateRandomSymmetricMM(0, 50, -100, 100, 1, "  ", true, false);
		runGenerateReaderTest();
	}

	@Test
	public void SymmetricTest0_5() {
		generateRandomSymmetricMM(0, 5, -5, 5, 1, ",", false, false);
		runGenerateReaderTest();
	}

	@Test
	public void SymmetricTest0_6() {
		generateRandomSymmetricMM(0, 50, -100, 100, 1, "  ", false, false);
		runGenerateReaderTest();
	}

	// Symmetric Index from 1
	@Test
	public void SymmetricTest1_1() {
		generateRandomSymmetricMM(1, 5, -5, 5, 1, ",", true, false);
		runGenerateReaderTest();
	}

	@Test
	public void SymmetricTest1_2() {
		generateRandomSymmetricMM(1, 50, -100, 100, 1, "  ", true, false);
		runGenerateReaderTest();
	}

	@Test
	public void SymmetricTest1_3() {
		generateRandomSymmetricMM(1, 50, -5, 5, 1, ",", false, false);
		runGenerateReaderTest();
	}

	@Test
	public void SymmetricTest1_4() {
		generateRandomSymmetricMM(1, 70, -100, 100, 1, "  ", false, false);
		runGenerateReaderTest();
	}

	// Skew-Symmetric Tests:
	// Skew-Symmetric Index from 0
	@Test
	public void SkewSymmetricTest0_1() {
		generateRandomSymmetricMM(0, 5, -100, 100, 1, ",", false, true);
		runGenerateReaderTest();
	}

	@Test
	public void SkewSymmetricTest0_2() {
		generateRandomSymmetricMM(0, 5, -100, 100, 1, "   ", true, true);
		runGenerateReaderTest();
	}

	// Skew-Symmetric Index from 1
	@Test
	public void SkewSymmetricTest0_3() {
		generateRandomSymmetricMM(1, 5, -100, 100, 1, ",", false, true);
		runGenerateReaderTest();
	}

	@Test
	public void SkewSymmetricTest0_4() {
		generateRandomSymmetricMM(1, 5, -100, 100, 1, "   ", true, true);
		runGenerateReaderTest();
	}

}
