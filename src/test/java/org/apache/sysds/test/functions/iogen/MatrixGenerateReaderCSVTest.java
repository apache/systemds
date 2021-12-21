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

import java.util.Random;

public class MatrixGenerateReaderCSVTest extends GenerateReaderMatrixTest {

	private final static String TEST_NAME = "MatrixGenerateReaderCSVTest";

	@Override
	protected String getTestName() {
		return TEST_NAME;
	}

	private void generateRandomCSV(int nrows, int ncols, double min, double max, double sparsity, String separator,
		String[] naString) {

		sampleMatrix = getRandomMatrix(nrows, ncols, min, max, sparsity, 714);
		StringBuilder sb = new StringBuilder();

		for(int r = 0; r < nrows; r++) {
			StringBuilder row = new StringBuilder();
			for(int c = 0; c < ncols; c++) {
				if(sampleMatrix[r][c] != 0) {
					row.append(sampleMatrix[r][c]).append(separator);
				}
				else {
					Random rn = new Random();
					int rni = rn.nextInt(naString.length);
					row.append(naString[rni]).append(separator);
				}
			}

			sb.append(row.substring(0, row.length() - separator.length()));
			if(r != nrows - 1)
				sb.append("\n");
		}
		sampleRaw = sb.toString();
	}

	@Test
	public void test1() {
		sampleRaw = "1,2,3,4,5\n" + "6,7,8,9,10\n" + "11,12,13,14,15";
		sampleMatrix = new double[][] {{1, 2, 3, 4, 5}, {6, 7, 8, 9, 10}, {11, 12, 13, 14, 15}};
		runGenerateReaderTest();
	}

	@Test
	public void test2() {
		String[] naString = {"NaN"};
		generateRandomCSV(5, 5, -10, 10, 1, ",", naString);
		runGenerateReaderTest();
	}

	@Test
	public void test3() {
		String[] naString = {"NaN"};
		generateRandomCSV(5, 5, -10, 10, 1, ",,,", naString);
		runGenerateReaderTest();
	}

	@Test
	public void test4() {
		String[] naString = {"Nan", "NAN", "", "inf", "null", "NULL"};
		generateRandomCSV(50, 50, -10, 10, 0.5, ",,", naString);
		runGenerateReaderTest();
	}

	@Test
	public void test5() {
		sampleRaw = "1.0,2.0,3.0,4.0,5.0\n" + "6.,7.,8.,9.,10.\n" + "11,12,13,14,15";
		sampleMatrix = new double[][] {{1, 2, 3, 4, 5}, {6, 7, 8, 9, 10}, {11, 12, 13, 14, 15}};
		runGenerateReaderTest();
	}

	@Test
	public void test6() {
		sampleRaw = "1.0,2.0,3.0,4.0,5.0\n" + "6.,7.,8.,9.,10.\n" + "11E0,12E0,13,14E0,15";
		sampleMatrix = new double[][] {{1, 2, 3, 4, 5}, {6, 7, 8, 9, 10}, {11, 12, 13, 14, 15}};
		runGenerateReaderTest();
	}

	@Test
	public void test7() {
		sampleRaw = "1.0,2.0,3.0,4.0,5.0\n" + "6.,7.,8.,9.,10.\n" + "1.1E1,1.2E1,13,1.4E1,15";
		sampleMatrix = new double[][] {{1, 2, 3, 4, 5}, {6, 7, 8, 9, 10}, {11, 12, 13, 14, 15}};
		runGenerateReaderTest();
	}

	@Test public void test8() {
		sampleRaw = "1.0,2.0,3.0,4.0,5.0\n" + "60.0E-1,7.,80.0E-1,9.,100.0E-1\n" + "1.1E1,1.2E1,13,1.4E1,15";
		sampleMatrix = new double[][] {{1, 2, 3, 4, 5}, {6, 7, 8, 9, 10}, {11, 12, 13, 14, 15}};
		runGenerateReaderTest();
	}

	@Test
	public void test9() {
		sampleRaw = ".1E1,.2E1,3.0,4.0,0.5E1\n" + "60.0E-1,7.,80.0E-1,9.,100.0E-1\n" + "1.1E1,1.2E1,13,1.4E1,15";
		sampleMatrix = new double[][] {{1, 2, 3, 4, 5}, {6, 7, 8, 9, 10}, {11, 12, 13, 14, 15}};
		runGenerateReaderTest();
	}

	@Test
	public void test10() {
		sampleRaw = "0.000001e6,2,3,4,5\n" + "6,7,8,9,10\n" + "11,12,13,14,15";
		sampleMatrix = new double[][] {{1, 2, 3, 4, 5}, {6, 7, 8, 9, 10}, {11, 12, 13, 14, 15}};
		runGenerateReaderTest();
	}

	@Test
	public void test11() {
		sampleRaw = "1,2,3,4,5,NAN\n" + "6,7,8,9,10,NAN\n" + "11,12,13,14,15,NAN";
		sampleMatrix = new double[][] {{1, 2, 3, 4, 5, 0}, {6, 7, 8, 9, 10, 0}, {11, 12, 13, 14, 15, 0}};
		runGenerateReaderTest();
	}

	@Test
	public void test12() {
		sampleRaw = "1,2,3,4,5,NAN,,\n" + "6,7,8,9,10,NAN,,\n" + "11,12,13,14,15,NAN,,";
		sampleMatrix = new double[][] {{1, 2, 3, 4, 5, 0, 0, 0}, {6, 7, 8, 9, 10, 0, 0, 0},
			{11, 12, 13, 14, 15, 0, 0, 0}};
		runGenerateReaderTest();
	}

	@Test
	public void test13() {
		String[] naString = {"Nan", "NAN", "", "inf", "null", "NULL"};
		generateRandomCSV(1000, 500, -10, 10, 0.5, ",,", naString);
		runGenerateReaderTest();
	}
}
