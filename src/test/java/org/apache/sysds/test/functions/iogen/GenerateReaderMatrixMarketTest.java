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

import org.apache.sysds.test.TestConfiguration;
import org.junit.Test;

import java.util.Random;

public class GenerateReaderMatrixMarketTest extends GenerateReaderTest {

	private final static String TEST_NAME = "GenerateReaderMatrixMarketTest";
	private final static String TEST_DIR = "functions/io/GenerateReaderMatrixMarketTest/";
	private final static String TEST_CLASS_DIR = TEST_DIR + GenerateReaderMatrixMarketTest.class.getSimpleName() + "/";

	private final static double eps = 1e-9;

	@Override public void setUp() {
		addTestConfiguration(TEST_NAME, new TestConfiguration(TEST_CLASS_DIR, TEST_NAME, new String[] {"Rout"}));
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
		System.out.println(sampleRaw);

	}

	// Index from 0
	@Test public void test0_1() throws Exception {
		sampleRaw = "1,1,1\n" + "1,2,4\n" + "2,2,2\n" + "3,3,3";
		sampleMatrix = new double[][] {{0, 1, 4, 0}, {0, 0, 2, 0}, {0, 0, 0, 3}};
		runGenerateReaderTest();
	}

	// Index from 0
	@Test public void test0_2() throws Exception {
		sampleRaw = "0,0,-1\n" + "0,1,1\n" + "0,2,2\n" + "0,3,3\n" + "1,0,4\n" + "1,1,5\n" + "1,2,6\n" + "1,3,7";
		sampleMatrix = new double[][] {{-1, 1, 2, 3}, {4,5,6,7}};
		runGenerateReaderTest();
	}

	// Index from 1
	@Test public void test1() throws Exception {
		sampleRaw = "1,1,1\n" + "1,2,4\n" + "2,2,2\n" + "3,3,3";
		sampleMatrix = new double[][] {{1, 4, 0}, {0, 2, 0}, {0, 0, 3}};
		runGenerateReaderTest();
	}
}