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

package org.apache.sysds.test.functions.iogen.Identify;

import org.apache.sysds.test.functions.iogen.GenerateReaderMatrixTest;
import org.junit.Test;

import java.util.Random;

public class MatrixGRRowColIdentifyTest extends GenerateReaderMatrixTest {

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
		sampleMatrix = new double[][] {{1, 2}, {6, 7}, {11, 12}};
		runGenerateReaderTest();
	}

	@Test
	public void test2() {
		sampleRaw = "1,a2,a3,a4,a5\n" + "6,a7,a8,a9,a10\n" + "11,a12,a13,a14,a15";
		sampleMatrix = new double[][] {{1, 5}, {6, 10}, {11, 15}};
		runGenerateReaderTest();
	}
	@Test
	public void test3() {
		sampleRaw = "1,,2,,3,,4,,5\n" + "6,,7,,8,,9,,10\n" + "11,,12,,13,,14,,15";
		sampleMatrix = new double[][] {{1, 5}, {6, 10}, {11, 15}};
		runGenerateReaderTest();
	}


}
