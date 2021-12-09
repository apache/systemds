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

public class FrameGenerateReaderLibSVMTest extends GenerateReaderFrameTest {

	private final static String TEST_NAME = "FrameGenerateReaderLibSVMTest";

	@Override
	protected String getTestName() {
		return TEST_NAME;
	}

	private void extractSampleRawLibSVM(int firstIndex, String separator, String indexSeparator) {

		int nrows = data.length;
		int ncols = data[0].length;
		int mid = ncols/2;
		String[][] dataLibSVM = new String[2 * nrows][ncols];
		StringBuilder sb = new StringBuilder();
		int indexRow = 0;
		for(int r = 0; r < nrows; r++) {
			StringBuilder row1 = new StringBuilder();
			StringBuilder row2 = new StringBuilder();
			row1.append("+1");
			for(int c = 0; c < ncols - 1; c++) {
				if(mid > c) {
					if(data[r][c] != null) {
						dataLibSVM[indexRow][c] = data[r][c];
						row1.append(separator).append(c + firstIndex).append(indexSeparator).append(data[r][c]);
					}
					else
						dataLibSVM[indexRow][c] = defaultValue(schema[c]);
				}
				else
					dataLibSVM[indexRow][c] = defaultValue(schema[c]);

			}
			dataLibSVM[indexRow++][ncols-1] = "+1";

			row2.append("-1");
			for(int c = 0; c < ncols - 1; c++) {
				if(mid <= c) {
					if(data[r][c] != null) {
						dataLibSVM[indexRow][c] = data[r][c];
						row2.append(separator).append(c + firstIndex).append(indexSeparator).append(data[r][c]);
					}
					else
						dataLibSVM[indexRow][c] = defaultValue(schema[c]);
				}
				else
					dataLibSVM[indexRow][c] = defaultValue(schema[c]);
			}
			dataLibSVM[indexRow++][ncols-1] = "-1";
			sb.append(row1).append("\n");
			sb.append(row2);
			if(r != nrows - 1)
				sb.append("\n");
		}
		sampleRaw = sb.toString();
		data = dataLibSVM;
	}

	@Test
	public void test1() {
		String[] naStrings = {"NULL", "inf", "NaN"};
		String separator = ",";
		String indexSeparator = ":";
		generateRandomData(10, 10, -100, 100, 1, naStrings);
		extractSampleRawLibSVM(0,separator, indexSeparator);
		runGenerateReaderTest();
	}

	@Test
	public void test2() {
		String[] naStrings = {"NULL", "inf", "NaN"};
		String separator = ",";
		String indexSeparator = ":";
		generateRandomData(100, 200, -100, 100, 1, naStrings);
		extractSampleRawLibSVM(0,separator, indexSeparator);
		runGenerateReaderTest();
	}

	@Test
	public void test3() {
		String[] naStrings = {"NULL", "inf", "NaN"};
		String separator = ",";
		String indexSeparator = ":";
		generateRandomData(1000, 200, -100, 100, 1, naStrings);
		extractSampleRawLibSVM(0,separator, indexSeparator);
		runGenerateReaderTest();
	}

	@Test
	public void test4() {
		String[] naStrings = {"NULL", "inf", "NaN"};
		String separator = ",,,,,,";
		String indexSeparator = ":";
		generateRandomData(20, 20, -100, 100, 0.6, naStrings);
		extractSampleRawLibSVM(0,separator, indexSeparator);
		runGenerateReaderTest();
	}

	@Test
	public void test5() {
		String[] naStrings = {"NULL", "inf", "NaN"};
		String separator = ",,,,,";
		String indexSeparator = ":";
		generateRandomData(100, 50, -100, 100, 0.5, naStrings);
		extractSampleRawLibSVM(0,separator, indexSeparator);
		runGenerateReaderTest();
	}

	@Test
	public void test6() {
		String[] naStrings = {"NULL", "inf", "NaN"};
		String separator = ",,,,,";
		String indexSeparator = ":";
		generateRandomData(10, 1000, -100, 100, 0.7, naStrings);
		extractSampleRawLibSVM(1,separator, indexSeparator);
		runGenerateReaderTest();
	}
}
