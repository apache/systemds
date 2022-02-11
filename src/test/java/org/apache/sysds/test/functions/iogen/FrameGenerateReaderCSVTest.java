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

public class FrameGenerateReaderCSVTest extends GenerateReaderFrameTest {

	private final static String TEST_NAME = "FrameGenerateReaderCSVTest";

	@Override
	protected String getTestName() {
		return TEST_NAME;
	}

	private void extractSampleRawCSV(String separator) {
		int nrows = data.length;
		int ncols = data[0].length;
		StringBuilder sb = new StringBuilder();
		for(int r = 0; r < nrows; r++) {
			for(int c = 0; c < ncols; c++) {
				sb.append(data[r][c]);
				if(c != ncols - 1)
					sb.append(separator);
			}
			if(r != nrows - 1)
				sb.append("\n");
		}
		sampleRaw = sb.toString();
	}

	@Test
	public void test1() {
		String[] naStrings = {"NULL", "inf", "NaN"};
		String separator = ",";
		generateRandomData(10, 10, -100, 100, 1, naStrings);
		extractSampleRawCSV(separator);
		runGenerateReaderTest();
	}

	@Test
	public void test2() {
		String[] naStrings = {"NULL", "inf", "NaN"};
		String separator = ",";
		generateRandomData(10, 10, -10, 10, 1, naStrings);
		extractSampleRawCSV(separator);
		runGenerateReaderTest();
	}

	@Test
	public void test3() {
		String[] naStrings = {"NULL", "inf", "NaN"};
		String separator = "****";
		generateRandomData(100, 500, -10, 10, 1, naStrings);
		extractSampleRawCSV(separator);
		runGenerateReaderTest();
	}

	@Test
	public void test4() {
		String[] naStrings = {"NULL", "inf", "NaN"};
		String separator = ",";
		generateRandomData(10, 10, -10, 10, 0.7, naStrings);
		extractSampleRawCSV(separator);
		runGenerateReaderTest();
	}

	@Test
	public void test5() {
		String[] naStrings = {"NULL", "inf", "NaN"};
		String separator = ",,,,";
		generateRandomData(10, 10, -10, 10, 0.5, naStrings);
		extractSampleRawCSV(separator);
		runGenerateReaderTest();
	}

	@Test
	public void test6() {
		String[] naStrings = {"NULL", "inf", "NaN"};
		String separator = "**";
		generateRandomData(1000, 100, -10, 10, 0.4, naStrings);
		extractSampleRawCSV(separator);
		runGenerateReaderTest();
	}

	@Test
	public void test7() {
		String[] naStrings = {"NULL", "inf", "NaN"};
		String separator = "**";
		generateRandomData(1000, 100, -10, 10, 0.8, naStrings);
		extractSampleRawCSV(separator);
		runGenerateReaderTest();
	}

	@Test
	public void test8() {
		String[] naStrings = {"NULL", "inf", "NaN"};
		String separator = "**";
		generateRandomData(10000, 100, -10, 10, 0.5, naStrings);
		extractSampleRawCSV(separator);
		runGenerateReaderTest();
	}
}
