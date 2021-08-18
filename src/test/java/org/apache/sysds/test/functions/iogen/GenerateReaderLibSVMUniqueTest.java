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

import com.google.gson.Gson;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;
import org.apache.sysds.test.AutomatedTestBase;
import org.apache.sysds.test.TestConfiguration;
import org.junit.Test;

import java.io.IOException;

public class GenerateReaderLibSVMUniqueTest extends GenerateReaderTest {

	private final static String TEST_NAME = "GenerateReaderLibSVMUniqueTest";
	private final static String TEST_DIR = "functions/io/GenerateReaderLibSVMUniqueTest/";
	private final static String TEST_CLASS_DIR = TEST_DIR + GenerateReaderLibSVMUniqueTest.class.getSimpleName() + "/";



	@Override public void setUp() {
		addTestConfiguration(TEST_NAME, new TestConfiguration(TEST_CLASS_DIR, TEST_NAME, new String[] {"Rout"}));
	}

	// Index start from 0
	@Test
	public void test0_1() throws Exception {

		sampleRaw = "+1 2:3 4:5 6:7\n"+
					"-1 8:9 10:11";

		sampleMatrix = new double[][] {{0,0,3,0,5,0,7,0,0,0,0,+1}, {0,0,0,0,0,0,0,0,9,0,11,-1}};
		runGenerateReaderTest();
	}
	@Test
	public void test0_2() throws Exception {
		int nrows = 100;
		int ncols = 51;
		double[][] random = getRandomMatrix(nrows, ncols, -100, 100, 1, 714);
		sampleMatrix = new double[2 * nrows][ncols];

		StringBuilder sb = new StringBuilder();
		int indexRow = 0;
		for(int r = 0; r < nrows; r ++) {
			StringBuilder row1 = new StringBuilder();
			StringBuilder row2 = new StringBuilder();
			row1.append("+1");

			for(int c = 0; c < ncols-1; c++) {
				if(random[r][c] > 0) {
					sampleMatrix[indexRow][c] = random[r][c];
					row1.append(" "+c + ":" + random[r][c]);
				}
				else {
					sampleMatrix[indexRow][c] = 0;
				}
			}
			sampleMatrix[indexRow++][ncols - 1] = 1;

			row2.append("-1");
			for(int c = 0; c < ncols-1; c++) {
				if(random[r][c] < 0) {
					sampleMatrix[indexRow][c] = random[r][c];
					row2.append(" "+c + ":" + random[r][c]);
				}
				else {
					sampleMatrix[indexRow][c] = 0;
				}
			}

			sampleMatrix[indexRow++][ncols - 1] = -1;

			sb.append(row1 + "\n");
			sb.append(row2 + "\n");
		}

		sampleRaw = sb.toString();
		System.out.println(sampleRaw);

		//Gson gson= new Gson();
		//System.out.println(gson.toJson(sampleMatrix));
		runGenerateReaderTest();
	}

	// Index start from 1
	@Test
	public void test0_3() throws Exception {

		sampleRaw = "0:-57.26153462146186 4:-90.11715823359168 5:-23.443753578851386 6:-16.661596780916284 -1";

		sampleMatrix = new double[][] {{-57.26153462146186,0,0,0,-90.11715823359168,-23.443753578851386,-16.661596780916284,-1}};
		runGenerateReaderTest();
	}



	// Index start from 1
	@Test
	public void test2() throws Exception {

		sampleRaw = "+1 2:3 4:5 6:7\n"+
					"-1 8:9 10:11";

		sampleMatrix = new double[][] {{0,3,0,5,0,7,0,0,0,0,+1}, {0,0,0,0,0,0,0,9,0,11,-1}};
		runGenerateReaderTest();
	}

	//0:-57.26153462146186 4:-90.11715823359168 5:-23.443753578851386 6:-16.661596780916284 -1

}
