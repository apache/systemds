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

package org.apache.sysds.test.functions.io.generateReader;

import org.apache.sysds.runtime.iogen.*;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;
import org.apache.sysds.runtime.util.DataConverter;
import org.apache.sysds.test.AutomatedTestBase;
import org.apache.sysds.test.TestConfiguration;
import org.junit.Test;

import java.io.IOException;

public class GenerateReaderTest2 extends AutomatedTestBase {

	private final static String TEST_NAME = "GenerateReaderTest2";
	private final static String TEST_DIR = "functions/io/GenerateReaderTest/";
	private final static String TEST_CLASS_DIR = TEST_DIR + GenerateReaderTest2.class.getSimpleName() + "/";

	private final static double eps = 1e-9;

	private String stream;
	private MatrixBlock sample;

	@Override public void setUp() {
		addTestConfiguration(TEST_NAME, new TestConfiguration(TEST_CLASS_DIR, TEST_NAME, new String[] {"Rout"}));
	}

	@Test public void tmyTest() throws IOException {
		String sv1 = "19.0";
		String sv2 = "";
		if(sv1.length() - 2 > 0 && sv1.substring(sv1.length() - 2).equals(".0")) { // check for: ".0"
			sv2 = sv1.substring(0, sv1.length() - 2);
		}
		String s = "19.0,19,20,19.0,19.0E";
		String s1 = s.replaceAll(sv1, "");
		String s2 = s1.replaceAll(sv2, "");

		System.out.println(sv1.substring(1));
		//System.out.println(s1);
		//System.out.println(s2);
	}

	//1. Generate CSV Test Data
	//1.a. The Data include Header and Unique Values
	@Test public void testCSV1_CP_CSV_Data_With_Header() throws IOException {
		stream = "a,b,c,d,e,f\n" + "1,2,3,4,5,6\n" + "7,8,9,10,11,12\n" + "2,3,1,5,4,6\n" + "1,5,3,4,2,6\n" + "8,9,10,11,12,7\n" + "1,2,3,4,5,6\n" + "7,8,9,10,11,12";

		double[][] sample = {{1, 2, 3, 4, 5, 6}, {7, 8, 9, 10, 11, 12}};
		//GenerateReader2.generateReader(stream, DataConverter.convertToMatrixBlock(sample));
	}

	//1. Generate CSV Test Data
	//1.a. The Data include Header and Unique Values
	@Test public void testLIBSVM1_CP_Data() throws Exception {
		stream = "1,2,3,4,5\n" + "6,7,8,9,10\n";

		double[][] sample = {{1, 2, 3, 4, 5}, {6, 7, 8, 9, 10}};
		//GenerateReader3.generateReader(stream, DataConverter.convertToMatrixBlock(sample));
	}

	@Test public void testCSV1_CP_Data() throws Exception {
		double[][] sample = getRandomMatrix(10, 10, -100, 100, 1, 714);
		StringBuilder raw = new StringBuilder();
		for(int r=0;r<sample.length;r++) {
			String row="";
			for(int c = 0; c < sample[0].length; c++)
				row +=sample[r][c] + ",";
			row = row.substring(0,row.length()-1);
			raw.append(row);
			raw.append("\n");
		}
		//System.out.println(raw.toString());
		//GenerateReader3.generateReader(raw.toString(), DataConverter.convertToMatrixBlock(sample));
	}

	@Test public void testCSV2_CP_Data() throws Exception {
		stream = "1.1,2.1,3.1,4.1,5.1\n" + "6.1,7.1,8.1,9.1,10.1\n";

		double[][] sample = {{1.1, 2.1, 3.1, 4.1, 5.1}, {6.1, 7.1, 8.1, 9.1, 10.1}};
		//GenerateReader3.generateReader(stream, DataConverter.convertToMatrixBlock(sample));
	}

	@Test public void testCSV3_CP_Data() throws Exception {
		stream = "1.1456,2.1758,3.1962,4.10002,5.122222333\n" + "6.15,7.18,8.19,9.120,10.1333\n";

		double[][] sample = {{1.1456,2.1758,3.1962,4.10002,5.122222333}, {6.15,7.18,8.19,9.120,10.1333}};
		//GenerateReader3.generateReader(stream, DataConverter.convertToMatrixBlock(sample));
	}

	@Test public void testCSV4_CP_Data() throws Exception {
		stream = "1.1456,2.1758,3.1962,4.10002,5.122222333\n" + "6.15,7.18,8.19,9.120,10.1333\n";

		double[][] sample = {{1.1456,2.1758,3.1962,4.10002,5.122222333}, {6.15,7.18,8.19,9.120,10.1333}};
		//GenerateReader3.generateReader(stream, DataConverter.convertToMatrixBlock(sample));
	}

	@Test public void testCSV5_CP_Data() throws Exception {
		stream = "1.,2.0,0.3,.4\n" + "5.00,6.,7.8,.8\n";

		double[][] sample = {{1,2,0.3,0.4}, {5,6,7.8,0.8}};
		//GenerateReader3.generateReader(stream, DataConverter.convertToMatrixBlock(sample));
	}

	@Test public void testCSV6_CP_Data() throws Exception {
		stream = "1.,2E00,1.1E-02,1.123E2\n" + "0.00005E5,.00006E5,0.00007E+5,.00008E05\n";

		double[][] sample = {{1.,2E0,1.1E-2,1.123E2}, {0.00005E5,.00006E5,0.00007E+5,.00008E05}};
		//GenerateReader3.generateReader(stream, DataConverter.convertToMatrixBlock(sample));
	}

	@Test public void testCSV7_CP_Data() throws Exception {
		stream = "1,2,3,,\n" +
				 ",,,4,5\n" +
				 "1,2,,,\n" +
				 ",,,null,5.000\n" +
				 "1,,3,,\n" +
				 ",,3,,";

		String stringValue="0.00123450000";
		//System.out.println(s.substring(4,8));

		int length = stringValue.length();
		int firstNZ=-1;
		int lastNZ=-1;
		for(int i=0; i<length; i++){
			char fChar = stringValue.charAt(i);
			char lChar = stringValue.charAt(length-i-1);
			System.out.println(">> "+lChar);
			if(Character.isDigit(fChar) && fChar!='0' && firstNZ==-1)
				firstNZ = i;

			if(Character.isDigit(lChar) && lChar!='0' && lastNZ==-1)
				lastNZ = length-i;

			if(firstNZ>0 && lastNZ>0)
				break;
		}
		System.out.println(firstNZ);
		System.out.println(lastNZ);
		System.out.println(stringValue.substring(firstNZ,lastNZ));

//		double[][] sample = {{1,2,3,0,0}, {0,0,0,4,5},{1,2,0,0,0},{0,0,0,0,5},{1,0,3,0,0},{0,0,3,0,0}};
//		GenerateReader3.generateReader(stream, DataConverter.convertToMatrixBlock(sample));
	}

	@Test public void testCSV8_CP_Data() throws Exception {
		stream = "1,,,,\n" + "2,,,,\n"+ "3,,,,\n"+ "4,,,,\n";

		double[][] sample = {{1,0,0,0,0}, {2,0,0,0,0}, {3,0,0,0,0}, {4,0,0,0,0}};
		//GenerateReader3.generateReader(stream, DataConverter.convertToMatrixBlock(sample));
	}

	@Test public void testCSV9_CP_Data() throws Exception {
		stream = "1,,,,\n" + "2,,,,\n"+ "3,,,,\n"+ "4,,,6,5\n";

		double[][] sample = {{1,0,0,5,1}, {2,0,0,0,0}, {3,0,0,0,0}, {4,0,0,6,5}};
		MatrixBlock matrixBlock = DataConverter.convertToMatrixBlock(sample);
		System.out.println(matrixBlock.getNonZeros());
		//GenerateReader3.generateReader(stream, DataConverter.convertToMatrixBlock(sample));
	}

	@Test public void testLibSVM_CP_Data() throws Exception {
		stream = "1:5\n" + "1:6\n"+ "1:7\n"+ "1:8\n";

		double[][] sample = {{5}, {6}, {7}, {8}};
		//GenerateReader3.generateReader(stream, DataConverter.convertToMatrixBlock(sample));
	}

	@Test public void testCSVNTF_CP_Data() throws Exception {
		double[][] sample = getRandomMatrix(10, 10, -100, 100, 1, 714);
		StringBuilder raw = new StringBuilder();
		for(int r=0;r<sample.length;r++) {
			String row="";
			for(int c = 0; c < sample[0].length; c++)
				row +=sample[r][c] + ",";
			row = row.substring(0,row.length()-1);
			raw.append(row);
			raw.append("\n");
		}
		//System.out.println(raw.toString());
		GenerateReader.generateReader(raw.toString(), DataConverter.convertToMatrixBlock(sample));
	}

}
