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

import org.apache.sysds.runtime.matrix.data.MatrixBlock;
import org.apache.sysds.test.AutomatedTestBase;
import org.apache.sysds.test.TestConfiguration;
import org.junit.Test;

public class GenerateReaderMatrixMarketTest1 extends AutomatedTestBase {

	private final static String TEST_NAME = "GenerateReaderCSVTest1";
	private final static String TEST_DIR = "functions/io/GenerateReaderCSVTest1/";
	private final static String TEST_CLASS_DIR = TEST_DIR + GenerateReaderMatrixMarketTest1.class.getSimpleName() + "/";

	private final static double eps = 1e-9;

	private String raw;
	private MatrixBlock sample;

	@Override public void setUp() {
		addTestConfiguration(TEST_NAME, new TestConfiguration(TEST_CLASS_DIR, TEST_NAME, new String[] {"Rout"}));
	}

	// Index start from 1
	//Type 0: Row, Col, Value
	@Test public void testMM1() throws Exception {
		raw = "1,1,4\n" +
			"1,2,5\n" +
			"1,3,6\n" +
			"2,1,7\n" +
			"2,2,8\n" +
			"2,3,9";

		double[][] sample = {{4, 5, 6}, {7, 8, 9}};
		//GenerateReader3.generateReader(raw, DataConverter.convertToMatrixBlock(sample));
	}
	//Type 0: Row, Col, Value
	@Test public void testMM11() throws Exception {
		raw = "0,0,4\n" +
			"0,1,5\n" +
			"0,2,6\n" +
			"1,0,7\n" +
			"1,1,8\n" +
			"1,2,9";

		double[][] sample = {{4, 5, 6}, {7, 8, 9}};
		//GenerateReader3.generateReader(raw, DataConverter.convertToMatrixBlock(sample));
	}

	//Type 1: Col, Row, Value
	@Test public void testMM2() throws Exception {
		raw = "1,1,4\n" +
			"2,1,5\n" +
			"3,1,6\n" +
			"1,2,7\n" +
			"2,2,8\n" +
			"3,2,9";

		double[][] sample = {{4, 5, 6}, {7, 8, 9}};
		//GenerateReader3.generateReader(raw, DataConverter.convertToMatrixBlock(sample));
	}
	//Type 2: Value, Row, Col
	@Test public void testMM3() throws Exception {
		raw = "4,1,1\n" +
			"5,1,2\n" +
			"6,1,3\n" +
			"7,2,1\n" +
			"8,2,2\n" +
			"9,2,3";

		double[][] sample = {{4, 5, 6}, {7, 8, 9}};
		//GenerateReader3.generateReader(raw, DataConverter.convertToMatrixBlock(sample));
	}

	//Type 3: Value, Col, Row
	@Test public void testMM4() throws Exception {
		raw = "4,1,1\n" +
			"5,2,1\n" +
			"6,3,1\n" +
			"7,1,2\n" +
			"8,2,2\n" +
			"9,3,2";
		double[][] sample = {{4, 5, 6}, {7, 8, 9}};
		//GenerateReader3.generateReader(raw, DataConverter.convertToMatrixBlock(sample));
	}

	@Test public void testMM5() throws Exception {
		raw = "1,2,1\n" +
			"1,3,2\n" +
			"2,1,3\n" +
			"3,3,7\n" +
			"4,4,4\n";

		double[][] sample = {{4, 5, 6}, {7, 8, 9}};
		//GenerateReader3.generateReader(raw, DataConverter.convertToMatrixBlock(sample));
	}


}
