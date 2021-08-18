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

import java.io.IOException;

public class GenerateReaderCSVTest_Unique extends AutomatedTestBase {

	private final static String TEST_NAME = "GenerateReaderCSVTest1";
	private final static String TEST_DIR = "functions/io/GenerateReaderCSVTest1/";
	private final static String TEST_CLASS_DIR = TEST_DIR + GenerateReaderCSVTest_Unique.class.getSimpleName() + "/";

	private final static double eps = 1e-9;

	private String sampleRaw;
	private MatrixBlock sampleMatrix;

	@Override public void setUp() {
		addTestConfiguration(TEST_NAME, new TestConfiguration(TEST_CLASS_DIR, TEST_NAME, new String[] {"Rout"}));
	}

	//1. Generate CSV Test Data
	//1.a. The Data include Header and Unique Values
	@Test public void testCSV1_CP_CSV_Data_With_Header() throws IOException {

		//stream = "a,b,c,d,e,f\n" + "1,2,3,4,5,6\n" + "7,8,9,10,11,12\n" + "2,3,1,5,4,6\n" + "1,5,3,4,2,6\n" + "8,9,10,11,12,7\n" + "1,2,3,4,5,6\n" + "7,8,9,10,11,12";

		double[][] sample = {{1, 2, 3, 4, 5, 6}, {7, 8, 9, 10, 11, 12}};
		//GenerateReader2.generateReader(stream, DataConverter.convertToMatrixBlock(sample));
	}





}
