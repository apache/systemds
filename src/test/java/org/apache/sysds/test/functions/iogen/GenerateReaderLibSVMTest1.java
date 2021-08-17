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

import org.apache.sysds.runtime.iogen.GenerateReader;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;
import org.apache.sysds.runtime.util.DataConverter;
import org.apache.sysds.test.AutomatedTestBase;
import org.apache.sysds.test.TestConfiguration;
import org.junit.Test;

public class GenerateReaderLibSVMTest1 extends AutomatedTestBase {

	private final static String TEST_NAME = "GenerateReaderCSVTest1";
	private final static String TEST_DIR = "functions/io/GenerateReaderCSVTest1/";
	private final static String TEST_CLASS_DIR = TEST_DIR + GenerateReaderLibSVMTest1.class.getSimpleName() + "/";

	private final static double eps = 1e-9;

	private String raw;
	private MatrixBlock sample;

	@Override public void setUp() {
		addTestConfiguration(TEST_NAME, new TestConfiguration(TEST_CLASS_DIR, TEST_NAME, new String[] {"Rout"}));
	}

	// Index start from 1
	//Type 0: Row, Col, Value
	@Test public void testLibSVM1() throws Exception {
		raw = "1 1:1 2:2\n"+
			  "-1 3:4 4:5 5:8\n";

		double[][] sample = {{1, 2, 0, 0, 0,1}, {0, 0, 4, 5, 8,-1}};
		GenerateReader.generateReader(raw, DataConverter.convertToMatrixBlock(sample));
	}

	@Test public void testLibSVM2() throws Exception {
		raw = "1 1.0:1.0 2.0:2.0\n"+
			"-1.0 3:4 4:5.0 5:8.0\n";

		double[][] sample = {{1, 2, 0, 0, 0,1}, {0, 0, 4, 5, 8,-1}};
		GenerateReader.generateReader(raw, DataConverter.convertToMatrixBlock(sample));
	}
}
