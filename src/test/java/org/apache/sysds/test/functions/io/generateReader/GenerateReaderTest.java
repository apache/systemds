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

import org.apache.sysds.common.Types.ExecMode;
import org.apache.sysds.hops.OptimizerUtils;
import org.apache.sysds.runtime.io.GenerateReader;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;
import org.apache.sysds.runtime.util.DataConverter;
import org.apache.sysds.test.AutomatedTestBase;
import org.apache.sysds.test.TestConfiguration;
import org.apache.sysds.test.TestUtils;
import org.junit.Test;

import java.io.IOException;

public class GenerateReaderTest extends AutomatedTestBase {

	private final static String TEST_NAME = "WriteCSVTest";
	private final static String TEST_DIR = "functions/io/csv/";
	private final static String TEST_CLASS_DIR = TEST_DIR + GenerateReaderTest.class.getSimpleName() + "/";

	private final static double eps = 1e-9;

	private String stream;
	private MatrixBlock sample;

	@Override public void setUp() {
		addTestConfiguration(TEST_NAME, new TestConfiguration(TEST_CLASS_DIR, TEST_NAME, new String[] {"Rout"}));
	}

	private void getCSV_Data1() {
		stream = "a,b,c,d,e\n" +
				 "1,2,3,4,5\n" +
				 "6,7,8,9,10\n" +
				 "11,12,13,14,15";

		double[][] sampleData = {{1, 2, 3, 4, 5}, {6, 7, 8, 9, 10}};
		sample = DataConverter.convertToMatrixBlock(sampleData);
	}

	private void getCSV_Data2() {
		stream = "1,2,3,4,5\n" +
				 "6,7,8,9,10\n" +
				 "11,12,13,14,15";

		double[][] sampleData = {{1, 2, 4, 3, 5}, {6, 7, 9, 8, 10}};
		sample = DataConverter.convertToMatrixBlock(sampleData);
	}

	private void getLIBSVM_Data1() {
		stream = "1 1:1 2:2 5:5\n" +
				 "2 3:3 4:4 6:6 7:7";

		double[][] sampleData = {{1, 2, 0, 0, 5, 0, 0}, {0, 0, 3, 4, 0, 6, 7}};
		sample = DataConverter.convertToMatrixBlock(sampleData);
	}

	@Test public void testCSV1_CP() throws IOException, InstantiationException, IllegalAccessException {
		getCSV_Data1();
		//GenerateReader.generateReader(stream, sample);
	}

	@Test public void testCSV2_CP() throws IOException, InstantiationException, IllegalAccessException {
		getCSV_Data2();

		OptimizerUtils.ALLOW_ALGEBRAIC_SIMPLIFICATION = true;
		GenerateReader.generateReader(stream, sample);
	}

	@Test public void testLIBSVM1_CP() throws IOException, InstantiationException, IllegalAccessException {
		getLIBSVM_Data1();
		//GenerateReader.generateReader(stream, sample);
	}

	private void runCSVWriteTest(ExecMode platform, boolean header, String sep, boolean sparse) throws IOException {

		ExecMode oldPlatform = rtplatform;
		rtplatform = platform;

		TestConfiguration config = getTestConfiguration(TEST_NAME);
		loadTestConfiguration(config);

		String HOME = SCRIPT_DIR + TEST_DIR;
		String inputMatrixName = HOME + INPUT_DIR + "transfusion_1"; // always read the same data, independent of
		// testNumber
		String dmlOutput = output("dml.scalar");
		String csvOutputName = output("transfusion_dml.data");
		String rOutput = output("R.scalar");

		fullDMLScriptName = HOME + TEST_NAME + ".dml";
		programArgs = new String[] {"-args", inputMatrixName, dmlOutput, csvOutputName, Boolean.toString(header), sep,
			Boolean.toString(sparse)};

		runTest(true, false, null, -1);

		// Verify produced CSV file w/ R
		csvOutputName = TestUtils.processMultiPartCSVForR(csvOutputName);

		fullRScriptName = HOME + "writecsv_verify.R";
		rCmd = "Rscript" + " " + fullRScriptName + " " + csvOutputName + " " + Boolean.toString(header)
			.toUpperCase() + " " + sep + " " + rOutput;
		runRScript(true);

		double dmlScalar = TestUtils.readDMLScalar(dmlOutput);
		double rScalar = TestUtils.readRScalar(rOutput);

		TestUtils.compareScalars(dmlScalar, rScalar, eps);

		rtplatform = oldPlatform;
	}
}
