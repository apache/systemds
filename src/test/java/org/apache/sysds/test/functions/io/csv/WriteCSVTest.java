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

package org.apache.sysds.test.functions.io.csv;

import java.io.IOException;

import org.apache.sysds.common.Types.ExecMode;
import org.apache.sysds.test.AutomatedTestBase;
import org.apache.sysds.test.TestConfiguration;
import org.apache.sysds.test.TestUtils;
import org.junit.Test;

/**
 * JUnit Test cases to evaluate the functionality of reading CSV files.
 * 
 * Test 1: write() w/ all properties. Test 2: read(format="csv") w/o mtd file. Test 3: read() w/ complete mtd file.
 *
 */

public class WriteCSVTest extends AutomatedTestBase {

	private final static String TEST_NAME = "WriteCSVTest";
	private final static String TEST_DIR = "functions/io/csv/";
	private final static String TEST_CLASS_DIR = TEST_DIR + WriteCSVTest.class.getSimpleName() + "/";

	private final static double eps = 1e-9;

	@Override
	public void setUp() {
		addTestConfiguration(TEST_NAME, new TestConfiguration(TEST_CLASS_DIR, TEST_NAME, new String[] {"Rout"}));
	}

	@Test
	public void testCSV1_CP() throws IOException {
		runCSVWriteTest(ExecMode.HYBRID, true, ":", true);
	}

	@Test
	public void testCSV2_CP() throws IOException {
		runCSVWriteTest(ExecMode.HYBRID, false, ":", true);
	}

	@Test
	public void testCSV3_CP() throws IOException {
		runCSVWriteTest(ExecMode.HYBRID, false, ":", false);
	}

	@Test
	public void testCSV4_CP() throws IOException {
		runCSVWriteTest(ExecMode.HYBRID, false, ".", false);
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
		rCmd = "Rscript" + " " + fullRScriptName + " " + csvOutputName + " " + Boolean.toString(header).toUpperCase()
			+ " " + sep + " " + rOutput;
		runRScript(true);

		double dmlScalar = TestUtils.readDMLScalar(dmlOutput);
		double rScalar = TestUtils.readRScalar(rOutput);

		TestUtils.compareScalars(dmlScalar, rScalar, eps);

		rtplatform = oldPlatform;
	}
}
