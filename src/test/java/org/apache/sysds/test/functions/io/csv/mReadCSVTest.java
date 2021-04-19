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

import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;
import org.apache.sysds.api.DMLScript;
import org.apache.sysds.common.Types.ExecMode;
import org.apache.sysds.conf.CompilerConfig;
import org.apache.sysds.test.AutomatedTestBase;
import org.apache.sysds.test.TestConfiguration;
import org.apache.sysds.test.TestUtils;
import org.junit.Test;

public class mReadCSVTest extends AutomatedTestBase {

	private final static String TEST_DIR = "functions/io/csv/";
	private static final Log LOG = LogFactory.getLog(CSVTestBase.class.getName());

	private final static String TEST_NAME = "ReadCSVTest";
	private final static String TEST_CLASS_DIR = TEST_DIR + ReadCSVTest1.class.getSimpleName() + "/";

	private final static double eps = 1e-9;

	protected String getInputCSVFileName() {
		return "transfusion_1" ;
	}

	@Override public void setUp() {
		TestUtils.clearAssertionInformation();
		addTestConfiguration(TEST_NAME, new TestConfiguration(TEST_CLASS_DIR, TEST_NAME, new String[] {"Rout"}));
	}

	@Test
	public void testCSV_Sequential_CP1() {
		runCSVTest(1, ExecMode.SINGLE_NODE, false);
	}

	@Test
	public void testCSV_Parallel_CP1() {
		runCSVTest(1, ExecMode.SINGLE_NODE, true);
	}

	@Test
	public void testCSV_Sequential_CP() {
		runCSVTest(1, ExecMode.HYBRID, false);
	}

	@Test
	public void testCSV_Parallel_CP() {
		runCSVTest(1, ExecMode.HYBRID, true);
	}

	@Test
	public void testCSV_SP() {
		runCSVTest(1, ExecMode.SPARK, false);
	}

	protected String runCSVTest(int testNumber, ExecMode platform, boolean parallel) {
		ExecMode oldPlatform = rtplatform;
		rtplatform = platform;

		boolean sparkConfigOld = DMLScript.USE_LOCAL_SPARK_CONFIG;
		if(rtplatform == ExecMode.SPARK)
			DMLScript.USE_LOCAL_SPARK_CONFIG = true;

		boolean oldpar = CompilerConfig.FLAG_PARREADWRITE_TEXT;

		String output;
		try {
			CompilerConfig.FLAG_PARREADWRITE_TEXT = parallel;

			TestConfiguration config = getTestConfiguration(TEST_NAME);
			loadTestConfiguration(config);
			setOutputBuffering(true); //otherwise NPEs

			String HOME = SCRIPT_DIR + TEST_DIR;
			String inputMatrixNameNoExtension = HOME + INPUT_DIR + getInputCSVFileName();
			String inputMatrixNameWithExtension = inputMatrixNameNoExtension + ".csv";
			String dmlOutput = output("dml.scalar");
			String rOutput = output("R.scalar");

			fullDMLScriptName = HOME + TEST_NAME + "_" + testNumber + ".dml";
			programArgs = new String[] {"-args", inputMatrixNameWithExtension, dmlOutput};

			fullRScriptName = HOME + "csv_verify2.R";
			rCmd = "Rscript" + " " + fullRScriptName + " " + inputMatrixNameNoExtension + ".single.csv " + rOutput;

			output = runTest(true, false, null, -1).toString();
//			runRScript(true);
//
//			double dmlScalar = TestUtils.readDMLScalar(dmlOutput);
//			double rScalar = TestUtils.readRScalar(rOutput);
//
//
//			TestUtils.compareScalars(dmlScalar, rScalar, eps);
		}
		finally {
			rtplatform = oldPlatform;
			CompilerConfig.FLAG_PARREADWRITE_TEXT = oldpar;
			DMLScript.USE_LOCAL_SPARK_CONFIG = sparkConfigOld;
		}
		return output;
	}
}
