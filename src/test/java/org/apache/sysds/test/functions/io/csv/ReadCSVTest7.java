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

import static org.junit.Assert.assertTrue;

import org.apache.sysds.api.DMLScript;
import org.apache.sysds.common.Types.ExecMode;
import org.apache.sysds.conf.CompilerConfig;
import org.apache.sysds.test.TestConfiguration;

public class ReadCSVTest7 extends ReadCSVTest4Nan {

	private final static String TEST_NAME = "ReadCSVTest";
	private final static String TEST_CLASS_DIR = TEST_DIR + ReadCSVTest1.class.getSimpleName() + "/";

	protected String expected;

	public ReadCSVTest7(){
		 expected = "1.000 1.000 1.000 1.000";
	}

	protected String getTestName() {
		return TEST_NAME;
	}

	protected String getTestClassDir() {
		return TEST_CLASS_DIR;
	}

	@Override
	protected String getInputCSVFileName() {
		return "sparse_" + getId();
	}

	protected int getId() {
		return 7;
	}

	@Override
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

			TestConfiguration config = getTestConfiguration(getTestName());
			loadTestConfiguration(config);
			setOutputBuffering(true); // otherwise NPEs

			String HOME = SCRIPT_DIR + TEST_DIR;
			String inputMatrixNameNoExtension = HOME + INPUT_DIR + getInputCSVFileName();
			String inputMatrixNameWithExtension = inputMatrixNameNoExtension + ".csv";

			fullDMLScriptName = HOME + getTestName() + "_" + testNumber + ".dml";
			programArgs = new String[] {"-args", inputMatrixNameWithExtension};

			output = runTest(true, false, null, -1).toString();
			assertTrue("was :" + output + " expected: " + expected, output.contains(expected));
		}
		finally {
			rtplatform = oldPlatform;
			CompilerConfig.FLAG_PARREADWRITE_TEXT = oldpar;
			DMLScript.USE_LOCAL_SPARK_CONFIG = sparkConfigOld;
		}
		return output;
	}
}
