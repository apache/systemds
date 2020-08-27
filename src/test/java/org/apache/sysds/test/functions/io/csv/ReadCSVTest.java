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

import org.apache.sysds.api.DMLScript;
import org.apache.sysds.common.Types.ExecMode;
import org.apache.sysds.conf.CompilerConfig;
import org.apache.sysds.test.TestConfiguration;
import org.apache.sysds.test.TestUtils;
import org.junit.Test;

public abstract class ReadCSVTest extends CSVTestBase {

	protected abstract int getId();
	protected String getInputCSVFileName() {
		return "transfusion_" + getId();
	}

	 @Test
	 public void testCSV_Sequential_CP1() {
	 	runCSVTest(getId(), ExecMode.SINGLE_NODE, false);
	 }

	 @Test
	 public void testCSV_Parallel_CP1() {
	 	runCSVTest(getId(), ExecMode.SINGLE_NODE, true);
	 }

	 @Test
	 public void testCSV_Sequential_CP() {
	 	runCSVTest(getId(), ExecMode.HYBRID, false);
	 }

	 @Test
	 public void testCSV_Parallel_CP() {
	 	runCSVTest(getId(), ExecMode.HYBRID, true);
	 }

	@Test
	public void testCSV_SP() {
		runCSVTest(getId(), ExecMode.SPARK, false);
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

			TestConfiguration config = getTestConfiguration(getTestName());

			loadTestConfiguration(config);

			String HOME = SCRIPT_DIR + TEST_DIR;
			String inputMatrixNameNoExtension = HOME + INPUT_DIR + getInputCSVFileName();
			String inputMatrixNameWithExtension = inputMatrixNameNoExtension + ".csv";
			String dmlOutput = output("dml.scalar");
			String rOutput = output("R.scalar");

			fullDMLScriptName = HOME + getTestName() + "_" + testNumber + ".dml";
			programArgs = new String[] {"-args", inputMatrixNameWithExtension, dmlOutput};

			fullRScriptName = HOME + "csv_verify2.R";
			rCmd = "Rscript" + " " + fullRScriptName + " " + inputMatrixNameNoExtension + ".single.csv " + rOutput;

			output = runTest(true, false, null, -1).toString();
			runRScript(true);

			double dmlScalar = TestUtils.readDMLScalar(dmlOutput);
			double rScalar = TestUtils.readRScalar(rOutput);

			TestUtils.compareScalars(dmlScalar, rScalar, eps);
		}
		finally {
			rtplatform = oldPlatform;
			CompilerConfig.FLAG_PARREADWRITE_TEXT = oldpar;
			DMLScript.USE_LOCAL_SPARK_CONFIG = sparkConfigOld;
		}
		return output;
	}
}
