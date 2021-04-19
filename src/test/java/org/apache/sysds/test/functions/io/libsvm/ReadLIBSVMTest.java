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

package org.apache.sysds.test.functions.io.libsvm;

import org.apache.sysds.api.DMLScript;
import org.apache.sysds.common.Types.ExecMode;
import org.apache.sysds.conf.CompilerConfig;
import org.apache.sysds.test.TestConfiguration;
import org.apache.sysds.test.TestUtils;
import org.junit.Test;

public abstract class ReadLIBSVMTest extends ReadLIBSVMTestBase {

	protected abstract int getId();

	protected abstract LIBSVMConfig getLIBSVMConfig();

	protected String getInputLIBSVMFileName() {
		return "transfusion_" + getId() + ".libsvm";
	}

	private final static double eps = 1e-9;

	@Test public void testlibsvm1_Seq_CP() {
		runlibsvmTest(getId(), ExecMode.SINGLE_NODE, false, getLIBSVMConfig());
	}

	@Test public void testlibsvm2_Pllel_CP() {
		runlibsvmTest(getId(), ExecMode.SINGLE_NODE, true, getLIBSVMConfig());
	}

	@Test public void testlibsvm3_SP() {
		runlibsvmTest(getId(), ExecMode.SPARK, false, getLIBSVMConfig());
	}

	protected void runlibsvmTest(int testNumber, ExecMode platform, boolean parallel, LIBSVMConfig libsvmConfig) {
		ExecMode oldPlatform = rtplatform;
		rtplatform = platform;

		boolean sparkConfigOld = DMLScript.USE_LOCAL_SPARK_CONFIG;
		if(rtplatform == ExecMode.SPARK)
			DMLScript.USE_LOCAL_SPARK_CONFIG = true;

		boolean oldpar = CompilerConfig.FLAG_PARREADWRITE_TEXT;

		try {
			CompilerConfig.FLAG_PARREADWRITE_TEXT = parallel;

			TestConfiguration config = getTestConfiguration(getTestName());
			loadTestConfiguration(config);

			String HOME = SCRIPT_DIR + TEST_DIR;
			String inputMatrix = HOME + INPUT_DIR + getInputLIBSVMFileName();
			String dmlOutput = output("dml.scalar");
			String rOutput = output("R.scalar");
			String sep = libsvmConfig.getOutSep();
			String indSep = libsvmConfig.getOutIndSep();

			fullDMLScriptName = HOME + getTestName() + "_" + testNumber + ".dml";
			programArgs = new String[] {"-explain", "hops", "-args", inputMatrix, dmlOutput};
			runTest(true, false, null, -1);

			fullRScriptName = HOME + "libsvm_verify.R";

			if(sep.equals(" ")) {
				sep = "NULL";
			}
			rCmd = "Rscript" + " " + fullRScriptName + " " + inputMatrix + " " + libsvmConfig
				.getColCount() + " " + sep + " " + indSep + " " + rOutput;
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
	}
}
