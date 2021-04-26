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
import org.apache.sysds.conf.CompilerConfig;
import org.junit.Test;
import org.apache.sysds.common.Types.ExecMode;
import org.apache.sysds.test.TestConfiguration;

public abstract class WriteLIBSVMTest extends WriteLIBSVMTestBase {

	protected abstract int getId();

	protected String getInputLIBSVMFileName() {
		return "transfusion_W" + getId() + ".libsvm";
	}

	@Test public void testlibsvm1_Seq_CP() {
		runWriteLIBSVMTest(getId(), ExecMode.SINGLE_NODE, false, " ", ":", false);
	}

	@Test public void testlibsvm2_Seq_CP() {
		runWriteLIBSVMTest(getId(), ExecMode.SINGLE_NODE, false, " ", ":", true);
	}

	@Test public void testlibsvm1_Pllel_CP() {
		runWriteLIBSVMTest(getId(), ExecMode.SINGLE_NODE, true, " ", ":", true);
	}

	@Test public void testlibsvm2_Pllel_CP() {
		runWriteLIBSVMTest(getId(), ExecMode.SINGLE_NODE, true, " ", ":", false);
	}

	@Test public void testlibsvm1_SP() {
		runWriteLIBSVMTest(getId(), ExecMode.SPARK, false, " ", ":", true);
	}

	@Test public void testlibsvm2_SP() {
		runWriteLIBSVMTest(getId(), ExecMode.SPARK, false, " ", ":", false);
	}

	protected void runWriteLIBSVMTest(int testNumber, ExecMode platform, boolean parallel, String sep, String indSep,
		boolean sparse) {

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
			String inputMatrixName = HOME + INPUT_DIR + getInputLIBSVMFileName();
			String dmlOutput = output("dml.scalar");
			String libsvmOutputName = output("libsvm_write" + testNumber + ".data");

			fullDMLScriptName = HOME + getTestName() + "_" + testNumber + ".dml";
			programArgs = new String[] {"-args", inputMatrixName, dmlOutput, libsvmOutputName, sep, indSep,
				Boolean.toString(sparse)};

			runTest(true, false, null, -1);
		}
		finally {
			rtplatform = oldPlatform;
			CompilerConfig.FLAG_PARREADWRITE_TEXT = oldpar;
			DMLScript.USE_LOCAL_SPARK_CONFIG = sparkConfigOld;
		}
	}
}
