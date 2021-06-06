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

package org.apache.sysds.test.functions.io.hdf5;

import org.apache.sysds.api.DMLScript;
import org.apache.sysds.common.Types.ExecMode;
import org.apache.sysds.conf.CompilerConfig;
import org.apache.sysds.test.TestConfiguration;
import org.junit.Test;

public abstract class WriteHDF5Test extends WriteHDF5TestBase {

	protected abstract int getId();


	protected String getInputHDF5FileName() {
		return "transfusion_" + getId() + ".csv";
	}

	private final static double eps = 1e-9;

	@Test public void testHDF51_Seq_CP() {
		runWriteHDF5Test(getId(), ExecMode.SINGLE_NODE, false, false);
	}

	protected void runWriteHDF5Test(int testNumber, ExecMode platform, boolean parallel, boolean sparse) {

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
			String inputMatrixName =HOME + INPUT_DIR + getInputHDF5FileName(); // "/home/sfathollahzadeh/adbc.h5";//
			String dmlOutput = output("dml.scalar");
			String rOutput = output("R.scalar");
			//String libsvmOutputName = output("libsvm_write" + testNumber + ".data");

			fullDMLScriptName = HOME + getTestName() + "_" + testNumber + ".dml";
			programArgs = new String[] {"-args", inputMatrixName, dmlOutput, Boolean.toString(sparse)};

			runTest(true, false, null, -1);
		}
		finally {
			rtplatform = oldPlatform;
			CompilerConfig.FLAG_PARREADWRITE_TEXT = oldpar;
			DMLScript.USE_LOCAL_SPARK_CONFIG = sparkConfigOld;
		}
	}
}
