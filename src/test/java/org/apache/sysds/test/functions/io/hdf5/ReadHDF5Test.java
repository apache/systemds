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

import java.util.HashMap;

import org.apache.sysds.api.DMLScript;
import org.apache.sysds.common.Types.ExecMode;
import org.apache.sysds.conf.CompilerConfig;
import org.apache.sysds.runtime.matrix.data.MatrixValue;
import org.apache.sysds.test.TestConfiguration;
import org.apache.sysds.test.TestUtils;
import org.junit.Test;

public abstract class ReadHDF5Test extends ReadHDF5TestBase {

	protected abstract int getId();

	protected String getInputHDF5FileName() {
		return "transfusion_" + getId() + ".h5";
	}

	private final static double eps = 1e-9;

	@Test
	public void testHDF51_Seq_CP() {
		runReadHDF5Test(getId(), ExecMode.SINGLE_NODE, false);
	}

	@Test
	public void testHDF51_Parallel_CP() {
		runReadHDF5Test(getId(), ExecMode.SINGLE_NODE, true);
	}

	protected void runReadHDF5Test(int testNumber, ExecMode platform, boolean parallel) {

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
			String inputMatrixName = HOME + INPUT_DIR + getInputHDF5FileName(); // always read the same data
			String datasetName = "DATASET_1";

			fullDMLScriptName = HOME + getTestName() + "_" + testNumber + ".dml";
			programArgs = new String[] {"-args", inputMatrixName, datasetName, output("Y")};

			fullRScriptName = HOME + "ReadHDF5_Verify.R";
			rCmd = "Rscript" + " " + fullRScriptName + " " + inputMatrixName + " " + datasetName + " " + expectedDir();

			runTest(true, false, null, -1);
			runRScript(true);

			HashMap<MatrixValue.CellIndex, Double> YR = readRMatrixFromExpectedDir("Y");
			HashMap<MatrixValue.CellIndex, Double> YSYSTEMDS = readDMLMatrixFromOutputDir("Y");
			TestUtils.compareMatrices(YR, YSYSTEMDS, eps, "YR", "YSYSTEMDS");
		}
		finally {
			rtplatform = oldPlatform;
			CompilerConfig.FLAG_PARREADWRITE_TEXT = oldpar;
			DMLScript.USE_LOCAL_SPARK_CONFIG = sparkConfigOld;
		}
	}
}
