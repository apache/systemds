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
import org.apache.sysds.runtime.matrix.data.MatrixValue;
import org.apache.sysds.test.TestConfiguration;
import org.apache.sysds.test.TestUtils;
import org.junit.Test;

import java.util.HashMap;

public abstract class WriteHDF5Test extends WriteHDF5TestBase {

	protected abstract int getId();

	protected String getOutputHDF5FileName() {
		return "transfusion_"+getId()+".h5";
	}

	private final static double eps = 1e-9;

	@Test public void testHDF51_Seq_CP() {
		runWriteHDF5Test(getId(), ExecMode.SINGLE_NODE, false, 50, 50);
	}

	@Test public void testHDF52_Seq_CP() {
		runWriteHDF5Test(getId(), ExecMode.SINGLE_NODE, false, 500, 800);
	}

	@Test public void testHDF53_Seq_CP() {
		runWriteHDF5Test(getId(), ExecMode.SINGLE_NODE, false, 1000, 2000);
	}

	@Test public void testHDF51_Parallel_CP() {
		runWriteHDF5Test(getId(), ExecMode.SINGLE_NODE, true, 500, 500);
	}

	@Test public void testHDF52_Parallel_CP() {
		runWriteHDF5Test(getId(), ExecMode.SINGLE_NODE, true, 1500, 1500);
	}

	protected void runWriteHDF5Test(int testNumber, ExecMode platform, boolean parallel, int rows, int cols) {

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
			String outMatrixName = HOME + OUTPUT_DIR + getOutputHDF5FileName();

			String datasetName = "DATASET_1";

			fullDMLScriptName = HOME + getTestName() + "_" + testNumber + ".dml";
			programArgs = new String[] {"-args", input("A"), outMatrixName, datasetName};

			//generate actual datasets
			double[][] A = getRandomMatrix(rows, cols, 0, 10, 1, 714);
			writeInputMatrixWithMTD("A", A, false);

			runTest(true, false, null, -1);

			fullRScriptName = HOME + "ReadHDF5_Verify.R";
			rCmd = "Rscript" + " " + fullRScriptName + " " + outMatrixName + " " + datasetName + " " + expectedDir();

			runRScript(true);

			HashMap<MatrixValue.CellIndex, Double> YR = readRMatrixFromExpectedDir("Y");
			HashMap<MatrixValue.CellIndex, Double> YA = TestUtils.convert2DDoubleArrayToHashMap(A);
			TestUtils.compareMatrices(YR, YA, eps, "YR", "YA");

		}
		finally {
			rtplatform = oldPlatform;
			CompilerConfig.FLAG_PARREADWRITE_TEXT = oldpar;
			DMLScript.USE_LOCAL_SPARK_CONFIG = sparkConfigOld;
		}
	}
}
