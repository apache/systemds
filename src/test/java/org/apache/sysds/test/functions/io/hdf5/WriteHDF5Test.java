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

	protected String getOutputHDF5FileName() {
		return "transfusion_2.h5";
	}

	private final static double eps = 1e-9;

	@Test public void testHDF51_Seq_CP() {
		runWriteHDF5Test(getId(), ExecMode.SINGLE_NODE, false, 20, 20);
	}

	@Test public void testHDF52_Seq_CP() {
		runWriteHDF5Test(getId(), ExecMode.SINGLE_NODE, false, 1000, 1000);
	}

	@Test public void testHDF53_Seq_CP() {
		runWriteHDF5Test(getId(), ExecMode.SINGLE_NODE, false, 5000, 5000);
	}

	@Test public void testHDF51_Parallel_CP() {
		runWriteHDF5Test(getId(), ExecMode.SINGLE_NODE, true, 1000, 1000);
	}

	@Test public void testHDF52_Parallel_CP() {
		runWriteHDF5Test(getId(), ExecMode.SINGLE_NODE, true, 5000, 5000);
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
			String dmlOutput = output("dml.scalar");

			String datasetName = "DATASET_1";

			//TODO: add RScript for verify file format

			fullDMLScriptName = HOME + getTestName() + "_" + testNumber + ".dml";
			programArgs = new String[] {"-args", input("A"), outMatrixName, datasetName};

			//generate actual datasets
			double[][] A = getRandomMatrix(rows, cols, 0, 10, 1, 714);
			writeInputMatrixWithMTD("A", A, false);

			runTest(true, false, null, -1);
		}
		finally {
			rtplatform = oldPlatform;
			CompilerConfig.FLAG_PARREADWRITE_TEXT = oldpar;
			DMLScript.USE_LOCAL_SPARK_CONFIG = sparkConfigOld;
		}
	}
}
