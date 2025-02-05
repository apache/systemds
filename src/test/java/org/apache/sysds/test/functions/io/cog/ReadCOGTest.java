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

package org.apache.sysds.test.functions.io.cog;

import org.apache.sysds.api.DMLScript;
import org.apache.sysds.common.Types;
import org.apache.sysds.conf.CompilerConfig;
import org.apache.sysds.test.TestConfiguration;
import org.apache.sysds.test.TestUtils;
import org.junit.Test;



public abstract class ReadCOGTest extends COGTestBase {
	protected abstract int getId();

	protected String getInputCOGFileName() {
		return "testCOG_" + getId();
	}

	protected abstract double getResult();

	@Test
	public void testCOG_Seq_CP() {
		runReadCOGTest(getId(), getResult(), Types.ExecMode.SINGLE_NODE, false);
	}
	@Test
	public void testCOG_Parallel_CP1() {
		runReadCOGTest(getId(), getResult(), Types.ExecMode.SINGLE_NODE, true);
	}

	@Test
	public void testCOG_Parallel_CP() {
		runReadCOGTest(getId(), getResult(), Types.ExecMode.HYBRID, true);
	}

	// TODO: Spark



	protected void runReadCOGTest(int testNumber, double result, Types.ExecMode platform, boolean parallel) {
		Types.ExecMode oldPlatform = rtplatform;
		rtplatform = platform;

		boolean sparkConfigOld = DMLScript.USE_LOCAL_SPARK_CONFIG;
		if(rtplatform == Types.ExecMode.SPARK)
			DMLScript.USE_LOCAL_SPARK_CONFIG = true;

		boolean oldpar = CompilerConfig.FLAG_PARREADWRITE_TEXT; // set to false for debugging maybeee

		try {
			CompilerConfig.FLAG_PARREADWRITE_TEXT = parallel;
			TestConfiguration config = getTestConfiguration(getTestName());
			loadTestConfiguration(config);

			String HOME = SCRIPT_DIR + TEST_DIR;
			String inputMatrixName = DATASET_DIR + "cog/" + getInputCOGFileName() + ".tif";

			String dmlOutput = output("dml.scalar");

			fullDMLScriptName = HOME + getTestName() + "_" + getScriptId() + ".dml";
			programArgs = new String[] {"-args", inputMatrixName, dmlOutput};

			runTest(true, false, null, -1);

			double dmlScalarOutput = TestUtils.readDMLScalar(dmlOutput);
			TestUtils.compareScalars(dmlScalarOutput, result, eps * getResult());
		}
		finally {
			rtplatform = oldPlatform;
			CompilerConfig.FLAG_PARREADWRITE_TEXT = oldpar;
			DMLScript.USE_LOCAL_SPARK_CONFIG = sparkConfigOld;
		}
	}
}

