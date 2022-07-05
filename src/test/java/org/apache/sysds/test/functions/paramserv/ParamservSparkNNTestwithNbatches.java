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

package org.apache.sysds.test.functions.paramserv;

import org.apache.sysds.api.DMLScript;
import org.apache.sysds.common.Types.ExecMode;
import org.apache.sysds.parser.Statement;
import org.apache.sysds.test.AutomatedTestBase;
import org.apache.sysds.test.TestConfiguration;
import org.junit.Ignore;
import org.junit.Test;

@net.jcip.annotations.NotThreadSafe
@Ignore
public class ParamservSparkNNTestwithNbatches extends AutomatedTestBase {

	private static final String TEST_NAME1 = "paramserv-nbatches-test";

	private static final String TEST_DIR = "functions/paramserv/";
	private static final String TEST_CLASS_DIR = TEST_DIR + ParamservSparkNNTestwithNbatches.class.getSimpleName() + "/";

	@Override
	public void setUp() {
		addTestConfiguration(TEST_NAME1, new TestConfiguration(TEST_CLASS_DIR, TEST_NAME1, new String[] {}));
	}

	@Test
	public void testParamservBSPNbatchesDisjointContiguous() {
		runDMLTest(2, 2, Statement.PSUpdateType.BSP, Statement.PSFrequency.BATCH, 16, Statement.PSScheme.DISJOINT_CONTIGUOUS, 16, false);
	}

	@Test
	public void testParamservASPNbatchesDisjointContiguous() {
		runDMLTest(2, 2, Statement.PSUpdateType.ASP, Statement.PSFrequency.BATCH, 16, Statement.PSScheme.DISJOINT_CONTIGUOUS, 16, false);
	}

	@Test
	public void testParamservSBPNbatchesDisjointContiguous() {
		runDMLTest(2, 3, Statement.PSUpdateType.SBP, Statement.PSFrequency.BATCH, 16, Statement.PSScheme.DISJOINT_CONTIGUOUS, 16, false);
	}

	private void internalRunDMLTest(String testname, boolean exceptionExpected, Class<?> expectedException,
		String errMessage) {
		ExecMode oldRtplatform = AutomatedTestBase.rtplatform;
		boolean oldUseLocalSparkConfig = DMLScript.USE_LOCAL_SPARK_CONFIG;
		AutomatedTestBase.rtplatform = ExecMode.HYBRID;
		DMLScript.USE_LOCAL_SPARK_CONFIG = true;

		try {
			TestConfiguration config = getTestConfiguration(testname);
			loadTestConfiguration(config);
			String HOME = SCRIPT_DIR + TEST_DIR;
			fullDMLScriptName = HOME + testname + ".dml";
			runTest(true, exceptionExpected, expectedException, errMessage, -1);
		} finally {
			AutomatedTestBase.rtplatform = oldRtplatform;
			DMLScript.USE_LOCAL_SPARK_CONFIG = oldUseLocalSparkConfig;
		}
	}

	private void runDMLTest(int epochs, int workers, Statement.PSUpdateType utype, Statement.PSFrequency freq, int batchsize, Statement.PSScheme scheme, int nbatches, boolean modelAvg) {
		programArgs = new String[] { "-nvargs", "mode=REMOTE_SPARK", "epochs=" + epochs, "workers=" + workers, "utype=" + utype, "freq=" + freq, "batchsize=" + batchsize, "scheme=" + scheme, "nbatches=" + nbatches, "modelAvg=" + modelAvg};
		internalRunDMLTest(TEST_NAME1, false, null, null);
	}
}
