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

package org.apache.sysds.test.functions.federated.algorithms;

import java.util.Arrays;
import java.util.Collection;

import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;
import org.apache.sysds.api.DMLScript;
import org.apache.sysds.common.Types;
import org.apache.sysds.runtime.meta.MatrixCharacteristics;
import org.apache.sysds.test.AutomatedTestBase;
import org.apache.sysds.test.TestConfiguration;
import org.apache.sysds.test.TestUtils;
import org.junit.Ignore;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.Parameterized;

@RunWith(value = Parameterized.class)
@net.jcip.annotations.NotThreadSafe
public class FederatedYL2SVMTest extends AutomatedTestBase {
	private static final Log LOG = LogFactory.getLog(FederatedYL2SVMTest.class.getName());

	private final static String TEST_DIR = "functions/federated/";
	private final static String TEST_NAME = "FederatedYL2SVMTest";
	private final static String TEST_NAME_2 = "FederatedYL2SVMTest2";
	private final static String TEST_CLASS_DIR = TEST_DIR + FederatedYL2SVMTest.class.getSimpleName() + "/";

	private final static int blocksize = 1024;
	@Parameterized.Parameter()
	public int rows;
	@Parameterized.Parameter(1)
	public int cols;

	@Override
	public void setUp() {
		TestUtils.clearAssertionInformation();
		addTestConfiguration(TEST_NAME, new TestConfiguration(TEST_CLASS_DIR, TEST_NAME, new String[] {"Z"}));
		addTestConfiguration(TEST_NAME_2, new TestConfiguration(TEST_CLASS_DIR, TEST_NAME_2, new String[] {"Z"}));
	}

	@Parameterized.Parameters
	public static Collection<Object[]> data() {
		// rows have to be even and > 1
		return Arrays.asList(new Object[][] {
			// {2, 1000}, {10, 100}, {100, 10}, {1000, 1}, {10, 2000},
			{2000, 10}});
	}

	@Test
	public void federatedL2SVMCP() {
		federatedL2SVM(Types.ExecMode.SINGLE_NODE, TEST_NAME);
	}

	@Test
	public void federatedL2SVMCP_2() {
		// This test is equal to the first tests, just with one worker location used instead.
		// making all federated matrices FULL type.
		federatedL2SVM(Types.ExecMode.SINGLE_NODE, TEST_NAME_2);
	}

	@Test
	@Ignore
	public void federatedL2SVMSP() {
		federatedL2SVM(Types.ExecMode.SPARK, TEST_NAME);
	}

	public void federatedL2SVM(Types.ExecMode execMode, String testName) {
		boolean sparkConfigOld = DMLScript.USE_LOCAL_SPARK_CONFIG;
		Types.ExecMode platformOld = rtplatform;
		rtplatform = execMode;
		if(rtplatform == Types.ExecMode.SPARK) {
			DMLScript.USE_LOCAL_SPARK_CONFIG = true;
		}

		getAndLoadTestConfiguration(testName);
		String HOME = SCRIPT_DIR + TEST_DIR;

		// write input matrices
		int halfRows = rows / 2;
		// We have two matrices handled by a single federated worker
		double[][] X1 = getRandomMatrix(halfRows, cols, 0, 1, 1, 42);
		double[][] X2 = getRandomMatrix(halfRows, cols, 0, 1, 1, 1340);
		double[][] Y1 = getRandomMatrix(halfRows, 1, -1, 1, 1, 1233);
		double[][] Y2 = getRandomMatrix(halfRows, 1, -1, 1, 1, 13);

		for(int i = 0; i < halfRows; i++) {
			Y1[i][0] = (Y1[i][0] > 0) ? 1 : -1;
			Y2[i][0] = (Y2[i][0] > 0) ? 1 : -1;
		}

		writeInputMatrixWithMTD("X1", X1, false, new MatrixCharacteristics(halfRows, cols, blocksize, halfRows * cols));
		writeInputMatrixWithMTD("X2", X2, false, new MatrixCharacteristics(halfRows, cols, blocksize, halfRows * cols));
		writeInputMatrixWithMTD("Y1", Y1, false, new MatrixCharacteristics(halfRows, 1, blocksize, halfRows));
		writeInputMatrixWithMTD("Y2", Y2, false, new MatrixCharacteristics(halfRows, 1, blocksize, halfRows));

		// empty script name because we don't execute any script, just start the worker
		fullDMLScriptName = "";
		int port1 = getRandomAvailablePort();
		int port2 = getRandomAvailablePort();
		Thread t1 = startLocalFedWorkerThread(port1, FED_WORKER_WAIT_S);
		Thread t2 = startLocalFedWorkerThread(port2);

		TestConfiguration config = availableTestConfigurations.get(testName);
		loadTestConfiguration(config);

		// Run reference dml script with normal matrix
		fullDMLScriptName = HOME + testName + "Reference.dml";
		programArgs = new String[] {"-args", input("X1"), input("X2"), input("Y1"), input("Y2"), expected("Z")};
		LOG.debug(runTest(null));

		// Run actual dml script with federated matrixz
		fullDMLScriptName = HOME + testName + ".dml";
		programArgs = new String[] {"-stats", "-nvargs", "in_X1=" + TestUtils.federatedAddress(port1, input("X1")),
			"in_X2=" + TestUtils.federatedAddress(port2, input("X2")), "rows=" + rows, "cols=" + cols,
			"in_Y1=" + TestUtils.federatedAddress(port1, input("Y1")),
			"in_Y2=" + TestUtils.federatedAddress(port2, input("Y2")), "out=" + output("Z")};
		LOG.debug(runTest(null));

		// compare via files
		compareResults(1e-9);

		TestUtils.shutdownThreads(t1, t2);

		rtplatform = platformOld;
		DMLScript.USE_LOCAL_SPARK_CONFIG = sparkConfigOld;
	}
}
