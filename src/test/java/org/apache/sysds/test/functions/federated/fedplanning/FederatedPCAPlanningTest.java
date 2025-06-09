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

package org.apache.sysds.test.functions.federated.fedplanning;

import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;
import org.apache.sysds.api.DMLScript;
import org.apache.sysds.common.Types;
import org.apache.sysds.runtime.meta.MatrixCharacteristics;
import org.apache.sysds.test.AutomatedTestBase;
import org.apache.sysds.test.TestConfiguration;
import org.apache.sysds.test.TestUtils;
import org.junit.Test;

import java.io.File;
import java.util.Arrays;

import static org.junit.Assert.fail;

public class FederatedPCAPlanningTest extends AutomatedTestBase {
	private static final Log LOG = LogFactory.getLog(FederatedPCAPlanningTest.class.getName());

	private final static String TEST_DIR = "functions/privacy/fedplanning/";
	private final static String TEST_NAME = "FederatedPCAPlanningTest";
	private final static String TEST_CLASS_DIR = TEST_DIR + FederatedPCAPlanningTest.class.getSimpleName() + "/";
	private static File TEST_CONF_FILE;

	private final static int blocksize = 1024;
	public final int rows = 1000;
	public final int cols = 100;

	@Override
	public void setUp() {
		TestUtils.clearAssertionInformation();
		addTestConfiguration(TEST_NAME, new TestConfiguration(TEST_CLASS_DIR, TEST_NAME, new String[] { "PC", "V" }));
	}

	@Test
	public void runPCAFOUTTest() {
		String[] expectedHeavyHitters = new String[] { "fed_fedinit", "fed_mean", "fed_tsmm", "fed_-", "fed_eigen" };
		setTestConf("SystemDS-config-fout.xml");
		loadAndRunTest(expectedHeavyHitters, TEST_NAME);
	}

	@Test
	public void runPCAHeuristicTest() {
		String[] expectedHeavyHitters = new String[] { "fed_fedinit", "fed_mean" };
		setTestConf("SystemDS-config-heuristic.xml");
		loadAndRunTest(expectedHeavyHitters, TEST_NAME);
	}

	@Test
	public void runPCACostBasedTestPrivate() {
		String[] expectedHeavyHitters = new String[] {};
		setTestConf("SystemDS-config-cost-based.xml");
		loadAndRunTestWithPrivacy(expectedHeavyHitters, TEST_NAME, "private");
	}

	@Test
	public void runPCACostBasedTestPrivateAggregate() {
		String[] expectedHeavyHitters = new String[] {};
		setTestConf("SystemDS-config-cost-based.xml");
		loadAndRunTestWithPrivacy(expectedHeavyHitters, TEST_NAME, "private-aggregate");
	}

	@Test
	public void runPCACostBasedTestPublic() {
		String[] expectedHeavyHitters = new String[] {};
		setTestConf("SystemDS-config-cost-based.xml");
		loadAndRunTestWithPrivacy(expectedHeavyHitters, TEST_NAME, "public");
	}

	@Test
	public void runRuntimeTest() {
		String[] expectedHeavyHitters = new String[] {};
		TEST_CONF_FILE = new File("src/test/config/SystemDS-config.xml");
		loadAndRunTest(expectedHeavyHitters, TEST_NAME);
	}

	private void setTestConf(String test_conf) {
		TEST_CONF_FILE = new File(SCRIPT_DIR + TEST_DIR, test_conf);
	}

	@Override
	protected File getConfigTemplateFile() {
		LOG.info("This test case overrides default configuration with " + TEST_CONF_FILE.getPath());
		return TEST_CONF_FILE;
	}

	private void writeInputMatrices() {
		writeStandardRowFedMatrix("X1", 65);
		writeStandardRowFedMatrix("X2", 75);
		writeStandardRowFedMatrix("X3", 85);
		writeStandardRowFedMatrix("X4", 95);
	}

	private void writeInputMatricesWithPrivacyConstraints(String privacyConstraints) {
		writeStandardRowFedMatrix("X1", 65, privacyConstraints);
		writeStandardRowFedMatrix("X2", 75, privacyConstraints);
		writeStandardRowFedMatrix("X3", 85, privacyConstraints);
		writeStandardRowFedMatrix("X4", 95, privacyConstraints);
	}

	private void writeStandardMatrix(String matrixName, int numRows, double[][] matrix) {
		MatrixCharacteristics mc = new MatrixCharacteristics(numRows, cols, blocksize, (long) numRows * cols);
		writeInputMatrixWithMTD(matrixName, matrix, false, mc);
	}

	private void writeStandardMatrix(String matrixName, int numRows, double[][] matrix, String privacyConstraints) {
		MatrixCharacteristics mc = new MatrixCharacteristics(numRows, cols, blocksize, (long) numRows * cols);
		writeInputMatrixWithMTD(matrixName, matrix, false, mc, privacyConstraints);
	}

	private void writeStandardMatrix(String matrixName, long seed, int numRows) {
		double[][] matrix = getRandomMatrix(numRows, cols, 0, 1, 1, seed);
		writeStandardMatrix(matrixName, numRows, matrix);
	}

	private void writeStandardMatrix(String matrixName, long seed, int numRows, String privacyConstraints) {
		double[][] matrix = getRandomMatrix(numRows, cols, 0, 1, 1, seed);
		writeStandardMatrix(matrixName, numRows, matrix, privacyConstraints);
	}

	private void writeStandardRowFedMatrix(String matrixName, long seed) {
		int quarterRows = rows / 4;
		writeStandardMatrix(matrixName, seed, quarterRows);
	}

	private void writeStandardRowFedMatrix(String matrixName, long seed, String privacyConstraints) {
		int quarterRows = rows / 4;
		writeStandardMatrix(matrixName, seed, quarterRows, privacyConstraints);
	}

	private void loadAndRunTest(String[] expectedHeavyHitters, String testName) {

		boolean sparkConfigOld = DMLScript.USE_LOCAL_SPARK_CONFIG;
		Types.ExecMode platformOld = rtplatform;
		rtplatform = Types.ExecMode.SINGLE_NODE;

		Thread t1 = null, t2 = null, t3 = null, t4 = null;

		try {
			getAndLoadTestConfiguration(testName);
			String HOME = SCRIPT_DIR + TEST_DIR;

			writeInputMatrices();

			int port1 = getRandomAvailablePort();
			int port2 = getRandomAvailablePort();
			int port3 = getRandomAvailablePort();
			int port4 = getRandomAvailablePort();
			t1 = startLocalFedWorkerThread(port1, FED_WORKER_WAIT_S);
			t2 = startLocalFedWorkerThread(port2);
			t3 = startLocalFedWorkerThread(port3);
			t4 = startLocalFedWorkerThread(port4);

			// Run actual dml script with federated matrix
			fullDMLScriptName = HOME + testName + ".dml";
			programArgs = new String[] { "-stats", "-nvargs",
					"X1=" + TestUtils.federatedAddress(port1, input("X1")),
					"X2=" + TestUtils.federatedAddress(port2, input("X2")),
					"X3=" + TestUtils.federatedAddress(port3, input("X3")),
					"X4=" + TestUtils.federatedAddress(port4, input("X4")),
					"r=" + rows, "c=" + cols, "K=2", "PC=" + output("PC"), "V=" + output("V") };
			runTest(true, false, null, -1);

			// Run reference dml script with normal matrix
			fullDMLScriptName = HOME + testName + "Reference.dml";
			programArgs = new String[] { "-nvargs", "X1=" + input("X1"), "X2=" + input("X2"),
					"X3=" + input("X3"), "X4=" + input("X4"), "PC=" + expected("PC"), "V=" + expected("V") };
			runTest(true, false, null, -1);

			// compare via files
			compareResults(1e-9);
			if (!heavyHittersContainsAllString(expectedHeavyHitters))
				fail("The following expected heavy hitters are missing: "
						+ Arrays.toString(missingHeavyHitters(expectedHeavyHitters)));
		} finally {
			TestUtils.shutdownThreads(t1, t2, t3, t4);
			rtplatform = platformOld;
			DMLScript.USE_LOCAL_SPARK_CONFIG = sparkConfigOld;
		}
	}

	private void loadAndRunTestWithPrivacy(String[] expectedHeavyHitters, String testName, String privacyConstraints) {
		boolean sparkConfigOld = DMLScript.USE_LOCAL_SPARK_CONFIG;
		Types.ExecMode platformOld = rtplatform;
		rtplatform = Types.ExecMode.SINGLE_NODE;

		Thread t1 = null, t2 = null, t3 = null, t4 = null;

		try {
			getAndLoadTestConfiguration(testName);
			String HOME = SCRIPT_DIR + TEST_DIR;

			writeInputMatricesWithPrivacyConstraints(privacyConstraints);

			int port1 = getRandomAvailablePort();
			int port2 = getRandomAvailablePort();
			int port3 = getRandomAvailablePort();
			int port4 = getRandomAvailablePort();
			t1 = startLocalFedWorkerThread(port1, FED_WORKER_WAIT_S);
			t2 = startLocalFedWorkerThread(port2);
			t3 = startLocalFedWorkerThread(port3);
			t4 = startLocalFedWorkerThread(port4);

			// Run actual dml script with federated matrix
			fullDMLScriptName = HOME + testName + ".dml";
			programArgs = new String[] { "-stats", "-nvargs",
					"X1=" + TestUtils.federatedAddress(port1, input("X1")),
					"X2=" + TestUtils.federatedAddress(port2, input("X2")),
					"X3=" + TestUtils.federatedAddress(port3, input("X3")),
					"X4=" + TestUtils.federatedAddress(port4, input("X4")),
					"r=" + rows, "c=" + cols, "K=2", "PC=" + output("PC"), "V=" + output("V") };
			runTest(true, false, null, -1);

			// Run reference dml script with normal matrix
			fullDMLScriptName = HOME + testName + "Reference.dml";
			programArgs = new String[] { "-nvargs", "X1=" + input("X1"), "X2=" + input("X2"),
					"X3=" + input("X3"), "X4=" + input("X4"), "PC=" + expected("PC"), "V=" + expected("V") };
			runTest(true, false, null, -1);

			// compare via files
			compareResults(1e-9);
			if (!heavyHittersContainsAllString(expectedHeavyHitters))
				fail("The following expected heavy hitters are missing: "
						+ Arrays.toString(missingHeavyHitters(expectedHeavyHitters)));
		} finally {
			TestUtils.shutdownThreads(t1, t2, t3, t4);
			rtplatform = platformOld;
			DMLScript.USE_LOCAL_SPARK_CONFIG = sparkConfigOld;
		}
	}
}