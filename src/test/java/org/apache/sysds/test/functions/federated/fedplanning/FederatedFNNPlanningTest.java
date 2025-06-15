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

public class FederatedFNNPlanningTest extends AutomatedTestBase {
	private static final Log LOG = LogFactory.getLog(FederatedFNNPlanningTest.class.getName());

	private final static String TEST_DIR = "functions/privacy/fedplanning/";
	private final static String TEST_NAME = "FederatedFNNPlanningTest";
	private final static String TEST_CLASS_DIR = TEST_DIR + FederatedFNNPlanningTest.class.getSimpleName() + "/";
	private static File TEST_CONF_FILE;

	private final static int blocksize = 1024;
	public final int rows = 1000;
	public final int cols = 100;
	public final int classes = 5;

	@Override
	public void setUp() {
		TestUtils.clearAssertionInformation();
		addTestConfiguration(TEST_NAME, new TestConfiguration(TEST_CLASS_DIR, TEST_NAME, new String[] { "model" }));
	}

	@Test
	public void runFNNFOUTTest() {
		String[] expectedHeavyHitters = new String[] { "fed_fedinit", "fed_ba+*", "fed_relu", "fed_dropout" };
		setTestConf("SystemDS-config-fout.xml");
		loadAndRunTest(expectedHeavyHitters, TEST_NAME);
	}

	@Test
	public void runFNNHeuristicTest() {
		String[] expectedHeavyHitters = new String[] { "fed_fedinit", "fed_ba+*" };
		setTestConf("SystemDS-config-heuristic.xml");
		loadAndRunTest(expectedHeavyHitters, TEST_NAME);
	}

	@Test
	public void runFNNCostBasedTestPrivate() {
		String[] expectedHeavyHitters = new String[] {};
		setTestConf("SystemDS-config-cost-based.xml");
		loadAndRunTestWithPrivacy(expectedHeavyHitters, TEST_NAME, "private");
	}

	@Test
	public void runFNNCostBasedTestPrivateAggregate() {
		String[] expectedHeavyHitters = new String[] {};
		setTestConf("SystemDS-config-cost-based.xml");
		loadAndRunTestWithPrivacy(expectedHeavyHitters, TEST_NAME, "private-aggregate");
	}

	@Test
	public void runFNNCostBasedTestPublic() {
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
		writeClassificationLabels("Y", 85);
	}

	private void writeInputMatricesWithPrivacyConstraints(String privacyConstraints) {
		writeStandardRowFedMatrix("X1", 65, privacyConstraints);
		writeStandardRowFedMatrix("X2", 75, privacyConstraints);
		writeClassificationLabels("Y", 85, privacyConstraints);
	}

	private void writeStandardMatrix(String matrixName, int numRows, double[][] matrix) {
		MatrixCharacteristics mc = new MatrixCharacteristics(numRows, cols, blocksize, (long) numRows * cols);
		writeInputMatrixWithMTD(matrixName, matrix, false, mc);
	}

	private void writeStandardMatrix(String matrixName, int numRows, double[][] matrix, String privacyConstraints) {
		MatrixCharacteristics mc = new MatrixCharacteristics(numRows, cols, blocksize, (long) numRows * cols);
		writeInputMatrixWithMTD(matrixName, matrix, false, mc, privacyConstraints);
	}

	private void writeClassificationLabels(String matrixName, long seed) {
		double[][] labels = getRandomMatrix(rows, classes, 0, 1, 1, seed);
		// Convert to one-hot encoded classification labels
		for(int i = 0; i < rows; i++) {
			int maxIdx = 0;
			for(int j = 1; j < classes; j++) {
				if(labels[i][j] > labels[i][maxIdx]) {
					maxIdx = j;
				}
			}
			for(int j = 0; j < classes; j++) {
				labels[i][j] = (j == maxIdx) ? 1.0 : 0.0;
			}
		}
		MatrixCharacteristics mc = new MatrixCharacteristics(rows, classes, blocksize, rows * classes);
		writeInputMatrixWithMTD(matrixName, labels, false, mc);
	}

	private void writeClassificationLabels(String matrixName, long seed, String privacyConstraints) {
		double[][] labels = getRandomMatrix(rows, classes, 0, 1, 1, seed);
		// Convert to one-hot encoded classification labels
		for(int i = 0; i < rows; i++) {
			int maxIdx = 0;
			for(int j = 1; j < classes; j++) {
				if(labels[i][j] > labels[i][maxIdx]) {
					maxIdx = j;
				}
			}
			for(int j = 0; j < classes; j++) {
				labels[i][j] = (j == maxIdx) ? 1.0 : 0.0;
			}
		}
		MatrixCharacteristics mc = new MatrixCharacteristics(rows, classes, blocksize, rows * classes);
		writeInputMatrixWithMTD(matrixName, labels, false, mc, privacyConstraints);
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
		int halfRows = rows / 2;
		writeStandardMatrix(matrixName, seed, halfRows);
	}

	private void writeStandardRowFedMatrix(String matrixName, long seed, String privacyConstraints) {
		int halfRows = rows / 2;
		writeStandardMatrix(matrixName, seed, halfRows, privacyConstraints);
	}

	private void loadAndRunTest(String[] expectedHeavyHitters, String testName) {

		boolean sparkConfigOld = DMLScript.USE_LOCAL_SPARK_CONFIG;
		Types.ExecMode platformOld = rtplatform;
		rtplatform = Types.ExecMode.SINGLE_NODE;

		Thread t1 = null, t2 = null;

		try {
			getAndLoadTestConfiguration(testName);
			String HOME = SCRIPT_DIR + TEST_DIR;

			writeInputMatrices();

			int port1 = getRandomAvailablePort();
			int port2 = getRandomAvailablePort();
			t1 = startLocalFedWorkerThread(port1, FED_WORKER_WAIT_S);
			t2 = startLocalFedWorkerThread(port2);

			// Run actual dml script with federated matrix
			fullDMLScriptName = HOME + testName + ".dml";
			programArgs = new String[] { "-stats", "-nvargs",
					"X1=" + TestUtils.federatedAddress(port1, input("X1")),
					"X2=" + TestUtils.federatedAddress(port2, input("X2")),
					"Y=" + input("Y"), "r=" + rows, "c=" + cols, "classes=" + classes,
					"epochs=3", "batch_size=64", "model=" + output("model") };
			runTest(true, false, null, -1);

			// Run reference dml script with normal matrix
			fullDMLScriptName = HOME + testName + "Reference.dml";
			programArgs = new String[] { "-nvargs", "X1=" + input("X1"), "X2=" + input("X2"),
					"Y=" + input("Y"), "model=" + expected("model") };
			runTest(true, false, null, -1);

			// compare via files
			compareResults(1e-9);
			if (!heavyHittersContainsAllString(expectedHeavyHitters))
				fail("The following expected heavy hitters are missing: "
						+ Arrays.toString(missingHeavyHitters(expectedHeavyHitters)));
		} finally {
			TestUtils.shutdownThreads(t1, t2);
			rtplatform = platformOld;
			DMLScript.USE_LOCAL_SPARK_CONFIG = sparkConfigOld;
		}
	}

	private void loadAndRunTestWithPrivacy(String[] expectedHeavyHitters, String testName, String privacyConstraints) {
		boolean sparkConfigOld = DMLScript.USE_LOCAL_SPARK_CONFIG;
		Types.ExecMode platformOld = rtplatform;
		rtplatform = Types.ExecMode.SINGLE_NODE;

		Thread t1 = null, t2 = null;

		try {
			getAndLoadTestConfiguration(testName);
			String HOME = SCRIPT_DIR + TEST_DIR;

			writeInputMatricesWithPrivacyConstraints(privacyConstraints);

			int port1 = getRandomAvailablePort();
			int port2 = getRandomAvailablePort();
			t1 = startLocalFedWorkerThread(port1, FED_WORKER_WAIT_S);
			t2 = startLocalFedWorkerThread(port2);

			// Run actual dml script with federated matrix
			fullDMLScriptName = HOME + testName + ".dml";
			programArgs = new String[] { "-stats", "-nvargs",
					"X1=" + TestUtils.federatedAddress(port1, input("X1")),
					"X2=" + TestUtils.federatedAddress(port2, input("X2")),
					"Y=" + input("Y"), "r=" + rows, "c=" + cols, "classes=" + classes,
					"epochs=3", "batch_size=64", "model=" + output("model") };
			runTest(true, false, null, -1);

			// Run reference dml script with normal matrix
			fullDMLScriptName = HOME + testName + "Reference.dml";
			programArgs = new String[] { "-nvargs", "X1=" + input("X1"), "X2=" + input("X2"),
					"Y=" + input("Y"), "model=" + expected("model") };
			runTest(true, false, null, -1);

			// compare via files
			compareResults(1e-9);
			if (!heavyHittersContainsAllString(expectedHeavyHitters))
				fail("The following expected heavy hitters are missing: "
						+ Arrays.toString(missingHeavyHitters(expectedHeavyHitters)));
		} finally {
			TestUtils.shutdownThreads(t1, t2);
			rtplatform = platformOld;
			DMLScript.USE_LOCAL_SPARK_CONFIG = sparkConfigOld;
		}
	}
}