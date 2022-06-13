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

package org.apache.sysds.test.functions.privacy.fedplanning;

import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;
import org.apache.sysds.api.DMLScript;
import org.apache.sysds.common.Types;
import org.apache.sysds.runtime.meta.MatrixCharacteristics;
import org.apache.sysds.runtime.privacy.PrivacyConstraint;
import org.apache.sysds.test.AutomatedTestBase;
import org.apache.sysds.test.TestConfiguration;
import org.apache.sysds.test.TestUtils;
import org.junit.Ignore;
import org.junit.Test;

import java.io.File;
import java.util.Arrays;

import static org.junit.Assert.fail;

@net.jcip.annotations.NotThreadSafe
public class FederatedL2SVMPlanningTest extends AutomatedTestBase {
	private static final Log LOG = LogFactory.getLog(FederatedL2SVMPlanningTest.class.getName());

	private final static String TEST_DIR = "functions/privacy/fedplanning/";
	private final static String TEST_NAME = "FederatedL2SVMPlanningTest";
	private final static String TEST_NAME_2 = "FederatedL2SVMFunctionPlanningTest";
	private final static String TEST_CLASS_DIR = TEST_DIR + FederatedL2SVMPlanningTest.class.getSimpleName() + "/";
	private static File TEST_CONF_FILE;

	private final static int blocksize = 1024;
	public final int rows = 1000;
	public final int cols = 100;

	@Override
	public void setUp() {
		TestUtils.clearAssertionInformation();
		addTestConfiguration(TEST_NAME, new TestConfiguration(TEST_CLASS_DIR, TEST_NAME, new String[] {"Z"}));
		addTestConfiguration(TEST_NAME_2, new TestConfiguration(TEST_CLASS_DIR, TEST_NAME_2, new String[] {"Z"}));
	}

	@Test
	public void runL2SVMFOUTTest(){
		String[] expectedHeavyHitters = new String[]{ "fed_fedinit", "fed_ba+*", "fed_tak+*", "fed_+*",
			"fed_max", "fed_1-*", "fed_tsmm", "fed_>"};
		setTestConf("SystemDS-config-fout.xml");
		loadAndRunTest(expectedHeavyHitters, TEST_NAME);
	}

	@Test
	public void runL2SVMHeuristicTest(){
		String[] expectedHeavyHitters = new String[]{ "fed_fedinit", "fed_ba+*"};
		setTestConf("SystemDS-config-heuristic.xml");
		loadAndRunTest(expectedHeavyHitters, TEST_NAME);
	}

	@Test
	public void runL2SVMCostBasedTest(){
		String[] expectedHeavyHitters = new String[]{ "fed_fedinit", "fed_ba+*", "fed_tak+*", "fed_+*",
			"fed_max", "fed_1-*", "fed_tsmm", "fed_>"};
		setTestConf("SystemDS-config-cost-based.xml");
		loadAndRunTest(expectedHeavyHitters, TEST_NAME);
	}

	@Test
	@Ignore
	public void runL2SVMFunctionFOUTTest(){
		String[] expectedHeavyHitters = new String[]{ "fed_fedinit", "fed_ba+*", "fed_tak+*", "fed_+*",
			"fed_max", "fed_1-*", "fed_tsmm", "fed_>"};
		setTestConf("SystemDS-config-fout.xml");
		loadAndRunTest(expectedHeavyHitters, TEST_NAME_2);
	}

	@Test
	@Ignore
	public void runL2SVMFunctionHeuristicTest(){
		String[] expectedHeavyHitters = new String[]{ "fed_fedinit", "fed_ba+*"};
		setTestConf("SystemDS-config-heuristic.xml");
		loadAndRunTest(expectedHeavyHitters, TEST_NAME_2);
	}

	@Test
	public void runL2SVMFunctionCostBasedTest(){
		String[] expectedHeavyHitters = new String[]{ "fed_fedinit", "fed_ba+*", "fed_tak+*", "fed_+*",
			"fed_max", "fed_1-*", "fed_tsmm", "fed_>"};
		setTestConf("SystemDS-config-cost-based.xml");
		loadAndRunTest(expectedHeavyHitters, TEST_NAME_2);
	}

	private void setTestConf(String test_conf){
		TEST_CONF_FILE = new File(SCRIPT_DIR + TEST_DIR, test_conf);
	}

	private void writeInputMatrices(){
		writeStandardRowFedMatrix("X1", 65, null);
		writeStandardRowFedMatrix("X2", 75, null);
		writeBinaryVector("Y", 44, null);
	}

	private void writeBinaryVector(String matrixName, long seed, PrivacyConstraint privacyConstraint){
		double[][] matrix = getRandomMatrix(rows, 1, -1, 1, 1, seed);
		for(int i = 0; i < rows; i++)
			matrix[i][0] = (matrix[i][0] > 0) ? 1 : -1;
		MatrixCharacteristics mc = new MatrixCharacteristics(rows, 1, blocksize, rows);
		writeInputMatrixWithMTD(matrixName, matrix, false, mc, privacyConstraint);
	}

	@SuppressWarnings("unused")
	private void writeStandardMatrix(String matrixName, long seed, PrivacyConstraint privacyConstraint){
		writeStandardMatrix(matrixName, seed, rows, privacyConstraint);
	}

	private void writeStandardMatrix(String matrixName, long seed, int numRows, PrivacyConstraint privacyConstraint){
		double[][] matrix = getRandomMatrix(numRows, cols, 0, 1, 1, seed);
		writeStandardMatrix(matrixName, numRows, privacyConstraint, matrix);
	}

	private void writeStandardMatrix(String matrixName, int numRows, PrivacyConstraint privacyConstraint, double[][] matrix){
		MatrixCharacteristics mc = new MatrixCharacteristics(numRows, cols, blocksize, (long) numRows * cols);
		writeInputMatrixWithMTD(matrixName, matrix, false, mc, privacyConstraint);
	}

	private void writeStandardRowFedMatrix(String matrixName, long seed, PrivacyConstraint privacyConstraint){
		int halfRows = rows/2;
		writeStandardMatrix(matrixName, seed, halfRows, privacyConstraint);
	}

	private void loadAndRunTest(String[] expectedHeavyHitters, String testName){

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
			programArgs = new String[] { "-stats", "-explain", "hops", "-nvargs",
				"X1=" + TestUtils.federatedAddress(port1, input("X1")),
				"X2=" + TestUtils.federatedAddress(port2, input("X2")),
				"Y=" + input("Y"), "r=" + rows, "c=" + cols, "Z=" + output("Z")};
			runTest(true, false, null, -1);

			// Run reference dml script with normal matrix
			fullDMLScriptName = HOME + testName + "Reference.dml";
			programArgs = new String[] {"-nvargs", "X1=" + input("X1"), "X2=" + input("X2"),
				"Y=" + input("Y"), "Z=" + expected("Z")};
			runTest(true, false, null, -1);

			// compare via files
			compareResults(1e-9);
			if (!heavyHittersContainsAllString(expectedHeavyHitters))
				fail("The following expected heavy hitters are missing: "
					+ Arrays.toString(missingHeavyHitters(expectedHeavyHitters)));
		}
		finally {
			TestUtils.shutdownThreads(t1, t2);
			rtplatform = platformOld;
			DMLScript.USE_LOCAL_SPARK_CONFIG = sparkConfigOld;
		}
	}

	/**
	 * Override default configuration with custom test configuration to ensure
	 * scratch space and local temporary directory locations are also updated.
	 */
	@Override
	protected File getConfigTemplateFile() {
		// Instrumentation in this test's output log to show custom configuration file used for template.
		LOG.info("This test case overrides default configuration with " + TEST_CONF_FILE.getPath());
		return TEST_CONF_FILE;
	}

}
