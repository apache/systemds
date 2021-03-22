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

import org.apache.sysds.hops.OptimizerUtils;
import org.apache.sysds.runtime.privacy.PrivacyConstraint;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.Parameterized;
import org.apache.sysds.api.DMLScript;
import org.apache.sysds.common.Types;
import org.apache.sysds.runtime.meta.MatrixCharacteristics;
import org.apache.sysds.test.AutomatedTestBase;
import org.apache.sysds.test.TestConfiguration;
import org.apache.sysds.test.TestUtils;

import java.util.Arrays;
import java.util.Collection;

@RunWith(value = Parameterized.class)
@net.jcip.annotations.NotThreadSafe
public class FederatedMultiplyPlanningTest extends AutomatedTestBase {
	private final static String TEST_DIR = "functions/privacy/";
	private final static String TEST_NAME = "FederatedMultiplyPlanningTest";
	private final static String TEST_NAME_2 = "FederatedMultiplyPlanningTest2";
	private final static String TEST_NAME_3 = "FederatedMultiplyPlanningTest3";
	private final static String TEST_CLASS_DIR = TEST_DIR + FederatedMultiplyPlanningTest.class.getSimpleName() + "/";

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
		addTestConfiguration(TEST_NAME_3, new TestConfiguration(TEST_CLASS_DIR, TEST_NAME_3, new String[] {"Z.scalar"}));
	}

	@Parameterized.Parameters
	public static Collection<Object[]> data() {
		// rows have to be even and > 1
		return Arrays.asList(new Object[][] {
			{100, 10}
		});
	}

	@Test
	public void federatedMultiplyCP() {
		OptimizerUtils.FEDERATED_COMPILATION = true;
		federatedTwoMatricesSingleNodeTest(TEST_NAME);
	}

	@Test
	public void federatedRowSum(){
		OptimizerUtils.FEDERATED_COMPILATION = true;
		federatedTwoMatricesSingleNodeTest(TEST_NAME_2);
	}

	@Test
	public void federatedTernarySequence(){
		OptimizerUtils.FEDERATED_COMPILATION = true;
		federatedTwoMatricesSingleNodeTest(TEST_NAME_3);
	}

	private void writeStandardMatrix(String matrixName, long seed){
		int halfRows = rows/2;
		double[][] matrix = getRandomMatrix(halfRows, cols, 0, 1, 1, seed);
		writeInputMatrixWithMTD(matrixName, matrix, false,
			new MatrixCharacteristics(halfRows, cols, blocksize, halfRows * cols),
			new PrivacyConstraint(PrivacyConstraint.PrivacyLevel.PrivateAggregation));
	}

	public void federatedTwoMatricesSingleNodeTest(String testName){
		federatedTwoMatricesTest(Types.ExecMode.SINGLE_NODE, testName);
	}

	public void federatedTwoMatricesTest(Types.ExecMode execMode, String testName) {
		boolean sparkConfigOld = DMLScript.USE_LOCAL_SPARK_CONFIG;
		Types.ExecMode platformOld = rtplatform;
		rtplatform = execMode;
		if(rtplatform == Types.ExecMode.SPARK) {
			DMLScript.USE_LOCAL_SPARK_CONFIG = true;
		}

		getAndLoadTestConfiguration(testName);
		String HOME = SCRIPT_DIR + TEST_DIR;

		// Write input matrices
		writeStandardMatrix("X1", 42);
		writeStandardMatrix("X2", 1340);
		writeStandardMatrix("Y1", 44);
		writeStandardMatrix("Y2", 21);

		int port1 = getRandomAvailablePort();
		int port2 = getRandomAvailablePort();
		Thread t1 = startLocalFedWorkerThread(port1, FED_WORKER_WAIT_S);
		Thread t2 = startLocalFedWorkerThread(port2);

		TestConfiguration config = availableTestConfigurations.get(testName);
		loadTestConfiguration(config);

		// Run actual dml script with federated matrix
		fullDMLScriptName = HOME + testName + ".dml";
		programArgs = new String[] {"-explain", "-nvargs", "X1=" + TestUtils.federatedAddress(port1, input("X1")),
			"X2=" + TestUtils.federatedAddress(port2, input("X2")),
			"Y1=" + TestUtils.federatedAddress(port1, input("Y1")),
			"Y2=" + TestUtils.federatedAddress(port2, input("Y2")), "r=" + rows, "c=" + cols, "Z=" + output("Z")};
		runTest(true, false, null, -1);

		OptimizerUtils.FEDERATED_COMPILATION = false;

		// Run reference dml script with normal matrix
		fullDMLScriptName = HOME + testName + "Reference.dml";
		programArgs = new String[] {"-nvargs", "X1=" + input("X1"), "X2=" + input("X2"), "Y1=" + input("Y1"),
			"Y2=" + input("Y2"), "Z=" + expected("Z")};
		runTest(true, false, null, -1);

		// compare via files
		compareResults(1e-9);
		heavyHittersContainsString("fed_*", "fed_ba+*");

		TestUtils.shutdownThreads(t1, t2);

		rtplatform = platformOld;
		DMLScript.USE_LOCAL_SPARK_CONFIG = sparkConfigOld;
	}

	public void federatedThreeMatricesTest(Types.ExecMode execMode, String testName) {
		boolean sparkConfigOld = DMLScript.USE_LOCAL_SPARK_CONFIG;
		Types.ExecMode platformOld = rtplatform;
		rtplatform = execMode;
		if(rtplatform == Types.ExecMode.SPARK) {
			DMLScript.USE_LOCAL_SPARK_CONFIG = true;
		}

		getAndLoadTestConfiguration(testName);
		String HOME = SCRIPT_DIR + TEST_DIR;

		// Write input matrices
		writeStandardMatrix("X1", 42);
		writeStandardMatrix("X2", 1340);
		writeStandardMatrix("Y1", 44);
		writeStandardMatrix("Y2", 21);
		writeStandardMatrix("W1", 55);

		int port1 = getRandomAvailablePort();
		int port2 = getRandomAvailablePort();
		Thread t1 = startLocalFedWorkerThread(port1, FED_WORKER_WAIT_S);
		Thread t2 = startLocalFedWorkerThread(port2);

		TestConfiguration config = availableTestConfigurations.get(testName);
		loadTestConfiguration(config);

		// Run actual dml script with federated matrix
		fullDMLScriptName = HOME + testName + ".dml";
		programArgs = new String[] {"-explain", "-nvargs", "X1=" + TestUtils.federatedAddress(port1, input("X1")),
			"X2=" + TestUtils.federatedAddress(port2, input("X2")),
			"Y1=" + TestUtils.federatedAddress(port1, input("Y1")),
			"Y2=" + TestUtils.federatedAddress(port2, input("Y2")), "r=" + rows, "c=" + cols, "Z=" + output("Z")};
		runTest(true, false, null, -1);

		OptimizerUtils.FEDERATED_COMPILATION = false;

		// Run reference dml script with normal matrix
		fullDMLScriptName = HOME + testName + "Reference.dml";
		programArgs = new String[] {"-nvargs", "X1=" + input("X1"), "X2=" + input("X2"), "Y1=" + input("Y1"),
			"Y2=" + input("Y2"), "Z=" + expected("Z")};
		runTest(true, false, null, -1);

		// compare via files
		compareResults(1e-9);
		heavyHittersContainsString("fed_*", "fed_ba+*");

		TestUtils.shutdownThreads(t1, t2);

		rtplatform = platformOld;
		DMLScript.USE_LOCAL_SPARK_CONFIG = sparkConfigOld;
	}
}

