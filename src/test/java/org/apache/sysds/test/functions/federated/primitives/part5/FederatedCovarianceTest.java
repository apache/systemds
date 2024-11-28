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

package org.apache.sysds.test.functions.federated.primitives.part5;

import java.util.Arrays;
import java.util.Collection;

import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;
import org.apache.sysds.api.DMLScript;
import org.apache.sysds.common.Types.ExecMode;
import org.apache.sysds.runtime.meta.MatrixCharacteristics;
import org.apache.sysds.test.AutomatedTestBase;
import org.apache.sysds.test.TestConfiguration;
import org.apache.sysds.test.TestUtils;
import org.junit.Assert;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.Parameterized;

@RunWith(value = Parameterized.class)
@net.jcip.annotations.NotThreadSafe
public class FederatedCovarianceTest extends AutomatedTestBase {

	protected static final Log LOG = LogFactory.getLog(FederatedCovarianceTest.class.getName());

	private final static String TEST_NAME1 = "FederatedCovarianceTest";
	private final static String TEST_NAME2 = "FederatedCovarianceAlignedTest";
	private final static String TEST_NAME3 = "FederatedCovarianceWeightedTest";
	private final static String TEST_NAME4 = "FederatedCovarianceAlignedWeightedTest";
	private final static String TEST_NAME5 = "FederatedCovarianceAllAlignedWeightedTest";
	private final static String TEST_DIR = "functions/federated/";
	private static final String TEST_CLASS_DIR = TEST_DIR + FederatedCovarianceTest.class.getSimpleName() + "/";

	private final static int blocksize = 1000;
	@Parameterized.Parameter
	public int rows;
	@Parameterized.Parameter(1)
	public int cols;

	@Parameterized.Parameters
	public static Collection<Object[]> data() {
		return Arrays.asList(new Object[][] {
			{120, 1},
			{1100, 1},
		});
	}

	@Override
	public void setUp() {
		TestUtils.clearAssertionInformation();
		addTestConfiguration(TEST_NAME1, new TestConfiguration(TEST_CLASS_DIR, TEST_NAME1, new String[] {"S.scalar"}));
		addTestConfiguration(TEST_NAME2, new TestConfiguration(TEST_CLASS_DIR, TEST_NAME2, new String[] {"S.scalar"}));
		addTestConfiguration(TEST_NAME3, new TestConfiguration(TEST_CLASS_DIR, TEST_NAME3, new String[] {"S.scalar"}));
		addTestConfiguration(TEST_NAME4, new TestConfiguration(TEST_CLASS_DIR, TEST_NAME4, new String[] {"S.scalar"}));
		addTestConfiguration(TEST_NAME5, new TestConfiguration(TEST_CLASS_DIR, TEST_NAME5, new String[] {"S.scalar"}));
	}

	@Test
	public void testCovCP() {
		runCovarianceTest(ExecMode.SINGLE_NODE, false);
	}

	@Test
	public void testAlignedCovCP() {
		runCovarianceTest(ExecMode.SINGLE_NODE, true);
	}

	@Test
	public void testCovarianceWeightedCP() {
		runWeightedCovarianceTest(ExecMode.SINGLE_NODE, false, false);
	}

	@Test
	public void testAlignedCovarianceWeightedCP() {
		runWeightedCovarianceTest(ExecMode.SINGLE_NODE, true, false);
	}

	@Test
	public void testAllAlignedCovarianceWeightedCP() {
		runWeightedCovarianceTest(ExecMode.SINGLE_NODE, true, true);
	}

	private void runCovarianceTest(ExecMode execMode, boolean alignedFedInput) {
		boolean sparkConfigOld = DMLScript.USE_LOCAL_SPARK_CONFIG;
		ExecMode platformOld = rtplatform;

		if(rtplatform == ExecMode.SPARK)
			DMLScript.USE_LOCAL_SPARK_CONFIG = true;

		String TEST_NAME = alignedFedInput ? TEST_NAME2 : TEST_NAME1;

		getAndLoadTestConfiguration(TEST_NAME);
		String HOME = SCRIPT_DIR + TEST_DIR;

		// write input matrices
		int r = rows / 4;
		int c = cols;

		// empty script name because we don't execute any script, just start the worker
		fullDMLScriptName = "";

		double[][] X1 = getRandomMatrix(r, c, 1, 5, 1, 3);
		double[][] X2 = getRandomMatrix(r, c, 1, 5, 1, 7);
		double[][] X3 = getRandomMatrix(r, c, 1, 5, 1, 8);
		double[][] X4 = getRandomMatrix(r, c, 1, 5, 1, 9);

		MatrixCharacteristics mc = new MatrixCharacteristics(r, c, blocksize, r * c);
		writeInputMatrixWithMTD("X1", X1, false, mc);
		writeInputMatrixWithMTD("X2", X2, false, mc);
		writeInputMatrixWithMTD("X3", X3, false, mc);
		writeInputMatrixWithMTD("X4", X4, false, mc);

		// empty script name because we don't execute any script, just start the worker
		fullDMLScriptName = "";
		int port1 = getRandomAvailablePort();
		int port2 = getRandomAvailablePort();
		int port3 = getRandomAvailablePort();
		int port4 = getRandomAvailablePort();
		Process t1 = startLocalFedWorker(port1, FED_WORKER_WAIT_S);
		Process t2 = startLocalFedWorker(port2, FED_WORKER_WAIT_S);
		Process t3 = startLocalFedWorker(port3, FED_WORKER_WAIT_S);
		Process t4 = startLocalFedWorker(port4, FED_WORKER_WAIT);

		try {
			if(!isAlive(t1, t2, t3, t4))
				throw new RuntimeException("Failed starting federated worker");

			setExecMode(execMode);
			TestConfiguration config = availableTestConfigurations.get(TEST_NAME);
			loadTestConfiguration(config);

			if(alignedFedInput) {
				double[][] Y1 = getRandomMatrix(r, c, 1, 5, 1, 3);
				double[][] Y2 = getRandomMatrix(r, c, 1, 5, 1, 7);
				double[][] Y3 = getRandomMatrix(r, c, 1, 5, 1, 8);
				double[][] Y4 = getRandomMatrix(r, c, 1, 5, 1, 9);

				writeInputMatrixWithMTD("Y1", Y1, false, mc);
				writeInputMatrixWithMTD("Y2", Y2, false, mc);
				writeInputMatrixWithMTD("Y3", Y3, false, mc);
				writeInputMatrixWithMTD("Y4", Y4, false, mc);

				// Run reference dml script with normal matrix
				fullDMLScriptName = HOME + TEST_NAME + "Reference.dml";
				programArgs = new String[] {"-stats", "100", "-args", input("X1"), input("X2"), input("X3"), input("X4"),
					input("Y1"), input("Y2"), input("Y3"), input("Y4"), expected("S")};
				runTest(null);

				fullDMLScriptName = HOME + TEST_NAME + ".dml";
				programArgs = new String[] {"-stats", "100", "-nvargs",
					"in_X1=" + TestUtils.federatedAddress(port1, input("X1")),
					"in_X2=" + TestUtils.federatedAddress(port2, input("X2")),
					"in_X3=" + TestUtils.federatedAddress(port3, input("X3")),
					"in_X4=" + TestUtils.federatedAddress(port4, input("X4")),
					"in_Y1=" + TestUtils.federatedAddress(port1, input("Y1")),
					"in_Y2=" + TestUtils.federatedAddress(port2, input("Y2")),
					"in_Y3=" + TestUtils.federatedAddress(port3, input("Y3")),
					"in_Y4=" + TestUtils.federatedAddress(port4, input("Y4")), "rows=" + rows, "cols=" + cols,
					"out_S=" + output("S")};
				runTest(null);

			}
			else {
				double[][] Y = getRandomMatrix(rows, c, 1, 5, 1, 3);
				writeInputMatrixWithMTD("Y", Y, false, new MatrixCharacteristics(rows, cols, blocksize, r * c));

				// Run reference dml script with normal matrix
				fullDMLScriptName = HOME + TEST_NAME + "Reference.dml";
				programArgs = new String[] {"-stats", "100", "-args", input("X1"), input("X2"), input("X3"), input("X4"),
					input("Y"), expected("S"),};
				runTest(null);
				fullDMLScriptName = HOME + TEST_NAME + ".dml";
				programArgs = new String[] {"-stats", "100", "-nvargs",
					"in_X1=" + TestUtils.federatedAddress(port1, input("X1")),
					"in_X2=" + TestUtils.federatedAddress(port2, input("X2")),
					"in_X3=" + TestUtils.federatedAddress(port3, input("X3")),
					"in_X4=" + TestUtils.federatedAddress(port4, input("X4")), "Y=" + input("Y"), "rows=" + rows,
					"cols=" + cols, "out_S=" + output("S")};
				runTest(null);
			}

			// compare via files
			compareResults(1e-2);
			Assert.assertTrue(heavyHittersContainsString("fed_cov"));

		}
		finally {
			TestUtils.shutdownThreads(t1, t2, t3, t4);
			rtplatform = platformOld;
			DMLScript.USE_LOCAL_SPARK_CONFIG = sparkConfigOld;
		}
	}

	private void runWeightedCovarianceTest(ExecMode execMode, boolean alignedInput, boolean alignedWeights) {
		boolean sparkConfigOld = DMLScript.USE_LOCAL_SPARK_CONFIG;
		ExecMode platformOld = setExecMode(execMode);
		String TEST_NAME = !alignedInput ? TEST_NAME3 : (!alignedWeights ? TEST_NAME4 : TEST_NAME5);
		getAndLoadTestConfiguration(TEST_NAME);

		String HOME = SCRIPT_DIR + TEST_DIR;
		
		int r = rows / 4;
		int c = cols;

		fullDMLScriptName = "";

		// Create 4 random 5x1 matrices
		double[][] X1 = getRandomMatrix(r, c, 1, 5, 1, 3);
		double[][] X2 = getRandomMatrix(r, c, 1, 5, 1, 7);
		double[][] X3 = getRandomMatrix(r, c, 1, 5, 1, 8);
		double[][] X4 = getRandomMatrix(r, c, 1, 5, 1, 9);

		// Create a 20x1 weights matrix 
		double[][] W = getRandomMatrix(rows, c, 0, 1, 1, 3);

		MatrixCharacteristics mc = new MatrixCharacteristics(r, c, blocksize, r * c);
		writeInputMatrixWithMTD("X1", X1, false, mc);
		writeInputMatrixWithMTD("X2", X2, false, mc);
		writeInputMatrixWithMTD("X3", X3, false, mc);
		writeInputMatrixWithMTD("X4", X4, false, mc);

		writeInputMatrixWithMTD("W", W, false, new MatrixCharacteristics(rows, cols, blocksize, r * c));

		// empty script name because we don't execute any script, just start the worker
		fullDMLScriptName = "";
		int port1 = getRandomAvailablePort();
		int port2 = getRandomAvailablePort();
		int port3 = getRandomAvailablePort();
		int port4 = getRandomAvailablePort();

		Process t1 = startLocalFedWorker(port1, FED_WORKER_WAIT_S);
		Process t2 = startLocalFedWorker(port2, FED_WORKER_WAIT_S);
		Process t3 = startLocalFedWorker(port3, FED_WORKER_WAIT_S);
		Process t4 = startLocalFedWorker(port4, FED_WORKER_WAIT);

		try {
			if(!isAlive(t1, t2, t3, t4))
				throw new RuntimeException("Failed starting federated worker");
			TestConfiguration config = availableTestConfigurations.get(TEST_NAME);
			loadTestConfiguration(config);

			if (alignedInput) {
				// Create 4 random 5x1 matrices
				double[][] Y1 = getRandomMatrix(r, c, 1, 5, 1, 3);
				double[][] Y2 = getRandomMatrix(r, c, 1, 5, 1, 7);
				double[][] Y3 = getRandomMatrix(r, c, 1, 5, 1, 8);
				double[][] Y4 = getRandomMatrix(r, c, 1, 5, 1, 9);

				writeInputMatrixWithMTD("Y1", Y1, false, mc);
				writeInputMatrixWithMTD("Y2", Y2, false, mc);
				writeInputMatrixWithMTD("Y3", Y3, false, mc);
				writeInputMatrixWithMTD("Y4", Y4, false, mc);

				if (!alignedWeights) {
					// Run reference dml script with a normal matrix
					fullDMLScriptName = HOME + TEST_NAME + "Reference.dml";
					programArgs = new String[] { "-stats", "100", "-args",
						input("X1"), input("X2"), input("X3"), input("X4"),
						input("Y1"), input("Y2"), input("Y3"), input("Y4"),
						input("W"), expected("S")
					};
					runTest(null);
					
					// Run the dml script with federated matrices
					fullDMLScriptName = HOME + TEST_NAME + ".dml";
					programArgs = new String[] {"-stats", "100", "-nvargs",
						"in_X1=" + TestUtils.federatedAddress(port1, input("X1")),
						"in_Y1=" + TestUtils.federatedAddress(port1, input("Y1")),
						"in_X2=" + TestUtils.federatedAddress(port2, input("X2")),
						"in_Y2=" + TestUtils.federatedAddress(port2, input("Y2")),
						"in_X3=" + TestUtils.federatedAddress(port3, input("X3")),
						"in_Y3=" + TestUtils.federatedAddress(port3, input("Y3")),
						"in_X4=" + TestUtils.federatedAddress(port4, input("X4")),
						"in_Y4=" + TestUtils.federatedAddress(port4, input("Y4")),
						"in_W1=" + input("W"), "rows=" + rows, "cols=" + cols, "out_S=" + output("S")};
					runTest(null);
				}
				else {
					double[][] W1 = getRandomMatrix(r, c, 0, 1, 1, 3);
					double[][] W2 = getRandomMatrix(r, c, 0, 1, 1, 7);
					double[][] W3 = getRandomMatrix(r, c, 0, 1, 1, 8);
					double[][] W4 = getRandomMatrix(r, c, 0, 1, 1, 9);

					writeInputMatrixWithMTD("W1", W1, false, mc);
					writeInputMatrixWithMTD("W2", W2, false, mc);
					writeInputMatrixWithMTD("W3", W3, false, mc);
					writeInputMatrixWithMTD("W4", W4, false, mc);

					// Run reference dml script with a normal matrix
					fullDMLScriptName = HOME + TEST_NAME + "Reference.dml";
					programArgs = new String[] {"-stats", "100", "-args",
						input("X1"), input("X2"), input("X3"), input("X4"),
						input("Y1"), input("Y2"), input("Y3"), input("Y4"),
						input("W1"), input("W2"), input("W3"), input("W4"), expected("S")
					};
					runTest(null);

					// Run the dml script with federated matrices and weights
					fullDMLScriptName = HOME + TEST_NAME + ".dml";
					programArgs = new String[] {"-stats", "100", "-nvargs",
						"in_X1=" + TestUtils.federatedAddress(port1, input("X1")),
						"in_Y1=" + TestUtils.federatedAddress(port1, input("Y1")),
						"in_W1=" + TestUtils.federatedAddress(port1, input("W1")),
						"in_X2=" + TestUtils.federatedAddress(port2, input("X2")),
						"in_Y2=" + TestUtils.federatedAddress(port2, input("Y2")),
						"in_W2=" + TestUtils.federatedAddress(port2, input("W2")),
						"in_X3=" + TestUtils.federatedAddress(port3, input("X3")),
						"in_Y3=" + TestUtils.federatedAddress(port3, input("Y3")),
						"in_W3=" + TestUtils.federatedAddress(port3, input("W3")),
						"in_X4=" + TestUtils.federatedAddress(port4, input("X4")),
						"in_Y4=" + TestUtils.federatedAddress(port4, input("Y4")),
						"in_W4=" + TestUtils.federatedAddress(port4, input("W4")),
						"rows=" + rows, "cols=" + cols, "out_S=" + output("S")};
					runTest(null);
				}
				
			}
			else {
				// Create a random 20x1 input matrix
				double[][] Y = getRandomMatrix(rows, c, 1, 5, 1, 3);
				writeInputMatrixWithMTD("Y", Y, false, new MatrixCharacteristics(rows, cols, blocksize, r * c));

				// Run reference dml script with a normal matrix
				fullDMLScriptName = HOME + TEST_NAME + "Reference.dml";
				programArgs = new String[] {"-stats", "100", "-args",
					input("X1"), input("X2"), input("X3"), input("X4"),
					input("Y"), input("W"), expected("S")
				};
				runTest(null);

				// Run the dml script with a federated matrix
				fullDMLScriptName = HOME + TEST_NAME + ".dml";
				programArgs = new String[] {"-stats", "100", "-nvargs",
					"in_X1=" + TestUtils.federatedAddress(port1, input("X1")),
					"in_X2=" + TestUtils.federatedAddress(port2, input("X2")),
					"in_X3=" + TestUtils.federatedAddress(port3, input("X3")),
					"in_X4=" + TestUtils.federatedAddress(port4, input("X4")),
					"in_W1=" + input("W"), "Y=" + input("Y"),
					"rows=" + rows, "cols=" + cols, "out_S=" + output("S")};
				runTest(null);
			}

			// compare via files
			compareResults(1e-2);
			Assert.assertTrue(heavyHittersContainsString("fed_cov"));

		}
		finally {
			TestUtils.shutdownThreads(t1, t2, t3, t4);
			rtplatform = platformOld;
			DMLScript.USE_LOCAL_SPARK_CONFIG = sparkConfigOld;
		}
	}
}
