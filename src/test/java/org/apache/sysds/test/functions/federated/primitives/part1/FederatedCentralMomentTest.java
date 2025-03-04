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

package org.apache.sysds.test.functions.federated.primitives.part1;

import java.util.Arrays;
import java.util.Collection;

import org.apache.sysds.api.DMLScript;
import org.apache.sysds.common.Types;
import org.apache.sysds.runtime.meta.MatrixCharacteristics;
import org.apache.sysds.runtime.util.HDFSTool;
import org.apache.sysds.test.AutomatedTestBase;
import org.apache.sysds.test.TestConfiguration;
import org.apache.sysds.test.TestUtils;
import org.junit.Assert;
import org.junit.Ignore;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.Parameterized;

@RunWith(value = Parameterized.class)
@net.jcip.annotations.NotThreadSafe
public class FederatedCentralMomentTest extends AutomatedTestBase {

	private final static String TEST_DIR = "functions/federated/";
	private final static String TEST_NAME1 = "FederatedCentralMomentTest";
	private final static String TEST_NAME2 = "FederatedCentralMomentWeightedTest";
	private final static String TEST_CLASS_DIR = TEST_DIR + FederatedCentralMomentTest.class.getSimpleName() + "/";

	private final static int blocksize = 1024;
	@Parameterized.Parameter()
	public int rows;

	@Parameterized.Parameter(1)
	public int cols;

	@Parameterized.Parameter(2)
	public int k;

	@Parameterized.Parameters
	public static Collection<Object[]> data() {
		return Arrays.asList(new Object[][] {{20, 1, 2}});
	}

	@Override
	public void setUp() {
		TestUtils.clearAssertionInformation();
		addTestConfiguration(TEST_NAME1, new TestConfiguration(TEST_CLASS_DIR, TEST_NAME1, new String[] {"S.scalar"}));
		addTestConfiguration(TEST_NAME2, new TestConfiguration(TEST_CLASS_DIR, TEST_NAME2, new String[] {"S.scalar"}));
	}

	@Test
	public void federatedCentralMomentCP() {
		federatedCentralMoment(Types.ExecMode.SINGLE_NODE, false);
	}

	@Test
	public void federatedCentralMomentWeightedCP() {
		federatedCentralMoment(Types.ExecMode.SINGLE_NODE, true);
	}

	@Test
	public void federatedCentralMomentSP() {
		federatedCentralMoment(Types.ExecMode.SPARK, false);
	}

	// The test fails due to an error while executing rmvar instruction after cm calculation
	// The CacheStatus of the weights variable is READ hence it can't be modified
	// In this test the input matrix is federated and weights are read from file
	@Ignore
	@Test
	public void federatedCentralMomentWeightedSP() {
		federatedCentralMoment(Types.ExecMode.SPARK, true);
	}

	public void federatedCentralMoment(Types.ExecMode execMode, boolean isWeighted) {
		boolean sparkConfigOld = DMLScript.USE_LOCAL_SPARK_CONFIG;
		Types.ExecMode platformOld = rtplatform;

		String TEST_NAME = isWeighted ? TEST_NAME2 : TEST_NAME1;
		getAndLoadTestConfiguration(TEST_NAME);
		String HOME = SCRIPT_DIR + TEST_DIR;

		int r = rows / 4;
		int c = cols;

		double[][] X1 = getRandomMatrix(r, c, 1, 5, 1, 3);
		double[][] X2 = getRandomMatrix(r, c, 1, 5, 1, 7);
		double[][] X3 = getRandomMatrix(r, c, 1, 5, 1, 8);
		double[][] X4 = getRandomMatrix(r, c, 1, 5, 1, 9);

		MatrixCharacteristics mc = new MatrixCharacteristics(r, 1, blocksize, r);
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
		Process t4 = startLocalFedWorker(port4, FED_WORKER_WAIT + 1000);

		
		try {
			if(!isAlive(t1, t2, t3, t4))
				throw new RuntimeException("Failed starting federated worker");

			// reference file should not be written to hdfs, so we set platform here
			rtplatform = execMode;
			if(rtplatform == Types.ExecMode.SPARK) {
				DMLScript.USE_LOCAL_SPARK_CONFIG = true;
			}
			TestConfiguration config = availableTestConfigurations.get(TEST_NAME);
			loadTestConfiguration(config);
			if (isWeighted) {
				double[][] W1 = getRandomMatrix(r, c, 0, 1, 1, 3);

				writeInputMatrixWithMTD("W1", W1, false, mc);

				// Run reference dml script with normal matrix
				fullDMLScriptName = HOME + TEST_NAME + "Reference.dml";
				programArgs = new String[] {"-stats", "100", "-args", input("X1"), input("X2"), input("X3"), input("X4"),
						input("W1"), expected("S"), "" + k};
				runTest(null);

				fullDMLScriptName = HOME + TEST_NAME + ".dml";
				programArgs = new String[] {"-stats", "100", "-nvargs",
						"in_X1=" + TestUtils.federatedAddress(port1, input("X1")),
						"in_X2=" + TestUtils.federatedAddress(port2, input("X2")),
						"in_X3=" + TestUtils.federatedAddress(port3, input("X3")),
						"in_X4=" + TestUtils.federatedAddress(port4, input("X4")),
						"in_W1=" + input("W1"),
						"rows=" + rows, "cols=" + cols, "k=" + k,
						"out_S=" + output("S")};
				runTest(null);
			}
			else {
				// Run reference dml script with normal matrix for Row/Col
				fullDMLScriptName = HOME + TEST_NAME + "Reference.dml";
				programArgs = new String[]{"-stats", "100", "-args", input("X1"), input("X2"), input("X3"), input("X4"),
						expected("S"), String.valueOf(k)};
				runTest(null);


				fullDMLScriptName = HOME + TEST_NAME + ".dml";
				programArgs = new String[]{"-stats", "100", "-nvargs",
						"in_X1=" + TestUtils.federatedAddress(port1, input("X1")),
						"in_X2=" + TestUtils.federatedAddress(port2, input("X2")),
						"in_X3=" + TestUtils.federatedAddress(port3, input("X3")),
						"in_X4=" + TestUtils.federatedAddress(port4, input("X4")), "rows=" + rows, "cols=" + 1,
						"out_S=" + output("S"), "k=" + k};
				runTest(null);
			}
			// compare all sums via files
			compareResults(0.01);

			Assert.assertTrue(heavyHittersContainsString("fed_cm"));

			// check that federated input files are still existing
			Assert.assertTrue(HDFSTool.existsFileOnHDFS(input("X1")));
			Assert.assertTrue(HDFSTool.existsFileOnHDFS(input("X2")));
			Assert.assertTrue(HDFSTool.existsFileOnHDFS(input("X3")));
			Assert.assertTrue(HDFSTool.existsFileOnHDFS(input("X4")));
		}
		finally {

			TestUtils.shutdownThreads(t1, t2, t3, t4);
			rtplatform = platformOld;
			DMLScript.USE_LOCAL_SPARK_CONFIG = sparkConfigOld;
		}
	}
}
