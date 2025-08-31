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

package org.apache.sysds.test.functions.federated.primitives.part2;

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
public class FederatedQuantileTest extends AutomatedTestBase {

	private final static String TEST_DIR = "functions/federated/quantile/";
	private final static String TEST_NAME1 = "FederatedQuantileTest";
	private final static String TEST_NAME2 = "FederatedMedianTest";
	private final static String TEST_NAME3 = "FederatedIQRTest";
	private final static String TEST_NAME4 = "FederatedQuantilesTest";
	private final static String TEST_CLASS_DIR = TEST_DIR + FederatedQuantileTest.class.getSimpleName() + "/";

	private final static int blocksize = 1024;
	@Parameterized.Parameter()
	public int rows;
	@Parameterized.Parameter(1)
	public int cols;
	@Parameterized.Parameter(2)
	public boolean rowPartitioned;

	@Parameterized.Parameters
	public static Collection<Object[]> data() {
		return Arrays.asList(new Object[][] {
			// {1000, 1, false},
			{128, 1, true}});
	}

	@Override
	public void setUp() {
		TestUtils.clearAssertionInformation();
		addTestConfiguration(TEST_NAME1, new TestConfiguration(TEST_CLASS_DIR, TEST_NAME1, new String[] {"S.scalar"}));
		addTestConfiguration(TEST_NAME2, new TestConfiguration(TEST_CLASS_DIR, TEST_NAME2, new String[] {"S.scalar"}));
		addTestConfiguration(TEST_NAME3, new TestConfiguration(TEST_CLASS_DIR, TEST_NAME3, new String[] {"S.scalar"}));
		addTestConfiguration(TEST_NAME4, new TestConfiguration(TEST_CLASS_DIR, TEST_NAME4, new String[] {"S"}));
	}

	@Test
	@Ignore
	public void federatedQuantile1CP() {
		federatedQuartile(Types.ExecMode.SINGLE_NODE, TEST_NAME1, 0.25);
	}

	@Test
	@Ignore
	public void federatedQuantile2CP() {
		federatedQuartile(Types.ExecMode.SINGLE_NODE, TEST_NAME1, 0.5);
	}

	@Test
	@Ignore
	public void federatedQuantile3CP() {
		federatedQuartile(Types.ExecMode.SINGLE_NODE, TEST_NAME1, 0.75);
	}

	@Test
	public void federatedMedianCP() {
		federatedQuartile(Types.ExecMode.SINGLE_NODE, TEST_NAME2, -1);
	}

	@Test
	public void federatedIQRCP() {
		federatedQuartile(Types.ExecMode.SINGLE_NODE, TEST_NAME3, -1);
	}

	@Test
	public void federatedQuantilesCP() {
		federatedQuartile(Types.ExecMode.SINGLE_NODE, TEST_NAME4, -1);
	}

	@Test
	@Ignore
	public void federatedQuantile1SP() {
		federatedQuartile(Types.ExecMode.SPARK, TEST_NAME1, 0.25);
	}

	@Test
	@Ignore
	public void federatedQuantile2SP() {
		federatedQuartile(Types.ExecMode.SPARK, TEST_NAME1, 0.5);
	}

	@Test
	@Ignore
	public void federatedQuantile3SP() {
		federatedQuartile(Types.ExecMode.SPARK, TEST_NAME1, 0.75);
	}

	@Test
	public void federatedMedianSP() {
		federatedQuartile(Types.ExecMode.SPARK, TEST_NAME2, -1);
	}

	@Test
	public void federatedIQRSP() {
		federatedQuartile(Types.ExecMode.SPARK, TEST_NAME3, -1);
	}

	@Test
	public void federatedQuantilesSP() {
		federatedQuartile(Types.ExecMode.SPARK, TEST_NAME4, -1);
	}

	public void federatedQuartile(Types.ExecMode execMode, String TEST_NAME, double p) {
		boolean sparkConfigOld = DMLScript.USE_LOCAL_SPARK_CONFIG;
		Types.ExecMode platformOld = rtplatform;

		getAndLoadTestConfiguration(TEST_NAME);
		String HOME = SCRIPT_DIR + TEST_DIR;

		double[][] X1, X2, X3, X4;
		int port1, port2, port3, port4;
		Process t1 = null, t2 = null, t3 = null, t4 = null;
		String[] programArgs1, programArgs2;
		try {
			if(rowPartitioned) {
				X1 = getRandomMatrix(rows / 4, cols, 1, 12, 1, 3);
				X2 = getRandomMatrix(rows / 4, cols, 1, 12, 1, 7);
				X3 = getRandomMatrix(rows / 4, cols, 1, 12, 1, 8);
				X4 = getRandomMatrix(rows / 4, cols, 1, 12, 1, 9);

				MatrixCharacteristics mc1 = new MatrixCharacteristics(rows / 4, 1, blocksize, rows);
				writeInputMatrixWithMTD("X1", X1, false, mc1);
				writeInputMatrixWithMTD("X2", X2, false, mc1);
				writeInputMatrixWithMTD("X3", X3, false, mc1);
				writeInputMatrixWithMTD("X4", X4, false, mc1);

				port1 = getRandomAvailablePort();
				port2 = getRandomAvailablePort();
				port3 = getRandomAvailablePort();
				port4 = getRandomAvailablePort();
				t1 = startLocalFedWorker(port1, FED_WORKER_WAIT_S);
				t2 = startLocalFedWorker(port2, FED_WORKER_WAIT_S);
				t3 = startLocalFedWorker(port3, FED_WORKER_WAIT_S);
				t4 = startLocalFedWorker(port4);

				if(!isAlive(t1, t2, t3, t4))
					throw new RuntimeException("Failed starting federated worker");

				programArgs1 = new String[] {"-explain", "-stats", "100", "-args", String.valueOf(p), expected("S"),
					Boolean.toString(rowPartitioned).toUpperCase(), input("X1"), input("X2"), input("X3"), input("X4")};
				programArgs2 = new String[] {"-explain", "-stats", "100", "-nvargs",
					"in_X1=" + TestUtils.federatedAddress(port1, input("X1")),
					"in_X2=" + TestUtils.federatedAddress(port2, input("X2")),
					"in_X3=" + TestUtils.federatedAddress(port3, input("X3")),
					"in_X4=" + TestUtils.federatedAddress(port4, input("X4")), "rows=" + rows, "cols=" + cols,
					"rP=" + Boolean.toString(rowPartitioned).toUpperCase(), "p=" + String.valueOf(p),
					"out_S=" + output("S")};
			}
			else {
				X1 = getRandomMatrix(rows, 1, 1, 12, 1, 3);
				MatrixCharacteristics mc = new MatrixCharacteristics(rows, 1, blocksize, rows);
				writeInputMatrixWithMTD("X1", X1, false, mc);

				port1 = getRandomAvailablePort();
				t1 = startLocalFedWorker(port1);

				if(!isAlive(t1))
					throw new RuntimeException("Failed starting federated worker");

				programArgs1 = new String[] {"-explain", "-stats", "100", "-args", String.valueOf(p), expected("S"),
					Boolean.toString(rowPartitioned).toUpperCase(), input("X1"), input("X1"), input("X1"), input("X1")};
				programArgs2 = new String[] {"-explain", "-stats", "100", "-nvargs",
					"in_X1=" + TestUtils.federatedAddress(port1, input("X1")),
					"in_X2=" + TestUtils.federatedAddress(port1, input("X1")),
					"in_X3=" + TestUtils.federatedAddress(port1, input("X1")),
					"in_X4=" + TestUtils.federatedAddress(port1, input("X1")), "rows=" + rows, "cols=" + cols,
					"p=" + String.valueOf(p), "out_S=" + output("S"),
					"rP=" + Boolean.toString(rowPartitioned).toUpperCase()};
			}

			// empty script name because we don't execute any script, just start the worker
			fullDMLScriptName = "";

			// we need the reference file to not be written to hdfs, so we get the correct format
			rtplatform = Types.ExecMode.SINGLE_NODE;
			// Run reference dml script with normal matrix for Row/Col
			fullDMLScriptName = HOME + TEST_NAME + "Reference.dml";

			programArgs = programArgs1;
			runTest(true, false, null, -1);

			// reference file should not be written to hdfs, so we set platform here
			rtplatform = execMode;
			if(rtplatform == Types.ExecMode.SPARK) {
				DMLScript.USE_LOCAL_SPARK_CONFIG = true;
			}
			TestConfiguration config = availableTestConfigurations.get(TEST_NAME);
			loadTestConfiguration(config);

			fullDMLScriptName = HOME + TEST_NAME + ".dml";
			programArgs = programArgs2;
			runTest(true, false, null, -1);

			// compare all sums via files
			compareResults(1e-9);
			Assert.assertTrue(heavyHittersContainsString("fed_qsort"));
			Assert.assertTrue(heavyHittersContainsString("fed_qpick"));

			// check that federated input files are still existing
			Assert.assertTrue(HDFSTool.existsFileOnHDFS(input("X1")));

		}
		finally {
			if(t1 != null) {

				TestUtils.shutdownThreads(t1);
				if(rowPartitioned && t2 != null) {
					Assert.assertTrue(HDFSTool.existsFileOnHDFS(input("X2")));
					Assert.assertTrue(HDFSTool.existsFileOnHDFS(input("X3")));
					Assert.assertTrue(HDFSTool.existsFileOnHDFS(input("X4")));

					TestUtils.shutdownThreads(t2, t3, t4);
				}
			}

			rtplatform = platformOld;
			DMLScript.USE_LOCAL_SPARK_CONFIG = sparkConfigOld;
		}
	}
}
