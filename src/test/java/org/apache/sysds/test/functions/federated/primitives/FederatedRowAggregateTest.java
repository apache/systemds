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

package org.apache.sysds.test.functions.federated.primitives;

import java.util.Arrays;
import java.util.Collection;

import org.apache.sysds.api.DMLScript;
import org.apache.sysds.common.Types.ExecMode;
import org.apache.sysds.runtime.meta.MatrixCharacteristics;
import org.apache.sysds.runtime.util.HDFSTool;
import org.apache.sysds.test.AutomatedTestBase;
import org.apache.sysds.test.TestConfiguration;
import org.apache.sysds.test.TestUtils;
import org.junit.Assert;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.Parameterized;

@RunWith(value = Parameterized.class)
@net.jcip.annotations.NotThreadSafe
public class FederatedRowAggregateTest extends AutomatedTestBase {
	private final static String TEST_NAME5 = "FederatedRowSumTest";
	private final static String TEST_NAME6 = "FederatedRowMeanTest";
	private final static String TEST_NAME7 = "FederatedRowMaxTest";
	private final static String TEST_NAME8 = "FederatedRowMinTest";
	private final static String TEST_NAME9 = "FederatedRowVarTest";
	private final static String TEST_NAME10 = "FederatedRowProdTest";
	private final static String TEST_NAME11 = "FederatedMMTest";

	private final static String TEST_DIR = "functions/federated/aggregate/";
	private static final String TEST_CLASS_DIR = TEST_DIR + FederatedRowAggregateTest.class.getSimpleName() + "/";

	private final static int blocksize = 1024;
	@Parameterized.Parameter()
	public int rows;
	@Parameterized.Parameter(1)
	public int cols;
	@Parameterized.Parameter(2)
	public boolean rowPartitioned;

	@Parameterized.Parameters
	public static Collection<Object[]> data() {
		return Arrays.asList(
			new Object[][] {
				{10, 1000, false},
				{1000, 40, true},
		});
	}

	private enum OpType {
		SUM, MEAN, MAX, MIN, VAR, PROD, MM
	}

	@Override
	public void setUp() {
		TestUtils.clearAssertionInformation();
		addTestConfiguration(TEST_NAME5, new TestConfiguration(TEST_CLASS_DIR, TEST_NAME5, new String[] {"S"}));
		addTestConfiguration(TEST_NAME6, new TestConfiguration(TEST_CLASS_DIR, TEST_NAME6, new String[] {"S"}));
		addTestConfiguration(TEST_NAME7, new TestConfiguration(TEST_CLASS_DIR, TEST_NAME7, new String[] {"S"}));
		addTestConfiguration(TEST_NAME8, new TestConfiguration(TEST_CLASS_DIR, TEST_NAME8, new String[] {"S"}));
		addTestConfiguration(TEST_NAME9, new TestConfiguration(TEST_CLASS_DIR, TEST_NAME9, new String[] {"S"}));
		addTestConfiguration(TEST_NAME10, new TestConfiguration(TEST_CLASS_DIR, TEST_NAME10, new String[] {"S"}));
		addTestConfiguration(TEST_NAME11, new TestConfiguration(TEST_CLASS_DIR, TEST_NAME11, new String[] {"S"}));
	}

	@Test
	public void testRowSumDenseMatrixCP() {
		runAggregateOperationTest(OpType.SUM, ExecMode.SINGLE_NODE);
	}

	@Test
	public void testRowMeanDenseMatrixCP() {
		runAggregateOperationTest(OpType.MEAN, ExecMode.SINGLE_NODE);
	}

	@Test
	public void testRowMaxDenseMatrixCP() {
		runAggregateOperationTest(OpType.MAX, ExecMode.SINGLE_NODE);
	}

	@Test
	public void testRowMinDenseMatrixCP() {
		runAggregateOperationTest(OpType.MIN, ExecMode.SINGLE_NODE);
	}

	@Test
	public void testRowVarDenseMatrixCP() {
		runAggregateOperationTest(OpType.VAR, ExecMode.SINGLE_NODE);
	}

	@Test
	public void testRowProdDenseMatrixCP() {
		runAggregateOperationTest(OpType.PROD, ExecMode.SINGLE_NODE);
	}

	@Test
	public void testMMDenseMatrixCP() {
		runAggregateOperationTest(OpType.MM, ExecMode.SINGLE_NODE);
	}

	@Test
	public void testRowSumDenseMatrixSP() {
		runAggregateOperationTest(OpType.SUM, ExecMode.SPARK);
	}

	@Test
	public void testRowMeanDenseMatrixSP() {
		runAggregateOperationTest(OpType.MEAN, ExecMode.SPARK);
	}

	@Test
	public void testRowMaxDenseMatrixSP() {
		runAggregateOperationTest(OpType.MAX, ExecMode.SPARK);
	}

	@Test
	public void testRowMinDenseMatrixSP() {
		runAggregateOperationTest(OpType.MIN, ExecMode.SPARK);
	}

	@Test
	public void testRowVarDenseMatrixSP() {
		runAggregateOperationTest(OpType.VAR, ExecMode.SPARK);
	}

	@Test
	public void testRowProdDenseMatrixSP() {
		runAggregateOperationTest(OpType.PROD, ExecMode.SPARK);
	}

	@Test
	public void testMMDenseMatrixSP() {
		runAggregateOperationTest(OpType.MM, ExecMode.SPARK);
	}

	private void runAggregateOperationTest(OpType type, ExecMode execMode) {
		boolean sparkConfigOld = DMLScript.USE_LOCAL_SPARK_CONFIG;
		ExecMode platformOld = rtplatform;

		if(rtplatform == ExecMode.SPARK)
			DMLScript.USE_LOCAL_SPARK_CONFIG = true;

		String TEST_NAME = null;
		switch(type) {
			case SUM:
				TEST_NAME = TEST_NAME5;
				break;
			case MEAN:
				TEST_NAME = TEST_NAME6;
				break;
			case MAX:
				TEST_NAME = TEST_NAME7;
				break;
			case MIN:
				TEST_NAME = TEST_NAME8;
				break;
			case VAR:
				TEST_NAME = TEST_NAME9;
				break;
			case PROD:
				TEST_NAME = TEST_NAME10;
				break;
			case MM:
				TEST_NAME = TEST_NAME11;
				break;
		}

		getAndLoadTestConfiguration(TEST_NAME);
		String HOME = SCRIPT_DIR + TEST_DIR;

		// write input matrices
		int r = rows;
		int c = cols / 4;
		if(rowPartitioned) {
			r = rows / 4;
			c = cols;
		}

		double[][] X1 = getRandomMatrix(r, c, 3, 3, 1, 3);
		double[][] X2 = getRandomMatrix(r, c, 3, 3, 1, 7);
		double[][] X3 = getRandomMatrix(r, c, 3, 3, 1, 8);
		double[][] X4 = getRandomMatrix(r, c, 3, 3, 1, 9);

		MatrixCharacteristics mc = new MatrixCharacteristics(r, c, blocksize, r * c);
		writeInputMatrixWithMTD("X1", X1, false, mc);
		writeInputMatrixWithMTD("X2", X2, false, mc);
		writeInputMatrixWithMTD("X3", X3, false, mc);
		writeInputMatrixWithMTD("X4", X4, false, mc);

		rtplatform = execMode;
		if(rtplatform == ExecMode.SPARK) {
			DMLScript.USE_LOCAL_SPARK_CONFIG = true;
		}
		TestConfiguration config = availableTestConfigurations.get(TEST_NAME);
		loadTestConfiguration(config);

		// empty script name because we don't execute any script, just start the worker
		fullDMLScriptName = "";
		int port1 = getRandomAvailablePort();
		int port2 = getRandomAvailablePort();
		int port3 = getRandomAvailablePort();
		int port4 = getRandomAvailablePort();
		Thread t1 = startLocalFedWorkerThread(port1, FED_WORKER_WAIT_S);
		Thread t2 = startLocalFedWorkerThread(port2, FED_WORKER_WAIT_S);
		Thread t3 = startLocalFedWorkerThread(port3, FED_WORKER_WAIT_S);
		Thread t4 = startLocalFedWorkerThread(port4);

		// Run reference dml script with normal matrix
		fullDMLScriptName = HOME + TEST_NAME + "Reference.dml";
		programArgs = new String[] {"-stats", "100", "-args", input("X1"), input("X2"), input("X3"), input("X4"),
			expected("S"), Boolean.toString(rowPartitioned).toUpperCase()};
		runTest(true, false, null, -1);

		// Run actual dml script with federated matrix

		fullDMLScriptName = HOME + TEST_NAME + ".dml";
		programArgs = new String[] {"-stats", "100", "-nvargs",
			"in_X1=" + TestUtils.federatedAddress(port1, input("X1")),
			"in_X2=" + TestUtils.federatedAddress(port2, input("X2")),
			"in_X3=" + TestUtils.federatedAddress(port3, input("X3")),
			"in_X4=" + TestUtils.federatedAddress(port4, input("X4")), "rows=" + rows, "cols=" + cols,
			"rP=" + Boolean.toString(rowPartitioned).toUpperCase(), "out_S=" + output("S")};

		runTest(true, false, null, -1);

		// compare via files
		compareResults(type == FederatedRowAggregateTest.OpType.VAR ? 1e-2 : 1e-9, "Stat-DML1", "Stat-DML2");

		String fedInst = "fed_uar";

		switch(type) {
			case SUM:
				Assert.assertTrue(heavyHittersContainsString(fedInst.concat("k+")));
				break;
			case MEAN:
				Assert.assertTrue(heavyHittersContainsString(fedInst.concat("mean")));
				break;
			case MAX:
				Assert.assertTrue(heavyHittersContainsString(fedInst.concat("max")));
				break;
			case MIN:
				Assert.assertTrue(heavyHittersContainsString(fedInst.concat("min")));
				break;
			case VAR:
				Assert.assertTrue(heavyHittersContainsString(fedInst.concat("var")));
				break;
			case PROD:
				Assert.assertTrue(heavyHittersContainsString(fedInst.concat("*")));
				break;
			case MM:
				Assert.assertTrue(heavyHittersContainsString(rtplatform == ExecMode.SPARK ? "fed_mapmm" : "fed_ba+*", 1, 2));
				break;
		}

		// check that federated input files are still existing
		Assert.assertTrue(HDFSTool.existsFileOnHDFS(input("X1")));
		Assert.assertTrue(HDFSTool.existsFileOnHDFS(input("X2")));
		Assert.assertTrue(HDFSTool.existsFileOnHDFS(input("X3")));
		Assert.assertTrue(HDFSTool.existsFileOnHDFS(input("X4")));

		TestUtils.shutdownThreads(t1, t2, t3, t4);

		rtplatform = platformOld;
		DMLScript.USE_LOCAL_SPARK_CONFIG = sparkConfigOld;

	}
}
