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
import org.apache.sysds.lops.LopProperties.ExecType;
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
public class FederatedFullAggregateTest extends AutomatedTestBase {
	private final static String TEST_NAME1 = "FederatedSumTest";
	private final static String TEST_NAME2 = "FederatedMeanTest";
	private final static String TEST_NAME3 = "FederatedMaxTest";
	private final static String TEST_NAME4 = "FederatedMinTest";

	private final static String TEST_DIR = "functions/federated/aggregate/";
	private static final String TEST_CLASS_DIR = TEST_DIR + FederatedFullAggregateTest.class.getSimpleName() + "/";

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
			new Object[][] {{10, 1000, false}, {100, 4, false}, {36, 1000, true}, {1000, 10, true}, {4, 100, true}});
	}

	private enum OpType {
		SUM, MEAN, MAX, MIN
	}

	@Override
	public void setUp() {
		TestUtils.clearAssertionInformation();
		addTestConfiguration(TEST_NAME1, new TestConfiguration(TEST_CLASS_DIR, TEST_NAME1, new String[] {"S.scalar"}));
		addTestConfiguration(TEST_NAME2, new TestConfiguration(TEST_CLASS_DIR, TEST_NAME2, new String[] {"S.scalar"}));
		addTestConfiguration(TEST_NAME3, new TestConfiguration(TEST_CLASS_DIR, TEST_NAME3, new String[] {"S.scalar"}));
		addTestConfiguration(TEST_NAME4, new TestConfiguration(TEST_CLASS_DIR, TEST_NAME4, new String[] {"S.scalar"}));
	}

	@Test
	public void testSumDenseMatrixCP() {
		runColAggregateOperationTest(OpType.SUM, ExecType.CP);
	}

	@Test
	public void testMeanDenseMatrixCP() {
		runColAggregateOperationTest(OpType.MEAN, ExecType.CP);
	}

	@Test
	public void testMaxDenseMatrixCP() {
		runColAggregateOperationTest(OpType.MAX, ExecType.CP);
	}

	@Test
	public void testMinDenseMatrixCP() {
		runColAggregateOperationTest(OpType.MIN, ExecType.CP);
	}

	@Test
	public void testSumDenseMatrixSP() {
		runColAggregateOperationTest(OpType.SUM, ExecType.SPARK);
	}

	@Test
	public void testMeanDenseMatrixSP() {
		runColAggregateOperationTest(OpType.MEAN, ExecType.SPARK);
	}

	@Test
	public void testMaxDenseMatrixSP() {
		runColAggregateOperationTest(OpType.MAX, ExecType.SPARK);
	}

	@Test
	public void testMinDenseMatrixSP() {
		runColAggregateOperationTest(OpType.MIN, ExecType.SPARK);
	}

	private void runColAggregateOperationTest(OpType type, ExecType instType) {
		ExecMode platformOld = rtplatform;
		switch(instType) {
			case SPARK:
				rtplatform = ExecMode.SPARK;
				break;
			default:
				rtplatform = ExecMode.HYBRID;
				break;
		}

		if(rtplatform == ExecMode.SPARK)
			DMLScript.USE_LOCAL_SPARK_CONFIG = true;

		String TEST_NAME = null;
		switch(type) {
			case SUM:
				TEST_NAME = TEST_NAME1;
				break;
			case MEAN:
				TEST_NAME = TEST_NAME2;
				break;
			case MAX:
				TEST_NAME = TEST_NAME3;
				break;
			case MIN:
				TEST_NAME = TEST_NAME4;
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
		Thread t1 = startLocalFedWorkerThread(port1);
		Thread t2 = startLocalFedWorkerThread(port2);
		Thread t3 = startLocalFedWorkerThread(port3);
		Thread t4 = startLocalFedWorkerThread(port4);

		TestConfiguration config = availableTestConfigurations.get(TEST_NAME);
		loadTestConfiguration(config);

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
		compareResults(1e-9);

		switch(type) {
			case SUM:
				Assert.assertTrue(heavyHittersContainsString("fed_uak+"));
				break;
			case MEAN:
				Assert.assertTrue(heavyHittersContainsString("fed_uamean"));
				break;
			case MAX:
				Assert.assertTrue(heavyHittersContainsString("fed_uamax"));
				break;
			case MIN:
				Assert.assertTrue(heavyHittersContainsString("fed_uamin"));
				break;
		}

		// check that federated input files are still existing
		Assert.assertTrue(HDFSTool.existsFileOnHDFS(input("X1")));
		Assert.assertTrue(HDFSTool.existsFileOnHDFS(input("X2")));
		Assert.assertTrue(HDFSTool.existsFileOnHDFS(input("X3")));
		Assert.assertTrue(HDFSTool.existsFileOnHDFS(input("X4")));

		TestUtils.shutdownThreads(t1, t2, t3, t4);
		resetExecMode(platformOld);

	}
}
