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

import org.apache.sysds.common.Types.ExecMode;
import org.apache.sysds.common.Types.ExecType;
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
public class FederatedFullCumulativeTest extends AutomatedTestBase {
	private final static String TEST_NAME1 = "FederatedCumsumTest";
	private final static String TEST_NAME2 = "FederatedCumprodTest";
	private final static String TEST_NAME3 = "FederatedCummaxTest";
	private final static String TEST_NAME4 = "FederatedCumminTest";
	private final static String TEST_NAME5 = "FederatedCumsumprodTest";

	private final static String TEST_DIR = "functions/federated/cumulative/";
	private static final String TEST_CLASS_DIR = TEST_DIR + FederatedFullCumulativeTest.class.getSimpleName() + "/";

	private final static int blocksize = 1024;
	@Parameterized.Parameter()
	public int rows;
	@Parameterized.Parameter(1)
	public int cols;
	@Parameterized.Parameter(2)
	public boolean rowPartitioned;

	@Parameterized.Parameters
	public static Collection<Object[]> data() {
		return Arrays.asList(new Object[][] {{240, 4, true}, {240, 4, false},});
	}

	private enum OpType {
		SUM, PROD, SUMPROD, MAX, MIN
	}

	@Override
	public void setUp() {
		TestUtils.clearAssertionInformation();
		addTestConfiguration(TEST_NAME1, new TestConfiguration(TEST_CLASS_DIR, TEST_NAME1, new String[] {"S"}));
		addTestConfiguration(TEST_NAME2, new TestConfiguration(TEST_CLASS_DIR, TEST_NAME2, new String[] {"S"}));
		addTestConfiguration(TEST_NAME3, new TestConfiguration(TEST_CLASS_DIR, TEST_NAME3, new String[] {"S"}));
		addTestConfiguration(TEST_NAME4, new TestConfiguration(TEST_CLASS_DIR, TEST_NAME4, new String[] {"S"}));
		addTestConfiguration(TEST_NAME5, new TestConfiguration(TEST_CLASS_DIR, TEST_NAME5, new String[] {"S"}));
	}

	@Test
	public void testSumDenseMatrixCP() {
		runCumOperationTest(OpType.SUM, ExecType.CP);
	}
	
	@Test
	public void testMaxDenseMatrixCP() {
		runCumOperationTest(OpType.MAX, ExecType.CP);
	}
	
	@Test
	public void testMinDenseMatrixCP() {
		runCumOperationTest(OpType.MIN, ExecType.CP);
	}

	@Test
	public void testProdDenseMatrixCP() {
		runCumOperationTest(OpType.PROD, ExecType.CP);
	}

	@Test
	public void testSumProdDenseMatrixCP() {
		runCumOperationTest(OpType.SUMPROD, ExecType.CP);
	}

	@Test
	@Ignore
	public void testSumDenseMatrixSP() {
		runCumOperationTest(OpType.SUM, ExecType.SPARK);
	}

	@Test
	@Ignore
	public void testMaxDenseMatrixSP() {
		runCumOperationTest(OpType.MAX, ExecType.SPARK);
	}

	@Test
	@Ignore
	public void testMinDenseMatrixSP() {
		runCumOperationTest(OpType.MIN, ExecType.SPARK);
	}

	private void runCumOperationTest(OpType type, ExecType instType) {
		ExecMode platformOld = setExecMode(instType);

		String TEST_NAME = null;
		switch(type) {
			case SUM:
				TEST_NAME = TEST_NAME1;
				break;
			case PROD:
				TEST_NAME = TEST_NAME2;
				break;
			case MAX:
				TEST_NAME = TEST_NAME3;
				break;
			case MIN:
				TEST_NAME = TEST_NAME4;
				break;
			case SUMPROD:
				TEST_NAME = TEST_NAME5;
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

		double[][] X1 = getRandomMatrix(r, c, 1, 3, 1, 3);
		double[][] X2 = getRandomMatrix(r, c, 1, 3, 1, 7);
		double[][] X3 = getRandomMatrix(r, c, 1, 3, 1, 8);
		double[][] X4 = getRandomMatrix(r, c, 1, 3, 1, 9);

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
		Process t4 = startLocalFedWorker(port4);

		
		try {
			if(!isAlive(t1, t2, t3, t4))
				throw new RuntimeException("Failed starting federated worker");

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
			if (type != OpType.SUMPROD && type != OpType.PROD)
				compareResults(1e-6, "DML1", "DML2");
			else // we sum over the cumsumprod matrix and get a very large number, hence the large tolerance
				compareResults(1e+73, "DML1", "DML2");

			switch(type) {
				case SUM:
					Assert
						.assertTrue(heavyHittersContainsString(instType == ExecType.SPARK ? "fed_bcumoffk+" : "fed_ucumk+"));
					break;
				case PROD:
					Assert.assertTrue(heavyHittersContainsString(instType == ExecType.SPARK ? "fed_bcumoff*" : "fed_ucum*"));
					break;
				case MAX:
					Assert.assertTrue(
						heavyHittersContainsString(instType == ExecType.SPARK ? "fed_bcumoffmax" : "fed_ucummax"));
					break;
				case MIN:
					Assert.assertTrue(
						heavyHittersContainsString(instType == ExecType.SPARK ? "fed_bcumoffmin" : "fed_ucummin"));
					break;
				case SUMPROD:
					// when input is column-partitioned, ucumk+* is executed instead of fed_ucumk+*
					Assert.assertTrue(heavyHittersContainsString(instType == ExecType.SPARK ? "fed_bcumoff+*" :
						rowPartitioned ? "fed_ucumk+*" : "ucumk+*"));
					break;
			}

			if(instType != ExecType.SPARK) { // verify output is federated
				if (type == OpType.SUMPROD && !rowPartitioned) {
					Assert.assertTrue(heavyHittersContainsString("uak+"));
				} else {
					Assert.assertTrue(heavyHittersContainsString("fed_uak+"));
				}
			}


			// check that federated input files are still existing
			Assert.assertTrue(HDFSTool.existsFileOnHDFS(input("X1")));
			Assert.assertTrue(HDFSTool.existsFileOnHDFS(input("X2")));
			Assert.assertTrue(HDFSTool.existsFileOnHDFS(input("X3")));
			Assert.assertTrue(HDFSTool.existsFileOnHDFS(input("X4")));

		}
		finally {
			TestUtils.shutdownThreads(t1, t2, t3, t4);
			resetExecMode(platformOld);
		}
	}
}
