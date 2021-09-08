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
import org.junit.Ignore;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.Parameterized;

@RunWith(value = Parameterized.class)
@net.jcip.annotations.NotThreadSafe
public class FederatedMisAlignedTest extends AutomatedTestBase {
	private final static String TEST_NAME1 = "FederatedMisAlignedTest";

	private final static String TEST_DIR = "functions/federated/";
	private static final String TEST_CLASS_DIR = TEST_DIR + FederatedMisAlignedTest.class.getSimpleName() + "/";

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
		MM,
		EW_MULT,
		EW_PLUS,
		EW_GREATER,
		BIND,
	}

	private enum MisAlignmentType {
		HOST,
		RANGE,
	}

	@Override
	public void setUp() {
		TestUtils.clearAssertionInformation();
		addTestConfiguration(TEST_NAME1, new TestConfiguration(TEST_CLASS_DIR, TEST_NAME1, new String[] {"S"}));
	}

	@Test
	public void testMMMisAlignedHostCP() {
		runMisAlignedTest(OpType.MM, ExecMode.SINGLE_NODE, MisAlignmentType.HOST);
	}

	@Test
	public void testMMMisAlignedHostSP() {
		runMisAlignedTest(OpType.MM, ExecMode.SPARK, MisAlignmentType.HOST);
	}

	@Test
	public void testMMMisAlignedRangeCP() {
		runMisAlignedTest(OpType.MM, ExecMode.SINGLE_NODE, MisAlignmentType.RANGE);
	}

	@Test
	public void testMMMisAlignedRangeSP() {
		runMisAlignedTest(OpType.MM, ExecMode.SPARK, MisAlignmentType.RANGE);
	}

	@Test
	public void testEWMultMisAlignedHostCP() {
		runMisAlignedTest(OpType.EW_MULT, ExecMode.SINGLE_NODE, MisAlignmentType.HOST);
	}

	@Test
	@Ignore
	public void testEWMultMisAlignedHostSP() {
		runMisAlignedTest(OpType.EW_MULT, ExecMode.SPARK, MisAlignmentType.HOST);
	}

	@Test
	@Ignore
	public void testEWMultMisAlignedRangeCP() {
		runMisAlignedTest(OpType.EW_MULT, ExecMode.SINGLE_NODE, MisAlignmentType.RANGE);
	}

	@Test
	public void testEWMultMisAlignedRangeSP() {
		runMisAlignedTest(OpType.EW_MULT, ExecMode.SPARK, MisAlignmentType.RANGE);
	}

	@Test
	@Ignore
	public void testEWPlusMisAlignedHostCP() {
		runMisAlignedTest(OpType.EW_PLUS, ExecMode.SINGLE_NODE, MisAlignmentType.HOST);
	}

	@Test
	public void testEWPlusMisAlignedHostSP() {
		runMisAlignedTest(OpType.EW_PLUS, ExecMode.SPARK, MisAlignmentType.HOST);
	}

	@Test
	public void testEWPlusMisAlignedRangeCP() {
		runMisAlignedTest(OpType.EW_PLUS, ExecMode.SINGLE_NODE, MisAlignmentType.RANGE);
	}

	@Test
	@Ignore
	public void testEWPlusMisAlignedRangeSP() {
		runMisAlignedTest(OpType.EW_PLUS, ExecMode.SPARK, MisAlignmentType.RANGE);
	}

	@Test
	public void testEWGreaterMisAlignedHostCP() {
		runMisAlignedTest(OpType.EW_GREATER, ExecMode.SINGLE_NODE, MisAlignmentType.HOST);
	}

	@Test
	@Ignore
	public void testEWGreaterMisAlignedHostSP() {
		runMisAlignedTest(OpType.EW_GREATER, ExecMode.SPARK, MisAlignmentType.HOST);
	}

	@Test
	@Ignore
	public void testEWGreaterMisAlignedRangeCP() {
		runMisAlignedTest(OpType.EW_GREATER, ExecMode.SINGLE_NODE, MisAlignmentType.RANGE);
	}

	@Test
	public void testEWGreaterMisAlignedRangeSP() {
		runMisAlignedTest(OpType.EW_GREATER, ExecMode.SPARK, MisAlignmentType.RANGE);
	}

	@Test
	public void testBindMisAlignedHostCP() {
		runMisAlignedTest(OpType.BIND, ExecMode.SINGLE_NODE, MisAlignmentType.HOST);
	}

	@Test
	public void testBindMisAlignedHostSP() {
		runMisAlignedTest(OpType.BIND, ExecMode.SPARK, MisAlignmentType.HOST);
	}

	@Test
	public void testBindMisAlignedRangeCP() {
		runMisAlignedTest(OpType.BIND, ExecMode.SINGLE_NODE, MisAlignmentType.RANGE);
	}

	@Test
	public void testBindMisAlignedRangeSP() {
		runMisAlignedTest(OpType.BIND, ExecMode.SPARK, MisAlignmentType.RANGE);
	}

	private void runMisAlignedTest(OpType type, ExecMode execMode, MisAlignmentType maType) {
		boolean sparkConfigOld = DMLScript.USE_LOCAL_SPARK_CONFIG;
		ExecMode platformOld = rtplatform;

		if(rtplatform == ExecMode.SPARK)
			DMLScript.USE_LOCAL_SPARK_CONFIG = true;

		String TEST_NAME = TEST_NAME1;

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
		programArgs = new String[] {"-stats", "100", "-nvargs",
			"in_X1=" + input("X1"), "in_X2=" + input("X2"), "in_X3=" + input("X3"), "in_X4=" + input("X4"),
			"testnum=" + Integer.toString(type.ordinal()), "misaligntype=" + Integer.toString(maType.ordinal()),
			"rP=" + Boolean.toString(rowPartitioned).toUpperCase(), "out_S=" + expected("S")};
		runTest(true, false, null, -1);
		
		// Run actual dml script with federated matrix
		
		fullDMLScriptName = HOME + TEST_NAME + ".dml";
		programArgs = new String[] {"-stats", "100", "-nvargs",
			"in_X1=" + TestUtils.federatedAddress(port1, input("X1")),
			"in_X2=" + TestUtils.federatedAddress(port2, input("X2")),
			"in_X3=" + TestUtils.federatedAddress(port3, input("X3")),
			"in_X4=" + TestUtils.federatedAddress(port4, input("X4")), "rows=" + rows, "cols=" + cols,
			"testnum=" + Integer.toString(type.ordinal()), "misaligntype=" + Integer.toString(maType.ordinal()),
			"rP=" + Boolean.toString(rowPartitioned).toUpperCase(), "out_S=" + output("S")};

		runTest(true, false, null, -1);

		// compare via files
		compareResults(1e-9, "Stat-DML1", "Stat-DML2");

		switch(type) {
			case MM:
				Assert.assertTrue(heavyHittersContainsString(rtplatform == ExecMode.SPARK ? "fed_mapmm" : "fed_ba+*"));
				break;
			case EW_MULT:
				Assert.assertTrue(heavyHittersContainsString("fed_*"));
				break;
			case EW_PLUS:
				Assert.assertTrue(heavyHittersContainsString("fed_+"));
				break;
			case EW_GREATER:
				Assert.assertTrue(heavyHittersContainsString("fed_>"));
				break;
			case BIND:
				Assert.assertTrue(heavyHittersContainsString(rtplatform == ExecMode.SPARK ? "fed_mappend" : "fed_append"));
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
