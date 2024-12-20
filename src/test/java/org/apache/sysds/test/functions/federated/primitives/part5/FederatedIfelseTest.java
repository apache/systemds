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
public class FederatedIfelseTest extends AutomatedTestBase {
	private final static String TEST_NAME1 = "FederatedIfelseTest";
	private final static String TEST_NAME2 = "FederatedIfelseAlignedTest";
	private final static String TEST_NAME3 = "FederatedIfelseSingleMatrixInputTest";

	private final static String TEST_DIR = "functions/federated/";
	private static final String TEST_CLASS_DIR = TEST_DIR + FederatedIfelseTest.class.getSimpleName() + "/";

	private final static int blocksize = 1024;

	@Parameterized.Parameter()
	public int rows;
	@Parameterized.Parameter(1)
	public int cols;
	@Parameterized.Parameter(2)
	public boolean rowPartitioned;

	@Parameterized.Parameters
	public static Collection<Object[]> data() {
		return Arrays.asList(new Object[][] {{64, 16, true}, {64, 16, false},});
	}

	@Override
	public void setUp() {
		TestUtils.clearAssertionInformation();
		addTestConfiguration(TEST_NAME1, new TestConfiguration(TEST_CLASS_DIR, TEST_NAME1, new String[] {"S"}));
		addTestConfiguration(TEST_NAME2, new TestConfiguration(TEST_CLASS_DIR, TEST_NAME2, new String[] {"S"}));
		addTestConfiguration(TEST_NAME3, new TestConfiguration(TEST_CLASS_DIR, TEST_NAME3, new String[] {"S"}));
	}

	@Test
	public void testIfelseDiffWorkersCP() {
		runTernaryTest(ExecMode.SINGLE_NODE, false, false);
	}

	@Test
	public void testIfelseDiffWorkersSingleMatInCP() {
		runTernaryTest(ExecMode.SINGLE_NODE, false, true);
	}

	@Test
	public void testIfelseAlignedCP() {
		runTernaryTest(ExecMode.SINGLE_NODE, true, false);
	}

	@Test
	public void testIfelseDiffWorkersSP() {
		runTernaryTest(ExecMode.SPARK, false, false);
	}

	@Test
	public void testIfelseAlignedSP() {
		runTernaryTest(ExecMode.SPARK, true, false);
	}

	private void runTernaryTest(ExecMode execMode, boolean aligned, boolean singleMatrixInput) {
		boolean sparkConfigOld = DMLScript.USE_LOCAL_SPARK_CONFIG;
		ExecMode platformOld = rtplatform;

		if(rtplatform == ExecMode.SPARK)
			DMLScript.USE_LOCAL_SPARK_CONFIG = true;

		String TEST_NAME = aligned ? TEST_NAME2 : (!singleMatrixInput ? TEST_NAME1 : TEST_NAME3);

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
		X1[0][0] = 0;
		MatrixCharacteristics mc = new MatrixCharacteristics(r, c, blocksize, r * c);
		writeInputMatrixWithMTD("X1", X1, false, mc);
		writeInputMatrixWithMTD("X2", X2, false, mc);
		writeInputMatrixWithMTD("X3", X3, false, mc);
		writeInputMatrixWithMTD("X4", X4, false, mc);

		double[][] Y1 = getRandomMatrix(r, c, 10, 15, 1, 3);
		double[][] Y2 = getRandomMatrix(r, c, 10, 15, 1, 7);
		double[][] Y3 = getRandomMatrix(r, c, 10, 15, 1, 8);
		double[][] Y4 = getRandomMatrix(r, c, 10, 15, 1, 9);
		MatrixCharacteristics mc2 = new MatrixCharacteristics(r, c, blocksize, r * c);
		writeInputMatrixWithMTD("Y1", Y1, false, mc2);
		writeInputMatrixWithMTD("Y2", Y2, false, mc2);
		writeInputMatrixWithMTD("Y3", Y3, false, mc2);
		writeInputMatrixWithMTD("Y4", Y4, false, mc2);

		// empty script name because we don't execute any script, just start the worker
		fullDMLScriptName = "";
		int port1 = getRandomAvailablePort();
		int port2 = getRandomAvailablePort();
		int port3 = getRandomAvailablePort();
		int port4 = getRandomAvailablePort();
		Process t1 = startLocalFedWorker(port1);
		Process t2 = startLocalFedWorker(port2);
		Process t3 = startLocalFedWorker(port3);
		Process t4 = startLocalFedWorker(port4);

		
		try {
			if(!isAlive(t1, t2, t3, t4))
				throw new RuntimeException("Failed starting federated worker");

			rtplatform = execMode;
			if(rtplatform == ExecMode.SPARK)
				DMLScript.USE_LOCAL_SPARK_CONFIG = true;

			TestConfiguration config = availableTestConfigurations.get(TEST_NAME);
			loadTestConfiguration(config);

			if(aligned) {
				// Run reference dml script with normal matrix
				fullDMLScriptName = HOME + TEST_NAME + "Reference.dml";
				programArgs = new String[] {"-stats", "100", "-args", input("X1"), input("X2"), input("X3"), input("X4"),
					input("Y1"), input("Y2"), input("Y3"), input("Y4"), expected("S"),
					Boolean.toString(rowPartitioned).toUpperCase()};
				runTest(true, false, null, -1);

				// Run actual dml script with federated matrix

				fullDMLScriptName = HOME + TEST_NAME + ".dml";
				programArgs = new String[] {"-stats", "100", "-nvargs", "in_X1=" + TestUtils.federatedAddress(port1, input("X1")),
					"in_X2=" + TestUtils.federatedAddress(port2, input("X2")),
					"in_X3=" + TestUtils.federatedAddress(port3, input("X3")),
					"in_X4=" + TestUtils.federatedAddress(port4, input("X4")),
					"in_Y1=" + TestUtils.federatedAddress(port1, input("Y1")),
					"in_Y2=" + TestUtils.federatedAddress(port2, input("Y2")),
					"in_Y3=" + TestUtils.federatedAddress(port3, input("Y3")),
					"in_Y4=" + TestUtils.federatedAddress(port4, input("Y4")), "rows=" + rows, "cols=" + cols,
					"rP=" + Boolean.toString(rowPartitioned).toUpperCase(), "out_S=" + output("S")};
				runTest(true, false, null, -1);
			} else {
				// Run reference dml script with normal matrix
				fullDMLScriptName = HOME + TEST_NAME + "Reference.dml";
				programArgs = new String[] {"-stats", "100", "-args", input("X1"), input("X2"), input("X3"),
					input("X4"), expected("S"), Boolean.toString(rowPartitioned).toUpperCase()};
				runTest(true, false, null, -1);

				// Run actual dml script with federated matrix
				double[][] x = getRandomMatrix(1, 1, 3.0, 3.0, 1, 1);
				double[][] y = getRandomMatrix(1, 1, 4.0, 4.0, 1, 1);
				MatrixCharacteristics mc1 = new MatrixCharacteristics(1, 1, blocksize, 1 * 1);
				writeInputMatrixWithMTD("x", x, false, mc1);
				writeInputMatrixWithMTD("y", y, false, mc1);

				fullDMLScriptName = HOME + TEST_NAME + ".dml";
				programArgs = new String[] {"-stats", "100", "-nvargs",
					"in_X1=" + TestUtils.federatedAddress(port1, input("X1")),
					"in_X2=" + TestUtils.federatedAddress(port2, input("X2")),
					"in_X3=" + TestUtils.federatedAddress(port3, input("X3")),
					"in_X4=" + TestUtils.federatedAddress(port4, input("X4")), "rows=" + rows, "cols=" + cols,
					"x=" + input("x"), "y=" + input("y"), "rP=" + Boolean.toString(rowPartitioned).toUpperCase(),
					"out_S=" + output("S")};
				runTest(true, false, null, -1);
			}

			// compare via files
			compareResults(1e-9, "DML1", "DML2");
			Assert.assertTrue(heavyHittersContainsString("fed_ifelse"));

			// check that federated input files are still existing
			Assert.assertTrue(HDFSTool.existsFileOnHDFS(input("X1")));
			Assert.assertTrue(HDFSTool.existsFileOnHDFS(input("X2")));
			Assert.assertTrue(HDFSTool.existsFileOnHDFS(input("X3")));
			Assert.assertTrue(HDFSTool.existsFileOnHDFS(input("X4")));

			if(aligned) {
				Assert.assertTrue(HDFSTool.existsFileOnHDFS(input("Y1")));
				Assert.assertTrue(HDFSTool.existsFileOnHDFS(input("Y2")));
				Assert.assertTrue(HDFSTool.existsFileOnHDFS(input("Y3")));
				Assert.assertTrue(HDFSTool.existsFileOnHDFS(input("Y4")));
			}

		}
		finally {
			TestUtils.shutdownThreads(t1, t2, t3, t4);

			rtplatform = platformOld;
			DMLScript.USE_LOCAL_SPARK_CONFIG = sparkConfigOld;
		}
	}
}
