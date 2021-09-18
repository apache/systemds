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
import org.apache.sysds.common.Types;
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
public class FederatedReshapeTest extends AutomatedTestBase {
	private final static String TEST_DIR = "functions/federated/";
	private final static String TEST_NAME = "FederatedReshapeTest";
	private final static String TEST_CLASS_DIR = TEST_DIR + FederatedReshapeTest.class.getSimpleName() + "/";

	private final static int blocksize = 1024;
	@Parameterized.Parameter()
	public int rows;

	@Parameterized.Parameter(1)
	public int cols;

	@Parameterized.Parameter(2)
	public int rRows;

	@Parameterized.Parameter(3)
	public int rCols;

	@Parameterized.Parameters
	public static Collection<Object[]> data() {
		return Arrays.asList(new Object[][] {
			{12, 12, 144, 1},
			{12, 12, 24, 6},
			{12, 12, 48, 3}
		});
	}

	@Override
	public void setUp() {
		TestUtils.clearAssertionInformation();
		addTestConfiguration(TEST_NAME, new TestConfiguration(TEST_CLASS_DIR, TEST_NAME, new String[] {"S"}));
	}

	@Test
	public void federatedReshapeCP() {
		federatedReshape(Types.ExecMode.SINGLE_NODE);
	}

	@Test
//	@Ignore
	public void federatedReshapeSP() {
		federatedReshape(Types.ExecMode.SPARK);
	}

	public void federatedReshape(Types.ExecMode execMode) {
		boolean sparkConfigOld = DMLScript.USE_LOCAL_SPARK_CONFIG;
		Types.ExecMode platformOld = rtplatform;

		getAndLoadTestConfiguration(TEST_NAME);
		String HOME = SCRIPT_DIR + TEST_DIR;

		double[][] X1 = getRandomMatrix(2, cols, 1, 5, 1, 3);
		double[][] X2 = getRandomMatrix(2, cols, 1, 5, 1, 7);
		double[][] X3 = getRandomMatrix(6, cols, 1, 5, 1, 8);
		double[][] X4 = getRandomMatrix(2, cols, 1, 5, 1, 9);

		MatrixCharacteristics mc1 = new MatrixCharacteristics(6, cols, blocksize, 6*cols);
		MatrixCharacteristics mc2 = new MatrixCharacteristics(2, cols, blocksize, 2*cols);
		writeInputMatrixWithMTD("X1", X1, false, mc2);
		writeInputMatrixWithMTD("X2", X2, false, mc2);
		writeInputMatrixWithMTD("X3", X3, false, mc1);
		writeInputMatrixWithMTD("X4", X4, false, mc2);

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

		// reference file should not be written to hdfs, so we set platform here
		rtplatform = execMode;
		if(rtplatform == Types.ExecMode.SPARK) {
			DMLScript.USE_LOCAL_SPARK_CONFIG = true;
		}
		// Run reference dml script with normal matrix for Row/Col
		fullDMLScriptName = HOME + TEST_NAME + "Reference.dml";
		programArgs = new String[] {"-stats", "100", "-args", 
			input("X1"), input("X2"), input("X3"), input("X4"), expected("S"), String.valueOf(rRows), String.valueOf(rCols)};
		runTest(null);

		TestConfiguration config = availableTestConfigurations.get(TEST_NAME);
		loadTestConfiguration(config);

		fullDMLScriptName = HOME + TEST_NAME + ".dml";
		programArgs = new String[] {"-stats", "100", "-nvargs",
			"in_X1=" + TestUtils.federatedAddress(port1, input("X1")),
			"in_X2=" + TestUtils.federatedAddress(port2, input("X2")),
			"in_X3=" + TestUtils.federatedAddress(port3, input("X3")),
			"in_X4=" + TestUtils.federatedAddress(port4, input("X4")),
			"rows=" + rows,
			"cols=" + cols,
			"r_rows=" + rRows,
			"r_cols=" + rCols,
			"out_S=" + output("S")};
		runTest(null);

		// compare all sums via files
		compareResults(0.01, "DML1", "DML2");

		Assert.assertTrue(heavyHittersContainsString("fed_rshape"));

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
