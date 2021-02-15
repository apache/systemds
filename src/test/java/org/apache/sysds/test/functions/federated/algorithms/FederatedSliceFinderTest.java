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

package org.apache.sysds.test.functions.federated.algorithms;

import java.util.Arrays;
import java.util.Collection;
import java.util.HashMap;

import org.apache.sysds.api.DMLScript;
import org.apache.sysds.common.Types;
import org.apache.sysds.runtime.matrix.data.MatrixValue;
import org.apache.sysds.runtime.meta.MatrixCharacteristics;
import org.apache.sysds.runtime.util.HDFSTool;
import org.apache.sysds.test.AutomatedTestBase;
import org.apache.sysds.test.TestConfiguration;
import org.apache.sysds.test.TestUtils;
import org.apache.sysds.test.functions.builtin.BuiltinSliceFinderTest;
import org.junit.Assert;
import org.junit.Ignore;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.Parameterized;

@RunWith(value = Parameterized.class)
@net.jcip.annotations.NotThreadSafe
public class FederatedSliceFinderTest extends AutomatedTestBase {

	private static final String PREP_NAME = "slicefinderPrep";
	private static final String TEST_NAME = "FederatedSliceFinder";
	private static final String TEST_DIR = "functions/federated/";
	private static final String TEST_CLASS_DIR = TEST_DIR + FederatedSliceFinderTest.class.getSimpleName() + "/";
	private static final boolean VERBOSE = true;
	private final static int blocksize = 1024;

	@Parameterized.Parameter()
	public int rows;
	@Parameterized.Parameter(1)
	public int cols;
	@Parameterized.Parameter(2)
	public boolean rowPartitioned;

	@Parameterized.Parameters
	public static Collection<Object[]> data() {
		return Arrays.asList(new Object[][] {{12, 4, true}});
	}


	@Override
	public void setUp() {
		addTestConfiguration(TEST_NAME, new TestConfiguration(TEST_CLASS_DIR, TEST_NAME, new String[]{"R"}));
	}

//	@Test
//	public void testTop4HybridDP() {
//		runSliceFinderTest(4, true, Types.ExecMode.HYBRID);
//	}

	@Test
	public void testTop4SinglenodeDP() {
		runSliceFinderTest(4, true, Types.ExecMode.SINGLE_NODE);
	}

//	@Test
//	public void testTop4HybridTP() {
//		runSliceFinderTest(4, false, Types.ExecMode.HYBRID);
//	}
//
//	@Test
//	public void testTop4SinglenodeTP() {
//		runSliceFinderTest(4, false, Types.ExecMode.SINGLE_NODE);
//	}
//
//	@Test
//	public void testTop10HybridDP() {
//		runSliceFinderTest(10, true, Types.ExecMode.HYBRID);
//	}
//
//	@Test
//	public void testTop10SinglenodeDP() {
//		runSliceFinderTest(10, true, Types.ExecMode.SINGLE_NODE);
//	}
//
//	@Test
//	public void testTop10HybridTP() {
//		runSliceFinderTest(10, false, Types.ExecMode.HYBRID);
//	}
//
//	@Test
//	public void testTop10SinglenodeTP() {
//		runSliceFinderTest(10, false, Types.ExecMode.SINGLE_NODE);
//	}

	private void runSliceFinderTest(int K, boolean dp, Types.ExecMode execMode) {
		boolean sparkConfigOld = DMLScript.USE_LOCAL_SPARK_CONFIG;
		Types.ExecMode platformOld = rtplatform;

		if(rtplatform == Types.ExecMode.SPARK)
			DMLScript.USE_LOCAL_SPARK_CONFIG = true;

		getAndLoadTestConfiguration(TEST_NAME);
		String HOME = SCRIPT_DIR + TEST_DIR;

		String data = SCRIPT_DIR + "functions/builtin/" + "/data/Salaries.csv";
		//run data preparation
		fullDMLScriptName = SCRIPT_DIR + "functions/builtin/" + PREP_NAME + ".dml";
		programArgs = new String[]{"-args", data, output("X"), output("e")};
		runTest(true, false, null, -1);

		//read output and resize it to / 4
		double[][] X_resize = TestUtils.convertHashMapToDoubleArray(readDMLMatrixFromOutputDir("X"));
		double[][] X = new double[X_resize.length-1][];
		System.arraycopy(X_resize, 0, X, 0, X_resize.length - 1);
		double[][] e_resize = TestUtils.convertHashMapToDoubleArray(readDMLMatrixFromOutputDir("e"));
		double[][] e = new double[e_resize.length-1][];
		System.arraycopy(e_resize, 0, e, 0, e_resize.length - 1);
		writeInputMatrixWithMTD("X", X, true);
		writeInputMatrixWithMTD("e", e, true);
//
//		// row fed
//		int r = X.length / 4;
//		int c = X[0].length;
//		double [][] X1 = new double[r][c];
//		double [][] X2 = new double[r][c];
//		double [][] X3 = new double[r][c];
//		double [][] X4 = new double[r][c];
//		System.arraycopy(X,0, X1,0, r);
//		System.arraycopy(X,r, X2, 0, r);
//		System.arraycopy(X,2 * r, X3,0, r);
//		System.arraycopy(X,3 * r, X4,0, r);

		int r = rows / 4;
		int c = cols;

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
		Thread t1 = startLocalFedWorkerThread(port1, FED_WORKER_WAIT_S);
		Thread t2 = startLocalFedWorkerThread(port2, FED_WORKER_WAIT_S);
		Thread t3 = startLocalFedWorkerThread(port3, FED_WORKER_WAIT_S);
		Thread t4 = startLocalFedWorkerThread(port4);

		TestConfiguration config = availableTestConfigurations.get(TEST_NAME);
		loadTestConfiguration(config);


//		//execute main test
		fullDMLScriptName = HOME + TEST_NAME + "Reference.dml";
		programArgs = new String[]{"-stats", "100", "-args",
			input("X1"), input("X2"), input("X3"), input("X4"), input("e"),
			String.valueOf(K),String.valueOf(!dp).toUpperCase(),
			String.valueOf(VERBOSE).toUpperCase(), expected("R")};
		runTest(true, false, null, -1);

		// Run actual dml script with federated matrix
		fullDMLScriptName = HOME + TEST_NAME + ".dml";
		programArgs = new String[] {"-stats", "100", "-nvargs",
			"in_X1=" + TestUtils.federatedAddress(port1, input("X1")),
			"in_X2=" + TestUtils.federatedAddress(port2, input("X2")),
			"in_X3=" + TestUtils.federatedAddress(port3, input("X3")),
			"in_X4=" + TestUtils.federatedAddress(port4, input("X4")),
//			"rows=" + X.length, "cols=" + X[0].length,
			"rows=" + rows, "cols=" + cols,
			"in_e=" + input("e"), "in_K=" + K,
			"in_tpEval=" + String.valueOf(!dp).toUpperCase(),
			"verbose="+ String.valueOf(VERBOSE).toUpperCase(),
			"out_R=" + output("R")
		};
		runTest(true, false, null, -1);

		// compare via files
		compareResults(1e-9);

		Assert.assertTrue(heavyHittersContainsString("fed_ba+*"));
		Assert.assertTrue(heavyHittersContainsString("fed_rightIndex"));
		Assert.assertTrue(heavyHittersContainsString("fed_transformencode"));
		Assert.assertTrue(heavyHittersContainsString("fed_uacmax"));
		Assert.assertTrue(heavyHittersContainsString("fed_uark+"));
		Assert.assertTrue(heavyHittersContainsString("fed_uarimax"));
		Assert.assertTrue(heavyHittersContainsString("fed_tsmm"));
		Assert.assertTrue(heavyHittersContainsString("fed_min"));
		Assert.assertTrue(heavyHittersContainsString("fed_uack+"));
		Assert.assertTrue(heavyHittersContainsString("fed_rshape"));
		Assert.assertTrue(heavyHittersContainsString("fed_replace"));
		Assert.assertTrue(heavyHittersContainsString("fed_uppertri"));
		//		Assert.assertTrue(heavyHittersContainsString("fed_leftIndex"));
		//		Assert.assertTrue(heavyHittersContainsString("fed_ucumk+"));
		//		Assert.assertTrue(heavyHittersContainsString("fed_uak+"));
		//		Assert.assertTrue(heavyHittersContainsString("fed_max"));
		//		Assert.assertTrue(heavyHittersContainsString("fed_ifelse"));


		// check that federated input files are still existing
		Assert.assertTrue(HDFSTool.existsFileOnHDFS(input("X1")));
		Assert.assertTrue(HDFSTool.existsFileOnHDFS(input("X2")));
		Assert.assertTrue(HDFSTool.existsFileOnHDFS(input("X3")));
		Assert.assertTrue(HDFSTool.existsFileOnHDFS(input("X4")));

		TestUtils.shutdownThreads(t1, t2, t3, t4);

		resetExecMode(platformOld);
	}
}
