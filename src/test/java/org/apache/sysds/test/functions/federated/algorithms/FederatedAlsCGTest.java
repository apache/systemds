/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.	See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.	The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.	You may obtain a copy of the License at
 *
 *	 http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.	See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

package org.apache.sysds.test.functions.federated.algorithms;

import org.apache.sysds.common.Types.ExecMode;
import org.apache.sysds.runtime.matrix.data.MatrixValue.CellIndex;
import org.apache.sysds.runtime.meta.MatrixCharacteristics;
import org.apache.sysds.runtime.util.HDFSTool;
import org.apache.sysds.test.AutomatedTestBase;
import org.apache.sysds.test.TestConfiguration;
import org.apache.sysds.test.TestUtils;
import org.junit.Assert;
import org.junit.BeforeClass;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.Parameterized;

import java.util.Arrays;
import java.util.Collection;
import java.util.HashMap;

@RunWith(value = Parameterized.class)
@net.jcip.annotations.NotThreadSafe
public class FederatedAlsCGTest extends AutomatedTestBase
{
	private final static String TEST_NAME = "FederatedAlsCGTest";
	private final static String TEST_DIR = "functions/federated/";
	private final static String TEST_CLASS_DIR = TEST_DIR + FederatedAlsCGTest.class.getSimpleName() + "/";

	private final static String OUTPUT_NAME = "Z";
	private final static double TOLERANCE = 0.01;
	private final static int BLOCKSIZE = 1024;

	@Parameterized.Parameter()
	public int rows;
	@Parameterized.Parameter(1)
	public int cols;
	@Parameterized.Parameter(2)
	public int rank;
	@Parameterized.Parameter(3)
	public String regression;
	@Parameterized.Parameter(4)
	public double lambda;
	@Parameterized.Parameter(5)
	public int max_iter;
	@Parameterized.Parameter(6)
	public double threshold;
	@Parameterized.Parameter(7)
	public double sparsity;

	@Override
	public void setUp() {
		addTestConfiguration(TEST_NAME, new TestConfiguration(TEST_CLASS_DIR, TEST_NAME, new String[]{OUTPUT_NAME}));
	}

	@Parameterized.Parameters
	public static Collection<Object[]> data() {
		// rows must be even
		return Arrays.asList(new Object[][] {
			// {rows, cols, rank, regression, lambda, max_iter, threshold, sparsity}
			{30, 15, 10, "L2", 0.0000001, 50, 0.000001, 1},
			{30, 15, 10, "wL2", 0.0000001, 50, 0.000001, 1}
		});
	}

	@BeforeClass
	public static void init() {
		TestUtils.clearDirectory(TEST_DATA_DIR + TEST_CLASS_DIR);
	}

	@Test
	public void federatedAlsCGSingleNode() {
		federatedAlsCG(TEST_NAME, ExecMode.SINGLE_NODE);
	}

//	@Test
//	public void federatedAlsCGSpark() {
//		federatedAlsCG(TEST_NAME, ExecMode.SPARK);
//	}

// -----------------------------------------------------------------------------

	public void federatedAlsCG(String testname, ExecMode execMode)
	{
		// store the previous platform config to restore it after the test
		ExecMode platform_old = setExecMode(execMode);

		getAndLoadTestConfiguration(testname);
		String HOME = SCRIPT_DIR + TEST_DIR;

		int fed_rows = rows / 2;
		int fed_cols = cols;

		double[][] X1 = getRandomMatrix(fed_rows, fed_cols, 1, 2, sparsity, 13);
		double[][] X2 = getRandomMatrix(fed_rows, fed_cols, 1, 2, sparsity, 2);

		writeInputMatrixWithMTD("X1", X1, false, new MatrixCharacteristics(
			fed_rows, fed_cols, BLOCKSIZE, fed_rows * fed_cols));
		writeInputMatrixWithMTD("X2", X2, false, new MatrixCharacteristics(
			fed_rows, fed_cols, BLOCKSIZE, fed_rows * fed_cols));

		// empty script name because we don't execute any script, just start the worker
		fullDMLScriptName = "";
		int port1 = getRandomAvailablePort();
		int port2 = getRandomAvailablePort();
		Thread thread1 = startLocalFedWorkerThread(port1, FED_WORKER_WAIT_S);
		Thread thread2 = startLocalFedWorkerThread(port2);

		getAndLoadTestConfiguration(testname);

		// Run reference dml script with normal matrix
		fullDMLScriptName = HOME + testname + "Reference.dml";
		programArgs = new String[] {"-stats", "-nvargs",
			"in_X1=" + input("X1"), "in_X2=" + input("X2"), "in_rank=" + Integer.toString(rank),
			"in_reg=" + regression, "in_lambda=" + Double.toString(lambda),
			"in_maxi=" + Integer.toString(max_iter), "in_thr=" + Double.toString(threshold),
			"out_Z=" + expected(OUTPUT_NAME)};
		runTest(true, false, null, -1);

		// Run actual dml script with federated matrix
		fullDMLScriptName = HOME + testname + ".dml";
		programArgs = new String[] {"-explain", "-stats", "-nvargs",
			"in_X1=" + TestUtils.federatedAddress(port1, input("X1")),
			"in_X2=" + TestUtils.federatedAddress(port2, input("X2")),
			"in_rank=" + Integer.toString(rank),
			"in_reg=" + regression,
			"in_lambda=" + Double.toString(lambda),
			"in_maxi=" + Integer.toString(max_iter),
			"in_thr=" + Double.toString(threshold),
			"rows=" + fed_rows, "cols=" + fed_cols,
			"out_Z=" + output(OUTPUT_NAME)};
		runTest(true, false, null, -1);

		// compare the results via files
		HashMap<CellIndex, Double> refResults  = readDMLMatrixFromExpectedDir(OUTPUT_NAME);
		HashMap<CellIndex, Double> fedResults = readDMLMatrixFromOutputDir(OUTPUT_NAME);
		TestUtils.compareMatrices(fedResults, refResults, TOLERANCE, "Fed", "Ref");

		TestUtils.shutdownThreads(thread1, thread2);

		// check for federated operations
		Assert.assertTrue(heavyHittersContainsString("fed_!="));
		Assert.assertTrue(heavyHittersContainsString("fed_fedinit"));
		Assert.assertTrue(heavyHittersContainsString("fed_wdivmm"));
		Assert.assertTrue(heavyHittersContainsString("fed_wsloss"));

		// check that federated input files are still existing
		Assert.assertTrue(HDFSTool.existsFileOnHDFS(input("X1")));
		Assert.assertTrue(HDFSTool.existsFileOnHDFS(input("X2")));

		resetExecMode(platform_old);
	}
}
