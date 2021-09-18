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

package org.apache.sysds.test.functions.federated.codegen;

import java.io.File;
import org.apache.sysds.common.Types.ExecMode;
import org.apache.sysds.runtime.matrix.data.MatrixValue.CellIndex;
import org.apache.sysds.runtime.meta.MatrixCharacteristics;
import org.apache.sysds.runtime.util.HDFSTool;
import org.apache.sysds.test.AutomatedTestBase;
import org.apache.sysds.test.TestConfiguration;
import org.apache.sysds.test.TestUtils;
import org.junit.Assert;
import org.junit.BeforeClass;
import org.junit.Ignore;
import org.junit.runner.RunWith;
import org.junit.runners.Parameterized;
import org.junit.Test;

import java.util.Arrays;
import java.util.Collection;
import java.util.HashMap;

@RunWith(value = Parameterized.class)
@net.jcip.annotations.NotThreadSafe
public class FederatedCodegenMultipleFedMOTest extends AutomatedTestBase
{
	private final static String TEST_NAME = "FederatedCodegenMultipleFedMOTest";

	private final static String TEST_DIR = "functions/federated/codegen/";
	private final static String TEST_CLASS_DIR = TEST_DIR + FederatedCodegenMultipleFedMOTest.class.getSimpleName() + "/";

	private final static String TEST_CONF = "SystemDS-config-codegen.xml";

	private final static String OUTPUT_NAME = "Z";
	private final static double TOLERANCE = 1e-7;
	private final static int BLOCKSIZE = 1024;

	@Parameterized.Parameter()
	public int test_num;
	@Parameterized.Parameter(1)
	public int rows_x;
	@Parameterized.Parameter(2)
	public int cols_x;
	@Parameterized.Parameter(3)
	public int rows_y;
	@Parameterized.Parameter(4)
	public int cols_y;
	@Parameterized.Parameter(5)
	public boolean row_partitioned;

	@Override
	public void setUp() {
		addTestConfiguration(TEST_NAME, new TestConfiguration(TEST_CLASS_DIR, TEST_NAME, new String[]{OUTPUT_NAME}));
	}

	@Parameterized.Parameters
	public static Collection<Object[]> data() {
		// rows must be even for row partitioned X and Y
		// cols must be even for col partitioned X and Y
		return Arrays.asList(new Object[][] {
			// {test_num, rows_x, cols_x, rows_y, cols_y row_partitioned}

			// cellwise
			// row partitioned
			{1, 4, 4, 4, 4, true},
			// {2, 4, 4, 4, 1, true},
			{3, 4, 1, 4, 1, true},
			{4, 1000, 1, 1000, 1, true},
			// {5, 500, 2, 500, 2, true},
			{6, 2, 500, 2, 500, true},
			{7, 2, 4, 2, 4, true},
			// column partitioned
			// {1, 4, 4, 4, 4, false},
			{2, 4, 4, 1, 4, false},
			{5, 500, 2, 500, 2, false},
			// {6, 2, 500, 2, 500, false},
			{7, 2, 4, 2, 4, false},

			// rowwise
			// {101, 6, 2, 6, 2, true},
			{102, 6, 1, 6, 4, true},
			// {103, 6, 4, 6, 2, true},
			{104, 150, 10, 150, 10, true},

			// multi aggregate
			// row partitioned
			// {201, 6, 4, 6, 4, true},
			{202, 6, 4, 6, 4, true},
			// FIXME: [SYSTEMDS-3110] {203, 20, 1, 20, 1, true},
			// col partitioned
			{201, 6, 4, 6, 4, false},
			{202, 6, 4, 6, 4, false},

			// outer product
			// row partitioned
			// {301, 1500, 1500, 1500, 10, true},
			{303, 4000, 2000, 4000, 10, true},
			// {305, 4000, 2000, 4000, 10, true},
			// {307, 1000, 2000, 1000, 10, true},
			// {309, 1000, 2000, 1000, 10, true},
			// col partitioned
			// {302, 2000, 2000, 10, 2000, false},
			// {304, 4000, 2000, 10, 2000, false},
			// {306, 4000, 2000, 10, 2000, false},
			{308, 1000, 2000, 10, 2000, false},
			// {310, 1000, 2000, 10, 2000, false},
			// row and col partitioned
			// {311, 1000, 2000, 1000, 10, true}, // FIXME: ArrayIndexOutOfBoundsException in dotProduct
			{312, 1000, 2000, 10, 2000, false},
			// {313, 4000, 2000, 4000, 10, true}, // FIXME: ArrayIndexOutOfBoundsException in dotProduct
			{314, 4000, 2000, 10, 2000, false},

			// combined tests
			{401, 20, 10, 20, 6, true}, // cellwise, rowwise, multiaggregate
			{402, 2000, 2000, 2000, 10, true}, // outerproduct

		});
	}

	@BeforeClass
	public static void init() {
		TestUtils.clearDirectory(TEST_DATA_DIR + TEST_CLASS_DIR);
	}

	@Test
	@Ignore
	public void federatedCodegenMultipleFedMOSingleNode() {
		testFederatedCodegenMultipleFedMO(ExecMode.SINGLE_NODE);
	}

	@Test
	@Ignore
	public void federatedCodegenMultipleFedMOSpark() {
		testFederatedCodegenMultipleFedMO(ExecMode.SPARK);
	}
	
	@Test
	public void federatedCodegenMultipleFedMOHybrid() {
		testFederatedCodegenMultipleFedMO(ExecMode.HYBRID);
	}

	private void testFederatedCodegenMultipleFedMO(ExecMode exec_mode) {
		// store the previous platform config to restore it after the test
		ExecMode platform_old = setExecMode(exec_mode);

		getAndLoadTestConfiguration(TEST_NAME);
		String HOME = SCRIPT_DIR + TEST_DIR;

		int fed_rows_x = rows_x;
		int fed_cols_x = cols_x;
		int fed_rows_y = rows_y;
		int fed_cols_y = cols_y;
		if(row_partitioned) {
			fed_rows_x /= 2;
			fed_rows_y /= 2;
		}
		else {
			fed_cols_x /= 2;
			fed_cols_y /= 2;
		}

		// generate dataset
		// matrix handled by two federated workers
		double[][] X1 = getRandomMatrix(fed_rows_x, fed_cols_x, 0, 1, 0.1, 3);
		double[][] X2 = getRandomMatrix(fed_rows_x, fed_cols_x, 0, 1, 0.1, 23);
		// matrix handled by two federated workers
		double[][] Y1 = getRandomMatrix(fed_rows_y, fed_cols_y, 0, 1, 0.1, 64);
		double[][] Y2 = getRandomMatrix(fed_rows_y, fed_cols_y, 0, 1, 0.1, 135);

		writeInputMatrixWithMTD("X1", X1, false, new MatrixCharacteristics(fed_rows_x, fed_cols_x, BLOCKSIZE, fed_rows_x * fed_cols_x));
		writeInputMatrixWithMTD("X2", X2, false, new MatrixCharacteristics(fed_rows_x, fed_cols_x, BLOCKSIZE, fed_rows_x * fed_cols_x));
		writeInputMatrixWithMTD("Y1", Y1, false, new MatrixCharacteristics(fed_rows_y, fed_cols_y, BLOCKSIZE, fed_rows_y * fed_cols_y));
		writeInputMatrixWithMTD("Y2", Y2, false, new MatrixCharacteristics(fed_rows_y, fed_cols_y, BLOCKSIZE, fed_rows_y * fed_cols_y));

		// empty script name because we don't execute any script, just start the worker
		fullDMLScriptName = "";
		int port1 = getRandomAvailablePort();
		int port2 = getRandomAvailablePort();
		Thread thread1 = startLocalFedWorkerThread(port1, FED_WORKER_WAIT_S);
		Thread thread2 = startLocalFedWorkerThread(port2);

		getAndLoadTestConfiguration(TEST_NAME);

		// Run reference dml script with normal matrix
		fullDMLScriptName = HOME + TEST_NAME + "Reference.dml";
		programArgs = new String[] {"-stats", "-nvargs",
			"in_X1=" + input("X1"), "in_X2=" + input("X2"),
			"in_Y1=" + input("Y1"), "in_Y2=" + input("Y2"),
			"in_rp=" + Boolean.toString(row_partitioned).toUpperCase(),
			"in_test_num=" + Integer.toString(test_num),
			"out_Z=" + expected(OUTPUT_NAME)};
		runTest(true, false, null, -1);

		// Run actual dml script with federated matrix
		fullDMLScriptName = HOME + TEST_NAME + ".dml";
		programArgs = new String[] {"-stats", "-nvargs",
			"in_X1=" + TestUtils.federatedAddress(port1, input("X1")),
			"in_X2=" + TestUtils.federatedAddress(port2, input("X2")),
			"in_Y1=" + TestUtils.federatedAddress(port1, input("Y1")),
			"in_Y2=" + TestUtils.federatedAddress(port2, input("Y2")),
			"in_rp=" + Boolean.toString(row_partitioned).toUpperCase(),
			"in_test_num=" + Integer.toString(test_num),
			"rows_x=" + rows_x, "cols_x=" + cols_x,
			"rows_y=" + rows_y, "cols_y=" + cols_y,
			"out_Z=" + output(OUTPUT_NAME)};
		runTest(true, false, null, -1);

		// compare the results via files
		HashMap<CellIndex, Double> refResults = readDMLMatrixFromExpectedDir(OUTPUT_NAME);
		HashMap<CellIndex, Double> fedResults = readDMLMatrixFromOutputDir(OUTPUT_NAME);
		TestUtils.compareMatrices(fedResults, refResults, TOLERANCE, "Fed", "Ref");

		TestUtils.shutdownThreads(thread1, thread2);

		// check for federated operations
		if(test_num >= 0 && test_num < 100)
			Assert.assertTrue(heavyHittersContainsSubString("fed_spoofCell"));
		else if(test_num < 200)
			Assert.assertTrue(heavyHittersContainsSubString("fed_spoofRA"));
		else if(test_num < 300)
			Assert.assertTrue(heavyHittersContainsSubString("fed_spoofMA"));
		else if(test_num < 400)
			Assert.assertTrue(heavyHittersContainsSubString("fed_spoofOP"));
		else if(test_num == 401) {
			Assert.assertTrue(heavyHittersContainsSubString("fed_spoofRA"));
			Assert.assertTrue(heavyHittersContainsSubString("fed_spoofCell"));
			Assert.assertTrue(heavyHittersContainsSubString("fed_spoofMA", exec_mode == ExecMode.SPARK ? 0 : 1));
		}
		else if(test_num == 402)
			Assert.assertTrue(heavyHittersContainsSubString("fed_spoofOP", 3, exec_mode == ExecMode.SPARK? 1 :2));

		// check that federated input files are still existing
		Assert.assertTrue(HDFSTool.existsFileOnHDFS(input("X1")));
		Assert.assertTrue(HDFSTool.existsFileOnHDFS(input("X2")));
		Assert.assertTrue(HDFSTool.existsFileOnHDFS(input("Y1")));
		Assert.assertTrue(HDFSTool.existsFileOnHDFS(input("Y2")));
		
		resetExecMode(platform_old);
	}

	/**
	 * Override default configuration with custom test configuration to ensure
	 * scratch space and local temporary directory locations are also updated.
	 */
	@Override
	protected File getConfigTemplateFile() {
		// Instrumentation in this test's output log to show custom configuration file used for template.
		File TEST_CONF_FILE = new File(SCRIPT_DIR + TEST_DIR, TEST_CONF);
		return TEST_CONF_FILE;
	}
}
