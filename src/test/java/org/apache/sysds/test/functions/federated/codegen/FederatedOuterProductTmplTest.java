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
public class FederatedOuterProductTmplTest extends AutomatedTestBase
{
	private final static String TEST_NAME = "FederatedOuterProductTmplTest";

	private final static String TEST_DIR = "functions/federated/codegen/";
	private final static String TEST_CLASS_DIR = TEST_DIR + FederatedOuterProductTmplTest.class.getSimpleName() + "/";

	private final static String TEST_CONF = "SystemDS-config-codegen.xml";

	private final static String OUTPUT_NAME = "Z";
	private final static double TOLERANCE = 1e-7;
	private final static int BLOCKSIZE = 1024;

	@Parameterized.Parameter()
	public int test_num;
	@Parameterized.Parameter(1)
	public int rows;
	@Parameterized.Parameter(2)
	public int cols;
	@Parameterized.Parameter(3)
	public boolean row_partitioned;

	@Override
	public void setUp() {
		addTestConfiguration(TEST_NAME, new TestConfiguration(TEST_CLASS_DIR, TEST_NAME, new String[]{OUTPUT_NAME}));
	}

	@Parameterized.Parameters
	public static Collection<Object[]> data() {
		// rows must be even for row partitioned X
		// cols must be even for col partitioned X
		return Arrays.asList(new Object[][] {
			// {test_num, rows, cols, row_partitioned}

			// row partitioned
			{1, 2000, 2000, true},
			{2, 4000, 2000, true},
			{3, 1000, 1000, true},
			{4, 4000, 2000, true},
			// {5, 4000, 2000, true},
			{6, 4000, 2000, true},
			// {7, 2000, 2000, true},
			// {8, 1000, 2000, true},
			{9, 1000, 2000, true},

			// column partitioned
			{1, 2000, 2000, false},
			// {2, 4000, 2000, false},
			// {3, 1000, 1000, false},
			{4, 4000, 2000, false},
			{5, 4000, 2000, false},
			// {6, 4000, 2000, false},
			//FIXME {7, 2000, 2000, false},
			{8, 1000, 2000, false},
			// {9, 1000, 2000, false},
		});
	}

	@BeforeClass
	public static void init() {
		TestUtils.clearDirectory(TEST_DATA_DIR + TEST_CLASS_DIR);
	}

	@Test
	@Ignore
	public void federatedCodegenOuterProductSingleNode() {
		testFederatedCodegenOuterProduct(ExecMode.SINGLE_NODE);
	}
	
	@Test
	@Ignore
	public void federatedCodegenOuterProductSpark() {
		testFederatedCodegenOuterProduct(ExecMode.SPARK);
	}
	
	@Test
	public void federatedCodegenOuterProductHybrid() {
		testFederatedCodegenOuterProduct(ExecMode.HYBRID);
	}
	
	private void testFederatedCodegenOuterProduct(ExecMode exec_mode) {
		// store the previous platform config to restore it after the test
		ExecMode platform_old = setExecMode(exec_mode);

		getAndLoadTestConfiguration(TEST_NAME);
		String HOME = SCRIPT_DIR + TEST_DIR;

		int fed_rows = rows;
		int fed_cols = cols;
		if(row_partitioned)
			fed_rows /= 2;
		else
			fed_cols /= 2;

		// generate dataset
		// matrix handled by two federated workers
		double[][] X1 = getRandomMatrix(fed_rows, fed_cols, 0, 1, 0.1, 3);
		double[][] X2 = getRandomMatrix(fed_rows, fed_cols, 0, 1, 0.1, 7);

		writeInputMatrixWithMTD("X1", X1, false, new MatrixCharacteristics(fed_rows, fed_cols, BLOCKSIZE, fed_rows * fed_cols));
		writeInputMatrixWithMTD("X2", X2, false, new MatrixCharacteristics(fed_rows, fed_cols, BLOCKSIZE, fed_rows * fed_cols));

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
			"in_rp=" + Boolean.toString(row_partitioned).toUpperCase(),
			"in_test_num=" + Integer.toString(test_num),
			"out_Z=" + expected(OUTPUT_NAME)};
		runTest(true, false, null, -1);

		// Run actual dml script with federated matrix
		fullDMLScriptName = HOME + TEST_NAME + ".dml";
		programArgs = new String[] {"-stats", "-nvargs",
			"in_X1=" + TestUtils.federatedAddress(port1, input("X1")),
			"in_X2=" + TestUtils.federatedAddress(port2, input("X2")),
			"in_rp=" + Boolean.toString(row_partitioned).toUpperCase(),
			"in_test_num=" + Integer.toString(test_num),
			"rows=" + rows, "cols=" + cols,
			"out_Z=" + output(OUTPUT_NAME)};
		runTest(true, false, null, -1);

		// compare the results via files
		HashMap<CellIndex, Double> refResults = readDMLMatrixFromExpectedDir(OUTPUT_NAME);
		HashMap<CellIndex, Double> fedResults = readDMLMatrixFromOutputDir(OUTPUT_NAME);
		TestUtils.compareMatrices(fedResults, refResults, TOLERANCE, "Fed", "Ref");

		TestUtils.shutdownThreads(thread1, thread2);

		// check for federated operations
		Assert.assertTrue(heavyHittersContainsSubString("fed_spoofOP"));

		// check that federated input files are still existing
		Assert.assertTrue(HDFSTool.existsFileOnHDFS(input("X1")));
		Assert.assertTrue(HDFSTool.existsFileOnHDFS(input("X2")));
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
