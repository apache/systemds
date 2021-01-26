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

package org.apache.sysds.test.functions.federated.primitives;

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
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.Parameterized;

import java.util.Arrays;
import java.util.Collection;
import java.util.HashMap;

@RunWith(value = Parameterized.class)
@net.jcip.annotations.NotThreadSafe
public class FederatedLogicalTest extends AutomatedTestBase
{
	private final static String SCALAR_TEST_NAME = "FederatedLogicalMatrixScalarTest";
	private final static String MATRIX_TEST_NAME = "FederatedLogicalMatrixMatrixTest";
	private final static String TEST_DIR = "functions/federated/binary/";
	private final static String TEST_CLASS_DIR = TEST_DIR + FederatedLogicalTest.class.getSimpleName() + "/";

	private final static String OUTPUT_NAME = "Z";
	private final static double TOLERANCE = 0;
	private final static int blocksize = 1024;

	public enum Type{
		GREATER,
		LESS,
		EQUALS,
		NOT_EQUALS,
		GREATER_EQUALS,
		LESS_EQUALS
	}

	@Parameterized.Parameter()
	public int rows;
	@Parameterized.Parameter(1)
	public int cols;
	@Parameterized.Parameter(2)
	public double sparsity;

	@Override
	public void setUp() {
		addTestConfiguration(SCALAR_TEST_NAME, new TestConfiguration(TEST_CLASS_DIR, SCALAR_TEST_NAME, new String[]{OUTPUT_NAME}));
		addTestConfiguration(MATRIX_TEST_NAME, new TestConfiguration(TEST_CLASS_DIR, MATRIX_TEST_NAME, new String[]{OUTPUT_NAME}));
	}

	@Parameterized.Parameters
	public static Collection<Object[]> data() {
		// rows must be even
		return Arrays.asList(new Object[][] {
			// {rows, cols, sparsity}
			{100, 75, 0.01},
			{100, 75, 0.9}
		});
	}

	@BeforeClass
	public static void init() {
		TestUtils.clearDirectory(TEST_DATA_DIR + TEST_CLASS_DIR);
	}

	//---------------------------MATRIX SCALAR--------------------------
	@Test
	public void federatedLogicalScalarGreaterSingleNode() {
		federatedLogicalTest(SCALAR_TEST_NAME, Type.GREATER, ExecMode.SINGLE_NODE);
	}

	@Test
	public void federatedLogicalScalarGreaterSpark() {
		federatedLogicalTest(SCALAR_TEST_NAME, Type.GREATER, ExecMode.SPARK);
	}

	@Test
	public void federatedLogicalScalarLessSingleNode() {
		federatedLogicalTest(SCALAR_TEST_NAME, Type.LESS, ExecMode.SINGLE_NODE);
	}

	@Test
	public void federatedLogicalScalarLessSpark() {
		federatedLogicalTest(SCALAR_TEST_NAME, Type.LESS, ExecMode.SPARK);
	}

	@Test
	public void federatedLogicalScalarEqualsSingleNode() {
		federatedLogicalTest(SCALAR_TEST_NAME, Type.EQUALS, ExecMode.SINGLE_NODE);
	}

	@Test
	public void federatedLogicalScalarEqualsSpark() {
		federatedLogicalTest(SCALAR_TEST_NAME, Type.EQUALS, ExecMode.SPARK);
	}

	@Test
	public void federatedLogicalScalarNotEqualsSingleNode() {
		federatedLogicalTest(SCALAR_TEST_NAME, Type.NOT_EQUALS, ExecMode.SINGLE_NODE);
	}

	@Test
	public void federatedLogicalScalarNotEqualsSpark() {
		federatedLogicalTest(SCALAR_TEST_NAME, Type.NOT_EQUALS, ExecMode.SPARK);
	}

	@Test
	public void federatedLogicalScalarGreaterEqualsSingleNode() {
		federatedLogicalTest(SCALAR_TEST_NAME, Type.GREATER_EQUALS, ExecMode.SINGLE_NODE);
	}

	@Test
	public void federatedLogicalScalarGreaterEqualsSpark() {
		federatedLogicalTest(SCALAR_TEST_NAME, Type.GREATER_EQUALS, ExecMode.SPARK);
	}

	@Test
	public void federatedLogicalScalarLessEqualsSingleNode() {
		federatedLogicalTest(SCALAR_TEST_NAME, Type.LESS_EQUALS, ExecMode.SINGLE_NODE);
	}

	@Test
	public void federatedLogicalScalarLessEqualsSpark() {
		federatedLogicalTest(SCALAR_TEST_NAME, Type.LESS_EQUALS, ExecMode.SPARK);
	}

	//---------------------------MATRIX MATRIX--------------------------
	@Test
	@Ignore
	public void federatedLogicalMatrixGreaterSingleNode() {
		federatedLogicalTest(MATRIX_TEST_NAME, Type.GREATER, ExecMode.SINGLE_NODE);
	}

	@Test
	@Ignore
	public void federatedLogicalMatrixGreaterSpark() {
		federatedLogicalTest(MATRIX_TEST_NAME, Type.GREATER, ExecMode.SPARK);
	}

	@Test
	@Ignore
	public void federatedLogicalMatrixLessSingleNode() {
		federatedLogicalTest(MATRIX_TEST_NAME, Type.LESS, ExecMode.SINGLE_NODE);
	}

	@Test
	@Ignore
	public void federatedLogicalMatrixLessSpark() {
		federatedLogicalTest(MATRIX_TEST_NAME, Type.LESS, ExecMode.SPARK);
	}

	@Test
	@Ignore
	public void federatedLogicalMatrixEqualsSingleNode() {
		federatedLogicalTest(MATRIX_TEST_NAME, Type.EQUALS, ExecMode.SINGLE_NODE);
	}

	@Test
	@Ignore
	public void federatedLogicalMatrixEqualsSpark() {
		federatedLogicalTest(MATRIX_TEST_NAME, Type.EQUALS, ExecMode.SPARK);
	}

	@Test
	@Ignore
	public void federatedLogicalMatrixNotEqualsSingleNode() {
		federatedLogicalTest(MATRIX_TEST_NAME, Type.NOT_EQUALS, ExecMode.SINGLE_NODE);
	}

	@Test
	@Ignore
	public void federatedLogicalMatrixNotEqualsSpark() {
		federatedLogicalTest(MATRIX_TEST_NAME, Type.NOT_EQUALS, ExecMode.SPARK);
	}

	@Test
	@Ignore
	public void federatedLogicalMatrixGreaterEqualsSingleNode() {
		federatedLogicalTest(MATRIX_TEST_NAME, Type.GREATER_EQUALS, ExecMode.SINGLE_NODE);
	}

	@Test
	@Ignore
	public void federatedLogicalMatrixGreaterEqualsSpark() {
		federatedLogicalTest(MATRIX_TEST_NAME, Type.GREATER_EQUALS, ExecMode.SPARK);
	}

	@Test
	@Ignore
	public void federatedLogicalMatrixLessEqualsSingleNode() {
		federatedLogicalTest(MATRIX_TEST_NAME, Type.LESS_EQUALS, ExecMode.SINGLE_NODE);
	}

	@Test
	@Ignore
	public void federatedLogicalMatrixLessEqualsSpark() {
		federatedLogicalTest(MATRIX_TEST_NAME, Type.LESS_EQUALS, ExecMode.SPARK);
	}

// -----------------------------------------------------------------------------

	public void federatedLogicalTest(String testname, Type op_type, ExecMode execMode)
	{
		// store the previous platform config to restore it after the test
		ExecMode platform_old = setExecMode(execMode);

		getAndLoadTestConfiguration(testname);
		String HOME = SCRIPT_DIR + TEST_DIR;

		int fed_rows = rows / 2;
		int fed_cols = cols;

		// generate dataset
		// matrix handled by two federated workers
		double[][] X1 = getRandomMatrix(fed_rows, fed_cols, 1, 2, 1, 13);
		double[][] X2 = getRandomMatrix(fed_rows, fed_cols, 1, 2, 1, 2);

		writeInputMatrixWithMTD("X1", X1, false, new MatrixCharacteristics(fed_rows, fed_cols, blocksize, fed_rows * fed_cols));
		writeInputMatrixWithMTD("X2", X2, false, new MatrixCharacteristics(fed_rows, fed_cols, blocksize, fed_rows * fed_cols));

		boolean is_matrix_test = testname.equals(MATRIX_TEST_NAME);

		double[][] Y_mat = null;
		double Y_scal = 0;
		if(is_matrix_test) {
			Y_mat = getRandomMatrix(rows, cols, 0, 1, sparsity, 5040);
			writeInputMatrixWithMTD("Y", Y_mat, true);
		}

		// empty script name because we don't execute any script, just start the worker
		fullDMLScriptName = "";
		int port1 = getRandomAvailablePort();
		int port2 = getRandomAvailablePort();
		Thread thread1 = startLocalFedWorkerThread(port1, FED_WORKER_WAIT_S);
		Thread thread2 = startLocalFedWorkerThread(port2);

		getAndLoadTestConfiguration(testname);

		// Run reference dml script with normal matrix
		fullDMLScriptName = HOME + testname + "Reference.dml";
		programArgs = new String[] {"-nvargs", "in_X1=" + input("X1"), "in_X2=" + input("X2"),
			"in_Y=" + (is_matrix_test ? input("Y") : Double.toString(Y_scal)),
			"in_op_type=" + Integer.toString(op_type.ordinal()),
			"out_Z=" + expected(OUTPUT_NAME)};
		runTest(true, false, null, -1);

		// Run actual dml script with federated matrix
		fullDMLScriptName = HOME + testname + ".dml";
		programArgs = new String[] {"-stats", "-nvargs",
			"in_X1=" + TestUtils.federatedAddress(port1, input("X1")), "in_X2=" + TestUtils.federatedAddress(port2, input("X2")),
			"in_Y=" + (is_matrix_test ? input("Y") : Double.toString(Y_scal)),
			"in_op_type=" + Integer.toString(op_type.ordinal()),
			"rows=" + fed_rows, "cols=" + fed_cols, "out_Z=" + output(OUTPUT_NAME)};
		runTest(true, false, null, -1);

		// compare the results via files
		HashMap<CellIndex, Double> refResults  = readDMLMatrixFromExpectedDir(OUTPUT_NAME);
		HashMap<CellIndex, Double> fedResults = readDMLMatrixFromOutputDir(OUTPUT_NAME);
		TestUtils.compareMatrices(fedResults, refResults, TOLERANCE, "Fed", "Ref");

		TestUtils.shutdownThreads(thread1, thread2);

		// check for federated operations
		switch(op_type)
		{
			case GREATER:
				Assert.assertTrue(heavyHittersContainsString("fed_>"));
				break;
			case LESS:
				Assert.assertTrue(heavyHittersContainsString("fed_<"));
				break;
			case EQUALS:
				Assert.assertTrue(heavyHittersContainsString("fed_=="));
				break;
			case NOT_EQUALS:
				Assert.assertTrue(heavyHittersContainsString("fed_!="));
				break;
			case GREATER_EQUALS:
				Assert.assertTrue(heavyHittersContainsString("fed_>="));
				break;
			case LESS_EQUALS:
				Assert.assertTrue(heavyHittersContainsString("fed_<="));
				break;
		}

		// check that federated input files are still existing
		Assert.assertTrue(HDFSTool.existsFileOnHDFS(input("X1")));
		Assert.assertTrue(HDFSTool.existsFileOnHDFS(input("X2")));

		resetExecMode(platform_old);
	}
}
