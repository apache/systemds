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

/*
 * Testing following logical operations:
 *   >, <, ==, !=, >=, <=
 * with a row/col partitioned federated matrix X and a scalar/vector/matrix Y
*/

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
	private final static int BLOCKSIZE = 1024;

	private enum Type {
		GREATER,
		LESS,
		EQUALS,
		NOT_EQUALS,
		GREATER_EQUALS,
		LESS_EQUALS
	}

	private enum FederationType {
		SINGLE_FED_WORKER,
		ROW_PARTITIONED,
		COL_PARTITIONED,
		FULL_PARTITIONED
	}

	private enum YType {
		MATRIX,
		ROW_VEC,
		COL_VEC,
		FED_MAT, // federated matrix Y
		FED_RV, // federated row vector Y
		FED_CV // federated col vector Y
	}

	@Parameterized.Parameter()
	public int rows;
	@Parameterized.Parameter(1)
	public int cols;
	@Parameterized.Parameter(2)
	public double sparsity;
	@Parameterized.Parameter(3)
	public FederationType fed_type;
	@Parameterized.Parameter(4)
	public YType y_type;

	@Override
	public void setUp() {
		addTestConfiguration(SCALAR_TEST_NAME, new TestConfiguration(TEST_CLASS_DIR, SCALAR_TEST_NAME, new String[]{OUTPUT_NAME}));
		addTestConfiguration(MATRIX_TEST_NAME, new TestConfiguration(TEST_CLASS_DIR, MATRIX_TEST_NAME, new String[]{OUTPUT_NAME}));
	}

	@Parameterized.Parameters
	public static Collection<Object[]> data() {
		// rows must be divisable by 4 for row partitioned data
		// cols must be divisable by 4 for col partitioned data
		// rows and cols must be divisable by 2 for full partitioned data
		return Arrays.asList(new Object[][] {
			// {rows, cols, sparsity, fed_type, y_type}

			// row partitioned MM
			{100, 75, 0.01, FederationType.ROW_PARTITIONED, YType.MATRIX},
			{100, 75, 0.9, FederationType.ROW_PARTITIONED, YType.MATRIX},
			// {4, 75, 0.01, FederationType.ROW_PARTITIONED, YType.MATRIX},
			// {4, 75, 0.9, FederationType.ROW_PARTITIONED, YType.MATRIX},
			// {100, 1, 0.01, FederationType.ROW_PARTITIONED, YType.MATRIX},
			// {100, 1, 0.9, FederationType.ROW_PARTITIONED, YType.MATRIX},
			{24, 16, 0.25, FederationType.ROW_PARTITIONED, YType.FED_MAT},

			// row partitioned MV row vector
			{100, 75, 0.01, FederationType.ROW_PARTITIONED, YType.ROW_VEC},
			{100, 75, 0.9, FederationType.ROW_PARTITIONED, YType.ROW_VEC},
			// {4, 75, 0.01, FederationType.ROW_PARTITIONED, YType.ROW_VEC},
			// {4, 75, 0.9, FederationType.ROW_PARTITIONED, YType.ROW_VEC},
			// {100, 1, 0.01, FederationType.ROW_PARTITIONED, YType.ROW_VEC},
			// {100, 1, 0.9, FederationType.ROW_PARTITIONED, YType.ROW_VEC},

			// row partitioned MV col vector
			{100, 75, 0.01, FederationType.ROW_PARTITIONED, YType.COL_VEC},
			{100, 75, 0.9, FederationType.ROW_PARTITIONED, YType.COL_VEC},
			// {4, 75, 0.01, FederationType.ROW_PARTITIONED, YType.COL_VEC},
			// {4, 75, 0.9, FederationType.ROW_PARTITIONED, YType.COL_VEC},
			// {100, 1, 0.01, FederationType.ROW_PARTITIONED, YType.COL_VEC},
			// {100, 1, 0.9, FederationType.ROW_PARTITIONED, YType.COL_VEC},
			{24, 16, 0.25, FederationType.ROW_PARTITIONED, YType.FED_CV},

			// col partitioned MM
			{100, 76, 0.01, FederationType.COL_PARTITIONED, YType.MATRIX},
			{100, 76, 0.9, FederationType.COL_PARTITIONED, YType.MATRIX},
			// {1, 76, 0.01, FederationType.COL_PARTITIONED, YType.MATRIX},
			// {1, 76, 0.9, FederationType.COL_PARTITIONED, YType.MATRIX},
			// {100, 4, 0.01, FederationType.COL_PARTITIONED, YType.MATRIX},
			// {100, 4, 0.9, FederationType.COL_PARTITIONED, YType.MATRIX},
			{24, 16, 0.25, FederationType.COL_PARTITIONED, YType.FED_MAT},

			// col partitioned MV row vector
			{100, 76, 0.01, FederationType.COL_PARTITIONED, YType.ROW_VEC},
			{100, 76, 0.9, FederationType.COL_PARTITIONED, YType.ROW_VEC},
			// {1, 76, 0.01, FederationType.COL_PARTITIONED, YType.ROW_VEC},
			// {1, 76, 0.9, FederationType.COL_PARTITIONED, YType.ROW_VEC},
			// {100, 4, 0.01, FederationType.COL_PARTITIONED, YType.ROW_VEC},
			// {100, 4, 0.9, FederationType.COL_PARTITIONED, YType.ROW_VEC},
			{24, 16, 0.25, FederationType.COL_PARTITIONED, YType.FED_RV},

			// col partitioned MV col vector
			{100, 76, 0.01, FederationType.COL_PARTITIONED, YType.COL_VEC},
			{100, 76, 0.9, FederationType.COL_PARTITIONED, YType.COL_VEC},
			// {1, 76, 0.01, FederationType.COL_PARTITIONED, YType.COL_VEC},
			// {1, 76, 0.9, FederationType.COL_PARTITIONED, YType.COL_VEC},
			// {100, 4, 0.01, FederationType.COL_PARTITIONED, YType.COL_VEC},
			// {100, 4, 0.9, FederationType.COL_PARTITIONED, YType.COL_VEC},

			// single federated worker MM
			{100, 75, 0.01, FederationType.SINGLE_FED_WORKER, YType.MATRIX},
			{100, 75, 0.9, FederationType.SINGLE_FED_WORKER, YType.MATRIX},
			// {1, 75, 0.01, FederationType.SINGLE_FED_WORKER, YType.MATRIX},
			// {1, 75, 0.9, FederationType.SINGLE_FED_WORKER, YType.MATRIX},
			// {100, 1, 0.01, FederationType.SINGLE_FED_WORKER, YType.MATRIX},
			// {100, 1, 0.9, FederationType.SINGLE_FED_WORKER, YType.MATRIX},
			{24, 16, 0.25, FederationType.SINGLE_FED_WORKER, YType.FED_MAT},

			// full partitioned (not supported yet)
			// {70, 80, 0.01, FederationType.FULL_PARTITIONED, YType.MATRIX},
			// {70, 80, 0.9, FederationType.FULL_PARTITIONED, YType.MATRIX},
			// {2, 2, 0.01, FederationType.FULL_PARTITIONED, YType.MATRIX},
			// {2, 2, 0.9, FederationType.FULL_PARTITIONED, YType.MATRIX}
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
	@Ignore
	public void federatedLogicalScalarLessSingleNode() {
		federatedLogicalTest(SCALAR_TEST_NAME, Type.LESS, ExecMode.SINGLE_NODE);
	}

	@Test
	@Ignore
	public void federatedLogicalScalarLessSpark() {
		federatedLogicalTest(SCALAR_TEST_NAME, Type.LESS, ExecMode.SPARK);
	}

	@Test
	@Ignore
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
	@Ignore
	public void federatedLogicalScalarNotEqualsSpark() {
		federatedLogicalTest(SCALAR_TEST_NAME, Type.NOT_EQUALS, ExecMode.SPARK);
	}

	@Test
	@Ignore
	public void federatedLogicalScalarGreaterEqualsSingleNode() {
		federatedLogicalTest(SCALAR_TEST_NAME, Type.GREATER_EQUALS, ExecMode.SINGLE_NODE);
	}

	@Test
	@Ignore
	public void federatedLogicalScalarGreaterEqualsSpark() {
		federatedLogicalTest(SCALAR_TEST_NAME, Type.GREATER_EQUALS, ExecMode.SPARK);
	}

	@Test
	@Ignore
	public void federatedLogicalScalarLessEqualsSingleNode() {
		federatedLogicalTest(SCALAR_TEST_NAME, Type.LESS_EQUALS, ExecMode.SINGLE_NODE);
	}

	@Test
	@Ignore
	public void federatedLogicalScalarLessEqualsSpark() {
		federatedLogicalTest(SCALAR_TEST_NAME, Type.LESS_EQUALS, ExecMode.SPARK);
	}

	//---------------------------MATRIX MATRIX--------------------------
	@Test
	public void federatedLogicalMatrixGreaterSingleNode() {
		federatedLogicalTest(MATRIX_TEST_NAME, Type.GREATER, ExecMode.SINGLE_NODE);
	}

	@Test
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

		int fed_rows = 0;
		int fed_cols = 0;
		switch(fed_type) {
			case SINGLE_FED_WORKER:
				fed_rows = rows;
				fed_cols = cols;
				break;
			case ROW_PARTITIONED:
				fed_rows = rows / 4;
				fed_cols = cols;
				break;
			case COL_PARTITIONED:
				fed_rows = rows;
				fed_cols = cols / 4;
				break;
			case FULL_PARTITIONED:
				fed_rows = rows / 2;
				fed_cols = cols / 2;
				break;
		}

		boolean single_fed_worker = (fed_type == FederationType.SINGLE_FED_WORKER);

		// generate dataset
		// matrix handled by four federated workers
		// X2, X3, X4 not used if single_fed_worker == true
		double[][] X1 = getRandomMatrix(fed_rows, fed_cols, 0, 1, sparsity, 13);
		double[][] X2 = (!single_fed_worker ? getRandomMatrix(fed_rows, fed_cols, 0, 1, sparsity, 2) : null);
		double[][] X3 = (!single_fed_worker ? getRandomMatrix(fed_rows, fed_cols, 0, 1, sparsity, 211) : null);
		double[][] X4 = (!single_fed_worker ? getRandomMatrix(fed_rows, fed_cols, 0, 1, sparsity, 65) : null);

		writeInputMatrixWithMTD("X1", X1, false, new MatrixCharacteristics(fed_rows, fed_cols, BLOCKSIZE, fed_rows * fed_cols));
		if(!single_fed_worker) {
			writeInputMatrixWithMTD("X2", X2, false, new MatrixCharacteristics(fed_rows, fed_cols, BLOCKSIZE, fed_rows * fed_cols));
			writeInputMatrixWithMTD("X3", X3, false, new MatrixCharacteristics(fed_rows, fed_cols, BLOCKSIZE, fed_rows * fed_cols));
			writeInputMatrixWithMTD("X4", X4, false, new MatrixCharacteristics(fed_rows, fed_cols, BLOCKSIZE, fed_rows * fed_cols));
		}

		boolean is_matrix_test = testname.equals(MATRIX_TEST_NAME);

		double[][] Y_mat = null;
		double Y_scal = 0;
		if(is_matrix_test) {
			int y_rows = ((y_type == YType.ROW_VEC || y_type == YType.FED_RV) ? 1 : rows);
			int y_cols = ((y_type == YType.COL_VEC || y_type == YType.FED_CV) ? 1 : cols);

			Y_mat = getRandomMatrix(y_rows, y_cols, 0, 1, sparsity, 5040);
			writeInputMatrixWithMTD("Y", Y_mat, false, new MatrixCharacteristics(y_rows, y_cols, BLOCKSIZE, y_rows * y_cols));
		}

		// empty script name because we don't execute any script, just start the worker
		fullDMLScriptName = "";
		int port1 = getRandomAvailablePort();
		int port2 = (!single_fed_worker ? getRandomAvailablePort() : 0);
		int port3 = (!single_fed_worker ? getRandomAvailablePort() : 0);
		int port4 = (!single_fed_worker ? getRandomAvailablePort() : 0);
		Thread thread1 = startLocalFedWorkerThread(port1, (!single_fed_worker ? FED_WORKER_WAIT_S : FED_WORKER_WAIT));
		Thread thread2 = (!single_fed_worker ? startLocalFedWorkerThread(port2, FED_WORKER_WAIT_S) : null);
		Thread thread3 = (!single_fed_worker ? startLocalFedWorkerThread(port3, FED_WORKER_WAIT_S) : null);
		Thread thread4 = (!single_fed_worker ? startLocalFedWorkerThread(port4) : null);

		getAndLoadTestConfiguration(testname);

		// Run reference dml script with normal matrix
		fullDMLScriptName = HOME + testname + "Reference.dml";
		programArgs = new String[] {"-nvargs",
			"in_X1=" + input("X1"),
			"in_X2=" + (!single_fed_worker ? input("X2") : input("X1")), // not needed in case of a single federated worker
			"in_X3=" + (!single_fed_worker ? input("X3") : input("X1")), // not needed in case of a single federated worker
			"in_X4=" + (!single_fed_worker ? input("X4") : input("X1")), // not needed in case of a single federated worker
			"in_Y=" + (is_matrix_test ? input("Y") : Double.toString(Y_scal)),
			"in_fed_type=" + Integer.toString(fed_type.ordinal()),
			"in_y_type=" + Integer.toString(y_type.ordinal()),
			"in_op_type=" + Integer.toString(op_type.ordinal()),
			"out_Z=" + expected(OUTPUT_NAME)};
		runTest(true, false, null, -1);

		// Run actual dml script with federated matrix
		fullDMLScriptName = HOME + testname + ".dml";
		programArgs = new String[] {"-stats", "-nvargs",
			"in_X1=" + TestUtils.federatedAddress(port1, input("X1")),
			"in_X2=" + (!single_fed_worker ? TestUtils.federatedAddress(port2, input("X2")) : null),
			"in_X3=" + (!single_fed_worker ? TestUtils.federatedAddress(port3, input("X3")) : null),
			"in_X4=" + (!single_fed_worker ? TestUtils.federatedAddress(port4, input("X4")) : null),
			"in_Y=" + (is_matrix_test ? input("Y") : Double.toString(Y_scal)),
			"in_fed_type=" + Integer.toString(fed_type.ordinal()),
			"in_y_type=" + Integer.toString(y_type.ordinal()),
			"in_op_type=" + Integer.toString(op_type.ordinal()),
			"rows=" + Integer.toString(fed_rows), "cols=" + Integer.toString(fed_cols),
			"out_Z=" + output(OUTPUT_NAME)};
		runTest(true, false, null, -1);

		// compare the results via files
		HashMap<CellIndex, Double> refResults  = readDMLMatrixFromExpectedDir(OUTPUT_NAME);
		HashMap<CellIndex, Double> fedResults = readDMLMatrixFromOutputDir(OUTPUT_NAME);
		TestUtils.compareMatrices(fedResults, refResults, TOLERANCE, "Fed", "Ref");

		TestUtils.shutdownThreads(thread1);
		if(!single_fed_worker)
			TestUtils.shutdownThreads(thread2, thread3, thread4);

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
		if(!single_fed_worker) {
			Assert.assertTrue(HDFSTool.existsFileOnHDFS(input("X2")));
			Assert.assertTrue(HDFSTool.existsFileOnHDFS(input("X3")));
			Assert.assertTrue(HDFSTool.existsFileOnHDFS(input("X4")));
		}

		resetExecMode(platform_old);
	}
}
