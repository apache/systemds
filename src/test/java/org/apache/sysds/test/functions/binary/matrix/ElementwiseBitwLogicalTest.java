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


package org.apache.sysds.test.functions.binary.matrix;

import org.junit.Test;
import org.apache.sysds.api.DMLScript;
import org.apache.sysds.common.Types.ExecMode;
import org.apache.sysds.lops.LopProperties.ExecType;
import org.apache.sysds.runtime.matrix.data.MatrixValue;
import org.apache.sysds.test.AutomatedTestBase;
import org.apache.sysds.test.TestConfiguration;
import org.apache.sysds.test.TestUtils;

import java.util.HashMap;

public class ElementwiseBitwLogicalTest extends AutomatedTestBase{

	private final static String TEST_NAME1 = "ElementwiseBitwAndTest";
	private final static String TEST_NAME2 = "ElementwiseBitwOrTest";
	private final static String TEST_NAME3 = "ElementwiseBitwXorTest";
	private final static String TEST_NAME4 = "ElementwiseBitwShiftLTest";
	private final static String TEST_NAME5 = "ElementwiseBitwShiftRTest";

	private final static String TEST_DIR   = "functions/binary/matrix/";
	private static final String TEST_CLASS_DIR = TEST_DIR + ElementwiseBitwLogicalTest.class.getSimpleName() + "/";

	private final static int rows = 2100;
	private final static int cols = 70;
	private final static double sparsity1 = 0.9;//dense
	private final static double sparsity2 = 0.1;//sparse
	private final static double eps = 1e-10;

	@Override
	public void setUp() {
		addTestConfiguration(TEST_NAME1, new TestConfiguration(TEST_CLASS_DIR, TEST_NAME1, new String[] { "C" }));
		addTestConfiguration(TEST_NAME2, new TestConfiguration(TEST_CLASS_DIR, TEST_NAME2, new String[] { "C" }));
		addTestConfiguration(TEST_NAME3, new TestConfiguration(TEST_CLASS_DIR, TEST_NAME3, new String[] { "C" }));
		addTestConfiguration(TEST_NAME4, new TestConfiguration(TEST_CLASS_DIR, TEST_NAME4, new String[] { "C" }));
		addTestConfiguration(TEST_NAME5, new TestConfiguration(TEST_CLASS_DIR, TEST_NAME5, new String[] { "C" }));
	}

	@Test
	public void testBitwAndDenseCP() {
		runBitwLogic(TEST_NAME1, false, ExecType.CP);
	}

	@Test
	public void testBitwAndDenseSP() {
		runBitwLogic(TEST_NAME1, false, ExecType.SPARK);
	}

	@Test
	public void testBitwAndSparseCP() {
		runBitwLogic(TEST_NAME1, true, ExecType.CP);
	}

	@Test
	public void testBitwAndSparseSP() {
		runBitwLogic(TEST_NAME1, true, ExecType.SPARK);
	}

	@Test
	public void testBitwOrDenseCP() {
		runBitwLogic(TEST_NAME2, false, ExecType.CP);
	}

	@Test
	public void testBitwOrDenseSP() {
		runBitwLogic(TEST_NAME2, false, ExecType.SPARK);
	}

	@Test
	public void testBitwOrSparseCP() {
		runBitwLogic(TEST_NAME2, true, ExecType.CP);
	}

	@Test
	public void testBitwOrSparseSP() {
		runBitwLogic(TEST_NAME2, true, ExecType.SPARK);
	}

	@Test
	public void testBitwXorDenseCP() {
		runBitwLogic(TEST_NAME3, false, ExecType.CP);
	}

	@Test
	public void testBitwXorDenseSP() {
		runBitwLogic(TEST_NAME3, false, ExecType.SPARK);
	}

	@Test
	public void testBitwXorSparseCP() {
		runBitwLogic(TEST_NAME3, true, ExecType.CP);
	}

	@Test
	public void testBitwXorSparseSP() {
		runBitwLogic(TEST_NAME3, true, ExecType.SPARK);
	}

	@Test
	public void testBitwShiftLDenseCP() {
		runBitwLogic(TEST_NAME4, false, ExecType.CP);
	}

	@Test
	public void testBitwShiftLDenseSP() {
		runBitwLogic(TEST_NAME4, false, ExecType.SPARK);
	}

	@Test
	public void testBitwShiftLSparseCP() {
		runBitwLogic(TEST_NAME4, true, ExecType.CP);
	}

	@Test
	public void testBitwShiftLSparseSP() {
		runBitwLogic(TEST_NAME4, true, ExecType.SPARK);
	}

	@Test
	public void testBitwShiftRDenseCP() {
		runBitwLogic(TEST_NAME5, false, ExecType.CP);
	}

	@Test
	public void testBitwShiftRDenseSP() {
		runBitwLogic(TEST_NAME5, false, ExecType.SPARK);
	}

	@Test
	public void testBitwShiftRSparseCP() {
		runBitwLogic(TEST_NAME5, true, ExecType.CP);
	}

	@Test
	public void testBitwShiftRSparseSP() {
		runBitwLogic(TEST_NAME5, true, ExecType.SPARK);
	}

	private void runBitwLogic(String testname, boolean sparse, ExecType et) {
		//rtplatform for MR
		ExecMode platformOld = rtplatform;

		switch( et ){
			case SPARK: rtplatform = ExecMode.SPARK; break;
			default: rtplatform = ExecMode.HYBRID; break;
		}

		boolean sparkConfigOld = DMLScript.USE_LOCAL_SPARK_CONFIG;
		if( rtplatform == ExecMode.SPARK || rtplatform == ExecMode.HYBRID )
			DMLScript.USE_LOCAL_SPARK_CONFIG = true;

		try {
			String TEST_NAME = testname;
			getAndLoadTestConfiguration(TEST_NAME);

			String HOME = SCRIPT_DIR + TEST_DIR;
			fullDMLScriptName = HOME + TEST_NAME + ".dml";
			programArgs = new String[]{"-explain","-args", input("A"), input("B"), output("C")};

			fullRScriptName = HOME + TEST_NAME + ".R";
			rCmd = "Rscript" + " " + fullRScriptName + " " + inputDir() + " " + expectedDir();

			//get a random matrix of values with 
			double[][] A = getRandomMatrix(rows, cols, 1, 31, sparse ? sparsity1 : sparsity2, 1234);
			double[][] B = getRandomMatrix(rows, cols, 1, 31, sparse ? sparsity1 : sparsity2, 5678);
			writeInputMatrixWithMTD("A", A, true);
			writeInputMatrixWithMTD("B", B, true);
			
			//run tests
			runTest(true, false, null, -1);
			runRScript(true);

			//compare matrices
			HashMap<MatrixValue.CellIndex, Double> dmlfile = readDMLMatrixFromHDFS("C");
			HashMap<MatrixValue.CellIndex, Double> rfile  = readRMatrixFromFS("C");
			TestUtils.compareMatrices(dmlfile, rfile, eps, "Stat-DML", "Stat-R", true);
		}
		finally {
			DMLScript.USE_LOCAL_SPARK_CONFIG = sparkConfigOld;
			rtplatform = platformOld;
		}
	}
}
