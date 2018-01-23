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

package org.apache.sysml.test.integration.functions.binary.scalar;


import org.apache.sysml.api.DMLScript;
import org.apache.sysml.api.DMLScript.RUNTIME_PLATFORM;
import org.apache.sysml.lops.LopProperties.ExecType;
import org.apache.sysml.runtime.matrix.data.MatrixValue.CellIndex;
import org.apache.sysml.test.integration.AutomatedTestBase;
import org.apache.sysml.test.integration.TestConfiguration;
import org.apache.sysml.test.utils.TestUtils;
import org.junit.Test;

import java.util.HashMap;


public class LogicalTest extends AutomatedTestBase {

	private final static String TEST_NAME1 = "LogicalAndTest";
	private final static String TEST_NAME2 = "LogicalOrTest";
	private final static String TEST_NAME3 = "LogicalXorTest";

	private final static String TEST_DIR   = "functions/binary/scalar/";
	private static final String TEST_CLASS_DIR = TEST_DIR + LogicalTest.class.getSimpleName() + "/";

	private final static int rows = 2100;
	private final static int cols = 70;
	private final static double sparsity1 = 0.1; //sparse
	private final static double sparsity2 = 0.9; //dense
	private final static double eps = 1e-10;

	@Override
	public void setUp() {
		addTestConfiguration(TEST_NAME1, new TestConfiguration(TEST_CLASS_DIR, TEST_NAME1, new String[] { "C" }));
		addTestConfiguration(TEST_NAME2, new TestConfiguration(TEST_CLASS_DIR, TEST_NAME2, new String[] { "C" }));
		addTestConfiguration(TEST_NAME3, new TestConfiguration(TEST_CLASS_DIR, TEST_NAME3, new String[] { "C" }));
	}

	// And Tests
	@Test
	public void testAndDenseCP() {
		runLogical(TEST_NAME1, false, ExecType.CP);
	}

	@Test
	public void testAndSparseCP() {
		runLogical(TEST_NAME1, true, ExecType.CP);
	}

	@Test
	public void testAndDenseSP() {
		runLogical(TEST_NAME1, false, ExecType.SPARK);
	}

	@Test
	public void testAndSparseSP() {
		runLogical(TEST_NAME1, true, ExecType.SPARK);
	}

	@Test
	public void testAndDenseMR() {
		runLogical(TEST_NAME1, false, ExecType.MR);
	}

	@Test
	public void testAndSparseMR() {
		runLogical(TEST_NAME1, true, ExecType.MR);
	}

	//Or Tests
	@Test
	public void testOrDenseCP() {
		runLogical(TEST_NAME2, false, ExecType.CP);
	}

	@Test
	public void testOrSparseCP() {
		runLogical(TEST_NAME2, true, ExecType.CP);
	}

	@Test
	public void testOrDenseSP() {
		runLogical(TEST_NAME2, false, ExecType.SPARK);
	}

	@Test
	public void testOrSparseSP() {
		runLogical(TEST_NAME2, true, ExecType.SPARK);
	}

	@Test
	public void testOrDenseMR() {
		runLogical(TEST_NAME2, false, ExecType.MR);
	}

	@Test
	public void testOrSparseMR() {
		runLogical(TEST_NAME2, true, ExecType.MR);
	}

	//XOR Tests
	@Test
	public void testXorDenseCP() {
		runLogical(TEST_NAME3, false, ExecType.CP);
	}

	@Test
	public void testXorSparseCP() {
		runLogical(TEST_NAME3, true, ExecType.CP);
	}

	@Test
	public void testXorDenseSP() {
		runLogical(TEST_NAME3, false, ExecType.SPARK);
	}

	@Test
	public void testXorSparseSP() {
		runLogical(TEST_NAME3, true, ExecType.SPARK);
	}

	@Test
	public void testXorDenseMR() {
		runLogical(TEST_NAME3, false, ExecType.MR);
	}

	@Test
	public void testXorSparseMR() {
		runLogical(TEST_NAME3, true, ExecType.MR);
	}


	public  void runLogical(String testname, boolean sparse, ExecType et) {
		//rtplatform for MR
		RUNTIME_PLATFORM platformOld = rtplatform;
		switch( et ){
			case MR: rtplatform = RUNTIME_PLATFORM.HADOOP; break;
			case SPARK: rtplatform = RUNTIME_PLATFORM.SPARK; break;
			default: rtplatform = RUNTIME_PLATFORM.HYBRID_SPARK; break;
		}

		boolean sparkConfigOld = DMLScript.USE_LOCAL_SPARK_CONFIG;
		if( rtplatform == RUNTIME_PLATFORM.SPARK || rtplatform == RUNTIME_PLATFORM.HYBRID_SPARK )
			DMLScript.USE_LOCAL_SPARK_CONFIG = true;

		try {
			String TEST_NAME = testname;
			getAndLoadTestConfiguration(TEST_NAME);

			String HOME = SCRIPT_DIR + TEST_DIR;
			fullDMLScriptName = HOME + TEST_NAME + ".dml";
			programArgs = new String[]{"-explain", "-args", input("A"), output("C")};

			fullRScriptName = HOME + TEST_NAME + ".R";
			rCmd = "Rscript" + " " + fullRScriptName + " " + inputDir() + " " + expectedDir();

			//get a random matrix of values with (-0.5, 0, 1)
			double[][] A = getRandomMatrix(rows, cols, -0.5, 1, sparse ? sparsity1:sparsity2, 1234);
			writeInputMatrixWithMTD("A", A, true);

			//run tests
			runTest(true, false, null, -1);
			runRScript(true);

			//compare matrices
			HashMap<CellIndex, Double> dmlfile = readDMLMatrixFromHDFS("C");
			HashMap<CellIndex, Double> rfile   = readRMatrixFromFS("C");
			TestUtils.compareMatrices(dmlfile, rfile, eps, "Stat-DML", "Stat-R");
		}
		finally {
			DMLScript.USE_LOCAL_SPARK_CONFIG = sparkConfigOld;
			rtplatform = platformOld;
		}
	}
}
