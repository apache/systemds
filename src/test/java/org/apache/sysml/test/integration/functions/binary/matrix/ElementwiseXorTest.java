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


package org.apache.sysml.test.integration.functions.binary.matrix;


import org.apache.sysml.api.DMLScript;
import org.apache.sysml.api.DMLScript.RUNTIME_PLATFORM;
import org.apache.sysml.runtime.matrix.MatrixCharacteristics;
import org.apache.sysml.runtime.matrix.data.MatrixValue.CellIndex;
import org.apache.sysml.test.integration.AutomatedTestBase;
import org.apache.sysml.test.integration.TestConfiguration;
import org.junit.Assert;
import org.junit.Test;

import java.util.HashMap;

/**
 * This tests elementwise xor operations
 *
 *
 */
public class ElementwiseXorTest extends AutomatedTestBase{

	private final static String TEST_NAME1 = "ElementwiseXorTest";
	private final static String TEST_DIR   = "functions/binary/matrix/";
	private static final String TEST_CLASS_DIR = TEST_DIR + ElementwiseXorTest.class.getSimpleName() + "/";

	private final static int rows1 = 10;
	private final static int rows2 = 2500;
	private final static int cols1 = 10;
	private final static int cols2 = 2500;
	private final static double sparsity1 = 0.9;//dense
	private final static double sparsity2 = 0.1;//sparse

	@Override
	public void setUp() {
		addTestConfiguration(
				TEST_NAME1,
				new TestConfiguration(TEST_CLASS_DIR, TEST_NAME1,
				new String[] { "D" }));
	}

	@Test
	public void testXorDenseCP() {
		runXor(rows1, cols1, false, RUNTIME_PLATFORM.SINGLE_NODE);
	}

	@Test
	public void testXorDenseSP() {
		runXor(rows1, cols1, false, RUNTIME_PLATFORM.SPARK);
	}

	@Test
	public void testXorDenseMR() {
		runXor(rows1, cols1, false, RUNTIME_PLATFORM.HADOOP);
	}

	@Test
	public void testXorDenseHybrid() {
		runXor(rows1, cols1, false, RUNTIME_PLATFORM.HYBRID_SPARK);
	}

	@Test
	public void testXorSparseCP() {
		runXor(rows1, cols1, true, RUNTIME_PLATFORM.SINGLE_NODE);
	}

	@Test
	public void testXorSparseSP() {
		runXor(rows1, cols1, true, RUNTIME_PLATFORM.SPARK);
	}

	@Test
	public void testXorSparseMR() {
		runXor(rows1, cols1, true, RUNTIME_PLATFORM.HADOOP);
	}

	@Test
	public void testXorSparseHybrid() {
		runXor(rows1, cols1, true, RUNTIME_PLATFORM.HYBRID_SPARK);
	}

	@Test
	public void testLargeXorDenseCP() {
		runXor(rows2, cols2, false, RUNTIME_PLATFORM.SINGLE_NODE);
	}

	@Test
	public void testLargeXorDenseSP() {
		runXor(rows2, cols2, false, RUNTIME_PLATFORM.SPARK);
	}

	@Test
	public void testLargeXorDenseMR() {
		runXor(rows2, cols2, false, RUNTIME_PLATFORM.HADOOP);
	}

	@Test
	public void testLargeXorDenseHybrid() {
		runXor(rows2, cols2, false, RUNTIME_PLATFORM.HYBRID_SPARK);
	}

	@Test
	public void testLargeXorSparseCP() {
		runXor(rows2, cols2, true, RUNTIME_PLATFORM.SINGLE_NODE);
	}

	@Test
	public void testLargeXorSparseSP() {
		runXor(rows2, cols2, true, RUNTIME_PLATFORM.SPARK);
	}

	@Test
	public void testLargeXorSparseMR() {
		runXor(rows2, cols2, true, RUNTIME_PLATFORM.HADOOP);
	}

	@Test
	public void testLargeXorSparseHybrid() {
		runXor(rows2, cols2, true, RUNTIME_PLATFORM.HYBRID_SPARK);
	}

	private void runXor(int rows, int cols, boolean sparsity, RUNTIME_PLATFORM rt) {
		RUNTIME_PLATFORM rtold = rtplatform;
		rtplatform = rt;

		boolean sparkConfigOld = DMLScript.USE_LOCAL_SPARK_CONFIG;
		if (rtplatform == RUNTIME_PLATFORM.SPARK)
			DMLScript.USE_LOCAL_SPARK_CONFIG = true;

		try {

			String TEST_NAME = TEST_NAME1;
			getAndLoadTestConfiguration(TEST_NAME);

			String HOME = SCRIPT_DIR + TEST_DIR;
			fullDMLScriptName = HOME + TEST_NAME + ".dml";
			programArgs = new String[]{"-args", input("A"), input("B"), input("computedMatrix"),output("D")};

			//get a random matrix of values with (-0.5, 0, 1]
			double[][] A = getRandomMatrix(rows, cols, -0.5, 1, sparsity ? sparsity1 : sparsity2, -1);
			double[][] B = getRandomMatrix(rows, cols, -0.5, 1, sparsity ? sparsity1 : sparsity2, -1);

			double[][] A1 = new double[rows][cols];
			double[][] B1 = new double[rows][cols];
			double[][] computedMatrix = new double[rows][cols];

			for(int i = 0; i < rows; i++) {
				for(int j = 0; j < cols; j++) {
					//Ceil tries to round up the cell values to either "0.0", or "1.0"
					A1[i][j] = Math.ceil(A[i][j]);
					B1[i][j] = Math.ceil(B[i][j]);
					computedMatrix[i][j] = (A1[i][j] != B1[i][j]) ? 1 : 0; // use operator(xor, and, not, or), depending on the checking
				}
			}

			MatrixCharacteristics mc = new MatrixCharacteristics(rows, cols, -1, -1, -1);
			writeInputMatrixWithMTD("A", A1, false, mc);
			writeInputMatrixWithMTD("B", B1, false, mc);
			writeInputMatrixWithMTD("computedMatrix", computedMatrix, false, mc);

			//run tests
			runTest(true, false, null, -1);

			Assert.assertEquals(0, readDMLMatrixFromHDFS("D").get(new CellIndex(1,1)), 1e-5);
		}
		finally {
			DMLScript.USE_LOCAL_SPARK_CONFIG = sparkConfigOld;
			rtplatform = rtold;
		}
	}
}
