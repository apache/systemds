/*
 * Modifications Copyright 2019 Graz University of Technology
 *
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

package org.tugraz.sysds.test.functions.unary.matrix;

import org.junit.Assert;
import org.junit.Test;
import org.tugraz.sysds.api.DMLScript;
import org.tugraz.sysds.common.Types.ExecMode;
import org.tugraz.sysds.runtime.matrix.data.MatrixValue.CellIndex;
import org.tugraz.sysds.runtime.meta.MatrixCharacteristics;
import org.tugraz.sysds.test.AutomatedTestBase;
import org.tugraz.sysds.test.TestConfiguration;

/**
 * Tests for the Cholesky matrix factorization
 * Note:
 * 1. Only tested for dense configuration of matrices (small, & large)
 * 2. The remaining tests for matrix dim check, positive definiteness
 *    already tested at commons-math.
 */

public class CholeskyTest extends AutomatedTestBase
{
	private final static String TEST_NAME = "cholesky";
	private final static String TEST_DIR  = "functions/unary/matrix/";
	private static final String TEST_CLASS_DIR = TEST_DIR + CholeskyTest.class.getSimpleName() + "/";

	private final static int rows1 = 500;
	private final static int rows2 = 2500;
	private final static int cols1 = 500;
	private final static int cols2 = 2500;
	private final static double sparsity = 0.9;

	@Override
	public void setUp() {
		addTestConfiguration(
				TEST_NAME,
				new TestConfiguration(TEST_CLASS_DIR, TEST_NAME,
				new String[] { "D" }) );
	}

	@Test
	public void testCholeskyDenseCP() {
		runTestCholesky( rows1, cols1, ExecMode.SINGLE_NODE );
	}

	@Test
	public void testCholeskyDenseSP() {
		runTestCholesky( rows1, cols1, ExecMode.SPARK );
	}

	@Test
	public void testCholeskyDenseHybrid() {
		runTestCholesky( rows1, cols1, ExecMode.HYBRID );
	}

	@Test
	public void testLargeCholeskyDenseCP() {
		runTestCholesky( rows2, cols2, ExecMode.SINGLE_NODE );
	}

	@Test
	public void testLargeCholeskyDenseSP() {
		runTestCholesky( rows2, cols2, ExecMode.SPARK );
	}

	@Test
	public void testLargeCholeskyDenseHybrid() {
		runTestCholesky( rows2, cols2, ExecMode.HYBRID );
	}

	private void runTestCholesky( int rows, int cols, ExecMode rt) {
		ExecMode rtold = rtplatform;
		rtplatform = rt;
		
		boolean sparkConfigOld = DMLScript.USE_LOCAL_SPARK_CONFIG;
		if( rtplatform == ExecMode.SPARK )
			DMLScript.USE_LOCAL_SPARK_CONFIG = true;
		
		try {
			getAndLoadTestConfiguration(TEST_NAME);
			
			String HOME = SCRIPT_DIR + TEST_DIR;
			fullDMLScriptName = HOME+ TEST_NAME +  ".dml";
			programArgs = new String[]{"-args", input("A"), output("D") };
			
			double[][] A = getRandomMatrix(rows, cols, 0, 1, sparsity, 10);
			MatrixCharacteristics mc = new MatrixCharacteristics(rows, cols, -1, -1);
			writeInputMatrixWithMTD("A", A, false, mc);
			
			//run tests and compare results
			runTest(true, false, null, -1);
			Assert.assertEquals(0, readDMLMatrixFromHDFS("D")
				.get(new CellIndex(1,1)), 1e-5);
		}
		finally {
			DMLScript.USE_LOCAL_SPARK_CONFIG = sparkConfigOld;
			rtplatform = rtold;
		}
	}
}
