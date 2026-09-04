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

package org.apache.sysds.test.functions.ternary;

import java.util.HashMap;

import org.junit.Test;
import org.apache.sysds.api.DMLScript;
import org.apache.sysds.common.Types.ExecMode;
import org.apache.sysds.hops.OptimizerUtils;
import org.apache.sysds.common.Types.ExecType;
import org.apache.sysds.runtime.matrix.data.MatrixValue.CellIndex;
import org.apache.sysds.test.AutomatedTestBase;
import org.apache.sysds.test.TestConfiguration;
import org.apache.sysds.test.TestUtils;

public class FullIfElseTest extends AutomatedTestBase 
{
	private final static String TEST_NAME1 = "TernaryIfElse";
	
	private final static String TEST_DIR = "functions/ternary/";
	private final static String TEST_CLASS_DIR = TEST_DIR + FullIfElseTest.class.getSimpleName() + "/";
	private final static double eps = 1e-10;
	
	private final static int rows = 2111;
	private final static int cols = 300;
	
	private final static double sparsity1 = 0.6;
	private final static double sparsity2 = 0.1;
	
	private enum MatType {
		MATRIX, COL, ROW, SCALAR
	}

	@Override
	public void setUp() {
		TestUtils.clearAssertionInformation();
		addTestConfiguration(TEST_NAME1, new TestConfiguration(TEST_CLASS_DIR, TEST_NAME1, new String[] { "R" }) );
	}

	@Test
	public void testScalarScalarScalarSparseCP() {
		runIfElseTest(MatType.SCALAR, MatType.SCALAR, MatType.SCALAR, true, ExecType.CP);
	}

	@Test
	public void testScalarScalarColSparseCP() {
		runIfElseTest(MatType.SCALAR, MatType.SCALAR, MatType.COL, true, ExecType.CP);
	}

	@Test
	public void testScalarScalarRowSparseCP() {
		runIfElseTest(MatType.SCALAR, MatType.SCALAR, MatType.ROW, true, ExecType.CP);
	}

	@Test
	public void testScalarScalarMatrixSparseCP() {
		runIfElseTest(MatType.SCALAR, MatType.SCALAR, MatType.MATRIX, true, ExecType.CP);
	}

	@Test
	public void testScalarColScalarSparseCP() {
		runIfElseTest(MatType.SCALAR, MatType.COL, MatType.SCALAR, true, ExecType.CP);
	}

	@Test
	public void testScalarColColSparseCP() {
		runIfElseTest(MatType.SCALAR, MatType.COL, MatType.COL, true, ExecType.CP);
	}

	@Test
	public void testScalarColRowSparseCP() {
		runIfElseTest(MatType.SCALAR, MatType.COL, MatType.ROW, true, ExecType.CP);
	}

	@Test
	public void testScalarColMatrixSparseCP() {
		runIfElseTest(MatType.SCALAR, MatType.COL, MatType.MATRIX, true, ExecType.CP);
	}

	@Test
	public void testScalarRowScalarSparseCP() {
		runIfElseTest(MatType.SCALAR, MatType.ROW, MatType.SCALAR, true, ExecType.CP);
	}

	@Test
	public void testScalarRowColSparseCP() {
		runIfElseTest(MatType.SCALAR, MatType.ROW, MatType.COL, true, ExecType.CP);
	}

	@Test
	public void testScalarRowRowSparseCP() {
		runIfElseTest(MatType.SCALAR, MatType.ROW, MatType.ROW, true, ExecType.CP);
	}

	@Test
	public void testScalarRowMatrixSparseCP() {
		runIfElseTest(MatType.SCALAR, MatType.ROW, MatType.MATRIX, true, ExecType.CP);
	}

	@Test
	public void testScalarMatrixScalarSparseCP() {
		runIfElseTest(MatType.SCALAR, MatType.MATRIX, MatType.SCALAR, true, ExecType.CP);
	}

	@Test
	public void testScalarMatrixColSparseCP() {
		runIfElseTest(MatType.SCALAR, MatType.MATRIX, MatType.COL, true, ExecType.CP);
	}

	@Test
	public void testScalarMatrixRowSparseCP() {
		runIfElseTest(MatType.SCALAR, MatType.MATRIX, MatType.ROW, true, ExecType.CP);
	}

	@Test
	public void testScalarMatrixMatrixSparseCP() {
		runIfElseTest(MatType.SCALAR, MatType.MATRIX, MatType.MATRIX, true, ExecType.CP);
	}

	@Test
	public void testColScalarScalarSparseCP() {
		runIfElseTest(MatType.COL, MatType.SCALAR, MatType.SCALAR, true, ExecType.CP);
	}

	@Test
	public void testColScalarColSparseCP() {
		runIfElseTest(MatType.COL, MatType.SCALAR, MatType.COL, true, ExecType.CP);
	}

	@Test
	public void testColScalarRowSparseCP() {
		runIfElseTest(MatType.COL, MatType.SCALAR, MatType.ROW, true, ExecType.CP);
	}

	@Test
	public void testColScalarMatrixSparseCP() {
		runIfElseTest(MatType.COL, MatType.SCALAR, MatType.MATRIX, true, ExecType.CP);
	}

	@Test
	public void testColColScalarSparseCP() {
		runIfElseTest(MatType.COL, MatType.COL, MatType.SCALAR, true, ExecType.CP);
	}

	@Test
	public void testColColColSparseCP() {
		runIfElseTest(MatType.COL, MatType.COL, MatType.COL, true, ExecType.CP);
	}

	@Test
	public void testColColRowSparseCP() {
		runIfElseTest(MatType.COL, MatType.COL, MatType.ROW, true, ExecType.CP);
	}

	@Test
	public void testColColMatrixSparseCP() {
		runIfElseTest(MatType.COL, MatType.COL, MatType.MATRIX, true, ExecType.CP);
	}

	@Test
	public void testColRowScalarSparseCP() {
		runIfElseTest(MatType.COL, MatType.ROW, MatType.SCALAR, true, ExecType.CP);
	}

	@Test
	public void testColRowColSparseCP() {
		runIfElseTest(MatType.COL, MatType.ROW, MatType.COL, true, ExecType.CP);
	}

	@Test
	public void testColRowRowSparseCP() {
		runIfElseTest(MatType.COL, MatType.ROW, MatType.ROW, true, ExecType.CP);
	}

	@Test
	public void testColRowMatrixSparseCP() {
		runIfElseTest(MatType.COL, MatType.ROW, MatType.MATRIX, true, ExecType.CP);
	}

	@Test
	public void testColMatrixScalarSparseCP() {
		runIfElseTest(MatType.COL, MatType.MATRIX, MatType.SCALAR, true, ExecType.CP);
	}

	@Test
	public void testColMatrixColSparseCP() {
		runIfElseTest(MatType.COL, MatType.MATRIX, MatType.COL, true, ExecType.CP);
	}

	@Test
	public void testColMatrixRowSparseCP() {
		runIfElseTest(MatType.COL, MatType.MATRIX, MatType.ROW, true, ExecType.CP);
	}

	@Test
	public void testColMatrixMatrixSparseCP() {
		runIfElseTest(MatType.COL, MatType.MATRIX, MatType.MATRIX, true, ExecType.CP);
	}

	@Test
	public void testRowScalarScalarSparseCP() {
		runIfElseTest(MatType.ROW, MatType.SCALAR, MatType.SCALAR, true, ExecType.CP);
	}

	@Test
	public void testRowScalarColSparseCP() {
		runIfElseTest(MatType.ROW, MatType.SCALAR, MatType.COL, true, ExecType.CP);
	}

	@Test
	public void testRowScalarRowSparseCP() {
		runIfElseTest(MatType.ROW, MatType.SCALAR, MatType.ROW, true, ExecType.CP);
	}

	@Test
	public void testRowScalarMatrixSparseCP() {
		runIfElseTest(MatType.ROW, MatType.SCALAR, MatType.MATRIX, true, ExecType.CP);
	}

	@Test
	public void testRowColScalarSparseCP() {
		runIfElseTest(MatType.ROW, MatType.COL, MatType.SCALAR, true, ExecType.CP);
	}

	@Test
	public void testRowColColSparseCP() {
		runIfElseTest(MatType.ROW, MatType.COL, MatType.COL, true, ExecType.CP);
	}

	@Test
	public void testRowColRowSparseCP() {
		runIfElseTest(MatType.ROW, MatType.COL, MatType.ROW, true, ExecType.CP);
	}

	@Test
	public void testRowColMatrixSparseCP() {
		runIfElseTest(MatType.ROW, MatType.COL, MatType.MATRIX, true, ExecType.CP);
	}

	@Test
	public void testRowRowScalarSparseCP() {
		runIfElseTest(MatType.ROW, MatType.ROW, MatType.SCALAR, true, ExecType.CP);
	}

	@Test
	public void testRowRowColSparseCP() {
		runIfElseTest(MatType.ROW, MatType.ROW, MatType.COL, true, ExecType.CP);
	}

	@Test
	public void testRowRowRowSparseCP() {
		runIfElseTest(MatType.ROW, MatType.ROW, MatType.ROW, true, ExecType.CP);
	}

	@Test
	public void testRowRowMatrixSparseCP() {
		runIfElseTest(MatType.ROW, MatType.ROW, MatType.MATRIX, true, ExecType.CP);
	}

	@Test
	public void testRowMatrixScalarSparseCP() {
		runIfElseTest(MatType.ROW, MatType.MATRIX, MatType.SCALAR, true, ExecType.CP);
	}

	@Test
	public void testRowMatrixColSparseCP() {
		runIfElseTest(MatType.ROW, MatType.MATRIX, MatType.COL, true, ExecType.CP);
	}

	@Test
	public void testRowMatrixRowSparseCP() {
		runIfElseTest(MatType.ROW, MatType.MATRIX, MatType.ROW, true, ExecType.CP);
	}

	@Test
	public void testRowMatrixMatrixSparseCP() {
		runIfElseTest(MatType.ROW, MatType.MATRIX, MatType.MATRIX, true, ExecType.CP);
	}

	@Test
	public void testMatrixScalarScalarSparseCP() {
		runIfElseTest(MatType.MATRIX, MatType.SCALAR, MatType.SCALAR, true, ExecType.CP);
	}

	@Test
	public void testMatrixScalarColSparseCP() {
		runIfElseTest(MatType.MATRIX, MatType.SCALAR, MatType.COL, true, ExecType.CP);
	}

	@Test
	public void testMatrixScalarRowSparseCP() {
		runIfElseTest(MatType.MATRIX, MatType.SCALAR, MatType.ROW, true, ExecType.CP);
	}

	@Test
	public void testMatrixScalarMatrixSparseCP() {
		runIfElseTest(MatType.MATRIX, MatType.SCALAR, MatType.MATRIX, true, ExecType.CP);
	}

	@Test
	public void testMatrixColScalarSparseCP() {
		runIfElseTest(MatType.MATRIX, MatType.COL, MatType.SCALAR, true, ExecType.CP);
	}

	@Test
	public void testMatrixColColSparseCP() {
		runIfElseTest(MatType.MATRIX, MatType.COL, MatType.COL, true, ExecType.CP);
	}

	@Test
	public void testMatrixColRowSparseCP() {
		runIfElseTest(MatType.MATRIX, MatType.COL, MatType.ROW, true, ExecType.CP);
	}

	@Test
	public void testMatrixColMatrixSparseCP() {
		runIfElseTest(MatType.MATRIX, MatType.COL, MatType.MATRIX, true, ExecType.CP);
	}

	@Test
	public void testMatrixRowScalarSparseCP() {
		runIfElseTest(MatType.MATRIX, MatType.ROW, MatType.SCALAR, true, ExecType.CP);
	}

	@Test
	public void testMatrixRowColSparseCP() {
		runIfElseTest(MatType.MATRIX, MatType.ROW, MatType.COL, true, ExecType.CP);
	}

	@Test
	public void testMatrixRowRowSparseCP() {
		runIfElseTest(MatType.MATRIX, MatType.ROW, MatType.ROW, true, ExecType.CP);
	}

	@Test
	public void testMatrixRowMatrixSparseCP() {
		runIfElseTest(MatType.MATRIX, MatType.ROW, MatType.MATRIX, true, ExecType.CP);
	}

	@Test
	public void testMatrixMatrixScalarSparseCP() {
		runIfElseTest(MatType.MATRIX, MatType.MATRIX, MatType.SCALAR, true, ExecType.CP);
	}

	@Test
	public void testMatrixMatrixColSparseCP() {
		runIfElseTest(MatType.MATRIX, MatType.MATRIX, MatType.COL, true, ExecType.CP);
	}

	@Test
	public void testMatrixMatrixRowSparseCP() {
		runIfElseTest(MatType.MATRIX, MatType.MATRIX, MatType.ROW, true, ExecType.CP);
	}

	@Test
	public void testMatrixMatrixMatrixSparseCP() {
		runIfElseTest(MatType.MATRIX, MatType.MATRIX, MatType.MATRIX, true, ExecType.CP);
	}

	@Test
	public void testScalarScalarScalarDenseCP() {
		runIfElseTest(MatType.SCALAR, MatType.SCALAR, MatType.SCALAR, false, ExecType.CP);
	}

	@Test
	public void testScalarScalarColDenseCP() {
		runIfElseTest(MatType.SCALAR, MatType.SCALAR, MatType.COL, false, ExecType.CP);
	}

	@Test
	public void testScalarScalarRowDenseCP() {
		runIfElseTest(MatType.SCALAR, MatType.SCALAR, MatType.ROW, false, ExecType.CP);
	}

	@Test
	public void testScalarScalarMatrixDenseCP() {
		runIfElseTest(MatType.SCALAR, MatType.SCALAR, MatType.MATRIX, false, ExecType.CP);
	}

	@Test
	public void testScalarColScalarDenseCP() {
		runIfElseTest(MatType.SCALAR, MatType.COL, MatType.SCALAR, false, ExecType.CP);
	}

	@Test
	public void testScalarColColDenseCP() {
		runIfElseTest(MatType.SCALAR, MatType.COL, MatType.COL, false, ExecType.CP);
	}

	@Test
	public void testScalarColRowDenseCP() {
		runIfElseTest(MatType.SCALAR, MatType.COL, MatType.ROW, false, ExecType.CP);
	}

	@Test
	public void testScalarColMatrixDenseCP() {
		runIfElseTest(MatType.SCALAR, MatType.COL, MatType.MATRIX, false, ExecType.CP);
	}

	@Test
	public void testScalarRowScalarDenseCP() {
		runIfElseTest(MatType.SCALAR, MatType.ROW, MatType.SCALAR, false, ExecType.CP);
	}

	@Test
	public void testScalarRowColDenseCP() {
		runIfElseTest(MatType.SCALAR, MatType.ROW, MatType.COL, false, ExecType.CP);
	}

	@Test
	public void testScalarRowRowDenseCP() {
		runIfElseTest(MatType.SCALAR, MatType.ROW, MatType.ROW, false, ExecType.CP);
	}

	@Test
	public void testScalarRowMatrixDenseCP() {
		runIfElseTest(MatType.SCALAR, MatType.ROW, MatType.MATRIX, false, ExecType.CP);
	}

	@Test
	public void testScalarMatrixScalarDenseCP() {
		runIfElseTest(MatType.SCALAR, MatType.MATRIX, MatType.SCALAR, false, ExecType.CP);
	}

	@Test
	public void testScalarMatrixColDenseCP() {
		runIfElseTest(MatType.SCALAR, MatType.MATRIX, MatType.COL, false, ExecType.CP);
	}

	@Test
	public void testScalarMatrixRowDenseCP() {
		runIfElseTest(MatType.SCALAR, MatType.MATRIX, MatType.ROW, false, ExecType.CP);
	}

	@Test
	public void testScalarMatrixMatrixDenseCP() {
		runIfElseTest(MatType.SCALAR, MatType.MATRIX, MatType.MATRIX, false, ExecType.CP);
	}

	@Test
	public void testColScalarScalarDenseCP() {
		runIfElseTest(MatType.COL, MatType.SCALAR, MatType.SCALAR, false, ExecType.CP);
	}

	@Test
	public void testColScalarColDenseCP() {
		runIfElseTest(MatType.COL, MatType.SCALAR, MatType.COL, false, ExecType.CP);
	}

	@Test
	public void testColScalarRowDenseCP() {
		runIfElseTest(MatType.COL, MatType.SCALAR, MatType.ROW, false, ExecType.CP);
	}

	@Test
	public void testColScalarMatrixDenseCP() {
		runIfElseTest(MatType.COL, MatType.SCALAR, MatType.MATRIX, false, ExecType.CP);
	}

	@Test
	public void testColColScalarDenseCP() {
		runIfElseTest(MatType.COL, MatType.COL, MatType.SCALAR, false, ExecType.CP);
	}

	@Test
	public void testColColColDenseCP() {
		runIfElseTest(MatType.COL, MatType.COL, MatType.COL, false, ExecType.CP);
	}

	@Test
	public void testColColRowDenseCP() {
		runIfElseTest(MatType.COL, MatType.COL, MatType.ROW, false, ExecType.CP);
	}

	@Test
	public void testColColMatrixDenseCP() {
		runIfElseTest(MatType.COL, MatType.COL, MatType.MATRIX, false, ExecType.CP);
	}

	@Test
	public void testColRowScalarDenseCP() {
		runIfElseTest(MatType.COL, MatType.ROW, MatType.SCALAR, false, ExecType.CP);
	}

	@Test
	public void testColRowColDenseCP() {
		runIfElseTest(MatType.COL, MatType.ROW, MatType.COL, false, ExecType.CP);
	}

	@Test
	public void testColRowRowDenseCP() {
		runIfElseTest(MatType.COL, MatType.ROW, MatType.ROW, false, ExecType.CP);
	}

	@Test
	public void testColRowMatrixDenseCP() {
		runIfElseTest(MatType.COL, MatType.ROW, MatType.MATRIX, false, ExecType.CP);
	}

	@Test
	public void testColMatrixScalarDenseCP() {
		runIfElseTest(MatType.COL, MatType.MATRIX, MatType.SCALAR, false, ExecType.CP);
	}

	@Test
	public void testColMatrixColDenseCP() {
		runIfElseTest(MatType.COL, MatType.MATRIX, MatType.COL, false, ExecType.CP);
	}

	@Test
	public void testColMatrixRowDenseCP() {
		runIfElseTest(MatType.COL, MatType.MATRIX, MatType.ROW, false, ExecType.CP);
	}

	@Test
	public void testColMatrixMatrixDenseCP() {
		runIfElseTest(MatType.COL, MatType.MATRIX, MatType.MATRIX, false, ExecType.CP);
	}

	@Test
	public void testRowScalarScalarDenseCP() {
		runIfElseTest(MatType.ROW, MatType.SCALAR, MatType.SCALAR, false, ExecType.CP);
	}

	@Test
	public void testRowScalarColDenseCP() {
		runIfElseTest(MatType.ROW, MatType.SCALAR, MatType.COL, false, ExecType.CP);
	}

	@Test
	public void testRowScalarRowDenseCP() {
		runIfElseTest(MatType.ROW, MatType.SCALAR, MatType.ROW, false, ExecType.CP);
	}

	@Test
	public void testRowScalarMatrixDenseCP() {
		runIfElseTest(MatType.ROW, MatType.SCALAR, MatType.MATRIX, false, ExecType.CP);
	}

	@Test
	public void testRowColScalarDenseCP() {
		runIfElseTest(MatType.ROW, MatType.COL, MatType.SCALAR, false, ExecType.CP);
	}

	@Test
	public void testRowColColDenseCP() {
		runIfElseTest(MatType.ROW, MatType.COL, MatType.COL, false, ExecType.CP);
	}

	@Test
	public void testRowColRowDenseCP() {
		runIfElseTest(MatType.ROW, MatType.COL, MatType.ROW, false, ExecType.CP);
	}

	@Test
	public void testRowColMatrixDenseCP() {
		runIfElseTest(MatType.ROW, MatType.COL, MatType.MATRIX, false, ExecType.CP);
	}

	@Test
	public void testRowRowScalarDenseCP() {
		runIfElseTest(MatType.ROW, MatType.ROW, MatType.SCALAR, false, ExecType.CP);
	}

	@Test
	public void testRowRowColDenseCP() {
		runIfElseTest(MatType.ROW, MatType.ROW, MatType.COL, false, ExecType.CP);
	}

	@Test
	public void testRowRowRowDenseCP() {
		runIfElseTest(MatType.ROW, MatType.ROW, MatType.ROW, false, ExecType.CP);
	}

	@Test
	public void testRowRowMatrixDenseCP() {
		runIfElseTest(MatType.ROW, MatType.ROW, MatType.MATRIX, false, ExecType.CP);
	}

	@Test
	public void testRowMatrixScalarDenseCP() {
		runIfElseTest(MatType.ROW, MatType.MATRIX, MatType.SCALAR, false, ExecType.CP);
	}

	@Test
	public void testRowMatrixColDenseCP() {
		runIfElseTest(MatType.ROW, MatType.MATRIX, MatType.COL, false, ExecType.CP);
	}

	@Test
	public void testRowMatrixRowDenseCP() {
		runIfElseTest(MatType.ROW, MatType.MATRIX, MatType.ROW, false, ExecType.CP);
	}

	@Test
	public void testRowMatrixMatrixDenseCP() {
		runIfElseTest(MatType.ROW, MatType.MATRIX, MatType.MATRIX, false, ExecType.CP);
	}

	@Test
	public void testMatrixScalarScalarDenseCP() {
		runIfElseTest(MatType.MATRIX, MatType.SCALAR, MatType.SCALAR, false, ExecType.CP);
	}

	@Test
	public void testMatrixScalarColDenseCP() {
		runIfElseTest(MatType.MATRIX, MatType.SCALAR, MatType.COL, false, ExecType.CP);
	}

	@Test
	public void testMatrixScalarRowDenseCP() {
		runIfElseTest(MatType.MATRIX, MatType.SCALAR, MatType.ROW, false, ExecType.CP);
	}

	@Test
	public void testMatrixScalarMatrixDenseCP() {
		runIfElseTest(MatType.MATRIX, MatType.SCALAR, MatType.MATRIX, false, ExecType.CP);
	}

	@Test
	public void testMatrixColScalarDenseCP() {
		runIfElseTest(MatType.MATRIX, MatType.COL, MatType.SCALAR, false, ExecType.CP);
	}

	@Test
	public void testMatrixColColDenseCP() {
		runIfElseTest(MatType.MATRIX, MatType.COL, MatType.COL, false, ExecType.CP);
	}

	@Test
	public void testMatrixColRowDenseCP() {
		runIfElseTest(MatType.MATRIX, MatType.COL, MatType.ROW, false, ExecType.CP);
	}

	@Test
	public void testMatrixColMatrixDenseCP() {
		runIfElseTest(MatType.MATRIX, MatType.COL, MatType.MATRIX, false, ExecType.CP);
	}

	@Test
	public void testMatrixRowScalarDenseCP() {
		runIfElseTest(MatType.MATRIX, MatType.ROW, MatType.SCALAR, false, ExecType.CP);
	}

	@Test
	public void testMatrixRowColDenseCP() {
		runIfElseTest(MatType.MATRIX, MatType.ROW, MatType.COL, false, ExecType.CP);
	}

	@Test
	public void testMatrixRowRowDenseCP() {
		runIfElseTest(MatType.MATRIX, MatType.ROW, MatType.ROW, false, ExecType.CP);
	}

	@Test
	public void testMatrixRowMatrixDenseCP() {
		runIfElseTest(MatType.MATRIX, MatType.ROW, MatType.MATRIX, false, ExecType.CP);
	}

	@Test
	public void testMatrixMatrixScalarDenseCP() {
		runIfElseTest(MatType.MATRIX, MatType.MATRIX, MatType.SCALAR, false, ExecType.CP);
	}

	@Test
	public void testMatrixMatrixColDenseCP() {
		runIfElseTest(MatType.MATRIX, MatType.MATRIX, MatType.COL, false, ExecType.CP);
	}

	@Test
	public void testMatrixMatrixRowDenseCP() {
		runIfElseTest(MatType.MATRIX, MatType.MATRIX, MatType.ROW, false, ExecType.CP);
	}

	@Test
	public void testMatrixMatrixMatrixDenseCP() {
		runIfElseTest(MatType.MATRIX, MatType.MATRIX, MatType.MATRIX, false, ExecType.CP);
	}


	//SPARK
	
	@Test
	public void testScalarScalarScalarDenseSP() {
		runIfElseTest(MatType.SCALAR, MatType.SCALAR, MatType.SCALAR, false, ExecType.SPARK);
	}
	
	@Test
	public void testMatrixScalarScalarDenseSP() {
		runIfElseTest(MatType.MATRIX, MatType.SCALAR, MatType.SCALAR, false, ExecType.SPARK);
	}
	
	@Test
	public void testScalarMatrixScalarDenseSP() {
		runIfElseTest(MatType.SCALAR, MatType.MATRIX, MatType.SCALAR, false, ExecType.SPARK);
	}
	
	@Test
	public void testMatrixMatrixScalarDenseSP() {
		runIfElseTest(MatType.MATRIX, MatType.MATRIX, MatType.SCALAR, false, ExecType.SPARK);
	}
	
	@Test
	public void testScalarScalarMatrixDenseSP() {
		runIfElseTest(MatType.SCALAR, MatType.SCALAR, MatType.MATRIX, false, ExecType.SPARK);
	}
	
	@Test
	public void testMatrixScalarMatrixDenseSP() {
		runIfElseTest(MatType.MATRIX, MatType.SCALAR, MatType.MATRIX, false, ExecType.SPARK);
	}
	
	@Test
	public void testScalarMatrixMatrixDenseSP() {
		runIfElseTest(MatType.SCALAR, MatType.MATRIX, MatType.MATRIX, false, ExecType.SPARK);
	}
	
	@Test
	public void testMatrixMatrixMatrixDenseSP() {
		runIfElseTest(MatType.MATRIX, MatType.MATRIX, MatType.MATRIX, false, ExecType.SPARK);
	}

	@Test
	public void testScalarScalarScalarSparseSP() {
		runIfElseTest(MatType.SCALAR, MatType.SCALAR, MatType.SCALAR, true, ExecType.SPARK);
	}
	
	@Test
	public void testMatrixScalarScalarSparseSP() {
		runIfElseTest(MatType.MATRIX, MatType.SCALAR, MatType.SCALAR, true, ExecType.SPARK);
	}
	
	@Test
	public void testScalarMatrixScalarSparseSP() {
		runIfElseTest(MatType.SCALAR, MatType.MATRIX, MatType.SCALAR, true, ExecType.SPARK);
	}
	
	@Test
	public void testMatrixMatrixScalarSparseSP() {
		runIfElseTest(MatType.MATRIX, MatType.MATRIX, MatType.SCALAR, true, ExecType.SPARK);
	}
	
	@Test
	public void testScalarScalarMatrixSparseSP() {
		runIfElseTest(MatType.SCALAR, MatType.SCALAR, MatType.MATRIX, true, ExecType.SPARK);
	}
	
	@Test
	public void testMatrixScalarMatrixSparseSP() {
		runIfElseTest(MatType.MATRIX, MatType.SCALAR, MatType.MATRIX, true, ExecType.SPARK);
	}
	
	@Test
	public void testScalarMatrixMatrixSparseSP() {
		runIfElseTest(MatType.SCALAR, MatType.MATRIX, MatType.MATRIX, true, ExecType.SPARK);
	}
	
	@Test
	public void testMatrixMatrixMatrixSparseSP() {
		runIfElseTest(MatType.MATRIX, MatType.MATRIX, MatType.MATRIX, true, ExecType.SPARK);
	}

	private void runIfElseTest(MatType mtype1, MatType mtype2, MatType mtype3, boolean sparse, ExecType et){
		setOutputBuffering(true);
		//rtplatform for MR
		ExecMode platformOld = rtplatform;
		switch( et ){
			case SPARK: rtplatform = ExecMode.SPARK; break;
			default: rtplatform = ExecMode.HYBRID; break;
		}
	
		boolean sparkConfigOld = DMLScript.USE_LOCAL_SPARK_CONFIG;
		boolean rewritesOld = OptimizerUtils.ALLOW_ALGEBRAIC_SIMPLIFICATION;
		if( rtplatform == ExecMode.SPARK || rtplatform == ExecMode.HYBRID )
			DMLScript.USE_LOCAL_SPARK_CONFIG = true;
		OptimizerUtils.ALLOW_ALGEBRAIC_SIMPLIFICATION = false; //test runtime ops
		
		try
		{
			TestConfiguration config = getTestConfiguration(TEST_NAME1);
			loadTestConfiguration(config);
			
			String HOME = SCRIPT_DIR + TEST_DIR;
			fullDMLScriptName = HOME + TEST_NAME1 + ".dml";
			programArgs = new String[]{"-explain", "-stats", "-args", input("A"), input("B"), input("C"), output("R")};
			fullRScriptName = HOME + TEST_NAME1 + ".R";
			rCmd = "Rscript" + " " + fullRScriptName + " " + inputDir() + " " + expectedDir();
	
			//generate actual datasets (matrices and scalars)
			double[][] A = getMatrixOfType(mtype1, sparse, 1);
			writeInputMatrixWithMTD("A", A, true);
			double[][] B = getMatrixOfType(mtype2, sparse, 2);
			writeInputMatrixWithMTD("B", B, true);
			double[][] C = getMatrixOfType(mtype3, sparse, 3);
			writeInputMatrixWithMTD("C", C, true);
			
			//run test cases
			runTest(null);
			runRScript(true); 
			
			//compare output matrices 
			HashMap<CellIndex, Double> dmlfile = readDMLMatrixFromOutputDir("R");
			HashMap<CellIndex, Double> rfile  = readRMatrixFromExpectedDir("R");
			TestUtils.compareMatrices(dmlfile, rfile, eps, "Stat-DML", "Stat-R");
		}
		finally {
			rtplatform = platformOld;
			DMLScript.USE_LOCAL_SPARK_CONFIG = sparkConfigOld;
			OptimizerUtils.ALLOW_ALGEBRAIC_SIMPLIFICATION = rewritesOld;
		}
	}
	
	private double[][] getMatrixOfType(MatType mtype, boolean sparse, long seed) {
		double[][] ret = null;
		double sparsity = sparse ? sparsity2 : sparsity1;
		switch(mtype) {
			case SCALAR:
				ret = getRandomMatrix(1, 1, 0, 1, sparsity, seed);
				break;
			case MATRIX:
				ret = getRandomMatrix(rows, cols, 0, 1, sparsity, seed);
				break;
			case COL:
				ret = getRandomMatrix(rows, 1, 0, 1, sparsity, seed);
				break;
			case ROW:
				ret = getRandomMatrix(1, cols, 0, 1, sparsity, seed);
				break;
			default:
		}
		return ret;
	}
}
