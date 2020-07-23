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

package org.apache.sysds.test.functions.binary.matrix_full_cellwise;

import java.util.HashMap;

import org.junit.AfterClass;
import org.junit.BeforeClass;
import org.junit.Test;
import org.apache.sysds.api.DMLScript;
import org.apache.sysds.common.Types.ExecMode;
import org.apache.sysds.lops.LopProperties.ExecType;
import org.apache.sysds.runtime.matrix.data.MatrixValue.CellIndex;
import org.apache.sysds.test.AutomatedTestBase;
import org.apache.sysds.test.TestConfiguration;
import org.apache.sysds.test.TestUtils;

public class FullMatrixMatrixCellwiseOperationTest extends AutomatedTestBase 
{
	private final static String TEST_NAME1 = "FullMatrixCellwiseOperation_Addition";
	private final static String TEST_NAME2 = "FullMatrixCellwiseOperation_Substraction";
	private final static String TEST_NAME3 = "FullMatrixCellwiseOperation_Multiplication";
	private final static String TEST_NAME4 = "FullMatrixCellwiseOperation_Division";
	
	private final static String TEST_DIR = "functions/binary/matrix_full_cellwise/";
	private final static String TEST_CLASS_DIR = TEST_DIR + FullMatrixMatrixCellwiseOperationTest.class.getSimpleName() + "/";
	private final static double eps = 1e-10;
	
	private final static int rows = 1010;
	private final static int cols = 300;
	private final static double sparsity1 = 0.7;
	private final static double sparsity2 = 0.1;
	
	private enum OpType{
		ADDITION,
		SUBTRACTION,
		MULTIPLICATION,
		DIVISION
	}
	
	private enum SparsityType{
		DENSE,
		SPARSE,
		EMPTY
	}
	
	@Override
	public void setUp() 
	{
		addTestConfiguration(TEST_NAME1,new TestConfiguration(TEST_CLASS_DIR, TEST_NAME1,new String[]{"C"})); 
		addTestConfiguration(TEST_NAME2,new TestConfiguration(TEST_CLASS_DIR, TEST_NAME2,new String[]{"C"})); 
		addTestConfiguration(TEST_NAME3,new TestConfiguration(TEST_CLASS_DIR, TEST_NAME3,new String[]{"C"})); 
		addTestConfiguration(TEST_NAME4,new TestConfiguration(TEST_CLASS_DIR, TEST_NAME4,new String[]{"C"})); 

		if (TEST_CACHE_ENABLED) {
			setOutAndExpectedDeletionDisabled(true);
		}
	}

	@BeforeClass
	public static void init()
	{
		TestUtils.clearDirectory(TEST_DATA_DIR + TEST_CLASS_DIR);
	}

	@AfterClass
	public static void cleanUp() {
		if (TEST_CACHE_ENABLED) {
			TestUtils.clearDirectory(TEST_DATA_DIR + TEST_CLASS_DIR);
		}
	}

	// ----------------
	
	@Test
	public void testAdditionDenseDenseSP() {
		runMatrixCellwiseOperationTest(OpType.ADDITION, SparsityType.DENSE, SparsityType.DENSE, ExecType.SPARK);
	}
	
	@Test
	public void testAdditionDenseSparseSP() {
		runMatrixCellwiseOperationTest(OpType.ADDITION, SparsityType.DENSE, SparsityType.SPARSE, ExecType.SPARK);
	}
	
	@Test
	public void testAdditionDenseEmptySP() {
		runMatrixCellwiseOperationTest(OpType.ADDITION, SparsityType.DENSE, SparsityType.EMPTY, ExecType.SPARK);
	}
	
	@Test
	public void testAdditionSparseDenseSP() {
		runMatrixCellwiseOperationTest(OpType.ADDITION, SparsityType.SPARSE, SparsityType.DENSE, ExecType.SPARK);
	}
	
	@Test
	public void testAdditionSparseSparseSP() {
		runMatrixCellwiseOperationTest(OpType.ADDITION, SparsityType.SPARSE, SparsityType.SPARSE, ExecType.SPARK);
	}
	
	@Test
	public void testAdditionSparseEmptySP() {
		runMatrixCellwiseOperationTest(OpType.ADDITION, SparsityType.SPARSE, SparsityType.EMPTY, ExecType.SPARK);
	}
	
	@Test
	public void testAdditionEmptyDenseSP() {
		runMatrixCellwiseOperationTest(OpType.ADDITION, SparsityType.EMPTY, SparsityType.DENSE, ExecType.SPARK);
	}
	
	@Test
	public void testAdditionEmptySparseSP() {
		runMatrixCellwiseOperationTest(OpType.ADDITION, SparsityType.EMPTY, SparsityType.SPARSE, ExecType.SPARK);
	}
	
	@Test
	public void testAdditionEmptyEmptySP() {
		runMatrixCellwiseOperationTest(OpType.ADDITION, SparsityType.EMPTY, SparsityType.EMPTY, ExecType.SPARK);
	}
	
	@Test
	public void testSubtractionDenseDenseSP() {
		runMatrixCellwiseOperationTest(OpType.SUBTRACTION, SparsityType.DENSE, SparsityType.DENSE, ExecType.SPARK);
	}
	
	@Test
	public void testSubtractionDenseSparseSP() {
		runMatrixCellwiseOperationTest(OpType.SUBTRACTION, SparsityType.DENSE, SparsityType.SPARSE, ExecType.SPARK);
	}
	
	@Test
	public void testSubtractionDenseEmptySP() {
		runMatrixCellwiseOperationTest(OpType.SUBTRACTION, SparsityType.DENSE, SparsityType.EMPTY, ExecType.SPARK);
	}
	
	@Test
	public void testSubtractionSparseDenseSP() {
		runMatrixCellwiseOperationTest(OpType.SUBTRACTION, SparsityType.SPARSE, SparsityType.DENSE, ExecType.SPARK);
	}
	
	@Test
	public void testSubtractionSparseSparseSP() {
		runMatrixCellwiseOperationTest(OpType.SUBTRACTION, SparsityType.SPARSE, SparsityType.SPARSE, ExecType.SPARK);
	}
	
	@Test
	public void testSubtractionSparseEmptySP() {
		runMatrixCellwiseOperationTest(OpType.SUBTRACTION, SparsityType.SPARSE, SparsityType.EMPTY, ExecType.SPARK);
	}
	
	@Test
	public void testSubtractionEmptyDenseSP() {
		runMatrixCellwiseOperationTest(OpType.SUBTRACTION, SparsityType.EMPTY, SparsityType.DENSE, ExecType.SPARK);
	}
	
	@Test
	public void testSubtractionEmptySparseSP() {
		runMatrixCellwiseOperationTest(OpType.SUBTRACTION, SparsityType.EMPTY, SparsityType.SPARSE, ExecType.SPARK);
	}
	
	@Test
	public void testSubtractionEmptyEmptySP() {
		runMatrixCellwiseOperationTest(OpType.SUBTRACTION, SparsityType.EMPTY, SparsityType.EMPTY, ExecType.SPARK);
	}
	
	@Test
	public void testMultiplicationDenseDenseSP() {
		runMatrixCellwiseOperationTest(OpType.MULTIPLICATION, SparsityType.DENSE, SparsityType.DENSE, ExecType.SPARK);
	}
	
	@Test
	public void testMultiplicationDenseSparseSP() {
		runMatrixCellwiseOperationTest(OpType.MULTIPLICATION, SparsityType.DENSE, SparsityType.SPARSE, ExecType.SPARK);
	}
	
	@Test
	public void testMultiplicationDenseEmptySP() {
		runMatrixCellwiseOperationTest(OpType.MULTIPLICATION, SparsityType.DENSE, SparsityType.EMPTY, ExecType.SPARK);
	}
	
	@Test
	public void testMultiplicationSparseDenseSP() {
		runMatrixCellwiseOperationTest(OpType.MULTIPLICATION, SparsityType.SPARSE, SparsityType.DENSE, ExecType.SPARK);
	}
	
	@Test
	public void testMultiplicationSparseSparseSP() {
		runMatrixCellwiseOperationTest(OpType.MULTIPLICATION, SparsityType.SPARSE, SparsityType.SPARSE, ExecType.SPARK);
	}
	
	@Test
	public void testMultiplicationSparseEmptySP() {
		runMatrixCellwiseOperationTest(OpType.MULTIPLICATION, SparsityType.SPARSE, SparsityType.EMPTY, ExecType.SPARK);
	}
	
	@Test
	public void testMultiplicationEmptyDenseSP() {
		runMatrixCellwiseOperationTest(OpType.MULTIPLICATION, SparsityType.EMPTY, SparsityType.DENSE, ExecType.SPARK);
	}
	
	@Test
	public void testMultiplicationEmptySparseSP() {
		runMatrixCellwiseOperationTest(OpType.MULTIPLICATION, SparsityType.EMPTY, SparsityType.SPARSE, ExecType.SPARK);
	}
	
	@Test
	public void testMultiplicationEmptyEmptySP() {
		runMatrixCellwiseOperationTest(OpType.MULTIPLICATION, SparsityType.EMPTY, SparsityType.EMPTY, ExecType.SPARK);
	}
	
	@Test
	public void testDivisionDenseDenseSP() {
		runMatrixCellwiseOperationTest(OpType.DIVISION, SparsityType.DENSE, SparsityType.DENSE, ExecType.SPARK);
	}
	
	@Test
	public void testDivisionDenseSparseSP() {
		runMatrixCellwiseOperationTest(OpType.DIVISION, SparsityType.DENSE, SparsityType.SPARSE, ExecType.SPARK);
	}
	
	@Test
	public void testDivisionDenseEmptySP() {
		runMatrixCellwiseOperationTest(OpType.DIVISION, SparsityType.DENSE, SparsityType.EMPTY, ExecType.SPARK);
	}
	
	@Test
	public void testDivisionSparseDenseSP() {
		runMatrixCellwiseOperationTest(OpType.DIVISION, SparsityType.SPARSE, SparsityType.DENSE, ExecType.SPARK);
	}
	
	@Test
	public void testDivisionSparseSparseSP() {
		runMatrixCellwiseOperationTest(OpType.DIVISION, SparsityType.SPARSE, SparsityType.SPARSE, ExecType.SPARK);
	}
	
	@Test
	public void testDivisionSparseEmptySP() {
		runMatrixCellwiseOperationTest(OpType.DIVISION, SparsityType.SPARSE, SparsityType.EMPTY, ExecType.SPARK);
	}
	
	@Test
	public void testDivisionEmptyDenseSP() {
		runMatrixCellwiseOperationTest(OpType.DIVISION, SparsityType.EMPTY, SparsityType.DENSE, ExecType.SPARK);
	}
	
	@Test
	public void testDivisionEmptySparseSP() {
		runMatrixCellwiseOperationTest(OpType.DIVISION, SparsityType.EMPTY, SparsityType.SPARSE, ExecType.SPARK);
	}
	
	@Test
	public void testDivisionEmptyEmptySP() {
		runMatrixCellwiseOperationTest(OpType.DIVISION, SparsityType.EMPTY, SparsityType.EMPTY, ExecType.SPARK);
	}
	
	// ----------------
	
	@Test
	public void testAdditionDenseDenseCP() {
		runMatrixCellwiseOperationTest(OpType.ADDITION, SparsityType.DENSE, SparsityType.DENSE, ExecType.CP);
	}
	
	@Test
	public void testAdditionDenseSparseCP() {
		runMatrixCellwiseOperationTest(OpType.ADDITION, SparsityType.DENSE, SparsityType.SPARSE, ExecType.CP);
	}
	
	@Test
	public void testAdditionDenseEmptyCP() {
		runMatrixCellwiseOperationTest(OpType.ADDITION, SparsityType.DENSE, SparsityType.EMPTY, ExecType.CP);
	}
	
	@Test
	public void testAdditionSparseDenseCP() {
		runMatrixCellwiseOperationTest(OpType.ADDITION, SparsityType.SPARSE, SparsityType.DENSE, ExecType.CP);
	}
	
	@Test
	public void testAdditionSparseSparseCP() {
		runMatrixCellwiseOperationTest(OpType.ADDITION, SparsityType.SPARSE, SparsityType.SPARSE, ExecType.CP);
	}
	
	@Test
	public void testAdditionSparseEmptyCP() {
		runMatrixCellwiseOperationTest(OpType.ADDITION, SparsityType.SPARSE, SparsityType.EMPTY, ExecType.CP);
	}
	
	@Test
	public void testAdditionEmptyDenseCP() {
		runMatrixCellwiseOperationTest(OpType.ADDITION, SparsityType.EMPTY, SparsityType.DENSE, ExecType.CP);
	}
	
	@Test
	public void testAdditionEmptySparseCP() {
		runMatrixCellwiseOperationTest(OpType.ADDITION, SparsityType.EMPTY, SparsityType.SPARSE, ExecType.CP);
	}
	
	@Test
	public void testAdditionEmptyEmptyCP() {
		runMatrixCellwiseOperationTest(OpType.ADDITION, SparsityType.EMPTY, SparsityType.EMPTY, ExecType.CP);
	}
	
	@Test
	public void testSubtractionDenseDenseCP() {
		runMatrixCellwiseOperationTest(OpType.SUBTRACTION, SparsityType.DENSE, SparsityType.DENSE, ExecType.CP);
	}
	
	@Test
	public void testSubtractionDenseSparseCP() {
		runMatrixCellwiseOperationTest(OpType.SUBTRACTION, SparsityType.DENSE, SparsityType.SPARSE, ExecType.CP);
	}
	
	@Test
	public void testSubtractionDenseEmptyCP() {
		runMatrixCellwiseOperationTest(OpType.SUBTRACTION, SparsityType.DENSE, SparsityType.EMPTY, ExecType.CP);
	}
	
	@Test
	public void testSubtractionSparseDenseCP() {
		runMatrixCellwiseOperationTest(OpType.SUBTRACTION, SparsityType.SPARSE, SparsityType.DENSE, ExecType.CP);
	}
	
	@Test
	public void testSubtractionSparseSparseCP() {
		runMatrixCellwiseOperationTest(OpType.SUBTRACTION, SparsityType.SPARSE, SparsityType.SPARSE, ExecType.CP);
	}
	
	@Test
	public void testSubtractionSparseEmptyCP() {
		runMatrixCellwiseOperationTest(OpType.SUBTRACTION, SparsityType.SPARSE, SparsityType.EMPTY, ExecType.CP);
	}
	
	@Test
	public void testSubtractionEmptyDenseCP() {
		runMatrixCellwiseOperationTest(OpType.SUBTRACTION, SparsityType.EMPTY, SparsityType.DENSE, ExecType.CP);
	}
	
	@Test
	public void testSubtractionEmptySparseCP() {
		runMatrixCellwiseOperationTest(OpType.SUBTRACTION, SparsityType.EMPTY, SparsityType.SPARSE, ExecType.CP);
	}
	
	@Test
	public void testSubtractionEmptyEmptyCP() {
		runMatrixCellwiseOperationTest(OpType.SUBTRACTION, SparsityType.EMPTY, SparsityType.EMPTY, ExecType.CP);
	}
	
	@Test
	public void testMultiplicationDenseDenseCP() {
		runMatrixCellwiseOperationTest(OpType.MULTIPLICATION, SparsityType.DENSE, SparsityType.DENSE, ExecType.CP);
	}
	
	@Test
	public void testMultiplicationDenseSparseCP() {
		runMatrixCellwiseOperationTest(OpType.MULTIPLICATION, SparsityType.DENSE, SparsityType.SPARSE, ExecType.CP);
	}
	
	@Test
	public void testMultiplicationDenseEmptyCP() {
		runMatrixCellwiseOperationTest(OpType.MULTIPLICATION, SparsityType.DENSE, SparsityType.EMPTY, ExecType.CP);
	}
	
	@Test
	public void testMultiplicationSparseDenseCP() {
		runMatrixCellwiseOperationTest(OpType.MULTIPLICATION, SparsityType.SPARSE, SparsityType.DENSE, ExecType.CP);
	}
	
	@Test
	public void testMultiplicationSparseSparseCP() {
		runMatrixCellwiseOperationTest(OpType.MULTIPLICATION, SparsityType.SPARSE, SparsityType.SPARSE, ExecType.CP);
	}
	
	@Test
	public void testMultiplicationSparseEmptyCP() {
		runMatrixCellwiseOperationTest(OpType.MULTIPLICATION, SparsityType.SPARSE, SparsityType.EMPTY, ExecType.CP);
	}
	
	@Test
	public void testMultiplicationEmptyDenseCP() {
		runMatrixCellwiseOperationTest(OpType.MULTIPLICATION, SparsityType.EMPTY, SparsityType.DENSE, ExecType.CP);
	}
	
	@Test
	public void testMultiplicationEmptySparseCP() {
		runMatrixCellwiseOperationTest(OpType.MULTIPLICATION, SparsityType.EMPTY, SparsityType.SPARSE, ExecType.CP);
	}
	
	@Test
	public void testMultiplicationEmptyEmptyCP() {
		runMatrixCellwiseOperationTest(OpType.MULTIPLICATION, SparsityType.EMPTY, SparsityType.EMPTY, ExecType.CP);
	}
	
	@Test
	public void testDivisionDenseDenseCP() {
		runMatrixCellwiseOperationTest(OpType.DIVISION, SparsityType.DENSE, SparsityType.DENSE, ExecType.CP);
	}
	
	@Test
	public void testDivisionDenseSparseCP() {
		runMatrixCellwiseOperationTest(OpType.DIVISION, SparsityType.DENSE, SparsityType.SPARSE, ExecType.CP);
	}
	
	@Test
	public void testDivisionDenseEmptyCP() {
		runMatrixCellwiseOperationTest(OpType.DIVISION, SparsityType.DENSE, SparsityType.EMPTY, ExecType.CP);
	}
	
	@Test
	public void testDivisionSparseDenseCP() {
		runMatrixCellwiseOperationTest(OpType.DIVISION, SparsityType.SPARSE, SparsityType.DENSE, ExecType.CP);
	}
	
	@Test
	public void testDivisionSparseSparseCP() {
		runMatrixCellwiseOperationTest(OpType.DIVISION, SparsityType.SPARSE, SparsityType.SPARSE, ExecType.CP);
	}
	
	@Test
	public void testDivisionSparseEmptyCP() {
		runMatrixCellwiseOperationTest(OpType.DIVISION, SparsityType.SPARSE, SparsityType.EMPTY, ExecType.CP);
	}
	
	@Test
	public void testDivisionEmptyDenseCP() {
		runMatrixCellwiseOperationTest(OpType.DIVISION, SparsityType.EMPTY, SparsityType.DENSE, ExecType.CP);
	}
	
	@Test
	public void testDivisionEmptySparseCP() {
		runMatrixCellwiseOperationTest(OpType.DIVISION, SparsityType.EMPTY, SparsityType.SPARSE, ExecType.CP);
	}
	
	@Test
	public void testDivisionEmptyEmptyCP() {
		runMatrixCellwiseOperationTest(OpType.DIVISION, SparsityType.EMPTY, SparsityType.EMPTY, ExecType.CP);
	}
	
	
	private void runMatrixCellwiseOperationTest( OpType type, SparsityType sparseM1, SparsityType sparseM2, ExecType instType)
	{
		//rtplatform for MR
		ExecMode platformOld = rtplatform;
		switch( instType ){
			case SPARK: rtplatform = ExecMode.SPARK; break;
			default: rtplatform = ExecMode.HYBRID; break;
		}
	
		boolean sparkConfigOld = DMLScript.USE_LOCAL_SPARK_CONFIG;
		if( rtplatform == ExecMode.SPARK )
			DMLScript.USE_LOCAL_SPARK_CONFIG = true;
	
		try
		{
			String TEST_NAME = null;
			switch( type )
			{
				case ADDITION: TEST_NAME = TEST_NAME1; break;
				case SUBTRACTION: TEST_NAME = TEST_NAME2; break;
				case MULTIPLICATION: TEST_NAME = TEST_NAME3; break;
				case DIVISION: TEST_NAME = TEST_NAME4; break;
			}
			
			TestConfiguration config = getTestConfiguration(TEST_NAME);
	
			//get sparsity
			double lsparsity1 = 1.0, lsparsity2 = 1.0;
			switch( sparseM1 ){
				case DENSE: lsparsity1 = sparsity1; break;
				case SPARSE: lsparsity1 = sparsity2; break;
				case EMPTY: lsparsity1 = 0.0; break;
			}
			switch( sparseM2 ){
				case DENSE: lsparsity2 = sparsity1; break;
				case SPARSE: lsparsity2 = sparsity2; break;
				case EMPTY: lsparsity2 = 0.0; break;
			}
			
			String TEST_CACHE_DIR = "";
			if (TEST_CACHE_ENABLED && (type != OpType.DIVISION))
			{
				TEST_CACHE_DIR = type.ordinal() + "_" + lsparsity1 + "_" + lsparsity2 + "/";
			}
			
			loadTestConfiguration(config, TEST_CACHE_DIR);
			
			/* This is for running the junit test the new way, i.e., construct the arguments directly */
			String HOME = SCRIPT_DIR + TEST_DIR;
			fullDMLScriptName = HOME + TEST_NAME + ".dml";
			programArgs = new String[]{"-args",
				input("A"), Integer.toString(rows), Integer.toString(cols),
				input("B"), Integer.toString(rows), Integer.toString(cols),
				output("C") };
			
			fullRScriptName = HOME + TEST_NAME + ".R";
			rCmd = "Rscript" + " " + fullRScriptName + " " + inputDir() + " " + expectedDir();
			
			//generate actual dataset
			double[][] A = getRandomMatrix(rows, cols, 0, (lsparsity1==0)?0:1, lsparsity1, 7); 
			writeInputMatrix("A", A, true);
			double[][] B = getRandomMatrix(rows, cols, 0, (lsparsity2==0)?0:1, lsparsity2, 3); 
			writeInputMatrix("B", B, true);
	
			boolean exceptionExpected = false;
			runTest(true, exceptionExpected, null, -1); 
			
			if( !(type==OpType.DIVISION) )
			{
				runRScript(true); 
			
				//compare matrices 
				HashMap<CellIndex, Double> dmlfile = readDMLMatrixFromHDFS("C");
				HashMap<CellIndex, Double> rfile  = readRMatrixFromFS("C");
				TestUtils.compareMatrices(dmlfile, rfile, eps, "Stat-DML", "Stat-R");
			}
			else
			{
				//For division, IEEE 754 defines x/0.0 as INFINITY and 0.0/0.0 as NaN.
				//Java handles this correctly while R always returns 1.0E308 in those cases.
				//Hence, we directly write the expected results.
				
				double C[][] = new double[rows][cols];
				for( int i=0; i<rows; i++ )
					for( int j=0; j<cols; j++ )
						C[i][j] = A[i][j]/B[i][j];
				writeExpectedMatrix("C", C);
				
				compareResults();
			}
		}
		finally {
			rtplatform = platformOld;
			DMLScript.USE_LOCAL_SPARK_CONFIG = sparkConfigOld;
		}
	}
}
