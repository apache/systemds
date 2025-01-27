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

import org.apache.sysds.common.Opcodes;
import org.junit.AfterClass;
import org.junit.BeforeClass;
import org.junit.Test;
import org.apache.sysds.api.DMLScript;
import org.apache.sysds.common.Types.ExecMode;
import org.apache.sysds.common.Types.ExecType;
import org.apache.sysds.runtime.matrix.data.MatrixValue.CellIndex;
import org.apache.sysds.runtime.meta.MatrixCharacteristics;
import org.apache.sysds.test.AutomatedTestBase;
import org.apache.sysds.test.TestConfiguration;
import org.apache.sysds.test.TestUtils;

/**
 * TODO cleanup outer(X,Y,z) definition to take two column vectors instead of column and row vector.
 * 
 */
public class FullVectorVectorCellwiseOperationTest extends AutomatedTestBase 
{
	private final static String TEST_NAME = "FullVectorVectorCellwiseOperation";
	private final static String TEST_DIR = "functions/binary/matrix_full_cellwise/";
	private final static String TEST_CLASS_DIR = TEST_DIR + FullVectorVectorCellwiseOperationTest.class.getSimpleName() + "/";
	private final static double eps = 1e-10;
	
	private final static int rows1 = 1001;
	private final static int rows2 = 1009;
	private final static double sparsity1 = 0.7;
	private final static double sparsity2 = 0.01;
	
	private enum OpType{
		ADDITION,
		SUBTRACTION,
		MULTIPLICATION,
		LESS_THAN,
	}
	
	private enum SparsityType{
		DENSE,
		SPARSE,
		EMPTY
	}
	
	@Override
	public void setUp() 
	{
		addTestConfiguration(TEST_NAME, new TestConfiguration(TEST_CLASS_DIR, TEST_NAME, new String[]{"C"}));

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
	public static void cleanUp()
	{
		if (TEST_CACHE_ENABLED) {
			TestUtils.clearDirectory(TEST_DATA_DIR + TEST_CLASS_DIR);
		}
	}
	
	@Test
	public void testAdditionDenseDenseSP() 
	{
		runMatrixVectorCellwiseOperationTest(OpType.ADDITION, SparsityType.DENSE, SparsityType.DENSE, ExecType.SPARK);
	}
	
	@Test
	public void testAdditionDenseSparseSP() 
	{
		runMatrixVectorCellwiseOperationTest(OpType.ADDITION, SparsityType.DENSE, SparsityType.SPARSE, ExecType.SPARK);
	}
	
	@Test
	public void testAdditionDenseEmptySP() 
	{
		runMatrixVectorCellwiseOperationTest(OpType.ADDITION, SparsityType.DENSE, SparsityType.EMPTY, ExecType.SPARK);
	}
	
	@Test
	public void testAdditionSparseDenseSP() 
	{
		runMatrixVectorCellwiseOperationTest(OpType.ADDITION, SparsityType.SPARSE, SparsityType.DENSE, ExecType.SPARK);
	}
	
	@Test
	public void testAdditionSparseSparseSP() 
	{
		runMatrixVectorCellwiseOperationTest(OpType.ADDITION, SparsityType.SPARSE, SparsityType.SPARSE, ExecType.SPARK);
	}
	
	@Test
	public void testAdditionSparseEmptySP() 
	{
		runMatrixVectorCellwiseOperationTest(OpType.ADDITION, SparsityType.SPARSE, SparsityType.EMPTY, ExecType.SPARK);
	}
	
	@Test
	public void testAdditionEmptyDenseSP() 
	{
		runMatrixVectorCellwiseOperationTest(OpType.ADDITION, SparsityType.EMPTY, SparsityType.DENSE, ExecType.SPARK);
	}
	
	@Test
	public void testAdditionEmptySparseSP() 
	{
		runMatrixVectorCellwiseOperationTest(OpType.ADDITION, SparsityType.EMPTY, SparsityType.SPARSE, ExecType.SPARK);
	}
	
	@Test
	public void testAdditionEmptyEmptySP() 
	{
		runMatrixVectorCellwiseOperationTest(OpType.ADDITION, SparsityType.EMPTY, SparsityType.EMPTY, ExecType.SPARK);
	}
	
	@Test
	public void testSubtractionDenseDenseSP() 
	{
		runMatrixVectorCellwiseOperationTest(OpType.SUBTRACTION, SparsityType.DENSE, SparsityType.DENSE, ExecType.SPARK);
	}
	
	@Test
	public void testSubtractionDenseSparseSP() 
	{
		runMatrixVectorCellwiseOperationTest(OpType.SUBTRACTION, SparsityType.DENSE, SparsityType.SPARSE, ExecType.SPARK);
	}
	
	@Test
	public void testSubtractionDenseEmptySP() 
	{
		runMatrixVectorCellwiseOperationTest(OpType.SUBTRACTION, SparsityType.DENSE, SparsityType.EMPTY, ExecType.SPARK);
	}
	
	@Test
	public void testSubtractionSparseDenseSP() 
	{
		runMatrixVectorCellwiseOperationTest(OpType.SUBTRACTION, SparsityType.SPARSE, SparsityType.DENSE, ExecType.SPARK);
	}
	
	@Test
	public void testSubtractionSparseSparseSP() 
	{
		runMatrixVectorCellwiseOperationTest(OpType.SUBTRACTION, SparsityType.SPARSE, SparsityType.SPARSE, ExecType.SPARK);
	}
	
	@Test
	public void testSubtractionSparseEmptySP() 
	{
		runMatrixVectorCellwiseOperationTest(OpType.SUBTRACTION, SparsityType.SPARSE, SparsityType.EMPTY, ExecType.SPARK);
	}
	
	@Test
	public void testSubtractionEmptyDenseSP() 
	{
		runMatrixVectorCellwiseOperationTest(OpType.SUBTRACTION, SparsityType.EMPTY, SparsityType.DENSE, ExecType.SPARK);
	}
	
	@Test
	public void testSubtractionEmptySparseSP() 
	{
		runMatrixVectorCellwiseOperationTest(OpType.SUBTRACTION, SparsityType.EMPTY, SparsityType.SPARSE, ExecType.SPARK);
	}
	
	@Test
	public void testSubtractionEmptyEmptySP() 
	{
		runMatrixVectorCellwiseOperationTest(OpType.SUBTRACTION, SparsityType.EMPTY, SparsityType.EMPTY, ExecType.SPARK);
	}
	
	@Test
	public void testMultiplicationDenseDenseSP() 
	{
		runMatrixVectorCellwiseOperationTest(OpType.MULTIPLICATION, SparsityType.DENSE, SparsityType.DENSE, ExecType.SPARK);
	}
	
	@Test
	public void testMultiplicationDenseSparseSP() 
	{
		runMatrixVectorCellwiseOperationTest(OpType.MULTIPLICATION, SparsityType.DENSE, SparsityType.SPARSE, ExecType.SPARK);
	}
	
	@Test
	public void testMultiplicationDenseEmptySP() 
	{
		runMatrixVectorCellwiseOperationTest(OpType.MULTIPLICATION, SparsityType.DENSE, SparsityType.EMPTY, ExecType.SPARK);
	}
	
	@Test
	public void testMultiplicationSparseDenseSP() 
	{
		runMatrixVectorCellwiseOperationTest(OpType.MULTIPLICATION, SparsityType.SPARSE, SparsityType.DENSE, ExecType.SPARK);
	}
	
	@Test
	public void testMultiplicationSparseSparseSP() 
	{
		runMatrixVectorCellwiseOperationTest(OpType.MULTIPLICATION, SparsityType.SPARSE, SparsityType.SPARSE, ExecType.SPARK);
	}
	
	@Test
	public void testMultiplicationSparseEmptySP() 
	{
		runMatrixVectorCellwiseOperationTest(OpType.MULTIPLICATION, SparsityType.SPARSE, SparsityType.EMPTY, ExecType.SPARK);
	}
	
	@Test
	public void testMultiplicationEmptyDenseSP() 
	{
		runMatrixVectorCellwiseOperationTest(OpType.MULTIPLICATION, SparsityType.EMPTY, SparsityType.DENSE, ExecType.SPARK);
	}
	
	@Test
	public void testMultiplicationEmptySparseSP() 
	{
		runMatrixVectorCellwiseOperationTest(OpType.MULTIPLICATION, SparsityType.EMPTY, SparsityType.SPARSE, ExecType.SPARK);
	}
	
	@Test
	public void testMultiplicationEmptyEmptySP() 
	{
		runMatrixVectorCellwiseOperationTest(OpType.MULTIPLICATION, SparsityType.EMPTY, SparsityType.EMPTY, ExecType.SPARK);
	}
	
	@Test
	public void testLessThanDenseDenseSP() 
	{
		runMatrixVectorCellwiseOperationTest(OpType.LESS_THAN, SparsityType.DENSE, SparsityType.DENSE, ExecType.SPARK);
	}
	
	@Test
	public void testLessThanDenseSparseSP() 
	{
		runMatrixVectorCellwiseOperationTest(OpType.LESS_THAN, SparsityType.DENSE, SparsityType.SPARSE, ExecType.SPARK);
	}
	
	@Test
	public void testLessThanDenseEmptySP() 
	{
		runMatrixVectorCellwiseOperationTest(OpType.LESS_THAN, SparsityType.DENSE, SparsityType.EMPTY, ExecType.SPARK);
	}
	
	@Test
	public void testLessThanSparseDenseSP() 
	{
		runMatrixVectorCellwiseOperationTest(OpType.LESS_THAN, SparsityType.SPARSE, SparsityType.DENSE, ExecType.SPARK);
	}
	
	@Test
	public void testLessThanSparseSparseSP() 
	{
		runMatrixVectorCellwiseOperationTest(OpType.LESS_THAN, SparsityType.SPARSE, SparsityType.SPARSE, ExecType.SPARK);
	}
	
	@Test
	public void testLessThanSparseEmptySP() 
	{
		runMatrixVectorCellwiseOperationTest(OpType.LESS_THAN, SparsityType.SPARSE, SparsityType.EMPTY, ExecType.SPARK);
	}
	
	@Test
	public void testLessThanEmptyDenseSP() 
	{
		runMatrixVectorCellwiseOperationTest(OpType.LESS_THAN, SparsityType.EMPTY, SparsityType.DENSE, ExecType.SPARK);
	}
	
	@Test
	public void testLessThanEmptySparseSP() 
	{
		runMatrixVectorCellwiseOperationTest(OpType.LESS_THAN, SparsityType.EMPTY, SparsityType.SPARSE, ExecType.SPARK);
	}
	
	@Test
	public void testLessThanEmptyEmptySP() 
	{
		runMatrixVectorCellwiseOperationTest(OpType.LESS_THAN, SparsityType.EMPTY, SparsityType.EMPTY, ExecType.SPARK);
	}
	
	// ---------------------------------------------
	@Test
	public void testAdditionDenseDenseCP() 
	{
		runMatrixVectorCellwiseOperationTest(OpType.ADDITION, SparsityType.DENSE, SparsityType.DENSE, ExecType.CP);
	}
	
	@Test
	public void testAdditionDenseSparseCP() 
	{
		runMatrixVectorCellwiseOperationTest(OpType.ADDITION, SparsityType.DENSE, SparsityType.SPARSE, ExecType.CP);
	}
	
	@Test
	public void testAdditionDenseEmptyCP() 
	{
		runMatrixVectorCellwiseOperationTest(OpType.ADDITION, SparsityType.DENSE, SparsityType.EMPTY, ExecType.CP);
	}
	
	@Test
	public void testAdditionSparseDenseCP() 
	{
		runMatrixVectorCellwiseOperationTest(OpType.ADDITION, SparsityType.SPARSE, SparsityType.DENSE, ExecType.CP);
	}
	
	@Test
	public void testAdditionSparseSparseCP() 
	{
		runMatrixVectorCellwiseOperationTest(OpType.ADDITION, SparsityType.SPARSE, SparsityType.SPARSE, ExecType.CP);
	}
	
	@Test
	public void testAdditionSparseEmptyCP() 
	{
		runMatrixVectorCellwiseOperationTest(OpType.ADDITION, SparsityType.SPARSE, SparsityType.EMPTY, ExecType.CP);
	}
	
	@Test
	public void testAdditionEmptyDenseCP() 
	{
		runMatrixVectorCellwiseOperationTest(OpType.ADDITION, SparsityType.EMPTY, SparsityType.DENSE, ExecType.CP);
	}
	
	@Test
	public void testAdditionEmptySparseCP() 
	{
		runMatrixVectorCellwiseOperationTest(OpType.ADDITION, SparsityType.EMPTY, SparsityType.SPARSE, ExecType.CP);
	}
	
	@Test
	public void testAdditionEmptyEmptyCP() 
	{
		runMatrixVectorCellwiseOperationTest(OpType.ADDITION, SparsityType.EMPTY, SparsityType.EMPTY, ExecType.CP);
	}
	
	
	@Test
	public void testSubtractionDenseDenseCP() 
	{
		runMatrixVectorCellwiseOperationTest(OpType.SUBTRACTION, SparsityType.DENSE, SparsityType.DENSE, ExecType.CP);
	}
	
	@Test
	public void testSubtractionDenseSparseCP() 
	{
		runMatrixVectorCellwiseOperationTest(OpType.SUBTRACTION, SparsityType.DENSE, SparsityType.SPARSE, ExecType.CP);
	}
	
	@Test
	public void testSubtractionDenseEmptyCP() 
	{
		runMatrixVectorCellwiseOperationTest(OpType.SUBTRACTION, SparsityType.DENSE, SparsityType.EMPTY, ExecType.CP);
	}
	
	@Test
	public void testSubtractionSparseDenseCP() 
	{
		runMatrixVectorCellwiseOperationTest(OpType.SUBTRACTION, SparsityType.SPARSE, SparsityType.DENSE, ExecType.CP);
	}
	
	@Test
	public void testSubtractionSparseSparseCP() 
	{
		runMatrixVectorCellwiseOperationTest(OpType.SUBTRACTION, SparsityType.SPARSE, SparsityType.SPARSE, ExecType.CP);
	}
	
	@Test
	public void testSubtractionSparseEmptyCP() 
	{
		runMatrixVectorCellwiseOperationTest(OpType.SUBTRACTION, SparsityType.SPARSE, SparsityType.EMPTY, ExecType.CP);
	}
	
	@Test
	public void testSubtractionEmptyDenseCP() 
	{
		runMatrixVectorCellwiseOperationTest(OpType.SUBTRACTION, SparsityType.EMPTY, SparsityType.DENSE, ExecType.CP);
	}
	
	@Test
	public void testSubtractionEmptySparseCP() 
	{
		runMatrixVectorCellwiseOperationTest(OpType.SUBTRACTION, SparsityType.EMPTY, SparsityType.SPARSE, ExecType.CP);
	}
	
	@Test
	public void testSubtractionEmptyEmptyCP() 
	{
		runMatrixVectorCellwiseOperationTest(OpType.SUBTRACTION, SparsityType.EMPTY, SparsityType.EMPTY, ExecType.CP);
	}
	
	@Test
	public void testMultiplicationDenseDenseCP() 
	{
		runMatrixVectorCellwiseOperationTest(OpType.MULTIPLICATION, SparsityType.DENSE, SparsityType.DENSE, ExecType.CP);
	}
	
	@Test
	public void testMultiplicationDenseSparseCP() 
	{
		runMatrixVectorCellwiseOperationTest(OpType.MULTIPLICATION, SparsityType.DENSE, SparsityType.SPARSE, ExecType.CP);
	}
	
	@Test
	public void testMultiplicationDenseEmptyCP() 
	{
		runMatrixVectorCellwiseOperationTest(OpType.MULTIPLICATION, SparsityType.DENSE, SparsityType.EMPTY, ExecType.CP);
	}
	
	@Test
	public void testMultiplicationSparseDenseCP() 
	{
		runMatrixVectorCellwiseOperationTest(OpType.MULTIPLICATION, SparsityType.SPARSE, SparsityType.DENSE, ExecType.CP);
	}
	
	@Test
	public void testMultiplicationSparseSparseCP() 
	{
		runMatrixVectorCellwiseOperationTest(OpType.MULTIPLICATION, SparsityType.SPARSE, SparsityType.SPARSE, ExecType.CP);
	}
	
	@Test
	public void testMultiplicationSparseEmptyCP() 
	{
		runMatrixVectorCellwiseOperationTest(OpType.MULTIPLICATION, SparsityType.SPARSE, SparsityType.EMPTY, ExecType.CP);
	}
	
	@Test
	public void testMultiplicationEmptyDenseCP() 
	{
		runMatrixVectorCellwiseOperationTest(OpType.MULTIPLICATION, SparsityType.EMPTY, SparsityType.DENSE, ExecType.CP);
	}
	
	@Test
	public void testMultiplicationEmptySparseCP() 
	{
		runMatrixVectorCellwiseOperationTest(OpType.MULTIPLICATION, SparsityType.EMPTY, SparsityType.SPARSE, ExecType.CP);
	}
	
	@Test
	public void testMultiplicationEmptyEmptyCP() 
	{
		runMatrixVectorCellwiseOperationTest(OpType.MULTIPLICATION, SparsityType.EMPTY, SparsityType.EMPTY, ExecType.CP);
	}
	
	@Test
	public void testLessThanDenseDenseCP() 
	{
		runMatrixVectorCellwiseOperationTest(OpType.LESS_THAN, SparsityType.DENSE, SparsityType.DENSE, ExecType.CP);
	}
	
	@Test
	public void testLessThanDenseSparseCP() 
	{
		runMatrixVectorCellwiseOperationTest(OpType.LESS_THAN, SparsityType.DENSE, SparsityType.SPARSE, ExecType.CP);
	}
	
	@Test
	public void testLessThanDenseEmptyCP() 
	{
		runMatrixVectorCellwiseOperationTest(OpType.LESS_THAN, SparsityType.DENSE, SparsityType.EMPTY, ExecType.CP);
	}
	
	@Test
	public void testLessThanSparseDenseCP() 
	{
		runMatrixVectorCellwiseOperationTest(OpType.LESS_THAN, SparsityType.SPARSE, SparsityType.DENSE, ExecType.CP);
	}
	
	@Test
	public void testLessThanSparseSparseCP() 
	{
		runMatrixVectorCellwiseOperationTest(OpType.LESS_THAN, SparsityType.SPARSE, SparsityType.SPARSE, ExecType.CP);
	}
	
	@Test
	public void testLessThanSparseEmptyCP() 
	{
		runMatrixVectorCellwiseOperationTest(OpType.LESS_THAN, SparsityType.SPARSE, SparsityType.EMPTY, ExecType.CP);
	}
	
	@Test
	public void testLessThanEmptyDenseCP() 
	{
		runMatrixVectorCellwiseOperationTest(OpType.LESS_THAN, SparsityType.EMPTY, SparsityType.DENSE, ExecType.CP);
	}
	
	@Test
	public void testLessThanEmptySparseCP() 
	{
		runMatrixVectorCellwiseOperationTest(OpType.LESS_THAN, SparsityType.EMPTY, SparsityType.SPARSE, ExecType.CP);
	}
	
	@Test
	public void testLessThanEmptyEmptyCP() 
	{
		runMatrixVectorCellwiseOperationTest(OpType.LESS_THAN, SparsityType.EMPTY, SparsityType.EMPTY, ExecType.CP);
	}
	
	
	private void runMatrixVectorCellwiseOperationTest( OpType type, SparsityType sparseM1, SparsityType sparseM2, ExecType instType)
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
			String opcode = null;
			String opcoder = null;
			switch( type )
			{
				case ADDITION: opcode = Opcodes.PLUS.toString(); opcoder="+";  break;
				case SUBTRACTION: opcode = Opcodes.MINUS.toString();  opcoder="-"; break;
				case MULTIPLICATION: opcode=Opcodes.MULT.toString();  opcoder="mult"; break;
				case LESS_THAN: opcode=Opcodes.LESS.toString(); opcoder="lt"; break;
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
			if (TEST_CACHE_ENABLED)
			{
				TEST_CACHE_DIR = type.ordinal() + "_" + lsparsity1 + "_" + lsparsity2 + "/";
			}
			
			loadTestConfiguration(config, TEST_CACHE_DIR);
			
			String HOME = SCRIPT_DIR + TEST_DIR;
			fullDMLScriptName = HOME + TEST_NAME + ".dml";
			programArgs = new String[]{"-args",
				input("A"), input("B"), opcode, output("C") };
			
			fullRScriptName = HOME + TEST_NAME + ".R";
			rCmd = "Rscript" + " " + fullRScriptName + " " + 
				inputDir() + " " + opcoder + " " + expectedDir();
			
			//generate actual dataset
			double[][] A = getRandomMatrix(rows1, 1, 0, (lsparsity1==0)?0:1, lsparsity1, 7); 
			writeInputMatrixWithMTD("A", A, true);
			double[][] B = getRandomMatrix(1, rows2, 0, (lsparsity2==0)?0:1, lsparsity2, 3); 
			writeInputMatrixWithMTD("B", B, true);
	
			boolean exceptionExpected = false;
			runTest(true, exceptionExpected, null, -1); 
			runRScript(true); 
			
			//compare matrices 
			HashMap<CellIndex, Double> dmlfile = readDMLMatrixFromOutputDir("C");
			HashMap<CellIndex, Double> rfile  = readRMatrixFromExpectedDir("C");
			TestUtils.compareMatrices(dmlfile, rfile, eps, "Stat-DML", "Stat-R");
			checkDMLMetaDataFile("C", new MatrixCharacteristics(rows1,rows2,1,1));
		}
		finally
		{
			rtplatform = platformOld;
			DMLScript.USE_LOCAL_SPARK_CONFIG = sparkConfigOld;
		}
	}		
}