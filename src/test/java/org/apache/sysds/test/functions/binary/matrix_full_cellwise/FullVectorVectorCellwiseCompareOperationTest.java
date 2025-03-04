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

import java.util.Arrays;
import java.util.HashMap;

import org.apache.sysds.common.Opcodes;
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
public class FullVectorVectorCellwiseCompareOperationTest extends AutomatedTestBase 
{
	
	private final static String TEST_NAME = "FullVectorVectorCellwiseOperation";
	private final static String TEST_DIR = "functions/binary/matrix_full_cellwise/";
	private final static String TEST_CLASS_DIR = TEST_DIR + FullVectorVectorCellwiseCompareOperationTest.class.getSimpleName() + "/";
	private final static double eps = 1e-10;
	
	private final static int rows1 = 1001;
	private final static int rows2 = 1009;
	private final static double sparsity1 = 0.7;
	private final static double sparsity2 = 0.01;
	
	private enum OpType{
		LESS_THAN,
		LESS_THAN_EQUALS,
		GREATER_THAN,
		GREATER_THAN_EQUALS,
		EQUALS,
		NOT_EQUALS
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

	//-------------------------------------
	@Test
	public void testLessThanEqualsDenseEmptyCP() 
	{
		runMatrixVectorCellwiseOperationTest(OpType.LESS_THAN_EQUALS, SparsityType.DENSE, SparsityType.EMPTY, ExecType.CP);
	}
	
	@Test
	public void testGreaterThanSparseDenseCP() 
	{
		runMatrixVectorCellwiseOperationTest(OpType.GREATER_THAN, SparsityType.SPARSE, SparsityType.DENSE, ExecType.CP);
	}
	
	@Test
	public void testGreaterThanEqualsSparseSparseCP() 
	{
		runMatrixVectorCellwiseOperationTest(OpType.GREATER_THAN_EQUALS, SparsityType.SPARSE, SparsityType.SPARSE, ExecType.CP);
	}
	
	@Test
	public void testEqualsSparseEmptyCP() 
	{
		runMatrixVectorCellwiseOperationTest(OpType.EQUALS, SparsityType.SPARSE, SparsityType.EMPTY, ExecType.CP);
	}
	
	@Test
	public void testNotEqualEmptyDenseCP() 
	{
		runMatrixVectorCellwiseOperationTest(OpType.NOT_EQUALS, SparsityType.EMPTY, SparsityType.DENSE, ExecType.CP);
	}
	
	 //-------------------------------------
	
	/**
	 * 
	 * @param sparseM1
	 * @param sparseM2
	 * @param instType
	 */
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
				case LESS_THAN: 			opcode= Opcodes.LESS.toString(); opcoder="lt"; break;
				case LESS_THAN_EQUALS: 		opcode=Opcodes.LESSEQUAL.toString(); opcoder="le"; break;
				case GREATER_THAN: 			opcode=Opcodes.GREATER.toString(); opcoder="gt"; break;
				case GREATER_THAN_EQUALS: 	opcode=Opcodes.GREATEREQUAL.toString(); opcoder="ge"; break;
				case EQUALS: 				opcode=Opcodes.EQUAL.toString(); opcoder="eq"; break;
				case NOT_EQUALS: 			opcode=Opcodes.NOTEQUAL.toString(); opcoder="ne"; break;
			}
			
			getAndLoadTestConfiguration(TEST_NAME);
			
			String HOME = SCRIPT_DIR + TEST_DIR;
			fullDMLScriptName = HOME + TEST_NAME + ".dml";
			programArgs = new String[]{"-stats", "-args", 
				input("A"), input("B"), opcode, output("C") };
			
			fullRScriptName = HOME + TEST_NAME + ".R";
			rCmd = "Rscript" + " " + fullRScriptName + " " + 
				inputDir() + " " + opcoder + " " + expectedDir();
	
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
			
			//generate actual dataset
			double[][] A = getRandomMatrix(rows1, 1, 0, (lsparsity1==0)?0:1, lsparsity1, 7); 
			writeInputMatrixWithMTD("A", A, true);
			double[][] B = getRandomMatrix(1, rows2, 0, (lsparsity2==0)?0:1, lsparsity2, 3);
			Arrays.sort(B[0]);
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
