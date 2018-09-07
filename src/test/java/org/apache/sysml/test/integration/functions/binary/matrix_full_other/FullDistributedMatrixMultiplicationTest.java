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

package org.apache.sysml.test.integration.functions.binary.matrix_full_other;

import java.util.HashMap;

import org.junit.AfterClass;
import org.junit.BeforeClass;
import org.junit.Test;

import org.apache.sysml.api.DMLScript;
import org.apache.sysml.api.DMLScript.RUNTIME_PLATFORM;
import org.apache.sysml.hops.AggBinaryOp;
import org.apache.sysml.hops.AggBinaryOp.MMultMethod;
import org.apache.sysml.lops.LopProperties.ExecType;
import org.apache.sysml.runtime.matrix.data.MatrixValue.CellIndex;
import org.apache.sysml.test.integration.AutomatedTestBase;
import org.apache.sysml.test.integration.TestConfiguration;
import org.apache.sysml.test.utils.TestUtils;

public class FullDistributedMatrixMultiplicationTest extends AutomatedTestBase 
{
	
	private final static String TEST_NAME = "FullDistributedMatrixMultiplication";
	private final static String TEST_DIR = "functions/binary/matrix_full_other/";
	private final static String TEST_CLASS_DIR = TEST_DIR + FullDistributedMatrixMultiplicationTest.class.getSimpleName() + "/";
	private final static double eps = 1e-10;
	
	private final static int rowsA = 1501;
	private final static int colsA = 1103;
	private final static int rowsB = 1103;
	private final static int colsB = 923;
	
	private final static double sparsity1 = 0.7;
	private final static double sparsity2 = 0.1;
	
	
	@Override
	public void setUp() 
	{
		TestUtils.clearAssertionInformation();
		addTestConfiguration(TEST_NAME, 
			new TestConfiguration(TEST_CLASS_DIR, TEST_NAME, new String[] { "C" }) ); 
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
	public void testDenseDenseMapmmMR() 
	{
		runDistributedMatrixMatrixMultiplicationTest(false, false, MMultMethod.MAPMM_R, ExecType.MR);
	}
	
	@Test
	public void testDenseSparseMapmmMR() 
	{
		runDistributedMatrixMatrixMultiplicationTest(false, true, MMultMethod.MAPMM_R, ExecType.MR);
	}
	
	@Test
	public void testSparseDenseMapmmMR() 
	{
		runDistributedMatrixMatrixMultiplicationTest(true, false, MMultMethod.MAPMM_R, ExecType.MR);
	}
	
	@Test
	public void testSparseSparseMapmmMR() 
	{
		runDistributedMatrixMatrixMultiplicationTest(true, true, MMultMethod.MAPMM_R, ExecType.MR);
	}
	
	@Test
	public void testDenseDenseCpmmMR() 
	{
		runDistributedMatrixMatrixMultiplicationTest(false, false, MMultMethod.CPMM, ExecType.MR);
	}
	
	@Test
	public void testDenseSparseCpmmMR() 
	{
		runDistributedMatrixMatrixMultiplicationTest(false, true, MMultMethod.CPMM, ExecType.MR);
	}
	
	@Test
	public void testSparseDenseCpmmMR() 
	{
		runDistributedMatrixMatrixMultiplicationTest(true, false, MMultMethod.CPMM, ExecType.MR);
	}
	
	@Test
	public void testSparseSparseCpmmMR() 
	{
		runDistributedMatrixMatrixMultiplicationTest(true, true, MMultMethod.CPMM, ExecType.MR);
	}
	
	@Test
	public void testDenseDenseRmmMR() 
	{
		runDistributedMatrixMatrixMultiplicationTest(false, false, MMultMethod.RMM, ExecType.MR);
	}
	
	@Test
	public void testDenseSparseRmmMR() 
	{
		runDistributedMatrixMatrixMultiplicationTest(false, true, MMultMethod.RMM, ExecType.MR);
	}
	
	@Test
	public void testSparseDenseRmmMR() 
	{
		runDistributedMatrixMatrixMultiplicationTest(true, false, MMultMethod.RMM, ExecType.MR);
	}
	
	@Test
	public void testSparseSparseRmmMR() 
	{
		runDistributedMatrixMatrixMultiplicationTest(true, true, MMultMethod.RMM, ExecType.MR);
	}

	@Test
	public void testDenseDenseMapmmSpark() 
	{
		runDistributedMatrixMatrixMultiplicationTest(false, false, MMultMethod.MAPMM_R, ExecType.SPARK);
	}
	
	@Test
	public void testDenseSparseMapmmSpark() 
	{
		runDistributedMatrixMatrixMultiplicationTest(false, true, MMultMethod.MAPMM_R, ExecType.SPARK);
	}
	
	@Test
	public void testSparseDenseMapmmSpark() 
	{
		runDistributedMatrixMatrixMultiplicationTest(true, false, MMultMethod.MAPMM_R, ExecType.SPARK);
	}
	
	@Test
	public void testSparseSparseMapmmSpark() 
	{
		runDistributedMatrixMatrixMultiplicationTest(true, true, MMultMethod.MAPMM_R, ExecType.SPARK);
	}
	
	@Test
	public void testDenseDenseCpmmSpark() 
	{
		runDistributedMatrixMatrixMultiplicationTest(false, false, MMultMethod.CPMM, ExecType.SPARK);
	}
	
	@Test
	public void testDenseSparseCpmmSpark() 
	{
		runDistributedMatrixMatrixMultiplicationTest(false, true, MMultMethod.CPMM, ExecType.SPARK);
	}
	
	@Test
	public void testSparseDenseCpmmSpark() 
	{
		runDistributedMatrixMatrixMultiplicationTest(true, false, MMultMethod.CPMM, ExecType.SPARK);
	}
	
	@Test
	public void testSparseSparseCpmmSpark() 
	{
		runDistributedMatrixMatrixMultiplicationTest(true, true, MMultMethod.CPMM, ExecType.SPARK);
	}
	
	@Test
	public void testDenseDenseRmmSpark() 
	{
		runDistributedMatrixMatrixMultiplicationTest(false, false, MMultMethod.RMM, ExecType.SPARK);
	}
	
	@Test
	public void testDenseSparseRmmSpark() 
	{
		runDistributedMatrixMatrixMultiplicationTest(false, true, MMultMethod.RMM, ExecType.SPARK);
	}
	
	@Test
	public void testSparseDenseRmmSpark() 
	{
		runDistributedMatrixMatrixMultiplicationTest(true, false, MMultMethod.RMM, ExecType.SPARK);
	}
	
	@Test
	public void testSparseSparseRmmSpark() 
	{
		runDistributedMatrixMatrixMultiplicationTest(true, true, MMultMethod.RMM, ExecType.SPARK);
	}
	

	/**
	 * 
	 * @param sparseM1
	 * @param sparseM2
	 * @param instType
	 */
	private void runDistributedMatrixMatrixMultiplicationTest( boolean sparseM1, boolean sparseM2, MMultMethod method, ExecType instType)
	{
		//rtplatform for MR
		RUNTIME_PLATFORM platformOld = rtplatform;
		switch( instType ){
			case MR: rtplatform = RUNTIME_PLATFORM.HADOOP; break;
			case SPARK: rtplatform = RUNTIME_PLATFORM.SPARK; break;
			default: rtplatform = RUNTIME_PLATFORM.HYBRID; break;
		}
	
		boolean sparkConfigOld = DMLScript.USE_LOCAL_SPARK_CONFIG;
		if( rtplatform == RUNTIME_PLATFORM.SPARK )
			DMLScript.USE_LOCAL_SPARK_CONFIG = true;

		MMultMethod methodOld = AggBinaryOp.FORCED_MMULT_METHOD;
		AggBinaryOp.FORCED_MMULT_METHOD = method;
		
		try
		{
			TestConfiguration config = getTestConfiguration(TEST_NAME);

			double sparsityA = sparseM1?sparsity2:sparsity1;
			double sparsityB = sparseM2?sparsity2:sparsity1;
			
			String TEST_CACHE_DIR = "";
			if (TEST_CACHE_ENABLED) {
				TEST_CACHE_DIR = String.valueOf(sparsityA) + "_" + String.valueOf(sparsityB) + "/";
			}
			
			loadTestConfiguration(config, TEST_CACHE_DIR);
			
			/* This is for running the junit test the new way, i.e., construct the arguments directly */
			String HOME = SCRIPT_DIR + TEST_DIR;
			fullDMLScriptName = HOME + TEST_NAME + ".dml";
			programArgs = new String[]{"-args", input("A"), input("B"), output("C") };
			
			fullRScriptName = HOME + TEST_NAME + ".R";
			rCmd = "Rscript" + " " + fullRScriptName + " " + inputDir() + " " + expectedDir();
	
			//generate actual dataset
			double[][] A = getRandomMatrix(rowsA, colsA, 0, 1, sparsityA, 12357); 
			writeInputMatrixWithMTD("A", A, true);
			double[][] B = getRandomMatrix(rowsB, colsB, 0, 1, sparsityB, 9873); 
			writeInputMatrixWithMTD("B", B, true);
	
			runTest(true, false, null, -1); 
			runRScript(true); 
			
			//compare matrices 
			HashMap<CellIndex, Double> dmlfile = readDMLMatrixFromHDFS("C");
			HashMap<CellIndex, Double> rfile  = readRMatrixFromFS("C");
			TestUtils.compareMatrices(dmlfile, rfile, eps, "Stat-DML", "Stat-R");
		}
		finally
		{
			rtplatform = platformOld;
			DMLScript.USE_LOCAL_SPARK_CONFIG = sparkConfigOld;
			AggBinaryOp.FORCED_MMULT_METHOD = methodOld;
		}
	}
}