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

import java.util.HashMap;

import org.junit.AfterClass;
import org.junit.BeforeClass;
import org.junit.Test;
import org.apache.sysds.api.DMLScript;
import org.apache.sysds.common.Types.ExecMode;
import org.apache.sysds.hops.OptimizerUtils;
import org.apache.sysds.lops.LopProperties.ExecType;
import org.apache.sysds.runtime.matrix.data.MatrixValue.CellIndex;
import org.apache.sysds.test.AutomatedTestBase;
import org.apache.sysds.test.TestConfiguration;
import org.apache.sysds.test.TestUtils;

public class DiagMatrixMultiplicationTest extends AutomatedTestBase 
{
	
	private final static String TEST_NAME1 = "DiagMatrixMultiplication";
	private final static String TEST_NAME2 = "DiagMatrixMultiplicationTranspose";
	private final static String TEST_DIR = "functions/binary/matrix/";
	private final static String TEST_CLASS_DIR = TEST_DIR + DiagMatrixMultiplicationTest.class.getSimpleName() + "/";
	private final static double eps = 1e-10;
	
	private final static int rowsA = 1107;
	private final static int colsA = 1107;
	private final static int rowsB = 1107;
	private final static int colsB = 1107;
	
	private final static double sparsity1 = 0.7;
	private final static double sparsity2 = 0.1;
	
	
	@Override
	public void setUp() 
	{
		addTestConfiguration(TEST_NAME1, 
			new TestConfiguration(TEST_CLASS_DIR, TEST_NAME1, new String[] { "C" }) ); 
		addTestConfiguration(TEST_NAME2, 
			new TestConfiguration(TEST_CLASS_DIR, TEST_NAME2, new String[] { "C" }) ); 

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
	public void testDiagMMDenseDenseCP() 
	{
		//should apply diag_mm rewrite
		runDiagMatrixMultiplicationTest(false, false, false, false, ExecType.CP, true);
	}
	
	@Test
	public void testDiagMMDenseDenseTransposeCP() 
	{
		//should apply diag_mm / t_t rewrite
		runDiagMatrixMultiplicationTest(false, false, true, false, ExecType.CP, true);
	}
	
	@Test
	public void testDiagMVDenseDenseCP() 
	{
		//should not apply diag_mm rewrite
		runDiagMatrixMultiplicationTest(false, false, false, true, ExecType.CP, true);
	}
	
	@Test
	public void testDiagMMSparseSparseCP() 
	{
		//should apply diag_mm rewrite
		runDiagMatrixMultiplicationTest(true, true, false, false, ExecType.CP, true);
	}
	
	@Test
	public void testDiagMMSparseSparseTransposeCP() 
	{
		//should apply diag_mm / t_t rewrite
		runDiagMatrixMultiplicationTest(true, true, true, false, ExecType.CP, true);
	}
	
	@Test
	public void testDiagMVSparseSparseCP() 
	{
		//should not apply diag_mm rewrite
		runDiagMatrixMultiplicationTest(true, true, false, true, ExecType.CP, true);
	}
	
	// --------------------------------------------------------

	@Test
	public void testDiagMMDenseDenseSP() 
	{
		//should apply diag_mm rewrite
		runDiagMatrixMultiplicationTest(false, false, false, false, ExecType.SPARK, true);
	}
	
	@Test
	public void testDiagMMDenseDenseTransposeSP() 
	{
		//should apply diag_mm / t_t rewrite
		runDiagMatrixMultiplicationTest(false, false, true, false, ExecType.SPARK, true);
	}
	
	@Test
	public void testDiagMVDenseDenseSP() 
	{
		//should not apply diag_mm rewrite
		runDiagMatrixMultiplicationTest(false, false, false, true, ExecType.SPARK, true);
	}
	
	@Test
	public void testDiagMMSparseSparseSP() 
	{
		//should apply diag_mm rewrite
		runDiagMatrixMultiplicationTest(true, true, false, false, ExecType.SPARK, true);
	}
	
	@Test
	public void testDiagMMSparseSparseTransposeSP() 
	{
		//should apply diag_mm / t_t rewrite
		runDiagMatrixMultiplicationTest(true, true, true, false, ExecType.SPARK, true);
	}
	
	@Test
	public void testDiagMVSparseSparseSP() 
	{
		//should not apply diag_mm rewrite
		runDiagMatrixMultiplicationTest(true, true, false, true, ExecType.SPARK, true);
	}
	
	// --------------------------------------------------------

	@Test
	public void testDiagMMDenseDenseNoSimplifyCP() 
	{
		runDiagMatrixMultiplicationTest(false, false, false, false, ExecType.CP, false);
	}
	
	@Test
	public void testDiagMMDenseDenseTransposeNoSimplifyCP() 
	{
		runDiagMatrixMultiplicationTest(false, false, true, false, ExecType.CP, false);
	}
	
	@Test
	public void testDiagMVDenseDenseNoSimplifyCP() 
	{
		runDiagMatrixMultiplicationTest(false, false, false, true, ExecType.CP, false);
	}
	
	@Test
	public void testDiagMMSparseSparseNoSimplifyCP() 
	{
		runDiagMatrixMultiplicationTest(true, true, false, false, ExecType.CP, false);
	}
	
	@Test
	public void testDiagMMSparseSparseTransposeNoSimplifyCP() 
	{
		runDiagMatrixMultiplicationTest(true, true, true, false, ExecType.CP, false);
	}
	
	@Test
	public void testDiagMVSparseSparseNoSimplifyCP() 
	{
		runDiagMatrixMultiplicationTest(true, true, false, true, ExecType.CP, false);
	}
	
	/**
	 * 
	 * @param sparseM1
	 * @param sparseM2
	 * @param instType
	 */
	private void runDiagMatrixMultiplicationTest( boolean sparseM1, boolean sparseM2, boolean rightTranspose, boolean rightVector, ExecType instType, boolean simplify)
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
		
		String TEST_NAME = rightTranspose ? TEST_NAME2 : TEST_NAME1;
		boolean oldFlagSimplify = OptimizerUtils.ALLOW_ALGEBRAIC_SIMPLIFICATION;
		
		try
		{
			TestConfiguration config = getTestConfiguration(TEST_NAME);
			
			OptimizerUtils.ALLOW_ALGEBRAIC_SIMPLIFICATION = simplify;
			
			double sparsityM1 = sparseM1?sparsity2:sparsity1;
			double sparsityM2 = sparseM2?sparsity2:sparsity1;
			
			String TEST_CACHE_DIR = "";
			if (TEST_CACHE_ENABLED)
			{
				TEST_CACHE_DIR = sparsityM1 + "_" + sparsityM2 + "_" + rightTranspose + "_" + rightVector + "/";
			}
			
			loadTestConfiguration(config, TEST_CACHE_DIR);
			
			/* This is for running the junit test the new way, i.e., construct the arguments directly */
			String HOME = SCRIPT_DIR + TEST_DIR;
			fullDMLScriptName = HOME + TEST_NAME + ".dml";
			programArgs = new String[]{"-explain","-args", 
				input("A"), Integer.toString(rowsA), Integer.toString(colsA),
				input("B"), Integer.toString(rowsB), Integer.toString(rightVector?1:colsB), output("C")};
			
			fullRScriptName = HOME + TEST_NAME + ".R";
			rCmd = "Rscript" + " " + fullRScriptName + " " + inputDir() + " " + expectedDir();
			
			//generate actual dataset
			double[][] A = getRandomMatrix(rowsA, colsA, 0, 1, sparsityM1, 7); 
			writeInputMatrix("A", A, true);
			double[][] B = getRandomMatrix(rowsB, rightVector?1:colsB, 0, 1, sparsityM2, 3); 
			writeInputMatrix("B", B, true);
	
			boolean exceptionExpected = false;
			runTest(true, exceptionExpected, null, -1); 
			runRScript(true); 
			
			//compare matrices 
			HashMap<CellIndex, Double> dmlfile = readDMLMatrixFromHDFS("C");
			HashMap<CellIndex, Double> rfile  = readRMatrixFromFS("C");
			TestUtils.compareMatrices(dmlfile, rfile, eps, "Stat-DML", "Stat-R");
		}
		finally
		{
			rtplatform = platformOld;
			OptimizerUtils.ALLOW_ALGEBRAIC_SIMPLIFICATION = oldFlagSimplify;
			DMLScript.USE_LOCAL_SPARK_CONFIG = sparkConfigOld;
		}
	}

}