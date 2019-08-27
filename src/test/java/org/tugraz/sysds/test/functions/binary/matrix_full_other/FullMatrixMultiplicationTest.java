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

package org.tugraz.sysds.test.functions.binary.matrix_full_other;

import java.util.HashMap;

import org.junit.AfterClass;
import org.junit.BeforeClass;
import org.junit.Test;
import org.tugraz.sysds.common.Types.ExecMode;
import org.tugraz.sysds.lops.LopProperties.ExecType;
import org.tugraz.sysds.runtime.matrix.data.MatrixValue.CellIndex;
import org.tugraz.sysds.test.AutomatedTestBase;
import org.tugraz.sysds.test.TestConfiguration;
import org.tugraz.sysds.test.TestUtils;

public class FullMatrixMultiplicationTest extends AutomatedTestBase 
{	
	private final static String TEST_NAME = "FullMatrixMultiplication";
	private final static String TEST_DIR = "functions/binary/matrix_full_other/";
	private final static String TEST_CLASS_DIR = TEST_DIR + FullMatrixMultiplicationTest.class.getSimpleName() + "/";
	private final static double eps = 1e-10;
	
	private final static int rowsA = 1501;
	private final static int colsA = 1703;
	private final static int rowsB = 1703;
	private final static int colsB = 1107;
	
	private final static double sparsity1 = 0.7;
	private final static double sparsity2 = 0.1;
	
	
	@Override
	public void setUp() 
	{
		TestUtils.clearAssertionInformation();
		addTestConfiguration(
				TEST_NAME, 
				new TestConfiguration(TEST_CLASS_DIR, TEST_NAME, 
				new String[] { "C" })   ); 
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
	public void testMMDenseDenseCP() 
	{
		runMatrixMatrixMultiplicationTest(false, false, ExecType.CP);
	}
	
	@Test
	public void testMMDenseSparseCP() 
	{
		runMatrixMatrixMultiplicationTest(false, true, ExecType.CP);
	}
	
	@Test
	public void testMMSparseDenseCP() 
	{
		runMatrixMatrixMultiplicationTest(true, false, ExecType.CP);
	}
	
	@Test
	public void testMMSparseSparseCP() 
	{
		runMatrixMatrixMultiplicationTest(true, true, ExecType.CP);
	}
	
	@Test
	public void testMVDenseDenseCP() 
	{
		runMatrixVectorMultiplicationTest(false, ExecType.CP);
	}
	
	@Test
	public void testMVSparseDenseCP() 
	{
		runMatrixVectorMultiplicationTest(true, ExecType.CP);
	}
	
	@Test
	public void testVMDenseDenseCP() 
	{
		runVectorMatrixMultiplicationTest(false, false, ExecType.CP);
	}
	
	@Test
	public void testVMDenseSparseCP() 
	{
		runVectorMatrixMultiplicationTest(false, true, ExecType.CP);
	}
	
	@Test
	public void testVMSparseDenseCP() 
	{
		runVectorMatrixMultiplicationTest(true, false, ExecType.CP);
	}
	
	@Test
	public void testVMSparseSparseCP() 
	{
		runVectorMatrixMultiplicationTest(true, true, ExecType.CP);
	}
	
	@Test
	public void testVVDenseDenseCP() 
	{
		runVectorVectorMultiplicationTest(false, false, ExecType.CP);
	}
	
	@Test
	public void testVVSparseDenseCP() 
	{
		runVectorVectorMultiplicationTest(true, false, ExecType.CP);
	}
	
	@Test
	public void testVtVtDenseDenseCP() 
	{
		runVectorVectorMultiplicationTest(false, true, ExecType.CP);
	}
	
	@Test
	public void testVtVtSparseDenseCP() 
	{
		runVectorVectorMultiplicationTest(true, true, ExecType.CP);
	}
	
	
	private void runMatrixMatrixMultiplicationTest( boolean sparseM1, boolean sparseM2, ExecType instType)
	{
		//setup exec type, rows, cols

		//rtplatform for MR
		ExecMode platformOld = rtplatform;
		rtplatform = ExecMode.HYBRID;
	
		try
		{
			TestConfiguration config = getTestConfiguration(TEST_NAME);

			double sparsityA = sparseM1?sparsity2:sparsity1; 
			double sparsityB = sparseM2?sparsity2:sparsity1; 

			String TEST_CACHE_DIR = "";
			if (TEST_CACHE_ENABLED) {
				TEST_CACHE_DIR = "mm" + String.valueOf(sparsityA) + "_" + String.valueOf(sparsityB) + "/";
			}
			
			loadTestConfiguration(config, TEST_CACHE_DIR);
			
			/* This is for running the junit test the new way, i.e., construct the arguments directly */
			String HOME = SCRIPT_DIR + TEST_DIR;
			fullDMLScriptName = HOME + TEST_NAME + ".dml";
			programArgs = new String[]{"-stats","-args",
				input("A"), Integer.toString(rowsA), Integer.toString(colsA),
				input("B"), Integer.toString(rowsB), Integer.toString(colsB), output("C") };
			
			fullRScriptName = HOME + TEST_NAME + ".R";
			rCmd = "Rscript" + " " + fullRScriptName + " " + inputDir() + " " + expectedDir();
	
			//generate actual dataset
			double[][] A = getRandomMatrix(rowsA, colsA, 0, 1, sparsityA, 7); 
			writeInputMatrix("A", A, true);
			double[][] B = getRandomMatrix(rowsB, colsB, 0, 1, sparsityB, 3); 
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
		}
	}
	
	/**
	 * Note: second matrix is always dense if vector.
	 * 
	 * @param sparseM1
	 * @param instType
	 */
	private void runMatrixVectorMultiplicationTest( boolean sparseM1, ExecType instType)
	{
		//setup exec type, rows, cols

		//rtplatform for MR
		ExecMode platformOld = rtplatform;
		rtplatform = ExecMode.HYBRID;
	
		try
		{
			TestConfiguration config = getTestConfiguration(TEST_NAME);

			double sparsityA = sparseM1?sparsity2:sparsity1; 

			String TEST_CACHE_DIR = "";
			if (TEST_CACHE_ENABLED) {
				TEST_CACHE_DIR = "mv" + String.valueOf(sparsityA) + "/";
			}
			
			loadTestConfiguration(config, TEST_CACHE_DIR);
			
			/* This is for running the junit test the new way, i.e., construct the arguments directly */
			String HOME = SCRIPT_DIR + TEST_DIR;
			fullDMLScriptName = HOME + TEST_NAME + ".dml";
			programArgs = new String[]{"-args",
				input("A"), Integer.toString(rowsA), Integer.toString(colsA),
				input("B"), Integer.toString(rowsB), Integer.toString(1), output("C") };
			
			fullRScriptName = HOME + TEST_NAME + ".R";
			rCmd = "Rscript" + " " + fullRScriptName + " " + inputDir() + " " + expectedDir();
	
			//generate actual dataset
			double[][] A = getRandomMatrix(rowsA, colsA, 0, 1, sparsityA, 7); 
			writeInputMatrix("A", A, true);
			double[][] B = getRandomMatrix(rowsB, 1, 0, 1, sparsity1, 3); 
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
		}
	}

	private void runVectorMatrixMultiplicationTest( boolean sparseM1, boolean sparseM2, ExecType instType)
	{
		//setup exec type, rows, cols

		//rtplatform for MR
		ExecMode platformOld = rtplatform;
		rtplatform = ExecMode.HYBRID;
	
		try
		{
			TestConfiguration config = getTestConfiguration(TEST_NAME);
			
			double sparsityA = sparseM1?sparsity2:sparsity1; 
			double sparsityB = sparseM2?sparsity2:sparsity1; 

			String TEST_CACHE_DIR = "";
			if (TEST_CACHE_ENABLED) {
				TEST_CACHE_DIR = "vm" + String.valueOf(sparsityA) + "_" + String.valueOf(sparsityB) + "/";
			}
			
			loadTestConfiguration(config, TEST_CACHE_DIR);
			
			/* This is for running the junit test the new way, i.e., construct the arguments directly */
			String HOME = SCRIPT_DIR + TEST_DIR;
			fullDMLScriptName = HOME + TEST_NAME + ".dml";
			programArgs = new String[]{"-args", input("A"), Integer.toString(1), Integer.toString(colsA),
				input("B"), Integer.toString(rowsB), Integer.toString(colsB), output("C") };
			
			fullRScriptName = HOME + TEST_NAME + ".R";
			rCmd = "Rscript" + " " + fullRScriptName + " " + inputDir() + " " + expectedDir();
	
			//generate actual dataset
			double[][] A = getRandomMatrix(1, colsA, 0, 1, sparsityA, 7); 
			writeInputMatrix("A", A, true);
			double[][] B = getRandomMatrix(rowsB, colsB, 0, 1, sparsityB, 3); 
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
		}
	}
	
	/**
	 * Note: second matrix is always dense if vector.
	 * 
	 * @param sparseM1
	 * @param instType
	 */
	private void runVectorVectorMultiplicationTest( boolean sparseM1, boolean outer, ExecType instType)
	{
		//setup exec type, rows, cols

		//rtplatform for MR
		ExecMode platformOld = rtplatform;
		rtplatform = ExecMode.HYBRID;
	
		try
		{
			TestConfiguration config = getTestConfiguration(TEST_NAME);
			
			int rows1 = outer?colsA:1;
			int rows2 = outer?1:rowsB;
			int cols1 = outer?1:colsA;
			int cols2 = outer?rowsB:1;
			
			double sparsityA = sparseM1?sparsity2:sparsity1; 

			String TEST_CACHE_DIR = "";
			if (TEST_CACHE_ENABLED) {
				TEST_CACHE_DIR = "vv" + rows1 + "_" + cols1 + "_" + rows2 + "_" + cols2 + "_" + String.valueOf(sparsityA) + "/";
			}
			
			loadTestConfiguration(config, TEST_CACHE_DIR);

			/* This is for running the junit test the new way, i.e., construct the arguments directly */
			String HOME = SCRIPT_DIR + TEST_DIR;
			fullDMLScriptName = HOME + TEST_NAME + ".dml";
			programArgs = new String[]{"-args", input("A"), Integer.toString(rows1), Integer.toString(cols1),
				input("B"), Integer.toString(rows2), Integer.toString(cols2), output("C") };
			
			fullRScriptName = HOME + TEST_NAME + ".R";
			rCmd = "Rscript" + " " + fullRScriptName + " " + inputDir() + " " + expectedDir();
	
			//generate actual dataset
			double[][] A = getRandomMatrix(rows1, cols1, 0, 1, sparsityA, 7); 
			writeInputMatrix("A", A, true);
			double[][] B = getRandomMatrix(rows2, cols2, 0, 1, sparsity1, 3); 
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
		}
	}

}