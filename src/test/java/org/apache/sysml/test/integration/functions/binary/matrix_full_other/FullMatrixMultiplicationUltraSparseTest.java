/**
 * (C) Copyright IBM Corp. 2010, 2015
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * 
 */

package com.ibm.bi.dml.test.integration.functions.binary.matrix_full_other;

import java.util.HashMap;

import org.junit.AfterClass;
import org.junit.BeforeClass;
import org.junit.Test;

import com.ibm.bi.dml.api.DMLScript.RUNTIME_PLATFORM;
import com.ibm.bi.dml.lops.LopProperties.ExecType;
import com.ibm.bi.dml.runtime.matrix.data.MatrixValue.CellIndex;
import com.ibm.bi.dml.test.integration.AutomatedTestBase;
import com.ibm.bi.dml.test.integration.TestConfiguration;
import com.ibm.bi.dml.test.utils.TestUtils;

public class FullMatrixMultiplicationUltraSparseTest extends AutomatedTestBase 
{
	
	private final static String TEST_NAME = "FullMatrixMultiplication";
	private final static String TEST_DIR = "functions/binary/matrix_full_other/";
	private final static String TEST_CLASS_DIR = TEST_DIR + FullMatrixMultiplicationUltraSparseTest.class.getSimpleName() + "/";
	private final static double eps = 1e-10;
	
	private final static int rowsA = 1501;
	private final static int colsA = 1703;
	private final static int rowsB = 1703;
	private final static int colsB = 1107;
	
	private final static double sparsity1 = 0.7;
	private final static double sparsity2 = 0.1;
	private final static double sparsity3 = 0.000005;
	
	private enum SparsityType{
		DENSE,
		SPARSE,
		ULTRA_SPARSE,
	}
	
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
	public void testMMDenseUltraSparseCP() 
	{
		runMatrixMatrixMultiplicationTest(SparsityType.DENSE, SparsityType.ULTRA_SPARSE, ExecType.CP);
	}
	
	@Test
	public void testMMSparseUltraSparseCP() 
	{
		runMatrixMatrixMultiplicationTest(SparsityType.SPARSE, SparsityType.ULTRA_SPARSE, ExecType.CP);
	}

	@Test
	public void testMMUltraSparseDenseCP() 
	{
		runMatrixMatrixMultiplicationTest(SparsityType.ULTRA_SPARSE, SparsityType.DENSE, ExecType.CP);
	}
	
	@Test
	public void testMMUltraSparseSparseCP() 
	{
		runMatrixMatrixMultiplicationTest(SparsityType.ULTRA_SPARSE, SparsityType.SPARSE, ExecType.CP);
	}
	
	@Test
	public void testMMUltraSparseUltraSparseCP() 
	{
		runMatrixMatrixMultiplicationTest(SparsityType.ULTRA_SPARSE, SparsityType.ULTRA_SPARSE, ExecType.CP);
	}
	
	@Test
	public void testMMDenseUltraSparseMR() 
	{
		runMatrixMatrixMultiplicationTest(SparsityType.DENSE, SparsityType.ULTRA_SPARSE, ExecType.MR);
	}
	
	@Test
	public void testMMSparseUltraSparseMR() 
	{
		runMatrixMatrixMultiplicationTest(SparsityType.SPARSE, SparsityType.ULTRA_SPARSE, ExecType.MR);
	}

	@Test
	public void testMMUltraSparseDenseMR() 
	{
		runMatrixMatrixMultiplicationTest(SparsityType.ULTRA_SPARSE, SparsityType.DENSE, ExecType.MR);
	}
	
	@Test
	public void testMMUltraSparseSparseMR() 
	{
		runMatrixMatrixMultiplicationTest(SparsityType.ULTRA_SPARSE, SparsityType.SPARSE, ExecType.MR);
	}
	
	@Test
	public void testMMUltraSparseUltraSparseMR() 
	{
		runMatrixMatrixMultiplicationTest(SparsityType.ULTRA_SPARSE, SparsityType.ULTRA_SPARSE, ExecType.MR);
	}

	/**
	 * 
	 * @param sparseM1
	 * @param sparseM2
	 * @param instType
	 */
	private void runMatrixMatrixMultiplicationTest( SparsityType sparseM1, SparsityType sparseM2, ExecType instType)
	{
		//setup exec type, rows, cols

		//rtplatform for MR
		RUNTIME_PLATFORM platformOld = rtplatform;
		rtplatform = (instType==ExecType.MR) ? RUNTIME_PLATFORM.HADOOP : RUNTIME_PLATFORM.HYBRID;
	
		try
		{
			TestConfiguration config = getTestConfiguration(TEST_NAME);
			
			double sparsityLeft = (sparseM1==SparsityType.DENSE)? sparsity1 : (sparseM1==SparsityType.SPARSE)? sparsity2 : sparsity3;  
			double sparsityRight = (sparseM2==SparsityType.DENSE)? sparsity1 : (sparseM2==SparsityType.SPARSE)? sparsity2 : sparsity3;  

			String TEST_CACHE_DIR = "";
			if (TEST_CACHE_ENABLED) {
				TEST_CACHE_DIR = String.valueOf(sparsityLeft) + "_" + String.valueOf(sparsityRight) + "/";
			}
			
			loadTestConfiguration(config, TEST_CACHE_DIR);
			
			/* This is for running the junit test the new way, i.e., construct the arguments directly */
			String HOME = SCRIPT_DIR + TEST_DIR;
			fullDMLScriptName = HOME + TEST_NAME + ".dml";
			programArgs = new String[]{"-args", input("A"), Integer.toString(rowsA), Integer.toString(colsA),
				input("B"), Integer.toString(rowsB), Integer.toString(colsB), output("C") };
			
			fullRScriptName = HOME + TEST_NAME + ".R";
			rCmd = "Rscript" + " " + fullRScriptName + " " + inputDir() + " " + expectedDir();
			
			//generate actual dataset
			double[][] A = getRandomMatrix(rowsA, colsA, 0, 1, sparsityLeft, 7); 
			writeInputMatrix("A", A, true);
			double[][] B = getRandomMatrix(rowsB, colsB, 0, 1, sparsityRight, 3); 
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
