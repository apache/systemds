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
import com.ibm.bi.dml.lops.MMTSJ.MMTSJType;
import com.ibm.bi.dml.runtime.matrix.data.MatrixValue.CellIndex;
import com.ibm.bi.dml.test.integration.AutomatedTestBase;
import com.ibm.bi.dml.test.integration.TestConfiguration;
import com.ibm.bi.dml.test.utils.TestUtils;

public class FullMatrixMultiplicationTransposeSelfTest extends AutomatedTestBase 
{

	private final static String TEST_NAME1 = "TransposeSelfMatrixMultiplication1";
	private final static String TEST_NAME2 = "TransposeSelfMatrixMultiplication2";
	private final static String TEST_DIR = "functions/binary/matrix_full_other/";
	private final static String TEST_CLASS_DIR = TEST_DIR + FullMatrixMultiplicationTransposeSelfTest.class.getSimpleName() + "/";
	private final static double eps = 1e-10;
	
	//for CP
	private final static int rows1 = 3500;
	private final static int cols1 = 1500;
	//for MR
	private final static int rows2 = 7000;//7000;  
	private final static int cols2 = 750;//750; 
	
	private final static double sparsity1 = 0.7;
	private final static double sparsity2 = 0.1;
	
	
	@Override
	public void setUp() 
	{
		addTestConfiguration(
				TEST_NAME1, 
				new TestConfiguration(TEST_CLASS_DIR, TEST_NAME1, 
				new String[] { "B" })   ); 
		addTestConfiguration(
				TEST_NAME2, 
				new TestConfiguration(TEST_CLASS_DIR, TEST_NAME2, 
				new String[] { "B" })   ); 
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
	public void testMMLeftDenseCP() 
	{
		runTransposeSelfMatrixMultiplicationTest(MMTSJType.LEFT, ExecType.CP, false);
	}
	
	@Test
	public void testMMRightDenseCP() 
	{
		runTransposeSelfMatrixMultiplicationTest(MMTSJType.RIGHT, ExecType.CP, false);
	}

	@Test
	public void testMMLeftSparseCP() 
	{
		runTransposeSelfMatrixMultiplicationTest(MMTSJType.LEFT, ExecType.CP, true);
	}
	
	@Test
	public void testMMRightSparseCP() 
	{
		runTransposeSelfMatrixMultiplicationTest(MMTSJType.RIGHT, ExecType.CP, true);
	}
	
	@Test
	public void testMMLeftDenseMR() 
	{
		runTransposeSelfMatrixMultiplicationTest(MMTSJType.LEFT, ExecType.MR, false);
	}
	
	@Test
	public void testMMRightDenseMR() 
	{
		runTransposeSelfMatrixMultiplicationTest(MMTSJType.RIGHT, ExecType.MR, false);
	}

	@Test
	public void testMMLeftSparseMR() 
	{
		runTransposeSelfMatrixMultiplicationTest(MMTSJType.LEFT, ExecType.MR, true);
	}
	
	@Test
	public void testMMRightSparseMR() 
	{
		runTransposeSelfMatrixMultiplicationTest(MMTSJType.RIGHT, ExecType.MR, true);
	}	
	
	@Test
	public void testVVLeftDenseCP() 
	{
		runTransposeSelfVectorMultiplicationTest(MMTSJType.LEFT, ExecType.CP, false);
	}
	
	@Test
	public void testVVRightDenseCP() 
	{
		runTransposeSelfVectorMultiplicationTest(MMTSJType.RIGHT, ExecType.CP, false);
	}

	@Test
	public void testVVLeftSparseCP() 
	{
		runTransposeSelfVectorMultiplicationTest(MMTSJType.LEFT, ExecType.CP, true);
	}
	
	@Test
	public void testVVRightSparseCP() 
	{
		runTransposeSelfVectorMultiplicationTest(MMTSJType.RIGHT, ExecType.CP, true);
	}
	
	@Test
	public void testVVLeftDenseMR() 
	{
		runTransposeSelfVectorMultiplicationTest(MMTSJType.LEFT, ExecType.MR, false);
	}
	
	@Test
	public void testVVRightDenseMR() 
	{
		runTransposeSelfVectorMultiplicationTest(MMTSJType.RIGHT, ExecType.MR, false);
	}

	@Test
	public void testVVLeftSparseMR() 
	{
		runTransposeSelfVectorMultiplicationTest(MMTSJType.LEFT, ExecType.MR, true);
	}
	
	@Test
	public void testVVRightSparseMR() 
	{
		runTransposeSelfVectorMultiplicationTest(MMTSJType.RIGHT, ExecType.MR, true);
	}

	/**
	 * 
	 * @param type
	 * @param instType
	 * @param sparse
	 */
	private void runTransposeSelfMatrixMultiplicationTest( MMTSJType type, ExecType instType, boolean sparse )
	{
		//setup exec type, rows, cols
		int rows = -1, cols = -1;
		String TEST_NAME = null;
		if( type == MMTSJType.LEFT ) {
			if( instType == ExecType.CP ) {
				rows = rows1;
				cols = cols1;
			}
			else { //if type MR
				rows = rows2;
				cols = cols2;
			}
			TEST_NAME = TEST_NAME1;
		}
		else {
			if( instType == ExecType.CP ) {
				rows = cols1;
				cols = rows1;
			}
			else { //if type MR
				rows = cols2;
				cols = rows2;
			}
			TEST_NAME = TEST_NAME2;
		}

		double sparsity = sparse ? sparsity2 : sparsity1;

		String TEST_CACHE_DIR = "";
		if (TEST_CACHE_ENABLED) {
			TEST_CACHE_DIR = rows + "_" + cols + "_" + sparsity + "/";
		}

		//rtplatform for MR
		RUNTIME_PLATFORM platformOld = rtplatform;
		rtplatform = (instType==ExecType.MR) ? RUNTIME_PLATFORM.HADOOP : RUNTIME_PLATFORM.HYBRID;
	
		try
		{
			TestConfiguration config = getTestConfiguration(TEST_NAME);
			loadTestConfiguration(config, TEST_CACHE_DIR);
			
			/* This is for running the junit test the new way, i.e., construct the arguments directly */
			String HOME = SCRIPT_DIR + TEST_DIR;
			fullDMLScriptName = HOME + TEST_NAME + ".dml";
			programArgs = new String[]{"-args", input("A"),
				Integer.toString(rows), Integer.toString(cols), output("B") };
			
			fullRScriptName = HOME + TEST_NAME + ".R";
			rCmd = "Rscript" + " " + fullRScriptName + " " + inputDir() + " " + expectedDir();
	
			//generate actual dataset
			double[][] A = getRandomMatrix(rows, cols, 0, 1, sparsity, 7); 
			writeInputMatrix("A", A, true);
	
			boolean exceptionExpected = false;
			runTest(true, exceptionExpected, null, -1); 
			runRScript(true); 
			
			//compare matrices 
			HashMap<CellIndex, Double> dmlfile = readDMLMatrixFromHDFS("B");
			HashMap<CellIndex, Double> rfile  = readRMatrixFromFS("B");
			TestUtils.compareMatrices(dmlfile, rfile, eps, "Stat-DML", "Stat-R");
		}
		finally
		{
			rtplatform = platformOld;
		}
	}
	
	/**
	 * 
	 * @param type
	 * @param instType
	 * @param sparse
	 */
	private void runTransposeSelfVectorMultiplicationTest( MMTSJType type, ExecType instType, boolean sparse )
	{
		//setup exec type, rows, cols
		int rows = -1, cols = -1;
		String TEST_NAME = null;
		if( type == MMTSJType.LEFT ) {
			if( instType == ExecType.CP ) {
				rows = rows1;
				cols = 1;
			}
			else { //if type MR
				rows = rows2;
				cols = 1;
			}
			TEST_NAME = TEST_NAME1;
		}
		else {
			if( instType == ExecType.CP ) {
				rows = 1;
				cols = rows1;
			}
			else { //if type MR
				rows = 1;
				cols = rows2;
			}
			TEST_NAME = TEST_NAME2;
		}

		double sparsity = sparse ? sparsity2 : sparsity1;

		String TEST_CACHE_DIR = "";
		if (TEST_CACHE_ENABLED) {
			TEST_CACHE_DIR = rows + "_" + cols + "_" + sparsity + "/";
		}

		//rtplatform for MR
		RUNTIME_PLATFORM platformOld = rtplatform;
		rtplatform = (instType==ExecType.MR) ? RUNTIME_PLATFORM.HADOOP : RUNTIME_PLATFORM.HYBRID;
	
		try
		{
			TestConfiguration config = getTestConfiguration(TEST_NAME);
			loadTestConfiguration(config, TEST_CACHE_DIR);
			
			/* This is for running the junit test the new way, i.e., construct the arguments directly */
			String HOME = SCRIPT_DIR + TEST_DIR;
			fullDMLScriptName = HOME + TEST_NAME + ".dml";
			programArgs = new String[]{"-args", input("A"),
				Integer.toString(rows), Integer.toString(cols), output("B") };
			
			fullRScriptName = HOME + TEST_NAME + ".R";
			rCmd = "Rscript" + " " + fullRScriptName + " " + inputDir() + " " + expectedDir();
	
			//generate actual dataset
			double[][] A = getRandomMatrix(rows, cols, 0, 1, sparsity, 7); 
			writeInputMatrix("A", A, true);
	
			boolean exceptionExpected = false;
			runTest(true, exceptionExpected, null, -1); 
			runRScript(true); 
			
			//compare matrices 
			HashMap<CellIndex, Double> dmlfile = readDMLMatrixFromHDFS("B");
			HashMap<CellIndex, Double> rfile  = readRMatrixFromFS("B");
			TestUtils.compareMatrices(dmlfile, rfile, eps, "Stat-DML", "Stat-R");
		}
		finally
		{
			rtplatform = platformOld;
		}
	}	
	
}