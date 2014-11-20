/**
 * IBM Confidential
 * OCO Source Materials
 * (C) Copyright IBM Corp. 2010, 2014
 * The source code for this program is not published or otherwise divested of its trade secrets, irrespective of what has been deposited with the U.S. Copyright Office.
 */

package com.ibm.bi.dml.test.integration.functions.binary.matrix;

import java.util.HashMap;

import org.junit.Test;

import com.ibm.bi.dml.api.DMLScript.RUNTIME_PLATFORM;
import com.ibm.bi.dml.lops.LopProperties.ExecType;
import com.ibm.bi.dml.runtime.matrix.data.MatrixValue.CellIndex;
import com.ibm.bi.dml.test.integration.AutomatedTestBase;
import com.ibm.bi.dml.test.integration.TestConfiguration;
import com.ibm.bi.dml.test.utils.TestUtils;

public class FullMatrixMultiplicationTest extends AutomatedTestBase 
{
	@SuppressWarnings("unused")
	private static final String _COPYRIGHT = "Licensed Materials - Property of IBM\n(C) Copyright IBM Corp. 2010, 2014\n" +
                                             "US Government Users Restricted Rights - Use, duplication  disclosure restricted by GSA ADP Schedule Contract with IBM Corp.";
	
	private final static String TEST_NAME = "FullMatrixMultiplication";
	private final static String TEST_DIR = "functions/binary/matrix/";
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
		addTestConfiguration(
				TEST_NAME, 
				new TestConfiguration(TEST_DIR, TEST_NAME, 
				new String[] { "C" })   ); 
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
	public void testMMDenseDenseMR() 
	{
		runMatrixMatrixMultiplicationTest(false, false, ExecType.MR);
	}
	
	@Test
	public void testMMDenseSparseMR() 
	{
		runMatrixMatrixMultiplicationTest(false, true, ExecType.MR);
	}
	
	@Test
	public void testMMSparseDenseMR() 
	{
		runMatrixMatrixMultiplicationTest(true, false, ExecType.MR);
	}
	
	@Test
	public void testMMSparseSparseMR() 
	{
		runMatrixMatrixMultiplicationTest(true, true, ExecType.MR);
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
	public void testMVDenseDenseMR() 
	{
		runMatrixVectorMultiplicationTest(false, ExecType.MR);
	}
	
	@Test
	public void testMVSparseDenseMR() 
	{
		runMatrixVectorMultiplicationTest(true, ExecType.MR);
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
	public void testVVDenseDenseMR() 
	{
		runVectorVectorMultiplicationTest(false, false, ExecType.MR);
	}
	
	@Test
	public void testVVSparseDenseMR() 
	{
		runVectorVectorMultiplicationTest(true, false, ExecType.MR);
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
	
	@Test
	public void testVtVtDenseDenseMR() 
	{
		runVectorVectorMultiplicationTest(false, true, ExecType.MR);
	}
	
	@Test
	public void testVtVtSparseDenseMR() 
	{
		runVectorVectorMultiplicationTest(true, true, ExecType.MR);
	}

	/**
	 * 
	 * @param sparseM1
	 * @param sparseM2
	 * @param instType
	 */
	private void runMatrixMatrixMultiplicationTest( boolean sparseM1, boolean sparseM2, ExecType instType)
	{
		//setup exec type, rows, cols

		//rtplatform for MR
		RUNTIME_PLATFORM platformOld = rtplatform;
		rtplatform = (instType==ExecType.MR) ? RUNTIME_PLATFORM.HADOOP : RUNTIME_PLATFORM.HYBRID;
	
		try
		{
			TestConfiguration config = getTestConfiguration(TEST_NAME);
			
			/* This is for running the junit test the new way, i.e., construct the arguments directly */
			String HOME = SCRIPT_DIR + TEST_DIR;
			fullDMLScriptName = HOME + TEST_NAME + ".dml";
			programArgs = new String[]{"-args", HOME + INPUT_DIR + "A",
					                        Integer.toString(rowsA),
					                        Integer.toString(colsA),
					                        HOME + INPUT_DIR + "B",
					                        Integer.toString(rowsB),
					                        Integer.toString(colsB),
					                        HOME + OUTPUT_DIR + "C"    };
			fullRScriptName = HOME + TEST_NAME + ".R";
			rCmd = "Rscript" + " " + fullRScriptName + " " + 
			       HOME + INPUT_DIR + " " + HOME + EXPECTED_DIR;
			
			loadTestConfiguration(config);
	
			//generate actual dataset
			double[][] A = getRandomMatrix(rowsA, colsA, 0, 1, sparseM1?sparsity2:sparsity1, 7); 
			writeInputMatrix("A", A, true);
			double[][] B = getRandomMatrix(rowsB, colsB, 0, 1, sparseM2?sparsity2:sparsity1, 3); 
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
		RUNTIME_PLATFORM platformOld = rtplatform;
		rtplatform = (instType==ExecType.MR) ? RUNTIME_PLATFORM.HADOOP : RUNTIME_PLATFORM.HYBRID;
	
		try
		{
			TestConfiguration config = getTestConfiguration(TEST_NAME);
			
			/* This is for running the junit test the new way, i.e., construct the arguments directly */
			String HOME = SCRIPT_DIR + TEST_DIR;
			fullDMLScriptName = HOME + TEST_NAME + ".dml";
			programArgs = new String[]{"-args", HOME + INPUT_DIR + "A",
					                        Integer.toString(rowsA),
					                        Integer.toString(colsA),
					                        HOME + INPUT_DIR + "B",
					                        Integer.toString(rowsB),
					                        Integer.toString(1),
					                        HOME + OUTPUT_DIR + "C"    };
			fullRScriptName = HOME + TEST_NAME + ".R";
			rCmd = "Rscript" + " " + fullRScriptName + " " + 
			       HOME + INPUT_DIR + " " + HOME + EXPECTED_DIR;
			
			loadTestConfiguration(config);
	
			//generate actual dataset
			double[][] A = getRandomMatrix(rowsA, colsA, 0, 1, sparseM1?sparsity2:sparsity1, 7); 
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
		RUNTIME_PLATFORM platformOld = rtplatform;
		rtplatform = (instType==ExecType.MR) ? RUNTIME_PLATFORM.HADOOP : RUNTIME_PLATFORM.HYBRID;
	
		try
		{
			TestConfiguration config = getTestConfiguration(TEST_NAME);
			
			int rows1 = outer?colsA:1;
			int rows2 = outer?1:rowsB;
			int cols1 = outer?1:colsA;
			int cols2 = outer?rowsB:1;
			
			/* This is for running the junit test the new way, i.e., construct the arguments directly */
			String HOME = SCRIPT_DIR + TEST_DIR;
			fullDMLScriptName = HOME + TEST_NAME + ".dml";
			programArgs = new String[]{"-args", HOME + INPUT_DIR + "A",
					                        Integer.toString(rows1),
					                        Integer.toString(cols1),
					                        HOME + INPUT_DIR + "B",
					                        Integer.toString(rows2),
					                        Integer.toString(cols2),
					                        HOME + OUTPUT_DIR + "C"    };
			fullRScriptName = HOME + TEST_NAME + ".R";
			rCmd = "Rscript" + " " + fullRScriptName + " " + 
			       HOME + INPUT_DIR + " " + HOME + EXPECTED_DIR;
			
			loadTestConfiguration(config);
	
			//generate actual dataset
			double[][] A = getRandomMatrix(rows1, cols1, 0, 1, sparseM1?sparsity2:sparsity1, 7); 
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