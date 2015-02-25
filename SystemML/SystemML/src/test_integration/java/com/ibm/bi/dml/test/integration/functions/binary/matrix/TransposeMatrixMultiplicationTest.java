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

/**
 * This test investigates the specific Hop-Lop rewrite t(X)%*%v -> t(t(v)%*%X).
 * 
 */
public class TransposeMatrixMultiplicationTest extends AutomatedTestBase 
{
	@SuppressWarnings("unused")
	private static final String _COPYRIGHT = "Licensed Materials - Property of IBM\n(C) Copyright IBM Corp. 2010, 2014\n" +
                                             "US Government Users Restricted Rights - Use, duplication  disclosure restricted by GSA ADP Schedule Contract with IBM Corp.";
	
	private final static String TEST_NAME1 = "TransposeMatrixMultiplication";
	private final static String TEST_NAME2 = "TransposeMatrixMultiplicationMinus";
	private final static String TEST_DIR = "functions/binary/matrix/";
	private final static double eps = 1e-10;
	
	//multiblock
	private final static int rowsA1 = 2407;
	private final static int colsA1 = 1199;
	private final static int rowsB1 = 2407;
	private final static int colsB1 = 73;
	
	//singleblock
	private final static int rowsA2 = 2407;
	private final static int colsA2 = 73;
	private final static int rowsB2 = 2407;
	private final static int colsB2 = 1;
	
	
	private final static double sparsity1 = 0.7;
	private final static double sparsity2 = 0.1;
	
	
	@Override
	public void setUp() 
	{
		addTestConfiguration( TEST_NAME1, 
				new TestConfiguration(TEST_DIR, TEST_NAME1, new String[] { "C" })   );
		addTestConfiguration( TEST_NAME2, 
				new TestConfiguration(TEST_DIR, TEST_NAME2, new String[] { "C" })   );
	}

	
	@Test
	public void testTransposeMMDenseDenseCP() 
	{
		runTransposeMatrixMultiplicationTest(false, false, ExecType.CP, false);
	}
	
	@Test
	public void testTransposeMMDenseSparseCP() 
	{
		runTransposeMatrixMultiplicationTest(false, true, ExecType.CP, false);
	}
	
	@Test
	public void testTransposeMMSparseDenseCP() 
	{
		runTransposeMatrixMultiplicationTest(true, false, ExecType.CP, false);
	}
	
	@Test
	public void testTransposeMMSparseSparseCP() 
	{
		runTransposeMatrixMultiplicationTest(true, true, ExecType.CP, false);
	}
	
	@Test
	public void testTransposeMMDenseDenseMR() 
	{
		runTransposeMatrixMultiplicationTest(false, false, ExecType.MR, false);
	}
	
	@Test
	public void testTransposeMMDenseSparseMR() 
	{
		runTransposeMatrixMultiplicationTest(false, true, ExecType.MR, false);
	}
	
	@Test
	public void testTransposeMMSparseDenseMR() 
	{
		runTransposeMatrixMultiplicationTest(true, false, ExecType.MR, false);
	}
	
	@Test
	public void testTransposeMMSparseSparseMR() 
	{
		runTransposeMatrixMultiplicationTest(true, true, ExecType.MR, false);
	}

	@Test
	public void testTransposeMVDenseDenseCP() 
	{
		runTransposeMatrixMultiplicationTest(false, false, ExecType.CP, true);
	}
	
	@Test
	public void testTransposeMVDenseSparseCP() 
	{
		runTransposeMatrixMultiplicationTest(false, true, ExecType.CP, true);
	}
	
	@Test
	public void testTransposeMVSparseDenseCP() 
	{
		runTransposeMatrixMultiplicationTest(true, false, ExecType.CP, true);
	}
	
	@Test
	public void testTransposeMVSparseSparseCP() 
	{
		runTransposeMatrixMultiplicationTest(true, true, ExecType.CP, true);
	}
	
	@Test
	public void testTransposeMVDenseDenseMR() 
	{
		runTransposeMatrixMultiplicationTest(false, false, ExecType.MR, true);
	}
	
	@Test
	public void testTransposeMVDenseSparseMR() 
	{
		runTransposeMatrixMultiplicationTest(false, true, ExecType.MR, true);
	}
	
	@Test
	public void testTransposeMVSparseDenseMR() 
	{
		runTransposeMatrixMultiplicationTest(true, false, ExecType.MR, true);
	}
	
	@Test
	public void testTransposeMVSparseSparseMR() 
	{
		runTransposeMatrixMultiplicationTest(true, true, ExecType.MR, true);
	}
	
	@Test
	public void testTransposeMMMinusDenseDenseCP() 
	{
		runTransposeMatrixMultiplicationTest(false, false, ExecType.CP, false, true);
	}
	
	@Test
	public void testTransposeMVMinusDenseDenseCP() 
	{
		runTransposeMatrixMultiplicationTest(false, false, ExecType.CP, true, true);
	}
	
	@Test
	public void testTransposeMMMinusDenseDenseMR() 
	{
		runTransposeMatrixMultiplicationTest(false, false, ExecType.MR, false, true);
	}
	
	@Test
	public void testTransposeMVMinusDenseDenseMR() 
	{
		runTransposeMatrixMultiplicationTest(false, false, ExecType.MR, true, true);
	}

	/**
	 * 
	 * @param sparseM1
	 * @param sparseM2
	 * @param instType
	 * @param vectorM2
	 */
	private void runTransposeMatrixMultiplicationTest( boolean sparseM1, boolean sparseM2, ExecType instType, boolean vectorM2)
	{
		runTransposeMatrixMultiplicationTest(sparseM1, sparseM2, instType, vectorM2, false);
	}
	
	/**
	 * 
	 * @param sparseM1
	 * @param sparseM2
	 * @param instType
	 */
	private void runTransposeMatrixMultiplicationTest( boolean sparseM1, boolean sparseM2, ExecType instType, boolean vectorM2, boolean minusM1)
	{
		//rtplatform for MR
		RUNTIME_PLATFORM platformOld = rtplatform;
		rtplatform = (instType==ExecType.MR) ? RUNTIME_PLATFORM.HADOOP : RUNTIME_PLATFORM.HYBRID;
	
		int rowsA = vectorM2 ? rowsA2 : rowsA1;
		int colsA = vectorM2 ? colsA2 : colsA1;
		int rowsB = vectorM2 ? rowsB2 : rowsB1;
		int colsB = vectorM2 ? colsB2 : colsB1;
	
		String TEST_NAME = minusM1 ? TEST_NAME2 : TEST_NAME1;
		
		try
		{
			TestConfiguration config = getTestConfiguration(TEST_NAME);
			
			/* This is for running the junit test the new way, i.e., construct the arguments directly */
			String HOME = SCRIPT_DIR + TEST_DIR;
			fullDMLScriptName = HOME + TEST_NAME + ".dml";
			programArgs = new String[]{"-explain","-args", HOME + INPUT_DIR + "A",
					                        Integer.toString(rowsA),
					                        Integer.toString(colsA),
					                        HOME + INPUT_DIR + "B",
					                        Integer.toString(rowsB),
					                        Integer.toString(colsB),
					                        HOME + OUTPUT_DIR + "C"};
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

}