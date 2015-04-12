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
import com.ibm.bi.dml.hops.OptimizerUtils;
import com.ibm.bi.dml.lops.LopProperties.ExecType;
import com.ibm.bi.dml.runtime.matrix.data.MatrixValue.CellIndex;
import com.ibm.bi.dml.test.integration.AutomatedTestBase;
import com.ibm.bi.dml.test.integration.TestConfiguration;
import com.ibm.bi.dml.test.utils.TestUtils;

public class DiagMatrixMultiplicationTest extends AutomatedTestBase 
{
	@SuppressWarnings("unused")
	private static final String _COPYRIGHT = "Licensed Materials - Property of IBM\n(C) Copyright IBM Corp. 2010, 2014\n" +
                                             "US Government Users Restricted Rights - Use, duplication  disclosure restricted by GSA ADP Schedule Contract with IBM Corp.";
	
	private final static String TEST_NAME1 = "DiagMatrixMultiplication";
	private final static String TEST_NAME2 = "DiagMatrixMultiplicationTranspose";
	private final static String TEST_DIR = "functions/binary/matrix/";
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
		addTestConfiguration(
				TEST_NAME1, 
				new TestConfiguration(TEST_DIR, TEST_NAME1, new String[] { "C" })   ); 
		addTestConfiguration(
				TEST_NAME2, 
				new TestConfiguration(TEST_DIR, TEST_NAME2, new String[] { "C" })   ); 
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
		if(rtplatform == RUNTIME_PLATFORM.SPARK)  
			runDiagMatrixMultiplicationTest(false, false, false, false, ExecType.SPARK, true);
	}
	
	@Test
	public void testDiagMMDenseDenseTransposeSP() 
	{
		//should apply diag_mm / t_t rewrite
		if(rtplatform == RUNTIME_PLATFORM.SPARK)  
			runDiagMatrixMultiplicationTest(false, false, true, false, ExecType.SPARK, true);
	}
	
	@Test
	public void testDiagMVDenseDenseSP() 
	{
		//should not apply diag_mm rewrite
		if(rtplatform == RUNTIME_PLATFORM.SPARK)  
			runDiagMatrixMultiplicationTest(false, false, false, true, ExecType.SPARK, true);
	}
	
	@Test
	public void testDiagMMSparseSparseSP() 
	{
		//should apply diag_mm rewrite
		if(rtplatform == RUNTIME_PLATFORM.SPARK)  
			runDiagMatrixMultiplicationTest(true, true, false, false, ExecType.SPARK, true);
	}
	
	@Test
	public void testDiagMMSparseSparseTransposeSP() 
	{
		//should apply diag_mm / t_t rewrite
		if(rtplatform == RUNTIME_PLATFORM.SPARK)  
			runDiagMatrixMultiplicationTest(true, true, true, false, ExecType.SPARK, true);
	}
	
	@Test
	public void testDiagMVSparseSparseSP() 
	{
		//should not apply diag_mm rewrite
		if(rtplatform == RUNTIME_PLATFORM.SPARK)  
			runDiagMatrixMultiplicationTest(true, true, false, true, ExecType.SPARK, true);
	}
	
	// --------------------------------------------------------
	
	@Test
	public void testDiagMMDenseDenseMR() 
	{
		//should apply diag_mm rewrite
		runDiagMatrixMultiplicationTest(false, false, false, false, ExecType.MR, true);
	}
	
	@Test
	public void testDiagMMDenseDenseTransposeMR() 
	{
		//should apply diag_mm / t_t rewrite
		runDiagMatrixMultiplicationTest(false, false, true, false, ExecType.MR, true);
	}
	
	@Test
	public void testDiagMVDenseDenseMR() 
	{
		//should not apply diag_mm rewrite
		runDiagMatrixMultiplicationTest(false, false, false, true, ExecType.MR, true);
	}
	
	@Test
	public void testDiagMMSparseSparseMR() 
	{
		//should apply diag_mm rewrite
		runDiagMatrixMultiplicationTest(true, true, false, false, ExecType.MR, true);
	}
	
	@Test
	public void testDiagMMSparseSparseTransposeMR() 
	{
		//should apply diag_mm / t_t rewrite
		runDiagMatrixMultiplicationTest(true, true, true, false, ExecType.MR, true);
	}
	
	@Test
	public void testDiagMVSparseSparseMR() 
	{
		runDiagMatrixMultiplicationTest(true, true, false, true, ExecType.MR, true);
	}

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

	@Test
	public void testDiagMMDenseDenseNoSimplifyMR() 
	{
		runDiagMatrixMultiplicationTest(false, false, false, false, ExecType.MR, false);
	}
	
	@Test
	public void testDiagMMDenseDenseTransposeNoSimplifyMR() 
	{
		runDiagMatrixMultiplicationTest(false, false, true, false, ExecType.MR, false);
	}
	
	@Test
	public void testDiagMVDenseDenseNoSimplifyMR() 
	{
		runDiagMatrixMultiplicationTest(false, false, false, true, ExecType.MR, false);
	}
	
	@Test
	public void testDiagMMSparseSparseNoSimplifyMR() 
	{
		runDiagMatrixMultiplicationTest(true, true, false, false, ExecType.MR, false);
	}
	
	@Test
	public void testDiagMMSparseSparseTransposeNoSimplifyMR() 
	{
		runDiagMatrixMultiplicationTest(true, true, true, false, ExecType.MR, false);
	}
	
	@Test
	public void testDiagMVSparseSparseNoSimplifyMR() 
	{
		runDiagMatrixMultiplicationTest(true, true, false, true, ExecType.MR, false);
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
		RUNTIME_PLATFORM platformOld = rtplatform;
		if(instType == ExecType.SPARK) {
	    	rtplatform = RUNTIME_PLATFORM.SPARK;
	    }
	    else {
			rtplatform = (instType==ExecType.MR) ? RUNTIME_PLATFORM.HADOOP : RUNTIME_PLATFORM.HYBRID;
	    }
		
		String TEST_NAME = rightTranspose ? TEST_NAME2 : TEST_NAME1;
		boolean oldFlagSimplify = OptimizerUtils.ALLOW_ALGEBRAIC_SIMPLIFICATION;
		
		try
		{
			TestConfiguration config = getTestConfiguration(TEST_NAME);
			
			OptimizerUtils.ALLOW_ALGEBRAIC_SIMPLIFICATION = simplify;
			
			/* This is for running the junit test the new way, i.e., construct the arguments directly */
			String HOME = SCRIPT_DIR + TEST_DIR;
			fullDMLScriptName = HOME + TEST_NAME + ".dml";
			programArgs = new String[]{"-explain","-args", HOME + INPUT_DIR + "A",
					                        Integer.toString(rowsA),
					                        Integer.toString(colsA),
					                        HOME + INPUT_DIR + "B",
					                        Integer.toString(rowsB),
					                        Integer.toString(rightVector?1:colsB),
					                        HOME + OUTPUT_DIR + "C"};
			fullRScriptName = HOME + TEST_NAME + ".R";
			rCmd = "Rscript" + " " + fullRScriptName + " " + 
			       HOME + INPUT_DIR + " " + HOME + EXPECTED_DIR;
			
			loadTestConfiguration(config);
	
			
			//generate actual dataset
			double[][] A = getRandomMatrix(rowsA, colsA, 0, 1, sparseM1?sparsity2:sparsity1, 7); 
			writeInputMatrix("A", A, true);
			double[][] B = getRandomMatrix(rowsB, rightVector?1:colsB, 0, 1, sparseM2?sparsity2:sparsity1, 3); 
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
		}
	}

}