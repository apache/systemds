/**
 * IBM Confidential
 * OCO Source Materials
 * (C) Copyright IBM Corp. 2010, 2013
 * The source code for this program is not published or otherwise divested of its trade secrets, irrespective of what has been deposited with the U.S. Copyright Office.
 */

package com.ibm.bi.dml.test.integration.functions.binary.matrix;

import org.junit.Test;

import com.ibm.bi.dml.api.DMLScript.RUNTIME_PLATFORM;
import com.ibm.bi.dml.runtime.matrix.MatrixCharacteristics;
import com.ibm.bi.dml.test.integration.AutomatedTestBase;
import com.ibm.bi.dml.test.integration.TestConfiguration;

public class MatrixVectorTest extends AutomatedTestBase 
{
	@SuppressWarnings("unused")
	private static final String _COPYRIGHT = "Licensed Materials - Property of IBM\n(C) Copyright IBM Corp. 2010, 2013\n" +
                                             "US Government Users Restricted Rights - Use, duplication  disclosure restricted by GSA ADP Schedule Contract with IBM Corp.";
	
	private final static String TEST_NAME1 = "MatrixVectorMultiplication";
	private final static String TEST_DIR = "functions/binary/matrix/";
	private final static double eps = 1e-10;
	
	private final static int rows = 3500;
	private final static int cols_wide = 1500;
	private final static int cols_skinny = 500;
	
	private final static double sparsity1 = 0.7;
	private final static double sparsity2 = 0.1;
	
	@Override
	public void setUp() 
	{
		addTestConfiguration(
				TEST_NAME1, 
				new TestConfiguration(TEST_DIR, TEST_NAME1, 
				new String[] { "y" })   ); 
	}
	
	@Test
	public void testMVwideSparseCP() 
	{
		runMatrixVectorMultiplicationTest(cols_wide, RUNTIME_PLATFORM.SINGLE_NODE, true);
	}
	
	@Test
	public void testMVwideSparseMR() 
	{
		runMatrixVectorMultiplicationTest(cols_wide, RUNTIME_PLATFORM.HADOOP, true);
	}

	@Test
	public void testMVwideSparseHYBRID() 
	{
		runMatrixVectorMultiplicationTest(cols_wide, RUNTIME_PLATFORM.HYBRID, true);
	}
	
	@Test
	public void testMVwideDenseCP() 
	{
		runMatrixVectorMultiplicationTest(cols_wide, RUNTIME_PLATFORM.SINGLE_NODE, false);
	}
	
	@Test
	public void testMVwideDenseMR() 
	{
		runMatrixVectorMultiplicationTest(cols_wide, RUNTIME_PLATFORM.HADOOP, false);
	}

	@Test
	public void testMVwideDenseHYBRID() 
	{
		runMatrixVectorMultiplicationTest(cols_wide, RUNTIME_PLATFORM.HYBRID, false);
	}
	
	@Test
	public void testMVskinnySparseCP() 
	{
		runMatrixVectorMultiplicationTest(cols_skinny, RUNTIME_PLATFORM.SINGLE_NODE, true);
	}
	
	@Test
	public void testMVskinnySparseMR() 
	{
		runMatrixVectorMultiplicationTest(cols_skinny, RUNTIME_PLATFORM.HADOOP, true);
	}

	@Test
	public void testMVskinnySparseHYBRID() 
	{
		runMatrixVectorMultiplicationTest(cols_skinny, RUNTIME_PLATFORM.HYBRID, true);
	}
	
	@Test
	public void testMVskinnyDenseCP() 
	{
		runMatrixVectorMultiplicationTest(cols_skinny, RUNTIME_PLATFORM.SINGLE_NODE, false);
	}
	
	@Test
	public void testMVskinnyDenseMR() 
	{
		runMatrixVectorMultiplicationTest(cols_skinny, RUNTIME_PLATFORM.HADOOP, false);
	}

	@Test
	public void testMVskinnyDenseHYBRID() 
	{
		runMatrixVectorMultiplicationTest(cols_skinny, RUNTIME_PLATFORM.HYBRID, false);
	}
	
	private void runMatrixVectorMultiplicationTest( int cols, RUNTIME_PLATFORM rt, boolean sparse )
	{

		RUNTIME_PLATFORM rtold = rtplatform;
		rtplatform = rt;

		try
		{
			TestConfiguration config = getTestConfiguration(TEST_NAME1);
			
			/* This is for running the junit test the new way, i.e., construct the arguments directly */
			String HOME = SCRIPT_DIR + TEST_DIR;
			fullDMLScriptName = HOME + TEST_NAME1 + ".dml";
			programArgs = new String[]{"-args", 
										HOME + INPUT_DIR + "A",
										HOME + INPUT_DIR + "x" ,
					                    HOME + OUTPUT_DIR + "y"
					                  };
			fullRScriptName = HOME + TEST_NAME1 + ".R";
			rCmd = "Rscript" + " " + fullRScriptName + " " + 
			       HOME + INPUT_DIR + " " + HOME + EXPECTED_DIR;
			
			loadTestConfiguration(config);
	
			//generate actual dataset
			double[][] A = getRandomMatrix(rows, cols, 0, 1, sparse?sparsity2:sparsity1, 10); 
			MatrixCharacteristics mc = new MatrixCharacteristics(rows, cols, -1, -1, -1);
			writeInputMatrixWithMTD("A", A, true, mc);
			double[][] x = getRandomMatrix(cols, 1, 0, 1, 1.0, 10); 
			mc = new MatrixCharacteristics(cols, 1, -1, -1, cols);
			writeInputMatrixWithMTD("x", x, true, mc);
	
			boolean exceptionExpected = false;
			runTest(true, exceptionExpected, null, -1); 
			runRScript(true);
			
			compareResultsWithR(eps);
			
		}
		finally
		{
			rtplatform = rtold;
		}
	}
	
}