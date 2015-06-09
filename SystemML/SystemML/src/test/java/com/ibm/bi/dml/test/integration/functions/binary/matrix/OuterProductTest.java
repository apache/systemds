/**
 * IBM Confidential
 * OCO Source Materials
 * (C) Copyright IBM Corp. 2010, 2015
 * The source code for this program is not published or otherwise divested of its trade secrets, irrespective of what has been deposited with the U.S. Copyright Office.
 */

package com.ibm.bi.dml.test.integration.functions.binary.matrix;

import java.util.HashMap;

import org.junit.Assert;
import org.junit.Test;

import com.ibm.bi.dml.api.DMLScript.RUNTIME_PLATFORM;
import com.ibm.bi.dml.lops.LopProperties.ExecType;
import com.ibm.bi.dml.runtime.matrix.data.MatrixValue.CellIndex;
import com.ibm.bi.dml.test.integration.AutomatedTestBase;
import com.ibm.bi.dml.test.integration.TestConfiguration;

public class OuterProductTest extends AutomatedTestBase 
{
	@SuppressWarnings("unused")
	private static final String _COPYRIGHT = "Licensed Materials - Property of IBM\n(C) Copyright IBM Corp. 2010, 2015\n" +
                                             "US Government Users Restricted Rights - Use, duplication  disclosure restricted by GSA ADP Schedule Contract with IBM Corp.";
	
	private final static String TEST_NAME = "OuterProduct";
	private final static String TEST_DIR = "functions/binary/matrix/";
	
	private final static int rows = 41456;
	private final static int cols = 9703;
	
	private final static double sparsity1 = 0.7;
	private final static double sparsity2 = 0.1;
	
	
	@Override
	public void setUp() 
	{
		addTestConfiguration(TEST_NAME, new TestConfiguration(TEST_DIR, TEST_NAME, new String[] { "C" })   ); 
	}

	
	@Test
	public void testMMDenseDenseMR() 
	{
		runMatrixMatrixMultiplicationTest(false, false, ExecType.MR);
	}
	
	@Test
	public void testMMSparseSparseMR() 
	{
		runMatrixMatrixMultiplicationTest(true, true, ExecType.MR);
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
			programArgs = new String[]{"-explain","-args", HOME + INPUT_DIR + "A",
					                            HOME + INPUT_DIR + "B",
					                            HOME + OUTPUT_DIR + "C"    };
			fullRScriptName = HOME + TEST_NAME + ".R";
			rCmd = "Rscript" + " " + fullRScriptName + " " + 
			       HOME + INPUT_DIR + " " + HOME + EXPECTED_DIR;
			
			loadTestConfiguration(config);
	
			//generate actual dataset
			double[][] A = getRandomMatrix(rows, 1, -1, 1, sparseM1?sparsity2:sparsity1, 7); 
			writeInputMatrixWithMTD("A", A, true);
			double[][] B = getRandomMatrix(1, cols, -1, 1, sparseM2?sparsity2:sparsity1, 3); 
			writeInputMatrixWithMTD("B", B, true);
	
			//run tests
			runTest(true, false, null, -1); 
			//runRScript(true); R fails here with out-of-memory 
			
			//compare matrices (single minimum)
			HashMap<CellIndex, Double> dmlfile = readDMLMatrixFromHDFS("C");
			//HashMap<CellIndex, Double> rfile  = readRMatrixFromFS("C");
			//TestUtils.compareMatrices(dmlfile, rfile, eps, "Stat-DML", "Stat-R");
			Double dmlret = dmlfile.get(new CellIndex(1,1));
			Double compare = computeMinOuterProduct(A, B, rows, cols);
			Assert.assertEquals("Wrong result value.", compare, dmlret);
			
			int expectedNumCompiled = 4; //REBLOCK, MMRJ, GMR, GMR write
			int expectedNumExecuted = 4; //REBLOCK, MMRJ, GMR, GMR write
			
			checkNumCompiledMRJobs(expectedNumCompiled); 
			checkNumExecutedMRJobs(expectedNumExecuted); 
		
		}
		finally
		{
			rtplatform = platformOld;
		}
	}
	
	/**
	 * Min over outer product for comparison because R runs out of memory.
	 * 
	 * @param A
	 * @param B
	 * @param rows
	 * @param cols
	 * @return
	 */
	public double computeMinOuterProduct( double[][] A, double[][] B, int rows, int cols )
	{
		double min = Double.MAX_VALUE;
		
		for( int i=0; i<rows; i++ )
		{
			double val1 = A[i][0];
			if( val1!=0 || min <= 0 )
				for( int j=0; j<cols; j++ )
				{
					double val2 = B[0][j];
					double val3 = val1 * val2;
					min = Math.min(min, val3);
				}
		}
		
		return min;
	}
}