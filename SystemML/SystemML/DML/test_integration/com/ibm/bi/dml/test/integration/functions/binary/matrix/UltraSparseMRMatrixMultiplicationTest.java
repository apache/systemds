/**
 * IBM Confidential
 * OCO Source Materials
 * (C) Copyright IBM Corp. 2010, 2015
 * The source code for this program is not published or otherwise divested of its trade secrets, irrespective of what has been deposited with the U.S. Copyright Office.
 */

package com.ibm.bi.dml.test.integration.functions.binary.matrix;

import java.util.HashMap;

import org.junit.Test;

import com.ibm.bi.dml.api.DMLScript.RUNTIME_PLATFORM;
import com.ibm.bi.dml.hops.AggBinaryOp;
import com.ibm.bi.dml.hops.AggBinaryOp.MMultMethod;
import com.ibm.bi.dml.lops.LopProperties.ExecType;
import com.ibm.bi.dml.runtime.matrix.data.MatrixValue.CellIndex;
import com.ibm.bi.dml.test.integration.AutomatedTestBase;
import com.ibm.bi.dml.test.integration.TestConfiguration;
import com.ibm.bi.dml.test.utils.TestUtils;

/**
 * Test for MMCJ MR because otherwise seldom (if at all) executed in our testsuite, ultrasparse 
 * in order to account for 'empty block rejection' optimization.
 * 
 * Furthermore, it is at the same time a test for removeEmpty-diag which has special
 * physical operators.
 */
public class UltraSparseMRMatrixMultiplicationTest extends AutomatedTestBase 
{
	@SuppressWarnings("unused")
	private static final String _COPYRIGHT = "Licensed Materials - Property of IBM\n(C) Copyright IBM Corp. 2010, 2015\n" +
                                             "US Government Users Restricted Rights - Use, duplication  disclosure restricted by GSA ADP Schedule Contract with IBM Corp.";
	
	private final static String TEST_NAME1 = "UltraSparseMatrixMultiplication";
	private final static String TEST_NAME2 = "UltraSparseMatrixMultiplication2";
	private final static String TEST_DIR = "functions/binary/matrix/";
	private final static double eps = 1e-10;
	
	private final static int rows = 4045; 
	private final static int cols = 23;
	
	private final static double sparsity1 = 0.7;
	private final static double sparsity2 = 0.1;
	
	
	@Override
	public void setUp() 
	{
		addTestConfiguration(
				TEST_NAME1, 
				new TestConfiguration(TEST_DIR, TEST_NAME1, 
				new String[] { "C" })   ); 
		addTestConfiguration(
				TEST_NAME2, 
				new TestConfiguration(TEST_DIR, TEST_NAME2, 
				new String[] { "C" })   ); 
	}


	
	@Test
	public void testMMRowDenseMR() 
	{
		runMatrixMatrixMultiplicationTest(false, false, ExecType.MR, true, false);
	}
	
	@Test
	public void testMMRowSparseMR() 
	{
		runMatrixMatrixMultiplicationTest(false, true, ExecType.MR, true, false);
	}
	
	@Test
	public void testMMRowDenseMR_PMMJ() 
	{
		runMatrixMatrixMultiplicationTest(false, false, ExecType.MR, true, true);
	}
	
	@Test
	public void testMMRowSparseMR_PMMJ() 
	{
		runMatrixMatrixMultiplicationTest(false, true, ExecType.MR, true, true);
	}
	
	@Test
	public void testMMColDenseMR() 
	{
		runMatrixMatrixMultiplicationTest(false, false, ExecType.MR, false, false);
	}
	
	@Test
	public void testMMColSparseMR() 
	{
		runMatrixMatrixMultiplicationTest(false, true, ExecType.MR, false, false);
	}
	
	

	/**
	 * 
	 * @param sparseM1
	 * @param sparseM2
	 * @param instType
	 */
	private void runMatrixMatrixMultiplicationTest( boolean sparseM1, boolean sparseM2, ExecType instType, boolean rowwise, boolean forcePMMJ)
	{
		//setup exec type, rows, cols

		//rtplatform for MR
		RUNTIME_PLATFORM platformOld = rtplatform;
		rtplatform = (instType==ExecType.MR) ? RUNTIME_PLATFORM.HADOOP : RUNTIME_PLATFORM.HYBRID;
	
		if(forcePMMJ)
			AggBinaryOp.FORCED_MMULT_METHOD = MMultMethod.PMM;
			
		try
		{
			String TEST_NAME = (rowwise) ? TEST_NAME1 : TEST_NAME2;
			TestConfiguration config = getTestConfiguration(TEST_NAME);
			
			/* This is for running the junit test the new way, i.e., construct the arguments directly */
			String HOME = SCRIPT_DIR + TEST_DIR;
			fullDMLScriptName = HOME + TEST_NAME + ".dml";
			programArgs = new String[]{"-explain","-args", HOME + INPUT_DIR + "A",
					                        Integer.toString(rows),
					                        Integer.toString(cols),
					                        HOME + INPUT_DIR + "B",
					                        HOME + OUTPUT_DIR + "C"    };
			fullRScriptName = HOME + TEST_NAME + ".R";
			rCmd = "Rscript" + " " + fullRScriptName + " " + 
			       HOME + INPUT_DIR + " " + HOME + EXPECTED_DIR;
			
			loadTestConfiguration(config);
	
			//generate actual dataset
			double[][] A = getRandomMatrix(rows, cols, 0, 1, sparseM1?sparsity2:sparsity1, 7); 
			writeInputMatrix("A", A, true);
			double[][] B = getRandomMatrix(rows, 1, 0.51, 3.49, 1.0, 3); 
			B = round(B);
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
			AggBinaryOp.FORCED_MMULT_METHOD = null;
		}
	}

	
	private double[][] round(double[][] data) {
		for(int i=0; i<data.length; i++)
			data[i][0]=Math.round(data[i][0]);
		return data;
	}
	
}