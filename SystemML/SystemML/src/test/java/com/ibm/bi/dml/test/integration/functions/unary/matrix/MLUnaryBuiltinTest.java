/**
 * IBM Confidential
 * OCO Source Materials
 * (C) Copyright IBM Corp. 2010, 2015
 * The source code for this program is not published or otherwise divested of its trade secrets, irrespective of what has been deposited with the U.S. Copyright Office.
 */

package com.ibm.bi.dml.test.integration.functions.unary.matrix;

import java.util.HashMap;

import junit.framework.Assert;

import org.junit.Test;

import com.ibm.bi.dml.api.DMLScript.RUNTIME_PLATFORM;
import com.ibm.bi.dml.lops.LopProperties.ExecType;
import com.ibm.bi.dml.runtime.matrix.data.MatrixValue.CellIndex;
import com.ibm.bi.dml.test.integration.AutomatedTestBase;
import com.ibm.bi.dml.test.integration.TestConfiguration;
import com.ibm.bi.dml.test.utils.TestUtils;
import com.ibm.bi.dml.utils.Statistics;

/**
 * 
 * 
 */
public class MLUnaryBuiltinTest extends AutomatedTestBase 
{
	@SuppressWarnings("unused")
	private static final String _COPYRIGHT = "Licensed Materials - Property of IBM\n(C) Copyright IBM Corp. 2010, 2015\n" +
                                             "US Government Users Restricted Rights - Use, duplication  disclosure restricted by GSA ADP Schedule Contract with IBM Corp.";
	
	private final static String TEST_NAME1 = "SProp";
	private final static String TEST_NAME2 = "Sigmoid";
	
	private final static String TEST_DIR = "functions/unary/matrix/";
	
	private final static double eps = 1e-10;
	
	private final static int rowsMatrix = 1201;
	private final static int colsMatrix = 1103;
	private final static double spSparse = 0.05;
	private final static double spDense = 0.5;
	
	private enum InputType {
		COL_VECTOR,
		MATRIX
	}
	
	@Override
	public void setUp() 
	{
		addTestConfiguration(TEST_NAME1,new TestConfiguration(TEST_DIR, TEST_NAME1,new String[]{"B"}));
		addTestConfiguration(TEST_NAME2,new TestConfiguration(TEST_DIR, TEST_NAME2,new String[]{"B"}));
	}

	
	@Test
	public void testSampleProportionVectorDenseCP() 
	{
		runMLUnaryBuiltinTest(TEST_NAME1, InputType.COL_VECTOR, false, ExecType.CP);
	}
	
	@Test
	public void testSampleProportionVectorSparseCP() 
	{
		runMLUnaryBuiltinTest(TEST_NAME1, InputType.COL_VECTOR, true, ExecType.CP);
	}
	
	@Test
	public void testSampleProportionMatrixDenseCP() 
	{
		runMLUnaryBuiltinTest(TEST_NAME1, InputType.MATRIX, false, ExecType.CP);
	}
	
	@Test
	public void testSampleProportionMatrixSparseCP() 
	{
		runMLUnaryBuiltinTest(TEST_NAME1, InputType.MATRIX, true, ExecType.CP);
	}
	
	@Test
	public void testSampleProportionVectorDenseMR() 
	{
		runMLUnaryBuiltinTest(TEST_NAME1, InputType.COL_VECTOR, false, ExecType.MR);
	}
	
	@Test
	public void testSampleProportionVectorSparseMR() 
	{
		runMLUnaryBuiltinTest(TEST_NAME1, InputType.COL_VECTOR, true, ExecType.MR);
	}
	
	@Test
	public void testSampleProportionMatrixDenseMR() 
	{
		runMLUnaryBuiltinTest(TEST_NAME1, InputType.MATRIX, false, ExecType.MR);
	}
	
	@Test
	public void testSampleProportionMatrixSparseMR() 
	{
		runMLUnaryBuiltinTest(TEST_NAME1, InputType.MATRIX, true, ExecType.MR);
	}
	
	@Test
	public void testSampleProportionVectorDenseSP() 
	{
		if(rtplatform == RUNTIME_PLATFORM.SPARK)
		runMLUnaryBuiltinTest(TEST_NAME1, InputType.COL_VECTOR, false, ExecType.SPARK);
	}
	
	@Test
	public void testSampleProportionVectorSparseSP() 
	{
		if(rtplatform == RUNTIME_PLATFORM.SPARK)
		runMLUnaryBuiltinTest(TEST_NAME1, InputType.COL_VECTOR, true, ExecType.SPARK);
	}
	
	@Test
	public void testSampleProportionMatrixDenseSP() 
	{
		if(rtplatform == RUNTIME_PLATFORM.SPARK)
		runMLUnaryBuiltinTest(TEST_NAME1, InputType.MATRIX, false, ExecType.SPARK);
	}
	
	@Test
	public void testSampleProportionMatrixSparseSP() 
	{
		if(rtplatform == RUNTIME_PLATFORM.SPARK)
		runMLUnaryBuiltinTest(TEST_NAME1, InputType.MATRIX, true, ExecType.SPARK);
	}
	

	@Test
	public void testSigmoidVectorDenseCP() 
	{
		runMLUnaryBuiltinTest(TEST_NAME2, InputType.COL_VECTOR, false, ExecType.CP);
	}
	
	@Test
	public void testSigmoidVectorSparseCP() 
	{
		runMLUnaryBuiltinTest(TEST_NAME2, InputType.COL_VECTOR, true, ExecType.CP);
	}
	
	@Test
	public void testSigmoidMatrixDenseCP() 
	{
		runMLUnaryBuiltinTest(TEST_NAME2, InputType.MATRIX, false, ExecType.CP);
	}
	
	@Test
	public void testSigmoidMatrixSparseCP() 
	{
		runMLUnaryBuiltinTest(TEST_NAME2, InputType.MATRIX, true, ExecType.CP);
	}
	
	@Test
	public void testSigmoidVectorDenseSP() 
	{
		if(rtplatform == RUNTIME_PLATFORM.SPARK)
		runMLUnaryBuiltinTest(TEST_NAME2, InputType.COL_VECTOR, false, ExecType.SPARK);
	}
	
	@Test
	public void testSigmoidVectorSparseSP() 
	{
		if(rtplatform == RUNTIME_PLATFORM.SPARK)
		runMLUnaryBuiltinTest(TEST_NAME2, InputType.COL_VECTOR, true, ExecType.SPARK);
	}
	
	@Test
	public void testSigmoidMatrixDenseSP() 
	{
		if(rtplatform == RUNTIME_PLATFORM.SPARK)
		runMLUnaryBuiltinTest(TEST_NAME2, InputType.MATRIX, false, ExecType.SPARK);
	}
	
	@Test
	public void testSigmoidMatrixSparseSP() 
	{
		if(rtplatform == RUNTIME_PLATFORM.SPARK)
		runMLUnaryBuiltinTest(TEST_NAME2, InputType.MATRIX, true, ExecType.SPARK);
	}
	
	@Test
	public void testSigmoidVectorDenseMR() 
	{
		runMLUnaryBuiltinTest(TEST_NAME2, InputType.COL_VECTOR, false, ExecType.MR);
	}
	
	@Test
	public void testSigmoidVectorSparseMR() 
	{
		runMLUnaryBuiltinTest(TEST_NAME2, InputType.COL_VECTOR, true, ExecType.MR);
	}
	
	@Test
	public void testSigmoidMatrixDenseMR() 
	{
		runMLUnaryBuiltinTest(TEST_NAME2, InputType.MATRIX, false, ExecType.MR);
	}
	
	@Test
	public void testSigmoidMatrixSparseMR() 
	{
		runMLUnaryBuiltinTest(TEST_NAME2, InputType.MATRIX, true, ExecType.MR);
	}

	
	
	/**
	 * 
	 * @param sparseM1
	 * @param sparseM2
	 * @param instType
	 */
	private void runMLUnaryBuiltinTest( String testname, InputType type, boolean sparse, ExecType instType)
	{
		//rtplatform for MR
		RUNTIME_PLATFORM platformOld = rtplatform;
		if(instType == ExecType.SPARK) {
	    	rtplatform = RUNTIME_PLATFORM.SPARK;
	    }
	    else {
	    	rtplatform = (instType==ExecType.MR) ? RUNTIME_PLATFORM.HADOOP : RUNTIME_PLATFORM.HYBRID;
	    }
		try
		{
			int rows = rowsMatrix;
			int cols = (type==InputType.COL_VECTOR) ? 1 : colsMatrix;
			double sparsity = (sparse) ? spSparse : spDense;
			String TEST_NAME = testname;
			
			TestConfiguration config = getTestConfiguration(TEST_NAME);
			
			// This is for running the junit test the new way, i.e., construct the arguments directly
			String HOME = SCRIPT_DIR + TEST_DIR;
			fullDMLScriptName = HOME + TEST_NAME + ".dml";
			programArgs = new String[]{"-stats", "-args", HOME + INPUT_DIR + "A",
					                        HOME + OUTPUT_DIR + "B"    };
			fullRScriptName = HOME + TEST_NAME + ".R";
			rCmd = "Rscript" + " " + fullRScriptName + " " + 
			       HOME + INPUT_DIR + " " + HOME + EXPECTED_DIR;
			
			loadTestConfiguration(config);
	
			//generate actual dataset 
			double[][] A = getRandomMatrix(rows, cols, -0.05, 1, sparsity, 7); 
			writeInputMatrixWithMTD("A", A, true);
	
			runTest(true, false, null, -1); 
			if( instType==ExecType.CP ) //in CP no MR jobs should be executed
				Assert.assertEquals("Unexpected number of executed MR jobs.", 0, Statistics.getNoOfExecutedMRJobs());
			
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