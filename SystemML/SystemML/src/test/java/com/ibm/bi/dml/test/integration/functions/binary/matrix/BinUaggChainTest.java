/**
 * IBM Confidential
 * OCO Source Materials
 * (C) Copyright IBM Corp. 2010, 2014
 * The source code for this program is not published or otherwise divested of its trade secrets, irrespective of what has been deposited with the U.S. Copyright Office.
 */

package com.ibm.bi.dml.test.integration.functions.binary.matrix;

import java.util.HashMap;

import junit.framework.Assert;

import org.junit.Test;

import com.ibm.bi.dml.api.DMLScript;
import com.ibm.bi.dml.api.DMLScript.RUNTIME_PLATFORM;
import com.ibm.bi.dml.lops.LopProperties.ExecType;
import com.ibm.bi.dml.runtime.matrix.data.MatrixValue.CellIndex;
import com.ibm.bi.dml.test.integration.AutomatedTestBase;
import com.ibm.bi.dml.test.integration.TestConfiguration;
import com.ibm.bi.dml.test.utils.TestUtils;
import com.ibm.bi.dml.utils.Statistics;

/**
 * TODO: extend test by various binary operator - unary aggregate operator combinations.
 * 
 */
public class BinUaggChainTest extends AutomatedTestBase 
{
	@SuppressWarnings("unused")
	private static final String _COPYRIGHT = "Licensed Materials - Property of IBM\n(C) Copyright IBM Corp. 2010, 2014\n" +
                                             "US Government Users Restricted Rights - Use, duplication  disclosure restricted by GSA ADP Schedule Contract with IBM Corp.";
	
	private final static String TEST_NAME1 = "BinUaggChain_Col";
	private final static String TEST_DIR = "functions/binary/matrix/";
	private final static double eps = 1e-8;
	
	private final static int rows = 1468;
	private final static int cols1 = 73; //single block
	private final static int cols2 = 1052; //multi block
	
	private final static double sparsity1 = 0.5; //dense
	private final static double sparsity2 = 0.1; //sparse
	
	@Override
	public void setUp() 
	{
		TestUtils.clearAssertionInformation();
		addTestConfiguration(TEST_NAME1, new TestConfiguration(TEST_DIR, TEST_NAME1, new String[] { "B" })); 
	}

	@Test
	public void testBinUaggChainColSingleDenseMR() 
	{
		runBinUaggTest(TEST_NAME1, true, false, ExecType.MR);
	}
	
	@Test
	public void testBinUaggChainColSingleSparseMR() 
	{
		runBinUaggTest(TEST_NAME1, true, true, ExecType.MR);
	}
	
	@Test
	public void testBinUaggChainColMultiDenseMR() 
	{
		runBinUaggTest(TEST_NAME1, false, false, ExecType.MR);
	}
	
	@Test
	public void testBinUaggChainColMultiSparseMR() 
	{
		runBinUaggTest(TEST_NAME1, false, true, ExecType.MR);
	}
	
	// -------------------------
	
	@Test
	public void testBinUaggChainColSingleDenseSP() 
	{
		 runBinUaggTest(TEST_NAME1, true, false, ExecType.SPARK);
	}
	
	@Test
	public void testBinUaggChainColSingleSparseSP() 
	{
		runBinUaggTest(TEST_NAME1, true, true, ExecType.SPARK);
	}
	
	@Test
	public void testBinUaggChainColMultiDenseSP() 
	{
		runBinUaggTest(TEST_NAME1, false, false, ExecType.SPARK);
	}
	
	@Test
	public void testBinUaggChainColMultiSparseSP() 
	{
		runBinUaggTest(TEST_NAME1, false, true, ExecType.SPARK);
	}
	
	// ----------------------
	


	/**
	 * 
	 * @param sparseM1
	 * @param sparseM2
	 * @param instType
	 */
	private void runBinUaggTest( String testname, boolean singleBlock, boolean sparse, ExecType instType)
	{
		//rtplatform for MR
		RUNTIME_PLATFORM platformOld = rtplatform;
		switch( instType ){
			case MR: rtplatform = RUNTIME_PLATFORM.HADOOP; break;
			case SPARK: rtplatform = RUNTIME_PLATFORM.SPARK; break;
			default: rtplatform = RUNTIME_PLATFORM.HYBRID; break;
		}
	
		boolean sparkConfigOld = DMLScript.USE_LOCAL_SPARK_CONFIG;
		if( rtplatform == RUNTIME_PLATFORM.SPARK )
			DMLScript.USE_LOCAL_SPARK_CONFIG = true;

		try
		{
			String TEST_NAME = testname;
			TestConfiguration config = getTestConfiguration(TEST_NAME);
			
			/* This is for running the junit test the new way, i.e., construct the arguments directly */
			String HOME = SCRIPT_DIR + TEST_DIR;
			fullDMLScriptName = HOME + TEST_NAME + ".dml";
			programArgs = new String[]{"-args", HOME + INPUT_DIR + "A",
					                            HOME + OUTPUT_DIR + "B"};
			fullRScriptName = HOME + TEST_NAME + ".R";
			rCmd = "Rscript" + " " + fullRScriptName + " " + 
			       HOME + INPUT_DIR + " " + HOME + EXPECTED_DIR;
			
			loadTestConfiguration(config);
	
			//generate actual datasets
			double[][] A = getRandomMatrix(rows, singleBlock?cols1:cols2, -1, 1, sparse?sparsity2:sparsity1, 7);
			writeInputMatrixWithMTD("A", A, true);
			
			runTest(true, false, null, -1); 
			runRScript(true); 
			
			//compare matrices 
			HashMap<CellIndex, Double> dmlfile = readDMLMatrixFromHDFS("B");
			HashMap<CellIndex, Double> rfile  = readRMatrixFromFS("B");
			TestUtils.compareMatrices(dmlfile, rfile, eps, "Stat-DML", "Stat-R");
			
			//check compiled/executed jobs
			if( rtplatform != RUNTIME_PLATFORM.SPARK ) {
				int expectedNumCompiled = (singleBlock)?1:3; 
				int expectedNumExecuted = (singleBlock)?1:3; 
				checkNumCompiledMRJobs(expectedNumCompiled); 
				checkNumExecutedMRJobs(expectedNumExecuted); 	
			}
		}
		finally
		{
			rtplatform = platformOld;
			DMLScript.USE_LOCAL_SPARK_CONFIG = sparkConfigOld;
		}
	}

}