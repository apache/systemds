/**
 * IBM Confidential
 * OCO Source Materials
 * (C) Copyright IBM Corp. 2010, 2015
 * The source code for this program is not published or otherwise divested of its trade secrets, irrespective of what has been deposited with the U.S. Copyright Office.
 */

package com.ibm.bi.dml.test.integration.functions.binary.matrix;

import java.util.HashMap;

import org.junit.Test;

import com.ibm.bi.dml.api.DMLScript;
import com.ibm.bi.dml.api.DMLScript.RUNTIME_PLATFORM;
import com.ibm.bi.dml.lops.LopProperties.ExecType;
import com.ibm.bi.dml.runtime.matrix.MatrixCharacteristics;
import com.ibm.bi.dml.runtime.matrix.data.MatrixValue.CellIndex;
import com.ibm.bi.dml.test.integration.AutomatedTestBase;
import com.ibm.bi.dml.test.integration.TestConfiguration;
import com.ibm.bi.dml.test.utils.TestUtils;

/**
 * TODO: extend test by various binary operator - unary aggregate operator combinations.
 * 
 */
public class UaggOuterChainTest extends AutomatedTestBase 
{
	@SuppressWarnings("unused")
	private static final String _COPYRIGHT = "Licensed Materials - Property of IBM\n(C) Copyright IBM Corp. 2010, 2015\n" +
                                             "US Government Users Restricted Rights - Use, duplication  disclosure restricted by GSA ADP Schedule Contract with IBM Corp.";
	
	private final static String TEST_NAME1 = "UaggOuterChain";
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
		addTestConfiguration(TEST_NAME1, new TestConfiguration(TEST_DIR, TEST_NAME1, new String[] { "C" })); 
	}

	@Test
	public void testUaggOuterChainSingleDenseMR() 
	{
		runBinUaggTest(TEST_NAME1, true, false, ExecType.MR);
	}
	
	@Test
	public void testUaggOuterChainSingleSparseMR() 
	{
		runBinUaggTest(TEST_NAME1, true, true, ExecType.MR);
	}
	
	@Test
	public void testUaggOuterChainMultiDenseMR() 
	{
		runBinUaggTest(TEST_NAME1, false, false, ExecType.MR);
	}
	
	@Test
	public void testUaggOuterChainMultiSparseMR() 
	{
		runBinUaggTest(TEST_NAME1, false, true, ExecType.MR);
	}
	
	// -------------------------
	/*
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
	*/
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
					                            HOME + INPUT_DIR + "B",
					                            HOME + OUTPUT_DIR + "C"};
			fullRScriptName = HOME + TEST_NAME + ".R";
			rCmd = "Rscript" + " " + fullRScriptName + " " + 
			       HOME + INPUT_DIR + " " + HOME + EXPECTED_DIR;
			
			loadTestConfiguration(config);
	
			//generate actual datasets
			double[][] A = getRandomMatrix(rows, 1, -1, 1, sparse?sparsity2:sparsity1, 235);
			writeInputMatrixWithMTD("A", A, true);
			double[][] B = getRandomMatrix(1, singleBlock?cols1:cols2, -1, 1, sparse?sparsity2:sparsity1, 124);
			writeInputMatrixWithMTD("B", B, true);
			
			runTest(true, false, null, -1); 
			runRScript(true); 
			
			//compare matrices 
			HashMap<CellIndex, Double> dmlfile = readDMLMatrixFromHDFS("C");
			HashMap<CellIndex, Double> rfile  = readRMatrixFromFS("C");
			TestUtils.compareMatrices(dmlfile, rfile, eps, "Stat-DML", "Stat-R");
			checkDMLMetaDataFile("C", new MatrixCharacteristics(rows,1,1,1)); //rowsums
			
			//check compiled/executed jobs
			if( rtplatform != RUNTIME_PLATFORM.SPARK ) {
				int expectedNumCompiled = 2; //reblock+gmr if uaggouterchain; otherwise 3 
				int expectedNumExecuted = expectedNumCompiled; 
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