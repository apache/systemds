/**
 * IBM Confidential
 * OCO Source Materials
 * (C) Copyright IBM Corp. 2010, 2015
 * The source code for this program is not published or otherwise divested of its trade secrets, irrespective of what has been deposited with the U.S. Copyright Office.
 */

package com.ibm.bi.dml.test.integration.functions.parfor;

import java.util.HashMap;

import org.junit.Assert;
import org.junit.Test;

import com.ibm.bi.dml.api.DMLScript;
import com.ibm.bi.dml.api.DMLScript.RUNTIME_PLATFORM;
import com.ibm.bi.dml.runtime.matrix.data.MatrixValue.CellIndex;
import com.ibm.bi.dml.test.integration.AutomatedTestBase;
import com.ibm.bi.dml.test.integration.TestConfiguration;
import com.ibm.bi.dml.test.utils.TestUtils;
import com.ibm.bi.dml.utils.Statistics;

public class ParForSerialRemoteResultMergeTest extends AutomatedTestBase 
{
	@SuppressWarnings("unused")
	private static final String _COPYRIGHT = "Licensed Materials - Property of IBM\n(C) Copyright IBM Corp. 2010, 2015\n" +
                                             "US Government Users Restricted Rights - Use, duplication  disclosure restricted by GSA ADP Schedule Contract with IBM Corp.";
	
	private final static String TEST_NAME1 = "parfor_pr_resultmerge1a"; //MR w/o compare
	private final static String TEST_NAME2 = "parfor_pr_resultmerge1b"; //MR w/ compare
	private final static String TEST_NAME3 = "parfor_pr_resultmerge1c"; //Spark w/o compare
	private final static String TEST_NAME4 = "parfor_pr_resultmerge1d"; //Spark w/ compare
	
	private final static String TEST_DIR = "functions/parfor/";
	private final static double eps = 1e-10;
	
	private final static int rows = 1100;  
	private final static int cols = 70;  
	
	private final static double sparsity1 = 0.7;
	private final static double sparsity2 = 0.1d;
	
	
	@Override
	public void setUp() 
	{
		addTestConfiguration(
				TEST_NAME1, new TestConfiguration(TEST_DIR, TEST_NAME1, 
				new String[] { "R" })   );
	
		addTestConfiguration(
				TEST_NAME2, new TestConfiguration(TEST_DIR, TEST_NAME2, 
				new String[] { "R" })   );
		
		addTestConfiguration(
				TEST_NAME3, new TestConfiguration(TEST_DIR, TEST_NAME3, 
				new String[] { "R" })   );
	
		addTestConfiguration(
				TEST_NAME4, new TestConfiguration(TEST_DIR, TEST_NAME4, 
				new String[] { "R" })   );
	}

	@Test
	public void testSingleResultMergeDenseMR() 
	{
		runParallelRemoteResultMerge(TEST_NAME1, false);
	}
	
	@Test
	public void testSingleResultMergeSparseMR() 
	{
		runParallelRemoteResultMerge(TEST_NAME1, true);
	}
	
	@Test
	public void testSingleResultMergeCompareDenseMR() 
	{
		runParallelRemoteResultMerge(TEST_NAME2, false);
	}
	
	@Test
	public void testSingleResultMergeCompareSparseMR() 
	{
		runParallelRemoteResultMerge(TEST_NAME2, true);
	}
	
	@Test
	public void testSingleResultMergeDenseSP() 
	{
		runParallelRemoteResultMerge(TEST_NAME3, false);
	}
	
	@Test
	public void testSingleResultMergeSparseSP() 
	{
		runParallelRemoteResultMerge(TEST_NAME3, true);
	}
	
	@Test
	public void testSingleResultMergeCompareDenseSP() 
	{
		runParallelRemoteResultMerge(TEST_NAME4, false);
	}
	
	@Test
	public void testSingleResultMergeCompareSparseSP() 
	{
		runParallelRemoteResultMerge(TEST_NAME4, true);
	}
	
	
	/**
	 * 
	 * @param outer execution mode of outer parfor loop
	 * @param inner execution mode of inner parfor loop
	 * @param instType execution mode of instructions
	 */
	private void runParallelRemoteResultMerge( String test_name, boolean sparse )
	{
		RUNTIME_PLATFORM oldRT = rtplatform;
		boolean oldUseSparkConfig = DMLScript.USE_LOCAL_SPARK_CONFIG;
		
		if( test_name.equals(TEST_NAME3) || test_name.equals(TEST_NAME4)  ) {
			DMLScript.USE_LOCAL_SPARK_CONFIG = true;
			rtplatform = RUNTIME_PLATFORM.HYBRID_SPARK;
		}
		
		//inst exec type, influenced via rows
		String TEST_NAME = test_name;
			
		try
		{
			//script
			TestConfiguration config = getTestConfiguration(TEST_NAME);
			config.addVariable("rows", rows);
			config.addVariable("cols", cols);
			
			/* This is for running the junit test the new way, i.e., construct the arguments directly */
			String HOME = SCRIPT_DIR + TEST_DIR;
			fullDMLScriptName = HOME + TEST_NAME + ".dml";
			programArgs = new String[]{"-args", HOME + INPUT_DIR + "V" , 
					                        Integer.toString(rows),
					                        Integer.toString(cols),
					                        HOME + OUTPUT_DIR + "R" };
			fullRScriptName = HOME + TEST_NAME + ".R";
			rCmd = "Rscript" + " " + fullRScriptName + " " + 
			       HOME + INPUT_DIR + " " + HOME + EXPECTED_DIR;
			
			loadTestConfiguration(config);
	
			long seed = System.nanoTime();
			double sparsity = -1;
			if( sparse )
				sparsity = sparsity2;
			else
				sparsity = sparsity1;
	        double[][] V = getRandomMatrix(rows, cols, 0, 1, sparsity, seed);
			writeInputMatrix("V", V, true);
	
			boolean exceptionExpected = false;
			runTest(true, exceptionExpected, null, -1);
			runRScript(true);
			
			//compare num MR jobs
			int expectedMRJobs = ( test_name.equals(TEST_NAME3) || test_name.equals(TEST_NAME4)  ) ? 0 : 2; 
			Assert.assertEquals("Unexpected number of executed MR jobs.", expectedMRJobs, Statistics.getNoOfExecutedMRJobs());	
			
			//compare matrices
			HashMap<CellIndex, Double> dmlfile = readDMLMatrixFromHDFS("R");
			HashMap<CellIndex, Double> rfile  = readRMatrixFromFS("Rout");
			TestUtils.compareMatrices(dmlfile, rfile, eps, "DML", "R");
		}
		finally
		{
			rtplatform = oldRT;
			DMLScript.USE_LOCAL_SPARK_CONFIG = oldUseSparkConfig;	
		}
	}
}