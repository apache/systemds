/**
 * IBM Confidential
 * OCO Source Materials
 * (C) Copyright IBM Corp. 2010, 2014
 * The source code for this program is not published or otherwise divested of its trade secrets, irrespective of what has been deposited with the U.S. Copyright Office.
 */

package com.ibm.bi.dml.test.integration.functions.indexing;

import java.util.HashMap;

import org.junit.Test;

import com.ibm.bi.dml.api.DMLScript.RUNTIME_PLATFORM;
import com.ibm.bi.dml.lops.LopProperties.ExecType;
import com.ibm.bi.dml.runtime.matrix.data.MatrixValue.CellIndex;
import com.ibm.bi.dml.test.integration.AutomatedTestBase;
import com.ibm.bi.dml.test.integration.TestConfiguration;
import com.ibm.bi.dml.test.utils.TestUtils;



public class Jdk7IssueRightIndexingTest extends AutomatedTestBase
{
	@SuppressWarnings("unused")
	private static final String _COPYRIGHT = "Licensed Materials - Property of IBM\n(C) Copyright IBM Corp. 2010, 2013\n" +
                                             "US Government Users Restricted Rights - Use, duplication  disclosure restricted by GSA ADP Schedule Contract with IBM Corp.";
	
	private final static String TEST_NAME = "Jdk7IssueTest";
	private final static String TEST_DIR = "functions/indexing/";

	private final static double eps = 0.0000000001;
	private final static int rows = 1000000;
	private final static int cols = 10;
	private final static double sparsity1 = 1.0;
	private final static double sparsity2 = 0.1;
					
	
	@Override
	public void setUp() {
		addTestConfiguration(TEST_NAME, new TestConfiguration(TEST_DIR, TEST_NAME, new String[] {"R"}));
	}
	
	@Test
	public void testIndexingDenseCP() 
	{
		runIndexingTest(false, ExecType.CP);
	}
	
	@Test
	public void testIndexingSparseCP() 
	{
		runIndexingTest(true, ExecType.CP);
	}
	
	@Test
	public void testIndexingDenseHybrid() 
	{
		runIndexingTest(false, null);
	}
	
	@Test
	public void testIndexingSparseHybrid() 
	{
		runIndexingTest(true, null);
	}
	
	@Test
	public void testIndexingDenseMR() 
	{
		runIndexingTest(false, ExecType.MR);
	}
	
	@Test
	public void testIndexingSparseMR() 
	{
		runIndexingTest(true, ExecType.MR);
	}
	
	/**
	 * 
	 * @param sparse
	 * @param et
	 */
	public void runIndexingTest( boolean sparse, ExecType et ) 
	{
		RUNTIME_PLATFORM oldRTP = rtplatform;
				
		try
		{
		    TestConfiguration config = getTestConfiguration(TEST_NAME);
		    rtplatform = (et==null) ? RUNTIME_PLATFORM.HYBRID : 
		    	         (et==ExecType.MR)? RUNTIME_PLATFORM.HADOOP : RUNTIME_PLATFORM.SINGLE_NODE;
			
		    config.addVariable("rows", rows);
	        config.addVariable("cols", cols);
	        
	        String RI_HOME = SCRIPT_DIR + TEST_DIR;
			fullDMLScriptName = RI_HOME + TEST_NAME + ".dml";
			programArgs = new String[]{"-args",  RI_HOME + INPUT_DIR + "M" ,
		                                         RI_HOME + OUTPUT_DIR + "R" };
			fullRScriptName = RI_HOME + TEST_NAME + ".R";
			rCmd = "Rscript" + " " + fullRScriptName + " " + 
			       RI_HOME + INPUT_DIR + " " + RI_HOME + EXPECTED_DIR;
	
			loadTestConfiguration(config);
			
			//generate input data
			double sparsity = sparse ? sparsity2 : sparsity1;
			double[][] M = getRandomMatrix(rows, cols, 0, 1, sparsity, 7);
	        writeInputMatrixWithMTD("M", M, true);
	        
	        //run test
			runTest(true, false, null, -1);			
			runRScript(true);
		
			//compare results
			HashMap<CellIndex, Double> dmlfile = readDMLMatrixFromHDFS("R");
			HashMap<CellIndex, Double> rfile  = readRMatrixFromFS("R");
			TestUtils.compareMatrices(dmlfile, rfile, eps, "Stat-DML", "Stat-R");
		}
		finally
		{
			rtplatform = oldRTP;
		}
	}
}
