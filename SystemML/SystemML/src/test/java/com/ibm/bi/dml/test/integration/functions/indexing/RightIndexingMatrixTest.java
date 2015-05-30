/**
 * IBM Confidential
 * OCO Source Materials
 * (C) Copyright IBM Corp. 2010, 2015
 * The source code for this program is not published or otherwise divested of its trade secrets, irrespective of what has been deposited with the U.S. Copyright Office.
 */

package com.ibm.bi.dml.test.integration.functions.indexing;

import java.util.HashMap;
import java.util.Random;

import org.junit.Test;

import com.ibm.bi.dml.api.DMLScript;
import com.ibm.bi.dml.api.DMLScript.RUNTIME_PLATFORM;
import com.ibm.bi.dml.lops.LopProperties.ExecType;
import com.ibm.bi.dml.runtime.matrix.data.MatrixValue.CellIndex;
import com.ibm.bi.dml.test.integration.AutomatedTestBase;
import com.ibm.bi.dml.test.integration.TestConfiguration;
import com.ibm.bi.dml.test.utils.TestUtils;



public class RightIndexingMatrixTest extends AutomatedTestBase
{
	@SuppressWarnings("unused")
	private static final String _COPYRIGHT = "Licensed Materials - Property of IBM\n(C) Copyright IBM Corp. 2010, 2015\n" +
                                             "US Government Users Restricted Rights - Use, duplication  disclosure restricted by GSA ADP Schedule Contract with IBM Corp.";
	
	private final static String TEST_NAME = "RightIndexingMatrixTest";
	private final static String TEST_DIR = "functions/indexing/";

	private final static double epsilon=0.0000000001;
	private final static int rows = 2279;
	private final static int cols = 1050;
	private final static int min=0;
	private final static int max=100;
	
	private final static double sparsity1 = 0.5;
	private final static double sparsity2 = 0.01;
	
	
	@Override
	public void setUp() {
		addTestConfiguration(TEST_NAME, new TestConfiguration(TEST_DIR, TEST_NAME, 
				new String[] {"B", "C", "D"}));
	}
	
	@Test
	public void testRightIndexingDenseCP() 
	{
		runRightIndexingTest(ExecType.CP, false);
	}
	
	@Test
	public void testRightIndexingDenseSP() 
	{
		runRightIndexingTest(ExecType.SPARK, false);
	}
	
	@Test
	public void testRightIndexingDenseMR() 
	{
		runRightIndexingTest(ExecType.MR, false);
	}
	
	@Test
	public void testRightIndexingSparseCP() 
	{
		runRightIndexingTest(ExecType.CP, true);
	}
	
	@Test
	public void testRightIndexingSparseSP() 
	{
		runRightIndexingTest(ExecType.SPARK, true);
	}
	
	@Test
	public void testRightIndexingSparseMR() 
	{
		runRightIndexingTest(ExecType.MR, true);
	}
	
	/**
	 * 
	 * @param et
	 * @param sparse
	 */
	public void runRightIndexingTest( ExecType et, boolean sparse ) 
	{
		RUNTIME_PLATFORM oldRTP = rtplatform;
			
		boolean sparkConfigOld = DMLScript.USE_LOCAL_SPARK_CONFIG;
		
		try
		{
		    TestConfiguration config = getTestConfiguration(TEST_NAME);
		    if(et == ExecType.SPARK) {
		    	rtplatform = RUNTIME_PLATFORM.SPARK;
		    }
		    else {
		    	rtplatform = (et==ExecType.MR)? RUNTIME_PLATFORM.HADOOP : RUNTIME_PLATFORM.SINGLE_NODE;
		    }
			if( rtplatform == RUNTIME_PLATFORM.SPARK )
				DMLScript.USE_LOCAL_SPARK_CONFIG = true;
			
		    
		    double sparsity = sparse ? sparsity2 : sparsity1;
		    
	        config.addVariable("rows", rows);
	        config.addVariable("cols", cols);
	        
	        long rowstart=216, rowend=429, colstart=967, colend=1009;
	        Random rand=new Random(System.currentTimeMillis());
	        rowstart=(long)(rand.nextDouble()*((double)rows))+1;
	        rowend=(long)(rand.nextDouble()*((double)(rows-rowstart+1)))+rowstart;
	        colstart=(long)(rand.nextDouble()*((double)cols))+1;
	        colend=(long)(rand.nextDouble()*((double)(cols-colstart+1)))+colstart;
	        config.addVariable("rowstart", rowstart);
	        config.addVariable("rowend", rowend);
	        config.addVariable("colstart", colstart);
	        config.addVariable("colend", colend);
	        
			/* This is for running the junit test the new way, i.e., construct the arguments directly */
			String RI_HOME = SCRIPT_DIR + TEST_DIR;
			fullDMLScriptName = RI_HOME + TEST_NAME + ".dml";
			programArgs = new String[]{"-args",  RI_HOME + INPUT_DIR + "A" , 
		               			Long.toString(rows), Long.toString(cols),
		                        Long.toString(rowstart), Long.toString(rowend),
		                        Long.toString(colstart), Long.toString(colend),
		                         RI_HOME + OUTPUT_DIR + "B" , 
		                         RI_HOME + OUTPUT_DIR + "C" , 
		                         RI_HOME + OUTPUT_DIR + "D" };
			fullRScriptName = RI_HOME + TEST_NAME + ".R";
			rCmd = "Rscript" + " " + fullRScriptName + " " + 
			       RI_HOME + INPUT_DIR + " "+rowstart+" "+rowend+" "+colstart+" "+colend+" " + RI_HOME + EXPECTED_DIR;
	
			loadTestConfiguration(config);
			double[][] A = getRandomMatrix(rows, cols, min, max, sparsity, System.currentTimeMillis());
	        writeInputMatrix("A", A, true);
	        
	        boolean exceptionExpected = false;
			int expectedNumberOfJobs = -1;
			runTest(true, exceptionExpected, null, expectedNumberOfJobs);
			
			runRScript(true);
			//disableOutAndExpectedDeletion();
		
			for(String file: config.getOutputFiles())
			{
				HashMap<CellIndex, Double> dmlfile = readDMLMatrixFromHDFS(file);
				HashMap<CellIndex, Double> rfile = readRMatrixFromFS(file);
				TestUtils.compareMatrices(dmlfile, rfile, epsilon, file+"-DML", file+"-R");
			}
		}
		finally
		{
			rtplatform = oldRTP;
			DMLScript.USE_LOCAL_SPARK_CONFIG = sparkConfigOld;
		}
	}
}
