/**
 * IBM Confidential
 * OCO Source Materials
 * (C) Copyright IBM Corp. 2010, 2015
 * The source code for this program is not published or otherwise divested of its trade secrets, irrespective of what has been deposited with the U.S. Copyright Office.
 */

package com.ibm.bi.dml.test.integration.functions.append;

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

public class AppendChainTest extends AutomatedTestBase
{
	@SuppressWarnings("unused")
	private static final String _COPYRIGHT = "Licensed Materials - Property of IBM\n(C) Copyright IBM Corp. 2010, 2015\n" +
                                             "US Government Users Restricted Rights - Use, duplication  disclosure restricted by GSA ADP Schedule Contract with IBM Corp.";
	
	private final static String TEST_NAME = "AppendChainTest";
	private final static String TEST_DIR = "functions/append/";

	private final static double epsilon=0.0000000001;
	private final static int min=1;
	private final static int max=100;
	
	private final static int rows = 1692;
	private final static int cols1 = 1059;
	private final static int cols2a = 1;
	private final static int cols3a = 1;
	private final static int cols2b = 1030;
	private final static int cols3b = 1770;
	
	private final static double sparsity1 = 0.7;
	private final static double sparsity2 = 0.07;
	
	@Override
	public void setUp() {
		TestUtils.clearAssertionInformation();
		addTestConfiguration(TEST_NAME, new TestConfiguration(TEST_DIR, TEST_NAME, 
				new String[] {"C"}));
	}

	@Test
	public void testAppendChainVectorDenseCP() {
		commonAppendTest(RUNTIME_PLATFORM.HYBRID, rows, cols1, cols2a, cols3a, false);
	}
	
	@Test
	public void testAppendChainMatrixDenseCP() {
		commonAppendTest(RUNTIME_PLATFORM.HYBRID, rows, cols1, cols2b, cols3b, false);
	}
	
	// ------------------------------------------------------
	@Test
	public void testAppendChainVectorDenseSP() {
		commonAppendTest(RUNTIME_PLATFORM.SPARK, rows, cols1, cols2a, cols3a, false);
	}
	
	@Test
	public void testAppendChainMatrixDenseSP() {
		commonAppendTest(RUNTIME_PLATFORM.SPARK, rows, cols1, cols2b, cols3b, false);
	}
	
	@Test
	public void testAppendChainVectorSparseSP() {
		commonAppendTest(RUNTIME_PLATFORM.SPARK, rows, cols1, cols2a, cols3a, true);
	}
	
	@Test
	public void testAppendChainMatrixSparseSP() {
		commonAppendTest(RUNTIME_PLATFORM.SPARK, rows, cols1, cols2b, cols3b, true);
	}
	
	// ------------------------------------------------------
	
	@Test
	public void testAppendChainVectorDenseMR() {
		commonAppendTest(RUNTIME_PLATFORM.HADOOP, rows, cols1, cols2a, cols3a, false);
	}
	
	@Test
	public void testAppendChainMatrixDenseMR() {
		commonAppendTest(RUNTIME_PLATFORM.HADOOP, rows, cols1, cols2b, cols3b, false);
	}
	
	@Test
	public void testAppendChainVectorSparseCP() {
		commonAppendTest(RUNTIME_PLATFORM.HYBRID, rows, cols1, cols2a, cols3a, true);
	}
	
	@Test
	public void testAppendChainMatrixSparseCP() {
		commonAppendTest(RUNTIME_PLATFORM.HYBRID, rows, cols1, cols2b, cols3b, true);
	}
	
	@Test
	public void testAppendChainVectorSparseMR() {
		commonAppendTest(RUNTIME_PLATFORM.HADOOP, rows, cols1, cols2a, cols3a, true);
	}
	
	@Test
	public void testAppendChainMatrixSparseMR() {
		commonAppendTest(RUNTIME_PLATFORM.HADOOP, rows, cols1, cols2b, cols3b, true);
	}
			
	/**
	 * 
	 * @param platform
	 * @param rows
	 * @param cols1
	 * @param cols2
	 * @param cols3
	 */
	public void commonAppendTest(RUNTIME_PLATFORM platform, int rows, int cols1, int cols2, int cols3, boolean sparse)
	{
		TestConfiguration config = getTestConfiguration(TEST_NAME);
	    
		RUNTIME_PLATFORM prevPlfm=rtplatform;
		boolean sparkConfigOld = DMLScript.USE_LOCAL_SPARK_CONFIG;
		
		try
		{
		    rtplatform = platform;
		    if( rtplatform == RUNTIME_PLATFORM.SPARK )
				DMLScript.USE_LOCAL_SPARK_CONFIG = true;
	
		    loadTestConfiguration(config);
			
	        config.addVariable("rows", rows);
	        config.addVariable("cols", cols1);
	          
			//This is for running the junit test the new way, i.e., construct the arguments directly
			String RI_HOME = SCRIPT_DIR + TEST_DIR;
			fullDMLScriptName = RI_HOME + TEST_NAME + ".dml";
			programArgs = new String[]{"-args",  RI_HOME + INPUT_DIR + "A" , 
					                             Long.toString(rows), 
					                             Long.toString(cols1),
								                 RI_HOME + INPUT_DIR + "B1" ,
								                 Long.toString(cols2),
								                 RI_HOME + INPUT_DIR + "B2" ,
								                 Long.toString(cols3),
		                                         RI_HOME + OUTPUT_DIR + "C" };
			fullRScriptName = RI_HOME + TEST_NAME + ".R";
			rCmd = "Rscript" + " " + fullRScriptName + " " + 
			       RI_HOME + INPUT_DIR + " "+ RI_HOME + EXPECTED_DIR;
	
			double sparsity = sparse ? sparsity2 : sparsity1; 
			double sparsity2 = 1-sparsity;
	        
			double[][] A = getRandomMatrix(rows, cols1, min, max, sparsity, 11);
	        writeInputMatrix("A", A, true);
	        double[][] B1= getRandomMatrix(rows, cols2, min, max, sparsity2, 21);
	        writeInputMatrix("B1", B1, true);
	        double[][] B2= getRandomMatrix(rows, cols2, min, max, sparsity, 31);
	        writeInputMatrix("B2", B2, true);
	        
	        boolean exceptionExpected = false;
			int expectedCompiledMRJobs = (rtplatform==RUNTIME_PLATFORM.HADOOP)? 2+((cols3>1)?1:0) : 1;
			int expectedExecutedMRJobs = (rtplatform==RUNTIME_PLATFORM.HADOOP)? 2+((cols3>1)?1:0) : 0; 
			runTest(true, exceptionExpected, null, expectedCompiledMRJobs);
			runRScript(true);
			Assert.assertEquals("Wrong number of executed MR jobs.",
					             expectedExecutedMRJobs, Statistics.getNoOfExecutedMRJobs());
			
			//compare result data
			for(String file: config.getOutputFiles())
			{
				HashMap<CellIndex, Double> dmlfile = readDMLMatrixFromHDFS(file);
				HashMap<CellIndex, Double> rfile = readRMatrixFromFS(file);
				TestUtils.compareMatrices(dmlfile, rfile, epsilon, file+"-DML", file+"-R");
			}
		}
		finally
		{
			rtplatform = prevPlfm;
			DMLScript.USE_LOCAL_SPARK_CONFIG = sparkConfigOld;
		}
	}
}
