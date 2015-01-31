/**
 * IBM Confidential
 * OCO Source Materials
 * (C) Copyright IBM Corp. 2010, 2015
 * The source code for this program is not published or otherwise divested of its trade secrets, irrespective of what has been deposited with the U.S. Copyright Office.
 */

package com.ibm.bi.dml.test.integration.functions.io.matrixmarket;

import org.junit.Test;

import com.ibm.bi.dml.api.DMLScript.RUNTIME_PLATFORM;
import com.ibm.bi.dml.hops.OptimizerUtils;
import com.ibm.bi.dml.test.integration.AutomatedTestBase;
import com.ibm.bi.dml.test.integration.TestConfiguration;
import com.ibm.bi.dml.test.utils.TestUtils;

public class ReadMMTest extends AutomatedTestBase 
{
	@SuppressWarnings("unused")
	private static final String _COPYRIGHT = "Licensed Materials - Property of IBM\n(C) Copyright IBM Corp. 2010, 2015\n" +
                                             "US Government Users Restricted Rights - Use, duplication  disclosure restricted by GSA ADP Schedule Contract with IBM Corp.";
	
	private final static String TEST_NAME = "ReadMMTest";
	private final static String TEST_DIR = "functions/io/matrixmarket/";
	
	private final static double eps = 1e-9;

	@Override
	public void setUp() 
	{
		addTestConfiguration(
				TEST_NAME, 
				new TestConfiguration(TEST_DIR, TEST_NAME, 
				new String[] { "Rout" })   );  
	}
	
	@Test
	public void testMatrixMarket1_Sequential_CP() {
		runMMTest(1, RUNTIME_PLATFORM.SINGLE_NODE, false);
	}
	
	@Test
	public void testMatrixMarket1_Parallel_CP() {
		runMMTest(1, RUNTIME_PLATFORM.SINGLE_NODE, true);
	}
	
	@Test
	public void testMatrixMarket1_MR() {
		runMMTest(1, RUNTIME_PLATFORM.HADOOP, true);
	}
	
	@Test
	public void testMatrixMarket2_Sequential_CP() {
		runMMTest(2, RUNTIME_PLATFORM.SINGLE_NODE, false);
	}
	
	@Test
	public void testMatrixMarket2_ParallelCP() {
		runMMTest(2, RUNTIME_PLATFORM.SINGLE_NODE, true);
	}
	
	@Test
	public void testMatrixMarket2_MR() {
		runMMTest(2, RUNTIME_PLATFORM.HADOOP, true);
	}
	
	@Test
	public void testMatrixMarket3_Sequential_CP() {
		runMMTest(3, RUNTIME_PLATFORM.SINGLE_NODE, false);
	}
	
	@Test
	public void testMatrixMarket3_Parallel_CP() {
		runMMTest(3, RUNTIME_PLATFORM.SINGLE_NODE, true);
	}
	
	@Test
	public void testMatrixMarket3_MR() {
		runMMTest(3, RUNTIME_PLATFORM.HADOOP, true);
	}
	
	private void runMMTest(int testNumber, RUNTIME_PLATFORM platform, boolean parallel) {
		
		RUNTIME_PLATFORM oldPlatform = rtplatform;
		boolean oldpar = OptimizerUtils.PARALLEL_READ_TEXTFORMATS;
		
		try
		{
			rtplatform = platform;
			OptimizerUtils.PARALLEL_READ_TEXTFORMATS = parallel;
			
			TestConfiguration config = getTestConfiguration(TEST_NAME);
			
			loadTestConfiguration(config);
			
			String HOME = SCRIPT_DIR + TEST_DIR;
			String inputMatrixName = HOME + INPUT_DIR + "ReadMMTest.mtx";
			String dmlOutput = HOME + OUTPUT_DIR + "dml.scalar";
			String rOutput = HOME + OUTPUT_DIR + "R.scalar";
			
			fullDMLScriptName = HOME + TEST_NAME + "_1.dml";
			programArgs = new String[]{"-args", inputMatrixName, dmlOutput};
			
			fullRScriptName = HOME + "mm_verify.R";
			rCmd = "Rscript" + " " + fullRScriptName + " " + inputMatrixName + " " + rOutput;
			
			runTest(true, false, null, -1);
			runRScript(true);
			
			double dmlScalar = TestUtils.readDMLScalar(dmlOutput); 
			double rScalar = TestUtils.readRScalar(rOutput); 
			
			TestUtils.compareScalars(dmlScalar, rScalar, eps);
		}
		finally
		{
			rtplatform = oldPlatform;
			OptimizerUtils.PARALLEL_READ_TEXTFORMATS = oldpar;
		}
	}
	
}