/**
 * IBM Confidential
 * OCO Source Materials
 * (C) Copyright IBM Corp. 2010, 2015
 * The source code for this program is not published or otherwise divested of its trade secrets, irrespective of what has been deposited with the U.S. Copyright Office.
 */

package com.ibm.bi.dml.test.integration.functions.io.csv;

import org.junit.Test;

import com.ibm.bi.dml.api.DMLScript.RUNTIME_PLATFORM;
import com.ibm.bi.dml.hops.OptimizerUtils;
import com.ibm.bi.dml.test.integration.AutomatedTestBase;
import com.ibm.bi.dml.test.integration.TestConfiguration;
import com.ibm.bi.dml.test.utils.TestUtils;

/**
 * JUnit Test cases to evaluate the functionality of reading CSV files.
 * 
 * Test 1: read() w/ mtd file.
 * Test 2: read(format="csv") w/o mtd file.
 * Test 3: read() w/ complete mtd file.
 *
 */

public class ReadCSVTest extends AutomatedTestBase 
{
	@SuppressWarnings("unused")
	private static final String _COPYRIGHT = "Licensed Materials - Property of IBM\n(C) Copyright IBM Corp. 2010, 2015\n" +
                                             "US Government Users Restricted Rights - Use, duplication  disclosure restricted by GSA ADP Schedule Contract with IBM Corp.";
	
	private final static String TEST_NAME = "ReadCSVTest";
	private final static String TEST_DIR = "functions/io/csv/";
	
	private final static double eps = 1e-9;

	@Override
	public void setUp() 
	{
		TestUtils.clearAssertionInformation();
		addTestConfiguration(
				TEST_NAME, 
				new TestConfiguration(TEST_DIR, TEST_NAME, 
				new String[] { "Rout" })   );  
	}
	
	@Test
	public void testCSV1_Sequential_CP1() {
		runCSVTest(1, RUNTIME_PLATFORM.SINGLE_NODE, false);
	}
	
	@Test
	public void testCSV1_Parallel_CP1() {
		runCSVTest(1, RUNTIME_PLATFORM.SINGLE_NODE, true);
	}
	
	@Test
	public void testCSV1_Sequential_CP() {
		runCSVTest(1, RUNTIME_PLATFORM.HYBRID, false);
	}
	
	@Test
	public void testCSV1_Parallel_CP() {
		runCSVTest(1, RUNTIME_PLATFORM.HYBRID, true);
	}
	
	@Test
	public void testCSV1_MR() {
		runCSVTest(1, RUNTIME_PLATFORM.HADOOP, true);
	}
	
	@Test
	public void testCSV2_Sequential_CP1() {
		runCSVTest(2, RUNTIME_PLATFORM.SINGLE_NODE, false);
	}
	
	@Test
	public void testCSV2_Parallel_CP1() {
		runCSVTest(2, RUNTIME_PLATFORM.SINGLE_NODE, true);
	}
	
	@Test
	public void testCSV2_Sequential_CP() {
		runCSVTest(2, RUNTIME_PLATFORM.HYBRID, false);
	}
	
	@Test
	public void testCSV2_Parallel_CP() {
		runCSVTest(2, RUNTIME_PLATFORM.HYBRID, true);
	}
	
	@Test
	public void testCSV2_MR() {
		runCSVTest(2, RUNTIME_PLATFORM.HADOOP, true);
	}

	@Test
	public void testCSV3_Sequential_CP1() {
		runCSVTest(3, RUNTIME_PLATFORM.SINGLE_NODE, false);
	}
	
	@Test
	public void testCSV3_Parallel_CP1() {
		runCSVTest(3, RUNTIME_PLATFORM.SINGLE_NODE, true);
	}
	
	@Test
	public void testCSV3_Sequential_CP() {
		runCSVTest(3, RUNTIME_PLATFORM.HYBRID, false);
	}
	
	@Test
	public void testCSV3_Parallel_CP() {
		runCSVTest(3, RUNTIME_PLATFORM.HYBRID, true);
	}
	
	@Test
	public void testCSV3_MR() {
		runCSVTest(3, RUNTIME_PLATFORM.HADOOP, false);
	}
	
	/**
	 * 
	 * @param testNumber
	 * @param platform
	 * @param parallel
	 */
	private void runCSVTest(int testNumber, RUNTIME_PLATFORM platform, boolean parallel) 
	{
		
		RUNTIME_PLATFORM oldPlatform = rtplatform;
		boolean oldpar = OptimizerUtils.PARALLEL_CP_READ_TEXTFORMATS;
		
		try
		{
			rtplatform = platform;
			OptimizerUtils.PARALLEL_CP_READ_TEXTFORMATS = parallel;
			
			TestConfiguration config = getTestConfiguration(TEST_NAME);
			
			loadTestConfiguration(config);
			
			String HOME = SCRIPT_DIR + TEST_DIR;
			String inputMatrixName = HOME + INPUT_DIR + "transfusion_" + testNumber + ".data";
			String dmlOutput = HOME + OUTPUT_DIR + "dml.scalar";
			String rOutput = HOME + OUTPUT_DIR + "R.scalar";
			
			fullDMLScriptName = HOME + TEST_NAME + "_" + testNumber + ".dml";
			programArgs = new String[]{"-args", inputMatrixName, dmlOutput};
			
			fullRScriptName = HOME + "csv_verify2.R";
			rCmd = "Rscript" + " " + fullRScriptName + " " + inputMatrixName + ".single " + rOutput;
			
			runTest(true, false, null, -1);
			runRScript(true);
			
			double dmlScalar = TestUtils.readDMLScalar(dmlOutput); 
			double rScalar = TestUtils.readRScalar(rOutput); 
			
			TestUtils.compareScalars(dmlScalar, rScalar, eps);
		}
		finally
		{
			rtplatform = oldPlatform;
			OptimizerUtils.PARALLEL_CP_READ_TEXTFORMATS = oldpar;		
		}
	}
	
}