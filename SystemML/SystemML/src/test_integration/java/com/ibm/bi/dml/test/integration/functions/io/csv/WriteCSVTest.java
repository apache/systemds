/**
 * IBM Confidential
 * OCO Source Materials
 * (C) Copyright IBM Corp. 2010, 2013
 * The source code for this program is not published or otherwise divested of its trade secrets, irrespective of what has been deposited with the U.S. Copyright Office.
 */

package com.ibm.bi.dml.test.integration.functions.io.csv;

import org.junit.Test;

import com.ibm.bi.dml.api.DMLScript.RUNTIME_PLATFORM;
import com.ibm.bi.dml.test.integration.AutomatedTestBase;
import com.ibm.bi.dml.test.integration.TestConfiguration;
import com.ibm.bi.dml.test.utils.TestUtils;

/**
 * JUnit Test cases to evaluate the functionality of reading CSV files.
 * 
 * Test 1: write() w/ all properties.
 * Test 2: read(format="csv") w/o mtd file.
 * Test 3: read() w/ complete mtd file.
 *
 */

public class WriteCSVTest extends AutomatedTestBase 
{
	@SuppressWarnings("unused")
	private static final String _COPYRIGHT = "Licensed Materials - Property of IBM\n(C) Copyright IBM Corp. 2010, 2013\n" +
                                             "US Government Users Restricted Rights - Use, duplication  disclosure restricted by GSA ADP Schedule Contract with IBM Corp.";
	
	private final static String TEST_NAME = "WriteCSVTest";
	private final static String TEST_DIR = "functions/io/csv/";
	
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
	public void testCSV1_CP() {
		runCSVWriteTest(RUNTIME_PLATFORM.HYBRID, true, ":", true);
	}
	
	@Test
	public void testCSV1_MR() {
		runCSVWriteTest(RUNTIME_PLATFORM.HADOOP, true, ":", true);
	}
	
	@Test
	public void testCSV2_CP() {
		runCSVWriteTest(RUNTIME_PLATFORM.HYBRID, false, ":", true);
	}
	
	@Test
	public void testCSV2_MR() {
		runCSVWriteTest(RUNTIME_PLATFORM.HADOOP, false, ":", true);
	}
	
	@Test
	public void testCSV3_CP() {
		runCSVWriteTest(RUNTIME_PLATFORM.HYBRID, false, ":", false);
	}
	
	@Test
	public void testCSV3_MR() {
		runCSVWriteTest(RUNTIME_PLATFORM.HADOOP, false, ":", false);
	}
	
	@Test
	public void testCSV4_CP() {
		runCSVWriteTest(RUNTIME_PLATFORM.HYBRID, false, ".", false);
	}
	
	@Test
	public void testCSV4_MR() {
		runCSVWriteTest(RUNTIME_PLATFORM.HADOOP, false, ".", false);
	}
	
	private void runCSVWriteTest(RUNTIME_PLATFORM platform, boolean header, String sep, boolean sparse) {
		
		RUNTIME_PLATFORM oldPlatform = rtplatform;
		rtplatform = platform;
		
		TestConfiguration config = getTestConfiguration(TEST_NAME);
		
		loadTestConfiguration(config);
		
		String HOME = SCRIPT_DIR + TEST_DIR;
		String inputMatrixName = HOME + INPUT_DIR + "transfusion_1"; // always read the same data, independent of testNumber
		String dmlOutput = HOME + OUTPUT_DIR + "dml.scalar";
		String csvOutputName = HOME + OUTPUT_DIR + "transfusion_dml.data";
		String rOutput = HOME + OUTPUT_DIR + "R.scalar";
		
		fullDMLScriptName = HOME + TEST_NAME + ".dml";
		programArgs = new String[]{"-args", inputMatrixName, dmlOutput, csvOutputName, Boolean.toString(header), sep, Boolean.toString(sparse) };
		
		runTest(true, false, null, -1);

		// Verify produced CSV file w/ R
		csvOutputName = TestUtils.processMultiPartCSVForR(csvOutputName);
		fullRScriptName = HOME + "writecsv_verify.R";
		rCmd = "Rscript" + " " + fullRScriptName + " " + csvOutputName + " " + Boolean.toString(header).toUpperCase() + " " + sep + " " + rOutput;
		runRScript(true);
		
		double dmlScalar = TestUtils.readDMLScalar(dmlOutput); 
		double rScalar = TestUtils.readRScalar(rOutput); 
		
		TestUtils.compareScalars(dmlScalar, rScalar, eps);

		rtplatform = oldPlatform;
	}
	
}