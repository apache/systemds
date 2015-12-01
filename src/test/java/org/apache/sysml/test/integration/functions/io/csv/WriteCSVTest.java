/**
 * (C) Copyright IBM Corp. 2010, 2015
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * 
 */

package org.apache.sysml.test.integration.functions.io.csv;

import java.io.IOException;

import org.junit.Test;

import org.apache.sysml.api.DMLScript.RUNTIME_PLATFORM;
import org.apache.sysml.test.integration.AutomatedTestBase;
import org.apache.sysml.test.integration.TestConfiguration;
import org.apache.sysml.test.utils.TestUtils;

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
	public void testCSV1_CP() throws IOException {
		runCSVWriteTest(RUNTIME_PLATFORM.HYBRID, true, ":", true);
	}
	
	@Test
	public void testCSV1_MR() throws IOException {
		runCSVWriteTest(RUNTIME_PLATFORM.HADOOP, true, ":", true);
	}
	
	@Test
	public void testCSV2_CP() throws IOException {
		runCSVWriteTest(RUNTIME_PLATFORM.HYBRID, false, ":", true);
	}
	
	@Test
	public void testCSV2_MR() throws IOException {
		runCSVWriteTest(RUNTIME_PLATFORM.HADOOP, false, ":", true);
	}
	
	@Test
	public void testCSV3_CP() throws IOException {
		runCSVWriteTest(RUNTIME_PLATFORM.HYBRID, false, ":", false);
	}
	
	@Test
	public void testCSV3_MR() throws IOException {
		runCSVWriteTest(RUNTIME_PLATFORM.HADOOP, false, ":", false);
	}
	
	@Test
	public void testCSV4_CP() throws IOException {
		runCSVWriteTest(RUNTIME_PLATFORM.HYBRID, false, ".", false);
	}
	
	@Test
	public void testCSV4_MR() throws IOException {
		runCSVWriteTest(RUNTIME_PLATFORM.HADOOP, false, ".", false);
	}
	
	private void runCSVWriteTest(RUNTIME_PLATFORM platform, boolean header, String sep, boolean sparse) throws IOException {
		
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
		programArgs = new String[]{"-explain" ,
				"-args", inputMatrixName, dmlOutput, csvOutputName, 
				Boolean.toString(header), sep, Boolean.toString(sparse) };
		
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