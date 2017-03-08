/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 * 
 *   http://www.apache.org/licenses/LICENSE-2.0
 * 
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

package org.apache.sysml.test.integration.functions.io.csv;

import org.junit.Test;
import org.apache.sysml.api.DMLScript;
import org.apache.sysml.api.DMLScript.RUNTIME_PLATFORM;
import org.apache.sysml.conf.CompilerConfig;
import org.apache.sysml.test.integration.AutomatedTestBase;
import org.apache.sysml.test.integration.TestConfiguration;
import org.apache.sysml.test.utils.TestUtils;

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
	private final static String TEST_NAME = "ReadCSVTest";
	private final static String TEST_DIR = "functions/io/csv/";
	private final static String TEST_CLASS_DIR = TEST_DIR + ReadCSVTest.class.getSimpleName() + "/";
	
	private final static double eps = 1e-9;

	@Override
	public void setUp() 
	{
		TestUtils.clearAssertionInformation();
		addTestConfiguration(TEST_NAME, 
			new TestConfiguration(TEST_CLASS_DIR, TEST_NAME, new String[] { "Rout" }) );  
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
	public void testCSV1_SP() {
		runCSVTest(1, RUNTIME_PLATFORM.SPARK, true);
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
	public void testCSV2_SP() {
		runCSVTest(2, RUNTIME_PLATFORM.SPARK, true);
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
	
	@Test
	public void testCSV3_SP() {
		runCSVTest(3, RUNTIME_PLATFORM.SPARK, false);
	}

	private void runCSVTest(int testNumber, RUNTIME_PLATFORM platform, boolean parallel) 
	{
		RUNTIME_PLATFORM oldPlatform = rtplatform;
		rtplatform = platform;
		
		boolean sparkConfigOld = DMLScript.USE_LOCAL_SPARK_CONFIG;
		if( rtplatform == RUNTIME_PLATFORM.SPARK )
			DMLScript.USE_LOCAL_SPARK_CONFIG = true;
		
		boolean oldpar = CompilerConfig.FLAG_PARREADWRITE_TEXT;
		
		try
		{
			CompilerConfig.FLAG_PARREADWRITE_TEXT = parallel;
			
			TestConfiguration config = getTestConfiguration(TEST_NAME);
			
			loadTestConfiguration(config);
			
			String HOME = SCRIPT_DIR + TEST_DIR;
			String inputMatrixNameNoExtension = HOME + INPUT_DIR + "transfusion_" + testNumber;
			String inputMatrixNameWithExtension = inputMatrixNameNoExtension + ".csv";
			String dmlOutput = output("dml.scalar");
			String rOutput = output("R.scalar");
			
			fullDMLScriptName = HOME + TEST_NAME + "_" + testNumber + ".dml";
			programArgs = new String[]{"-args", inputMatrixNameWithExtension, dmlOutput};
			
			fullRScriptName = HOME + "csv_verify2.R";
			rCmd = "Rscript" + " " + fullRScriptName + " " + inputMatrixNameNoExtension + ".single.csv " + rOutput;
			
			runTest(true, false, null, -1);
			runRScript(true);
			
			double dmlScalar = TestUtils.readDMLScalar(dmlOutput); 
			double rScalar = TestUtils.readRScalar(rOutput); 
			
			TestUtils.compareScalars(dmlScalar, rScalar, eps);
		}
		finally {
			rtplatform = oldPlatform;
			CompilerConfig.FLAG_PARREADWRITE_TEXT = oldpar;		
			DMLScript.USE_LOCAL_SPARK_CONFIG = sparkConfigOld;
		}
	}
	
}