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

package org.apache.sysds.test.functions.io.csv;

import org.junit.Test;
import org.apache.sysds.api.DMLScript;
import org.apache.sysds.common.Types.ExecMode;
import org.apache.sysds.conf.CompilerConfig;
import org.apache.sysds.test.AutomatedTestBase;
import org.apache.sysds.test.TestConfiguration;
import org.apache.sysds.test.TestUtils;

/**
 * JUnit Test cases to evaluate the functionality of reading CSV files.
 * 
 * Test 1: read() w/ mtd file.
 * Test 2: read(format="csv") w/o mtd file.
 * Test 3: read() w/ complete mtd file.
 *
 */

@net.jcip.annotations.NotThreadSafe
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
		runCSVTest(1, ExecMode.SINGLE_NODE, false);
	}
	
	@Test
	public void testCSV1_Parallel_CP1() {
		runCSVTest(1, ExecMode.SINGLE_NODE, true);
	}
	
	@Test
	public void testCSV1_Sequential_CP() {
		runCSVTest(1, ExecMode.HYBRID, false);
	}
	
	@Test
	public void testCSV1_Parallel_CP() {
		runCSVTest(1, ExecMode.HYBRID, true);
	}
	
	@Test
	public void testCSV1_SP() {
		runCSVTest(1, ExecMode.SPARK, true);
	}
	
	@Test
	public void testCSV2_Sequential_CP1() {
		runCSVTest(2, ExecMode.SINGLE_NODE, false);
	}
	
	@Test
	public void testCSV2_Parallel_CP1() {
		runCSVTest(2, ExecMode.SINGLE_NODE, true);
	}
	
	@Test
	public void testCSV2_Sequential_CP() {
		runCSVTest(2, ExecMode.HYBRID, false);
	}
	
	@Test
	public void testCSV2_Parallel_CP() {
		runCSVTest(2, ExecMode.HYBRID, true);
	}
	
	@Test
	public void testCSV2_SP() {
		runCSVTest(2, ExecMode.SPARK, true);
	}

	@Test
	public void testCSV3_Sequential_CP1() {
		runCSVTest(3, ExecMode.SINGLE_NODE, false);
	}
	
	@Test
	public void testCSV3_Parallel_CP1() {
		runCSVTest(3, ExecMode.SINGLE_NODE, true);
	}
	
	@Test
	public void testCSV3_Sequential_CP() {
		runCSVTest(3, ExecMode.HYBRID, false);
	}
	
	@Test
	public void testCSV3_Parallel_CP() {
		runCSVTest(3, ExecMode.HYBRID, true);
	}
	
	@Test
	public void testCSV3_SP() {
		runCSVTest(3, ExecMode.SPARK, false);
	}

	private void runCSVTest(int testNumber, ExecMode platform, boolean parallel) 
	{
		ExecMode oldPlatform = rtplatform;
		rtplatform = platform;
		
		boolean sparkConfigOld = DMLScript.USE_LOCAL_SPARK_CONFIG;
		if( rtplatform == ExecMode.SPARK )
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