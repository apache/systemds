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

package org.apache.sysml.test.integration.functions.io.matrixmarket;

import org.junit.Test;

import org.apache.sysml.api.DMLScript.RUNTIME_PLATFORM;
import org.apache.sysml.hops.OptimizerUtils;
import org.apache.sysml.test.integration.AutomatedTestBase;
import org.apache.sysml.test.integration.TestConfiguration;
import org.apache.sysml.test.utils.TestUtils;

public class ReadMMTest extends AutomatedTestBase 
{
	
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
		boolean oldpar = OptimizerUtils.PARALLEL_CP_READ_TEXTFORMATS;
		
		try
		{
			rtplatform = platform;
			OptimizerUtils.PARALLEL_CP_READ_TEXTFORMATS = parallel;
			
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
			OptimizerUtils.PARALLEL_CP_READ_TEXTFORMATS = oldpar;
		}
	}
	
}