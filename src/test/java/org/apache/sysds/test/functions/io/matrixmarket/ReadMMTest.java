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

package org.apache.sysds.test.functions.io.matrixmarket;

import org.junit.Test;
import org.apache.sysds.common.Types.ExecMode;
import org.apache.sysds.conf.CompilerConfig;
import org.apache.sysds.test.AutomatedTestBase;
import org.apache.sysds.test.TestConfiguration;
import org.apache.sysds.test.TestUtils;

public class ReadMMTest extends AutomatedTestBase 
{
	private final static String TEST_NAME = "ReadMMTest";
	private final static String TEST_DIR = "functions/io/matrixmarket/";
	private final static String TEST_CLASS_DIR = TEST_DIR + ReadMMTest.class.getSimpleName() + "/";
	
	private final static double eps = 1e-9;

	@Override
	public void setUp() 
	{
		addTestConfiguration(TEST_NAME, 
			new TestConfiguration(TEST_CLASS_DIR, TEST_NAME, new String[] { "Rout" }) );
	}
	
	@Test
	public void testMatrixMarket1_Sequential_CP() {
		runMMTest(1, ExecMode.SINGLE_NODE, false);
	}
	
	@Test
	public void testMatrixMarket1_Parallel_CP() {
		runMMTest(1, ExecMode.SINGLE_NODE, true);
	}
	
	@Test
	public void testMatrixMarket2_Sequential_CP() {
		runMMTest(2, ExecMode.SINGLE_NODE, false);
	}
	
	@Test
	public void testMatrixMarket2_ParallelCP() {
		runMMTest(2, ExecMode.SINGLE_NODE, true);
	}
	
	@Test
	public void testMatrixMarket3_Sequential_CP() {
		runMMTest(3, ExecMode.SINGLE_NODE, false);
	}
	
	@Test
	public void testMatrixMarket3_Parallel_CP() {
		runMMTest(3, ExecMode.SINGLE_NODE, true);
	}
	
	private void runMMTest(int testNumber, ExecMode platform, boolean parallel) {
		
		ExecMode oldPlatform = rtplatform;
		boolean oldpar = CompilerConfig.FLAG_PARREADWRITE_TEXT;
		
		try
		{
			rtplatform = platform;
			CompilerConfig.FLAG_PARREADWRITE_TEXT = parallel;
			
			TestConfiguration config = getTestConfiguration(TEST_NAME);
			loadTestConfiguration(config);
			
			String HOME = SCRIPT_DIR + TEST_DIR;
			String inputMatrixName = HOME + INPUT_DIR + "ReadMMTest.mtx";
			String dmlOutput = output("dml.scalar");
			String rOutput = output("R.scalar");
			
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
			CompilerConfig.FLAG_PARREADWRITE_TEXT = oldpar;
		}
	}
	
}