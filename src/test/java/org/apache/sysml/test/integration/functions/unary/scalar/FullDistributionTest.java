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

package org.apache.sysml.test.integration.functions.unary.scalar;

import java.util.HashMap;
import java.util.Random;

import org.junit.Test;
import org.apache.sysml.api.DMLScript;
import org.apache.sysml.api.DMLScript.RUNTIME_PLATFORM;
import org.apache.sysml.lops.LopProperties.ExecType;
import org.apache.sysml.runtime.matrix.data.MatrixValue.CellIndex;
import org.apache.sysml.test.integration.AutomatedTestBase;
import org.apache.sysml.test.integration.TestConfiguration;
import org.apache.sysml.test.utils.TestUtils;

/**
 * Test case for all cdf distribution functions, where we test the specific builtin 
 * functions (which are equivalent to the generic cdf with specific parameterizations) 
 *
 */
public class FullDistributionTest extends AutomatedTestBase 
{
	private final static String TEST_NAME = "DFTest";
	private final static String TEST_DIR = "functions/unary/scalar/";
	private final static String TEST_CLASS_DIR = TEST_DIR + FullDistributionTest.class.getSimpleName() + "/";
	
	private enum TEST_TYPE { 
		NORMAL, NORMAL_NOPARAMS, NORMAL_MEAN, 
		NORMAL_SD, F, T, CHISQ, EXP, EXP_NOPARAMS 
	};
	
	
	@Override
	public void setUp() {
		TestUtils.clearAssertionInformation();
		addTestConfiguration(TEST_NAME, new TestConfiguration(TEST_CLASS_DIR, TEST_NAME, new String[] { "dfout" }));
	}
	
	@Test
	public void testNormalCP() {
		runDFTest(TEST_TYPE.NORMAL, true, 1.0, 2.0, ExecType.CP);
	}
	
	@Test
	public void testNormalNoParamsCP() {
		runDFTest(TEST_TYPE.NORMAL_NOPARAMS, true, null, null, ExecType.CP);
	}
	
	@Test
	public void testNormalMeanCP() {
		runDFTest(TEST_TYPE.NORMAL_MEAN, true, 1.0, null, ExecType.CP);
	}
	
	@Test
	public void testNormalSdCP() {
		runDFTest(TEST_TYPE.NORMAL_SD, true, 2.0, null, ExecType.CP);
	}
	
	@Test
	public void testTCP() {
		runDFTest(TEST_TYPE.T, true, 10.0, null, ExecType.CP);
	}
	
	@Test
	public void testFCP() {
		runDFTest(TEST_TYPE.T, true, 10.0, 20.0, ExecType.CP);
	}
	
	@Test
	public void testChisqCP() {
		runDFTest(TEST_TYPE.CHISQ, true, 10.0, null, ExecType.CP);
	}
	
	@Test
	public void testExpCP() {
		runDFTest(TEST_TYPE.EXP, true, 5.0, null, ExecType.CP);
	}

	@Test
	public void testNormalSpark() {
		runDFTest(TEST_TYPE.NORMAL, true, 1.0, 2.0, ExecType.SPARK);
	}
	
	@Test
	public void testNormalNoParamsSpark() {
		runDFTest(TEST_TYPE.NORMAL_NOPARAMS, true, null, null, ExecType.SPARK);
	}
	
	@Test
	public void testNormalMeanSpark() {
		runDFTest(TEST_TYPE.NORMAL_MEAN, true, 1.0, null, ExecType.SPARK);
	}
	
	@Test
	public void testNormalSdSpark() {
		runDFTest(TEST_TYPE.NORMAL_SD, true, 2.0, null, ExecType.SPARK);
	}
	
	@Test
	public void testTSpark() {
		runDFTest(TEST_TYPE.T, true, 10.0, null, ExecType.SPARK);
	}
	
	@Test
	public void testFSpark() {
		runDFTest(TEST_TYPE.T, true, 10.0, 20.0, ExecType.SPARK);
	}
	
	@Test
	public void testChisqSpark() {
		runDFTest(TEST_TYPE.CHISQ, true, 10.0, null, ExecType.SPARK);
	}
	
	@Test
	public void testExpSpark() {
		runDFTest(TEST_TYPE.EXP, true, 5.0, null, ExecType.SPARK);
	}
	
	@Test
	public void testNormalMR() {
		runDFTest(TEST_TYPE.NORMAL, true, 1.0, 2.0, ExecType.MR);
	}
	
	@Test
	public void testNormalNoParamsMR() {
		runDFTest(TEST_TYPE.NORMAL_NOPARAMS, true, null, null, ExecType.MR);
	}
	
	@Test
	public void testNormalMeanMR() {
		runDFTest(TEST_TYPE.NORMAL_MEAN, true, 1.0, null, ExecType.MR);
	}
	
	@Test
	public void testNormalSdMR() {
		runDFTest(TEST_TYPE.NORMAL_SD, true, 2.0, null, ExecType.MR);
	}
	
	@Test
	public void testTMR() {
		runDFTest(TEST_TYPE.T, true, 10.0, null, ExecType.MR);
	}
	
	@Test
	public void testFMR() {
		runDFTest(TEST_TYPE.T, true, 10.0, 20.0, ExecType.MR);
	}
	
	@Test
	public void testChisqMR() {
		runDFTest(TEST_TYPE.CHISQ, true, 10.0, null, ExecType.MR);
	}
	
	@Test
	public void testExpMR() {
		runDFTest(TEST_TYPE.EXP, true, 5.0, null, ExecType.MR);
	}
	
	/**
	 * Internal test method - all these tests are expected to run in CP independent of the passed
	 * instType. However, we test all backends to ensure correct compilation in the presence of
	 * forced execution types.
	 * 
	 * @param type
	 * @param inverse
	 * @param param1
	 * @param param2
	 * @param instType
	 */
	private void runDFTest(TEST_TYPE type, boolean inverse, Double param1, Double param2, ExecType instType) 
	{
		//setup multi backend configuration
		RUNTIME_PLATFORM platformOld = rtplatform;
		switch( instType ){
			case MR: rtplatform = RUNTIME_PLATFORM.HADOOP; break;
			case SPARK: rtplatform = RUNTIME_PLATFORM.SPARK; break;
			default: rtplatform = RUNTIME_PLATFORM.HYBRID; break;
		}
		boolean sparkConfigOld = DMLScript.USE_LOCAL_SPARK_CONFIG;
		if( rtplatform == RUNTIME_PLATFORM.SPARK )
			DMLScript.USE_LOCAL_SPARK_CONFIG = true;
		
		try
		{
			//set test and parameters
			getAndLoadTestConfiguration(TEST_NAME);
			double in = (new Random(System.nanoTime())).nextDouble();
			String HOME = SCRIPT_DIR + TEST_DIR;
			fullDMLScriptName = HOME + TEST_NAME + "_" + type.toString() + ".dml";
			fullRScriptName = HOME + TEST_NAME + "_" + type.toString() + ".R";
			
			switch(type) {
				case NORMAL_NOPARAMS:
					programArgs = new String[]{"-args", Double.toString(in), output("dfout") };
					rCmd = "Rscript" + " " + fullRScriptName + " " + Double.toString(in) + " " + expected("dfout");
					break;
					
				case NORMAL_MEAN:
				case NORMAL_SD:
				case T:
				case CHISQ:
				case EXP:
					programArgs = new String[]{"-args", Double.toString(in), Double.toString(param1), output("dfout") };
					rCmd = "Rscript" + " " + fullRScriptName + " " + Double.toString(in) + " " + Double.toString(param1) + " " + expected("dfout");
					break;
					
				case NORMAL:
				case F:
					programArgs = new String[]{"-args", Double.toString(in), Double.toString(param1), Double.toString(param2), output("dfout") };
					rCmd = "Rscript" + " " + fullRScriptName + " " + Double.toString(in) + " " + Double.toString(param1) + " " + Double.toString(param2) + " " + expected("dfout");
					break;
				
				default: 
					throw new RuntimeException("Invalid distribution function: " + type);
			}
			
			//run test
			runTest(true, false, null, -1); 
			runRScript(true); 
			
			//compare results
			HashMap<CellIndex, Double> dmlfile = readDMLMatrixFromHDFS("dfout");
			HashMap<CellIndex, Double> rfile  = readRMatrixFromFS("dfout");
			TestUtils.compareMatrices(dmlfile, rfile, 1e-8, "DMLout", "Rout");
		}
		finally {
			rtplatform = platformOld;
			DMLScript.USE_LOCAL_SPARK_CONFIG = sparkConfigOld;
		}
	}
}
