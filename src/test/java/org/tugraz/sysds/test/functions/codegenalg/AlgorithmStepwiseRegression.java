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

package org.tugraz.sysds.test.functions.codegenalg;

import java.io.File;

import org.junit.Assert;
import org.junit.Test;
import org.tugraz.sysds.api.DMLScript;
import org.tugraz.sysds.common.Types.ExecMode;
import org.tugraz.sysds.hops.OptimizerUtils;
import org.tugraz.sysds.lops.LopProperties.ExecType;
import org.tugraz.sysds.test.AutomatedTestBase;
import org.tugraz.sysds.test.TestConfiguration;
import org.tugraz.sysds.test.TestUtils;

public class AlgorithmStepwiseRegression extends AutomatedTestBase 
{
	private final static String TEST_NAME1 = "Algorithm_Stepwise";
	private final static String TEST_DIR = "functions/codegenalg/";
	private final static String TEST_CLASS_DIR = TEST_DIR + AlgorithmStepwiseRegression.class.getSimpleName() + "/";
	private final static String TEST_CONF_DEFAULT = "SystemDS-config-codegen.xml";
	private final static File TEST_CONF_FILE_DEFAULT = new File(SCRIPT_DIR + TEST_DIR, TEST_CONF_DEFAULT);
	private final static String TEST_CONF_FUSE_ALL = "SystemDS-config-codegen-fuse-all.xml";
	private final static File TEST_CONF_FILE_FUSE_ALL = new File(SCRIPT_DIR + TEST_DIR, TEST_CONF_FUSE_ALL);
	private final static String TEST_CONF_FUSE_NO_REDUNDANCY = "SystemDS-config-codegen-fuse-no-redundancy.xml";
	private final static File TEST_CONF_FILE_FUSE_NO_REDUNDANCY = new File(SCRIPT_DIR + TEST_DIR,
			TEST_CONF_FUSE_NO_REDUNDANCY);

	private enum TestType { DEFAULT,FUSE_ALL,FUSE_NO_REDUNDANCY }

	private final static int rows = 2468;
	private final static int cols = 200;
	
	private final static double sparsity1 = 0.7; //dense
	private final static double sparsity2 = 0.1; //sparse
	
	private final static int icpt = 0;
	private final static double thr = 0.01;
	
	public enum StepwiseType {
		GLM_PROBIT,
		LINREG_DS,
	}
	
	private TestType currentTestType = TestType.DEFAULT;
	
	@Override
	public void setUp() {
		TestUtils.clearAssertionInformation();
		addTestConfiguration(TEST_NAME1, new TestConfiguration(TEST_CLASS_DIR, TEST_NAME1, new String[] { "w" })); 
	}

	@Test
	public void testStepwiseGLMDenseRewritesCP() {
		runStepwiseTest(StepwiseType.GLM_PROBIT, false, true, ExecType.CP, TestType.DEFAULT);
	}
	
	@Test
	public void testStepwiseGLMSparseRewritesCP() {
		runStepwiseTest(StepwiseType.GLM_PROBIT, true, true, ExecType.CP, TestType.DEFAULT);
	}
	
	@Test
	public void testStepwiseGLMDenseNoRewritesCP() {
		runStepwiseTest(StepwiseType.GLM_PROBIT, false, false, ExecType.CP, TestType.DEFAULT);
	}
	
	@Test
	public void testStepwiseGLMSparseNoRewritesCP() {
		runStepwiseTest(StepwiseType.GLM_PROBIT, true, false, ExecType.CP, TestType.DEFAULT);
	}
	
//	@Test
//	public void testStepwiseGLMDenseRewritesSP() {
//		runStepwiseTest(StepwiseType.GLM_PROBIT, false, true, ExecType.SPARK);
//	}
//	
//	@Test
//	public void testStepwiseGLMSparseRewritesSP() {
//		runStepwiseTest(StepwiseType.GLM_PROBIT, true, true, ExecType.SPARK);
//	}
//	
//	@Test
//	public void testStepwiseGLMDenseNoRewritesSP() {
//		runStepwiseTest(StepwiseType.GLM_PROBIT, false, false, ExecType.SPARK);
//	}
//	
//	@Test
//	public void testStepwiseGLMSparseNoRewritesSP() {
//		runStepwiseTest(StepwiseType.GLM_PROBIT, true, false, ExecType.SPARK);
//	}
	
	@Test
	public void testStepwiseLinregDSDenseRewritesCP() {
		runStepwiseTest(StepwiseType.LINREG_DS, false, true, ExecType.CP, TestType.DEFAULT);
	}
	
	@Test
	public void testStepwiseLinregDSSparseRewritesCP() {
		runStepwiseTest(StepwiseType.LINREG_DS, true, true, ExecType.CP, TestType.DEFAULT);
	}
	
	@Test
	public void testStepwiseLinregDSDenseNoRewritesCP() {
		runStepwiseTest(StepwiseType.LINREG_DS, false, false, ExecType.CP, TestType.DEFAULT);
	}
	
	@Test
	public void testStepwiseLinregDSSparseNoRewritesCP() {
		runStepwiseTest(StepwiseType.LINREG_DS, true, false, ExecType.CP, TestType.DEFAULT);
	}

	@Test
	public void testStepwiseGLMDenseRewritesCPFuseAll() {
		runStepwiseTest(StepwiseType.GLM_PROBIT, false, true, ExecType.CP, TestType.FUSE_ALL);
	}

	@Test
	public void testStepwiseGLMSparseRewritesCPFuseAll() {
		runStepwiseTest(StepwiseType.GLM_PROBIT, true, true, ExecType.CP, TestType.FUSE_ALL);
	}

	@Test
	public void testStepwiseLinregDSDenseRewritesCPFuseAll() {
		runStepwiseTest(StepwiseType.LINREG_DS, false, true, ExecType.CP, TestType.FUSE_ALL);
	}

	@Test
	public void testStepwiseLinregDSSparseRewritesCPFuseAll() {
		runStepwiseTest(StepwiseType.LINREG_DS, true, true, ExecType.CP, TestType.FUSE_ALL);
	}

	@Test
	public void testStepwiseGLMDenseRewritesCPFuseNoRedundancy() {
		runStepwiseTest(StepwiseType.GLM_PROBIT, false, true, ExecType.CP, TestType.FUSE_NO_REDUNDANCY);
	}

	@Test
	public void testStepwiseGLMSparseRewritesCPFuseNoRedundancy() {
		runStepwiseTest(StepwiseType.GLM_PROBIT, true, true, ExecType.CP, TestType.FUSE_NO_REDUNDANCY);
	}

	@Test
	public void testStepwiseLinregDSDenseRewritesCPFuseNoRedundancy() {
		runStepwiseTest(StepwiseType.LINREG_DS, false, true, ExecType.CP, TestType.FUSE_NO_REDUNDANCY);
	}

	@Test
	public void testStepwiseLinregDSSparseRewritesCPFuseNoRedundancy() {
		runStepwiseTest(StepwiseType.LINREG_DS, true, true, ExecType.CP, TestType.FUSE_NO_REDUNDANCY);
	}

	
	private void runStepwiseTest( StepwiseType type, boolean sparse, boolean rewrites, ExecType instType, TestType testType)
	{
		boolean oldFlag = OptimizerUtils.ALLOW_ALGEBRAIC_SIMPLIFICATION;
		ExecMode platformOld = rtplatform;
		switch( instType ){
			case SPARK: rtplatform = ExecMode.SPARK; break;
			default: rtplatform = ExecMode.HYBRID; break;
		}
		currentTestType = testType;
		boolean sparkConfigOld = DMLScript.USE_LOCAL_SPARK_CONFIG;
		if( rtplatform == ExecMode.SPARK || rtplatform == ExecMode.HYBRID )
			DMLScript.USE_LOCAL_SPARK_CONFIG = true;

		try
		{
			String TEST_NAME = TEST_NAME1;
			TestConfiguration config = getTestConfiguration(TEST_NAME);
			loadTestConfiguration(config);
			
			if( type ==  StepwiseType.LINREG_DS) {
				fullDMLScriptName = "scripts/algorithms/StepLinearRegDS.dml";
				programArgs = new String[]{ "-explain", "-stats", "-nvargs",
					"X="+input("X"), "Y="+input("Y"), "icpt="+String.valueOf(icpt),
					"thr="+String.valueOf(thr), "B="+output("B"), "S="+output("S")};
			}
			else { //GLM binomial probit
				fullDMLScriptName = "scripts/algorithms/StepGLM.dml";
				programArgs = new String[]{ "-explain", "-stats", "-nvargs",
					"X="+input("X"), "Y="+input("Y"), "icpt="+String.valueOf(icpt),
					"thr="+String.valueOf(thr), "link=3", "yneg=0",
					"moi=5", "mii=5", "B="+output("B"), "S="+output("S")};
			}
			
			OptimizerUtils.ALLOW_ALGEBRAIC_SIMPLIFICATION = rewrites;
			
			//generate actual datasets
			double[][] X = getRandomMatrix(rows, cols, 0, 1, sparse?sparsity2:sparsity1, 714);
			writeInputMatrixWithMTD("X", X, true);
			double[][] y = TestUtils.round(getRandomMatrix(rows, 1, 0, 1, 1.0, 136));
			writeInputMatrixWithMTD("Y", y, true);
			
			runTest(true, false, null, -1); 

			Assert.assertTrue(heavyHittersContainsSubString("spoof")
				|| heavyHittersContainsSubString("sp_spoof"));
		}
		finally {
			rtplatform = platformOld;
			DMLScript.USE_LOCAL_SPARK_CONFIG = sparkConfigOld;
			OptimizerUtils.ALLOW_ALGEBRAIC_SIMPLIFICATION = oldFlag;
			OptimizerUtils.ALLOW_AUTO_VECTORIZATION = true;
			OptimizerUtils.ALLOW_OPERATOR_FUSION = true;
		}
	}

	/**
	 * Override default configuration with custom test configuration to ensure
	 * scratch space and local temporary directory locations are also updated.
	 */
	@Override
	protected File getConfigTemplateFile() {
		// Instrumentation in this test's output log to show custom configuration file used for template.
		String message = "This test case overrides default configuration with ";
		if(currentTestType == TestType.FUSE_ALL){
			System.out.println(message + TEST_CONF_FILE_FUSE_ALL.getPath());
			return TEST_CONF_FILE_FUSE_ALL;
		} else if(currentTestType == TestType.FUSE_NO_REDUNDANCY){
			System.out.println(message + TEST_CONF_FILE_FUSE_NO_REDUNDANCY.getPath());
			return TEST_CONF_FILE_FUSE_NO_REDUNDANCY;
		} else {
			System.out.println(message + TEST_CONF_FILE_DEFAULT.getPath());
			return TEST_CONF_FILE_DEFAULT;
		}
	}
}
