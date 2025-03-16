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

package org.apache.sysds.test.functions.codegenalg.parttwo;

import java.io.File;

import org.apache.sysds.api.DMLScript;
import org.apache.sysds.common.Opcodes;
import org.apache.sysds.common.Types.ExecMode;
import org.apache.sysds.hops.OptimizerUtils;
import org.apache.sysds.common.Types.ExecType;
import org.apache.sysds.test.AutomatedTestBase;
import org.apache.sysds.test.TestConfiguration;
import org.apache.sysds.test.TestUtils;
import org.junit.Assert;
import org.junit.Ignore;
import org.junit.Test;

public class AlgorithmStepwiseRegression extends AutomatedTestBase 
{
	private final static String TEST_NAME1 = "Algorithm_StepLM";
	private final static String TEST_NAME2 = "Algorithm_StepGLM";
	private final static String TEST_DIR = "functions/codegenalg/";
	private final static String TEST_CLASS_DIR = TEST_DIR + AlgorithmStepwiseRegression.class.getSimpleName() + "/";

	private final static int rows = 1468;
	private final static int cols = 200;
	
	private final static double sparsity1 = 0.7; //dense
	private final static double sparsity2 = 0.1; //sparse
	
	private final static int icpt = 0;
	private final static double thr = 0.01;
	
	public enum StepwiseType {
		GLM_PROBIT,
		LINREG_DS,
	}
	
	private CodegenTestType currentTestType = CodegenTestType.DEFAULT;
	
	@Override
	public void setUp() {
		TestUtils.clearAssertionInformation();
		addTestConfiguration(TEST_NAME1, new TestConfiguration(TEST_CLASS_DIR, TEST_NAME1, new String[] { "w" })); 
		addTestConfiguration(TEST_NAME2, new TestConfiguration(TEST_CLASS_DIR, TEST_NAME2, new String[] { "w" })); 
	}

	@Test
	@Ignore
	public void testStepwiseGLMDenseRewritesCP() {
		runStepwiseTest(StepwiseType.GLM_PROBIT, false, true, ExecType.CP, CodegenTestType.DEFAULT);
	}
	
	@Test
	@Ignore
	public void testStepwiseGLMSparseRewritesCP() {
		runStepwiseTest(StepwiseType.GLM_PROBIT, true, true, ExecType.CP, CodegenTestType.DEFAULT);
	}
	
	@Test
	@Ignore
	public void testStepwiseGLMDenseNoRewritesCP() {
		runStepwiseTest(StepwiseType.GLM_PROBIT, false, false, ExecType.CP, CodegenTestType.DEFAULT);
	}
	
	@Test
	@Ignore
	public void testStepwiseGLMSparseNoRewritesCP() {
		runStepwiseTest(StepwiseType.GLM_PROBIT, true, false, ExecType.CP, CodegenTestType.DEFAULT);
	}
	
	@Test
	@Ignore
	public void testStepwiseGLMDenseRewritesSP() {
		runStepwiseTest(StepwiseType.GLM_PROBIT, false, true, ExecType.SPARK, CodegenTestType.DEFAULT);
	}
	
	@Test
	@Ignore
	public void testStepwiseGLMSparseRewritesSP() {
		runStepwiseTest(StepwiseType.GLM_PROBIT, true, true, ExecType.SPARK, CodegenTestType.DEFAULT);
	}
	
	@Test
	@Ignore
	public void testStepwiseGLMDenseNoRewritesSP() {
		runStepwiseTest(StepwiseType.GLM_PROBIT, false, false, ExecType.SPARK, CodegenTestType.DEFAULT);
	}
	
	@Test
	@Ignore
	public void testStepwiseGLMSparseNoRewritesSP() {
		runStepwiseTest(StepwiseType.GLM_PROBIT, true, false, ExecType.SPARK, CodegenTestType.DEFAULT);
	}
	
	@Test
	public void testStepwiseLinregDSDenseRewritesCP() {
		runStepwiseTest(StepwiseType.LINREG_DS, false, true, ExecType.CP, CodegenTestType.DEFAULT);
	}
	
	@Test
	public void testStepwiseLinregDSSparseRewritesCP() {
		runStepwiseTest(StepwiseType.LINREG_DS, true, true, ExecType.CP, CodegenTestType.DEFAULT);
	}
	
	@Test
	public void testStepwiseLinregDSDenseNoRewritesCP() {
		runStepwiseTest(StepwiseType.LINREG_DS, false, false, ExecType.CP, CodegenTestType.DEFAULT);
	}
	
	@Test
	public void testStepwiseLinregDSSparseNoRewritesCP() {
		runStepwiseTest(StepwiseType.LINREG_DS, true, false, ExecType.CP, CodegenTestType.DEFAULT);
	}

	@Test
	@Ignore
	public void testStepwiseGLMDenseRewritesCPFuseAll() {
		runStepwiseTest(StepwiseType.GLM_PROBIT, false, true, ExecType.CP, CodegenTestType.FUSE_ALL);
	}

	@Test
	@Ignore
	public void testStepwiseGLMSparseRewritesCPFuseAll() {
		runStepwiseTest(StepwiseType.GLM_PROBIT, true, true, ExecType.CP, CodegenTestType.FUSE_ALL);
	}

	@Test
	public void testStepwiseLinregDSDenseRewritesCPFuseAll() {
		runStepwiseTest(StepwiseType.LINREG_DS, false, true, ExecType.CP, CodegenTestType.FUSE_ALL);
	}

	@Test
	public void testStepwiseLinregDSSparseRewritesCPFuseAll() {
		runStepwiseTest(StepwiseType.LINREG_DS, true, true, ExecType.CP, CodegenTestType.FUSE_ALL);
	}

	@Test
	@Ignore
	public void testStepwiseGLMDenseRewritesCPFuseNoRedundancy() {
		runStepwiseTest(StepwiseType.GLM_PROBIT, false, true, ExecType.CP, CodegenTestType.FUSE_NO_REDUNDANCY);
	}

	@Test
	@Ignore
	public void testStepwiseGLMSparseRewritesCPFuseNoRedundancy() {
		runStepwiseTest(StepwiseType.GLM_PROBIT, true, true, ExecType.CP, CodegenTestType.FUSE_NO_REDUNDANCY);
	}

	@Test
	public void testStepwiseLinregDSDenseRewritesCPFuseNoRedundancy() {
		runStepwiseTest(StepwiseType.LINREG_DS, false, true, ExecType.CP, CodegenTestType.FUSE_NO_REDUNDANCY);
	}

	@Test
	public void testStepwiseLinregDSSparseRewritesCPFuseNoRedundancy() {
		runStepwiseTest(StepwiseType.LINREG_DS, true, true, ExecType.CP, CodegenTestType.FUSE_NO_REDUNDANCY);
	}

	
	private void runStepwiseTest( StepwiseType type, boolean sparse, boolean rewrites, ExecType instType, CodegenTestType CodegenTestType)
	{
		boolean oldFlag = OptimizerUtils.ALLOW_ALGEBRAIC_SIMPLIFICATION;
		ExecMode platformOld = rtplatform;
		switch( instType ){
			case SPARK: rtplatform = ExecMode.SPARK; break;
			default: rtplatform = ExecMode.HYBRID; break;
		}
		currentTestType = CodegenTestType;
		boolean sparkConfigOld = DMLScript.USE_LOCAL_SPARK_CONFIG;
		if( rtplatform == ExecMode.SPARK || rtplatform == ExecMode.HYBRID )
			DMLScript.USE_LOCAL_SPARK_CONFIG = true;

		try
		{
			String TEST_NAME = (type==StepwiseType.LINREG_DS) ? TEST_NAME1 : TEST_NAME2;
			TestConfiguration config = getTestConfiguration(TEST_NAME);
			loadTestConfiguration(config);
			String HOME = SCRIPT_DIR + TEST_DIR;
			fullDMLScriptName = HOME + TEST_NAME + ".dml";
			
			if( type ==  StepwiseType.LINREG_DS) {
				programArgs = new String[]{ "-stats", "-nvargs",
					"X="+input("X"), "Y="+input("Y"), "icpt="+String.valueOf(icpt),
					"thr="+String.valueOf(thr), "B="+output("B"), "S="+output("S")};
			}
			else { //GLM binomial probit
				programArgs = new String[]{ "-stats", "-nvargs",
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

			Assert.assertTrue(heavyHittersContainsSubString(Opcodes.SPOOF.toString())
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
		return getCodegenConfigFile(SCRIPT_DIR + TEST_DIR, currentTestType);
	}
}
