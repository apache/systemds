/*
 * Modifications Copyright 2019 Graz University of Technology
 *
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

public class AlgorithmDatagen extends AutomatedTestBase 
{
	private final static String TEST_NAME1 = "Algorithm_Datagen";
	private final static String TEST_DIR = "functions/codegenalg/";
	private final static String TEST_CLASS_DIR = TEST_DIR + AlgorithmDatagen.class.getSimpleName() + "/";
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
	
	private final static double sparsity1 = 0.9; //dense
	private final static double sparsity2 = 0.1; //sparse

	public enum DatagenType {
		LINREG,
		LOGREG,
	}

	private TestType currentTestType = TestType.DEFAULT;
	
	@Override
	public void setUp() {
		TestUtils.clearAssertionInformation();
		addTestConfiguration(TEST_NAME1, new TestConfiguration(TEST_CLASS_DIR, TEST_NAME1, new String[] { "X","Y","w" })); 
	}

	@Test
	public void testDatagenLinregDenseRewritesCP() {
		runStepwiseTest(DatagenType.LINREG, false, true, ExecType.CP, TestType.DEFAULT);
	}
	
	@Test
	public void testDatagenLinregSparseRewritesCP() {
		runStepwiseTest(DatagenType.LINREG, true, true, ExecType.CP, TestType.DEFAULT);
	}
	
	@Test
	public void testDatagenLinregDenseNoRewritesCP() {
		runStepwiseTest(DatagenType.LINREG, false, false, ExecType.CP, TestType.DEFAULT);
	}
	
	@Test
	public void testDatagenLinregSparseNoRewritesCP() {
		runStepwiseTest(DatagenType.LINREG, true, false, ExecType.CP, TestType.DEFAULT);
	}
	
	@Test
	public void testDatagenLogregDenseRewritesCP() {
		runStepwiseTest(DatagenType.LOGREG, false, true, ExecType.CP, TestType.DEFAULT);
	}
	
	@Test
	public void testDatagenLogregSparseRewritesCP() {
		runStepwiseTest(DatagenType.LOGREG, true, true, ExecType.CP, TestType.DEFAULT);
	}
	
	@Test
	public void testDatagenLogregDenseNoRewritesCP() {
		runStepwiseTest(DatagenType.LOGREG, false, false, ExecType.CP, TestType.DEFAULT);
	}
	
	@Test
	public void testDatagenLogregSparseNoRewritesCP() {
		runStepwiseTest(DatagenType.LOGREG, true, false, ExecType.CP, TestType.DEFAULT);
	}

	@Test
	public void testDatagenLinregDenseRewritesSP() {
		runStepwiseTest(DatagenType.LINREG, false, true, ExecType.SPARK, TestType.DEFAULT);
	}
	
	@Test
	public void testDatagenLinregSparseRewritesSP() {
		runStepwiseTest(DatagenType.LINREG, true, true, ExecType.SPARK, TestType.DEFAULT);
	}
	
	@Test
	public void testDatagenLinregDenseNoRewritesSP() {
		runStepwiseTest(DatagenType.LINREG, false, false, ExecType.SPARK, TestType.DEFAULT);
	}
	
	@Test
	public void testDatagenLinregSparseNoRewritesSP() {
		runStepwiseTest(DatagenType.LINREG, true, false, ExecType.SPARK, TestType.DEFAULT);
	}
	
	@Test
	public void testDatagenLogregDenseRewritesSP() {
		runStepwiseTest(DatagenType.LOGREG, false, true, ExecType.SPARK, TestType.DEFAULT);
	}
	
	@Test
	public void testDatagenLogregSparseRewritesSP() {
		runStepwiseTest(DatagenType.LOGREG, true, true, ExecType.SPARK, TestType.DEFAULT);
	}
	
	@Test
	public void testDatagenLogregDenseNoRewritesSP() {
		runStepwiseTest(DatagenType.LOGREG, false, false, ExecType.SPARK, TestType.DEFAULT);
	}
	
	@Test
	public void testDatagenLogregSparseNoRewritesSP() {
		runStepwiseTest(DatagenType.LOGREG, true, false, ExecType.SPARK, TestType.DEFAULT);
	}

	@Test
	public void testDatagenLinregDenseRewritesCPFuseAll() {
		runStepwiseTest(DatagenType.LINREG, false, true, ExecType.CP, TestType.FUSE_ALL);
	}

	@Test
	public void testDatagenLinregSparseRewritesCPFuseAll() {
		runStepwiseTest(DatagenType.LINREG, true, true, ExecType.CP, TestType.FUSE_ALL);
	}

	@Test
	public void testDatagenLogregDenseRewritesCPFuseAll() {
		runStepwiseTest(DatagenType.LOGREG, false, true, ExecType.CP, TestType.FUSE_ALL);
	}

	@Test
	public void testDatagenLogregSparseRewritesCPFuseAll() {
		runStepwiseTest(DatagenType.LOGREG, true, true, ExecType.CP, TestType.FUSE_ALL);
	}

	@Test
	public void testDatagenLinregDenseRewritesSPFuseAll() {
		runStepwiseTest(DatagenType.LINREG, false, true, ExecType.SPARK, TestType.FUSE_ALL);
	}

	@Test
	public void testDatagenLinregSparseRewritesSPFuseAll() {
		runStepwiseTest(DatagenType.LINREG, true, true, ExecType.SPARK, TestType.FUSE_ALL);
	}

	@Test
	public void testDatagenLogregDenseRewritesSPFuseAll() {
		runStepwiseTest(DatagenType.LOGREG, false, true, ExecType.SPARK, TestType.FUSE_ALL);
	}

	@Test
	public void testDatagenLogregSparseRewritesSPFuseAll() {
		runStepwiseTest(DatagenType.LOGREG, true, true, ExecType.SPARK, TestType.FUSE_ALL);
	}

	@Test
	public void testDatagenLinregDenseRewritesCPFuseNoRedundancy() {
		runStepwiseTest(DatagenType.LINREG, false, true, ExecType.CP, TestType.FUSE_NO_REDUNDANCY);
	}

	@Test
	public void testDatagenLinregSparseRewritesCPFuseNoRedundancy() {
		runStepwiseTest(DatagenType.LINREG, true, true, ExecType.CP, TestType.FUSE_NO_REDUNDANCY);
	}

	@Test
	public void testDatagenLogregDenseRewritesCPFuseNoRedundancy() {
		runStepwiseTest(DatagenType.LOGREG, false, true, ExecType.CP, TestType.FUSE_NO_REDUNDANCY);
	}

	@Test
	public void testDatagenLogregSparseRewritesCPFuseNoRedundancy() {
		runStepwiseTest(DatagenType.LOGREG, true, true, ExecType.CP, TestType.FUSE_NO_REDUNDANCY);
	}

	@Test
	public void testDatagenLinregDenseRewritesSPFuseNoRedundancy() {
		runStepwiseTest(DatagenType.LINREG, false, true, ExecType.SPARK, TestType.FUSE_NO_REDUNDANCY);
	}

	@Test
	public void testDatagenLinregSparseRewritesSPFuseNoRedundancy() {
		runStepwiseTest(DatagenType.LINREG, true, true, ExecType.SPARK, TestType.FUSE_NO_REDUNDANCY);
	}

	@Test
	public void testDatagenLogregDenseRewritesSPFuseNoRedundancy() {
		runStepwiseTest(DatagenType.LOGREG, false, true, ExecType.SPARK, TestType.FUSE_NO_REDUNDANCY);
	}

	@Test
	public void testDatagenLogregSparseRewritesSPFuseNoRedundancy() {
		runStepwiseTest(DatagenType.LOGREG, true, true, ExecType.SPARK, TestType.FUSE_NO_REDUNDANCY);
	}
	
	private void runStepwiseTest( DatagenType type, boolean sparse, boolean rewrites, ExecType instType, TestType testType)
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
		
		OptimizerUtils.ALLOW_ALGEBRAIC_SIMPLIFICATION = rewrites;
		
		try
		{
			String TEST_NAME = TEST_NAME1;
			TestConfiguration config = getTestConfiguration(TEST_NAME);
			loadTestConfiguration(config);
			
			double sparsity = sparse ? sparsity2 : sparsity1;
			
			if( type ==  DatagenType.LINREG) {
				fullDMLScriptName = "scripts/datagen/genRandData4LinearRegression.dml";
				programArgs = new String[]{ "-explain", "-stats", "-args",
					String.valueOf(rows), String.valueOf(cols), "10", "1", output("w"),
					output("X"), output("y"), "1", "1", String.valueOf(sparsity), "binary"};
			}
			else { //LOGREG
				fullDMLScriptName = "scripts/datagen/genRandData4LogisticRegression.dml";
				programArgs = new String[]{ "-explain", "-stats", "-args",
					String.valueOf(rows), String.valueOf(cols), "10", "1", output("w"),
					output("X"), output("y"), "1", "1", String.valueOf(sparsity), "binary", "1"};
			}
			
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
