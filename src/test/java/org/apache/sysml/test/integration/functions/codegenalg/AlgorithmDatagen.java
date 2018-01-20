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

package org.apache.sysml.test.integration.functions.codegenalg;

import java.io.File;

import org.junit.Assert;
import org.junit.Test;
import org.apache.sysml.api.DMLScript;
import org.apache.sysml.api.DMLScript.RUNTIME_PLATFORM;
import org.apache.sysml.hops.OptimizerUtils;
import org.apache.sysml.lops.LopProperties.ExecType;
import org.apache.sysml.test.integration.AutomatedTestBase;
import org.apache.sysml.test.integration.TestConfiguration;
import org.apache.sysml.test.utils.TestUtils;

public class AlgorithmDatagen extends AutomatedTestBase 
{
	private final static String TEST_NAME1 = "Algorithm_Datagen";
	private final static String TEST_DIR = "functions/codegenalg/";
	private final static String TEST_CLASS_DIR = TEST_DIR + AlgorithmDatagen.class.getSimpleName() + "/";
	private final static String TEST_CONF = "SystemML-config-codegen.xml";
	private final static File   TEST_CONF_FILE = new File(SCRIPT_DIR + TEST_DIR, TEST_CONF);
	
	private final static int rows = 2468;
	private final static int cols = 200;
	
	private final static double sparsity1 = 0.9; //dense
	private final static double sparsity2 = 0.1; //sparse
	
	public enum DatagenType {
		LINREG,
		LOGREG,
	}
	
	@Override
	public void setUp() {
		TestUtils.clearAssertionInformation();
		addTestConfiguration(TEST_NAME1, new TestConfiguration(TEST_CLASS_DIR, TEST_NAME1, new String[] { "X","Y","w" })); 
	}

	@Test
	public void testDatagenLinregDenseRewritesCP() {
		runStepwiseTest(DatagenType.LINREG, false, true, ExecType.CP);
	}
	
	@Test
	public void testDatagenLinregSparseRewritesCP() {
		runStepwiseTest(DatagenType.LINREG, true, true, ExecType.CP);
	}
	
	@Test
	public void testDatagenLinregDenseNoRewritesCP() {
		runStepwiseTest(DatagenType.LINREG, false, false, ExecType.CP);
	}
	
	@Test
	public void testDatagenLinregSparseNoRewritesCP() {
		runStepwiseTest(DatagenType.LINREG, true, false, ExecType.CP);
	}
	
	@Test
	public void testDatagenLogregDenseRewritesCP() {
		runStepwiseTest(DatagenType.LOGREG, false, true, ExecType.CP);
	}
	
	@Test
	public void testDatagenLogregSparseRewritesCP() {
		runStepwiseTest(DatagenType.LOGREG, true, true, ExecType.CP);
	}
	
	@Test
	public void testDatagenLogregDenseNoRewritesCP() {
		runStepwiseTest(DatagenType.LOGREG, false, false, ExecType.CP);
	}
	
	@Test
	public void testDatagenLogregSparseNoRewritesCP() {
		runStepwiseTest(DatagenType.LOGREG, true, false, ExecType.CP);
	}

	@Test
	public void testDatagenLinregDenseRewritesSP() {
		runStepwiseTest(DatagenType.LINREG, false, true, ExecType.SPARK);
	}
	
	@Test
	public void testDatagenLinregSparseRewritesSP() {
		runStepwiseTest(DatagenType.LINREG, true, true, ExecType.SPARK);
	}
	
	@Test
	public void testDatagenLinregDenseNoRewritesSP() {
		runStepwiseTest(DatagenType.LINREG, false, false, ExecType.SPARK);
	}
	
	@Test
	public void testDatagenLinregSparseNoRewritesSP() {
		runStepwiseTest(DatagenType.LINREG, true, false, ExecType.SPARK);
	}
	
	@Test
	public void testDatagenLogregDenseRewritesSP() {
		runStepwiseTest(DatagenType.LOGREG, false, true, ExecType.SPARK);
	}
	
	@Test
	public void testDatagenLogregSparseRewritesSP() {
		runStepwiseTest(DatagenType.LOGREG, true, true, ExecType.SPARK);
	}
	
	@Test
	public void testDatagenLogregDenseNoRewritesSP() {
		runStepwiseTest(DatagenType.LOGREG, false, false, ExecType.SPARK);
	}
	
	@Test
	public void testDatagenLogregSparseNoRewritesSP() {
		runStepwiseTest(DatagenType.LOGREG, true, false, ExecType.SPARK);
	}
	
	private void runStepwiseTest( DatagenType type, boolean sparse, boolean rewrites, ExecType instType)
	{
		boolean oldFlag = OptimizerUtils.ALLOW_ALGEBRAIC_SIMPLIFICATION;
		RUNTIME_PLATFORM platformOld = rtplatform;
		switch( instType ){
			case SPARK: rtplatform = RUNTIME_PLATFORM.SPARK; break;
			default: rtplatform = RUNTIME_PLATFORM.HYBRID_SPARK; break;
		}
		
		boolean sparkConfigOld = DMLScript.USE_LOCAL_SPARK_CONFIG;
		if( rtplatform == RUNTIME_PLATFORM.SPARK || rtplatform == RUNTIME_PLATFORM.HYBRID_SPARK )
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
		System.out.println("This test case overrides default configuration with " + TEST_CONF_FILE.getPath());
		return TEST_CONF_FILE;
	}
}
