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

package org.apache.sysml.test.integration.functions.codegen;

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

public class AlgorithmAutoEncoder extends AutomatedTestBase 
{
	private final static String TEST_NAME1 = "Algorithm_AutoEncoder";
	private final static String TEST_DIR = "functions/codegen/";
	private final static String TEST_CLASS_DIR = TEST_DIR + AlgorithmAutoEncoder.class.getSimpleName() + "/";
	private final static String TEST_CONF = "SystemML-config-codegen.xml";
	private final static File   TEST_CONF_FILE = new File(SCRIPT_DIR + TEST_DIR, TEST_CONF);
	
	private final static int rows = 2468;
	private final static int cols = 784;
	
	private final static double sparsity1 = 0.7; //dense
	private final static double sparsity2 = 0.1; //sparse
	
	private final static int H1 = 500;
	private final static int H2 = 2;
	private final static double epochs = 2; 
	
	@Override
	public void setUp() {
		TestUtils.clearAssertionInformation();
		addTestConfiguration(TEST_NAME1, new TestConfiguration(TEST_CLASS_DIR, TEST_NAME1, new String[] { "w" })); 
	}

	@Test
	public void testAutoEncoder256DenseCP() {
		runGLMTest(256, false, false, ExecType.CP);
	}
	
	@Test
	public void testAutoEncoder256DenseRewritesCP() {
		runGLMTest(256, false, true, ExecType.CP);
	}
	
	@Test
	public void testAutoEncoder256SparseCP() {
		runGLMTest(256, true, false, ExecType.CP);
	}
	
	@Test
	public void testAutoEncoder256SparseRewritesCP() {
		runGLMTest(256, true, true, ExecType.CP);
	}
	
	@Test
	public void testAutoEncoder512DenseCP() {
		runGLMTest(512, false, false, ExecType.CP);
	}
	
	@Test
	public void testAutoEncoder512DenseRewritesCP() {
		runGLMTest(512, false, true, ExecType.CP);
	}
	
	@Test
	public void testAutoEncoder512SparseCP() {
		runGLMTest(512, true, false, ExecType.CP);
	}
	
	@Test
	public void testAutoEncoder512SparseRewritesCP() {
		runGLMTest(512, true, true, ExecType.CP);
	}
	
	//Note: limited cases for SPARK, as lazy evaluation 
	//causes very long execution time for this algorithm
	
	@Test
	public void testAutoEncoder256DenseRewritesSpark() {
		runGLMTest(256, false, true, ExecType.SPARK);
	}
	
	@Test
	public void testAutoEncoder256SparseRewritesSpark() {
		runGLMTest(256, true, true, ExecType.SPARK);
	}
	
	@Test
	public void testAutoEncoder512DenseRewritesSpark() {
		runGLMTest(512, false, true, ExecType.SPARK);
	}
	
	@Test
	public void testAutoEncoder512SparseRewritesSpark() {
		runGLMTest(512, true, true, ExecType.SPARK);
	}
	
	private void runGLMTest(int batchsize, boolean sparse, boolean rewrites, ExecType instType)
	{
		boolean oldFlag = OptimizerUtils.ALLOW_ALGEBRAIC_SIMPLIFICATION;
		RUNTIME_PLATFORM platformOld = rtplatform;
		switch( instType ){
			case MR: rtplatform = RUNTIME_PLATFORM.HADOOP; break;
			case SPARK: rtplatform = RUNTIME_PLATFORM.SPARK; break;
			default: rtplatform = RUNTIME_PLATFORM.HYBRID_SPARK; break;
		}
	
		boolean sparkConfigOld = DMLScript.USE_LOCAL_SPARK_CONFIG;
		if( rtplatform == RUNTIME_PLATFORM.SPARK || rtplatform == RUNTIME_PLATFORM.HYBRID_SPARK )
			DMLScript.USE_LOCAL_SPARK_CONFIG = true;

		try
		{
			String TEST_NAME = TEST_NAME1;
			TestConfiguration config = getTestConfiguration(TEST_NAME);
			loadTestConfiguration(config);
			
			fullDMLScriptName = "scripts/staging/autoencoder-2layer.dml";
			programArgs = new String[]{ "-explain", "-stats", "-nvargs", "X="+input("X"),
				"H1="+H1, "H2="+H2, "EPOCH="+epochs, "BATCH="+batchsize, 
				"W1_out="+output("W1"), "b1_out="+output("b1"),
				"W2_out="+output("W2"), "b2_out="+output("b2"),
				"W3_out="+output("W3"), "b3_out="+output("b3"),
				"W4_out="+output("W4"), "b4_out="+output("b4")};
			OptimizerUtils.ALLOW_ALGEBRAIC_SIMPLIFICATION = rewrites;
			
			//generate actual datasets
			double[][] X = getRandomMatrix(rows, cols, 0, 1, sparse?sparsity2:sparsity1, 714);
			writeInputMatrixWithMTD("X", X, true);
			
			//run script
			runTest(true, false, null, -1); 
			//TODO R script
			
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
