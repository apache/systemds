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
 
package org.apache.sysds.test.functions.codegenalg.partone;

import java.io.File;

import org.junit.Assert;
import org.junit.Test;
import org.apache.sysds.api.DMLScript;
import org.apache.sysds.common.Types.ExecMode;
import org.apache.sysds.hops.OptimizerUtils;
import org.apache.sysds.lops.LopProperties.ExecType;
import org.apache.sysds.test.AutomatedTestBase;
import org.apache.sysds.test.TestConfiguration;
import org.apache.sysds.test.TestUtils;

public class AlgorithmAutoEncoder extends AutomatedTestBase 
{
	private final static String TEST_NAME1 = "Algorithm_AutoEncoder";
	private final static String TEST_DIR = "functions/codegenalg/";
	private final static String TEST_CLASS_DIR = TEST_DIR + AlgorithmAutoEncoder.class.getSimpleName() + "/";
	
	private final static int rows = 2468;
	private final static int cols = 784;
	
	private final static double sparsity1 = 0.7; //dense
	private final static double sparsity2 = 0.1; //sparse
	
	private final static int H1 = 500;
	private final static int H2 = 2;
	private final static double epochs = 2; 
	
	private CodegenTestType currentTestType = CodegenTestType.DEFAULT;
	
	@Override
	public void setUp() {
		TestUtils.clearAssertionInformation();
		addTestConfiguration(TEST_NAME1, new TestConfiguration(TEST_CLASS_DIR, TEST_NAME1, new String[] { "w" })); 
	}

	//Note: limited cases for SPARK, as lazy evaluation 
	//causes very long execution time for this algorithm

	@Test
	public void testAutoEncoder256DenseCP() {
		runAutoEncoderTest(256, false, false, ExecType.CP, CodegenTestType.DEFAULT);
	}
	
	@Test
	public void testAutoEncoder256DenseRewritesCP() {
		runAutoEncoderTest(256, false, true, ExecType.CP, CodegenTestType.DEFAULT);
	}
	
	@Test
	public void testAutoEncoder256SparseCP() {
		runAutoEncoderTest(256, true, false, ExecType.CP, CodegenTestType.DEFAULT);
	}
	
	@Test
	public void testAutoEncoder256SparseRewritesCP() {
		runAutoEncoderTest(256, true, true, ExecType.CP, CodegenTestType.DEFAULT);
	}
	
	@Test
	public void testAutoEncoder512DenseCP() {
		runAutoEncoderTest(512, false, false, ExecType.CP, CodegenTestType.DEFAULT);
	}
	
	@Test
	public void testAutoEncoder512DenseRewritesCP() {
		runAutoEncoderTest(512, false, true, ExecType.CP, CodegenTestType.DEFAULT);
	}
	
	@Test
	public void testAutoEncoder512SparseCP() {
		runAutoEncoderTest(512, true, false, ExecType.CP, CodegenTestType.DEFAULT);
	}
	
	@Test
	public void testAutoEncoder512SparseRewritesCP() {
		runAutoEncoderTest(512, true, true, ExecType.CP, CodegenTestType.DEFAULT);
	}
	
	@Test
	public void testAutoEncoder256DenseRewritesSpark() {
		runAutoEncoderTest(256, false, true, ExecType.SPARK, CodegenTestType.DEFAULT);
	}
	
	@Test
	public void testAutoEncoder256SparseRewritesSpark() {
		runAutoEncoderTest(256, true, true, ExecType.SPARK, CodegenTestType.DEFAULT);
	}
	
	@Test
	public void testAutoEncoder512DenseRewritesSpark() {
		runAutoEncoderTest(512, false, true, ExecType.SPARK, CodegenTestType.DEFAULT);
	}
	
	@Test
	public void testAutoEncoder512SparseRewritesSpark() {
		runAutoEncoderTest(512, true, true, ExecType.SPARK, CodegenTestType.DEFAULT);
	}

	@Test
	public void testAutoEncoder512DenseRewritesCPFuseAll() {
		runAutoEncoderTest(512, false, true, ExecType.CP, CodegenTestType.FUSE_ALL);
	}

	@Test
	public void testAutoEncoder512SparseRewritesCPFuseAll() {
		runAutoEncoderTest(512, true, true, ExecType.CP, CodegenTestType.FUSE_ALL);
	}

	@Test
	public void testAutoEncoder512DenseRewritesSparkFuseAll() {
		runAutoEncoderTest(512, false, true, ExecType.SPARK, CodegenTestType.FUSE_ALL);
	}

	@Test
	public void testAutoEncoder512SparseRewritesSparkFuseAll() {
		runAutoEncoderTest(512, true, true, ExecType.SPARK, CodegenTestType.FUSE_ALL);
	}

	@Test
	public void testAutoEncoder512DenseRewritesCPFuseNoRedundancy() {
		runAutoEncoderTest(512, false, true, ExecType.CP, CodegenTestType.FUSE_NO_REDUNDANCY);
	}

	@Test
	public void testAutoEncoder512SparseRewritesCPFuseNoRedundancy() {
		runAutoEncoderTest(512, true, true, ExecType.CP, CodegenTestType.FUSE_NO_REDUNDANCY);
	}

	@Test
	public void testAutoEncoder512DenseRewritesSparkFuseNoRedundancy() {
		runAutoEncoderTest(512, false, true, ExecType.SPARK, CodegenTestType.FUSE_NO_REDUNDANCY);
	}

	@Test
	public void testAutoEncoder512SparseRewritesSparkFuseNoRedundancy() {
		runAutoEncoderTest(512, true, true, ExecType.SPARK, CodegenTestType.FUSE_NO_REDUNDANCY);
	}

	private void runAutoEncoderTest(int batchsize, boolean sparse, boolean rewrites, ExecType instType, CodegenTestType CodegenTestType)
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
			String TEST_NAME = TEST_NAME1;
			TestConfiguration config = getTestConfiguration(TEST_NAME);
			loadTestConfiguration(config);
			
			fullDMLScriptName = "scripts/staging/autoencoder-2layer.dml";
			programArgs = new String[]{ "-stats", "-nvargs", "X="+input("X"),
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
		return getCodegenConfigFile(SCRIPT_DIR + TEST_DIR, currentTestType);
	}
}
