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

public class AlgorithmAutoEncoder extends AutomatedTestBase 
{
	private final static String TEST_NAME1 = "Algorithm_AutoEncoder";
	private final static String TEST_DIR = "functions/codegenalg/";
	private final static String TEST_CLASS_DIR = TEST_DIR + AlgorithmAutoEncoder.class.getSimpleName() + "/";
	private final static String TEST_CONF_DEFAULT = "SystemDS-config-codegen.xml";
	private final static File TEST_CONF_FILE_DEFAULT = new File(SCRIPT_DIR + TEST_DIR, TEST_CONF_DEFAULT);
	private final static String TEST_CONF_FUSE_ALL = "SystemDS-config-codegen-fuse-all.xml";
	private final static File TEST_CONF_FILE_FUSE_ALL = new File(SCRIPT_DIR + TEST_DIR, TEST_CONF_FUSE_ALL);
	private final static String TEST_CONF_FUSE_NO_REDUNDANCY = "SystemDS-config-codegen-fuse-no-redundancy.xml";
	private final static File TEST_CONF_FILE_FUSE_NO_REDUNDANCY = new File(SCRIPT_DIR + TEST_DIR,
			TEST_CONF_FUSE_NO_REDUNDANCY);

	private enum TestType { DEFAULT,FUSE_ALL,FUSE_NO_REDUNDANCY }
	
	private final static int rows = 2468;
	private final static int cols = 784;
	
	private final static double sparsity1 = 0.7; //dense
	private final static double sparsity2 = 0.1; //sparse
	
	private final static int H1 = 500;
	private final static int H2 = 2;
	private final static double epochs = 2; 
	
	private TestType currentTestType = TestType.DEFAULT;
	
	@Override
	public void setUp() {
		TestUtils.clearAssertionInformation();
		addTestConfiguration(TEST_NAME1, new TestConfiguration(TEST_CLASS_DIR, TEST_NAME1, new String[] { "w" })); 
	}

	//Note: limited cases for SPARK, as lazy evaluation 
	//causes very long execution time for this algorithm

	@Test
	public void testAutoEncoder256DenseCP() {
		runAutoEncoderTest(256, false, false, ExecType.CP, TestType.DEFAULT);
	}
	
	@Test
	public void testAutoEncoder256DenseRewritesCP() {
		runAutoEncoderTest(256, false, true, ExecType.CP, TestType.DEFAULT);
	}
	
	@Test
	public void testAutoEncoder256SparseCP() {
		runAutoEncoderTest(256, true, false, ExecType.CP, TestType.DEFAULT);
	}
	
	@Test
	public void testAutoEncoder256SparseRewritesCP() {
		runAutoEncoderTest(256, true, true, ExecType.CP, TestType.DEFAULT);
	}
	
	@Test
	public void testAutoEncoder512DenseCP() {
		runAutoEncoderTest(512, false, false, ExecType.CP, TestType.DEFAULT);
	}
	
	@Test
	public void testAutoEncoder512DenseRewritesCP() {
		runAutoEncoderTest(512, false, true, ExecType.CP, TestType.DEFAULT);
	}
	
	@Test
	public void testAutoEncoder512SparseCP() {
		runAutoEncoderTest(512, true, false, ExecType.CP, TestType.DEFAULT);
	}
	
	@Test
	public void testAutoEncoder512SparseRewritesCP() {
		runAutoEncoderTest(512, true, true, ExecType.CP, TestType.DEFAULT);
	}
	
	@Test
	public void testAutoEncoder256DenseRewritesSpark() {
		runAutoEncoderTest(256, false, true, ExecType.SPARK, TestType.DEFAULT);
	}
	
	@Test
	public void testAutoEncoder256SparseRewritesSpark() {
		runAutoEncoderTest(256, true, true, ExecType.SPARK, TestType.DEFAULT);
	}
	
	@Test
	public void testAutoEncoder512DenseRewritesSpark() {
		runAutoEncoderTest(512, false, true, ExecType.SPARK, TestType.DEFAULT);
	}
	
	@Test
	public void testAutoEncoder512SparseRewritesSpark() {
		runAutoEncoderTest(512, true, true, ExecType.SPARK, TestType.DEFAULT);
	}

	@Test
	public void testAutoEncoder512DenseRewritesCPFuseAll() {
		runAutoEncoderTest(512, false, true, ExecType.CP, TestType.FUSE_ALL);
	}

	@Test
	public void testAutoEncoder512SparseRewritesCPFuseAll() {
		runAutoEncoderTest(512, true, true, ExecType.CP, TestType.FUSE_ALL);
	}

	@Test
	public void testAutoEncoder512DenseRewritesSparkFuseAll() {
		runAutoEncoderTest(512, false, true, ExecType.SPARK, TestType.FUSE_ALL);
	}

	@Test
	public void testAutoEncoder512SparseRewritesSparkFuseAll() {
		runAutoEncoderTest(512, true, true, ExecType.SPARK, TestType.FUSE_ALL);
	}

	@Test
	public void testAutoEncoder512DenseRewritesCPFuseNoRedundancy() {
		runAutoEncoderTest(512, false, true, ExecType.CP, TestType.FUSE_NO_REDUNDANCY);
	}

	@Test
	public void testAutoEncoder512SparseRewritesCPFuseNoRedundancy() {
		runAutoEncoderTest(512, true, true, ExecType.CP, TestType.FUSE_NO_REDUNDANCY);
	}

	@Test
	public void testAutoEncoder512DenseRewritesSparkFuseNoRedundancy() {
		runAutoEncoderTest(512, false, true, ExecType.SPARK, TestType.FUSE_NO_REDUNDANCY);
	}

	@Test
	public void testAutoEncoder512SparseRewritesSparkFuseNoRedundancy() {
		runAutoEncoderTest(512, true, true, ExecType.SPARK, TestType.FUSE_NO_REDUNDANCY);
	}

	private void runAutoEncoderTest(int batchsize, boolean sparse, boolean rewrites, ExecType instType, TestType testType)
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
