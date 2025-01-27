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
import java.util.HashMap;

import org.apache.sysds.common.Opcodes;
import org.apache.sysds.runtime.matrix.data.MatrixValue;
import org.junit.Assert;
import org.junit.Test;
import org.apache.sysds.api.DMLScript;
import org.apache.sysds.common.Types.ExecMode;
import org.apache.sysds.hops.OptimizerUtils;
import org.apache.sysds.common.Types.ExecType;
import org.apache.sysds.test.AutomatedTestBase;
import org.apache.sysds.test.TestConfiguration;
import org.apache.sysds.test.TestUtils;

public class AlgorithmAutoEncoder extends AutomatedTestBase 
{
	private final static String TEST_NAME1 = "Algorithm_AutoEncoder";
	private final static String TEST_DIR = "functions/codegenalg/";
	private final static String TEST_CLASS_DIR = TEST_DIR + AlgorithmAutoEncoder.class.getSimpleName() + "/";
	
	private final static int rows = 1068;
	private final static int cols = 784;
	
	private final static double sparsity1 = 0.7; //dense
	private final static double sparsity2 = 0.1; //sparse
	private final static double eps       = 4e-4;
	
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
			
			fullDMLScriptName = SCRIPT_DIR + TEST_DIR + "/Algorithm_AutoEncoder.dml";
			//"scripts/staging/autoencoder-2layer.dml";
			programArgs = new String[]{ "-stats", "-nvargs", "X="+input("X"),
				"H1="+H1, "H2="+H2, "EPOCH="+epochs, "BATCH="+batchsize,
				"W1_rand="+input("W1_rand"),"W2_rand="+input("W2_rand"),
				"W3_rand="+input("W3_rand"), "W4_rand="+input("W4_rand"),
				"order_rand="+input("order_rand"),
				"W1_out="+output("W1"), "b1_out="+output("b1"),
				"W2_out="+output("W2"), "b2_out="+output("b2"),
				"W3_out="+output("W3"), "b3_out="+output("b3"),
				"W4_out="+output("W4"), "b4_out="+output("b4")};

			rCmd = getRCmd(inputDir(), String.valueOf(H1), String.valueOf(H2),
					String.valueOf(epochs), String.valueOf(batchsize), expectedDir());
			OptimizerUtils.ALLOW_ALGEBRAIC_SIMPLIFICATION = rewrites;
			
			//generate actual datasets
			double[][] X = getRandomMatrix(rows, cols, 0, 1, sparse?sparsity2:sparsity1, 714);
			writeInputMatrixWithMTD("X", X, true);

			//generate rand matrices for W1, W2, W3, W4 here itself for passing onto both DML and R scripts
			double[][] W1_rand = getRandomMatrix(H1, cols, 0, 1, sparse?sparsity2:sparsity1, 800);
			writeInputMatrixWithMTD("W1_rand", W1_rand, true);
			double[][] W2_rand = getRandomMatrix(H2, H1, 0, 1, sparse?sparsity2:sparsity1, 900);
			writeInputMatrixWithMTD("W2_rand", W2_rand, true);
			double[][] W3_rand = getRandomMatrix(H1, H2, 0, 1, sparse?sparsity2:sparsity1, 589);
			writeInputMatrixWithMTD("W3_rand", W3_rand, true);
			double[][] W4_rand = getRandomMatrix(cols, H1, 0, 1, sparse?sparsity2:sparsity1, 150);
			writeInputMatrixWithMTD("W4_rand", W4_rand, true);
			double[][] order_rand = getRandomMatrix(rows, 1, 0, 1, sparse?sparsity2:sparsity1, 143);
			writeInputMatrixWithMTD("order_rand", order_rand, true); //for the permut operation on input X

			//run script
			runTest(true, false, null, -1); 
			runRScript(true);

			HashMap<MatrixValue.CellIndex, Double> dmlW1 = readDMLMatrixFromOutputDir("W1");
			HashMap<MatrixValue.CellIndex, Double> dmlW2 = readDMLMatrixFromOutputDir("W2");
			HashMap<MatrixValue.CellIndex, Double> dmlW3 = readDMLMatrixFromOutputDir("W3");
			HashMap<MatrixValue.CellIndex, Double> dmlW4 = readDMLMatrixFromOutputDir("W4");
			HashMap<MatrixValue.CellIndex, Double> dmlb1 = readDMLMatrixFromOutputDir("b1");
			HashMap<MatrixValue.CellIndex, Double> dmlb2 = readDMLMatrixFromOutputDir("b2");
			HashMap<MatrixValue.CellIndex, Double> dmlb3 = readDMLMatrixFromOutputDir("b3");
			HashMap<MatrixValue.CellIndex, Double> dmlb4 = readDMLMatrixFromOutputDir("b4");
			HashMap<MatrixValue.CellIndex, Double> rW1  = readRMatrixFromExpectedDir("W1");
			HashMap<MatrixValue.CellIndex, Double> rW2  = readRMatrixFromExpectedDir("W2");
			HashMap<MatrixValue.CellIndex, Double> rW3  = readRMatrixFromExpectedDir("W3");
			HashMap<MatrixValue.CellIndex, Double> rW4  = readRMatrixFromExpectedDir("W4");
			HashMap<MatrixValue.CellIndex, Double> rb1  = readRMatrixFromExpectedDir("b1");
			HashMap<MatrixValue.CellIndex, Double> rb2  = readRMatrixFromExpectedDir("b2");
			HashMap<MatrixValue.CellIndex, Double> rb3  = readRMatrixFromExpectedDir("b3");
			HashMap<MatrixValue.CellIndex, Double> rb4  = readRMatrixFromExpectedDir("b4");
			TestUtils.compareMatrices(dmlW1, rW1, eps, "Stat-DML", "Stat-R");
			TestUtils.compareMatrices(dmlW2, rW2, eps, "Stat-DML", "Stat-R");
			TestUtils.compareMatrices(dmlW3, rW3, eps, "Stat-DML", "Stat-R");
			TestUtils.compareMatrices(dmlW4, rW4, eps, "Stat-DML", "Stat-R");
			TestUtils.compareMatrices(dmlb1, rb1, eps, "Stat-DML", "Stat-R");
			TestUtils.compareMatrices(dmlb2, rb2, eps, "Stat-DML", "Stat-R");
			TestUtils.compareMatrices(dmlb3, rb3, eps, "Stat-DML", "Stat-R");
			TestUtils.compareMatrices(dmlb4, rb4, eps, "Stat-DML", "Stat-R");
			
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
