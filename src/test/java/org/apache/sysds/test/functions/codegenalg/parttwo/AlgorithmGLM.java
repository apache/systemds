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
import org.junit.Test;

public class AlgorithmGLM extends AutomatedTestBase 
{	
	private final static String TEST_NAME1 = "Algorithm_GLM";
	private final static String TEST_DIR = "functions/codegenalg/";
	private final static String TEST_CLASS_DIR = TEST_DIR + AlgorithmGLM.class.getSimpleName() + "/";

	private final static int rows = 1468;
	private final static int cols = 1007;
		
	private final static double sparsity1 = 0.7; //dense
	private final static double sparsity2 = 0.1; //sparse
	
	private final static int intercept = 0;
	private final static double epsilon = 0.000000001;
	private final static double maxiter = 5; //inner/outer

	public enum GLMType {
		POISSON_LOG,
		GAMMA_LOG,
		BINOMIAL_PROBIT,
	}
	
	private CodegenTestType currentTestType = CodegenTestType.DEFAULT;
	
	@Override
	public void setUp() {
		TestUtils.clearAssertionInformation();
		addTestConfiguration(TEST_NAME1, new TestConfiguration(TEST_CLASS_DIR, TEST_NAME1, new String[] { "w" })); 
	}

	@Test
	public void testGLMPoissonDenseRewritesCP() {
		runGLMTest(GLMType.POISSON_LOG, true, false, ExecType.CP, CodegenTestType.DEFAULT);
	}
	
	@Test
	public void testGLMPoissonSparseRewritesCP() {
		runGLMTest(GLMType.POISSON_LOG, true, true, ExecType.CP, CodegenTestType.DEFAULT);
	}
	
	@Test
	public void testGLMPoissonDenseCP() {
		runGLMTest(GLMType.POISSON_LOG, false, false, ExecType.CP, CodegenTestType.DEFAULT);
	}
	
	@Test
	public void testGLMPoissonSparseCP() {
		runGLMTest(GLMType.POISSON_LOG, false, true, ExecType.CP, CodegenTestType.DEFAULT);
	}

//TODO debugging GLM builtin GAMMA
//	@Test
//	public void testGLMGammaDenseRewritesCP() {
//		runGLMTest(GLMType.GAMMA_LOG, true, false, ExecType.CP, CodegenTestType.DEFAULT);
//	}
//	
//	@Test
//	public void testGLMGammaSparseRewritesCP() {
//		runGLMTest(GLMType.GAMMA_LOG, true, true, ExecType.CP, CodegenTestType.DEFAULT);
//	}
//	
//	@Test
//	public void testGLMGammaDenseCP() {
//		runGLMTest(GLMType.GAMMA_LOG, false, false, ExecType.CP, CodegenTestType.DEFAULT);
//	}
//	
//	@Test
//	public void testGLMGammaSparseCP() {
//		runGLMTest(GLMType.GAMMA_LOG, false, true, ExecType.CP, CodegenTestType.DEFAULT);
//	}
	
	@Test
	public void testGLMBinomialDenseRewritesCP() {
		runGLMTest(GLMType.BINOMIAL_PROBIT, true, false, ExecType.CP, CodegenTestType.DEFAULT);
	}
	
	@Test
	public void testGLMBinomialSparseRewritesCP() {
		runGLMTest(GLMType.BINOMIAL_PROBIT, true, true, ExecType.CP, CodegenTestType.DEFAULT);
	}
	
	@Test
	public void testGLMBinomialDenseCP() {
		runGLMTest(GLMType.BINOMIAL_PROBIT, false, false, ExecType.CP, CodegenTestType.DEFAULT);
	}
	
	@Test
	public void testGLMBinomialSparseCP() {
		runGLMTest(GLMType.BINOMIAL_PROBIT, false, true, ExecType.CP, CodegenTestType.DEFAULT);
	}
	
	@Test
	public void testGLMPoissonDenseRewritesSP() {
		runGLMTest(GLMType.POISSON_LOG, true, false, ExecType.SPARK, CodegenTestType.DEFAULT);
	}
	
	@Test
	public void testGLMPoissonSparseRewritesSP() {
		runGLMTest(GLMType.POISSON_LOG, true, true, ExecType.SPARK, CodegenTestType.DEFAULT);
	}
	
//	@Test
//	public void testGLMGammaDenseRewritesSP() {
//		runGLMTest(GLMType.GAMMA_LOG, true, false, ExecType.SPARK, CodegenTestType.DEFAULT);
//	}
//	
//	@Test
//	public void testGLMGammaSparseRewritesSP() {
//		runGLMTest(GLMType.GAMMA_LOG, true, true, ExecType.SPARK, CodegenTestType.DEFAULT);
//	}
//	
	@Test
	public void testGLMBinomialDenseRewritesSP() {
		runGLMTest(GLMType.BINOMIAL_PROBIT, true, false, ExecType.SPARK, CodegenTestType.DEFAULT);
	}
	
	@Test
	public void testGLMBinomialSparseRewritesSP() {
		runGLMTest(GLMType.BINOMIAL_PROBIT, true, true, ExecType.SPARK, CodegenTestType.DEFAULT);
	}

	@Test
	public void testGLMPoissonDenseRewritesCPFuseAll() {
		runGLMTest(GLMType.POISSON_LOG, true, false, ExecType.CP, CodegenTestType.FUSE_ALL);
	}

	@Test
	public void testGLMPoissonSparseRewritesCPFuseAll() {
		runGLMTest(GLMType.POISSON_LOG, true, true, ExecType.CP, CodegenTestType.FUSE_ALL);
	}

//	@Test
//	public void testGLMGammaDenseRewritesCPFuseAll() {
//		runGLMTest(GLMType.GAMMA_LOG, true, false, ExecType.CP, CodegenTestType.FUSE_ALL);
//	}
//
//	@Test
//	public void testGLMGammaSparseRewritesCPFuseAll() {
//		runGLMTest(GLMType.GAMMA_LOG, true, true, ExecType.CP, CodegenTestType.FUSE_ALL);
//	}

	@Test
	public void testGLMBinomialDenseRewritesCPFuseAll() {
		runGLMTest(GLMType.BINOMIAL_PROBIT, true, false, ExecType.CP, CodegenTestType.FUSE_ALL);
	}

	@Test
	public void testGLMBinomialSparseRewritesCPFuseAll() {
		runGLMTest(GLMType.BINOMIAL_PROBIT, true, true, ExecType.CP, CodegenTestType.FUSE_ALL);
	}

	@Test
	public void testGLMPoissonDenseRewritesSPFuseAll() {
		runGLMTest(GLMType.POISSON_LOG, true, false, ExecType.SPARK, CodegenTestType.FUSE_ALL);
	}

	@Test
	public void testGLMPoissonSparseRewritesSPFuseAll() {
		runGLMTest(GLMType.POISSON_LOG, true, true, ExecType.SPARK, CodegenTestType.FUSE_ALL);
	}

//	@Test
//	public void testGLMGammaDenseRewritesSPFuseAll() {
//		runGLMTest(GLMType.GAMMA_LOG, true, false, ExecType.SPARK, CodegenTestType.FUSE_ALL);
//	}
//
//	@Test
//	public void testGLMGammaSparseRewritesSPFuseAll() {
//		runGLMTest(GLMType.GAMMA_LOG, true, true, ExecType.SPARK, CodegenTestType.FUSE_ALL);
//	}

	@Test
	public void testGLMBinomialDenseRewritesSPFuseAll() {
		runGLMTest(GLMType.BINOMIAL_PROBIT, true, false, ExecType.SPARK, CodegenTestType.FUSE_ALL);
	}

	@Test
	public void testGLMBinomialSparseRewritesSPFuseAll() {
		runGLMTest(GLMType.BINOMIAL_PROBIT, true, true, ExecType.SPARK, CodegenTestType.FUSE_ALL);
	}

	@Test
	public void testGLMPoissonDenseRewritesCPFuseNoRedundancy() {
		runGLMTest(GLMType.POISSON_LOG, true, false, ExecType.CP, CodegenTestType.FUSE_NO_REDUNDANCY);
	}

	@Test
	public void testGLMPoissonSparseRewritesCPFuseNoRedundancy() {
		runGLMTest(GLMType.POISSON_LOG, true, true, ExecType.CP, CodegenTestType.FUSE_NO_REDUNDANCY);
	}

//	@Test
//	public void testGLMGammaDenseRewritesCPFuseNoRedundancy() {
//		runGLMTest(GLMType.GAMMA_LOG, true, false, ExecType.CP, CodegenTestType.FUSE_NO_REDUNDANCY);
//	}
//
//	@Test
//	public void testGLMGammaSparseRewritesCPFuseNoRedundancy() {
//		runGLMTest(GLMType.GAMMA_LOG, true, true, ExecType.CP, CodegenTestType.FUSE_NO_REDUNDANCY);
//	}

	@Test
	public void testGLMBinomialDenseRewritesCPFuseNoRedundancy() {
		runGLMTest(GLMType.BINOMIAL_PROBIT, true, false, ExecType.CP, CodegenTestType.FUSE_NO_REDUNDANCY);
	}

	@Test
	public void testGLMBinomialSparseRewritesCPFuseNoRedundancy() {
		runGLMTest(GLMType.BINOMIAL_PROBIT, true, true, ExecType.CP, CodegenTestType.FUSE_NO_REDUNDANCY);
	}

	@Test
	public void testGLMPoissonDenseRewritesSPFuseNoRedundancy() {
		runGLMTest(GLMType.POISSON_LOG, true, false, ExecType.SPARK, CodegenTestType.FUSE_NO_REDUNDANCY);
	}

	@Test
	public void testGLMPoissonSparseRewritesSPFuseNoRedundancy() {
		runGLMTest(GLMType.POISSON_LOG, true, true, ExecType.SPARK, CodegenTestType.FUSE_NO_REDUNDANCY);
	}

//	@Test
//	public void testGLMGammaDenseRewritesSPFuseNoRedundancy() {
//		runGLMTest(GLMType.GAMMA_LOG, true, false, ExecType.SPARK, CodegenTestType.FUSE_NO_REDUNDANCY);
//	}
//
//	@Test
//	public void testGLMGammaSparseRewritesSPFuseNoRedundancy() {
//		runGLMTest(GLMType.GAMMA_LOG, true, true, ExecType.SPARK, CodegenTestType.FUSE_NO_REDUNDANCY);
//	}

	@Test
	public void testGLMBinomialDenseRewritesSPFuseNoRedundancy() {
		runGLMTest(GLMType.BINOMIAL_PROBIT, true, false, ExecType.SPARK, CodegenTestType.FUSE_NO_REDUNDANCY);
	}

	@Test
	public void testGLMBinomialSparseRewritesSPFuseNoRedundancy() {
		runGLMTest(GLMType.BINOMIAL_PROBIT, true, true, ExecType.SPARK, CodegenTestType.FUSE_NO_REDUNDANCY);
	}

	private void runGLMTest( GLMType type, boolean rewrites, boolean sparse, ExecType instType, CodegenTestType CodegenTestType)
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
			
			String[] addArgs = new String[4];
			String param4Name = "lpow=";
			switch(type) {
				case POISSON_LOG: //dfam, vpow, link, lpow
					addArgs[0] = "1"; addArgs[1] = "1.0"; addArgs[2] = "1"; addArgs[3] = "0.0";
					break;
				case GAMMA_LOG:   //dfam, vpow, link, lpow
					addArgs[0] = "1"; addArgs[1] = "2.0"; addArgs[2] = "1"; addArgs[3] = "0.0";
					break;
				case BINOMIAL_PROBIT: //dfam, vpow, link, yneg 
					addArgs[0] = "2"; addArgs[1] = "0.0"; addArgs[2] = "3"; addArgs[3] = "0";
					param4Name = "yneg=";
					break;
			}
			
			fullDMLScriptName = "src/test/scripts/applications/glm/GLM.dml";
			programArgs = new String[]{ "-stats", "-nvargs", "X="+input("X"), "Y="+input("Y"),
				"icpt="+String.valueOf(intercept), "tol="+String.valueOf(epsilon), "moi="+String.valueOf(maxiter), 
				"dfam="+addArgs[0], "vpow="+addArgs[1], "link="+addArgs[2], param4Name+addArgs[3], "B="+output("w")};

			rCmd = getRCmd(inputDir(), String.valueOf(intercept),String.valueOf(epsilon),
				String.valueOf(maxiter), addArgs[0], addArgs[1], addArgs[2], addArgs[3], expectedDir());

			OptimizerUtils.ALLOW_ALGEBRAIC_SIMPLIFICATION = rewrites;
			
			//generate actual datasets
			double[][] X = getRandomMatrix(rows, cols, 0, 1, sparse?sparsity2:sparsity1, 714);
			writeInputMatrixWithMTD("X", X, true);
			double[][] y = TestUtils.round(getRandomMatrix(rows, 1, 0, 1, 1.0, 136));
			writeInputMatrixWithMTD("Y", y, true);
			
			runTest(true, false, null, -1); 
			//TODO fix R glm script
			//runRScript(true); 
			
			//compare matrices 
			//HashMap<CellIndex, Double> dmlfile = readDMLMatrixFromHDFS("w");
			//HashMap<CellIndex, Double> rfile  = readRMatrixFromFS("w");
			//TestUtils.compareMatrices(dmlfile, rfile, eps, "Stat-DML", "Stat-R");
			Assert.assertTrue(heavyHittersContainsSubString(Opcodes.SPOOF.toString()) || heavyHittersContainsSubString("sp_spoof"));
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
