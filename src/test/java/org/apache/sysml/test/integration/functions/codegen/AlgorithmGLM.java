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

public class AlgorithmGLM extends AutomatedTestBase 
{	
	private final static String TEST_NAME1 = "Algorithm_GLM";
	private final static String TEST_DIR = "functions/codegen/";
	private final static String TEST_CLASS_DIR = TEST_DIR + AlgorithmGLM.class.getSimpleName() + "/";
	private final static String TEST_CONF = "SystemML-config-codegen.xml";
	private final static File   TEST_CONF_FILE = new File(SCRIPT_DIR + TEST_DIR, TEST_CONF);
	
	//private final static double eps = 1e-5;
	
	private final static int rows = 2468;
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
	
	@Override
	public void setUp() {
		TestUtils.clearAssertionInformation();
		addTestConfiguration(TEST_NAME1, new TestConfiguration(TEST_CLASS_DIR, TEST_NAME1, new String[] { "w" })); 
	}

	@Test
	public void testGLMPoissonDenseRewritesCP() {
		runGLMTest(GLMType.POISSON_LOG, true, false, ExecType.CP);
	}
	
	@Test
	public void testGLMPoissonSparseRewritesCP() {
		runGLMTest(GLMType.POISSON_LOG, true, true, ExecType.CP);
	}
	
	@Test
	public void testGLMPoissonDenseCP() {
		runGLMTest(GLMType.POISSON_LOG, false, false, ExecType.CP);
	}
	
	@Test
	public void testGLMPoissonSparseCP() {
		runGLMTest(GLMType.POISSON_LOG, false, true, ExecType.CP);
	}
	
	@Test
	public void testGLMGammaDenseRewritesCP() {
		runGLMTest(GLMType.GAMMA_LOG, true, false, ExecType.CP);
	}
	
	@Test
	public void testGLMGammaSparseRewritesCP() {
		runGLMTest(GLMType.GAMMA_LOG, true, true, ExecType.CP);
	}
	
	@Test
	public void testGLMGammaDenseCP() {
		runGLMTest(GLMType.GAMMA_LOG, false, false, ExecType.CP);
	}
	
	@Test
	public void testGLMGammaSparseCP() {
		runGLMTest(GLMType.GAMMA_LOG, false, true, ExecType.CP);
	}
	
	@Test
	public void testGLMBinomialDenseRewritesCP() {
		runGLMTest(GLMType.BINOMIAL_PROBIT, true, false, ExecType.CP);
	}
	
	@Test
	public void testGLMBinomialSparseRewritesCP() {
		runGLMTest(GLMType.BINOMIAL_PROBIT, true, true, ExecType.CP);
	}
	
	@Test
	public void testGLMBinomialDenseCP() {
		runGLMTest(GLMType.BINOMIAL_PROBIT, false, false, ExecType.CP);
	}
	
	@Test
	public void testGLMBinomialSparseCP() {
		runGLMTest(GLMType.BINOMIAL_PROBIT, false, true, ExecType.CP);
	}
	
	private void runGLMTest( GLMType type, boolean rewrites, boolean sparse, ExecType instType)
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
					addArgs[0] = "2"; addArgs[1] = "0.0"; addArgs[2] = "3"; addArgs[3] = "2";
					param4Name = "yneg=";
					break;
			}
			
			fullDMLScriptName = "scripts/algorithms/GLM.dml";
			programArgs = new String[]{ "-explain", "-stats", "-nvargs", "X="+input("X"), "Y="+input("Y"),
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
			Assert.assertTrue(heavyHittersContainsSubString("spoof") || heavyHittersContainsSubString("sp_spoof"));
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
