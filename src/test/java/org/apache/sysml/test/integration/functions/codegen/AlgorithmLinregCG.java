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
import java.util.HashMap;

import org.junit.Assert;
import org.junit.Test;
import org.apache.sysml.api.DMLScript;
import org.apache.sysml.api.DMLScript.RUNTIME_PLATFORM;
import org.apache.sysml.hops.OptimizerUtils;
import org.apache.sysml.lops.LopProperties.ExecType;
import org.apache.sysml.runtime.matrix.data.MatrixValue.CellIndex;
import org.apache.sysml.test.integration.AutomatedTestBase;
import org.apache.sysml.test.integration.TestConfiguration;
import org.apache.sysml.test.utils.TestUtils;

public class AlgorithmLinregCG extends AutomatedTestBase 
{	
	private final static String TEST_NAME1 = "Algorithm_LinregCG";
	private final static String TEST_DIR = "functions/codegen/";
	private final static String TEST_CLASS_DIR = TEST_DIR + AlgorithmLinregCG.class.getSimpleName() + "/";
	private final static String TEST_CONF = "SystemML-config-codegen.xml";
	private final static File   TEST_CONF_FILE = new File(SCRIPT_DIR + TEST_DIR, TEST_CONF);
	
	private final static double eps = 1e-1;
	
	private final static int rows = 2468;
	private final static int cols = 507;
		
	private final static double sparsity1 = 0.7; //dense
	private final static double sparsity2 = 0.1; //sparse
	
	private final static double epsilon = 0.000000001;
	private final static double maxiter = 10;
	
	@Override
	public void setUp() {
		TestUtils.clearAssertionInformation();
		addTestConfiguration(TEST_NAME1, new TestConfiguration(TEST_CLASS_DIR, TEST_NAME1, new String[] { "w" })); 
	}

	@Test
	public void testLinregCG0DenseRewritesCP() {
		runLinregCGTest(TEST_NAME1, true, false, 0, ExecType.CP);
	}
	
	@Test
	public void testLinregCG0SparseRewritesCP() {
		runLinregCGTest(TEST_NAME1, true, true, 0, ExecType.CP);
	}
	
	@Test
	public void testLinregCG0DenseCP() {
		runLinregCGTest(TEST_NAME1, false, false, 0, ExecType.CP);
	}
	
	@Test
	public void testLinregCG0SparseCP() {
		runLinregCGTest(TEST_NAME1, false, true, 0, ExecType.CP);
	}

	@Test
	public void testLinregCG0DenseRewritesSP() {
		runLinregCGTest(TEST_NAME1, true, false, 0, ExecType.SPARK);
	}
	
	@Test
	public void testLinregCG0SparseRewritesSP() {
		runLinregCGTest(TEST_NAME1, true, true, 0, ExecType.SPARK);
	}
	
	@Test
	public void testLinregCG0DenseSP() {
		runLinregCGTest(TEST_NAME1, false, false, 0, ExecType.SPARK);
	}
	
	@Test
	public void testLinregCG0SparseSP() {
		runLinregCGTest(TEST_NAME1, false, true, 0, ExecType.SPARK);
	}
	
	@Test
	public void testLinregCG1DenseRewritesCP() {
		runLinregCGTest(TEST_NAME1, true, false, 1, ExecType.CP);
	}
	
	@Test
	public void testLinregCG1SparseRewritesCP() {
		runLinregCGTest(TEST_NAME1, true, true, 1, ExecType.CP);
	}
	
	@Test
	public void testLinregCG1DenseCP() {
		runLinregCGTest(TEST_NAME1, false, false, 1, ExecType.CP);
	}
	
	@Test
	public void testLinregCG1SparseCP() {
		runLinregCGTest(TEST_NAME1, false, true, 1, ExecType.CP);
	}

	@Test
	public void testLinregCG1DenseRewritesSP() {
		runLinregCGTest(TEST_NAME1, true, false, 1, ExecType.SPARK);
	}
	
	@Test
	public void testLinregCG1SparseRewritesSP() {
		runLinregCGTest(TEST_NAME1, true, true, 1, ExecType.SPARK);
	}
	
	@Test
	public void testLinregCG1DenseSP() {
		runLinregCGTest(TEST_NAME1, false, false, 1, ExecType.SPARK);
	}
	
	@Test
	public void testLinregCG1SparseSP() {
		runLinregCGTest(TEST_NAME1, false, true, 1, ExecType.SPARK);
	}
	
	@Test
	public void testLinregCG2DenseRewritesCP() {
		runLinregCGTest(TEST_NAME1, true, false, 2, ExecType.CP);
	}
	
	@Test
	public void testLinregCG2SparseRewritesCP() {
		runLinregCGTest(TEST_NAME1, true, true, 2, ExecType.CP);
	}
	
	@Test
	public void testLinregCG2DenseCP() {
		runLinregCGTest(TEST_NAME1, false, false, 2, ExecType.CP);
	}
	
	@Test
	public void testLinregCG2SparseCP() {
		runLinregCGTest(TEST_NAME1, false, true, 2, ExecType.CP);
	}

	@Test
	public void testLinregCG2DenseRewritesSP() {
		runLinregCGTest(TEST_NAME1, true, false, 2, ExecType.SPARK);
	}
	
	@Test
	public void testLinregCG2SparseRewritesSP() {
		runLinregCGTest(TEST_NAME1, true, true, 2, ExecType.SPARK);
	}
	
	@Test
	public void testLinregCG2DenseSP() {
		runLinregCGTest(TEST_NAME1, false, false, 2, ExecType.SPARK);
	}
	
	@Test
	public void testLinregCG2SparseSP() {
		runLinregCGTest(TEST_NAME1, false, true, 2, ExecType.SPARK);
	}
	
	private void runLinregCGTest( String testname, boolean rewrites, boolean sparse, int intercept, ExecType instType)
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
			String TEST_NAME = testname;
			TestConfiguration config = getTestConfiguration(TEST_NAME);
			loadTestConfiguration(config);
			
			fullDMLScriptName = "scripts/algorithms/LinearRegCG.dml";
			programArgs = new String[]{ "-explain", "-stats", "-nvargs", "X="+input("X"), "Y="+input("y"),
				"icpt="+String.valueOf(intercept), "tol="+String.valueOf(epsilon),
				"maxi="+String.valueOf(maxiter), "reg=0.001", "B="+output("w")};

			rCmd = getRCmd(inputDir(), String.valueOf(intercept),String.valueOf(epsilon),
				String.valueOf(maxiter), "0.001", expectedDir());
	
			OptimizerUtils.ALLOW_ALGEBRAIC_SIMPLIFICATION = rewrites;
			
			//generate actual datasets
			double[][] X = getRandomMatrix(rows, cols, 0, 1, sparse?sparsity2:sparsity1, 7);
			writeInputMatrixWithMTD("X", X, true);
			double[][] y = getRandomMatrix(rows, 1, 0, 10, 1.0, 3);
			writeInputMatrixWithMTD("y", y, true);
			
			runTest(true, false, null, -1); 
			runRScript(true); 
			
			//compare matrices 
			HashMap<CellIndex, Double> dmlfile = readDMLMatrixFromHDFS("w");
			HashMap<CellIndex, Double> rfile  = readRMatrixFromFS("w");
			TestUtils.compareMatrices(dmlfile, rfile, eps, "Stat-DML", "Stat-R");
			Assert.assertTrue(heavyHittersContainsSubString("spoofRA") 
					|| heavyHittersContainsSubString("sp_spoofRA"));
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
