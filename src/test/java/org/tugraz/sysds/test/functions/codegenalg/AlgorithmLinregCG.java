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
import java.util.HashMap;

import org.junit.Assert;
import org.junit.Test;
import org.tugraz.sysds.api.DMLScript;
import org.tugraz.sysds.common.Types.ExecMode;
import org.tugraz.sysds.hops.OptimizerUtils;
import org.tugraz.sysds.lops.LopProperties.ExecType;
import org.tugraz.sysds.runtime.matrix.data.MatrixValue.CellIndex;
import org.tugraz.sysds.test.AutomatedTestBase;
import org.tugraz.sysds.test.TestConfiguration;
import org.tugraz.sysds.test.TestUtils;

public class AlgorithmLinregCG extends AutomatedTestBase 
{	
	private final static String TEST_NAME1 = "Algorithm_LinregCG";
	private final static String TEST_DIR = "functions/codegenalg/";
	private final static String TEST_CLASS_DIR = TEST_DIR + AlgorithmLinregCG.class.getSimpleName() + "/";
	private final static String TEST_CONF_DEFAULT = "SystemDS-config-codegen.xml";
	private final static File TEST_CONF_FILE_DEFAULT = new File(SCRIPT_DIR + TEST_DIR, TEST_CONF_DEFAULT);
	private final static String TEST_CONF_FUSE_ALL = "SystemDS-config-codegen-fuse-all.xml";
	private final static File TEST_CONF_FILE_FUSE_ALL = new File(SCRIPT_DIR + TEST_DIR, TEST_CONF_FUSE_ALL);
	private final static String TEST_CONF_FUSE_NO_REDUNDANCY = "SystemDS-config-codegen-fuse-no-redundancy.xml";
	private final static File TEST_CONF_FILE_FUSE_NO_REDUNDANCY = new File(SCRIPT_DIR + TEST_DIR,
			TEST_CONF_FUSE_NO_REDUNDANCY);

	private enum TestType { DEFAULT,FUSE_ALL,FUSE_NO_REDUNDANCY }
	private static TestType currentTestType = TestType.DEFAULT;

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
		runLinregCGTest(TEST_NAME1, true, false, 0, ExecType.CP, TestType.DEFAULT);
	}
	
	@Test
	public void testLinregCG0SparseRewritesCP() {
		runLinregCGTest(TEST_NAME1, true, true, 0, ExecType.CP, TestType.DEFAULT);
	}
	
	@Test
	public void testLinregCG0DenseCP() {
		runLinregCGTest(TEST_NAME1, false, false, 0, ExecType.CP, TestType.DEFAULT);
	}
	
	@Test
	public void testLinregCG0SparseCP() {
		runLinregCGTest(TEST_NAME1, false, true, 0, ExecType.CP, TestType.DEFAULT);
	}

	@Test
	public void testLinregCG0DenseRewritesSP() {
		runLinregCGTest(TEST_NAME1, true, false, 0, ExecType.SPARK, TestType.DEFAULT);
	}
	
	@Test
	public void testLinregCG0SparseRewritesSP() {
		runLinregCGTest(TEST_NAME1, true, true, 0, ExecType.SPARK, TestType.DEFAULT);
	}
	
	@Test
	public void testLinregCG0DenseSP() {
		runLinregCGTest(TEST_NAME1, false, false, 0, ExecType.SPARK, TestType.DEFAULT);
	}
	
	@Test
	public void testLinregCG0SparseSP() {
		runLinregCGTest(TEST_NAME1, false, true, 0, ExecType.SPARK, TestType.DEFAULT);
	}
	
	@Test
	public void testLinregCG1DenseRewritesCP() {
		runLinregCGTest(TEST_NAME1, true, false, 1, ExecType.CP, TestType.DEFAULT);
	}
	
	@Test
	public void testLinregCG1SparseRewritesCP() {
		runLinregCGTest(TEST_NAME1, true, true, 1, ExecType.CP, TestType.DEFAULT);
	}
	
	@Test
	public void testLinregCG1DenseCP() {
		runLinregCGTest(TEST_NAME1, false, false, 1, ExecType.CP, TestType.DEFAULT);
	}
	
	@Test
	public void testLinregCG1SparseCP() {
		runLinregCGTest(TEST_NAME1, false, true, 1, ExecType.CP, TestType.DEFAULT);
	}

	@Test
	public void testLinregCG1DenseRewritesSP() {
		runLinregCGTest(TEST_NAME1, true, false, 1, ExecType.SPARK, TestType.DEFAULT);
	}
	
	@Test
	public void testLinregCG1SparseRewritesSP() {
		runLinregCGTest(TEST_NAME1, true, true, 1, ExecType.SPARK, TestType.DEFAULT);
	}
	
	@Test
	public void testLinregCG1DenseSP() {
		runLinregCGTest(TEST_NAME1, false, false, 1, ExecType.SPARK, TestType.DEFAULT);
	}
	
	@Test
	public void testLinregCG1SparseSP() {
		runLinregCGTest(TEST_NAME1, false, true, 1, ExecType.SPARK, TestType.DEFAULT);
	}
	
	@Test
	public void testLinregCG2DenseRewritesCP() {
		runLinregCGTest(TEST_NAME1, true, false, 2, ExecType.CP, TestType.DEFAULT);
	}
	
	@Test
	public void testLinregCG2SparseRewritesCP() {
		runLinregCGTest(TEST_NAME1, true, true, 2, ExecType.CP, TestType.DEFAULT);
	}
	
	@Test
	public void testLinregCG2DenseCP() {
		runLinregCGTest(TEST_NAME1, false, false, 2, ExecType.CP, TestType.DEFAULT);
	}
	
	@Test
	public void testLinregCG2SparseCP() {
		runLinregCGTest(TEST_NAME1, false, true, 2, ExecType.CP, TestType.DEFAULT);
	}

	@Test
	public void testLinregCG2DenseRewritesSP() {
		runLinregCGTest(TEST_NAME1, true, false, 2, ExecType.SPARK, TestType.DEFAULT);
	}
	
	@Test
	public void testLinregCG2SparseRewritesSP() {
		runLinregCGTest(TEST_NAME1, true, true, 2, ExecType.SPARK, TestType.DEFAULT);
	}
	
	@Test
	public void testLinregCG2DenseSP() {
		runLinregCGTest(TEST_NAME1, false, false, 2, ExecType.SPARK, TestType.DEFAULT);
	}
	
	@Test
	public void testLinregCG2SparseSP() {
		runLinregCGTest(TEST_NAME1, false, true, 2, ExecType.SPARK, TestType.DEFAULT);
	}

	@Test
	public void testLinregCG0DenseRewritesCPFuseAll() {
		runLinregCGTest(TEST_NAME1, true, false, 0, ExecType.CP, TestType.FUSE_ALL);
	}

	@Test
	public void testLinregCG0SparseRewritesCPFuseAll() {
		runLinregCGTest(TEST_NAME1, true, true, 0, ExecType.CP, TestType.FUSE_ALL);
	}

	@Test
	public void testLinregCG0DenseRewritesSPFuseAll() {
		runLinregCGTest(TEST_NAME1, true, false, 0, ExecType.SPARK, TestType.FUSE_ALL);
	}

	@Test
	public void testLinregCG0SparseRewritesSPFuseAll() {
		runLinregCGTest(TEST_NAME1, true, true, 0, ExecType.SPARK, TestType.FUSE_ALL);
	}

	@Test
	public void testLinregCG1DenseRewritesCPFuseAll() {
		runLinregCGTest(TEST_NAME1, true, false, 1, ExecType.CP, TestType.FUSE_ALL);
	}

	@Test
	public void testLinregCG1SparseRewritesCPFuseAll() {
		runLinregCGTest(TEST_NAME1, true, true, 1, ExecType.CP, TestType.FUSE_ALL);
	}

	@Test
	public void testLinregCG1DenseRewritesSPFuseAll() {
		runLinregCGTest(TEST_NAME1, true, false, 1, ExecType.SPARK, TestType.FUSE_ALL);
	}

	@Test
	public void testLinregCG1SparseRewritesSPFuseAll() {
		runLinregCGTest(TEST_NAME1, true, true, 1, ExecType.SPARK, TestType.FUSE_ALL);
	}

	@Test
	public void testLinregCG2DenseRewritesCPFuseAll() {
		runLinregCGTest(TEST_NAME1, true, false, 2, ExecType.CP, TestType.FUSE_ALL);
	}

	@Test
	public void testLinregCG2SparseRewritesCPFuseAll() {
		runLinregCGTest(TEST_NAME1, true, true, 2, ExecType.CP, TestType.FUSE_ALL);
	}

	@Test
	public void testLinregCG2DenseRewritesSPFuseAll() {
		runLinregCGTest(TEST_NAME1, true, false, 2, ExecType.SPARK, TestType.FUSE_ALL);
	}

	@Test
	public void testLinregCG2SparseRewritesSPFuseAll() {
		runLinregCGTest(TEST_NAME1, true, true, 2, ExecType.SPARK, TestType.FUSE_ALL);
	}

	@Test
	public void testLinregCG0DenseRewritesCPFuseNoRedundancy() {
		runLinregCGTest(TEST_NAME1, true, false, 0, ExecType.CP, TestType.FUSE_NO_REDUNDANCY);
	}

	@Test
	public void testLinregCG0SparseRewritesCPFuseNoRedundancy() {
		runLinregCGTest(TEST_NAME1, true, true, 0, ExecType.CP, TestType.FUSE_NO_REDUNDANCY);
	}

	@Test
	public void testLinregCG0DenseRewritesSPFuseNoRedundancy() {
		runLinregCGTest(TEST_NAME1, true, false, 0, ExecType.SPARK, TestType.FUSE_NO_REDUNDANCY);
	}

	@Test
	public void testLinregCG0SparseRewritesSPFuseNoRedundancy() {
		runLinregCGTest(TEST_NAME1, true, true, 0, ExecType.SPARK, TestType.FUSE_NO_REDUNDANCY);
	}

	@Test
	public void testLinregCG1DenseRewritesCPFuseNoRedundancy() {
		runLinregCGTest(TEST_NAME1, true, false, 1, ExecType.CP, TestType.FUSE_NO_REDUNDANCY);
	}

	@Test
	public void testLinregCG1SparseRewritesCPFuseNoRedundancy() {
		runLinregCGTest(TEST_NAME1, true, true, 1, ExecType.CP, TestType.FUSE_NO_REDUNDANCY);
	}

	@Test
	public void testLinregCG1DenseRewritesSPFuseNoRedundancy() {
		runLinregCGTest(TEST_NAME1, true, false, 1, ExecType.SPARK, TestType.FUSE_NO_REDUNDANCY);
	}

	@Test
	public void testLinregCG1SparseRewritesSPFuseNoRedundancy() {
		runLinregCGTest(TEST_NAME1, true, true, 1, ExecType.SPARK, TestType.FUSE_NO_REDUNDANCY);
	}

	@Test
	public void testLinregCG2DenseRewritesCPFuseNoRedundancy() {
		runLinregCGTest(TEST_NAME1, true, false, 2, ExecType.CP, TestType.FUSE_NO_REDUNDANCY);
	}

	@Test
	public void testLinregCG2SparseRewritesCPFuseNoRedundancy() {
		runLinregCGTest(TEST_NAME1, true, true, 2, ExecType.CP, TestType.FUSE_NO_REDUNDANCY);
	}

	@Test
	public void testLinregCG2DenseRewritesSPFuseNoRedundancy() {
		runLinregCGTest(TEST_NAME1, true, false, 2, ExecType.SPARK, TestType.FUSE_NO_REDUNDANCY);
	}

	@Test
	public void testLinregCG2SparseRewritesSPFuseNoRedundancy() {
		runLinregCGTest(TEST_NAME1, true, true, 2, ExecType.SPARK, TestType.FUSE_NO_REDUNDANCY);
	}

	private void runLinregCGTest( String testname, boolean rewrites, boolean sparse, int intercept, ExecType instType, TestType testType)
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
