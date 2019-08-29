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

public class AlgorithmMLogreg extends AutomatedTestBase 
{
	private final static String TEST_NAME1 = "Algorithm_MLogreg";
	private final static String TEST_DIR = "functions/codegenalg/";
	private final static String TEST_CLASS_DIR = TEST_DIR + AlgorithmMLogreg.class.getSimpleName() + "/";
	private final static String TEST_CONF_DEFAULT = "SystemDS-config-codegen.xml";
	private final static File TEST_CONF_FILE_DEFAULT = new File(SCRIPT_DIR + TEST_DIR, TEST_CONF_DEFAULT);
	private final static String TEST_CONF_FUSE_ALL = "SystemDS-config-codegen-fuse-all.xml";
	private final static File TEST_CONF_FILE_FUSE_ALL = new File(SCRIPT_DIR + TEST_DIR, TEST_CONF_FUSE_ALL);
	private final static String TEST_CONF_FUSE_NO_REDUNDANCY = "SystemDS-config-codegen-fuse-no-redundancy.xml";
	private final static File TEST_CONF_FILE_FUSE_NO_REDUNDANCY = new File(SCRIPT_DIR + TEST_DIR,
			TEST_CONF_FUSE_NO_REDUNDANCY);

	private enum TestType { DEFAULT,FUSE_ALL,FUSE_NO_REDUNDANCY }

	private final static double eps = 1e-5;
	
	private final static int rows = 2468;
	private final static int cols = 227;
	
	private final static double sparsity1 = 0.7; //dense
	private final static double sparsity2 = 0.1; //sparse
	
	private final static double epsilon = 0.000000001;
	private final static double maxiter = 10;
	
	private TestType currentTestType = TestType.DEFAULT;
	
	
	@Override
	public void setUp() {
		TestUtils.clearAssertionInformation();
		addTestConfiguration(TEST_NAME1, new TestConfiguration(TEST_CLASS_DIR, TEST_NAME1, new String[] { "w" })); 
	}

	@Test
	public void testMlogregBin0DenseRewritesCP() {
		runMlogregTest(TEST_NAME1, 2, 0, true, false, ExecType.CP, TestType.DEFAULT);
	}
	
	@Test
	public void testMlogregBin0SparseRewritesCP() {
		runMlogregTest(TEST_NAME1, 2, 0, true, true, ExecType.CP, TestType.DEFAULT);
	}
	
	@Test
	public void testMlogregBin0DenseCP() {
		runMlogregTest(TEST_NAME1, 2, 0, false, false, ExecType.CP, TestType.DEFAULT);
	}
	
	@Test
	public void testMlogregBin0SparseCP() {
		runMlogregTest(TEST_NAME1, 2, 0, false, true, ExecType.CP, TestType.DEFAULT);
	}
	
	@Test
	public void testMlogregMul0DenseRewritesCP() {
		runMlogregTest(TEST_NAME1, 5, 0, true, false, ExecType.CP, TestType.DEFAULT);
	}
	
	@Test
	public void testMlogregMul0SparseRewritesCP() {
		runMlogregTest(TEST_NAME1, 5, 0, true, true, ExecType.CP, TestType.DEFAULT);
	}
	
	@Test
	public void testMlogregMul0DenseCP() {
		runMlogregTest(TEST_NAME1, 5, 0, false, false, ExecType.CP, TestType.DEFAULT);
	}
	
	@Test
	public void testMlogregMul0SparseCP() {
		runMlogregTest(TEST_NAME1, 5, 0, false, true, ExecType.CP, TestType.DEFAULT);
	}

	@Test
	public void testMlogregBin0DenseRewritesSP() {
		runMlogregTest(TEST_NAME1, 2, 0, true, false, ExecType.SPARK, TestType.DEFAULT);
	}
	
	@Test
	public void testMlogregBin0SparseRewritesSP() {
		runMlogregTest(TEST_NAME1, 2, 0, true, true, ExecType.SPARK, TestType.DEFAULT);
	}
	
	@Test
	public void testMlogregBin0DenseSP() {
		runMlogregTest(TEST_NAME1, 2, 0, false, false, ExecType.SPARK, TestType.DEFAULT);
	}
	
	@Test
	public void testMlogregBin0SparseSP() {
		runMlogregTest(TEST_NAME1, 2, 0, false, true, ExecType.SPARK, TestType.DEFAULT);
	}
	
	@Test
	public void testMlogregMul0DenseRewritesSP() {
		runMlogregTest(TEST_NAME1, 5, 0, true, false, ExecType.SPARK, TestType.DEFAULT);
	}
	
	@Test
	public void testMlogregMul0SparseRewritesSP() {
		runMlogregTest(TEST_NAME1, 5, 0, true, true, ExecType.SPARK, TestType.DEFAULT);
	}
	
	@Test
	public void testMlogregMul0DenseSP() {
		runMlogregTest(TEST_NAME1, 5, 0, false, false, ExecType.SPARK, TestType.DEFAULT);
	}
	
	@Test
	public void testMlogregMul0SparseSP() {
		runMlogregTest(TEST_NAME1, 5, 0, false, true, ExecType.SPARK, TestType.DEFAULT);
	}
	
	@Test
	public void testMlogregBin1DenseRewritesCP() {
		runMlogregTest(TEST_NAME1, 2, 1, true, false, ExecType.CP, TestType.DEFAULT);
	}
	
	@Test
	public void testMlogregBin1SparseRewritesCP() {
		runMlogregTest(TEST_NAME1, 2, 1, true, true, ExecType.CP, TestType.DEFAULT);
	}
	
	@Test
	public void testMlogregBin1DenseCP() {
		runMlogregTest(TEST_NAME1, 2, 1, false, false, ExecType.CP, TestType.DEFAULT);
	}
	
	@Test
	public void testMlogregBin1SparseCP() {
		runMlogregTest(TEST_NAME1, 2, 1, false, true, ExecType.CP, TestType.DEFAULT);
	}
	
	@Test
	public void testMlogregMul1DenseRewritesCP() {
		runMlogregTest(TEST_NAME1, 5, 1, true, false, ExecType.CP, TestType.DEFAULT);
	}
	
	@Test
	public void testMlogregMul1SparseRewritesCP() {
		runMlogregTest(TEST_NAME1, 5, 1, true, true, ExecType.CP, TestType.DEFAULT);
	}
	
	@Test
	public void testMlogregMul1DenseCP() {
		runMlogregTest(TEST_NAME1, 5, 1, false, false, ExecType.CP, TestType.DEFAULT);
	}
	
	@Test
	public void testMlogregMul1SparseCP() {
		runMlogregTest(TEST_NAME1, 5, 1, false, true, ExecType.CP, TestType.DEFAULT);
	}

	@Test
	public void testMlogregBin2DenseRewritesCP() {
		runMlogregTest(TEST_NAME1, 2, 2, true, false, ExecType.CP, TestType.DEFAULT);
	}
	
	@Test
	public void testMlogregBin2SparseRewritesCP() {
		runMlogregTest(TEST_NAME1, 2, 2, true, true, ExecType.CP, TestType.DEFAULT);
	}
	
	@Test
	public void testMlogregBin2DenseCP() {
		runMlogregTest(TEST_NAME1, 2, 2, false, false, ExecType.CP, TestType.DEFAULT);
	}
	
	@Test
	public void testMlogregBin2SparseCP() {
		runMlogregTest(TEST_NAME1, 2, 2, false, true, ExecType.CP, TestType.DEFAULT);
	}
	
	@Test
	public void testMlogregMul2DenseRewritesCP() {
		runMlogregTest(TEST_NAME1, 5, 2, true, false, ExecType.CP, TestType.DEFAULT);
	}
	
	@Test
	public void testMlogregMul2SparseRewritesCP() {
		runMlogregTest(TEST_NAME1, 5, 2, true, true, ExecType.CP, TestType.DEFAULT);
	}
	
	@Test
	public void testMlogregMul2DenseCP() {
		runMlogregTest(TEST_NAME1, 5, 2, false, false, ExecType.CP, TestType.DEFAULT);
	}
	
	@Test
	public void testMlogregMul2SparseCP() {
		runMlogregTest(TEST_NAME1, 5, 2, false, true, ExecType.CP, TestType.DEFAULT);
	}

	@Test
	public void testMlogregBin0DenseRewritesCPFuseAll() {
		runMlogregTest(TEST_NAME1, 2, 0, true, false, ExecType.CP, TestType.FUSE_ALL);
	}

	@Test
	public void testMlogregBin0SparseRewritesCPFuseAll() {
		runMlogregTest(TEST_NAME1, 2, 0, true, true, ExecType.CP, TestType.FUSE_ALL);
	}

	@Test
	public void testMlogregMul0DenseRewritesCPFuseAll() {
		runMlogregTest(TEST_NAME1, 5, 0, true, false, ExecType.CP, TestType.FUSE_ALL);
	}

	@Test
	public void testMlogregMul0SparseRewritesCPFuseAll() {
		runMlogregTest(TEST_NAME1, 5, 0, true, true, ExecType.CP, TestType.FUSE_ALL);
	}

	@Test
	public void testMlogregBin0DenseRewritesSPFuseAll() {
		runMlogregTest(TEST_NAME1, 2, 0, true, false, ExecType.SPARK, TestType.FUSE_ALL);
	}

	@Test
	public void testMlogregBin0SparseRewritesSPFuseAll() {
		runMlogregTest(TEST_NAME1, 2, 0, true, true, ExecType.SPARK, TestType.FUSE_ALL);
	}

	@Test
	public void testMlogregMul0DenseRewritesSPFuseAll() {
		runMlogregTest(TEST_NAME1, 5, 0, true, false, ExecType.SPARK, TestType.FUSE_ALL);
	}

	@Test
	public void testMlogregMul0SparseRewritesSPFuseAll() {
		runMlogregTest(TEST_NAME1, 5, 0, true, true, ExecType.SPARK, TestType.FUSE_ALL);
	}

	@Test
	public void testMlogregBin1DenseRewritesCPFuseAll() {
		runMlogregTest(TEST_NAME1, 2, 1, true, false, ExecType.CP, TestType.FUSE_ALL);
	}

	@Test
	public void testMlogregBin1SparseRewritesCPFuseAll() {
		runMlogregTest(TEST_NAME1, 2, 1, true, true, ExecType.CP, TestType.FUSE_ALL);
	}

	@Test
	public void testMlogregMul1DenseRewritesCPFuseAll() {
		runMlogregTest(TEST_NAME1, 5, 1, true, false, ExecType.CP, TestType.FUSE_ALL);
	}

	@Test
	public void testMlogregMul1SparseRewritesCPFuseAll() {
		runMlogregTest(TEST_NAME1, 5, 1, true, true, ExecType.CP, TestType.FUSE_ALL);
	}

	@Test
	public void testMlogregBin2DenseRewritesCPFuseAll() {
		runMlogregTest(TEST_NAME1, 2, 2, true, false, ExecType.CP, TestType.FUSE_ALL);
	}

	@Test
	public void testMlogregBin2SparseRewritesCPFuseAll() {
		runMlogregTest(TEST_NAME1, 2, 2, true, true, ExecType.CP, TestType.FUSE_ALL);
	}

	@Test
	public void testMlogregMul2DenseRewritesCPFuseAll() {
		runMlogregTest(TEST_NAME1, 5, 2, true, false, ExecType.CP, TestType.FUSE_ALL);
	}

	@Test
	public void testMlogregMul2SparseRewritesCPFuseAll() {
		runMlogregTest(TEST_NAME1, 5, 2, true, true, ExecType.CP, TestType.FUSE_ALL);
	}

	@Test
	public void testMlogregBin0DenseRewritesCPFuseNoRedundancy() {
		runMlogregTest(TEST_NAME1, 2, 0, true, false, ExecType.CP, TestType.FUSE_NO_REDUNDANCY);
	}

	@Test
	public void testMlogregBin0SparseRewritesCPFuseNoRedundancy() {
		runMlogregTest(TEST_NAME1, 2, 0, true, true, ExecType.CP, TestType.FUSE_NO_REDUNDANCY);
	}

	@Test
	public void testMlogregMul0DenseRewritesCPFuseNoRedundancy() {
		runMlogregTest(TEST_NAME1, 5, 0, true, false, ExecType.CP, TestType.FUSE_NO_REDUNDANCY);
	}

	@Test
	public void testMlogregMul0SparseRewritesCPFuseNoRedundancy() {
		runMlogregTest(TEST_NAME1, 5, 0, true, true, ExecType.CP, TestType.FUSE_NO_REDUNDANCY);
	}

	@Test
	public void testMlogregBin0DenseRewritesSPFuseNoRedundancy() {
		runMlogregTest(TEST_NAME1, 2, 0, true, false, ExecType.SPARK, TestType.FUSE_NO_REDUNDANCY);
	}

	@Test
	public void testMlogregBin0SparseRewritesSPFuseNoRedundancy() {
		runMlogregTest(TEST_NAME1, 2, 0, true, true, ExecType.SPARK, TestType.FUSE_NO_REDUNDANCY);
	}

	@Test
	public void testMlogregMul0DenseRewritesSPFuseNoRedundancy() {
		runMlogregTest(TEST_NAME1, 5, 0, true, false, ExecType.SPARK, TestType.FUSE_NO_REDUNDANCY);
	}

	@Test
	public void testMlogregMul0SparseRewritesSPFuseNoRedundancy() {
		runMlogregTest(TEST_NAME1, 5, 0, true, true, ExecType.SPARK, TestType.FUSE_NO_REDUNDANCY);
	}

	@Test
	public void testMlogregBin1DenseRewritesCPFuseNoRedundancy() {
		runMlogregTest(TEST_NAME1, 2, 1, true, false, ExecType.CP, TestType.FUSE_NO_REDUNDANCY);
	}

	@Test
	public void testMlogregBin1SparseRewritesCPFuseNoRedundancy() {
		runMlogregTest(TEST_NAME1, 2, 1, true, true, ExecType.CP, TestType.FUSE_NO_REDUNDANCY);
	}

	@Test
	public void testMlogregMul1DenseRewritesCPFuseNoRedundancy() {
		runMlogregTest(TEST_NAME1, 5, 1, true, false, ExecType.CP, TestType.FUSE_NO_REDUNDANCY);
	}

	@Test
	public void testMlogregMul1SparseRewritesCPFuseNoRedundancy() {
		runMlogregTest(TEST_NAME1, 5, 1, true, true, ExecType.CP, TestType.FUSE_NO_REDUNDANCY);
	}

	@Test
	public void testMlogregBin2DenseRewritesCPFuseNoRedundancy() {
		runMlogregTest(TEST_NAME1, 2, 2, true, false, ExecType.CP, TestType.FUSE_NO_REDUNDANCY);
	}

	@Test
	public void testMlogregBin2SparseRewritesCPFuseNoRedundancy() {
		runMlogregTest(TEST_NAME1, 2, 2, true, true, ExecType.CP, TestType.FUSE_NO_REDUNDANCY);
	}

	@Test
	public void testMlogregMul2DenseRewritesCPFuseNoRedundancy() {
		runMlogregTest(TEST_NAME1, 5, 2, true, false, ExecType.CP, TestType.FUSE_NO_REDUNDANCY);
	}

	@Test
	public void testMlogregMul2SparseRewritesCPFuseNoRedundancy() {
		runMlogregTest(TEST_NAME1, 5, 2, true, true, ExecType.CP, TestType.FUSE_NO_REDUNDANCY);
	}
	
	private void runMlogregTest( String testname, int classes, int intercept, boolean rewrites, boolean sparse, ExecType instType, TestType testType)
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
			
			fullDMLScriptName = "scripts/algorithms/MultiLogReg.dml";
			programArgs = new String[]{ "-explain", "-stats", "-nvargs", "X="+input("X"), "Y="+input("Y"),
				"icpt="+String.valueOf(intercept), "tol="+String.valueOf(epsilon),
				"moi="+String.valueOf(maxiter), "reg=0.001", "B="+output("w")};

			rCmd = getRCmd(inputDir(), String.valueOf(intercept),String.valueOf(epsilon),
				String.valueOf(maxiter), expectedDir());

			OptimizerUtils.ALLOW_ALGEBRAIC_SIMPLIFICATION = rewrites;
			
			//generate actual datasets
			double[][] X = getRandomMatrix(rows, cols, 0, 1, sparse?sparsity2:sparsity1, 2384);
			writeInputMatrixWithMTD("X", X, true);
			double[][] y = TestUtils.round(getRandomMatrix(rows, 1, 0.51, classes+0.49, 1.0, 9283));
			writeInputMatrixWithMTD("Y", y, true);
			
			runTest(true, false, null, -1); 
			runRScript(true); 
			
			//compare matrices 
			HashMap<CellIndex, Double> dmlfile = readDMLMatrixFromHDFS("w");
			HashMap<CellIndex, Double> rfile  = readRMatrixFromFS("w");
			TestUtils.compareMatrices(dmlfile, rfile, eps, "Stat-DML", "Stat-R");
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
