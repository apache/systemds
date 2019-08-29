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

import org.junit.Assert;
import org.junit.Test;
import org.tugraz.sysds.api.DMLScript;
import org.tugraz.sysds.common.Types.ExecMode;
import org.tugraz.sysds.hops.OptimizerUtils;
import org.tugraz.sysds.lops.LopProperties.ExecType;
import org.tugraz.sysds.test.AutomatedTestBase;
import org.tugraz.sysds.test.TestConfiguration;
import org.tugraz.sysds.test.TestUtils;

public class AlgorithmKMeans extends AutomatedTestBase 
{
	private final static String TEST_NAME1 = "Algorithm_KMeans";
	private final static String TEST_DIR = "functions/codegenalg/";
	private final static String TEST_CLASS_DIR = TEST_DIR + AlgorithmKMeans.class.getSimpleName() + "/";
	private final static String TEST_CONF_DEFAULT = "SystemDS-config-codegen.xml";
	private final static File TEST_CONF_FILE_DEFAULT = new File(SCRIPT_DIR + TEST_DIR, TEST_CONF_DEFAULT);
	private final static String TEST_CONF_FUSE_ALL = "SystemDS-config-codegen-fuse-all.xml";
	private final static File TEST_CONF_FILE_FUSE_ALL = new File(SCRIPT_DIR + TEST_DIR, TEST_CONF_FUSE_ALL);
	private final static String TEST_CONF_FUSE_NO_REDUNDANCY = "SystemDS-config-codegen-fuse-no-redundancy.xml";
	private final static File TEST_CONF_FILE_FUSE_NO_REDUNDANCY = new File(SCRIPT_DIR + TEST_DIR,
			TEST_CONF_FUSE_NO_REDUNDANCY);

	private enum TestType { DEFAULT,FUSE_ALL,FUSE_NO_REDUNDANCY }

	//private final static double eps = 1e-5;
	
	private final static int rows = 3972;
	private final static int cols = 972;
		
	private final static double sparsity1 = 0.7; //dense
	private final static double sparsity2 = 0.1; //sparse
	
	private final static double epsilon = 0.000000001;
	private final static double maxiter = 10;
	
	private TestType currentTestType = TestType.DEFAULT;
	
	@Override
	public void setUp() {
		TestUtils.clearAssertionInformation();
		addTestConfiguration(TEST_NAME1, new TestConfiguration(TEST_CLASS_DIR, TEST_NAME1, new String[] { "C" })); 
	}

	@Test
	public void testKMeansDenseBinSingleRewritesCP() {
		runKMeansTest(TEST_NAME1, true, false, 2, 1, ExecType.CP, TestType.DEFAULT);
	}
	
	@Test
	public void testKMeansSparseBinSingleRewritesCP() {
		runKMeansTest(TEST_NAME1, true, true, 2, 1, ExecType.CP, TestType.DEFAULT);
	}
	
	@Test
	public void testKMeansDenseBinSingleCP() {
		runKMeansTest(TEST_NAME1, false, false, 2, 1, ExecType.CP, TestType.DEFAULT);
	}
	
	@Test
	public void testKMeansSparseBinSingleCP() {
		runKMeansTest(TEST_NAME1, false, true, 2, 1, ExecType.CP, TestType.DEFAULT);
	}
	
	@Test
	public void testKMeansDenseBinMultiRewritesCP() {
		runKMeansTest(TEST_NAME1, true, false, 2, 10, ExecType.CP, TestType.DEFAULT);
	}
	
	@Test
	public void testKMeansSparseBinMultiRewritesCP() {
		runKMeansTest(TEST_NAME1, true, true, 2, 10, ExecType.CP, TestType.DEFAULT);
	}
	
	@Test
	public void testKMeansDenseBinMultiCP() {
		runKMeansTest(TEST_NAME1, false, false, 2, 10, ExecType.CP, TestType.DEFAULT);
	}
	
	@Test
	public void testKMeansSparseBinMultiCP() {
		runKMeansTest(TEST_NAME1, false, true, 2, 10, ExecType.CP, TestType.DEFAULT);
	}
	
	@Test
	public void testKMeansDenseMulSingleRewritesCP() {
		runKMeansTest(TEST_NAME1, true, false, 20, 1, ExecType.CP, TestType.DEFAULT);
	}
	
	@Test
	public void testKMeansSparseMulSingleRewritesCP() {
		runKMeansTest(TEST_NAME1, true, true, 20, 1, ExecType.CP, TestType.DEFAULT);
	}
	
	@Test
	public void testKMeansDenseMulSingleCP() {
		runKMeansTest(TEST_NAME1, false, false, 20, 1, ExecType.CP, TestType.DEFAULT);
	}
	
	@Test
	public void testKMeansSparseMulSingleCP() {
		runKMeansTest(TEST_NAME1, false, true, 20, 1, ExecType.CP, TestType.DEFAULT);
	}
	
	@Test
	public void testKMeansDenseMulMultiRewritesCP() {
		runKMeansTest(TEST_NAME1, true, false, 20, 10, ExecType.CP, TestType.DEFAULT);
	}
	
	@Test
	public void testKMeansSparseMulMultiRewritesCP() {
		runKMeansTest(TEST_NAME1, true, true, 20, 10, ExecType.CP, TestType.DEFAULT);
	}
	
	@Test
	public void testKMeansDenseMulMultiCP() {
		runKMeansTest(TEST_NAME1, false, false, 20, 10, ExecType.CP, TestType.DEFAULT);
	}
	
	@Test
	public void testKMeansSparseMulMultiCP() {
		runKMeansTest(TEST_NAME1, false, true, 20, 10, ExecType.CP, TestType.DEFAULT);
	}

	@Test
	public void testKMeansDenseBinSingleRewritesCPFuseAll() {
		runKMeansTest(TEST_NAME1, true, false, 2, 1, ExecType.CP, TestType.FUSE_ALL);
	}

	@Test
	public void testKMeansSparseBinSingleRewritesCPFuseAll() {
		runKMeansTest(TEST_NAME1, true, true, 2, 1, ExecType.CP, TestType.FUSE_ALL);
	}

	@Test
	public void testKMeansDenseBinSingleCPFuseAll() {
		runKMeansTest(TEST_NAME1, false, false, 2, 1, ExecType.CP, TestType.FUSE_ALL);
	}

	@Test
	public void testKMeansSparseBinSingleCPFuseAll() {
		runKMeansTest(TEST_NAME1, false, true, 2, 1, ExecType.CP, TestType.FUSE_ALL);
	}

	@Test
	public void testKMeansDenseBinMultiRewritesCPFuseAll() {
		runKMeansTest(TEST_NAME1, true, false, 2, 10, ExecType.CP, TestType.FUSE_ALL);
	}

	@Test
	public void testKMeansSparseBinMultiRewritesCPFuseAll() {
		runKMeansTest(TEST_NAME1, true, true, 2, 10, ExecType.CP, TestType.FUSE_ALL);
	}

	@Test
	public void testKMeansDenseBinMultiCPFuseAll() {
		runKMeansTest(TEST_NAME1, false, false, 2, 10, ExecType.CP, TestType.FUSE_ALL);
	}

	@Test
	public void testKMeansSparseBinMultiCPFuseAll() {
		runKMeansTest(TEST_NAME1, false, true, 2, 10, ExecType.CP, TestType.FUSE_ALL);
	}

	@Test
	public void testKMeansDenseMulSingleRewritesCPFuseAll() {
		runKMeansTest(TEST_NAME1, true, false, 20, 1, ExecType.CP, TestType.FUSE_ALL);
	}

	@Test
	public void testKMeansSparseMulSingleRewritesCPFuseAll() {
		runKMeansTest(TEST_NAME1, true, true, 20, 1, ExecType.CP, TestType.FUSE_ALL);
	}

	@Test
	public void testKMeansDenseMulSingleCPFuseAll() {
		runKMeansTest(TEST_NAME1, false, false, 20, 1, ExecType.CP, TestType.FUSE_ALL);
	}

	@Test
	public void testKMeansSparseMulSingleCPFuseAll() {
		runKMeansTest(TEST_NAME1, false, true, 20, 1, ExecType.CP, TestType.FUSE_ALL);
	}

	@Test
	public void testKMeansDenseMulMultiRewritesCPFuseAll() {
		runKMeansTest(TEST_NAME1, true, false, 20, 10, ExecType.CP, TestType.FUSE_ALL);
	}

	@Test
	public void testKMeansSparseMulMultiRewritesCPFuseAll() {
		runKMeansTest(TEST_NAME1, true, true, 20, 10, ExecType.CP, TestType.FUSE_ALL);
	}

	@Test
	public void testKMeansDenseMulMultiCPFuseAll() {
		runKMeansTest(TEST_NAME1, false, false, 20, 10, ExecType.CP, TestType.FUSE_ALL);
	}

	@Test
	public void testKMeansSparseMulMultiCPFuseAll() {
		runKMeansTest(TEST_NAME1, false, true, 20, 10, ExecType.CP, TestType.FUSE_ALL);
	}

	@Test
	public void testKMeansDenseBinSingleRewritesCPFuseNoRedundancy() {
		runKMeansTest(TEST_NAME1, true, false, 2, 1, ExecType.CP, TestType.FUSE_NO_REDUNDANCY);
	}

	@Test
	public void testKMeansSparseBinSingleRewritesCPFuseNoRedundancy() {
		runKMeansTest(TEST_NAME1, true, true, 2, 1, ExecType.CP, TestType.FUSE_NO_REDUNDANCY);
	}

	@Test
	public void testKMeansDenseBinSingleCPFuseNoRedundancy() {
		runKMeansTest(TEST_NAME1, false, false, 2, 1, ExecType.CP, TestType.FUSE_NO_REDUNDANCY);
	}

	@Test
	public void testKMeansSparseBinSingleCPFuseNoRedundancy() {
		runKMeansTest(TEST_NAME1, false, true, 2, 1, ExecType.CP, TestType.FUSE_NO_REDUNDANCY);
	}

	@Test
	public void testKMeansDenseBinMultiRewritesCPFuseNoRedundancy() {
		runKMeansTest(TEST_NAME1, true, false, 2, 10, ExecType.CP, TestType.FUSE_NO_REDUNDANCY);
	}

	@Test
	public void testKMeansSparseBinMultiRewritesCPFuseNoRedundancy() {
		runKMeansTest(TEST_NAME1, true, true, 2, 10, ExecType.CP, TestType.FUSE_NO_REDUNDANCY);
	}

	@Test
	public void testKMeansDenseBinMultiCPFuseNoRedundancy() {
		runKMeansTest(TEST_NAME1, false, false, 2, 10, ExecType.CP, TestType.FUSE_NO_REDUNDANCY);
	}

	@Test
	public void testKMeansSparseBinMultiCPFuseNoRedundancy() {
		runKMeansTest(TEST_NAME1, false, true, 2, 10, ExecType.CP, TestType.FUSE_NO_REDUNDANCY);
	}

	@Test
	public void testKMeansDenseMulSingleRewritesCPFuseNoRedundancy() {
		runKMeansTest(TEST_NAME1, true, false, 20, 1, ExecType.CP, TestType.FUSE_NO_REDUNDANCY);
	}

	@Test
	public void testKMeansSparseMulSingleRewritesCPFuseNoRedundancy() {
		runKMeansTest(TEST_NAME1, true, true, 20, 1, ExecType.CP, TestType.FUSE_NO_REDUNDANCY);
	}

	@Test
	public void testKMeansDenseMulSingleCPFuseNoRedundancy() {
		runKMeansTest(TEST_NAME1, false, false, 20, 1, ExecType.CP, TestType.FUSE_NO_REDUNDANCY);
	}

	@Test
	public void testKMeansSparseMulSingleCPFuseNoRedundancy() {
		runKMeansTest(TEST_NAME1, false, true, 20, 1, ExecType.CP, TestType.FUSE_NO_REDUNDANCY);
	}

	@Test
	public void testKMeansDenseMulMultiRewritesCPFuseNoRedundancy() {
		runKMeansTest(TEST_NAME1, true, false, 20, 10, ExecType.CP, TestType.FUSE_NO_REDUNDANCY);
	}

	@Test
	public void testKMeansSparseMulMultiRewritesCPFuseNoRedundancy() {
		runKMeansTest(TEST_NAME1, true, true, 20, 10, ExecType.CP, TestType.FUSE_NO_REDUNDANCY);
	}

	@Test
	public void testKMeansDenseMulMultiCPFuseNoRedundancy() {
		runKMeansTest(TEST_NAME1, false, false, 20, 10, ExecType.CP, TestType.FUSE_NO_REDUNDANCY);
	}

	@Test
	public void testKMeansSparseMulMultiCPFuseNoRedundancy() {
		runKMeansTest(TEST_NAME1, false, true, 20, 10, ExecType.CP, TestType.FUSE_NO_REDUNDANCY);
	}
	
	private void runKMeansTest( String testname, boolean rewrites, boolean sparse, int centroids, int runs, ExecType instType, TestType testType)
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
			
			fullDMLScriptName = "scripts/algorithms/Kmeans.dml";
			programArgs = new String[]{ "-explain", "-stats",
				"-nvargs", "X="+input("X"), "k="+String.valueOf(centroids), "runs="+String.valueOf(runs), 
				"tol="+String.valueOf(epsilon), "maxi="+String.valueOf(maxiter), "C="+output("C")};

			//rCmd = getRCmd(inputDir(), String.valueOf(intercept),String.valueOf(epsilon),
			//	String.valueOf(maxiter), expectedDir());

			OptimizerUtils.ALLOW_ALGEBRAIC_SIMPLIFICATION = rewrites;
			
			//generate actual datasets
			double[][] X = getRandomMatrix(rows, cols, 0, 1, sparse?sparsity2:sparsity1, 714);
			writeInputMatrixWithMTD("X", X, true);
			
			runTest(true, false, null, -1); 
			
			Assert.assertTrue(heavyHittersContainsSubString("spoofCell") || heavyHittersContainsSubString("sp_spoofCell"));
			Assert.assertTrue(heavyHittersContainsSubString("spoofRA") || heavyHittersContainsSubString("sp_spoofRA"));
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
