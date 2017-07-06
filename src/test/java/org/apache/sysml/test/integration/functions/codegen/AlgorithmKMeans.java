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

public class AlgorithmKMeans extends AutomatedTestBase 
{	
	private final static String TEST_NAME1 = "Algorithm_KMeans";
	private final static String TEST_DIR = "functions/codegen/";
	private final static String TEST_CLASS_DIR = TEST_DIR + AlgorithmKMeans.class.getSimpleName() + "/";
	private final static String TEST_CONF = "SystemML-config-codegen.xml";
	private final static File   TEST_CONF_FILE = new File(SCRIPT_DIR + TEST_DIR, TEST_CONF);
	
	//private final static double eps = 1e-5;
	
	private final static int rows = 3972;
	private final static int cols = 972;
		
	private final static double sparsity1 = 0.7; //dense
	private final static double sparsity2 = 0.1; //sparse
	
	private final static double epsilon = 0.000000001;
	private final static double maxiter = 10;
	
	@Override
	public void setUp() {
		TestUtils.clearAssertionInformation();
		addTestConfiguration(TEST_NAME1, new TestConfiguration(TEST_CLASS_DIR, TEST_NAME1, new String[] { "C" })); 
	}

	@Test
	public void testKMeansDenseBinSingleRewritesCP() {
		runKMeansTest(TEST_NAME1, true, false, 2, 1, ExecType.CP);
	}
	
	@Test
	public void testKMeansSparseBinSingleRewritesCP() {
		runKMeansTest(TEST_NAME1, true, true, 2, 1, ExecType.CP);
	}
	
	@Test
	public void testKMeansDenseBinSingleCP() {
		runKMeansTest(TEST_NAME1, false, false, 2, 1, ExecType.CP);
	}
	
	@Test
	public void testKMeansSparseBinSingleCP() {
		runKMeansTest(TEST_NAME1, false, true, 2, 1, ExecType.CP);
	}
	
	@Test
	public void testKMeansDenseBinMultiRewritesCP() {
		runKMeansTest(TEST_NAME1, true, false, 2, 10, ExecType.CP);
	}
	
	@Test
	public void testKMeansSparseBinMultiRewritesCP() {
		runKMeansTest(TEST_NAME1, true, true, 2, 10, ExecType.CP);
	}
	
	@Test
	public void testKMeansDenseBinMultiCP() {
		runKMeansTest(TEST_NAME1, false, false, 2, 10, ExecType.CP);
	}
	
	@Test
	public void testKMeansSparseBinMultiCP() {
		runKMeansTest(TEST_NAME1, false, true, 2, 10, ExecType.CP);
	}
	
	@Test
	public void testKMeansDenseMulSingleRewritesCP() {
		runKMeansTest(TEST_NAME1, true, false, 20, 1, ExecType.CP);
	}
	
	@Test
	public void testKMeansSparseMulSingleRewritesCP() {
		runKMeansTest(TEST_NAME1, true, true, 20, 1, ExecType.CP);
	}
	
	@Test
	public void testKMeansDenseMulSingleCP() {
		runKMeansTest(TEST_NAME1, false, false, 20, 1, ExecType.CP);
	}
	
	@Test
	public void testKMeansSparseMulSingleCP() {
		runKMeansTest(TEST_NAME1, false, true, 20, 1, ExecType.CP);
	}
	
	@Test
	public void testKMeansDenseMulMultiRewritesCP() {
		runKMeansTest(TEST_NAME1, true, false, 20, 10, ExecType.CP);
	}
	
	@Test
	public void testKMeansSparseMulMultiRewritesCP() {
		runKMeansTest(TEST_NAME1, true, true, 20, 10, ExecType.CP);
	}
	
	@Test
	public void testKMeansDenseMulMultiCP() {
		runKMeansTest(TEST_NAME1, false, false, 20, 10, ExecType.CP);
	}
	
	@Test
	public void testKMeansSparseMulMultiCP() {
		runKMeansTest(TEST_NAME1, false, true, 20, 10, ExecType.CP);
	}
	
	private void runKMeansTest( String testname, boolean rewrites, boolean sparse, int centroids, int runs, ExecType instType)
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
