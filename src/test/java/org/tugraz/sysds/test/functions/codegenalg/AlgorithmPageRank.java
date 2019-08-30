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

public class AlgorithmPageRank extends AutomatedTestBase 
{
	private final static String TEST_NAME1 = "Algorithm_PageRank";
	private final static String TEST_DIR = "functions/codegenalg/";
	private final static String TEST_CLASS_DIR = TEST_DIR + AlgorithmPageRank.class.getSimpleName() + "/";
	private final static String TEST_CONF_DEFAULT = "SystemDS-config-codegen.xml";
	private final static File TEST_CONF_FILE_DEFAULT = new File(SCRIPT_DIR + TEST_DIR, TEST_CONF_DEFAULT);
	private final static String TEST_CONF_FUSE_ALL = "SystemDS-config-codegen-fuse-all.xml";
	private final static File TEST_CONF_FILE_FUSE_ALL = new File(SCRIPT_DIR + TEST_DIR, TEST_CONF_FUSE_ALL);
	private final static String TEST_CONF_FUSE_NO_REDUNDANCY = "SystemDS-config-codegen-fuse-no-redundancy.xml";
	private final static File TEST_CONF_FILE_FUSE_NO_REDUNDANCY = new File(SCRIPT_DIR + TEST_DIR, TEST_CONF_FUSE_NO_REDUNDANCY);

	private enum TestType { DEFAULT,FUSE_ALL,FUSE_NO_REDUNDANCY }

	//absolute diff for large output scale in the +E12
	private final static double eps = 0.1;
	
	private final static int rows = 1468;
	private final static int cols = 1468;
	
	private final static double sparsity1 = 0.41; //dense
	private final static double sparsity2 = 0.05; //sparse
	
	private final static double alpha = 0.85;
	private final static double maxiter = 10;
	
	private TestType currentTestType = TestType.DEFAULT;
	
	@Override
	public void setUp() {
		TestUtils.clearAssertionInformation();
		addTestConfiguration(TEST_NAME1, new TestConfiguration(TEST_CLASS_DIR, TEST_NAME1, new String[] { "w" })); 
	}

	@Test
	public void testPageRankDenseCP() {
		runPageRankTest(TEST_NAME1, true, false, ExecType.CP, TestType.DEFAULT);
	}
	
	@Test
	public void testPageRankSparseCP() {
		runPageRankTest(TEST_NAME1, true, true, ExecType.CP, TestType.DEFAULT);
	}

	@Test
	public void testPageRankDenseCPFuseAll() {
		runPageRankTest(TEST_NAME1, true, false, ExecType.CP, TestType.FUSE_ALL);
	}

	@Test
	public void testPageRankSparseCPFuseAll() {
		runPageRankTest(TEST_NAME1, true, true, ExecType.CP, TestType.FUSE_ALL);
	}

	@Test
	public void testPageRankDenseCPFuseNoRedundancy() {
		runPageRankTest(TEST_NAME1, true, false, ExecType.CP, TestType.FUSE_NO_REDUNDANCY);
	}

	@Test
	public void testPageRankSparseCPFuseNoRedundancy() {
		runPageRankTest(TEST_NAME1, true, true, ExecType.CP, TestType.FUSE_NO_REDUNDANCY);
	}
	
	@Test
	public void testPageRankDenseCPNoR() {
		runPageRankTest(TEST_NAME1, false, false, ExecType.CP, TestType.DEFAULT);
	}
	
	@Test
	public void testPageRankSparseCPNoR() {
		runPageRankTest(TEST_NAME1, false, true, ExecType.CP, TestType.DEFAULT);
	}

	@Test
	public void testPageRankDenseCPFuseAllNoR() {
		runPageRankTest(TEST_NAME1, false, false, ExecType.CP, TestType.FUSE_ALL);
	}

	@Test
	public void testPageRankSparseCPFuseAllNoR() {
		runPageRankTest(TEST_NAME1, false, true, ExecType.CP, TestType.FUSE_ALL);
	}

	@Test
	public void testPageRankDenseCPFuseNoRedundancyNoR() {
		runPageRankTest(TEST_NAME1, false, false, ExecType.CP, TestType.FUSE_NO_REDUNDANCY);
	}

	@Test
	public void testPageRankSparseCPFuseNoRedundancyNoR() {
		runPageRankTest(TEST_NAME1, false, true, ExecType.CP, TestType.FUSE_NO_REDUNDANCY);
	}

	private void runPageRankTest( String testname, boolean rewrites, boolean sparse, ExecType instType, TestType testType)
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
			
			fullDMLScriptName = "scripts/staging/PageRank.dml";
			programArgs = new String[]{ "-explain", "-stats", "-args", input("G"), 
				input("p"), input("e"), input("u"), String.valueOf(alpha), 
				String.valueOf(maxiter), output("p")};
			rCmd = getRCmd(inputDir(), String.valueOf(alpha),
				String.valueOf(maxiter), expectedDir());

			OptimizerUtils.ALLOW_ALGEBRAIC_SIMPLIFICATION = rewrites;
			OptimizerUtils.ALLOW_OPERATOR_FUSION = rewrites;
			
			//generate actual datasets
			double[][] G = getRandomMatrix(rows, cols, 1, 1, sparse?sparsity2:sparsity1, 234);
			writeInputMatrixWithMTD("G", G, true);
			writeInputMatrixWithMTD("p", getRandomMatrix(cols, 1, 0, 1e-14, 1, 71), true);
			writeInputMatrixWithMTD("e", getRandomMatrix(rows, 1, 0, 1e-14, 1, 72), true);
			writeInputMatrixWithMTD("u", getRandomMatrix(1, cols, 0, 1e-14, 1, 73), true);
			
			runTest(true, false, null, -1);
			runRScript(true); 
			
			//compare matrices 
			HashMap<CellIndex, Double> dml = readDMLMatrixFromHDFS("p");
			HashMap<CellIndex, Double> r = readRMatrixFromFS("p");
			TestUtils.compareMatrices(dml, r, eps, "Stat-DML", "Stat-R");
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
