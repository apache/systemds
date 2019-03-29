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

import org.junit.Test;
import org.apache.sysds.api.DMLScript;
import org.apache.sysds.common.Types.ExecMode;
import org.apache.sysds.hops.OptimizerUtils;
import org.apache.sysds.lops.LopProperties.ExecType;
import org.apache.sysds.runtime.matrix.data.MatrixValue.CellIndex;
import org.apache.sysds.test.AutomatedTestBase;
import org.apache.sysds.test.TestConfiguration;
import org.apache.sysds.test.TestUtils;
import org.junit.Assert;

public class AlgorithmPCA extends AutomatedTestBase
{
	private final static String TEST_NAME1 = "Algorithm_PCA";
	private final static String TEST_DIR = "functions/codegenalg/";
	private final static String TEST_CLASS_DIR = TEST_DIR + AlgorithmPCA.class.getSimpleName() + "/";
	private final static String TEST_CONF_DEFAULT = "SystemDS-config-codegen.xml";
	private final static File TEST_CONF_FILE_DEFAULT = new File(SCRIPT_DIR + TEST_DIR, TEST_CONF_DEFAULT);
	private final static String TEST_CONF_FUSE_ALL = "SystemDS-config-codegen-fuse-all.xml";
	private final static File TEST_CONF_FILE_FUSE_ALL = new File(SCRIPT_DIR + TEST_DIR, TEST_CONF_FUSE_ALL);
	private final static String TEST_CONF_FUSE_NO_REDUNDANCY = "SystemDS-config-codegen-fuse-no-redundancy.xml";
	private final static File TEST_CONF_FILE_FUSE_NO_REDUNDANCY = new File(SCRIPT_DIR + TEST_DIR,
			TEST_CONF_FUSE_NO_REDUNDANCY);

	private enum TestType { DEFAULT, FUSE_ALL, FUSE_NO_REDUNDANCY }

	private final static double eps = 1e-5;

	private final static int rows = 3468;
	private final static int cols1 = 1007;
	private final static int cols2 = 987;

	private final static double sparsity1 = 0.7; //dense
	private final static double sparsity2 = 0.1; //sparse

	private TestType currentTestType = TestType.DEFAULT;

	@Override
	public void setUp() {
		TestUtils.clearAssertionInformation();
		addTestConfiguration(TEST_NAME1, new TestConfiguration(TEST_CLASS_DIR, TEST_NAME1, new String[] { "w" }));
	}

	@Test
	public void testPCADenseRewritesCP() {
		runPCATest(TEST_NAME1, true, false, ExecType.CP, TestType.DEFAULT);
	}

	@Test
	public void testPCASparseRewritesCP() {
		runPCATest(TEST_NAME1, true, true, ExecType.CP, TestType.DEFAULT);
	}

	@Test
	public void testPCADenseCP() {
		runPCATest(TEST_NAME1, false, false, ExecType.CP, TestType.DEFAULT);
	}

	@Test
	public void testPCASparseCP() {
		runPCATest(TEST_NAME1, false, true, ExecType.CP, TestType.DEFAULT);
	}

	@Test
	public void testPCADenseRewritesSP() {
		runPCATest(TEST_NAME1, true, false, ExecType.SPARK, TestType.DEFAULT);
	}

	@Test
	public void testPCASparseRewritesSP() {
		runPCATest(TEST_NAME1, true, true, ExecType.SPARK, TestType.DEFAULT);
	}

	@Test
	public void testPCADenseSP() {
		runPCATest(TEST_NAME1, false, false, ExecType.SPARK, TestType.DEFAULT);
	}

	@Test
	public void testPCASparseSP() {
		runPCATest(TEST_NAME1, false, true, ExecType.SPARK, TestType.DEFAULT);
	}

	@Test
	public void testPCADenseRewritesCPFuseAll() {
		runPCATest(TEST_NAME1, true, false, ExecType.CP, TestType.FUSE_ALL);
	}

	@Test
	public void testPCASparseRewritesCPFuseAll() {
		runPCATest(TEST_NAME1, true, true, ExecType.CP, TestType.FUSE_ALL);
	}

	@Test
	public void testPCADenseRewritesSPFuseAll() {
		runPCATest(TEST_NAME1, true, false, ExecType.SPARK, TestType.FUSE_ALL);
	}

	@Test
	public void testPCASparseRewritesSPFuseAll() {
		runPCATest(TEST_NAME1, true, true, ExecType.SPARK, TestType.FUSE_ALL);
	}

	@Test
	public void testPCADenseRewritesCPFuseNoRedundancy() {
		runPCATest(TEST_NAME1, true, false, ExecType.CP, TestType.FUSE_NO_REDUNDANCY);
	}

	@Test
	public void testPCASparseRewritesCPFuseNoRedundancy() {
		runPCATest(TEST_NAME1, true, true, ExecType.CP, TestType.FUSE_NO_REDUNDANCY);
	}

	@Test
	public void testPCADenseRewritesSPFuseNoRedundancy() {
		runPCATest(TEST_NAME1, true, false, ExecType.SPARK, TestType.FUSE_NO_REDUNDANCY);
	}

	@Test
	public void testPCASparseRewritesSPFuseNoRedundancy() {
		runPCATest(TEST_NAME1, true, true, ExecType.SPARK, TestType.FUSE_NO_REDUNDANCY);
	}

	private void runPCATest( String testname, boolean rewrites, boolean sparse, ExecType instType, TestType testType)
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

			fullDMLScriptName = "scripts/algorithms/PCA.dml";
			// pass OFMT=text flag, since readDMLMatrixFromHDFS() uses " " separator, not a "," separator.
			programArgs = new String[]{ "-explain", "-stats", "-nvargs", "OFMT=TEXT","INPUT="+input("A"),
					"OUTPUT="+output("")};

			rCmd = getRCmd(inputDir(), expectedDir());

			OptimizerUtils.ALLOW_ALGEBRAIC_SIMPLIFICATION = rewrites;

			//generate actual datasets
			int cols = (instType==ExecType.SPARK) ? cols2 : cols1;
			double[][] A = getRandomMatrix(rows, cols, 0, 1, sparse?sparsity1:sparsity1, 714);
			System.out.println(A);
			writeInputMatrixWithMTD("A", A, true);

			runTest(true, false, null, -1);
			runRScript(true);

			//compare matrices
			HashMap<CellIndex, Double> dmleval = readDMLMatrixFromHDFS("dominant.eigen.values");
			HashMap<CellIndex, Double> reval   = readRMatrixFromFS("dominant.eigen.values");
			HashMap<CellIndex, Double> dmlevec = readDMLMatrixFromHDFS("dominant.eigen.vectors");
			HashMap<CellIndex, Double> revec = readDMLMatrixFromHDFS("dominant.eigen.vectors");
			HashMap<CellIndex, Double> dmlstd = readDMLMatrixFromHDFS("dominant.eigen.standard.deviations");
			HashMap<CellIndex, Double> rstd   = readRMatrixFromFS("dominant.eigen.standard.deviations");
			TestUtils.compareMatrices(dmleval, reval, eps, "Stat-DML", "Stat-R");
			TestUtils.compareMatrices(dmlevec, revec, eps, "Stat-DML", "Stat-R");
			TestUtils.compareMatrices(dmlstd, rstd, eps, "Stat-DML", "Stat-R");
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
		String message = "This test case overrides default configuration with ";
		if(currentTestType == AlgorithmPCA.TestType.FUSE_ALL){
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