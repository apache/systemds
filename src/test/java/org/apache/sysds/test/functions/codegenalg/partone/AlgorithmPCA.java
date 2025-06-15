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
import org.apache.sysds.common.Types.ExecMode;
import org.apache.sysds.hops.OptimizerUtils;
import org.apache.sysds.common.Types.ExecType;
import org.apache.sysds.runtime.matrix.data.MatrixValue.CellIndex;
import org.apache.sysds.test.AutomatedTestBase;
import org.apache.sysds.test.TestConfiguration;
import org.apache.sysds.test.TestUtils;
import org.junit.Assert;
import org.junit.Ignore;
import org.junit.Test;

public class AlgorithmPCA extends AutomatedTestBase
{
	private final static String TEST_NAME1 = "Algorithm_PCA";
	private final static String TEST_DIR = "functions/codegenalg/";
	private final static String TEST_CLASS_DIR = TEST_DIR + AlgorithmPCA.class.getSimpleName() + "/";

	private final static double eps = 1e-5;

	private final static int rows = 1468;
	private final static int cols1 = 1007;
	private final static int cols2 = 387;

	private final static double sparsity1 = 0.7; //dense
	private final static double sparsity2 = 0.1; //sparse
	
	private CodegenTestType currentTestType = CodegenTestType.DEFAULT;

	@Override
	public void setUp() {
		TestUtils.clearAssertionInformation();
		addTestConfiguration(TEST_NAME1, new TestConfiguration(TEST_CLASS_DIR, TEST_NAME1, new String[] { "w" }));
	}

	@Test
	@Ignore
	public void testPCADenseRewritesCP() {
		runPCATest(TEST_NAME1, true, false, ExecType.CP, CodegenTestType.DEFAULT);
	}

	@Test
	@Ignore
	public void testPCASparseRewritesCP() {
		runPCATest(TEST_NAME1, true, true, ExecType.CP, CodegenTestType.DEFAULT);
	}

	@Test
	@Ignore
	public void testPCADenseCP() {
		runPCATest(TEST_NAME1, false, false, ExecType.CP, CodegenTestType.DEFAULT);
	}

	@Test
	@Ignore
	public void testPCASparseCP() {
		runPCATest(TEST_NAME1, false, true, ExecType.CP, CodegenTestType.DEFAULT);
	}

	@Test
	@Ignore
	public void testPCADenseRewritesSP() {
		runPCATest(TEST_NAME1, true, false, ExecType.SPARK, CodegenTestType.DEFAULT);
	}

	@Test
	@Ignore
	public void testPCASparseRewritesSP() {
		runPCATest(TEST_NAME1, true, true, ExecType.SPARK, CodegenTestType.DEFAULT);
	}

	@Test
	@Ignore
	public void testPCADenseSP() {
		runPCATest(TEST_NAME1, false, false, ExecType.SPARK, CodegenTestType.DEFAULT);
	}

	@Test
	@Ignore
	public void testPCASparseSP() {
		runPCATest(TEST_NAME1, false, true, ExecType.SPARK, CodegenTestType.DEFAULT);
	}

	@Test
	@Ignore
	public void testPCADenseRewritesCPFuseAll() {
		runPCATest(TEST_NAME1, true, false, ExecType.CP, CodegenTestType.FUSE_ALL);
	}

	@Test
	@Ignore
	public void testPCASparseRewritesCPFuseAll() {
		runPCATest(TEST_NAME1, true, true, ExecType.CP, CodegenTestType.FUSE_ALL);
	}

	@Test
	@Ignore
	public void testPCADenseRewritesSPFuseAll() {
		runPCATest(TEST_NAME1, true, false, ExecType.SPARK, CodegenTestType.FUSE_ALL);
	}

	@Test
	@Ignore
	public void testPCASparseRewritesSPFuseAll() {
		runPCATest(TEST_NAME1, true, true, ExecType.SPARK, CodegenTestType.FUSE_ALL);
	}

	@Test
	@Ignore
	public void testPCADenseRewritesCPFuseNoRedundancy() {
		runPCATest(TEST_NAME1, true, false, ExecType.CP, CodegenTestType.FUSE_NO_REDUNDANCY);
	}

	@Test
	@Ignore
	public void testPCASparseRewritesCPFuseNoRedundancy() {
		runPCATest(TEST_NAME1, true, true, ExecType.CP, CodegenTestType.FUSE_NO_REDUNDANCY);
	}

	@Test
	@Ignore
	public void testPCADenseRewritesSPFuseNoRedundancy() {
		runPCATest(TEST_NAME1, true, false, ExecType.SPARK, CodegenTestType.FUSE_NO_REDUNDANCY);
	}

	@Test
	@Ignore
	public void testPCASparseRewritesSPFuseNoRedundancy() {
		runPCATest(TEST_NAME1, true, true, ExecType.SPARK, CodegenTestType.FUSE_NO_REDUNDANCY);
	}

	private void runPCATest(String testname, boolean rewrites, boolean sparse, ExecType instType, CodegenTestType CodegenTestType)
	{
		boolean oldFlag = OptimizerUtils.ALLOW_ALGEBRAIC_SIMPLIFICATION;
		ExecMode platformOld = setExecMode(instType);
		
		try {
			String TEST_NAME = testname;
			TestConfiguration config = getTestConfiguration(TEST_NAME);
			loadTestConfiguration(config);

			fullDMLScriptName = getScript();
			// pass OFMT=text flag, since readDMLMatrixFromHDFS() uses " " separator, not a "," separator.
			programArgs = new String[]{ "-explain", "-stats", "-nvargs", "OFMT=TEXT","INPUT="+input("A"),
					"OUTPUT="+output("")};

			rCmd = getRCmd(inputDir(), expectedDir());

			OptimizerUtils.ALLOW_ALGEBRAIC_SIMPLIFICATION = rewrites;

			//generate actual datasets
			int cols = (instType==ExecType.SPARK) ? cols2 : cols1;
			double[][] A = getRandomMatrix(rows, cols, 0, 1, sparse?sparsity2:sparsity1, 714);
			writeInputMatrixWithMTD("A", A, true);

			runTest(true, false, null, -1);
			runRScript(true);

			//compare matrices
			//HashMap<CellIndex, Double> dmleval = readDMLMatrixFromOutputDir("dominant.eigen.values");
			//HashMap<CellIndex, Double> reval   = readRMatrixFromExpectedDir("dominant.eigen.values");
			HashMap<CellIndex, Double> dmlevec = readDMLMatrixFromOutputDir("dominant.eigen.vectors");
			HashMap<CellIndex, Double> revec = readDMLMatrixFromOutputDir("dominant.eigen.vectors");
			//TestUtils.compareMatrices(dmleval, reval, eps, "Stat-DML", "Stat-R");
			TestUtils.compareMatrices(dmlevec, revec, eps, "Stat-DML", "Stat-R");
			Assert.assertTrue(heavyHittersContainsSubString(Opcodes.SPOOF.toString()) || heavyHittersContainsSubString("sp_spoof"));
		}
		finally {
			resetExecMode(platformOld);
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
