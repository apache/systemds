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

package org.apache.sysds.test.functions.quaternary;

import java.util.HashMap;

import org.apache.sysds.common.Opcodes;
import org.junit.AfterClass;
import org.junit.Assert;
import org.junit.BeforeClass;
import org.junit.Test;
import org.apache.sysds.api.DMLScript;
import org.apache.sysds.common.Types.ExecMode;
import org.apache.sysds.hops.OptimizerUtils;
import org.apache.sysds.hops.QuaternaryOp;
import org.apache.sysds.lops.WeightedSquaredLoss;
import org.apache.sysds.common.Types.ExecType;
import org.apache.sysds.runtime.instructions.Instruction;
import org.apache.sysds.runtime.matrix.data.MatrixValue.CellIndex;
import org.apache.sysds.runtime.meta.MatrixCharacteristics;
import org.apache.sysds.test.AutomatedTestBase;
import org.apache.sysds.test.TestConfiguration;
import org.apache.sysds.test.TestUtils;
import org.apache.sysds.utils.Statistics;

/**
 * 
 * 
 */
public class WeightedSquaredLossTest extends AutomatedTestBase {
	private final static String TEST_NAME1 = "WeightedSquaredLossPost";
	private final static String TEST_NAME2 = "WeightedSquaredLossPre";
	private final static String TEST_NAME3 = "WeightedSquaredLossNo";
	private final static String TEST_NAME4 = "WeightedSquaredLossPost2";
	private final static String TEST_NAME5 = "WeightedSquaredLossPre2";
	private final static String TEST_NAME6 = "WeightedSquaredLossNo2";
	private final static String TEST_NAME7 = "WeightedSquaredLossPostNz";

	private final static String TEST_DIR = "functions/quaternary/";
	private final static String TEST_CLASS_DIR = TEST_DIR + WeightedSquaredLossTest.class.getSimpleName() + "/";

	private final static double eps = 1e-6;

	private final static int rows = 1201;
	private final static int cols = 1103;
	private final static int rank = 10;
	private final static double spSparse = 0.001;
	private final static double spDense = 0.6;

	@Override
	public void setUp() {
		addTestConfiguration(TEST_NAME1, new TestConfiguration(TEST_CLASS_DIR, TEST_NAME1, new String[] {"R"}));
		addTestConfiguration(TEST_NAME2, new TestConfiguration(TEST_CLASS_DIR, TEST_NAME2, new String[] {"R"}));
		addTestConfiguration(TEST_NAME3, new TestConfiguration(TEST_CLASS_DIR, TEST_NAME3, new String[] {"R"}));
		addTestConfiguration(TEST_NAME4, new TestConfiguration(TEST_CLASS_DIR, TEST_NAME4, new String[] {"R"}));
		addTestConfiguration(TEST_NAME5, new TestConfiguration(TEST_CLASS_DIR, TEST_NAME5, new String[] {"R"}));
		addTestConfiguration(TEST_NAME6, new TestConfiguration(TEST_CLASS_DIR, TEST_NAME6, new String[] {"R"}));
		addTestConfiguration(TEST_NAME7, new TestConfiguration(TEST_CLASS_DIR, TEST_NAME7, new String[] {"R"}));

		if(TEST_CACHE_ENABLED) {
			setOutAndExpectedDeletionDisabled(true);
		}
	}

	@BeforeClass
	public static void init() {
		TestUtils.clearDirectory(TEST_DATA_DIR + TEST_CLASS_DIR);
	}

	@AfterClass
	public static void cleanUp() {
		if(TEST_CACHE_ENABLED) {
			TestUtils.clearDirectory(TEST_DATA_DIR + TEST_CLASS_DIR);
		}
	}

	@Test
	public void testSquaredLossDensePostWeightsNoRewritesCP() {
		runMLUnaryBuiltinTest(TEST_NAME1, false, false, ExecType.CP);
	}

	@Test
	public void testSquaredLossDensePreWeightsNoRewritesCP() {
		runMLUnaryBuiltinTest(TEST_NAME2, false, false, ExecType.CP);
	}

	@Test
	public void testSquaredLossDenseNoWeightsNoRewritesCP() {
		runMLUnaryBuiltinTest(TEST_NAME3, false, false, ExecType.CP);
	}

	@Test
	public void testSquaredLossSparsePostWeightsNoRewritesCP() {
		runMLUnaryBuiltinTest(TEST_NAME1, true, false, ExecType.CP);
	}

	@Test
	public void testSquaredLossSparsePreWeightsNoRewritesCP() {
		runMLUnaryBuiltinTest(TEST_NAME2, true, false, ExecType.CP);
	}

	@Test
	public void testSquaredLossSparseNoWeightsNoRewritesCP() {
		runMLUnaryBuiltinTest(TEST_NAME3, true, false, ExecType.CP);
	}

	@Test
	public void testSquaredLossDensePostWeightsRewritesCP() {
		runMLUnaryBuiltinTest(TEST_NAME1, false, true, ExecType.CP);
	}

	@Test
	public void testSquaredLossDensePreWeightsRewritesCP() {
		runMLUnaryBuiltinTest(TEST_NAME2, false, true, ExecType.CP);
	}

	@Test
	public void testSquaredLossDenseNoWeightsRewritesCP() {
		runMLUnaryBuiltinTest(TEST_NAME3, false, true, ExecType.CP);
	}

	@Test
	public void testSquaredLossSparsePostWeightsRewritesCP() {
		runMLUnaryBuiltinTest(TEST_NAME1, true, true, ExecType.CP);
	}

	@Test
	public void testSquaredLossSparsePreWeightsRewritesCP() {
		runMLUnaryBuiltinTest(TEST_NAME2, true, true, ExecType.CP);
	}

	@Test
	public void testSquaredLossSparseNoWeightsRewritesCP() {
		runMLUnaryBuiltinTest(TEST_NAME3, true, true, ExecType.CP);
	}

	@Test
	public void testSquaredLossDensePostWeightsRewritesSP() {
		runMLUnaryBuiltinTest(TEST_NAME1, false, true, ExecType.SPARK);
	}

	@Test
	public void testSquaredLossDensePreWeightsRewritesSP() {
		runMLUnaryBuiltinTest(TEST_NAME2, false, true, ExecType.SPARK);
	}

	@Test
	public void testSquaredLossDenseNoWeightsRewritesSP() {
		runMLUnaryBuiltinTest(TEST_NAME3, false, true, ExecType.SPARK);
	}

	@Test
	public void testSquaredLossSparsePostWeightsRewritesSP() {
		runMLUnaryBuiltinTest(TEST_NAME1, true, true, ExecType.SPARK);
	}

	@Test
	public void testSquaredLossSparsePreWeightsRewritesSP() {
		runMLUnaryBuiltinTest(TEST_NAME2, true, true, ExecType.SPARK);
	}

	@Test
	public void testSquaredLossSparseNoWeightsRewritesSP() {
		runMLUnaryBuiltinTest(TEST_NAME3, true, true, ExecType.SPARK);
	}

	@Test
	public void testSquaredLossDensePostNzWeightsRewritesCP() {
		runMLUnaryBuiltinTest(TEST_NAME7, false, true, ExecType.CP);
	}

	@Test
	public void testSquaredLossSparsePostNzWeightsRewritesCP() {
		runMLUnaryBuiltinTest(TEST_NAME7, true, true, ExecType.CP);
	}

	@Test
	public void testSquaredLossDensePostNzWeightsRewritesSP() {
		runMLUnaryBuiltinTest(TEST_NAME7, false, true, ExecType.SPARK);
	}

	@Test
	public void testSquaredLossSparsePostNzWeightsRewritesSP() {
		runMLUnaryBuiltinTest(TEST_NAME7, true, true, ExecType.SPARK);
	}

	// the following tests force the replication based mr operator because
	// otherwise we would always choose broadcasts for this small input data

	@Test
	public void testSquaredLossSparsePostWeightsRewritesRepSP() {
		runMLUnaryBuiltinTest(TEST_NAME1, true, true, true, ExecType.SPARK);
	}

	@Test
	public void testSquaredLossSparsePreWeightsRewritesRepSP() {
		runMLUnaryBuiltinTest(TEST_NAME2, true, true, true, ExecType.SPARK);
	}

	@Test
	public void testSquaredLossSparseNoWeightsRewritesRepSP() {
		runMLUnaryBuiltinTest(TEST_NAME3, true, true, true, ExecType.SPARK);
	}

	@Test
	public void testSquaredLossDensePostWeightsRewritesRepSP() {
		runMLUnaryBuiltinTest(TEST_NAME1, false, true, true, ExecType.SPARK);
	}

	@Test
	public void testSquaredLossDensePreWeightsRewritesRepSP() {
		runMLUnaryBuiltinTest(TEST_NAME2, false, true, true, ExecType.SPARK);
	}

	@Test
	public void testSquaredLossDenseNoWeightsRewritesRepSP() {
		runMLUnaryBuiltinTest(TEST_NAME3, false, true, true, ExecType.SPARK);
	}

	@Test
	public void testSquaredLossSparsePostNzWeightsRewritesRepSP() {
		runMLUnaryBuiltinTest(TEST_NAME7, true, true, true, ExecType.SPARK);
	}

	@Test
	public void testSquaredLossDensePostNzWeightsRewritesRepSP() {
		runMLUnaryBuiltinTest(TEST_NAME7, false, true, true, ExecType.SPARK);
	}

	// the following tests use a sightly different pattern of U%*%t(V)-X
	// which applies as well due to the subsequent squaring.

	@Test
	public void testSquaredLossDensePostWeightsRewrites2CP() {
		runMLUnaryBuiltinTest(TEST_NAME4, false, true, ExecType.CP);
	}

	@Test
	public void testSquaredLossDensePreWeightsRewrites2CP() {
		runMLUnaryBuiltinTest(TEST_NAME5, false, true, ExecType.CP);
	}

	@Test
	public void testSquaredLossDenseNoWeightsRewrites2CP() {
		runMLUnaryBuiltinTest(TEST_NAME6, false, true, ExecType.CP);
	}

	/**
	 * 
	 * @param testname
	 * @param sparse
	 * @param rewrites
	 * @param instType
	 */
	private void runMLUnaryBuiltinTest(String testname, boolean sparse, boolean rewrites, ExecType instType) {
		runMLUnaryBuiltinTest(testname, sparse, rewrites, false, instType);
	}

	private void runMLUnaryBuiltinTest(String testname, boolean sparse, boolean rewrites, boolean rep,
		ExecType instType) {
		ExecMode platformOld = rtplatform;
		switch(instType) {
			case SPARK:
				rtplatform = ExecMode.SPARK;
				break;
			default:
				rtplatform = ExecMode.HYBRID;
				break;
		}

		boolean sparkConfigOld = DMLScript.USE_LOCAL_SPARK_CONFIG;
		if(rtplatform == ExecMode.SPARK)
			DMLScript.USE_LOCAL_SPARK_CONFIG = true;

		boolean rewritesOld = OptimizerUtils.ALLOW_ALGEBRAIC_SIMPLIFICATION;
		boolean forceOld = QuaternaryOp.FORCE_REPLICATION;

		OptimizerUtils.ALLOW_ALGEBRAIC_SIMPLIFICATION = rewrites;
		QuaternaryOp.FORCE_REPLICATION = rep;

		try {
			double sparsity = (sparse) ? spSparse : spDense;
			String TEST_NAME = testname;

			TestConfiguration config = getTestConfiguration(TEST_NAME);

			String TEST_CACHE_DIR = "";
			if(TEST_CACHE_ENABLED) {
				TEST_CACHE_DIR = TEST_NAME + "_" + sparsity + "/";
			}

			loadTestConfiguration(config, TEST_CACHE_DIR);

			// This is for running the junit test the new way, i.e., construct the arguments directly
			String HOME = SCRIPT_DIR + TEST_DIR;
			fullDMLScriptName = HOME + TEST_NAME + ".dml";
			programArgs = new String[] {"-stats", "-explain", "runtime", "-args", input("X"), input("U"), input("V"),
				input("W"), output("R")};

			fullRScriptName = HOME + TEST_NAME + ".R";
			rCmd = "Rscript" + " " + fullRScriptName + " " + inputDir() + " " + expectedDir();

			// generate actual dataset
			double[][] X = getRandomMatrix(rows, cols, 0, 1, sparsity, 7);
			writeInputMatrixWithMTD("X", X, true);
			double[][] U = getRandomMatrix(rows, rank, 0, 1, 1.0, 213);
			writeInputMatrixWithMTD("U", U, true);
			double[][] V = getRandomMatrix(cols, rank, 0, 1, 1.0, 312);
			writeInputMatrixWithMTD("V", V, true);
			if(!TEST_NAME.equals(TEST_NAME3)) {
				double[][] W = getRandomMatrix(rows, cols, 0, 1, sparsity, 1467);
				writeInputMatrixWithMTD("W", W, true);
			}

			runTest(true, false, null, -1);
			runRScript(true);

			// compare matrices
			HashMap<CellIndex, Double> dmlfile = readDMLMatrixFromOutputDir("R");
			HashMap<CellIndex, Double> rfile = readRMatrixFromExpectedDir("R");
			TestUtils.compareMatrices(dmlfile, rfile, eps, "Stat-DML", "Stat-R");
			checkDMLMetaDataFile("R", new MatrixCharacteristics(1, 1, 1, 1));

			// check statistics for right operator in cp
			if(instType == ExecType.CP && rewrites)
				Assert.assertTrue("Rewrite not applied.",
					Statistics.getCPHeavyHitterOpCodes().contains(WeightedSquaredLoss.OPCODE_CP));
			else if(instType == ExecType.SPARK && rewrites) {
				boolean noWeights = testname.equals(TEST_NAME3) || testname.equals(TEST_NAME6) ||
					testname.equals(TEST_NAME7);
				String opcode = Instruction.SP_INST_PREFIX +
					((rep || !noWeights) ? Opcodes.WEIGHTEDSQUAREDLOSSR.toString() : Opcodes.WEIGHTEDSQUAREDLOSS.toString());
				Assert.assertTrue("Rewrite not applied.", Statistics.getCPHeavyHitterOpCodes().contains(opcode));
			}
		}
		finally {
			rtplatform = platformOld;
			DMLScript.USE_LOCAL_SPARK_CONFIG = sparkConfigOld;
			OptimizerUtils.ALLOW_ALGEBRAIC_SIMPLIFICATION = rewritesOld;
			QuaternaryOp.FORCE_REPLICATION = forceOld;
		}
	}
}
