package org.apache.sysds.test.functions.rewrite;

import org.apache.sysds.api.DMLScript;
import org.apache.sysds.common.Types;
import org.apache.sysds.hops.OptimizerUtils;
import org.apache.sysds.runtime.matrix.data.MatrixValue;
import org.apache.sysds.test.AutomatedTestBase;
import org.apache.sysds.test.TestConfiguration;
import org.apache.sysds.test.TestUtils;
import org.apache.sysds.utils.Statistics;
import org.junit.Assert;
import org.junit.Test;

import java.util.HashMap;

public class RewriteMatrixMultChainOptSparseTest extends AutomatedTestBase {

	private static final String TEST_NAME = "RewriteMatrixMultChainOpSparse";
	private static final String TEST_DIR = "functions/rewrite/";
	private static final String TEST_CLASS_DIR =
		TEST_DIR + RewriteMatrixMultChainOptSparseTest.class.getSimpleName() + "/";

	private static final int rows = 1000;
	private static final int cols = 300;
	private static final double eps = Math.pow(10, -10);

	@Override
	public void setUp() {
		TestUtils.clearAssertionInformation();
		addTestConfiguration(TEST_NAME, new TestConfiguration(TEST_CLASS_DIR, TEST_NAME, new String[] {"R"}));
	}

	@Test
	public void testMatrixMultChainOptSparseNoRewrites() {
		testRewriteMatrixMultChainOpSparse(false);
	}

	@Test
	public void testMatrixMultChainOptSparseRewrites() {
		testRewriteMatrixMultChainOpSparse(true);
	}

	private void testRewriteMatrixMultChainOpSparse(boolean rewrites) {
		boolean oldFlag1 = OptimizerUtils.ALLOW_ADVANCED_MMCHAIN_REWRITES;
		boolean oldFlag2 = OptimizerUtils.ALLOW_SUM_PRODUCT_REWRITES;

		try {
			TestConfiguration config = getTestConfiguration(TEST_NAME);
			loadTestConfiguration(config);

			String HOME = SCRIPT_DIR + TEST_DIR;
			fullDMLScriptName = HOME + TEST_NAME + ".dml";
			programArgs = new String[] {"-explain", "hops", "-stats", "-args", input("X"), input("Y"), output("R")};
			fullRScriptName = HOME + TEST_NAME + ".R";
			rCmd = getRCmd(inputDir(), expectedDir());

			OptimizerUtils.ALLOW_ADVANCED_MMCHAIN_REWRITES = rewrites;
			OptimizerUtils.ALLOW_SUM_PRODUCT_REWRITES = rewrites;
			double[][] X = getRandomMatrix(rows, cols, -1, 1, 0.10d, 7);
			double[][] Y = getRandomMatrix(cols, 1, -1, 1, 0.10d, 3);
			writeInputMatrixWithMTD("X", X, true);
			writeInputMatrixWithMTD("Y", Y, true);

			//execute tests
			runTest(true, false, null, -1);
			runRScript(true);

			//compare matrices
			HashMap<MatrixValue.CellIndex, Double> dmlfile = readDMLMatrixFromOutputDir("R");
			HashMap<MatrixValue.CellIndex, Double> rfile = readRMatrixFromExpectedDir("R");
			TestUtils.compareMatrices(dmlfile, rfile, eps, "Stat-DML", "Stat-R");

			if(rewrites) {
				Assert.assertTrue(heavyHittersContainsSubString("mmchain") ||
					heavyHittersContainsSubString("sp_mapmmchain"));
			}
			else {
				Assert.assertFalse(heavyHittersContainsSubString("mmchain") ||
						heavyHittersContainsSubString("sp_mapmmchain"));
			}

		}
		finally {
			OptimizerUtils.ALLOW_ADVANCED_MMCHAIN_REWRITES = oldFlag1;
			OptimizerUtils.ALLOW_SUM_PRODUCT_REWRITES = oldFlag2;
		}
	}

}
