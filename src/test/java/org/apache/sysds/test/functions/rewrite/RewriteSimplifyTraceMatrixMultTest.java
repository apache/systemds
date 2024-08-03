package org.apache.sysds.test.functions.rewrite;

import org.apache.sysds.hops.OptimizerUtils;
import org.apache.sysds.runtime.matrix.data.MatrixValue;
import org.apache.sysds.test.AutomatedTestBase;
import org.apache.sysds.test.TestConfiguration;
import org.apache.sysds.test.TestUtils;
import org.apache.sysds.utils.Statistics;
import org.junit.Assert;
import org.junit.Test;

import java.util.HashMap;

public class RewriteSimplifyTraceMatrixMultTest extends AutomatedTestBase {
	private static final String TEST_NAME = "RewriteSimplifyTraceMatrixMult";
	private static final String TEST_DIR = "functions/rewrite/";
	private static final String TEST_CLASS_DIR =
		TEST_DIR + RewriteSimplifyTraceMatrixMultTest.class.getSimpleName() + "/";

	private static final int rows = 500;
	private static final int cols = 500;
	private static final double eps = Math.pow(10, -10);

	@Override
	public void setUp() {
		TestUtils.clearAssertionInformation();
		addTestConfiguration(TEST_NAME, new TestConfiguration(TEST_CLASS_DIR, TEST_NAME, new String[] {"R"}));
	}

	@Test
	public void testSimplifyTraceMatrixMultNoRewrite() {
		testRewriteTraceMatrixMult(TEST_NAME, false);
	}

	@Test
	public void testSimplifyTraceMatrixMultRewrite() {
		testRewriteTraceMatrixMult(TEST_NAME, true);
	}

	private void testRewriteTraceMatrixMult(String testname, boolean rewrites) {
		boolean oldFlag = OptimizerUtils.ALLOW_ALGEBRAIC_SIMPLIFICATION;
		try {
			TestConfiguration config = getTestConfiguration(testname);
			loadTestConfiguration(config);

			String HOME = SCRIPT_DIR + TEST_DIR;
			fullDMLScriptName = HOME + testname + ".dml";
			programArgs = new String[] {"-stats", "-args", input("A"), input("B"), output("R")};

			fullRScriptName = HOME + testname + ".R";
			rCmd = getRCmd(inputDir(), expectedDir());

			OptimizerUtils.ALLOW_ALGEBRAIC_SIMPLIFICATION = rewrites;
			//create dense matrices so that rewrites are possible
			double[][] A = getRandomMatrix(rows, cols, -1, 1, 0.70d, 7);
			double[][] B = getRandomMatrix(cols, rows, -1, 1, 0.70d, 6);
			writeInputMatrixWithMTD("A", A, true);
			writeInputMatrixWithMTD("B", B, true);

			runTest(true, false, null, -1);
			runRScript(true);

			//compare matrices
			HashMap<MatrixValue.CellIndex, Double> dmlfile = readDMLScalarFromOutputDir("R");
			HashMap<MatrixValue.CellIndex, Double> rfile = readRScalarFromExpectedDir("R");
			TestUtils.compareMatrices(dmlfile, rfile, eps, "Stat-DML", "Stat-R");

			//check trace operator existence
			String uaktrace = "uaktrace";
			long numTrace = Statistics.getCPHeavyHitterCount(uaktrace);

			if(rewrites)
				Assert.assertTrue(numTrace == 0);
			else
				Assert.assertTrue(numTrace == 1);

		}
		finally {
			OptimizerUtils.ALLOW_ALGEBRAIC_SIMPLIFICATION = oldFlag;
		}

	}
}
