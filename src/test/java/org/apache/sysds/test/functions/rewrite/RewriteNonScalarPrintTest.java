package org.apache.sysds.test.functions.rewrite;

import org.apache.sysds.hops.OptimizerUtils;
import org.apache.sysds.runtime.matrix.data.MatrixValue;
import org.apache.sysds.test.AutomatedTestBase;
import org.apache.sysds.test.TestConfiguration;
import org.apache.sysds.test.TestUtils;
import org.junit.Test;

import java.util.HashMap;

public class RewriteNonScalarPrintTest extends AutomatedTestBase {

	private static final String TEST_NAME = "RewriteNonScalarPrint";
	private static final String TEST_DIR = "functions/rewrite/";

	private static final String TEST_CLASS_DIR =
		TEST_DIR + RewriteNonScalarPrintTest.class.getSimpleName() + "/";

	private static final double eps = Math.pow(10, -10);

	@Override
	public void setUp() {
		TestUtils.clearAssertionInformation();
		addTestConfiguration(TEST_NAME, new TestConfiguration(TEST_CLASS_DIR, TEST_NAME, new String[] {"R"}));
	}

	@Test
	public void testNonScalarPrint() {
		testRewriteNonScalarPrint(true);
	}

	private void testRewriteNonScalarPrint(boolean rewrites){
		boolean oldFlag = OptimizerUtils.ALLOW_ALGEBRAIC_SIMPLIFICATION;
		try{
			TestConfiguration config = getTestConfiguration(TEST_NAME);
			loadTestConfiguration(config);

			String HOME = SCRIPT_DIR + TEST_DIR;
			fullDMLScriptName = HOME + TEST_NAME + ".dml";
			programArgs = new String[] {"-stats", "-args", output("R")};
			fullRScriptName = HOME + TEST_NAME + ".R";
			rCmd = getRCmd(inputDir(), expectedDir());

			OptimizerUtils.ALLOW_ALGEBRAIC_SIMPLIFICATION = rewrites;

			runTest(true, false, null, -1);
			runRScript(true);

			//compare matrices
			HashMap<MatrixValue.CellIndex, Double> dmlfile = readDMLMatrixFromOutputDir("R");
			HashMap<MatrixValue.CellIndex, Double> rfile = readRMatrixFromExpectedDir("R");
			TestUtils.compareMatrices(dmlfile, rfile, eps, "Stat-DML", "Stat-R");

		} finally {
			OptimizerUtils.ALLOW_ALGEBRAIC_SIMPLIFICATION = oldFlag;
		}
	}
}
