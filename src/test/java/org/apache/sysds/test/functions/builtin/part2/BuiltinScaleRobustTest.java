package org.apache.sysds.test.functions.builtin.part2;

import java.util.HashMap;

import org.junit.Assert;
import org.junit.Test;

import org.apache.sysds.common.Types.ExecMode;
import org.apache.sysds.common.Types.ExecType;
import org.apache.sysds.runtime.matrix.data.MatrixValue.CellIndex;
import org.apache.sysds.test.AutomatedTestBase;
import org.apache.sysds.test.TestConfiguration;
import org.apache.sysds.test.TestUtils;
import org.apache.sysds.utils.Statistics;

public class BuiltinScaleRobustTest extends AutomatedTestBase {
	private final static String TEST_NAME = "scaleRobust";
	private final static String TEST_DIR = "functions/builtin/";
	private final static String TEST_CLASS_DIR = TEST_DIR + BuiltinScaleRobustTest.class.getSimpleName() + "/";
	private final static double eps = 1e-10;
	private final static int rows = 70;
	private final static int cols = 50;

	@Override
	public void setUp() {
		addTestConfiguration(TEST_NAME, new TestConfiguration(TEST_CLASS_DIR, TEST_NAME, new String[]{"B"}));
	}

	@Test
	public void testScaleRobustDenseCP() {
		runTest(false, ExecType.CP);
	}

	private void runTest(boolean sparse, ExecType et) {
		ExecMode old = setExecMode(et);
		try {
			loadTestConfiguration(getTestConfiguration(TEST_NAME));
			double sparsity = sparse ? 0.1 : 0.9;
			String HOME = SCRIPT_DIR + TEST_DIR;
			fullDMLScriptName = HOME + TEST_NAME + ".dml";
			fullRScriptName = HOME + TEST_NAME + ".R";
			programArgs = new String[]{"-args", input("A"), output("B")};
			rCmd = "Rscript " + fullRScriptName + " " + inputDir() + " " + expectedDir();

			double[][] A = getRandomMatrix(rows, cols, -10, 10, sparsity, 7);
			writeInputMatrixWithMTD("A", A, true);

			runTest(true, false, null, -1);
			runRScript(true);

			HashMap<CellIndex, Double> dmlfile = readDMLMatrixFromOutputDir("B");
			HashMap<CellIndex, Double> rfile  = readRMatrixFromExpectedDir("B");
			TestUtils.compareMatrices(dmlfile, rfile, eps, "DML", "R");
		}
		finally {
			rtplatform = old;
		}
	}
}
