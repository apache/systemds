package org.apache.sysds.test.functions.builtin;

import org.apache.sysds.common.Types;
import org.apache.sysds.lops.LopProperties;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;
import org.apache.sysds.runtime.matrix.data.MatrixValue;
import org.apache.sysds.test.AutomatedTestBase;
import org.apache.sysds.test.TestConfiguration;
import org.apache.sysds.test.TestUtils;
import org.junit.Test;

import java.util.HashMap;

public class BuiltinGMMTest extends AutomatedTestBase {
	private final static String TEST_NAME = "GMM";
	private final static String TEST_DIR = "functions/builtin/";
	private static final String TEST_CLASS_DIR = TEST_DIR + BuiltinGMMTest.class.getSimpleName() + "/";

	private final static double eps = 1;
	private final static int rows = 100;
	private final static double spDense = 0.99;

	@Override
	public void setUp() {
		addTestConfiguration(TEST_NAME,new TestConfiguration(TEST_CLASS_DIR, TEST_NAME,new String[]{"B"}));
	}

	@Test
	public void testGMMCP() {
		runGMMTest(3,3,"VVI", LopProperties.ExecType.CP);
	}



	private void runGMMTest(int G_mixtures, int iter, String model,  LopProperties.ExecType instType)
	{
		Types.ExecMode platformOld = setExecMode(instType);

		try
		{
			loadTestConfiguration(getTestConfiguration(TEST_NAME));
			String HOME = SCRIPT_DIR + TEST_DIR;
			fullDMLScriptName = HOME + TEST_NAME + ".dml";
			programArgs = new String[]{ "-args", input("A"), String.valueOf(G_mixtures), String.valueOf(iter), model, String.valueOf("0.00000001"),output("B") };

			fullRScriptName = HOME + TEST_NAME + ".R";
			rCmd = "Rscript" + " " + fullRScriptName + " " + inputDir()+ " " + model + " "  +String.valueOf(G_mixtures)+ " "+ expectedDir();

			//generate actual dataset
			double[][] X = getRandomMatrix(rows, 2, 0.5, 1, 1.0, 714);
			writeInputMatrixWithMTD("A", X, true);

			runTest(true, false, null, -1);
			runRScript(true);

			//compare matrices
			HashMap<MatrixValue.CellIndex, Double> dmlfile = readDMLMatrixFromHDFS("B");
			HashMap<MatrixValue.CellIndex, Double> rfile  = readRMatrixFromFS("B");
			TestUtils.compareMatrices(dmlfile, rfile, eps, "Stat-DML", "Stat-R");
		}
		finally {
			rtplatform = platformOld;
		}
	}


}
