package org.apache.sysds.test.functions.builtin;

import org.apache.sysds.common.Types;
import org.apache.sysds.lops.LopProperties;
import org.apache.sysds.test.AutomatedTestBase;
import org.apache.sysds.test.TestConfiguration;
import org.junit.Test;

public class BuiltinDecisionTreeTest extends AutomatedTestBase
{
    private final static String TEST_NAME = "decisionTree";
    private final static String TEST_DIR = "functions/builtin/";
    private static final String TEST_CLASS_DIR = TEST_DIR + BuiltinDecisionTreeTest.class.getSimpleName() + "/";

    private final static double eps = 1e-10;
    private final static int rows = 6;
    private final static int cols = 4;

    @Override
    public void setUp() {
        addTestConfiguration(TEST_NAME, new TestConfiguration(TEST_CLASS_DIR, TEST_NAME, new String[]{"C"}));
    }

    @Test
    public void testDecisionTreeDefaultCP() { runDecisionTree(true, LopProperties.ExecType.CP); }

    @Test
    public void testDecisionTreeSP() {
        runDecisionTree(true, LopProperties.ExecType.SPARK);
    }

    private void runDecisionTree(boolean defaultProb, LopProperties.ExecType instType)
    {
        Types.ExecMode platformOld = setExecMode(instType);

        try
        {
            loadTestConfiguration(getTestConfiguration(TEST_NAME));

            String HOME = SCRIPT_DIR + TEST_DIR;
            fullDMLScriptName = HOME + TEST_NAME + ".dml";
            programArgs = new String[]{"-args", input("X"), input("Y"), input("R"), output("M") };
            fullRScriptName = HOME + TEST_NAME + ".R";
            rCmd = "Rscript" + " " + fullRScriptName + " " + inputDir() + " "  + expectedDir();

            double[][] Y = getRandomMatrix(rows, 1, 0, 1, 1.0, 3);
            for (int row = 0; row < rows; row++) {
                Y[row][0] = (Y[row][0] > 0.5)? 1.0 : 0.0;
            }

            //generate actual dataset
            double[][] X = getRandomMatrix(rows, cols, 0, 100, 1.0, 7);
            for (int row = 0; row < rows/2; row++) {
                X[row][2] = (Y[row][0] > 0.5)? 2.0 : 1.0;
                X[row][3] = 1.0;
            }
            for (int row = rows/2; row < rows; row++) {
                X[row][2] = 1.0;
                X[row][3] = (Y[row][0] > 0.5)? 2.0 : 1.0;
            }
            writeInputMatrixWithMTD("X", X, true);
            writeInputMatrixWithMTD("Y", Y, true);



            double[][] R = getRandomMatrix(1, cols, 1, 1, 1.0, 1);
            R[0][3] = 3.0;
            R[0][2] = 3.0;
            writeInputMatrixWithMTD("R", R, true);

            runTest(true, false, null, -1);

//            runRScript(true);
//            HashMap<MatrixValue.CellIndex, Double> dmlfile = readDMLMatrixFromOutputDir("C");
//            HashMap<MatrixValue.CellIndex, Double> rfile  = readRMatrixFromExpectedDir("C");
//            TestUtils.compareMatrices(dmlfile, rfile, eps, "Stat-DML", "Stat-R");
        }
        finally {
            rtplatform = platformOld;
        }
    }
}