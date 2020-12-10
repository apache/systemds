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
    private final static int rows = 50;
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
            programArgs = new String[]{"-args", input("A"), input("B"), " ", output("C") };
            fullRScriptName = HOME + TEST_NAME + ".R";
            rCmd = "Rscript" + " " + fullRScriptName + " " + inputDir() + " "  + expectedDir();

            //generate actual dataset
            double[][] A = getRandomMatrix(rows, cols, 0, 100, 1.0, 7);
            writeInputMatrixWithMTD("A", A, true);
            double[][] B = getRandomMatrix(rows, 1, 0, 1, 1.0, 3);
            for (int row = 0; row < rows; row++) {
                B[row][0] = (B[row][0] > 0.5)? 1.0 : 0.0;
            }
            writeInputMatrixWithMTD("B", B, true);

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