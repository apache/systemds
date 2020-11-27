package org.apache.sysds.test.functions.builtin;

import org.apache.sysds.common.Types;
import org.apache.sysds.lops.LopProperties;
import org.apache.sysds.runtime.matrix.data.MatrixValue;
import org.apache.sysds.test.AutomatedTestBase;
import org.apache.sysds.test.TestConfiguration;
import org.apache.sysds.test.TestUtils;
import org.junit.Test;

import java.util.HashMap;

public class BuiltinDecisionTreeTest extends AutomatedTestBase
{
    private final static String TEST_NAME = "decisionTree";
    private final static String TEST_DIR = "functions/builtin/";
    private static final String TEST_CLASS_DIR = TEST_DIR + BuiltinDistTest.class.getSimpleName() + "/";

    private final static double eps = 1e-3;
    private final static int rows = 1765;
    private final static double spDense = 0.99;

    @Override
    public void setUp() {
        addTestConfiguration(TEST_NAME,new TestConfiguration(TEST_CLASS_DIR, TEST_NAME,new String[]{"B"}));
    }

    @Test
    public void testDistDefaultCP() { runDist(true, LopProperties.ExecType.CP); }

    @Test
    public void testDistSP() {
        runDist(true, LopProperties.ExecType.SPARK);
    }

    private void runDist(boolean defaultProb, LopProperties.ExecType instType)
    {
        Types.ExecMode platformOld = setExecMode(instType);

        try
        {
            loadTestConfiguration(getTestConfiguration(TEST_NAME));

            String HOME = SCRIPT_DIR + TEST_DIR;
            fullDMLScriptName = HOME + TEST_NAME + ".dml";
            programArgs = new String[]{"-args", input("A"), output("B") };
            fullRScriptName = HOME + TEST_NAME + ".R";
            rCmd = "Rscript" + " " + fullRScriptName + " " + inputDir() + " " + expectedDir();

            //generate actual dataset
            double[][] A = getRandomMatrix(rows, 10, -1, 1, spDense, 7);
            writeInputMatrixWithMTD("A", A, true);

            runTest(true, false, null, -1);
            runRScript(true);

            //compare matrices
            HashMap<MatrixValue.CellIndex, Double> dmlfile = readDMLMatrixFromOutputDir("B");
            HashMap<MatrixValue.CellIndex, Double> rfile  = readRMatrixFromExpectedDir("B");
            TestUtils.compareMatrices(dmlfile, rfile, eps, "Stat-DML", "Stat-R");
        }
        finally {
            rtplatform = platformOld;
        }
    }
}