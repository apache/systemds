package org.apache.sysds.test.functions.builtin;

import org.apache.sysds.common.Types;
import org.apache.sysds.runtime.matrix.data.MatrixValue;
import org.apache.sysds.test.AutomatedTestBase;
import org.apache.sysds.test.TestConfiguration;
import org.apache.sysds.test.TestUtils;
import org.junit.Test;

import java.util.HashMap;

public class BuiltinDifferenceTest extends AutomatedTestBase {
    private final static String TEST_NAME = "difference";
    private final static String TEST_DIR = "functions/builtin/";
    private static final String TEST_CLASS_DIR = TEST_DIR + BuiltinDifferenceTest.class.getSimpleName() + "/";

    @Override
    public void setUp() {
        TestUtils.clearAssertionInformation();
        addTestConfiguration(TEST_NAME, new TestConfiguration(TEST_CLASS_DIR, TEST_NAME, new String[]{"R"}));
    }

    @Test
    public void testDifference1CP() {
        double[][] X = {{1}, {2}, {3}};
        double[][] Y = {{2}, {3}, {4}};

        double[][] expected = {{1}};
        runUnionTests(X, Y,expected, Types.ExecType.CP);
    }




    private void runUnionTests(double[][] X, double[][]Y, double[][] expected, Types.ExecType instType) {
        Types.ExecMode platformOld = setExecMode(instType);
        try {
            loadTestConfiguration(getTestConfiguration(TEST_NAME));
            String HOME = SCRIPT_DIR + TEST_DIR;
            fullDMLScriptName = HOME + TEST_NAME + ".dml";
            programArgs = new String[]{ "-args", input("X"),input("Y"), output("R")};
            fullRScriptName = HOME + TEST_NAME + ".R";
            rCmd = "Rscript" + " " + fullRScriptName + " " + inputDir() + " " + expectedDir();

            writeInputMatrixWithMTD("X", X, true);
            writeInputMatrixWithMTD("Y", Y, true);

            runTest(true, false, null, -1);
            HashMap<MatrixValue.CellIndex, Double> dmlfile = readDMLMatrixFromOutputDir("R");

            HashMap<MatrixValue.CellIndex, Double> R = new HashMap<>();
            for(int r=0; r< expected.length; r++)
                for(int c=0; c< expected[r].length; c++)
                    R.put(new MatrixValue.CellIndex(r+1,c+1), expected[r][c]);

            TestUtils.compareMatrices(dmlfile, R, 1e-10, "dml", "expected");

            runRScript(true);
            HashMap<MatrixValue.CellIndex, Double> rfile  = readRMatrixFromExpectedDir("R");

            TestUtils.compareMatrices(dmlfile, rfile, 1e-10, "dml", "expected");
        }
        finally {
            rtplatform = platformOld;
        }
    }
}