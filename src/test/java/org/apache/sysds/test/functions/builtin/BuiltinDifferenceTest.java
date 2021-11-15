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
        runUnionTests(X, Y, Types.ExecType.CP);
    }

    @Test
    public void testDifference1SP() {
        double[][] X = {{1}, {2}, {3}};
        double[][] Y = {{2}, {3}, {4}};

        runUnionTests(X, Y, Types.ExecType.SPARK);
    }

    @Test
    public void testDifference2CP() {
        double[][] X = {{9}, {2}, {3}};
        double[][] Y = {{2}, {3}, {4}};

        runUnionTests(X, Y, Types.ExecType.CP);
    }

    @Test
    public void testDifference2SP() {
        double[][] X = {{9}, {2}, {3}};
        double[][] Y = {{2}, {3}, {4}};

        runUnionTests(X, Y, Types.ExecType.SPARK);
    }

    @Test
    public void testDifference3CP() { //fails because element order in R is wrong
        double[][] X =  {{12},{22},{13},{4},{6},{7},{8},{9},{12},{12}};
        double[][] Y = {{1},{2},{11},{12},{13},{18},{20},{21},{12}};
        runUnionTests(X, Y, Types.ExecType.CP);
    }

    @Test
    public void testDifference3Spark() {
        double[][] X = {{12},{22},{13},{4},{6},{7},{8},{9},{12},{12}};
        double[][] Y = {{1},{2},{11},{12},{13},{18},{20},{21},{12}};
        runUnionTests(X, Y, Types.ExecType.SPARK);
    }

    @Test
    public void testDifference4CP() { //fails because element order in R is wrong
        double[][] X =  {{1.4}, {-1.3}, {10}, {4}};
        double[][] Y = {{1.3},{-1.4},{10},{9}};
        runUnionTests(X, Y, Types.ExecType.CP);
    }

    @Test
    public void testDifference4Spark() {
        double[][] X =  {{1.4}, {-1.3}, {10}, {4}};
        double[][] Y = {{1.3},{-1.4},{10},{9}};
        runUnionTests(X, Y, Types.ExecType.SPARK);
    }

    private void runUnionTests(double[][] X, double[][]Y, Types.ExecType instType) {
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
            runRScript(true);

            HashMap<MatrixValue.CellIndex, Double> dmlfile = readDMLMatrixFromOutputDir("R");
            HashMap<MatrixValue.CellIndex, Double> rfile  = readRMatrixFromExpectedDir("R");

            TestUtils.compareMatrices(dmlfile, rfile, 1e-10, "dml", "expected");
        }
        finally {
            rtplatform = platformOld;
        }
    }
}