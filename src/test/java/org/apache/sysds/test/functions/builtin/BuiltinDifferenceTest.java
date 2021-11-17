package org.apache.sysds.test.functions.builtin;

import org.apache.spark.sql.catalyst.expressions.Ascending;
import org.apache.sysds.common.Types;
import org.apache.sysds.runtime.matrix.data.MatrixValue;
import org.apache.sysds.test.AutomatedTestBase;
import org.apache.sysds.test.TestConfiguration;
import org.apache.sysds.test.TestUtils;
import org.junit.Assert;
import org.junit.Test;

import java.util.ArrayList;
import java.util.Comparator;
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
    public void testDifference3CP() {
        double[][] X =  {{12},{22},{13},{4},{6},{7},{8},{9},{12},{12}};
        double[][] Y = {{1},{2},{11},{12},{13},{18},{20},{21},{12}};
        runUnionTests(X, Y, Types.ExecType.CP);
    }

    @Test
    public void testDifference3SP() {
        double[][] X = {{12},{22},{13},{4},{6},{7},{8},{9},{12},{12}};
        double[][] Y = {{1},{2},{11},{12},{13},{18},{20},{21},{12}};
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

            ArrayList<Double> dml_values = new ArrayList<>(dmlfile.values());
            ArrayList<Double> r_values = new ArrayList<>(rfile.values());

            //Junit way collection equal ignore order.
            Assert.assertTrue(dml_values.size() == r_values.size() && dml_values.containsAll(r_values) && r_values.containsAll(dml_values));
        }
        finally {
            rtplatform = platformOld;
        }
    }
}