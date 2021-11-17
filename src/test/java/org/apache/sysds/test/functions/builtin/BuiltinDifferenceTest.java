package org.apache.sysds.test.functions.builtin;

import org.apache.sysds.common.Types;
import org.apache.sysds.runtime.matrix.data.MatrixValue;
import org.apache.sysds.test.AutomatedTestBase;
import org.apache.sysds.test.TestConfiguration;
import org.apache.sysds.test.TestUtils;
import org.junit.Assert;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.Parameterized;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collection;
import java.util.HashMap;


@RunWith(Parameterized.class)
public class BuiltinDifferenceTest extends AutomatedTestBase {
    private final static String TEST_NAME = "difference";
    private final static String TEST_DIR = "functions/builtin/";
    private static final String TEST_CLASS_DIR = TEST_DIR + BuiltinDifferenceTest.class.getSimpleName() + "/";

    private final Types.ExecType execType;

    public BuiltinDifferenceTest(Types.ExecType execType){

        this.execType = execType;
    }

    @Parameterized.Parameters
    public static Collection<Object[]> types(){
        return Arrays.asList(new Object[][]{
                {Types.ExecType.CP},
                {Types.ExecType.SPARK}
        });
    }

    @Override
    public void setUp() {
        TestUtils.clearAssertionInformation();
        addTestConfiguration(TEST_NAME, new TestConfiguration(TEST_CLASS_DIR, TEST_NAME, new String[]{"R"}));
    }

    @Test
    public void testPosNumbersAscending() {
        double[][] X = {{1}, {2}, {3}};
        double[][] Y = {{2}, {3}, {4}};

        runUnionTests(X, Y, execType);
    }

    @Test
    public void testPosNumbersRandomOrder() {
        double[][] X = {{9}, {2}, {3}};
        double[][] Y = {{2}, {3}, {4}};

        runUnionTests(X, Y, Types.ExecType.CP);
    }

    @Test
    public void testComplexPosNumbers() {
        double[][] X =  {{12},{22},{13},{4},{6},{7},{8},{9},{12},{12}};
        double[][] Y = {{1},{2},{11},{12},{13},{18},{20},{21},{12}};
        runUnionTests(X, Y, Types.ExecType.CP);
    }

    @Test
    public void testNegNumbers() {
        double[][] X = {{-10},{-5},{2}};
        double[][] Y = {{2},{-3}};
        runUnionTests(X, Y, Types.ExecType.CP);
    }

    @Test
    public void testFloatingPNumbers() { //floating point numbers do not work with table()...
        double[][] X = {{2},{2.5},{4}};
        double[][] Y = {{2.4},{2}};
        runUnionTests(X, Y, Types.ExecType.CP);
    }

    @Test
    public void testNegAndFloating() {
        double[][] X =  {{1.4}, {-1.3}, {10}, {4}};
        double[][] Y = {{1.3},{-1.4},{10},{9}};
        runUnionTests(X, Y, Types.ExecType.CP);
    }

    @Test
    public void testMaxValue() {
        double[][] X =  {{Double.MAX_VALUE}, {2},{4}}; //Max value does not work. because of x/y row size hack probably.
        double[][] Y = {{2},{15}};
        runUnionTests(X, Y, Types.ExecType.CP);
    }

    @Test
    public void testMinValue() {
        double[][] X =  {{Double.MIN_VALUE}, {2},{4}};
        double[][] Y = {{2},{15}};
        runUnionTests(X, Y, Types.ExecType.CP);
    }

    @Test
    public void testCombined() {
        double[][] X =  {{Double.MIN_VALUE}, {-Double.MAX_VALUE},{4}, {-1.3}, {10}, {4}};
        double[][] Y = {{Double.MIN_VALUE},{15},{-1.2},{-25.3}};
        runUnionTests(X, Y, Types.ExecType.CP);
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