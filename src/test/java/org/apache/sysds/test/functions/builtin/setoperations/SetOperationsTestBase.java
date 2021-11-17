package org.apache.sysds.test.functions.builtin.setoperations;

import org.apache.sysds.common.Types;
import org.apache.sysds.runtime.matrix.data.MatrixValue;
import org.apache.sysds.test.AutomatedTestBase;
import org.apache.sysds.test.TestConfiguration;
import org.apache.sysds.test.TestUtils;
import org.junit.Assert;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.Parameterized;

import java.util.*;

@RunWith(Parameterized.class)
public abstract class SetOperationsTestBase extends AutomatedTestBase {
    private final String TEST_NAME;
    private final String TEST_DIR ;
    private final String TEST_CLASS_DIR;

    private final Types.ExecType execType;

    public SetOperationsTestBase(String test_name, String test_dir, String test_class_dir, Types.ExecType execType){
        TEST_NAME = test_name;
        TEST_DIR = test_dir;
        TEST_CLASS_DIR = test_class_dir;

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

        runUnitTest(X, Y, execType);
    }

    @Test
    public void testPosNumbersRandomOrder() {
        double[][] X = {{9}, {2}, {3}};
        double[][] Y = {{2}, {3}, {4}};

        runUnitTest(X, Y, Types.ExecType.CP);
    }

    @Test
    public void testComplexPosNumbers() {
        double[][] X =  {{12},{22},{13},{4},{6},{7},{8},{9},{12},{12}};
        double[][] Y = {{1},{2},{11},{12},{13},{18},{20},{21},{12}};
        runUnitTest(X, Y, Types.ExecType.CP);
    }

    @Test
    public void testNegNumbers() {
        double[][] X = {{-10},{-5},{2}};
        double[][] Y = {{2},{-3}};
        runUnitTest(X, Y, Types.ExecType.CP);
    }

    @Test
    public void testFloatingPNumbers() { //floating point numbers do not work with table()...
        double[][] X = {{2},{2.5},{4}};
        double[][] Y = {{2.4},{2}};
        runUnitTest(X, Y, Types.ExecType.CP);
    }

    @Test
    public void testNegAndFloating() {
        double[][] X =  {{1.4}, {-1.3}, {10}, {4}};
        double[][] Y = {{1.3},{-1.4},{10},{9}};
        runUnitTest(X, Y, Types.ExecType.CP);
    }

//    @Test  //TODO max value tests do not work with R since R and java have a different max value.
//    public void testMaxValue() {
//        double[][] X =  {{Double.MAX_VALUE}, {2},{4}};
//        double[][] Y = {{2},{15}};
//        runUnitTest(X, Y, Types.ExecType.CP);
//    }

    @Test
    public void testMinValue() {
        double[][] X =  {{Double.MIN_VALUE}, {2},{4}};
        double[][] Y = {{2},{15}};
        runUnitTest(X, Y, Types.ExecType.CP);
    }

    @Test
    public void testCombined() {
        double[][] X =  {{Double.MIN_VALUE}, {4}, {-1.3}, {10}, {4}};
        double[][] Y = {{Double.MIN_VALUE},{15},{-1.2},{-25.3}};
        runUnitTest(X, Y, Types.ExecType.CP);
    }

    private void runUnitTest(double[][] X, double[][]Y, Types.ExecType instType) {
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
            Collections.sort(dml_values);
            Collections.sort(r_values);

            Assert.assertEquals(dml_values.size(), r_values.size());
            Assert.assertEquals(dml_values, r_values);


            //Junit way collection equal ignore order.
            //Assert.assertTrue(dml_values.size() == r_values.size() && dml_values.containsAll(r_values) && r_values.containsAll(dml_values));
        }
        finally {
            rtplatform = platformOld;
        }
    }


}
