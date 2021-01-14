package org.apache.sysds.test.functions.builtin;

import org.apache.sysds.runtime.matrix.data.MatrixValue;
import org.apache.sysds.test.AutomatedTestBase;
import org.apache.sysds.test.TestConfiguration;
import org.apache.sysds.test.TestUtils;
import org.junit.Test;

import java.util.HashMap;

public class BuiltinStatsNATest extends AutomatedTestBase {
    private final static String TEST_NAME = "statsNATest";
    private final static String TEST_DIR = "functions/builtin/";
    private final static String TEST_CLASS_DIR = TEST_DIR + BuiltinSplitTest.class.getSimpleName() + "/";
    private final static double eps = 1e-3;

    @Override
    public void setUp() {
        TestUtils.clearAssertionInformation();
        addTestConfiguration(TEST_NAME, new TestConfiguration(TEST_CLASS_DIR, TEST_NAME, new String[]{"B",}));
    }

    @Test
    public void testStatsNA1() {
        runStatsNA(1, 100);
    }

    @Test
    public void testStatsNA2() {
        runStatsNA(4, 100);
    }

    @Test
    public void testStatsNA3() {
        runStatsNA(100, 1000);
    }

    @Test
    public void testStatsNA4() {
        runStatsNA(100, 100);
    }
    
    private void runStatsNA(int bins, int size) {
        loadTestConfiguration(getTestConfiguration(TEST_NAME));
        String HOME = SCRIPT_DIR + TEST_DIR;
        fullDMLScriptName = HOME + TEST_NAME + ".dml";
        programArgs = new String[]{
                "-nvargs",
                "X=" + input("A"),
                "bins=" + bins,
                "Out1=" + output("Out1"),
                "Out2=" + output("Out2"),
                "Out3=" + output("Out3"),
                "Out4=" + output("Out4"),
                "Out5=" + output("Out5"),
                "Out6=" + output("Out6"),
                "Out7=" + output("Out7"),
                "Out8=" + output("Out8")
        };

        double[][] A = getRandomMatrix(size, 1, -10, 10, 0.6, 7);
        writeInputMatrixWithMTD("A", A, true);

        fullRScriptName = HOME + TEST_NAME + ".R";
        rCmd = getRCmd(inputDir(), Integer.toString(bins), expectedDir());

        runTest(true, false, null, -1);
        runRScript(true);
        //compare matrices
        HashMap<MatrixValue.CellIndex, Double> dmlfileOut1 = readDMLScalarFromOutputDir("Out1");
        HashMap<MatrixValue.CellIndex, Double> dmlfileOut2 = readDMLScalarFromOutputDir("Out2");
        HashMap<MatrixValue.CellIndex, Double> dmlfileOut3 = readDMLScalarFromOutputDir("Out3");
        HashMap<MatrixValue.CellIndex, Double> dmlfileOut4 = readDMLScalarFromOutputDir("Out4");
        HashMap<MatrixValue.CellIndex, Double> dmlfileOut5 = readDMLScalarFromOutputDir("Out5");
        HashMap<MatrixValue.CellIndex, Double> dmlfileOut6 = readDMLScalarFromOutputDir("Out6");
        HashMap<MatrixValue.CellIndex, Double> dmlfileOut7 = readDMLScalarFromOutputDir("Out7");
        HashMap<MatrixValue.CellIndex, Double> dmlfileOut8 = readDMLScalarFromOutputDir("Out8");

        HashMap<MatrixValue.CellIndex, Double> rfileOut1 = readRScalarFromExpectedDir("Out1");
        HashMap<MatrixValue.CellIndex, Double> rfileOut2 = readRScalarFromExpectedDir("Out2");
        HashMap<MatrixValue.CellIndex, Double> rfileOut3 = readRScalarFromExpectedDir("Out3");
        HashMap<MatrixValue.CellIndex, Double> rfileOut4 = readRScalarFromExpectedDir("Out4");
        HashMap<MatrixValue.CellIndex, Double> rfileOut5 = readRScalarFromExpectedDir("Out5");
        HashMap<MatrixValue.CellIndex, Double> rfileOut6 = readRScalarFromExpectedDir("Out6");
        HashMap<MatrixValue.CellIndex, Double> rfileOut7 = readRScalarFromExpectedDir("Out7");
        HashMap<MatrixValue.CellIndex, Double> rfileOut8 = readRScalarFromExpectedDir("Out8");

        MatrixValue.CellIndex key_ce = new MatrixValue.CellIndex(1, 1);

        TestUtils.compareScalars(dmlfileOut1.get(key_ce), rfileOut1.get(key_ce), eps);
        TestUtils.compareScalars(dmlfileOut2.get(key_ce), rfileOut2.get(key_ce), eps);
        TestUtils.compareScalars(dmlfileOut3.get(key_ce), rfileOut3.get(key_ce), eps);
        TestUtils.compareScalars(dmlfileOut4.get(key_ce), rfileOut4.get(key_ce), eps);
        TestUtils.compareScalars(dmlfileOut5.get(key_ce), rfileOut5.get(key_ce), eps);
        TestUtils.compareScalars(dmlfileOut6.get(key_ce), rfileOut6.get(key_ce), eps);
        TestUtils.compareScalars(dmlfileOut7.get(key_ce), rfileOut7.get(key_ce), eps);
        TestUtils.compareScalars(dmlfileOut8.get(key_ce), rfileOut8.get(key_ce), eps);
    }
}
