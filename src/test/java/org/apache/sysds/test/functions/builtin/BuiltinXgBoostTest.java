package org.apache.sysds.test.functions.builtin;

import org.apache.sysds.common.Types;
import org.apache.sysds.lops.LopProperties;
import org.apache.sysds.runtime.matrix.data.MatrixValue;
import org.apache.sysds.test.AutomatedTestBase;
import org.apache.sysds.test.TestConfiguration;
import org.apache.sysds.test.TestUtils;
import org.junit.Test;

import java.util.HashMap;

public class BuiltinXgBoostTest extends AutomatedTestBase {
    private final static String TEST_NAME = "xgboost";
    private final static String TEST_DIR = "functions/builtin/";
    private static final String TEST_CLASS_DIR = TEST_DIR + BuiltinXgBoostTest.class.getSimpleName() + "/";

    @Override
    public void setUp() {
        addTestConfiguration(TEST_NAME, new TestConfiguration(TEST_CLASS_DIR, TEST_NAME, new String[]{"C"}));
    }

    @Test
    public void testXgBoost() {
        executeXgBoost(LopProperties.ExecType.CP);
    }

    private void executeXgBoost(LopProperties.ExecType instType) {
        Types.ExecMode platformOld = setExecMode(instType);
        try {
            loadTestConfiguration(getTestConfiguration(TEST_NAME));

            String HOME = SCRIPT_DIR + TEST_DIR;
            fullDMLScriptName = HOME + TEST_NAME + ".dml";
            programArgs = new String[]{"-args", input("X"), input("Y"), input("R"), output("M")};

            // TODO: Replace with expected input
            double[][] Y = {{1.0}, {2.0}, {3.0}, {4.0}, {5.0}, {6.0}, {7.0}, {8.0}};

            double[][] X = {{15.0, 25.0, 35.0, 45.0, 17.0,27.0,37.0,99.0}, {20.0,30.0,87.0,90.0,30.0,40.0,97.0,80.0}, {15.0,24.0,38.0,43.0,25.0,34.0,48.0,53.0},};

            double[][] R = {{1.0,1.0,1.0},};

            writeInputMatrixWithMTD("X", X, true);
            writeInputMatrixWithMTD("Y", Y, true);
            writeInputMatrixWithMTD("R", R, true);

            runTest(true, false, null, -1);

            HashMap<MatrixValue.CellIndex, Double> actual_M = readDMLMatrixFromOutputDir("M");
            HashMap<MatrixValue.CellIndex, Double> expected_M = new HashMap<>();
            //TODO: Replace with expected outputs
            expected_M.put(new MatrixValue.CellIndex(1, 1), 1.0);
            TestUtils.compareMatrices(expected_M, actual_M, 0, "Expected-DML", "Actual-DML");
        } catch (Exception ex) {
            System.out.println("[ERROR] Xgboost test failed, cause: " + ex);
            throw ex;
        } finally {
            rtplatform = platformOld;
        }
    }
}
