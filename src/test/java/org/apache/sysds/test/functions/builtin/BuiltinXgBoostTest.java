package org.apache.sysds.test.functions.builtin;

import org.apache.sysds.common.Types;
import org.apache.sysds.lops.LopProperties;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;
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

    private final static double eps = 1e-10;

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
            programArgs = new String[]{"-args", input("X"), input("y"), input("R"), output("M")};

            // TODO: Replace with expected input
            double[][] y = { // 5 x 1
                {1.0},
                {0.0},
                {0.0},
                {1.0},
                {0.0}};

            double[][] X = { // 5 x 5
                {4.5, 4.0, 3.0, 2.8, 3.5},
                {1.9, 2.4, 1.0, 3.4, 2.9},
                {2.0, 1.1, 1.0, 4.9, 3.4},
                {2.3, 5.0, 2.0, 1.4, 1.8},
                {2.1, 1.1, 3.0, 1.0, 1.9},};

            double[][] R = { // 1 x 5
                {1.0, 1.0, 1.0, 1.0, 1.0},};

            writeInputMatrixWithMTD("X", X, true);
            writeInputMatrixWithMTD("y", y, true);
            writeInputMatrixWithMTD("R", R, true);

            runTest(true, false, null, -1);

            HashMap<MatrixValue.CellIndex, Double> actual_M = readDMLMatrixFromOutputDir("M");
            HashMap<MatrixValue.CellIndex, Double> expected_M = new HashMap<>();
            //TODO: Replace with expected outputs
            expected_M.put(new MatrixValue.CellIndex(1, 1), 1.0); // node ide
            expected_M.put(new MatrixValue.CellIndex(2, 1), 1.0); // tree id
            expected_M.put(new MatrixValue.CellIndex(3, 1), 1.0); // offset
            expected_M.put(new MatrixValue.CellIndex(4, 1), 2.0); // used feature
            expected_M.put(new MatrixValue.CellIndex(5, 1), 1.0); // scalar
            expected_M.put(new MatrixValue.CellIndex(6, 1), 3.2); // split value

            expected_M.put(new MatrixValue.CellIndex(1, 2), 2.0); // node ide
            expected_M.put(new MatrixValue.CellIndex(2, 2), 1.0); // tree id
            expected_M.put(new MatrixValue.CellIndex(3, 2), 2.0); // offset
            expected_M.put(new MatrixValue.CellIndex(4, 2), 1.0); // used feature
            expected_M.put(new MatrixValue.CellIndex(5, 2), 1.0); // scalar
            expected_M.put(new MatrixValue.CellIndex(6, 2), 2.05); // split value

            expected_M.put(new MatrixValue.CellIndex(1, 3), 3.0); // node ide
            expected_M.put(new MatrixValue.CellIndex(2, 3), 1.0); // tree id
            expected_M.put(new MatrixValue.CellIndex(3, 3), 3.0); // offset
            expected_M.put(new MatrixValue.CellIndex(4, 3), 2.0); // used feature
            expected_M.put(new MatrixValue.CellIndex(5, 3), 1.0); // scalar
            expected_M.put(new MatrixValue.CellIndex(6, 3), 4.5); // split value

            expected_M.put(new MatrixValue.CellIndex(1, 4), 4.0); // node ide
            expected_M.put(new MatrixValue.CellIndex(2, 4), 1.0); // tree id
            expected_M.put(new MatrixValue.CellIndex(3, 4), 4.0); // offset
            expected_M.put(new MatrixValue.CellIndex(4, 4), 1.0); // used feature
            expected_M.put(new MatrixValue.CellIndex(5, 4), 1.0); // scalar
            expected_M.put(new MatrixValue.CellIndex(6, 4), 1.95); // split value

            expected_M.put(new MatrixValue.CellIndex(1, 5), 5.0); // node ide
            expected_M.put(new MatrixValue.CellIndex(2, 5), 1.0); // tree id
            expected_M.put(new MatrixValue.CellIndex(3, 5), 0.0); // offset
            expected_M.put(new MatrixValue.CellIndex(4, 5), 0.0); // used feature
            expected_M.put(new MatrixValue.CellIndex(5, 5), 0.0); // scalar
            expected_M.put(new MatrixValue.CellIndex(6, 5), 0.0); // split value

            expected_M.put(new MatrixValue.CellIndex(1, 6), 6.0); // node ide
            expected_M.put(new MatrixValue.CellIndex(2, 6), 1.0); // tree id
            expected_M.put(new MatrixValue.CellIndex(3, 6), 0.0); // offset
            expected_M.put(new MatrixValue.CellIndex(4, 6), 0.0); // used feature
            expected_M.put(new MatrixValue.CellIndex(5, 6), 0.0); // scalar
            expected_M.put(new MatrixValue.CellIndex(6, 6), 0.0); // split value

            expected_M.put(new MatrixValue.CellIndex(1, 7), 7.0); // node ide
            expected_M.put(new MatrixValue.CellIndex(2, 7), 1.0); // tree id
            expected_M.put(new MatrixValue.CellIndex(3, 7), 0.0); // offset
            expected_M.put(new MatrixValue.CellIndex(4, 7), 0.0); // used feature
            expected_M.put(new MatrixValue.CellIndex(5, 7), 0.0); // scalar
            expected_M.put(new MatrixValue.CellIndex(6, 7), 0.0); // split value

            expected_M.put(new MatrixValue.CellIndex(1, 8), 8.0); // node ide
            expected_M.put(new MatrixValue.CellIndex(2, 8), 1.0); // tree id
            expected_M.put(new MatrixValue.CellIndex(3, 8), 0.0); // offset
            expected_M.put(new MatrixValue.CellIndex(4, 8), 0.0); // used feature
            expected_M.put(new MatrixValue.CellIndex(5, 8), 0.0); // scalar
            expected_M.put(new MatrixValue.CellIndex(6, 8), 0.0); // split value

            expected_M.put(new MatrixValue.CellIndex(1, 9), 9.0); // node ide
            expected_M.put(new MatrixValue.CellIndex(2, 9), 1.0); // tree id
            expected_M.put(new MatrixValue.CellIndex(3, 9), 0.0); // offset
            expected_M.put(new MatrixValue.CellIndex(4, 9), 0.0); // used feature
            expected_M.put(new MatrixValue.CellIndex(5, 9), 0.0); // scalar
            expected_M.put(new MatrixValue.CellIndex(6, 9), 0.0); // split value

            TestUtils.compareMatrices(expected_M, actual_M, eps, "Expected-DML", "Actual-DML");

        } catch (Exception ex) {
            System.out.println("[ERROR] Xgboost test failed, cause: " + ex);
            throw ex;
        } finally {
            rtplatform = platformOld;
        }
    }
}
