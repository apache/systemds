package org.apache.sysds.test.functions.builtin;

import org.apache.sysds.common.Types;
import org.apache.sysds.runtime.matrix.data.MatrixValue;
import org.apache.sysds.test.AutomatedTestBase;
import org.apache.sysds.test.TestConfiguration;
import org.apache.sysds.test.TestUtils;
import org.junit.Test;

import java.util.HashMap;

public class BuiltinXgBoostPredictTest_regression extends AutomatedTestBase {
    private final static String TEST_NAME = "xgboost_predict_regression";
    private final static String TEST_DIR = "functions/builtin/";
    private static final String TEST_CLASS_DIR = TEST_DIR + BuiltinXgBoostPredictTest_regression.class.getSimpleName() + "/";
    double eps = 1e-10;

    @Override
    public void setUp() {
        TestUtils.clearAssertionInformation();
        addTestConfiguration(TEST_NAME, new TestConfiguration(TEST_CLASS_DIR, TEST_NAME, new String[]{"C"}));
    }

    @Test
    public void testXgBoost() {
        executeXgBoost(Types.ExecMode.SINGLE_NODE, 2.0);
    }

    private void executeXgBoost(Types.ExecMode mode, double threshold) {
        Types.ExecMode platformOld = setExecMode(mode);
        try {
            loadTestConfiguration(getTestConfiguration(TEST_NAME));

            String HOME = SCRIPT_DIR + TEST_DIR;
            fullDMLScriptName = HOME + TEST_NAME + ".dml";
            programArgs = new String[]{"-args", output("P"), output("y")};

            runTest(true, false, null, -1);

            HashMap<MatrixValue.CellIndex, Double> predicted_values = readDMLMatrixFromOutputDir("P");
            HashMap<MatrixValue.CellIndex, Double> actual_values = readDMLMatrixFromOutputDir("y");

            TestUtils.compareMatrices(predicted_values, actual_values, threshold, "predicted_val", "actual_value");

        } catch (Exception ex) {
            System.out.println("[ERROR] Xgboost test failed, cause: " + ex);
            throw ex;
        } finally {
            rtplatform = platformOld;
        }
    }
}
