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
            programArgs = new String[]{"-args", input("X"), input("Y"), output("M")};

            // TODO: Replace with expected input
            double[][] Y = {{1.0}, {0.0}, {0.0}, {1.0}, {0.0}};

            double[][] X = {{4.5, 4.0, 3.0, 2.8, 3.5}, {1.9, 2.4, 1.0, 3.4, 2.9}, {2.0, 1.1, 1.0, 4.9, 3.4},
                    {2.3, 5.0, 2.0, 1.4, 1.8}, {2.1, 1.1, 3.0, 1.0, 1.9},};
            writeInputMatrixWithMTD("X", X, true);
            writeInputMatrixWithMTD("Y", Y, true);

            runTest(true, false, null, -1);

            HashMap<MatrixValue.CellIndex, Double> actual_M = readDMLMatrixFromOutputDir("M");
            HashMap<MatrixValue.CellIndex, Double> expected_M = new HashMap<>();
            //TODO: Replace with expected outputs
            expected_M.put(new MatrixValue.CellIndex(1, 1), 987.0);
            TestUtils.compareMatrices(expected_M, actual_M, 0, "Expected-DML", "Actual-DML");
        } catch (Exception ex) {
            System.out.println("[ERROR] Xgboost test failed, cause: " + ex);
            throw ex;
        } finally {
            rtplatform = platformOld;
        }
    }
}
