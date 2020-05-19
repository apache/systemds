package org.apache.sysds.test.functions.builtin;

import org.apache.sysds.common.Types;
import org.apache.sysds.lops.LopProperties;
import org.apache.sysds.runtime.matrix.data.MatrixValue;
import org.apache.sysds.test.AutomatedTestBase;
import org.apache.sysds.test.TestConfiguration;
import org.apache.sysds.test.TestUtils;
import org.junit.Test;

import java.util.Arrays;
import java.util.HashMap;

public class BuiltinToOneHotTest extends AutomatedTestBase {
    private final static String TEST_NAME = "to_one_hot";
    private final static String TEST_DIR = "functions/builtin/";
    private static final String TEST_CLASS_DIR = TEST_DIR + BuiltinToOneHotTest.class.getSimpleName() + "/";

    private final static double eps = 0;
    private final static int rows = 10;
    private final static int cols = 1;
    private final static int numClasses = 10;

    @Override
    public void setUp() {
        addTestConfiguration(TEST_NAME,new TestConfiguration(TEST_CLASS_DIR, TEST_NAME, new String[]{"B"}));
    }

    @Test
    public void runSimpleTest() {
        runToOneHotTest(false, false, LopProperties.ExecType.CP);
    }

    private void runToOneHotTest(boolean scalar, boolean sparse, LopProperties.ExecType instType) {
        Types.ExecMode platformOld = setExecMode(instType);

        try
        {
            loadTestConfiguration(getTestConfiguration(TEST_NAME));

            String HOME = SCRIPT_DIR + TEST_DIR;
            fullDMLScriptName = HOME + TEST_NAME + ".dml";
            programArgs = new String[]{"-explain", "-args", input("A"), String.format("%d", numClasses),
                    output("B") };

            //generate actual dataset
            double[][] doubles = getRandomMatrix(rows, cols, 1, numClasses, 1, 7);

            // round them
            double[][] A = new double[rows][cols];
            for(int i = 0; i < rows; i++) {
                for(int j = 0; j < cols; j++) {
                    A[i][j] = Math.round(doubles[i][j]);
                }
            }

            writeInputMatrixWithMTD("A", A, false);

            runTest(true, false, null, -1);

            HashMap<MatrixValue.CellIndex, Double> expected = new HashMap<MatrixValue.CellIndex, Double>();
            for(int i = 0; i < A.length; i++) {
                for(int j = 0; j < A[i].length; j++) {
                    // indices start with 1 here
                    expected.put(new MatrixValue.CellIndex(i + 1, (int) A[i][j]), 1.0);
                }
            }

            //compare matrices
            HashMap<MatrixValue.CellIndex, Double> result = readDMLMatrixFromHDFS("B");
            TestUtils.compareMatrices(result, expected, eps, "Stat-DML", "Stat-Java");
        }
        finally {
            rtplatform = platformOld;
        }
    }
}
