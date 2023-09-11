
package org.apache.sysds.test.functions.builtin.part1;

import org.junit.Test;
import org.apache.sysds.common.Types.ExecMode;
import org.apache.sysds.common.Types.ExecType;
import org.apache.sysds.runtime.matrix.data.MatrixValue;
import org.apache.sysds.test.AutomatedTestBase;
import org.apache.sysds.test.TestConfiguration;
import org.apache.sysds.test.TestUtils;

import java.util.HashMap;

public class BuiltinImageMirrorLinearizedTest extends AutomatedTestBase {
    private final static String TEST_NAME_LINEARIZED = "image_mirror_linearized";
    private final static String TEST_NAME = "image_mirror";
    private final static String TEST_DIR = "functions/builtin/";
    private final static String TEST_CLASS_DIR = TEST_DIR + BuiltinImageMirrorLinearizedTest.class.getSimpleName() + "/";
    
   private final static double eps = 1e-10;
   private final static int rows = 64; 
   private final static int cols = 64; 
   private final static double spSparse = 0.05; 
   private final static double spDense = 0.5; 

    @Override
    public void setUp() {
        addTestConfiguration(TEST_NAME_LINEARIZED, new TestConfiguration(TEST_CLASS_DIR, TEST_NAME_LINEARIZED, new String[]{"B"}));
        addTestConfiguration(TEST_NAME, new TestConfiguration(TEST_CLASS_DIR, TEST_NAME, new String[]{"B"}));
    }

    @Test
    public void testImageMirrorLinearizedDenseCP() {
        runImageMirrorLinearizedTest(false, ExecType.CP);
    }

    private void runImageMirrorLinearizedTest(boolean sparse, ExecType instType) {
        ExecMode platformOld = setExecMode(instType);
        disableOutAndExpectedDeletion();

        try {
            loadTestConfiguration(getTestConfiguration(TEST_NAME_LINEARIZED));
            loadTestConfiguration(getTestConfiguration(TEST_NAME));

            double sparsity = sparse ? spSparse : spDense;
            String HOME = SCRIPT_DIR + TEST_DIR;

            // For image_mirror_linearized
            fullDMLScriptName = HOME + TEST_NAME_LINEARIZED + ".dml";
            programArgs = new String[]{"-nvargs",
                    "in_file=" + input("A"), "x_out_reshape_file=" + output("B_x_reshape"), "y_out_reshape_file=" + output("B_y_reshape")
            };
            double[][] A = getRandomMatrix(rows, cols, 0, 255, sparsity, 7);
            writeInputMatrixWithMTD("A", A, true);

            runTest(true, false, null, -1);

            HashMap<MatrixValue.CellIndex, Double> dmlfileLinearizedX = readDMLMatrixFromOutputDir("B_x_reshape");
            HashMap<MatrixValue.CellIndex, Double> dmlfileLinearizedY = readDMLMatrixFromOutputDir("B_y_reshape");

            // For image_mirror
            fullDMLScriptName = HOME + TEST_NAME + ".dml";
            programArgs = new String[]{"-nvargs",
                    "in_file=" + input("A"), "x_out_file=" + output("B_x"), "y_out_file=" + output("B_y")
            };

            runTest(true, false, null, -1);

            HashMap<MatrixValue.CellIndex, Double> dmlfileX = readDMLMatrixFromOutputDir("B_x");
            HashMap<MatrixValue.CellIndex, Double> dmlfileY = readDMLMatrixFromOutputDir("B_y");

            // Compare matrices
            TestUtils.compareMatrices(dmlfileLinearizedX, dmlfileX, eps, "Stat-DML-LinearizedX", "Stat-DML-X");
            TestUtils.compareMatrices(dmlfileLinearizedY, dmlfileY, eps, "Stat-DML-LinearizedY", "Stat-DML-Y");
        

        } finally {
            rtplatform = platformOld;
        }
    }
}
