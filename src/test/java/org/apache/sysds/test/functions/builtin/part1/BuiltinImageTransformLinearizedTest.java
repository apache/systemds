package org.apache.sysds.test.functions.builtin.part1;

import org.junit.Ignore;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.Parameterized;
import org.apache.sysds.common.Types.ExecMode;
import org.apache.sysds.common.Types.ExecType;
import org.apache.sysds.runtime.matrix.data.MatrixValue;
import org.apache.sysds.test.AutomatedTestBase;
import org.apache.sysds.test.TestConfiguration;
import org.apache.sysds.test.TestUtils;

import java.util.Arrays;
import java.util.Collection;
import java.util.HashMap;

@RunWith(Parameterized.class)
@net.jcip.annotations.NotThreadSafe

public class BuiltinImageTransformLinearizedTest extends AutomatedTestBase {
    private final static String TEST_NAME_LINEARIZED = "image_transform_linearized";
    private final static String TEST_DIR = "functions/builtin/";
    private final static String TEST_CLASS_DIR = TEST_DIR + BuiltinImageTransformLinearizedTest.class.getSimpleName() + "/";

    private final static double eps = 1e-10;
    private final static double spSparse = 0.05;
    private final static double spDense = 0.5;

    @Parameterized.Parameter(0)
    public int rows; //number of linearized images
    @Parameterized.Parameter(1)
    public int width;
    @Parameterized.Parameter(2)
    public int height;
    @Parameterized.Parameter(3)
    public int out_w;
    @Parameterized.Parameter(4)
    public int out_h;
    @Parameterized.Parameter(5)
    public int a;
    @Parameterized.Parameter(6)
    public int b;
    @Parameterized.Parameter(7)
    public int c;
    @Parameterized.Parameter(8)
    public int d;
    @Parameterized.Parameter(9)
    public int e;
    @Parameterized.Parameter(10)
    public int f;
    @Parameterized.Parameter(11)
    public int fill_value;
    @Parameterized.Parameter(12)
    public int s_cols;
    @Parameterized.Parameter(13)
    public int s_rows;


    @Parameterized.Parameters
    public static Collection<Object[]> data() {
        return Arrays.asList(new Object[][] {{1,512, 512, 512, 512, 1,0,0,0,1,0,1, 512, 512}});
    }

    @Override
    public void setUp() {
        addTestConfiguration(TEST_NAME_LINEARIZED,
                new TestConfiguration(TEST_CLASS_DIR, TEST_NAME_LINEARIZED, new String[] {"B_x"}));
    }

    @Test
    public void testImageTranslateLinearized() {
        runImageTranslateLinearizedTest(false, ExecType.CP);
    }

    @Test
    @Ignore
    public void testImageTranslateLinearizedSP() {
        runImageTranslateLinearizedTest(true, ExecType.SPARK);
    }

    private void runImageTranslateLinearizedTest(boolean sparse, ExecType instType) {
        ExecMode platformOld = setExecMode(instType);
        disableOutAndExpectedDeletion();

        try {
            loadTestConfiguration(getTestConfiguration(TEST_NAME_LINEARIZED));

            double sparsity = sparse ? spSparse : spDense;
            String HOME = SCRIPT_DIR + TEST_DIR;

            fullDMLScriptName = HOME + TEST_NAME_LINEARIZED + ".dml";
            programArgs = new String[] {"-nvargs", "in_file=" + input("A"), "width=" + width, "height=" + height,
                    "out_w=" + out_w, "out_h=" + out_h, "a=" + a, "b=" + b, "c=" + c, "d=" + d, "e=" + e, "f=" + f,
                    "fill_value=" + fill_value, "s_cols=" + s_cols, "s_rows=" + s_rows,
                    "out_file=" + output("B_x")};

            double[][] A = getRandomMatrix(rows, height*width, 0, 255, sparsity, 7);
            writeInputMatrixWithMTD("A", A, true);

            runTest(true, false, null, -1);

            //HashMap<MatrixValue.CellIndex, Double> dmlfileLinearizedX = readDMLMatrixFromOutputDir("B_x");

            //HashMap<MatrixValue.CellIndex, Double> dmlfileX = readDMLMatrixFromOutputDir("B_x_reshape");

            //TestUtils.compareMatrices(dmlfileLinearizedX, dmlfileX, eps, "Stat-DML-LinearizedX", "Stat-DML-X");

        }
        finally {
            rtplatform = platformOld;
        }
    }
}
