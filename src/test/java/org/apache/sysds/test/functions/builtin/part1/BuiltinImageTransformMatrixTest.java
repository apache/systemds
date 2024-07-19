package org.apache.sysds.test.functions.builtin.part1;

import org.apache.sysds.test.AutomatedTestBase;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.Parameterized;
import org.apache.sysds.common.Types.ExecMode;
import org.apache.sysds.common.Types.ExecType;
import org.apache.sysds.test.TestConfiguration;

import java.util.Arrays;
import java.util.Collection;

@RunWith(Parameterized.class)
@net.jcip.annotations.NotThreadSafe

public class BuiltinImageTransformMatrixTest extends AutomatedTestBase {
    private final static String TEST_NAME_LINEARIZED = "image_transform_matrix";
    private final static String TEST_DIR = "functions/builtin/";
    private final static String TEST_CLASS_DIR = TEST_DIR + BuiltinImageTransformMatrixTest.class.getSimpleName() + "/";

    @Parameterized.Parameter(0)
    public double[][] transMat;
    @Parameterized.Parameter(1)
    public double[][] dimMat;

    @Parameterized.Parameters
    public static Collection<Object[]> data() {
        return Arrays.asList(new Object[][] {{new double[][] {{2,0,0},{0,1,0},{0,0,1}}, new double[][] {{10, 10},{15,15}}}});
    }

    @Override
    public void setUp() {
        addTestConfiguration(TEST_NAME_LINEARIZED,
                new TestConfiguration(TEST_CLASS_DIR, TEST_NAME_LINEARIZED, new String[] {"B_x"}));
    }

    @Test
    public void testImageTransformMatrix() {
        runImageTransformMatrixTest(ExecType.CP);
    }

    private void runImageTransformMatrixTest(ExecType instType) {
        ExecMode platformOld = setExecMode(instType);
        disableOutAndExpectedDeletion();

        try {
            loadTestConfiguration(getTestConfiguration(TEST_NAME_LINEARIZED));

            String HOME = SCRIPT_DIR + TEST_DIR;

            writeInputMatrixWithMTD("transMat", transMat, true);
            writeInputMatrixWithMTD("dimMat", dimMat, true);

            fullDMLScriptName = HOME + TEST_NAME_LINEARIZED + ".dml";
            programArgs = new String[]{"-nvargs", "transMat=" + input("transMat"), "dimMat=" + input("dimMat"), "out_file=" + output("B_x"), "--debug"};

            //double[][] A = getRandomMatrix(rows, height*width, 0, 255, sparsity, 7);


            runTest(true, false, null, -1);

            //HashMap<MatrixValue.CellIndex, Double> dmlfileLinearizedX = readDMLMatrixFromOutputDir("B_x");

            //HashMap<MatrixValue.CellIndex, Double> dmlfileX = readDMLMatrixFromOutputDir("B_x_reshape");

            //TestUtils.compareMatrices(dmlfileLinearizedX, dmlfileX, eps, "Stat-DML-LinearizedX", "Stat-DML-X");

        } catch (Exception e) {
            e.printStackTrace();
        } finally {
            rtplatform = platformOld;
        }
    }


}
