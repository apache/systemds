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

    private static double [][] t1 = new double[][] {{2,0,0},{0,1,0},{0,0,1}};
    private static double [][] d1 = new double[][] {{10, 10},{15,15}};
    private static double [][] t2 = new double[][] {{4,0,0},{0,2,0},{0,0,1}};
    private static double [][] d2 = new double[][] {{10, 10},{15,15}};
    private static double [][] t3 = new double[][] {{2,0,0},{0,1,0},{0,0,1}};
    private static double [][] d3 = new double[][] {{100, 100},{150,150}};
    private static double [][] t4 = new double[][] {{4,0,0},{0,2,0},{0,0,1}};
    private static double [][] d4 = new double[][] {{100, 100},{150,150}};
    private static double [][] t5 = new double[][] {{-5,0,0},{0,-1,0},{0,0,1}};
    private static double [][] d5 = new double[][] {{100, 100},{150,150}};
    private static double [][] t6 = new double[][] {{2,0,0},{0,1,0},{0,0,1}};
    private static double [][] d6 = new double[][] {{1920, 1080},{1980, 1080}};
    private static double [][] t7 = new double[][] {{2,0,0},{0,1,0},{0,0,1}};
    private static double [][] d7 = new double[][] {{1920, 1080},{3840, 2160}};
    private static double [][] t8 = new double[][] {{2,0,0},{0,1,0},{0,0,1}};
    private static double [][] d8 = new double[][] {{3840, 2160},{1980, 1080}};

    @Parameterized.Parameters
    public static Collection<Object[]> data() {
        return Arrays.asList(new Object[][] {{t1, d1},{t2, d2},{t3, d3},{t4, d4},{t5, d5},{t6, d6},{t7, d7},{t8, d8}});
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
