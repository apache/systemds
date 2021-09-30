package org.apache.sysds.test.functions.builtin;

import org.apache.sysds.common.Types;
import org.apache.sysds.hops.OptimizerUtils;
import org.apache.sysds.runtime.matrix.data.MatrixValue;
import org.apache.sysds.test.AutomatedTestBase;
import org.apache.sysds.test.TestConfiguration;
import org.apache.sysds.test.TestUtils;
import org.junit.Test;

import java.util.HashMap;


public class BuiltInInPlaceTest extends AutomatedTestBase{
    private final static String TEST_NAME = "updateInPlaceTest";
    private final static String TEST_DIR = "functions/builtin/";
    private final static String TEST_CLASS_DIR = TEST_DIR + BuiltinSplitTest.class.getSimpleName() + "/";
    private final static double eps = 1e-3;

    @Override
    public void setUp() {
        TestUtils.clearAssertionInformation();
        addTestConfiguration(TEST_NAME, new TestConfiguration(TEST_CLASS_DIR, TEST_NAME, new String[]{"B",}));
    }

    @Test
    public void testInPlace() {
        runInPlaceTest(Types.ExecType.CP);
    }


    private void runInPlaceTest(Types.ExecType instType) {
        Types.ExecMode platformOld = setExecMode(instType);
        try {
            loadTestConfiguration(getTestConfiguration(TEST_NAME));
            String HOME = SCRIPT_DIR + TEST_DIR;
            fullDMLScriptName = HOME + TEST_NAME + ".dml";
            programArgs = new String[]{"-nvargs","Out=" + output("Out") };

            //double[][] A = getRandomMatrix(size, 1, -10, 10, 0.6, 7);
            //writeInputMatrixWithMTD("A", A, true);
            OptimizerUtils.ENABLE_UNARY_UPDATE_IN_PLACE = true;
            runTest(true, false, null, -1);
            HashMap<MatrixValue.CellIndex, Double> dmlfileOut1 = readDMLMatrixFromOutputDir("Out");
            OptimizerUtils.ENABLE_UNARY_UPDATE_IN_PLACE = false;
            runTest(true, false, null, -1);
            HashMap<MatrixValue.CellIndex, Double> dmlfileOut2 = readDMLMatrixFromOutputDir("Out");

            //compare matrices
            // HashMap<MatrixValue.CellIndex, Double> dmlfileOut1 = readDMLMatrixFromOutputDir("Out");
            // HashMap<MatrixValue.CellIndex, Double> rfileOut1 = readRMatrixFromExpectedDir("Out");
            // TestUtils.compareMatrices(dmlfileOut1, rfileOut1, eps, "Stat-DML", "Stat-R");
            // TestUtils.compareScalars(1,1,eps);
            TestUtils.compareMatrices(dmlfileOut1,dmlfileOut2,eps,"Stat-DML1","Stat-DML2");
        }
        catch(Exception e) {
            e.printStackTrace();
        }
        finally {
            rtplatform = platformOld;
        }
    }
}