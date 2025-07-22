package org.apache.sysds.test.functions.rewrite;

import org.apache.sysds.api.DMLScript;
import org.apache.sysds.common.Opcodes;
import org.apache.sysds.common.Types.ExecMode;
import org.apache.sysds.common.Types.ExecType;
import org.apache.sysds.hops.OptimizerUtils;
import org.apache.sysds.runtime.matrix.data.MatrixValue;
import org.apache.sysds.test.AutomatedTestBase;
import org.apache.sysds.test.TestConfiguration;
import org.apache.sysds.test.TestUtils;
import org.apache.sysds.utils.Statistics;
import org.junit.AfterClass;
import org.junit.Assert;
import org.junit.BeforeClass;
import org.junit.Test;

import java.util.HashMap;


public class RewriteSimplifyTransposedCumsumTest extends AutomatedTestBase{
    private static final String TEST_NAME = "RewriteSimplifyTransposedCumsum";
    private static final String TEST_DIR = "functions/rewrite/";
    private static final String TEST_CLASS_DIR = TEST_DIR + RewriteSimplifyTransposedCumsumTest.class.getSimpleName() + "/";

    private static final double eps = 1e-10;

    private static final int rowsMatrix = 1201;
    private static final int colsMatrix = 1103;
    private static final double spSparse = 0.1;
    private static final double spDense = 0.9;

    private enum InputType {
        COL_VECTOR,
        ROW_VECTOR,
        MATRIX
    }

    @Override
    public void setUp() {
        addTestConfiguration(TEST_NAME, new TestConfiguration(TEST_CLASS_DIR, TEST_NAME, new String[] {"B"}));
        if (TEST_CACHE_ENABLED) {
            setOutAndExpectedDeletionDisabled(true);
        }
    }

    @BeforeClass
    public static void init() {
        TestUtils.clearDirectory(TEST_DATA_DIR + TEST_CLASS_DIR);
    }

    @AfterClass
    public static void cleanUp() {
        if (TEST_CACHE_ENABLED) {
            TestUtils.clearDirectory(TEST_DATA_DIR + TEST_CLASS_DIR);
        }
    }

    // dense cp
    @Test public void testRewriteMatrixDenseCPNoRewrite() { testRewriteSimplifyRowcumsum(InputType.MATRIX, false, ExecType.CP, false); }
    @Test public void testRewriteMatrixDenseCP()    { testRewriteSimplifyRowcumsum(InputType.MATRIX, false, ExecType.CP, true); }

    @Test public void testRewriteColVectorDenseCPNoRewrite() { testRewriteSimplifyRowcumsum(InputType.COL_VECTOR, false, ExecType.CP, false); }
    @Test public void testRewriteColVectorDenseCP()    { testRewriteSimplifyRowcumsum(InputType.COL_VECTOR, false, ExecType.CP, true); }

    @Test public void testRewriteRowVectorDenseCPNoRewrite() { testRewriteSimplifyRowcumsum(InputType.ROW_VECTOR, false, ExecType.CP, false); }
    @Test public void testRewriteRowVectorDenseCP()    { testRewriteSimplifyRowcumsum(InputType.ROW_VECTOR, false, ExecType.CP, true); }

    // sparse cp
    @Test public void testRewriteMatrixSparseCPNoRewrite() { testRewriteSimplifyRowcumsum(InputType.MATRIX, true, ExecType.CP, false); }
    @Test public void testRewriteMatrixSparseCP()    { testRewriteSimplifyRowcumsum(InputType.MATRIX, true, ExecType.CP, true); }

    @Test public void testRewriteColVectorSparseCPNoRewrite() { testRewriteSimplifyRowcumsum(InputType.COL_VECTOR, true, ExecType.CP, false); }
    @Test public void testRewriteColVectorSparseCP()    { testRewriteSimplifyRowcumsum(InputType.COL_VECTOR, true, ExecType.CP, true); }

    @Test public void testRewriteRowVectorSparseCPNoRewrite() { testRewriteSimplifyRowcumsum(InputType.ROW_VECTOR, true, ExecType.CP, false); }
    @Test public void testRewriteRowVectorSparseCP()    { testRewriteSimplifyRowcumsum(InputType.ROW_VECTOR, true, ExecType.CP, true); }


    // dense sp
    @Test public void testRewriteMatrixDenseSPNoRewrite() { testRewriteSimplifyRowcumsum(InputType.MATRIX, false, ExecType.SPARK, false); }
    @Test public void testRewriteMatrixDenseSP()    { testRewriteSimplifyRowcumsum(InputType.MATRIX, false, ExecType.SPARK, true); }

    @Test public void testRewriteColVectorDenseSPNoRewrite() { testRewriteSimplifyRowcumsum(InputType.COL_VECTOR, false, ExecType.SPARK, false); }
    @Test public void testRewriteColVectorDenseSP()    { testRewriteSimplifyRowcumsum(InputType.COL_VECTOR, false, ExecType.SPARK, true); }

    @Test public void testRewriteRowVectorDenseSPNoRewrite() { testRewriteSimplifyRowcumsum(InputType.ROW_VECTOR, false, ExecType.SPARK, false); }
    @Test public void testRewriteRowVectorDenseSP()    { testRewriteSimplifyRowcumsum(InputType.ROW_VECTOR, false, ExecType.SPARK, true); }

    // sparse sp
    @Test public void testRewriteMatrixSparseSPNoRewrite() { testRewriteSimplifyRowcumsum(InputType.MATRIX, true, ExecType.SPARK, false); }
    @Test public void testRewriteMatrixSparseSP()    { testRewriteSimplifyRowcumsum(InputType.MATRIX, true, ExecType.SPARK, true); }

    @Test public void testRewriteColVectorSparseSPNoRewrite() { testRewriteSimplifyRowcumsum(InputType.COL_VECTOR, true, ExecType.SPARK, false); }
    @Test public void testRewriteColVectorSparseSP()    { testRewriteSimplifyRowcumsum(InputType.COL_VECTOR, true, ExecType.SPARK, true); }

    @Test public void testRewriteRowVectorSparseSPNoRewrite() { testRewriteSimplifyRowcumsum(InputType.ROW_VECTOR, true, ExecType.SPARK, false); }
    @Test public void testRewriteRowVectorSparseSP()    { testRewriteSimplifyRowcumsum(InputType.ROW_VECTOR, true, ExecType.SPARK, true); }


    private void testRewriteSimplifyRowcumsum(InputType type, boolean sparse, ExecType instType, boolean rewrites) {

        ExecMode platformOld = rtplatform;
        switch( instType ){
            case SPARK: rtplatform = ExecMode.SPARK; break;
            default: rtplatform = ExecMode.HYBRID; break;
        }

        boolean sparkConfigOld = DMLScript.USE_LOCAL_SPARK_CONFIG;
        if( rtplatform == ExecMode.SPARK )
            DMLScript.USE_LOCAL_SPARK_CONFIG = true;

        //rewrites
        boolean oldFlagRewrites = OptimizerUtils.ALLOW_ALGEBRAIC_SIMPLIFICATION;
        OptimizerUtils.ALLOW_ALGEBRAIC_SIMPLIFICATION = rewrites;


        try {
            // Determine matrix dimensions based on InputType
            int rows = (type == InputType.ROW_VECTOR) ? 1 : rowsMatrix;
            int cols = (type == InputType.COL_VECTOR) ? 1 : colsMatrix;
            double sparsity = (sparse) ? spSparse : spDense;

            String TEST_CACHE_DIR = !TEST_CACHE_ENABLED ? "" :
                    type.ordinal() + "_" + sparsity + "/";

            TestConfiguration config = getTestConfiguration(TEST_NAME);
            loadTestConfiguration(config, TEST_CACHE_DIR);

            String HOME = SCRIPT_DIR + TEST_DIR;
            fullDMLScriptName = HOME + TEST_NAME + ".dml";
            programArgs = new String[] {"-stats", "-args", input("A"), output("B")};

            fullRScriptName = HOME + TEST_NAME + ".R";
            rCmd = "Rscript" + " " + fullRScriptName + " " + inputDir() + " " + expectedDir();

            // create and write matrix
            double[][] A = getRandomMatrix(rows, cols, -0.05, 1, sparsity, 7);
            writeInputMatrixWithMTD("A", A, true);

            runTest(true, false, null, -1);
            if( instType == ExecType.CP ) {
                Assert.assertEquals("Unexpected number of executed Spark jobs.", 0, Statistics.getNoOfExecutedSPInst());
            }

            runRScript(true);

            //compare matrices
            HashMap<MatrixValue.CellIndex, Double> dmlfile = readDMLMatrixFromOutputDir("B");
            HashMap<MatrixValue.CellIndex, Double> rfile = readRMatrixFromExpectedDir("B");
            TestUtils.compareMatrices(dmlfile, rfile, eps, "Stat-DML", "Stat-R");

            // Assertions for opcodes
            if(rewrites) {
                // rewrite is enabled: double transposed CUMSUM and CUMSUM is not found, ROWCUMSUM operation is found
                Assert.assertFalse(heavyHittersContainsString(Opcodes.TRANSPOSE.toString()) || heavyHittersContainsString("sp_r'"));
                Assert.assertFalse(heavyHittersContainsString(Opcodes.UCUMKP.toString()) || heavyHittersContainsString("sp_bcumoffk+"));
                Assert.assertTrue(heavyHittersContainsString(Opcodes.UROWCUMKP.toString()) || heavyHittersContainsString("sp_urowcumk+"));
            } else {
                // rewrite is disabled: double transposed CUMSUM and CUMSUM is found, ROWCUMSUM operation is not found
                Assert.assertTrue(heavyHittersContainsString(Opcodes.TRANSPOSE.toString()) || heavyHittersContainsString("sp_r'"));
                Assert.assertTrue(heavyHittersContainsString(Opcodes.UCUMKP.toString()) || heavyHittersContainsString("sp_bcumoffk+"));
                Assert.assertFalse(heavyHittersContainsString(Opcodes.UROWCUMKP.toString()) || heavyHittersContainsString("sp_urowcumk+"));
            }
        }
        finally {
            rtplatform = platformOld;
            DMLScript.USE_LOCAL_SPARK_CONFIG = sparkConfigOld;
            OptimizerUtils.ALLOW_ALGEBRAIC_SIMPLIFICATION = oldFlagRewrites;
        }
    }
}
