/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

package org.apache.sysml.test.integration.functions.aggregate;

import org.apache.sysml.api.DMLScript;
import org.apache.sysml.api.DMLScript.RUNTIME_PLATFORM;
import org.apache.sysml.hops.OptimizerUtils;
import org.apache.sysml.lops.LopProperties.ExecType;
import org.apache.sysml.runtime.instructions.Instruction;
import org.apache.sysml.runtime.matrix.data.MatrixValue.CellIndex;
import org.apache.sysml.test.integration.AutomatedTestBase;
import org.apache.sysml.test.integration.TestConfiguration;
import org.apache.sysml.test.utils.TestUtils;
import org.apache.sysml.utils.Statistics;
import org.junit.Assert;
import org.junit.Test;

import java.util.HashMap;

/**
 * Test the row variances function, "rowVars(X)".
 */
public class RowVariancesTest extends AutomatedTestBase {

    private static final String TEST_NAME = "RowVariances";
    private static final String TEST_DIR = "functions/aggregate/";
    private static final String TEST_CLASS_DIR =
            TEST_DIR + RowVariancesTest.class.getSimpleName() + "/";
    private static final String INPUT_NAME = "X";
    private static final String OUTPUT_NAME = "rowVariances";

    private static final String rowVarOp = "uarvar";
    private static final String varOp = "uavar";
    private static final int rows = 1234;
    private static final int cols = 567;
    private static final double sparsity1 = 1;
    private static final double sparsity2 = 0.2;
    private static final double eps = Math.pow(10, -10);

    private enum VectorType {NONE, ROWVECTOR, COLUMNVECTOR}

    @Override
    public void setUp() {
        TestUtils.clearAssertionInformation();
        TestConfiguration config = new TestConfiguration(TEST_CLASS_DIR, TEST_NAME);
        addTestConfiguration(TEST_NAME, config);
    }

    // Dense matrix w/ rewrites
    @Test
    public void testRowVariancesDenseMatrixRewritesCP() {
        testRowVariances(TEST_NAME, false, VectorType.NONE, true, ExecType.CP);
    }

    @Test
    public void testRowVariancesDenseMatrixRewritesSpark() {
        testRowVariances(TEST_NAME, false, VectorType.NONE, true, ExecType.SPARK);
    }

    @Test
    public void testRowVariancesDenseMatrixRewritesMR() {
        testRowVariances(TEST_NAME, false, VectorType.NONE, true, ExecType.MR);
    }

    // Dense matrix w/o rewrites
    @Test
    public void testRowVariancesDenseMatrixNoRewritesCP() {
        testRowVariances(TEST_NAME, false, VectorType.NONE, false, ExecType.CP);
    }

    @Test
    public void testRowVariancesDenseMatrixNoRewritesSpark() {
        testRowVariances(TEST_NAME, false, VectorType.NONE, false, ExecType.SPARK);
    }

    @Test
    public void testRowVariancesDenseMatrixNoRewritesMR() {
        testRowVariances(TEST_NAME, false, VectorType.NONE, false, ExecType.MR);
    }

    // Dense vector w/ rewrites
    //  - Row vector
    @Test
    public void testRowVariancesDenseRowVectorRewritesCP() {
        testRowVariances(TEST_NAME, false, VectorType.ROWVECTOR, true, ExecType.CP);
    }

    @Test
    public void testRowVariancesDenseRowVectorRewritesSpark() {
        testRowVariances(TEST_NAME, false, VectorType.ROWVECTOR, true, ExecType.SPARK);
    }

    @Test
    public void testRowVariancesDenseRowVectorRewritesMR() {
        testRowVariances(TEST_NAME, false, VectorType.ROWVECTOR, true, ExecType.MR);
    }

    //  - Column vector
    @Test
    public void testRowVariancesDenseColVectorRewritesCP() {
        testRowVariances(TEST_NAME, false, VectorType.COLUMNVECTOR, true, ExecType.CP);
    }

    @Test
    public void testRowVariancesDenseColVectorRewritesSpark() {
        testRowVariances(TEST_NAME, false, VectorType.COLUMNVECTOR, true, ExecType.SPARK);
    }

    @Test
    public void testRowVariancesDenseColVectorRewritesMR() {
        testRowVariances(TEST_NAME, false, VectorType.COLUMNVECTOR, true, ExecType.MR);
    }

    // Dense vector w/o rewrites
    //  - Row vector
    @Test
    public void testRowVariancesDenseRowVectorNoRewritesCP() {
        testRowVariances(TEST_NAME, false, VectorType.ROWVECTOR, false, ExecType.CP);
    }

    @Test
    public void testRowVariancesDenseRowVectorNoRewritesSpark() {
        testRowVariances(TEST_NAME, false, VectorType.ROWVECTOR, false, ExecType.SPARK);
    }

    @Test
    public void testRowVariancesDenseRowVectorNoRewritesMR() {
        testRowVariances(TEST_NAME, false, VectorType.ROWVECTOR, false, ExecType.MR);
    }

    //  - Column vector
    @Test
    public void testRowVariancesDenseColVectorNoRewritesCP() {
        testRowVariances(TEST_NAME, false, VectorType.COLUMNVECTOR, false, ExecType.CP);
    }

    @Test
    public void testRowVariancesDenseColVectorNoRewritesSpark() {
        testRowVariances(TEST_NAME, false, VectorType.COLUMNVECTOR, false, ExecType.SPARK);
    }

    @Test
    public void testRowVariancesDenseColVectorNoRewritesMR() {
        testRowVariances(TEST_NAME, false, VectorType.COLUMNVECTOR, false, ExecType.MR);
    }

    // Sparse matrix w/ rewrites
    @Test
    public void testRowVariancesSparseMatrixRewritesCP() {
        testRowVariances(TEST_NAME, true, VectorType.NONE, true, ExecType.CP);
    }

    @Test
    public void testRowVariancesSparseMatrixRewritesSpark() {
        testRowVariances(TEST_NAME, true, VectorType.NONE, true, ExecType.SPARK);
    }

    @Test
    public void testRowVariancesSparseMatrixRewritesMR() {
        testRowVariances(TEST_NAME, true, VectorType.NONE, true, ExecType.MR);
    }

    // Sparse matrix w/o rewrites
    @Test
    public void testRowVariancesSparseMatrixNoRewritesCP() {
        testRowVariances(TEST_NAME, true, VectorType.NONE, false, ExecType.CP);
    }

    @Test
    public void testRowVariancesSparseMatrixNoRewritesSpark() {
        testRowVariances(TEST_NAME, true, VectorType.NONE, false, ExecType.SPARK);
    }

    @Test
    public void testRowVariancesSparseMatrixNoRewritesMR() {
        testRowVariances(TEST_NAME, true, VectorType.NONE, false, ExecType.MR);
    }

    // Sparse vector w/ rewrites
    //  - Row vector
    @Test
    public void testRowVariancesSparseRowVectorRewritesCP() {
        testRowVariances(TEST_NAME, true, VectorType.ROWVECTOR, true, ExecType.CP);
    }

    @Test
    public void testRowVariancesSparseRowVectorRewritesSpark() {
        testRowVariances(TEST_NAME, true, VectorType.ROWVECTOR, true, ExecType.SPARK);
    }

    @Test
    public void testRowVariancesSparseRowVectorRewritesMR() {
        testRowVariances(TEST_NAME, true, VectorType.ROWVECTOR, true, ExecType.MR);
    }

    //  - Column vector
    @Test
    public void testRowVariancesSparseColVectorRewritesCP() {
        testRowVariances(TEST_NAME, true, VectorType.COLUMNVECTOR, true, ExecType.CP);
    }

    @Test
    public void testRowVariancesSparseColVectorRewritesSpark() {
        testRowVariances(TEST_NAME, true, VectorType.COLUMNVECTOR, true, ExecType.SPARK);
    }

    @Test
    public void testRowVariancesSparseColVectorRewritesMR() {
        testRowVariances(TEST_NAME, true, VectorType.COLUMNVECTOR, true, ExecType.MR);
    }

    // Sparse vector w/o rewrites
    //  - Row vector
    @Test
    public void testRowVariancesSparseRowVectorNoRewritesCP() {
        testRowVariances(TEST_NAME, true, VectorType.ROWVECTOR, false, ExecType.CP);
    }

    @Test
    public void testRowVariancesSparseRowVectorNoRewritesSpark() {
        testRowVariances(TEST_NAME, true, VectorType.ROWVECTOR, false, ExecType.SPARK);
    }

    @Test
    public void testRowVariancesSparseRowVectorNoRewritesMR() {
        testRowVariances(TEST_NAME, true, VectorType.ROWVECTOR, false, ExecType.MR);
    }

    //  - Column vector
    @Test
    public void testRowVariancesSparseColVectorNoRewritesCP() {
        testRowVariances(TEST_NAME, true, VectorType.COLUMNVECTOR, false, ExecType.CP);
    }

    @Test
    public void testRowVariancesSparseColVectorNoRewritesSpark() {
        testRowVariances(TEST_NAME, true, VectorType.COLUMNVECTOR, false, ExecType.SPARK);
    }

    @Test
    public void testRowVariancesSparseColVectorNoRewritesMR() {
        testRowVariances(TEST_NAME, true, VectorType.COLUMNVECTOR, false, ExecType.MR);
    }

    /**
     * Test the row variances function, "rowVars(X)", on
     * dense/sparse matrices/vectors on the CP/Spark/MR platforms.
     *
     * @param testName The name of this test case.
     * @param sparse Whether or not the matrix/vector should be sparse.
     * @param vectorType Selection between a matrix, a row vector, and
     *                   a column vector.
     * @param rewrites Whether or not to employ algebraic rewrites.
     * @param platform Selection between CP/Spark/MR platforms.
     */
    private void testRowVariances(String testName, boolean sparse, VectorType vectorType,
                                  boolean rewrites, ExecType platform) {
        // Configure settings for this test case
        boolean rewritesOld = OptimizerUtils.ALLOW_ALGEBRAIC_SIMPLIFICATION;
        OptimizerUtils.ALLOW_ALGEBRAIC_SIMPLIFICATION = rewrites;

        RUNTIME_PLATFORM platformOld = rtplatform;
        switch (platform) {
            case MR:
                rtplatform = RUNTIME_PLATFORM.HADOOP;
                break;
            case SPARK:
                rtplatform = RUNTIME_PLATFORM.SPARK;
                break;
            default:
                rtplatform = RUNTIME_PLATFORM.SINGLE_NODE;
                break;
        }

        boolean sparkConfigOld = DMLScript.USE_LOCAL_SPARK_CONFIG;
        if (rtplatform == RUNTIME_PLATFORM.SPARK)
            DMLScript.USE_LOCAL_SPARK_CONFIG = true;

        try {
            // Create and load test configuration
            getAndLoadTestConfiguration(testName);
            String HOME = SCRIPT_DIR + TEST_DIR;
            fullDMLScriptName = HOME + testName + ".dml";
            programArgs = new String[]{"-explain", "-stats", "-args",
                    input(INPUT_NAME), output(OUTPUT_NAME)};
            fullRScriptName = HOME + testName + ".R";
            rCmd = "Rscript" + " " + fullRScriptName + " " + inputDir() + " " + expectedDir();

            // Generate data
            double sparsity = sparse ? sparsity2 : sparsity1;
            int r;
            int c;
            switch (vectorType) {
                case ROWVECTOR:
                    r = 1;
                    c = cols;
                    break;
                case COLUMNVECTOR:
                    r = rows;
                    c = 1;
                    break;
                case NONE:
                default:
                    r = rows;
                    c = cols;
            }
            double[][] X = getRandomMatrix(r, c, -1, 1, sparsity, 7);
            writeInputMatrixWithMTD(INPUT_NAME, X, true);

            // Run DML and R scripts
            runTest(true, false, null, -1);
            runRScript(true);

            // Compare output matrices
            HashMap<CellIndex, Double> dmlfile = readDMLMatrixFromHDFS(OUTPUT_NAME);
            HashMap<CellIndex, Double> rfile  = readRMatrixFromFS(OUTPUT_NAME);
            TestUtils.compareMatrices(dmlfile, rfile, eps, "Stat-DML", "Stat-R");

            // On CP and Spark modes, check that the following
            // rewrites occurred:
            //  - rowVars(X), where X is a row vector, should be
            //    rewritten to var(X).
            //  - rowVars(X), where X is a column vector, should be
            //    rewritten to an empty row vector of zeros.
            if (rewrites && (platform == ExecType.SPARK || platform == ExecType.CP)) {
                String prefix = (platform == ExecType.SPARK) ? Instruction.SP_INST_PREFIX : "";
                if (vectorType == VectorType.ROWVECTOR) {
                    String opcode = prefix + varOp;
                    boolean rewriteApplied = Statistics.getCPHeavyHitterOpCodes().contains(opcode);
                    Assert.assertTrue("Rewrite not applied to row vector case.", rewriteApplied);

                } else if (vectorType == VectorType.COLUMNVECTOR) {
                    String opcode = prefix + rowVarOp;
                    boolean rewriteApplied = !Statistics.getCPHeavyHitterOpCodes().contains(opcode);
                    Assert.assertTrue("Rewrite not applied to column vector case.", rewriteApplied);
                }
            }
        }
        finally {
            // Reset settings
            OptimizerUtils.ALLOW_ALGEBRAIC_SIMPLIFICATION = rewritesOld;
            rtplatform = platformOld;
            DMLScript.USE_LOCAL_SPARK_CONFIG = sparkConfigOld;
        }
    }
}
