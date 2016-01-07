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
    private static final int cols = 1432;
    private static final double sparsitySparse = 0.2;
    private static final double sparsityDense = 0.7;
    private static final double eps = Math.pow(10, -10);

    private enum Sparsity {EMPTY, SPARSE, DENSE}
    private enum DataType {MATRIX, ROWVECTOR, COLUMNVECTOR}

    @Override
    public void setUp() {
        TestUtils.clearAssertionInformation();
        TestConfiguration config = new TestConfiguration(TEST_CLASS_DIR, TEST_NAME);
        addTestConfiguration(TEST_NAME, config);
    }

    // Dense matrix w/ rewrites
    @Test
    public void testRowVariancesDenseMatrixRewritesCP() {
        testRowVariances(TEST_NAME, Sparsity.DENSE, DataType.MATRIX, true, ExecType.CP);
    }

    @Test
    public void testRowVariancesDenseMatrixRewritesSpark() {
        testRowVariances(TEST_NAME, Sparsity.DENSE, DataType.MATRIX, true, ExecType.SPARK);
    }

    @Test
    public void testRowVariancesDenseMatrixRewritesMR() {
        testRowVariances(TEST_NAME, Sparsity.DENSE, DataType.MATRIX, true, ExecType.MR);
    }

    // Dense matrix w/o rewrites
    @Test
    public void testRowVariancesDenseMatrixNoRewritesCP() {
        testRowVariances(TEST_NAME, Sparsity.DENSE, DataType.MATRIX, false, ExecType.CP);
    }

    @Test
    public void testRowVariancesDenseMatrixNoRewritesSpark() {
        testRowVariances(TEST_NAME, Sparsity.DENSE, DataType.MATRIX, false, ExecType.SPARK);
    }

    @Test
    public void testRowVariancesDenseMatrixNoRewritesMR() {
        testRowVariances(TEST_NAME, Sparsity.DENSE, DataType.MATRIX, false, ExecType.MR);
    }

    // Dense vector w/ rewrites
    //  - Row vector
    @Test
    public void testRowVariancesDenseRowVectorRewritesCP() {
        testRowVariances(TEST_NAME, Sparsity.DENSE, DataType.ROWVECTOR, true, ExecType.CP);
    }

    @Test
    public void testRowVariancesDenseRowVectorRewritesSpark() {
        testRowVariances(TEST_NAME, Sparsity.DENSE, DataType.ROWVECTOR, true, ExecType.SPARK);
    }

    @Test
    public void testRowVariancesDenseRowVectorRewritesMR() {
        testRowVariances(TEST_NAME, Sparsity.DENSE, DataType.ROWVECTOR, true, ExecType.MR);
    }

    //  - Column vector
    @Test
    public void testRowVariancesDenseColVectorRewritesCP() {
        testRowVariances(TEST_NAME, Sparsity.DENSE, DataType.COLUMNVECTOR, true, ExecType.CP);
    }

    @Test
    public void testRowVariancesDenseColVectorRewritesSpark() {
        testRowVariances(TEST_NAME, Sparsity.DENSE, DataType.COLUMNVECTOR, true, ExecType.SPARK);
    }

    @Test
    public void testRowVariancesDenseColVectorRewritesMR() {
        testRowVariances(TEST_NAME, Sparsity.DENSE, DataType.COLUMNVECTOR, true, ExecType.MR);
    }

    // Dense vector w/o rewrites
    //  - Row vector
    @Test
    public void testRowVariancesDenseRowVectorNoRewritesCP() {
        testRowVariances(TEST_NAME, Sparsity.DENSE, DataType.ROWVECTOR, false, ExecType.CP);
    }

    @Test
    public void testRowVariancesDenseRowVectorNoRewritesSpark() {
        testRowVariances(TEST_NAME, Sparsity.DENSE, DataType.ROWVECTOR, false, ExecType.SPARK);
    }

    @Test
    public void testRowVariancesDenseRowVectorNoRewritesMR() {
        testRowVariances(TEST_NAME, Sparsity.DENSE, DataType.ROWVECTOR, false, ExecType.MR);
    }

    //  - Column vector
    @Test
    public void testRowVariancesDenseColVectorNoRewritesCP() {
        testRowVariances(TEST_NAME, Sparsity.DENSE, DataType.COLUMNVECTOR, false, ExecType.CP);
    }

    @Test
    public void testRowVariancesDenseColVectorNoRewritesSpark() {
        testRowVariances(TEST_NAME, Sparsity.DENSE, DataType.COLUMNVECTOR, false, ExecType.SPARK);
    }

    @Test
    public void testRowVariancesDenseColVectorNoRewritesMR() {
        testRowVariances(TEST_NAME, Sparsity.DENSE, DataType.COLUMNVECTOR, false, ExecType.MR);
    }

    // Sparse matrix w/ rewrites
    @Test
    public void testRowVariancesSparseMatrixRewritesCP() {
        testRowVariances(TEST_NAME, Sparsity.SPARSE, DataType.MATRIX, true, ExecType.CP);
    }

    @Test
    public void testRowVariancesSparseMatrixRewritesSpark() {
        testRowVariances(TEST_NAME, Sparsity.SPARSE, DataType.MATRIX, true, ExecType.SPARK);
    }

    @Test
    public void testRowVariancesSparseMatrixRewritesMR() {
        testRowVariances(TEST_NAME, Sparsity.SPARSE, DataType.MATRIX, true, ExecType.MR);
    }

    // Sparse matrix w/o rewrites
    @Test
    public void testRowVariancesSparseMatrixNoRewritesCP() {
        testRowVariances(TEST_NAME, Sparsity.SPARSE, DataType.MATRIX, false, ExecType.CP);
    }

    @Test
    public void testRowVariancesSparseMatrixNoRewritesSpark() {
        testRowVariances(TEST_NAME, Sparsity.SPARSE, DataType.MATRIX, false, ExecType.SPARK);
    }

    @Test
    public void testRowVariancesSparseMatrixNoRewritesMR() {
        testRowVariances(TEST_NAME, Sparsity.SPARSE, DataType.MATRIX, false, ExecType.MR);
    }

    // Sparse vector w/ rewrites
    //  - Row vector
    @Test
    public void testRowVariancesSparseRowVectorRewritesCP() {
        testRowVariances(TEST_NAME, Sparsity.SPARSE, DataType.ROWVECTOR, true, ExecType.CP);
    }

    @Test
    public void testRowVariancesSparseRowVectorRewritesSpark() {
        testRowVariances(TEST_NAME, Sparsity.SPARSE, DataType.ROWVECTOR, true, ExecType.SPARK);
    }

    @Test
    public void testRowVariancesSparseRowVectorRewritesMR() {
        testRowVariances(TEST_NAME, Sparsity.SPARSE, DataType.ROWVECTOR, true, ExecType.MR);
    }

    //  - Column vector
    @Test
    public void testRowVariancesSparseColVectorRewritesCP() {
        testRowVariances(TEST_NAME, Sparsity.SPARSE, DataType.COLUMNVECTOR, true, ExecType.CP);
    }

    @Test
    public void testRowVariancesSparseColVectorRewritesSpark() {
        testRowVariances(TEST_NAME, Sparsity.SPARSE, DataType.COLUMNVECTOR, true, ExecType.SPARK);
    }

    @Test
    public void testRowVariancesSparseColVectorRewritesMR() {
        testRowVariances(TEST_NAME, Sparsity.SPARSE, DataType.COLUMNVECTOR, true, ExecType.MR);
    }

    // Sparse vector w/o rewrites
    //  - Row vector
    @Test
    public void testRowVariancesSparseRowVectorNoRewritesCP() {
        testRowVariances(TEST_NAME, Sparsity.SPARSE, DataType.ROWVECTOR, false, ExecType.CP);
    }

    @Test
    public void testRowVariancesSparseRowVectorNoRewritesSpark() {
        testRowVariances(TEST_NAME, Sparsity.SPARSE, DataType.ROWVECTOR, false, ExecType.SPARK);
    }

    @Test
    public void testRowVariancesSparseRowVectorNoRewritesMR() {
        testRowVariances(TEST_NAME, Sparsity.SPARSE, DataType.ROWVECTOR, false, ExecType.MR);
    }

    //  - Column vector
    @Test
    public void testRowVariancesSparseColVectorNoRewritesCP() {
        testRowVariances(TEST_NAME, Sparsity.SPARSE, DataType.COLUMNVECTOR, false, ExecType.CP);
    }

    @Test
    public void testRowVariancesSparseColVectorNoRewritesSpark() {
        testRowVariances(TEST_NAME, Sparsity.SPARSE, DataType.COLUMNVECTOR, false, ExecType.SPARK);
    }

    @Test
    public void testRowVariancesSparseColVectorNoRewritesMR() {
        testRowVariances(TEST_NAME, Sparsity.SPARSE, DataType.COLUMNVECTOR, false, ExecType.MR);
    }

    // Empty matrix w/ rewrites
    @Test
    public void testRowVariancesEmptyMatrixRewritesCP() {
        testRowVariances(TEST_NAME, Sparsity.EMPTY, DataType.MATRIX, true, ExecType.CP);
    }

    @Test
    public void testRowVariancesEmptyMatrixRewritesSpark() {
        testRowVariances(TEST_NAME, Sparsity.EMPTY, DataType.MATRIX, true, ExecType.SPARK);
    }

    @Test
    public void testRowVariancesEmptyMatrixRewritesMR() {
        testRowVariances(TEST_NAME, Sparsity.EMPTY, DataType.MATRIX, true, ExecType.MR);
    }

    // Empty matrix w/o rewrites
    @Test
    public void testRowVariancesEmptyMatrixNoRewritesCP() {
        testRowVariances(TEST_NAME, Sparsity.EMPTY, DataType.MATRIX, false, ExecType.CP);
    }

    @Test
    public void testRowVariancesEmptyMatrixNoRewritesSpark() {
        testRowVariances(TEST_NAME, Sparsity.EMPTY, DataType.MATRIX, false, ExecType.SPARK);
    }

    @Test
    public void testRowVariancesEmptyMatrixNoRewritesMR() {
        testRowVariances(TEST_NAME, Sparsity.EMPTY, DataType.MATRIX, false, ExecType.MR);
    }

    // Empty vector w/ rewrites
    //  - Row vector
    @Test
    public void testRowVariancesEmptyRowVectorRewritesCP() {
        testRowVariances(TEST_NAME, Sparsity.EMPTY, DataType.ROWVECTOR, true, ExecType.CP);
    }

    @Test
    public void testRowVariancesEmptyRowVectorRewritesSpark() {
        testRowVariances(TEST_NAME, Sparsity.EMPTY, DataType.ROWVECTOR, true, ExecType.SPARK);
    }

    @Test
    public void testRowVariancesEmptyRowVectorRewritesMR() {
        testRowVariances(TEST_NAME, Sparsity.EMPTY, DataType.ROWVECTOR, true, ExecType.MR);
    }

    //  - Column vector
    @Test
    public void testRowVariancesEmptyColVectorRewritesCP() {
        testRowVariances(TEST_NAME, Sparsity.EMPTY, DataType.COLUMNVECTOR, true, ExecType.CP);
    }

    @Test
    public void testRowVariancesEmptyColVectorRewritesSpark() {
        testRowVariances(TEST_NAME, Sparsity.EMPTY, DataType.COLUMNVECTOR, true, ExecType.SPARK);
    }

    @Test
    public void testRowVariancesEmptyColVectorRewritesMR() {
        testRowVariances(TEST_NAME, Sparsity.EMPTY, DataType.COLUMNVECTOR, true, ExecType.MR);
    }

    // Empty vector w/o rewrites
    //  - Row vector
    @Test
    public void testRowVariancesEmptyRowVectorNoRewritesCP() {
        testRowVariances(TEST_NAME, Sparsity.EMPTY, DataType.ROWVECTOR, false, ExecType.CP);
    }

    @Test
    public void testRowVariancesEmptyRowVectorNoRewritesSpark() {
        testRowVariances(TEST_NAME, Sparsity.EMPTY, DataType.ROWVECTOR, false, ExecType.SPARK);
    }

    @Test
    public void testRowVariancesEmptyRowVectorNoRewritesMR() {
        testRowVariances(TEST_NAME, Sparsity.EMPTY, DataType.ROWVECTOR, false, ExecType.MR);
    }

    //  - Column vector
    @Test
    public void testRowVariancesEmptyColVectorNoRewritesCP() {
        testRowVariances(TEST_NAME, Sparsity.EMPTY, DataType.COLUMNVECTOR, false, ExecType.CP);
    }

    @Test
    public void testRowVariancesEmptyColVectorNoRewritesSpark() {
        testRowVariances(TEST_NAME, Sparsity.EMPTY, DataType.COLUMNVECTOR, false, ExecType.SPARK);
    }

    @Test
    public void testRowVariancesEmptyColVectorNoRewritesMR() {
        testRowVariances(TEST_NAME, Sparsity.EMPTY, DataType.COLUMNVECTOR, false, ExecType.MR);
    }

    /**
     * Test the row variances function, "rowVars(X)", on
     * dense/sparse matrices/vectors on the CP/Spark/MR platforms.
     *
     * @param testName The name of this test case.
     * @param sparsity Selection between empty, sparse, and dense data.
     * @param dataType Selection between a matrix, a row vector, and a
     *                 column vector.
     * @param rewrites Whether or not to employ algebraic rewrites.
     * @param platform Selection between CP/Spark/MR platforms.
     */
    private void testRowVariances(String testName, Sparsity sparsity, DataType dataType,
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
            // - sparsity
            double sparsityVal;
            switch (sparsity) {
                case EMPTY:
                    sparsityVal = 0;
                    break;
                case SPARSE:
                    sparsityVal = sparsitySparse;
                    break;
                case DENSE:
                default:
                    sparsityVal = sparsityDense;
            }
            // - size
            int r;
            int c;
            switch (dataType) {
                case ROWVECTOR:
                    r = 1;
                    c = cols;
                    break;
                case COLUMNVECTOR:
                    r = rows;
                    c = 1;
                    break;
                case MATRIX:
                default:
                    r = rows;
                    c = cols;
            }
            // - generation
            double[][] X = getRandomMatrix(r, c, -1, 1, sparsityVal, 7);
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
                if (dataType == DataType.ROWVECTOR) {
                    String opcode = prefix + varOp;
                    boolean rewriteApplied = Statistics.getCPHeavyHitterOpCodes().contains(opcode);
                    Assert.assertTrue("Rewrite not applied to row vector case.", rewriteApplied);

                } else if (dataType == DataType.COLUMNVECTOR) {
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
