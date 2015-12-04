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
 * Test for the column sums of squared values function, "colSums(X^2)".
 */
public class ColSumsSqTest extends AutomatedTestBase {

    private static final String TEST_NAME = "ColSumsSq";
    private static final String TEST_DIR = "functions/aggregate/";
    private static final String INPUT_NAME = "X";
    private static final String OUTPUT_NAME = "colSumsSq";

    private static final String op = "uacsqk+";
    private static final int rows = 1234;
    private static final int cols = 567;
    private static final double sparsity1 = 1;
    private static final double sparsity2 = 0.2;
    private static final double eps = Math.pow(10, -10);

    @Override
    public void setUp() {
        TestUtils.clearAssertionInformation();
        TestConfiguration config = new TestConfiguration(TEST_DIR, TEST_NAME);
        addTestConfiguration(TEST_NAME, config);
    }

    // Dense matrix w/ rewrites
    @Test
    public void testColSumsSquaredDenseMatrixRewriteCP() {
        testColSumsSquared(TEST_NAME, false, false, true, ExecType.CP);
    }

    @Test
    public void testColSumsSquaredDenseMatrixRewriteSpark() {
        testColSumsSquared(TEST_NAME, false, false, true, ExecType.SPARK);
    }

    @Test
    public void testColSumsSquaredDenseMatrixRewriteMR() {
        testColSumsSquared(TEST_NAME, false, false, true, ExecType.MR);
    }

    // Dense matrix w/o rewrites
    @Test
    public void testColSumsSquaredDenseMatrixNoRewriteCP() {
        testColSumsSquared(TEST_NAME, false, false, false, ExecType.CP);
    }

    @Test
    public void testColSumsSquaredDenseMatrixNoRewriteSpark() {
        testColSumsSquared(TEST_NAME, false, false, false, ExecType.SPARK);
    }

    @Test
    public void testColSumsSquaredDenseMatrixNoRewriteMR() {
        testColSumsSquared(TEST_NAME, false, false, false, ExecType.MR);
    }

    // Dense vector w/ rewrites
    @Test
    public void testColSumsSquaredDenseVectorRewriteCP() {
        testColSumsSquared(TEST_NAME, false, true, true, ExecType.CP);
    }

    @Test
    public void testColSumsSquaredDenseVectorRewriteSpark() {
        testColSumsSquared(TEST_NAME, false, true, true, ExecType.SPARK);
    }

    @Test
    public void testColSumsSquaredDenseVectorRewriteMR() {
        testColSumsSquared(TEST_NAME, false, true, true, ExecType.MR);
    }

    // Dense vector w/o rewrites
    @Test
    public void testColSumsSquaredDenseVectorNoRewriteCP() {
        testColSumsSquared(TEST_NAME, false, true, false, ExecType.CP);
    }

    @Test
    public void testColSumsSquaredDenseVectorNoRewriteSpark() {
        testColSumsSquared(TEST_NAME, false, true, false, ExecType.SPARK);
    }

    @Test
    public void testColSumsSquaredDenseVectorNoRewriteMR() {
        testColSumsSquared(TEST_NAME, false, true, false, ExecType.MR);
    }

    // Sparse matrix w/ rewrites
    @Test
    public void testColSumsSquaredSparseMatrixRewriteCP() {
        testColSumsSquared(TEST_NAME, true, false, true, ExecType.CP);
    }

    @Test
    public void testColSumsSquaredSparseMatrixRewriteSpark() {
        testColSumsSquared(TEST_NAME, true, false, true, ExecType.SPARK);
    }

    @Test
    public void testColSumsSquaredSparseMatrixRewriteMR() {
        testColSumsSquared(TEST_NAME, true, false, true, ExecType.MR);
    }

    // Sparse matrix w/o rewrites
    @Test
    public void testColSumsSquaredSparseMatrixNoRewriteCP() {
        testColSumsSquared(TEST_NAME, true, false, false, ExecType.CP);
    }

    @Test
    public void testColSumsSquaredSparseMatrixNoRewriteSpark() {
        testColSumsSquared(TEST_NAME, true, false, false, ExecType.SPARK);
    }

    @Test
    public void testColSumsSquaredSparseMatrixNoRewriteMR() {
        testColSumsSquared(TEST_NAME, true, false, false, ExecType.MR);
    }

    // Sparse vector w/ rewrites
    @Test
    public void testColSumsSquaredSparseVectorRewriteCP() {
        testColSumsSquared(TEST_NAME, true, true, true, ExecType.CP);
    }

    @Test
    public void testColSumsSquaredSparseVectorRewriteSpark() {
        testColSumsSquared(TEST_NAME, true, true, true, ExecType.SPARK);
    }

    @Test
    public void testColSumsSquaredSparseVectorRewriteMR() {
        testColSumsSquared(TEST_NAME, true, true, true, ExecType.MR);
    }

    // Sparse vector w/o rewrites
    @Test
    public void testColSumsSquaredSparseVectorNoRewriteCP() {
        testColSumsSquared(TEST_NAME, true, true, false, ExecType.CP);
    }

    @Test
    public void testColSumsSquaredSparseVectorNoRewriteSpark() {
        testColSumsSquared(TEST_NAME, true, true, false, ExecType.SPARK);
    }

    @Test
    public void testColSumsSquaredSparseVectorNoRewriteMR() {
        testColSumsSquared(TEST_NAME, true, true, false, ExecType.MR);
    }

    /**
     * Test the column sums of squared values function, "colSums(X^2)",
     * on dense/sparse matrices/vectors with rewrites/no rewrites on
     * the CP/Spark/MR platforms.
     *
     * @param testName The name of this test case.
     * @param sparse Whether or not the matrix/vector should be sparse.
     * @param vector Boolean value choosing between a vector and a matrix.
     * @param rewrites Whether or not to employ algebraic rewrites.
     * @param platform Selection between CP/Spark/MR platforms.
     */
    private void testColSumsSquared(String testName, boolean sparse, boolean vector,
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
            TestConfiguration config = getTestConfiguration(testName);
            String HOME = SCRIPT_DIR + TEST_DIR;
            fullDMLScriptName = HOME + testName + ".dml";
            programArgs = new String[]{"-explain", "-stats", "-args",
                    HOME + INPUT_DIR + INPUT_NAME,
                    HOME + OUTPUT_DIR + OUTPUT_NAME};
            fullRScriptName = HOME + testName + ".R";
            rCmd = "Rscript" + " " + fullRScriptName + " " +
                    HOME + INPUT_DIR + " " + HOME + EXPECTED_DIR;
            loadTestConfiguration(config);

            // Generate data
            double sparsity = sparse ? sparsity2 : sparsity1;
            int columns = vector ? 1 : cols;
            double[][] X = getRandomMatrix(rows, columns, -1, 1, sparsity, 7);
            writeInputMatrixWithMTD(INPUT_NAME, X, true);

            // Run DML and R scripts
            runTest(true, false, null, -1);
            runRScript(true);

            // Compare output matrices
            HashMap<CellIndex, Double> dmlfile = readDMLMatrixFromHDFS(OUTPUT_NAME);
            HashMap<CellIndex, Double> rfile  = readRMatrixFromFS(OUTPUT_NAME);
            TestUtils.compareMatrices(dmlfile, rfile, eps, "Stat-DML", "Stat-R");

            // On CP and Spark modes, check that the rewrite actually
            // occurred for matrix cases and not for vector cases.
            if (rewrites && (platform == ExecType.SPARK || platform == ExecType.CP)) {
                String prefix = (platform == ExecType.SPARK) ? Instruction.SP_INST_PREFIX : "";
                String opcode = prefix + op;
                boolean rewriteApplied = Statistics.getCPHeavyHitterOpCodes().contains(opcode);
                if (vector)
                    Assert.assertFalse("Rewrite applied to vector case.", rewriteApplied);
                else
                    Assert.assertTrue("Rewrite not applied to matrix case.", rewriteApplied);
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
