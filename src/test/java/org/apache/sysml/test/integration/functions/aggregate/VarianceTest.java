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
import org.apache.sysml.lops.LopProperties.ExecType;
import org.apache.sysml.runtime.matrix.data.MatrixValue.CellIndex;
import org.apache.sysml.test.integration.AutomatedTestBase;
import org.apache.sysml.test.integration.TestConfiguration;
import org.apache.sysml.test.utils.TestUtils;
import org.junit.Test;

import java.util.HashMap;

/**
 * Test the variance function, "var(X)".
 */
public class VarianceTest extends AutomatedTestBase {

    private static final String TEST_NAME = "Variance";
    private static final String TEST_DIR = "functions/aggregate/";
    private static final String TEST_CLASS_DIR =
            TEST_DIR + VarianceTest.class.getSimpleName() + "/";
    private static final String INPUT_NAME = "X";
    private static final String OUTPUT_NAME = "variance";

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

    // Dense matrix
    @Test
    public void testVarianceDenseMatrixCP() {
        testVariance(TEST_NAME, false, VectorType.NONE, ExecType.CP);
    }

    @Test
    public void testVarianceDenseMatrixSpark() {
        testVariance(TEST_NAME, false, VectorType.NONE, ExecType.SPARK);
    }

    @Test
    public void testVarianceDenseMatrixMR() {
        testVariance(TEST_NAME, false, VectorType.NONE, ExecType.MR);
    }

    // Dense vector
    //  - Row vector
    @Test
    public void testVarianceDenseRowVectorCP() {
        testVariance(TEST_NAME, false, VectorType.ROWVECTOR, ExecType.CP);
    }

    @Test
    public void testVarianceDenseRowVectorSpark() {
        testVariance(TEST_NAME, false, VectorType.ROWVECTOR, ExecType.SPARK);
    }

    @Test
    public void testVarianceDenseRowVectorMR() {
        testVariance(TEST_NAME, false, VectorType.ROWVECTOR, ExecType.MR);
    }

    //  - Column vector
    @Test
    public void testVarianceDenseColVectorCP() {
        testVariance(TEST_NAME, false, VectorType.COLUMNVECTOR, ExecType.CP);
    }

    @Test
    public void testVarianceDenseColVectorSpark() {
        testVariance(TEST_NAME, false, VectorType.COLUMNVECTOR, ExecType.SPARK);
    }

    @Test
    public void testVarianceDenseColVectorMR() {
        testVariance(TEST_NAME, false, VectorType.COLUMNVECTOR, ExecType.MR);
    }

    // Sparse matrix
    @Test
    public void testVarianceSparseMatrixCP() {
        testVariance(TEST_NAME, true, VectorType.NONE, ExecType.CP);
    }

    @Test
    public void testVarianceSparseMatrixSpark() {
        testVariance(TEST_NAME, true, VectorType.NONE, ExecType.SPARK);
    }

    @Test
    public void testVarianceSparseMatrixMR() {
        testVariance(TEST_NAME, true, VectorType.NONE, ExecType.MR);
    }

    // Sparse vector
    //  - Row vector
    @Test
    public void testVarianceSparseRowVectorCP() {
        testVariance(TEST_NAME, true, VectorType.ROWVECTOR, ExecType.CP);
    }

    @Test
    public void testVarianceSparseRowVectorSpark() {
        testVariance(TEST_NAME, true, VectorType.ROWVECTOR, ExecType.SPARK);
    }

    @Test
    public void testVarianceSparseRowVectorMR() {
        testVariance(TEST_NAME, true, VectorType.ROWVECTOR, ExecType.MR);
    }

    //  - Column vector
    @Test
    public void testVarianceSparseColVectorCP() {
        testVariance(TEST_NAME, true, VectorType.COLUMNVECTOR, ExecType.CP);
    }

    @Test
    public void testVarianceSparseColVectorSpark() {
        testVariance(TEST_NAME, true, VectorType.COLUMNVECTOR, ExecType.SPARK);
    }

    @Test
    public void testVarianceSparseColVectorMR() {
        testVariance(TEST_NAME, true, VectorType.COLUMNVECTOR, ExecType.MR);
    }
    
    /**
     * Test the variance function, "var(X)", on
     * dense/sparse matrices/vectors on the CP/Spark/MR platforms.
     *
     * @param testName The name of this test case.
     * @param sparse Whether or not the matrix/vector should be sparse.
     * @param vectorType Selection between a matrix, a row vector, and
     *                   a column vector.
     * @param platform Selection between CP/Spark/MR platforms.
     */
    private void testVariance(String testName, boolean sparse, VectorType vectorType,
                              ExecType platform) {
        // Configure settings for this test case
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
        }
        finally {
            // Reset settings
            rtplatform = platformOld;
            DMLScript.USE_LOCAL_SPARK_CONFIG = sparkConfigOld;
        }
    }
}
