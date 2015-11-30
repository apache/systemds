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
 * Test the column standard deviations function, "colSds(X)".
 */
public class ColStdDevsTest extends AutomatedTestBase {

    private static final String TEST_NAME = "ColStdDevs";
    private static final String TEST_DIR = "functions/aggregate/";
    private static final String TEST_CLASS_DIR =
            TEST_DIR + ColStdDevsTest.class.getSimpleName() + "/";
    private static final String INPUT_NAME = "X";
    private static final String OUTPUT_NAME = "colStdDevs";

    private static final int rows = 1234;
    private static final int cols1 = 1;
    private static final int cols2 = 567;
    private static final double sparsity1 = 1;
    private static final double sparsity2 = 0.2;
    private static final double eps = Math.pow(10, -10);

    @Override
    public void setUp() {
        TestUtils.clearAssertionInformation();
        TestConfiguration config = new TestConfiguration(TEST_CLASS_DIR, TEST_NAME);
        addTestConfiguration(TEST_NAME, config);
    }

    // Dense matrix
    @Test
    public void testColStdDevsDenseMatrixCP() {
        testColStdDevs(TEST_NAME, false, false, ExecType.CP);
    }

    @Test
    public void testColStdDevsDenseMatrixSpark() {
        testColStdDevs(TEST_NAME, false, false, ExecType.SPARK);
    }

    @Test
    public void testColStdDevsDenseMatrixMR() {
        testColStdDevs(TEST_NAME, false, false, ExecType.MR);
    }

    // Dense vector
    @Test
    public void testColStdDevsDenseVectorCP() {
        testColStdDevs(TEST_NAME, false, true, ExecType.CP);
    }

    @Test
    public void testColStdDevsDenseVectorSpark() {
        testColStdDevs(TEST_NAME, false, true, ExecType.SPARK);
    }

    @Test
    public void testColStdDevsDenseVectorMR() {
        testColStdDevs(TEST_NAME, false, true, ExecType.MR);
    }

    // Sparse matrix
    @Test
    public void testColStdDevsSparseMatrixCP() {
        testColStdDevs(TEST_NAME, true, false, ExecType.CP);
    }

    @Test
    public void testColStdDevsSparseMatrixSpark() {
        testColStdDevs(TEST_NAME, true, false, ExecType.SPARK);
    }

    @Test
    public void testColStdDevsSparseMatrixMR() {
        testColStdDevs(TEST_NAME, true, false, ExecType.MR);
    }

    // Sparse vector
    @Test
    public void testColStdDevsSparseVectorCP() {
        testColStdDevs(TEST_NAME, true, true, ExecType.CP);
    }

    @Test
    public void testColStdDevsSparseVectorSpark() {
        testColStdDevs(TEST_NAME, true, true, ExecType.SPARK);
    }

    @Test
    public void testColStdDevsSparseVectorMR() {
        testColStdDevs(TEST_NAME, true, true, ExecType.MR);
    }

    /**
     * Test the column standard deviations function, "colSds(X)", on
     * dense/sparse matrices/vectors on the CP/Spark/MR platforms.
     *
     * @param testName The name of this test case.
     * @param sparse Whether or not the matrix/vector should be sparse.
     * @param vector Boolean value choosing between a vector and a matrix.
     * @param platform Selection between CP/Spark/MR platforms.
     */
    private void testColStdDevs(String testName, boolean sparse, boolean vector,
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
            int cols = vector ? cols1 : cols2;
            double[][] X = getRandomMatrix(rows, cols, -1, 1, sparsity, 7);
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
