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

package org.apache.sysds.test.functions.rewrite;

import org.apache.sysds.common.Opcodes;
import org.apache.sysds.utils.Statistics;
import org.apache.sysds.test.AutomatedTestBase;
import org.apache.sysds.test.TestConfiguration;
import org.apache.sysds.test.TestUtils;
import org.junit.Assert;
import org.junit.Test;
import org.apache.sysds.hops.OptimizerUtils;
import java.util.Arrays;

/**
 * Test for the rewrite that replaces a sequence of X = floor(M * sf) Y = compress(X) to a fused quantize_compress(M,
 * sf).
 * 
 */
public class RewriteQuantizationFusedCompressionTest extends AutomatedTestBase {
    private static final String TEST_NAME1 = "RewriteQuantizationFusedCompressionScalar";
    private static final String TEST_NAME2 = "RewriteQuantizationFusedCompressionMatrix";
    private static final String TEST_DIR = "functions/rewrite/";
    private static final String TEST_CLASS_DIR = TEST_DIR
        + RewriteQuantizationFusedCompressionTest.class.getSimpleName() + "/";

    private static final int rows = 500;
    private static final int cols = 500;
    private static final double sfValue = 0.5; // Value used to fill the scale factor matrix or as a standalone scalar

    @Override
    public void setUp() {
        TestUtils.clearAssertionInformation();
        addTestConfiguration(TEST_NAME1, new TestConfiguration(TEST_CLASS_DIR, TEST_NAME1, new String[] {"R"}));
        addTestConfiguration(TEST_NAME2, new TestConfiguration(TEST_CLASS_DIR, TEST_NAME2, new String[] {"R"}));
    }

    @Test
    public void testRewriteQuantizationFusedCompressionScalar() {
        testRewriteQuantizationFusedCompression(TEST_NAME1, true, true);
    }

    @Test
    public void testRewriteQuantizationFusedCompressionNoRewriteScalar() {
        testRewriteQuantizationFusedCompression(TEST_NAME1, false, true);
    }

    @Test
    public void testRewriteQuantizationFusedCompression() {
        testRewriteQuantizationFusedCompression(TEST_NAME2, true, false);
    }

    @Test
    public void testRewriteQuantizationFusedCompressionNoRewrite() {
        testRewriteQuantizationFusedCompression(TEST_NAME2, false, false);
    }

    /**
     * Unified method to test both scalar and matrix scale factors.
     * 
     * @param testname Test name
     * @param rewrites Whether to enable fusion rewrites
     * @param isScalar Whether the scale factor is a scalar or a matrix
     */
    private void testRewriteQuantizationFusedCompression(String testname, boolean rewrites, boolean isScalar) {
        boolean oldRewriteFlag = OptimizerUtils.ALLOW_QUANTIZE_COMPRESS_REWRITE;
        OptimizerUtils.ALLOW_QUANTIZE_COMPRESS_REWRITE = rewrites;

        try {
            TestConfiguration config = getTestConfiguration(testname);
            loadTestConfiguration(config);

            String HOME = SCRIPT_DIR + TEST_DIR;
            fullDMLScriptName = HOME + testname + ".dml";

            double[][] A = getRandomMatrix(rows, cols, -1, 1, 0.70d, 7);

            String[] programArgs;
            if(isScalar) {
                // Scalar case: pass sfValue as a string
                String s = Double.toString(sfValue);
                programArgs = new String[] {"-stats", "-args", input("A"), s, output("R")};
                writeInputMatrixWithMTD("A", A, 174522, false);
            }
            else {
                // Matrix case: pass S as a separate matrix
                double[][] S = new double[rows][1];
                for(int i = 0; i < rows; i++) {
                    S[i][0] = sfValue;
                }
                programArgs = new String[] {"-stats", "-args", input("A"), input("S"), output("R")};
                writeInputMatrixWithMTD("A", A, 174522, false);
                writeInputMatrixWithMTD("S", S, 500, false);
            }

            this.programArgs = programArgs;
            runTest(true, false, null, -1);

            // Simple check if quantization indeed occured by computing expected sum
            // Even if compression is aborted, the quantization step should still take effect
            double expectedR = Arrays.stream(A).flatMapToDouble(Arrays::stream).map(x -> Math.floor(x * sfValue)).sum();
            double actualR = TestUtils.readDMLScalar(output("R"));

            Assert.assertEquals("Mismatch in expected sum after quantization and compression", expectedR, actualR, 0.0);

            // Check if fusion occurred
            if(rewrites) {
                Assert.assertEquals("Expected fused operation count mismatch", 1,
                    Statistics.getCPHeavyHitterCount(Opcodes.QUANTIZE_COMPRESS.toString()));
                Assert.assertEquals("Expected no separate floor op", 0,
                    Statistics.getCPHeavyHitterCount(Opcodes.FLOOR.toString()));
                Assert.assertEquals("Expected no separate compress op", 0,
                    Statistics.getCPHeavyHitterCount(Opcodes.COMPRESS.toString()));
                Assert.assertEquals("Expected no separate multiplication op", 0,
                    Statistics.getCPHeavyHitterCount(Opcodes.MULT.toString()));
            }
            else {
                Assert.assertEquals("Expected no fused op", 0,
                    Statistics.getCPHeavyHitterCount(Opcodes.QUANTIZE_COMPRESS.toString()));
                Assert.assertEquals("Expected separate floor op", 1,
                    Statistics.getCPHeavyHitterCount(Opcodes.FLOOR.toString()));
                Assert.assertEquals("Expected separate compress op", 1,
                    Statistics.getCPHeavyHitterCount(Opcodes.COMPRESS.toString()));
                Assert.assertEquals("Expected separate multiplication op", 1,
                    Statistics.getCPHeavyHitterCount(Opcodes.MULT.toString()));
            }
        }
        finally {
            OptimizerUtils.ALLOW_QUANTIZE_COMPRESS_REWRITE = oldRewriteFlag;
        }
    }
}
