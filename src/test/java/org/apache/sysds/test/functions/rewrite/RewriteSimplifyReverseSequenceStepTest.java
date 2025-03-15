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

import org.apache.sysds.hops.OptimizerUtils;
import org.apache.sysds.runtime.matrix.data.MatrixValue;
import org.apache.sysds.test.AutomatedTestBase;
import org.apache.sysds.test.TestConfiguration;
import org.apache.sysds.test.TestUtils;
import org.junit.Assert;
import org.junit.Test;

public class RewriteSimplifyReverseSequenceStepTest extends AutomatedTestBase {
    private static final String TEST_NAME1 = "RewriteSimplifyReverseSequenceStep";

    private static final String TEST_DIR = "functions/rewrite/";
    private static final String TEST_CLASS_DIR = TEST_DIR + RewriteSimplifyReverseSequenceStepTest.class.getSimpleName() + "/";

    @Override
    public void setUp() {
        TestUtils.clearAssertionInformation();
        addTestConfiguration(TEST_NAME1, new TestConfiguration(TEST_CLASS_DIR, TEST_NAME1, new String[]{"R"}));
    }

    @Test
    public void testRewriteReverseSeqStep() {
        testRewriteReverseSeq(TEST_NAME1, true);
    }

    @Test
    public void testNoRewriteReverseSeqStep() {
        testRewriteReverseSeq(TEST_NAME1, false);
    }

    private void testRewriteReverseSeq(String testname, boolean rewrites) {
        boolean oldFlag = OptimizerUtils.ALLOW_ALGEBRAIC_SIMPLIFICATION;
        int rows = 10;

        try {
            TestConfiguration config = getTestConfiguration(testname);
            loadTestConfiguration(config);

            String HOME = SCRIPT_DIR + TEST_DIR;
            fullDMLScriptName = HOME + testname + ".dml";
            programArgs = new String[]{"-stats", "-args", String.valueOf(rows), output("Scalar")};
            OptimizerUtils.ALLOW_ALGEBRAIC_SIMPLIFICATION = rewrites;

            runTest(true, false, null, -1);

            // Calculate expected sums for each sequence
            double sum1 = calculateSum(0, rows-1, 1);    // A1 = rev(seq(0, rows-1, 1))
            double sum2 = calculateSum(0, rows, 2);      // A2 = rev(seq(0, rows, 2))
            double sum3 = calculateSum(2, rows, 2);      // A3 = rev(seq(2, rows, 2))
            double sum4 = calculateSum(0, 100, 5);       // A4 = rev(seq(0, 100, 5))
            double sum5 = calculateSum(15, 5, -0.5);        // A5 = rev(seq(15, 5, -0.5))

            double expected = sum1 + sum2 + sum3 + sum4 + sum5;

            double ret = readDMLScalarFromOutputDir("Scalar").get(new MatrixValue.CellIndex(1, 1)).doubleValue();

            Assert.assertEquals("Incorrect sum computed", expected, ret, 1e-10);

            if (rewrites) {
                // With bidirectional rewrite, REV operations should be removed
                Assert.assertFalse("Rewrite should have removed REV operation!",
                        heavyHittersContainsString("rev"));
            }
        }
        finally {
            OptimizerUtils.ALLOW_ALGEBRAIC_SIMPLIFICATION = oldFlag;
        }
    }

    // Helper method to calculate sum of a sequence
    private double calculateSum(double from, double to, double incr) {
        double sum = 0;
        int n = 0;

        if ((incr > 0 && from <= to) || (incr < 0 && from >= to)) {
            // Calculate number of elements in the sequence
            n = (int)Math.floor(Math.abs((to - from) / incr)) + 1;

            // Calculate the last element in the sequence
            double last = from + (n - 1) * incr;

            // Use arithmetic sequence sum formula: n * (first + last) / 2
            sum = n * (from + last) / 2;
        }

        return sum;
    }
}
