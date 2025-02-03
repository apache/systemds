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

package org.apache.sysds.test.functions.builtin.part1;

import org.apache.sysds.common.Types;
import org.apache.sysds.runtime.matrix.data.MatrixValue;
import org.apache.sysds.test.AutomatedTestBase;
import org.apache.sysds.test.TestConfiguration;
import org.apache.sysds.test.TestUtils;
import org.junit.Test;

import java.util.HashMap;

public class BuiltinCooccurrenceMatrixTest extends AutomatedTestBase {

    private static final String TEST_NAME = "cooccurrenceMatrix";
    private static final String TEST_DIR = "functions/builtin/";
    private static final String RESOURCE_DIRECTORY = "src/test/resources/datasets/";
    private static final String TEST_CLASS_DIR = TEST_DIR + BuiltinCooccurrenceMatrixTest.class.getSimpleName() + "/";
    private static final double EPSILON = 1e-10; // Tolerance for comparison

    @Override
    public void setUp() {
        addTestConfiguration(TEST_NAME,new TestConfiguration(TEST_CLASS_DIR, TEST_NAME,new String[]{"TestResult",}));
    }

    @Test
    public void cooccurrenceMatrixTest() {
        runCooccurrenceMatrix(20, 2, "FALSE", "TRUE");
        HashMap<MatrixValue.CellIndex, Double> cooccurrenceMatrix = readDMLMatrixFromOutputDir("TestResult");
        double[][] computedC = TestUtils.convertHashMapToDoubleArray(cooccurrenceMatrix);

        // Unique words: {apple, banana, orange, grape}
        // Co-occurrence based on word pairs in same sentences
        double[][] expectedC = new double[][] {
                {0, 1, 2, 0},  // apple with {banana, orange}
                {1, 0, 3, 1},  // banana with {apple, orange, grape}
                {2, 3, 0, 2},  // orange with {apple, banana, grape}
                {0, 1, 2, 0}   // grape with {banana, orange, grape}
        };

        TestUtils.compareMatrices(expectedC, computedC, expectedC.length, expectedC[0].length, EPSILON);

    }

    public void runCooccurrenceMatrix(Integer maxTokens, Integer windowSize, String distanceWeighting, String symmetric) {
        // Load test configuration
        Types.ExecMode platformOld = setExecMode(Types.ExecType.CP);
        try{
            loadTestConfiguration(getTestConfiguration(TEST_NAME));

            String HOME = SCRIPT_DIR + TEST_DIR;

            fullDMLScriptName = HOME + TEST_NAME + ".dml";

            programArgs = new String[]{"-nvargs",
                    "input=" + RESOURCE_DIRECTORY + "GloVe/coocMatrixTest.csv",
                    "maxTokens=" + maxTokens,
                    "windowSize=" + windowSize,
                    "distanceWeighting=" + distanceWeighting,
                    "symmetric=" + symmetric,
                    "out_file=" + output("TestResult")};
            System.out.println("Run dml script..");
            runTest(true, false, null, -1);
            System.out.println("DONE");
        }
        finally {
            rtplatform = platformOld;
        }
    }


}
