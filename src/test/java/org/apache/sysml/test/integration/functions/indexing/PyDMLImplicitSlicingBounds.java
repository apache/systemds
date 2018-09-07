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

package org.apache.sysml.test.integration.functions.indexing;

import org.apache.sysml.runtime.matrix.data.MatrixValue.CellIndex;
import org.apache.sysml.test.integration.AutomatedTestBase;
import org.apache.sysml.test.integration.TestConfiguration;
import org.apache.sysml.test.utils.TestUtils;
import org.junit.Test;

import java.util.HashMap;

/**
 * Test the PyDML implicit slicing.
 */
public class PyDMLImplicitSlicingBounds extends AutomatedTestBase {

    private static final String TEST_NAME1 = "RightImplicitRowLowerImplicitRowUpper";
    private static final String TEST_NAME2 = "RightImplicitRowLowerImplicitRowUpperComma";
    private static final String TEST_NAME3 = "RightImplicitRowColLowerImplicitRowColUpper";
    private static final String TEST_NAME4 = "RightImplicitRowLowerImplicitColUpper";
    private static final String TEST_NAME5 = "RightImplicitColLowerImplicitRowUpper";
    private static final String TEST_NAME6 = "LeftImplicitRowLowerImplicitRowUpper";
    private static final String TEST_NAME7 = "LeftImplicitRowLowerImplicitRowUpperComma";
    private static final String TEST_NAME8 = "LeftImplicitRowColLowerImplicitRowColUpper";
    private static final String TEST_NAME9 = "LeftImplicitRowLowerImplicitColUpper";
    private static final String TEST_NAME10 = "LeftImplicitColLowerImplicitRowUpper";
    private static final String TEST_DIR = "functions/indexing/";
    private static final String TEST_CLASS_DIR =
            TEST_DIR + PyDMLImplicitSlicingBounds.class.getSimpleName() + "/";
    private static final String INPUT_NAME = "X";
    private static final String OUTPUT_NAME_IMPLICIT = "X_implicit";
    private static final String OUTPUT_NAME_EXPLICIT = "X_explicit";

    private static final int rows = 123;
    private static final int cols = 143;
    private static final double sparsity = 0.7;
    private static final double eps = Math.pow(10, -10);

    @Override
    public void setUp() {
        TestUtils.clearAssertionInformation();
        addTestConfiguration(TEST_NAME1, new TestConfiguration(TEST_CLASS_DIR, TEST_NAME1));
        addTestConfiguration(TEST_NAME2, new TestConfiguration(TEST_CLASS_DIR, TEST_NAME2));
        addTestConfiguration(TEST_NAME3, new TestConfiguration(TEST_CLASS_DIR, TEST_NAME3));
        addTestConfiguration(TEST_NAME4, new TestConfiguration(TEST_CLASS_DIR, TEST_NAME4));
        addTestConfiguration(TEST_NAME5, new TestConfiguration(TEST_CLASS_DIR, TEST_NAME5));
        addTestConfiguration(TEST_NAME6, new TestConfiguration(TEST_CLASS_DIR, TEST_NAME6));
        addTestConfiguration(TEST_NAME7, new TestConfiguration(TEST_CLASS_DIR, TEST_NAME7));
        addTestConfiguration(TEST_NAME8, new TestConfiguration(TEST_CLASS_DIR, TEST_NAME8));
        addTestConfiguration(TEST_NAME9, new TestConfiguration(TEST_CLASS_DIR, TEST_NAME9));
        addTestConfiguration(TEST_NAME10, new TestConfiguration(TEST_CLASS_DIR, TEST_NAME10));
    }

    // Right indexing
    @Test
    public void testRightImplicitRowLowerImplicitRowUpper() {
       testPyDMLImplicitSlicingBounds(TEST_NAME1);
    }

    @Test
    public void testRightImplicitRowLowerImplicitRowUpperComma() {
        testPyDMLImplicitSlicingBounds(TEST_NAME2);
    }

    @Test
    public void testRightImplicitRowColLowerImplicitRowColUpper() {
        testPyDMLImplicitSlicingBounds(TEST_NAME3);
    }

    @Test
    public void testRightImplicitRowLowerImplicitColUpper() {
        testPyDMLImplicitSlicingBounds(TEST_NAME4);
    }

    @Test
    public void testRightImplicitColLowerImplicitRowUpper() {
        testPyDMLImplicitSlicingBounds(TEST_NAME5);
    }

    // Left indexing
    @Test
    public void testLeftImplicitRowLowerImplicitRowUpper() {
        testPyDMLImplicitSlicingBounds(TEST_NAME6);
    }

    @Test
    public void testLeftImplicitRowLowerImplicitRowUpperComma() {
        testPyDMLImplicitSlicingBounds(TEST_NAME7);
    }

    @Test
    public void testLeftImplicitRowColLowerImplicitRowColUpper() {
        testPyDMLImplicitSlicingBounds(TEST_NAME8);
    }

    @Test
    public void testLeftImplicitRowLowerImplicitColUpper() {
        testPyDMLImplicitSlicingBounds(TEST_NAME9);
    }

    @Test
    public void testLeftImplicitColLowerImplicitRowUpper() {
        testPyDMLImplicitSlicingBounds(TEST_NAME10);
    }
    /**
     * Test the implicit bounds slicing in PyDML.
     *
     * @param testName The name of this test case.
     */
    private void testPyDMLImplicitSlicingBounds(String testName) {
        // Create and load test configuration
        getAndLoadTestConfiguration(testName);
        String HOME = SCRIPT_DIR + TEST_DIR;
        fullDMLScriptName = HOME + testName + ".pydml";
        programArgs = new String[]{"-python", "-args",
                input(INPUT_NAME), output(OUTPUT_NAME_IMPLICIT), output(OUTPUT_NAME_EXPLICIT)};

        // Generate data
        double[][] X = getRandomMatrix(rows, cols, -1, 1, sparsity, 7);
        writeInputMatrixWithMTD(INPUT_NAME, X, true);

        // Run PyDML script
        runTest(true, false, null, -1);

        // Compare output matrices
        HashMap<CellIndex, Double> pydmlImplicit = readDMLMatrixFromHDFS(OUTPUT_NAME_IMPLICIT);
        HashMap<CellIndex, Double> pydmlExplicit = readDMLMatrixFromHDFS(OUTPUT_NAME_EXPLICIT);
        TestUtils.compareMatrices(pydmlImplicit, pydmlExplicit, eps, "Implicit", "Explicit");

    }
}
