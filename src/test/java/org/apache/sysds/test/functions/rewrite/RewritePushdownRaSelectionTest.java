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

import java.util.HashMap;

import org.apache.sysds.common.Opcodes;
import org.apache.sysds.common.Types;
import org.apache.sysds.hops.OptimizerUtils;
import org.apache.sysds.runtime.matrix.data.MatrixValue.CellIndex;
import org.apache.sysds.test.AutomatedTestBase;
import org.apache.sysds.test.TestConfiguration;
import org.apache.sysds.test.TestUtils;
import org.junit.Test;

public class RewritePushdownRaSelectionTest extends AutomatedTestBase
{
	private static final String TEST_NAME = "RewritePushdownRaSelection";
	private static final String TEST_DIR = "functions/rewrite/";
	private static final String TEST_CLASS_DIR = TEST_DIR + RewritePushdownRaSelectionTest.class.getSimpleName() + "/";

	private static final double eps = 1e-8;

	@Override
	public void setUp() {
		TestUtils.clearAssertionInformation();
		addTestConfiguration(TEST_NAME, new TestConfiguration(TEST_CLASS_DIR, TEST_NAME, new String[] {"result"}));
	}

	@Test
	public void testRewritePushdownRaSelectionNoRewrite() {
        int col = 1;
        String op = Opcodes.EQUAL.toString();
        double val = 4.0;

        // Expected output matrix
        double[][] Y = {
                {4,7,8,4,7,8},
                {4,7,8,4,5,10},
                {4,3,5,4,7,8},
                {4,3,5,4,5,10},
        };

        testRewritePushdownRaSelection(col, op, val, Y, "nested-loop", false);
	}

	@Test
	public void testRewritePushdownRaSelection1() {
        int col = 1;
        String op = Opcodes.EQUAL.toString();
        double val = 4.0;

        // Expected output matrix
        double[][] Y = {
                {4,7,8,4,7,8},
                {4,7,8,4,5,10},
                {4,3,5,4,7,8},
                {4,3,5,4,5,10},
        };

        testRewritePushdownRaSelection(col, op, val, Y, "sort-merge", true);
	}

    @Test
    public void testRewritePushdownRaSelection2() {
        int col = 5;
        String op = Opcodes.EQUAL.toString();
        double val = 7.0;

        // Expected output matrix
        double[][] Y = {
                {4,7,8,4,7,8},
                {4,3,5,4,7,8},
        };

        testRewritePushdownRaSelection(col, op, val, Y, "sort-merge", true);
    }

    private void testRewritePushdownRaSelection(int col, String op, double val, double[][] Y,
                                                String method, boolean rewrites) {

        //generate actual dataset and variables
        double[][] A = {
                {1, 2, 3},
                {4, 7, 8},
                {1, 3, 6},
                {4, 3, 5},
                {5, 8, 9}
        };
        double[][] B = {
                {1, 2, 9},
                {3, 7, 6},
                {2, 8, 5},
                {4, 7, 8},
                {4, 5, 10}
        };
        int colA = 1;
        int colB = 1;

        runRewritePushdownRaSelectionTest(A, colA, B, colB, Y, col, op, val, method, rewrites);
    }


    private void runRewritePushdownRaSelectionTest(double [][] A, int colA, double [][] B, int colB, double [][] Y,
                                                   int col, String op, double val, String method, boolean rewrites)
    {
        Types.ExecMode platformOld = setExecMode(Types.ExecMode.SINGLE_NODE);
        boolean oldFlag = OptimizerUtils.ALLOW_RA_REWRITES;

        try
        {
            loadTestConfiguration(getTestConfiguration(TEST_NAME));
            String HOME = SCRIPT_DIR + TEST_DIR;

            fullDMLScriptName = HOME + TEST_NAME + ".dml";
            programArgs = new String[]{"-explain", "hops", "-args",
                    input("A"), String.valueOf(colA), input("B"),
                    String.valueOf(colB), String.valueOf(col), op, String.valueOf(val), method, output("result") };
            writeInputMatrixWithMTD("A", A, true);
            writeInputMatrixWithMTD("B", B, true);

            OptimizerUtils.ALLOW_RA_REWRITES = rewrites;

            // run dmlScript
            runTest(null);

            //compare matrices
            HashMap<CellIndex, Double> dmlfile = readDMLMatrixFromOutputDir("result");
            HashMap<CellIndex, Double> expectedOutput = TestUtils.convert2DDoubleArrayToHashMap(Y);
            TestUtils.compareMatrices(dmlfile, expectedOutput, eps, "Stat-DML", "Expected");
        }
        finally {
            rtplatform = platformOld;
            OptimizerUtils.ALLOW_RA_REWRITES = oldFlag;
        }
    }
}
