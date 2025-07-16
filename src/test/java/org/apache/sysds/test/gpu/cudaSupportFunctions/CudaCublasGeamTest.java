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

package org.apache.sysds.test.gpu.cudaSupportFunctions;

import org.apache.sysds.runtime.matrix.data.MatrixValue;
import org.apache.sysds.test.AutomatedTestBase;
import org.apache.sysds.test.TestConfiguration;
import org.apache.sysds.test.TestUtils;
import org.junit.Assert;
import org.junit.BeforeClass;
import org.junit.Test;

import java.util.HashMap;

public class CudaCublasGeamTest extends AutomatedTestBase {

    private static final String TEST_NAME = "CudaCublasGeam";
    private static final String TEST_DIR = "gpu/cudaSupportFunctions/";
    private static final String TEST_CLASS_DIR = TEST_DIR + CudaCublasGeamTest.class.getSimpleName() + "/";

    private static final int rows = 200;
    private static final int cols = 200;

    private static final double eps = Math.pow(10, -10);

    @BeforeClass
    public static void ensureGPU() {
        TestUtils.checkGPU();
    }

    @Override
    public void setUp() {
        TestUtils.clearAssertionInformation();
        addTestConfiguration(TEST_NAME, new TestConfiguration(TEST_CLASS_DIR, TEST_NAME, new String[] {"R"}));
    }

    @Test
    public void testCublasGeamNoTranspose() {
        testCublasGeam(1);
    }

    @Test
    public void testCublasGeamLeftTranspose() {
        testCublasGeam(2);
    }

    @Test
    public void testCublasGeamRightTranspose() {
        testCublasGeam(3);
    }

    @Test
    public void testCublasGeamBothTranspose() {
        testCublasGeam(4);
    }

    private void testCublasGeam(int ID) {

        TestConfiguration config = getTestConfiguration(TEST_NAME);
        loadTestConfiguration(config);

        String HOME = SCRIPT_DIR + TEST_DIR;
        fullDMLScriptName = HOME + TEST_NAME + ".dml";
        programArgs = new String[] {"-stats", "-gpu", "-args", input("A"), input("B"), String.valueOf(ID), output("R")};
        fullRScriptName = HOME + TEST_NAME + ".R";
        rCmd = getRCmd(inputDir(), String.valueOf(ID), expectedDir());

        // both matrices have to be dense
        double[][] A = getRandomMatrix(rows, cols, -1, 1, 0.70d, 5);
        double[][] B = getRandomMatrix(rows, cols, -1, 1, 0.60d, 3);
        writeInputMatrixWithMTD("A", A, true);
        writeInputMatrixWithMTD("B", B, true);

        runTest(true, false, null, -1);
        runRScript(true);

        //compare matrices
        HashMap<MatrixValue.CellIndex, Double> dmlfile = readDMLMatrixFromOutputDir("R");
        HashMap<MatrixValue.CellIndex, Double> rfile = readRMatrixFromExpectedDir("R");
        TestUtils.compareMatrices(dmlfile, rfile, eps, "Stat-DML", "Stat-R");

        Assert.assertTrue(heavyHittersContainsString("gpu_+"));
    }
}
