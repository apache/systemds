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
import org.apache.sysds.common.Types.ExecType;
import org.apache.sysds.runtime.matrix.data.MatrixValue.CellIndex;
import org.apache.sysds.test.AutomatedTestBase;
import org.apache.sysds.test.TestConfiguration;
import org.apache.sysds.test.TestUtils;
import org.apache.sysds.utils.Statistics;
import org.junit.Assert;
import org.junit.Test;
import java.util.HashMap;

public class BuiltinAmputeTest extends AutomatedTestBase {
    private final static String TEST_NAME = "builtinAmputeTest";
    private final static String TEST_DIR = "functions/builtin/";
    private final static String TEST_CLASS_DIR = TEST_DIR + BuiltinAmputeTest.class.getSimpleName() + "/";
    private final static String OUTPUT_NAME = "R";

    private final static String WINE_DATA = DATASET_DIR + "wine/winequality-red-white.csv";
    private final static String DIABETES_DATA = DATASET_DIR + "diabetes/diabetes.csv";
    private final static String MNIST_DATA = DATASET_DIR + "MNIST/mnist_test.csv";
    private final static double EPSILON = 0.05;
    private final static double SMALL_SAMPLE_EPSILON = 0.1; // More leeway given to smaller proportions of amputed rows.
    private final static int SEED = 42;

    @Override
    public void setUp() {
        for(int i = 1; i <= 4; i++) {
            addTestConfiguration(TEST_NAME + i, new TestConfiguration(TEST_CLASS_DIR, TEST_NAME, new String[] {OUTPUT_NAME}));
        }
    }

    @Test
    public void testAmputeWine_prop25() {
        runAmpute(1, WINE_DATA, ExecType.CP, false, 0.25, SMALL_SAMPLE_EPSILON);
    }

    @Test
    public void testAmputeDiabetes_prop25() {
        runAmpute(2, DIABETES_DATA, ExecType.CP, true, 0.25, SMALL_SAMPLE_EPSILON);
    }

    @Test
    public void testAmputeMNIST_prop25() {
        runAmpute(3, MNIST_DATA, ExecType.CP, false, 0.25, SMALL_SAMPLE_EPSILON);
    }

    @Test
    public void testAmputeWine_prop50() {
        runAmpute(1, WINE_DATA, ExecType.CP, false, 0.5, EPSILON);
    }

    @Test
    public void testAmputeDiabetes_prop50() {
        runAmpute(2, DIABETES_DATA, ExecType.CP, true, 0.5, EPSILON);
    }

    @Test
    public void testAmputeMNIST_prop50() {
        runAmpute(3, MNIST_DATA, ExecType.CP, false, 0.5, EPSILON);
    }

    @Test
    public void testAmputeWine_prop75() {
        runAmpute(1, WINE_DATA, ExecType.CP, false, 0.75, EPSILON);
    }

    @Test
    public void testAmputeDiabetes_prop75() {
        runAmpute(2, DIABETES_DATA, ExecType.CP, true, 0.75, EPSILON);
    }

    @Test
    public void testAmputeMNIST_prop75() {
        runAmpute(3, MNIST_DATA, ExecType.CP, false, 0.75, EPSILON);
    }

    @Test
    public void testAmputeMNIST_singleRow() {
        runSingleRowDMLAmpute(4, WINE_DATA, ExecType.CP, false, 0.5, EPSILON);
    }

    // This function tests whether MICE ampute (R) and SystemDS ampute.dml produce approximately the same proportion of amputed rows
    // and pattern frequencies in their output under the same input settings. This information is compiled into a single matrix by each script.
    private void runAmpute(int test, String data, ExecType instType, boolean header, double prop, double eps) {
        Types.ExecMode platformOld = setExecMode(instType);

        try {
            loadTestConfiguration(getTestConfiguration(TEST_NAME + test));

            String HOME = SCRIPT_DIR + TEST_DIR;

            fullDMLScriptName = HOME + "ampute.dml";
            programArgs = new String[]{"-stats", "-args", data, output(OUTPUT_NAME), String.valueOf(SEED), String.valueOf(header), String.valueOf(prop), String.valueOf(false)};

            fullRScriptName = HOME + "ampute.R";
            String outPath = expectedDir() + OUTPUT_NAME;
            rCmd = getRCmd( data, outPath, String.valueOf(SEED), String.valueOf(header), String.valueOf(prop));

            runRScript(true);
            runTest(true, false, null, -1);

            HashMap<CellIndex, Double> dmlfile = readDMLMatrixFromOutputDir(OUTPUT_NAME);
            HashMap<CellIndex, Double> rfile  = readRMatrixFromExpectedDir(OUTPUT_NAME);
            TestUtils.compareMatrices(dmlfile, rfile, eps, "Stat-DML", "Stat-R");

            Assert.assertEquals(0, Statistics.getNoOfExecutedSPInst());
        } finally {
            rtplatform = platformOld;
        }
    }

    // This function simply tests implicitly whether an exception is thrown when running DML ampute with a single input data row:
    private void runSingleRowDMLAmpute(int test, String data, ExecType instType, boolean header, double prop, double eps) {
        Types.ExecMode platformOld = setExecMode(instType);
        try {
            loadTestConfiguration(getTestConfiguration(TEST_NAME + test));
            String HOME = SCRIPT_DIR + TEST_DIR;

            fullDMLScriptName = HOME + "ampute.dml";
            programArgs = new String[]{"-stats", "-args", data, output(OUTPUT_NAME), String.valueOf(SEED), String.valueOf(header), String.valueOf(prop), String.valueOf(true)};
            runTest(true, false, null, -1);

            HashMap<CellIndex, Double> dmlfile = readDMLMatrixFromOutputDir(OUTPUT_NAME);
            Assert.assertEquals(0, Statistics.getNoOfExecutedSPInst());
        } finally {
            rtplatform = platformOld;
        }
    }
}
