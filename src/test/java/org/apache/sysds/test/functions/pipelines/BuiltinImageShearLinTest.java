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
package org.apache.sysds.test.functions.pipelines;

import org.apache.sysds.common.Types.ExecMode;
import org.apache.sysds.common.Types.ExecType;
import org.apache.sysds.runtime.matrix.data.MatrixValue;
import org.apache.sysds.test.AutomatedTestBase;
import org.apache.sysds.test.TestConfiguration;
import org.apache.sysds.test.TestUtils;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.Parameterized;

import java.util.HashMap;
import java.util.Arrays;
import java.util.Collection;

@RunWith(Parameterized.class)
@net.jcip.annotations.NotThreadSafe

public class BuiltinImageShearLinTest extends AutomatedTestBase {
    private final static String TEST_NAME = "image_shear_linearized";
    private final static String TEST_DIR = "functions/builtin/";
    private final static String TEST_CLASS_DIR = TEST_DIR + BuiltinImageShearLinTest.class.getSimpleName() + "/";

    private final static double eps = 1e-10;
    private final static double spSparse = 0.1;
    private final static double spDense = 0.9;
    /*
     * private final static int s_rows = 200;
     * private final static int s_cols = 120;
     * private final static int n_imgs = 5;
     * 
     * private final static double shearX = 0.0;
     * private final static double shearY = 0.2;
     * private final static double fill_value = 128.0;
     */
    @Parameterized.Parameter(0)
    public int s_rows;
    @Parameterized.Parameter(1)
    public int s_cols;
    @Parameterized.Parameter(2)
    public int n_imgs;
    @Parameterized.Parameter(3)
    public double shearX;
    @Parameterized.Parameter(4)
    public double shearY;
    @Parameterized.Parameter(5)
    public double fill_value;

    @Parameterized.Parameters
    public static Collection<Object[]> data() {
        return Arrays.asList(new Object[][] {
                { 16, 15, 200, 0.0, 0.7, 255.0 },
                { 31, 32, 100, 0.9, -0.2, 0.0 },
                { 64, 64, 100, -0.3, 0.01, 50.0 },
                { 128, 127, 100, 0.0, 0.55, 80.0 },
                { 256, 256, 100, 0.11, 0.0, 25.0 }

        });
    }

    @Override
    public void setUp() {
        addTestConfiguration(TEST_NAME, new TestConfiguration(TEST_CLASS_DIR, TEST_NAME, new String[] { "B" }));
    }

    @Test
    public void testImageShearLinDenseCP() {
        runImageShearLinTest(false, ExecType.CP);
    }

    @Test
    public void testImageShearLinSparseCP() {
        runImageShearLinTest(true, ExecType.CP);
    }

    private void runImageShearLinTest(boolean sparse, ExecType instType) {
        ExecMode platformOld = setExecMode(instType);
        disableOutAndExpectedDeletion();

        try {
            loadTestConfiguration(getTestConfiguration(TEST_NAME));
            double sparsity = sparse ? spSparse : spDense;

            String HOME = SCRIPT_DIR + TEST_DIR;
            fullDMLScriptName = HOME + TEST_NAME + ".dml";
            programArgs = new String[] { "-nvargs", "in_file=" + input("A"), "out_file=" + output("B"),
                    "width=" + s_cols * s_rows,
                    "height=" + n_imgs, "shear_x=" + shearX, "shear_y=" + shearY, "fill_value=" + fill_value,
                    "s_cols=" + s_cols,
                    "s_rows=" + s_rows };

            double[][] A = getRandomMatrix(n_imgs, s_rows * s_cols, 0, 255, sparsity, 7);
            writeInputMatrixWithMTD("A", A, true);

            fullRScriptName = HOME + "image_transform_linearized.R";
            rCmd = "Rscript" + " " + fullRScriptName + " " + inputDir() + " " + expectedDir() + " " + s_cols * s_rows
                    + " " + n_imgs
                    + " " + s_cols + " " + s_rows + " " + 1 + " " + shearX + " " + 0 + " " + shearY + " " + 1 + " " + 0
                    + " " + fill_value + " " + s_cols + " " + s_rows;

            runTest(true, false, null, -1);
            runRScript(true);

            HashMap<MatrixValue.CellIndex, Double> dmlfile = readDMLMatrixFromOutputDir("B");
            HashMap<MatrixValue.CellIndex, Double> rfile = readRMatrixFromExpectedDir("B");
            TestUtils.compareMatrices(dmlfile, rfile, eps, "Stat-DML", "Stat-R");
        } finally {
            rtplatform = platformOld;
        }
    }

}
