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

import org.junit.Test;
import org.apache.sysds.common.Types.ExecMode;
import org.apache.sysds.common.Types.ExecType;
import org.apache.sysds.runtime.matrix.data.MatrixValue;
import org.apache.sysds.test.AutomatedTestBase;
import org.apache.sysds.test.TestConfiguration;
import org.apache.sysds.test.TestUtils;
import org.junit.runner.RunWith;
import org.junit.runners.Parameterized;

import java.util.HashMap;
import java.util.Arrays;
import java.util.Collection;

@RunWith(Parameterized.class)
@net.jcip.annotations.NotThreadSafe

public class BuiltinImageCropLinTest extends AutomatedTestBase {
    private final static String TEST_NAME = "image_crop_linearized";
    private final static String TEST_DIR = "functions/builtin/";
    private final static String TEST_CLASS_DIR = TEST_DIR + BuiltinImageCropLinTest.class.getSimpleName() + "/";

    private final static double eps = 1e-10;
    private final static double spSparse = 0.1;
    private final static double spDense = 0.9;

    @Parameterized.Parameter(0)
    public int s_cols;
    @Parameterized.Parameter(1)
    public int s_rows;
    @Parameterized.Parameter(2)
    public int rows;
    @Parameterized.Parameter(3)
    public int x_offset;
    @Parameterized.Parameter(4)
    public int y_offset;
    @Parameterized.Parameter(5)
    public double size;

    public int cols;
    public int new_w;
    public int new_h;

    @Parameterized.Parameters
    public static Collection<Object[]> data() {
        return Arrays.asList(new Object[][] {
                { 10, 12, 20, 2, 3, 0.5 },
                { 12, 12, 40, 5, 5, 0.4 },
                { 32, 32, 200, 13, 10, 0.2 },
                { 31, 33, 200, 7, 10, 0.2 }, 
                { 64, 64, 50, 2, 0, 0.8 },
                { 125, 123, 32, 7, 37, 0.3 },
                { 128, 128, 83, 23, 14, 0.123 },
                { 256, 50, 2, 0, 0, 0.8 } 
        });
    }

    @Override
    public void setUp() {
        cols = s_cols * s_rows;
        new_w = (int) Math.floor(s_cols * size);
        new_h = (int) Math.floor(s_rows * size);
        addTestConfiguration(TEST_NAME, new TestConfiguration(TEST_CLASS_DIR, TEST_NAME, new String[] { "B" }));
    }

    @Test
    public void testImageCropMatrixDenseCP() {
        runImageCropLinTest(false, ExecType.CP);
    }

    @Test
    public void testImageCropMatrixSparseCP() {
        runImageCropLinTest(true, ExecType.CP);
    }

    @Test
    public void testImageCropMatrixDenseSP() {
        runImageCropLinTest(false, ExecType.SPARK);
    }

    @Test
    public void testImageCropMatrixSparseSP() {
        runImageCropLinTest(false, ExecType.SPARK);
    }

    private void runImageCropLinTest(boolean sparse, ExecType instType) {
        ExecMode platformOld = setExecMode(instType);
        disableOutAndExpectedDeletion();

        try {
            loadTestConfiguration(getTestConfiguration(TEST_NAME));
            double sparsity = sparse ? spSparse : spDense;

            String HOME = SCRIPT_DIR + TEST_DIR;
            fullDMLScriptName = HOME + TEST_NAME + ".dml";
            programArgs = new String[] { "-nvargs",
                    "in_file=" + input("A"), "out_file=" + output("B"),
                    "x_offset=" + x_offset, "y_offset=" + y_offset, "cols=" + cols, "rows=" + rows,
                    "s_cols=" + s_cols, "s_rows=" + s_rows, "new_w=" + new_w, "new_h=" + new_h };
            // print the command
            System.out.println("COMMAND:" + fullDMLScriptName + " " + String.join(" ", programArgs));

            // generate actual dataset
            double[][] A = getRandomMatrix(rows, cols, 0, 255, sparsity, 7);
            writeInputMatrixWithMTD("A", A, true);

            // crop functionality in java
            double[][] ref = new double[rows][new_h * new_w];
            int start_h = (int) Math.ceil((double) (s_rows - new_h) / 2) + y_offset;
            int start_w = (int) Math.ceil((double) (s_cols - new_w) / 2) + x_offset;
            if(s_cols == 64){
                System.out.println("start_h: " + start_h + ", start_w: " + start_w);
            }

            for (int i = 0; i < rows; i++) {
                if(start_w == 0 && start_h == 0){
                    ref[i]=A[i];
                }else{
                    for (int j = 0; j < new_h * new_w; j++) {
                        int ja = ((j / new_w) + start_h - 1) * s_cols + (j % new_w) + start_w - 1;
                        ref[i][j] = A[i][ja];
                    }
                }
            }

            writeInputMatrixWithMTD("ref", ref, true);

            runTest(true, false, null, -1);

            HashMap<MatrixValue.CellIndex, Double> dmlfile = readDMLMatrixFromOutputDir("B");
            double[][] dml_res = TestUtils.convertHashMapToDoubleArray(dmlfile, rows,
                    (new_h * new_w));
            TestUtils.compareMatrices(ref, dml_res, eps, "Java vs. DML");

        } finally {
            rtplatform = platformOld;
        }

    }

}