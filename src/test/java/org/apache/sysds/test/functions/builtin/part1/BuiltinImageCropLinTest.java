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

import java.util.HashMap;

public class BuiltinImageCropLinTest extends AutomatedTestBase
{
	private final static String TEST_NAME = "image_crop_linearized";
	private final static String TEST_DIR = "functions/builtin/";
	private final static String TEST_CLASS_DIR = TEST_DIR + BuiltinImageCropTest.class.getSimpleName() + "/";

	private final static double eps = 1e-10;

    private final static int s_cols = 27;
    private final static int s_rows = 8;
	private final static int rows = s_cols * s_rows;
	private final static int cols = 13;
	private final static double spSparse = 0.1;
	private final static double spDense = 0.9;
	private final static int x_offset = 1;
	private final static int y_offset = 2;
	private final static float size = 0.6f;
    



    @Override 
    public void setUp() {
        addTestConfiguration(TEST_NAME,new TestConfiguration(TEST_CLASS_DIR, TEST_NAME,new String[]{"B"}));
    }

    @Test
    public void testImageCropMatrixDenseCP() {runImageCropLinTest(false, ExecType.CP);
    }

    @Test
    public void testImageCropMatrixSparseCP() {runImageCropLinTest(true, ExecType.CP);
    }

    @Test
    public void testImageCropMatrixDenseSP() {runImageCropLinTest(false, ExecType.SPARK);
    }

    @Test
    public void testImageCropMatrixSparseSP() {runImageCropLinTest(false,ExecType.SPARK);
    }

    private void runImageCropLinTest (boolean sparse, ExecType instType)
    {
        ExecMode platformOld = setExecMode(instType);
        disableOutAndExpectedDeletion();

        try{
            loadTestConfiguration(getTestConfiguration(TEST_NAME));
            double sparsity = sparse ? spSparse : spDense;

            int new_w = (int) Math.floor(s_cols * size);
            int new_h = (int) Math.floor(s_rows * size);
            //int new_w = 20;
            //int new_h = 5;

            String HOME = SCRIPT_DIR + TEST_DIR;
            fullDMLScriptName = HOME + TEST_NAME + ".dml";
            programArgs = new String[]{"-nvargs",
					"in_file=" + input("A"), "out_file=" + output("B"),
					"x_offset=" + x_offset, "y_offset=" + y_offset, "cols=" + cols, "rows=" + rows, 
                    "s_cols=" + s_cols, "s_rows=" + s_rows, "new_w=" + new_w, "new_h=" + new_h};

            
            fullRScriptName = HOME + TEST_NAME + ".R";
            rCmd = "Rscript" + " " + fullRScriptName + " " + inputDir() + " " + expectedDir() 
            + " " + size + " " + cols + " " + rows + " " + s_cols + " " + s_rows + " " + x_offset + " " + y_offset
            + " " + new_w + " " + new_h;

            //generate actual dataset
            double[][] A = getRandomMatrix(rows, cols, 0, 255, sparsity, 7);
            writeInputMatrixWithMTD("A", A, true);

            runTest(true, false, null, -1);
            runRScript(true);

            HashMap<MatrixValue.CellIndex, Double> dmlfile = readDMLMatrixFromOutputDir("B");
            HashMap<MatrixValue.CellIndex, Double> rfile = readRMatrixFromExpectedDir("B");
            TestUtils.compareMatrices(dmlfile, rfile, eps, "Stat-DML", "Stat-R");

        }
        finally {
            rtplatform = platformOld;
        }



    }

}