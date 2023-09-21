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
import org.junit.runner.RunWith;
import org.junit.runners.Parameterized;
import org.apache.sysds.common.Types.ExecMode;
import org.apache.sysds.common.Types.ExecType;
import org.apache.sysds.runtime.matrix.data.MatrixValue;
import org.apache.sysds.test.AutomatedTestBase;
import org.apache.sysds.test.TestConfiguration;
import org.apache.sysds.test.TestUtils;

import java.util.Arrays;
import java.util.Collection;
import java.util.HashMap;

@RunWith(Parameterized.class)
@net.jcip.annotations.NotThreadSafe
public class BuiltinImageMirrorLinearizedTest extends AutomatedTestBase {

    private final static String TEST_NAME_LINEARIZED = "image_mirror_linearized";
    private final static String TEST_DIR = "functions/builtin/";
    private final static String TEST_CLASS_DIR = TEST_DIR + BuiltinImageMirrorLinearizedTest.class.getSimpleName() + "/";
    private final static double eps = 1e-10;
    private final static double spSparse = 0.05; 
    private final static double spDense = 0.5; 

    @Parameterized.Parameter()
    public int img_rows;
    @Parameterized.Parameter(1)
    public int img_cols;
    @Parameterized.Parameter(2)
    public int rows;  // number of images
    public int cols;  // Initialized based on img_rows * img_cols

    @Parameterized.Parameters
    public static Collection<Object[]> data() {
        return Arrays.asList(new Object[][] {
            {12, 12, 4},
            {13, 11, 5},
            {64, 64, 32},
           {256, 256, 5},
            {256, 253, 5},
            {1024, 1024, 5},
            {1024, 1048, 5}
            

           
        });
    }

    @Override
    public void setUp() {
        cols = img_rows * img_cols;
        addTestConfiguration(TEST_NAME_LINEARIZED, new TestConfiguration(TEST_CLASS_DIR, TEST_NAME_LINEARIZED, new String[]{"B_x", "B_y"}));
    }

    @Test
    public void testImageMirrorLinearized() {
        runImageMirrorLinearizedTest(false, ExecType.CP);
    }
    
     @Test
    public void testImageMirrorLinearizedSparse() {
        runImageMirrorLinearizedTest(true, ExecType.CP);
    }

    private void runImageMirrorLinearizedTest(boolean sparse, ExecType instType) {
        ExecMode platformOld = setExecMode(instType);
        disableOutAndExpectedDeletion();

        try {
            loadTestConfiguration(getTestConfiguration(TEST_NAME_LINEARIZED));

            double sparsity = sparse ? spSparse : spDense;
            String HOME = SCRIPT_DIR + TEST_DIR;

            fullDMLScriptName = HOME + TEST_NAME_LINEARIZED + ".dml";
            programArgs = new String[]{"-nvargs",
                "in_file=" + input("A"), 
                "x_out_reshape_file=" + output("B_x_reshape"), 
                "y_out_reshape_file=" + output("B_y_reshape"),
                "x_out_file=" + output("B_x"), 
                "y_out_file=" + output("B_y"),
                "img_rows=" + img_rows, 
                "img_cols=" + img_cols    
            };

            double[][] A = getRandomMatrix(rows, cols, 0, 255, sparsity, 7);
            writeInputMatrixWithMTD("A", A, true);

            runTest(true, false, null, -1);

            HashMap<MatrixValue.CellIndex, Double> dmlfileLinearizedX = readDMLMatrixFromOutputDir("B_x");
            HashMap<MatrixValue.CellIndex, Double> dmlfileLinearizedY = readDMLMatrixFromOutputDir("B_y");
            
            HashMap<MatrixValue.CellIndex, Double> dmlfileX = readDMLMatrixFromOutputDir("B_x_reshape");
            HashMap<MatrixValue.CellIndex, Double> dmlfileY = readDMLMatrixFromOutputDir("B_y_reshape");

            TestUtils.compareMatrices(dmlfileLinearizedX, dmlfileX, eps, "Stat-DML-LinearizedX", "Stat-DML-X");
            TestUtils.compareMatrices(dmlfileLinearizedY, dmlfileY, eps, "Stat-DML-LinearizedY", "Stat-DML-Y");

        } finally {
            rtplatform = platformOld;
        }
    }
}
