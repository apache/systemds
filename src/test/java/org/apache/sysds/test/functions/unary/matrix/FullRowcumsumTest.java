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

package org.apache.sysds.test.functions.unary.matrix;

import org.apache.sysds.api.DMLScript;
import org.apache.sysds.common.Types.ExecMode;
import org.apache.sysds.common.Types.ExecType;
import org.apache.sysds.hops.OptimizerUtils;
import org.apache.sysds.runtime.matrix.data.MatrixValue.CellIndex;
import org.apache.sysds.test.AutomatedTestBase;
import org.apache.sysds.test.TestConfiguration;
import org.apache.sysds.test.TestUtils;
import org.apache.sysds.utils.Statistics;
import org.junit.AfterClass;
import org.junit.Assert;
import org.junit.BeforeClass;
import org.junit.Test;

import java.util.HashMap;

public class FullRowcumsumTest extends AutomatedTestBase
{
    private final static String TEST_NAME = "rowcumsum";
    private final static String TEST_DIR = "functions/unary/matrix/";
    private static final String TEST_CLASS_DIR = TEST_DIR + FullRowcumsumTest.class.getSimpleName() + "/";

    private final static double eps = 1e-10;

    private final static int rowsMatrix = 1201;
    private final static int colsMatrix = 1103;
    private final static double spSparse = 0.1;
    private final static double spDense = 0.9;

    private enum InputType {
        COL_VECTOR,
        ROW_VECTOR,
        MATRIX
    }

    @Override
    public void setUp()
    {
        addTestConfiguration(TEST_NAME,new TestConfiguration(TEST_CLASS_DIR, TEST_NAME,new String[]{"B"}));

        if (TEST_CACHE_ENABLED) {
            setOutAndExpectedDeletionDisabled(true);
        }
    }

    @BeforeClass
    public static void init() {
        TestUtils.clearDirectory(TEST_DATA_DIR + TEST_CLASS_DIR);
    }

    @AfterClass
    public static void cleanUp() {
        if (TEST_CACHE_ENABLED) {
            TestUtils.clearDirectory(TEST_DATA_DIR + TEST_CLASS_DIR);
        }
    }

    @Test
    public void testCumsumColVectorDenseCP() {
        runColAggregateOperationTest(InputType.COL_VECTOR, false, ExecType.CP);
    }

    @Test
    public void testCumsumRowVectorDenseCP() {
        runColAggregateOperationTest(InputType.ROW_VECTOR, false, ExecType.CP);
    }

    @Test
    public void testCumsumRowVectorDenseNoRewritesCP() {
        runColAggregateOperationTest(InputType.ROW_VECTOR, false, ExecType.CP, false);
    }

    @Test
    public void testCumsumMatrixDenseCP() {
        runColAggregateOperationTest(InputType.MATRIX, false, ExecType.CP);
    }

    @Test
    public void testCumsumColVectorSparseCP() {
        runColAggregateOperationTest(InputType.COL_VECTOR, true, ExecType.CP);
    }

    @Test
    public void testCumsumRowVectorSparseCP() {
        runColAggregateOperationTest(InputType.ROW_VECTOR, true, ExecType.CP);
    }

    @Test
    public void testCumsumRowVectorSparseNoRewritesCP() {
        runColAggregateOperationTest(InputType.ROW_VECTOR, true, ExecType.CP, false);
    }

    @Test
    public void testCumsumMatrixSparseCP() {
        runColAggregateOperationTest(InputType.MATRIX, true, ExecType.CP);
    }

    @Test
    public void testCumsumColVectorDenseSP() {
        runColAggregateOperationTest(InputType.COL_VECTOR, false, ExecType.SPARK);
    }

    @Test
    public void testCumsumRowVectorDenseSP() {
        runColAggregateOperationTest(InputType.ROW_VECTOR, false, ExecType.SPARK);
    }

    @Test
    public void testCumsumRowVectorDenseNoRewritesSP() {
        runColAggregateOperationTest(InputType.ROW_VECTOR, false, ExecType.SPARK, false);
    }

    @Test
    public void testCumsumMatrixDenseSP() {
        runColAggregateOperationTest(InputType.MATRIX, false, ExecType.SPARK);
    }

    @Test
    public void testCumsumColVectorSparseSP() { //das hier testen
        runColAggregateOperationTest(InputType.COL_VECTOR, true, ExecType.SPARK);
    }

    @Test
    public void testCumsumRowVectorSparseSP() {
        runColAggregateOperationTest(InputType.ROW_VECTOR, true, ExecType.SPARK);
    }

    @Test
    public void testCumsumRowVectorSparseNoRewritesSP() {
        runColAggregateOperationTest(InputType.ROW_VECTOR, true, ExecType.SPARK, false);
    }

    @Test
    public void testCumsumMatrixSparseSP() {
        runColAggregateOperationTest(InputType.MATRIX, true, ExecType.SPARK);
    }

    private void runColAggregateOperationTest( InputType type, boolean sparse, ExecType instType) {
        //by default we apply algebraic simplification rewrites
        runColAggregateOperationTest(type, sparse, instType, true);
    }

    private void runColAggregateOperationTest( InputType type, boolean sparse, ExecType instType, boolean rewrites)
    {
        ExecMode platformOld = rtplatform;
        switch( instType ){
            case SPARK: rtplatform = ExecMode.SPARK; break;
            default: rtplatform = ExecMode.HYBRID; break;
        }

        boolean sparkConfigOld = DMLScript.USE_LOCAL_SPARK_CONFIG;
        if( rtplatform == ExecMode.SPARK )
            DMLScript.USE_LOCAL_SPARK_CONFIG = true;

        //rewrites
        boolean oldFlagRewrites = OptimizerUtils.ALLOW_ALGEBRAIC_SIMPLIFICATION;
        OptimizerUtils.ALLOW_ALGEBRAIC_SIMPLIFICATION = rewrites;

        try
        {
            int cols = (type== InputType.COL_VECTOR) ? 1 : colsMatrix;
            int rows = (type== InputType.ROW_VECTOR) ? 1 : rowsMatrix;
            double sparsity = (sparse) ? spSparse : spDense;

            String TEST_CACHE_DIR = !TEST_CACHE_ENABLED ? "" :
                    type.ordinal() + "_" + sparsity + "/";

            TestConfiguration config = getTestConfiguration(TEST_NAME);
            loadTestConfiguration(config, TEST_CACHE_DIR);

            // This is for running the junit test the new way, i.e., construct the arguments directly
            String HOME = SCRIPT_DIR + TEST_DIR;
            fullDMLScriptName = HOME + TEST_NAME + ".dml";
            programArgs = new String[]{"-explain", "-args", input("A"), output("B") };

            fullRScriptName = HOME + TEST_NAME + ".R";
            rCmd = "Rscript" + " " + fullRScriptName + " " + inputDir() + " " + expectedDir();

            //generate actual dataset
            double[][] A = getRandomMatrix(rows, cols, -0.05, 1, sparsity, 7);
            writeInputMatrixWithMTD("A", A, true);

            runTest(true, false, null, -1);
            if( instType==ExecType.CP ) //in CP no spark jobs should be executed
                Assert.assertEquals("Unexpected number of executed MR jobs.", 0, Statistics.getNoOfExecutedSPInst());

            runRScript(true);

            //compare matrices
            HashMap<CellIndex, Double> dmlfile = readDMLMatrixFromOutputDir("B");
            HashMap<CellIndex, Double> rfile  = readRMatrixFromExpectedDir("B");
            TestUtils.compareMatrices(dmlfile, rfile, eps, "Stat-DML", "Stat-R");
        }
        finally
        {
            rtplatform = platformOld;
            DMLScript.USE_LOCAL_SPARK_CONFIG = sparkConfigOld;
            OptimizerUtils.ALLOW_ALGEBRAIC_SIMPLIFICATION = oldFlagRewrites;
        }
    }
}
