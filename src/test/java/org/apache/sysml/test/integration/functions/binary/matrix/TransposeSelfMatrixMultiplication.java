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

package org.apache.sysml.test.integration.functions.binary.matrix;

import java.util.HashMap;

import org.junit.AfterClass;
import org.junit.BeforeClass;
import org.junit.Test;

import org.apache.sysml.api.DMLScript;
import org.apache.sysml.api.DMLScript.RUNTIME_PLATFORM;
import org.apache.sysml.lops.LopProperties.ExecType;
import org.apache.sysml.runtime.matrix.data.MatrixValue.CellIndex;
import org.apache.sysml.test.integration.AutomatedTestBase;
import org.apache.sysml.test.integration.TestConfiguration;
import org.apache.sysml.test.utils.TestUtils;

/**
 * This test investigates the specific Hop-Lop rewrite t(X)%*%v -> t(t(v)%*%X).
 * 
 */
public class TransposeSelfMatrixMultiplication extends AutomatedTestBase
{

        private final static String TEST_NAME1 = "TransposeSelfMatrixMultiplication";
        private final static String TEST_DIR = "functions/binary/matrix/";
        private final static String TEST_CLASS_DIR = TEST_DIR + TransposeSelfMatrixMultiplication.class.getSimpleName() + "/";
        private final static double eps = 1e-10;

        //multiblock
        private final static int rowsA1 = 3;
        private final static int colsA1 = 3;

        //singleblock
        private final static int rowsA2 = 2407;
        private final static int colsA2 = 73;


        private final static double sparsity1 = 0.7;
        private final static double sparsity2 = 0.1;


        @Override
        public void setUp()
        {
                addTestConfiguration( TEST_NAME1,
                        new TestConfiguration(TEST_CLASS_DIR, TEST_NAME1, new String[] { "C" }) );

                if (TEST_CACHE_ENABLED) {
                        setOutAndExpectedDeletionDisabled(true);
                }
        }

        @BeforeClass
        public static void init()
        {
                TestUtils.clearDirectory(TEST_DATA_DIR + TEST_CLASS_DIR);
        }

        @AfterClass
        public static void cleanUp()
        {
                if (TEST_CACHE_ENABLED) {
                        TestUtils.clearDirectory(TEST_DATA_DIR + TEST_CLASS_DIR);
                }
        }

        @Test
        public void testTransposeMMDenseDenseCP1()
        {
        		/**
        		 * test case to test the pattern X %*% t(X) and t(X) %*% X 
        		 * @param1	isSparse
        		 * @param2	ExecType
        		 * @param3	isVector
        		 * @param4	isLeftTransposed	for A %*% A', it's false; for A' %*% A, it's true
        		 */
                runTransposeSelfMatrixMultiplication(false, ExecType.CP, false, true);
        }

        @Test
        public void testTransposeMMDenseDenseCP2()
        {
                runTransposeSelfMatrixMultiplication(false, ExecType.CP, false, false);
        }
        
        /**
         * 
         * @param sparseM1
         * @param sparseM2
         * @param instType
         */
        private void runTransposeSelfMatrixMultiplication( boolean sparseM1, ExecType instType, boolean vectorM2, boolean isLeftTransposed)
        {
                //rtplatform for MR
                RUNTIME_PLATFORM platformOld = rtplatform;
                switch( instType ){
                        case MR: rtplatform = RUNTIME_PLATFORM.HADOOP; break;
                        case SPARK: rtplatform = RUNTIME_PLATFORM.SPARK; break;
                        default: rtplatform = RUNTIME_PLATFORM.HYBRID; break;
                }

                boolean sparkConfigOld = DMLScript.USE_LOCAL_SPARK_CONFIG;
                if( rtplatform == RUNTIME_PLATFORM.SPARK )
                	DMLScript.USE_LOCAL_SPARK_CONFIG = true;

                int rowsA = vectorM2 ? rowsA2 : rowsA1;
                int colsA = vectorM2 ? colsA2 : colsA1;

                String TEST_NAME = TEST_NAME1;

                try
                {
                        TestConfiguration config = getTestConfiguration(TEST_NAME);

                        double sparsityM1 = sparseM1?sparsity2:sparsity1;

                        String TEST_CACHE_DIR = "";
                        if (TEST_CACHE_ENABLED)
                        {
                                TEST_CACHE_DIR = sparsityM1 + "_" + vectorM2 + "_" + isLeftTransposed + "/";
                        }

                        loadTestConfiguration(config, TEST_CACHE_DIR);

                        /* This is for running the junit test the new way, i.e., construct the arguments directly */
                        String HOME = SCRIPT_DIR + TEST_DIR;
                        fullDMLScriptName = HOME + TEST_NAME + ".dml";
                        programArgs = new String[]{"-explain","-args",
                                input("A"), Integer.toString(rowsA), Integer.toString(colsA),
                                ("" + isLeftTransposed).toUpperCase(),
                                output("C")};

                        fullRScriptName = HOME + TEST_NAME + ".R";
                        rCmd = "Rscript" + " " + fullRScriptName + " " +
                        inputDir() + " " + isLeftTransposed + " " + expectedDir();

                        //generate actual dataset
                        double[][] A = getRandomMatrix(rowsA, colsA, 0, 1, sparsityM1, 7);
                        writeInputMatrix("A", A, true);

                        boolean exceptionExpected = false;
                        runTest(true, exceptionExpected, null, -1);
                        runRScript(true);

                        //compare matrices 
                        HashMap<CellIndex, Double> dmlfile = readDMLMatrixFromHDFS("C");
                        HashMap<CellIndex, Double> rfile  = readRMatrixFromFS("C");
                        TestUtils.compareMatrices(dmlfile, rfile, eps, "Stat-DML", "Stat-R");
                }
                finally
                {
                        rtplatform = platformOld;
                        DMLScript.USE_LOCAL_SPARK_CONFIG = sparkConfigOld;
                }
        }
}