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

package org.apache.sysml.test.integration.mlcontext.algorithms;

import static org.apache.sysml.api.mlcontext.ScriptFactory.dmlFromFile;

import org.apache.log4j.Logger;
import org.apache.sysml.api.mlcontext.Script;
import org.apache.sysml.hops.OptimizerUtils;
import org.apache.sysml.test.integration.mlcontext.MLContextTestBase;
import org.apache.sysml.test.utils.TestUtils;
import org.junit.Assert;
import org.junit.Test;


public class MLContextSVMTest extends MLContextTestBase {
        protected static Logger log = Logger.getLogger(MLContextSVMTest.class);

        protected final static String TEST_SCRIPT_L2SVM = "scripts/algorithms/l2-svm.dml";
        protected final static String TEST_SCRIPT_MSVM  = "scripts/algorithms/m-svm.dml";

        private final static double sparsity1 = 0.7;//dense
        private final static double sparsity2 = 0.1;//sparse

        private final static int rows = 3468;
        private final static int cols = 1007;

        public enum SVMtype {
                L2SVM,
                MSVM,
        }

        @Test
        public void testL2SVMDenseRewrites() {
                runSVMTestMLC(SVMtype.L2SVM, 1, true, false);
        }

        @Test
        public void testL2SVMSparseRewrites() {
                runSVMTestMLC(SVMtype.L2SVM, 1, true, true);
        }

        @Test
        public void testL2SVMDense() {
                runSVMTestMLC(SVMtype.L2SVM, 1, false, false);
        }

        @Test
        public void testL2SVMSparse() {
                runSVMTestMLC(SVMtype.L2SVM, 1, false, true);
        }

        @Test
        public void testMSVMDenseBinaryRewrites() {
                runSVMTestMLC(SVMtype.MSVM, 2, true, false);
        }

        @Test
        public void testMSVMSparseBinaryRewrites() {
                runSVMTestMLC(SVMtype.MSVM, 2, true, true);
        }

        @Test
        public void testMSVMDenseMulRewrites() {
                runSVMTestMLC(SVMtype.MSVM, 4, true, false);
        }

        @Test
        public void testMSVMSparseMulRewrites() {
                runSVMTestMLC(SVMtype.MSVM, 4, true, true);
        }

        @Test
        public void testMSVMDenseBinary() {
                runSVMTestMLC(SVMtype.MSVM, 2, false, false);
        }

        @Test
        public void testMSVMSparseBinary() {
                runSVMTestMLC(SVMtype.MSVM, 2, false, true);
        }

        @Test
        public void testMSVMDenseMul() {
                runSVMTestMLC(SVMtype.MSVM, 4, false, false);
        }

        @Test
        public void testMSVMSparseMul() {
                runSVMTestMLC(SVMtype.MSVM, 4, false, true);
        }

        private void runSVMTestMLC(SVMtype type, int numClasses,boolean rewrites, boolean sparse) {

                 OptimizerUtils.ALLOW_ALGEBRAIC_SIMPLIFICATION = rewrites;

                 double[][] X = getRandomMatrix(rows, cols, 0, 1, sparse ? sparsity2: sparsity1, 714);
                 double[][] Y = TestUtils.round(getRandomMatrix(rows, 1, 1, numClasses, 1.0, 136));

                 switch (type) {
                         case L2SVM:
                                    Script l2svm = dmlFromFile(TEST_SCRIPT_L2SVM);
                                    l2svm.in("X", X).in("Y", Y).in("$icpt", "0").in("$tol", "0.001")
                                           .in("$reg", "1.0").in("$maxiter", "100").out("w");
                                    double[][] w = ml.execute(l2svm).getMatrix("w").to2DDoubleArray();
                                    log.debug("output matrix weights:\n" + getMatrixAsString(w));
                                     Assert.assertEquals(0.0, w[cols+2][0], 0); //this col is intercept value(=0).

                                    break;

                         case MSVM:
                                   Script msvm = dmlFromFile(TEST_SCRIPT_MSVM);
                                   msvm.in("X", X).in("Y", Y).in("$icpt", "0").in("$tol", "0.001")
                                         .in("$reg", "1.0").in("$maxiter", "100").out("w");
                                   w = ml.execute(msvm).getMatrix("w").to2DDoubleArray();
                         log.debug("output matrix weights:\n" + getMatrixAsString(w));
                         Assert.assertEquals(0, w[cols][0], 0); //this col is intercept value(=0).

                         break;
        }

    }
}
