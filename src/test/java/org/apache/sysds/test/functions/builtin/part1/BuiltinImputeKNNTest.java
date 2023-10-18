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
import org.apache.sysds.common.Types.ExecMode;
import org.apache.sysds.common.Types.ExecType;
import org.apache.sysds.runtime.matrix.data.MatrixValue.CellIndex;
import org.apache.sysds.test.AutomatedTestBase;
import org.apache.sysds.test.TestConfiguration;
import org.apache.sysds.test.TestUtils;
import org.junit.Assert;
import org.junit.Test;

import java.io.IOException;

public class BuiltinImputeKNNTest extends AutomatedTestBase {

    private final static String TEST_NAME = "imputeByKNN";
    private final static String TEST_DIR = "functions/builtin/";
    private static final String TEST_CLASS_DIR = TEST_DIR + BuiltinImputeKNNTest.class.getSimpleName() + "/";

    private double eps = 10;
    @Override
    public void setUp() {
        TestUtils.clearAssertionInformation();
        addTestConfiguration(TEST_NAME, new TestConfiguration(TEST_CLASS_DIR, TEST_NAME, new String[] {"B","B2","B3"}));
    }

    @Test
    public void testDefaultCP()throws IOException{
        runImputeKNN(Types.ExecType.CP);
    }

    @Test
    public void testDefaultSpark()throws IOException{
        runImputeKNN(Types.ExecType.SPARK);
    }

    private void runImputeKNN(ExecType instType) throws IOException {
        ExecMode platform_old = setExecMode(instType);
        try {
            loadTestConfiguration(getTestConfiguration(TEST_NAME));
            String HOME = SCRIPT_DIR + TEST_DIR;
            fullDMLScriptName = HOME + TEST_NAME + ".dml";
            programArgs = new String[] {"-args", DATASET_DIR+"Salaries.csv", 
            	"dist", "dist_missing", "dist_sample", "42", "0.9", output("B"), output("B2"), output("B3")};

            runTest(true, false, null, -1);

            //Compare matrices, check if the sum of the imputed value is roughly the same
            double sum1 = readDMLMatrixFromOutputDir("B").get(new CellIndex(1,1));
            double sum2 = readDMLMatrixFromOutputDir("B2").get(new CellIndex(1,1));
            double sum3 = readDMLMatrixFromOutputDir("B3").get(new CellIndex(1,1));
            Assert.assertEquals(sum1, sum2, eps);
            Assert.assertEquals(sum2, sum3, eps);
        }
        finally {
            rtplatform = platform_old;
        }
    }
}
