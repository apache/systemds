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

package org.apache.sysds.test.functions.builtin.part2;


import org.junit.Assert;
import org.junit.Test;

import java.util.HashMap;

import org.apache.sysds.common.Types.ExecMode;
import org.apache.sysds.runtime.matrix.data.MatrixValue.CellIndex;
import org.apache.sysds.test.AutomatedTestBase;
import org.apache.sysds.test.TestConfiguration;
import org.apache.sysds.test.TestUtils;

public class BuiltinShapExplainerTest extends AutomatedTestBase
{
    private static final String TEST_NAME = "shapExplainer";
    private static final String TEST_DIR = "functions/builtin/";
    private static final String TEST_CLASS_DIR = TEST_DIR + BuiltinShapExplainerTest.class.getSimpleName() + "/";
    private static final boolean VERBOSE = true;

    @Override
    public void setUp() {
        addTestConfiguration(TEST_NAME, new TestConfiguration(TEST_CLASS_DIR, TEST_NAME, new String[]{"R"}));
    }

    @Test
    public void testPrepareMaskForPermutation() {
        runShapExplainerUnitTest("prepare_mask_for_permutation");
    }

    private void runShapExplainerUnitTest(String testType) {
        ExecMode platformOld = setExecMode(ExecMode.SINGLE_NODE);
        loadTestConfiguration(getTestConfiguration(TEST_NAME));
        String HOME = SCRIPT_DIR + TEST_DIR;

        try {
            loadTestConfiguration(getTestConfiguration(TEST_NAME));

            //execute given unit test
            fullDMLScriptName = HOME + TEST_NAME + "Unit.dml";
            programArgs = new String[]{"-args", testType, output("R"), output("R_expected")};
            runTest(true, false, null, -1);

            //compare to expected result
            HashMap<CellIndex, Double> result = readDMLMatrixFromOutputDir("R");
            HashMap<CellIndex, Double> result_expected = readDMLMatrixFromOutputDir("R_expected");

            TestUtils.compareMatrices(result, result_expected, 1e-3, testType+"_result", testType+"_expected");

        }
        finally {
            rtplatform = platformOld;
        }
    }
}
