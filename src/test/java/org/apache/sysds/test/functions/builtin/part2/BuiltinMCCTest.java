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

import org.apache.sysds.common.Types.ExecMode;


import org.apache.commons.lang.ArrayUtils;
import org.apache.sysds.runtime.lineage.LineageCacheConfig.ReuseCacheType;
import org.apache.sysds.runtime.matrix.data.MatrixValue.CellIndex;
import org.apache.sysds.test.AutomatedTestBase;
import org.apache.sysds.test.TestConfiguration;
import org.apache.sysds.test.TestUtils;
import org.junit.Assert;
import org.junit.Test;


public class BuiltinMCCTest extends AutomatedTestBase {
    private final static String TEST_NAME = "mcc";
	private final static String TEST_DIR = "functions/builtin/";
	private final static String TEST_CLASS_DIR = TEST_DIR + BuiltinMCCTest.class.getSimpleName() + "/";

    private final static String OUTPUT_IDENTIFIER = "mattCorrCoeff.scalar";
    private final static double epsilon = 1e-10;

    @Override
	public void setUp() {
        TestConfiguration tc = new TestConfiguration(TEST_CLASS_DIR, TEST_NAME, new String[]{OUTPUT_IDENTIFIER});
		addTestConfiguration(TEST_NAME, tc);
	}

    @Test
    public void testMCCCorrect1() {
        double[][] predictions = {{1},{1},{1},{0},{1},{1},{0},{0},{0},{1}};
        double[][] labels = {{1},{1},{1},{1},{1},{0},{0},{0},{0},{0}};
        boolean expectException = false;
        runMCCTest(predictions, labels, false, ExecMode.HYBRID, expectException);
    }

    @Test
    public void testMCCCorrect_2() {
        double[][] predictions = {{0},{0},{0},{0},{0},{0},{0},{0},{0},{0}};
        double[][] labels = {{1},{1},{1},{0},{1},{0},{1},{1},{0},{1}};
        boolean expectException = false;
        runMCCTest(predictions, labels, false, ExecMode.HYBRID, expectException);
    }

    @Test
    public void testMCCIncorrectSP() {
        double[][] predictions = {{0},{1},{1},{0},{1},{1},{0},{0},{0},{1}};
        double[][] labels = {{1},{1},{1},{1},{1},{0},{0},{0},{0},{0}};
        boolean expectException = false;
        runMCCTest(predictions, labels, false, ExecMode.SPARK, expectException);
    }

    @Test
    public void testMCCCorrectLarge() {
        double[][] predictions = getRandomMatrix(100000, 1, 0.0, 1.0, 1.0, 7);
        double[][] labels = getRandomMatrix(100000, 1, 0.0, 1.0, 1.0, 11);
        for (int row = 0; row < predictions.length; row++) {
            predictions[row][0] = Math.round(predictions[row][0]);
            labels[row][0] = Math.round(labels[row][0]);
        }
        boolean expectException = false;
        runMCCTest(predictions, labels, false, ExecMode.HYBRID, expectException);
    }

    @Test
    public void testMCCIncorrect_1() {
        double[][] predictions = {{-1},{-1},{-1},{-1},{-1},{-1},{-1},{-1},{-1},{-1}};
        double[][] labels = {{99},{99},{99},{99},{99},{99},{99},{99},{99},{99}};
        boolean expectException = true;
        runMCCTest(predictions, labels, false, ExecMode.HYBRID, expectException);
    }

    @Test
    public void testMCCIncorrect_2() {
        double[][] predictions = {{1},{1},{1},{0},{1},{1},{0},{0},{0},{-1}};
        double[][] labels = {{99},{1},{1},{1},{1},{0},{0},{0},{0},{0}};
        boolean expectException = true;
        runMCCTest(predictions, labels, false, ExecMode.HYBRID, expectException);
    }
    
    private void runMCCTest(double[][] predictions, double[][] labels, boolean lineage, ExecMode mode, boolean expectException) {
        ExecMode execModeOld = setExecMode(mode);
        try {
            loadTestConfiguration(getTestConfiguration(TEST_NAME));
            String HOME = SCRIPT_DIR + TEST_DIR;
			fullDMLScriptName = HOME + TEST_NAME + ".dml";
            programArgs = new String[]{
                "-nvargs", 
                "predictions="+input("predictions"),
                "labels=" + input("labels"),
                "mattCorrCoeff=" + output(OUTPUT_IDENTIFIER),
            };
            if (lineage) {
                programArgs = (String[]) ArrayUtils.addAll(programArgs, new String[] {
                    "-stats","-lineage", ReuseCacheType.REUSE_HYBRID.name().toLowerCase()});
            }
            writeInputMatrixWithMTD("labels", labels, true);
            writeInputMatrixWithMTD("predictions", predictions, true);

            fullRScriptName = HOME + TEST_NAME + ".R";
			rCmd = getRCmd(inputDir(), expected(OUTPUT_IDENTIFIER));

            runTest(true, expectException, null, -1); 
            if (!expectException) {
                runRScript(true);
                Double mattCorrCoeffDML = readDMLScalarFromOutputDir(OUTPUT_IDENTIFIER).get(new CellIndex(1,1));
                Assert.assertTrue(-1 <= mattCorrCoeffDML && mattCorrCoeffDML <= 1);
                Double mattCorrCoeffR = readRScalarFromExpectedDir(OUTPUT_IDENTIFIER).get(new CellIndex(1,1));
                TestUtils.compareScalars(mattCorrCoeffDML, mattCorrCoeffR, epsilon);
            }
            
        } finally {
            resetExecMode(execModeOld);
        }
    }

}
