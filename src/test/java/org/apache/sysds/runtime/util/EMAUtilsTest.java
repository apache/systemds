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

package org.apache.sysds.runtime.util;

import org.apache.sysds.runtime.matrix.data.MatrixValue;
import org.apache.sysds.test.AutomatedTestBase;
import org.apache.sysds.test.TestConfiguration;
import org.junit.Assert;
import org.junit.Test;

import java.util.HashMap;

import static org.apache.sysds.test.TestUtils.readRMatrixFromFS;
import static  org.apache.sysds.runtime.util.EMAUtils.*;

public class EMAUtilsTest extends AutomatedTestBase {
    private static final String TEST_NAME = "exponentialSmoothing";
    private static final String TEST_DIR = "functions/builtin/";
    private static final String TEST_CLASS_DIR = TEST_DIR + EMAUtilsTest.class.getSimpleName() + "/";

    private static final double REFERENCE_EPS = 1e-4;
    private static final Double[] TS_DATA = {
            112., 118., 132., 129., 121., 135., 148., 148., 136., 119., 104., 118.,
            115., 126., 141., 135., 125., 149., 170., 170., 158., 133., 114., 140.,
            145., 150., 178., 163., 172., 178., 199., 199., 184., 162., 146., 166.,
            171., 180., 193., 181., 183., 218., 230., 242., 209., 191., 172., 194.,
            196., 196., 236., 235., 229., 243., 264., 272., 237., 211., 180., 201.
    };
    @Override
    public void setUp() {
        addTestConfiguration(TEST_NAME,
                new TestConfiguration(TEST_CLASS_DIR, TEST_NAME, new String[] { "R_double", "R_triple", "R_triple_odd" }));
    }

    @Test
    public void testDoubleExponentialSmoothing() {
        fullRScriptName = SCRIPT_DIR + TEST_DIR + TEST_NAME + ".R";
        String resultPath = output("R_double");

        rCmd = getRCmd("double", resultPath);
        runRScript(true);

        HashMap<MatrixValue.CellIndex, Double> rOutput = readRMatrixFromFS(resultPath);
        Assert.assertFalse("R output should contain values", rOutput.isEmpty());

        int numRows = (int) rOutput.keySet().stream().mapToLong(idx -> idx.row).max().orElse(0);
        double[] rFittedValues = new double[numRows];
        for (int i = 1; i <= numRows; i++) {
            rFittedValues[i - 1] = rOutput.get(new MatrixValue.CellIndex(i, 1));
        }


        Container javaFittedValues = double_exponential_smoothing(TS_DATA, 0.3, 0.1);

        Assert.assertEquals("Array length mismatch", rFittedValues.length, javaFittedValues.values.length-2);
        for (int i = 0; i < rFittedValues.length; i++) {
            Assert.assertEquals("Mismatch at index " + i, rFittedValues[i], javaFittedValues.values[i+2], REFERENCE_EPS);
        }
    }

    @Test
    public void testTripleExponentialSmoothing(){
        int freq = 12;
        fullRScriptName = SCRIPT_DIR + TEST_DIR + TEST_NAME + ".R";
        String resultPath = output("R_triple");

        rCmd = getRCmd("triple", resultPath);
        runRScript(true);

        HashMap<MatrixValue.CellIndex, Double> rOutputTriple = readRMatrixFromFS(resultPath);
        Assert.assertFalse(
                "R output should contain values",
                rOutputTriple.isEmpty()
        );

        int numRows = (int) rOutputTriple.keySet().stream()
                .mapToLong(idx -> idx.row)
                .max()
                .orElse(0);

        double[] rFittedValues = new double[numRows];

        for (int i = 0; i < numRows; i++) {
            rFittedValues[i] = rOutputTriple.getOrDefault(
                    new MatrixValue.CellIndex(i + 1, 1), 0.0);
        }

        Container javaResult = triple_exponential_smoothing(
                TS_DATA, 0.3, 0.1, 0.2, freq);

        Assert.assertEquals("Array length mismatch", rFittedValues.length, javaResult.values.length - freq);

        for (int i = 0; i < rFittedValues.length; i++) {
            Assert.assertEquals("Mismatch at index " + i, rFittedValues[i], javaResult.values[i + freq], REFERENCE_EPS);
        }
    }

    @Test
    public void testOddTripleExponentialSmoothing(){
        int freq = 7;
        fullRScriptName = SCRIPT_DIR + TEST_DIR + TEST_NAME + ".R";
        String resultPath = output("R_triple_odd");

        rCmd = getRCmd("odd_triple", resultPath);
        runRScript(true);

        HashMap<MatrixValue.CellIndex, Double> rOutputTriple = readRMatrixFromFS(resultPath);
        Assert.assertFalse(
                "R output should contain values",
                rOutputTriple.isEmpty()
        );

        int numRows = (int) rOutputTriple.keySet().stream()
                .mapToLong(idx -> idx.row)
                .max()
                .orElse(0);

        double[] rFittedValues = new double[numRows];

        for (int i = 0; i < numRows; i++) {
            rFittedValues[i] = rOutputTriple.getOrDefault(
                    new MatrixValue.CellIndex(i + 1, 1), 0.0);
        }

        Container javaResult = triple_exponential_smoothing(
                TS_DATA, 0.3, 0.1, 0.2, freq);

        Assert.assertEquals("Array length mismatch", rFittedValues.length, javaResult.values.length - freq);

        for (int i = 0; i < rFittedValues.length; i++) {
            Assert.assertEquals("Mismatch at index " + i, rFittedValues[i], javaResult.values[i + freq], REFERENCE_EPS);
        }
    }
}