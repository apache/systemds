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
import java.util.HashMap;

public class BuiltinImputeMARTest extends AutomatedTestBase {
    private final static String TEST_NAME = "imputeMARTest";
    private final static String TEST_DIR = "functions/builtin/";
    private static final String TEST_CLASS_DIR = TEST_DIR + BuiltinImputeMARTest.class.getSimpleName() + "/";
    private String DATASET = DATASET_DIR + "ChickWeight.csv";
    private final static double eps = 0.16;
    private final static int iter = 3;

    @Override
    public void setUp() {
        addTestConfiguration(TEST_NAME, new TestConfiguration(TEST_CLASS_DIR, TEST_NAME, new String[]{"LogReg"}));
    }

    @Test
    public void testMiceImputation() {
        runImputationTest("MICE", ExecType.CP);
    }

    @Test
    public void testMeanImputation() {
        runImputationTest("MEAN", ExecType.CP);
    }

    @Test
    public void compareImputationMethods() {
        runImputationTest("MICE", ExecType.CP);
        runImputationTest("MEAN", ExecType.CP);
        compareResults();
    }

    private void runImputationTest(String method, ExecType instType) {
        Types.ExecMode platformOld = setExecMode(instType);
        try {
            System.out.println("Dataset " + DATASET);
            loadTestConfiguration(getTestConfiguration(TEST_NAME));
            String HOME = SCRIPT_DIR + TEST_DIR;
            fullDMLScriptName = HOME + TEST_NAME + ".dml";
            
            double[][] mask = {{0.0, 0.0, 1.0, 1.0, 0.0}};
            writeInputMatrixWithMTD("M", mask, true);

            programArgs = new String[]{"-nvargs", 
                "X=" + DATASET, 
                "Mask=" + input("M"),
                "method=" + method,
                "iteration=" + iter, 
                "LogReg=" + output("LogReg_" + method), 
                "targetCol=4"};

            runTest(true, false, null, -1);
            testLogisticRegression("LogReg_" + method);
        } finally {
            rtplatform = platformOld;
        }
    }

    private void testLogisticRegression(String outputName) {
        HashMap<MatrixValue.CellIndex, Double> logRegResults = readDMLMatrixFromOutputDir(outputName);

        Assert.assertEquals("Incorrect number of logistic regression coefficients", 6, logRegResults.size() - 1);

        for (int i = 1; i <= logRegResults.size() - 1; i++) {
            Double coefficient = logRegResults.get(new MatrixValue.CellIndex(i, 1));
            Assert.assertTrue("Logistic regression coefficient is out of reasonable range: " + coefficient,
                    Math.abs(coefficient) < 10);
        }

        Double auc = logRegResults.get(new MatrixValue.CellIndex(logRegResults.size(), 1));
        Assert.assertTrue("Logistic regression model performance (AUC) is below threshold: " + auc, auc > 0.7);

        Double rSquared = logRegResults.get(new MatrixValue.CellIndex(logRegResults.size(), 1));
        Assert.assertTrue("R-squared value is out of range: " + rSquared, rSquared >= 0 && rSquared <= 1);
        Assert.assertTrue("Model fit (R-squared) is below acceptable threshold: " + rSquared, rSquared > 0.3);
        
        int n = 1000;
        for (int i = 1; i <= logRegResults.size() - 2; i++) {
            Double coefficient = logRegResults.get(new MatrixValue.CellIndex(i, 1));
            Double standardError = Math.abs(coefficient / Math.sqrt(n));
            Double zScore = coefficient / standardError;
            Double pValue = 2 * (1 - cdf(Math.abs(zScore)));
            Assert.assertTrue("P-value for coefficient " + i + " is not significant: " + pValue, pValue < 0.05);
        }
        System.out.println("Model fit (R-squared): " + rSquared);
    }

    private void compareResults() {
        HashMap<MatrixValue.CellIndex, Double> miceResults = readDMLMatrixFromOutputDir("LogReg_MICE");
        HashMap<MatrixValue.CellIndex, Double> meanResults = readDMLMatrixFromOutputDir("LogReg_MEAN");

        System.out.println("Comparison of MICE and Mean Imputation:");
        System.out.println("Coefficient\tMICE\t\tMean\t\tMICE p-value\tMean p-value");

        int n = 1000;
        for (int i = 1; i <= miceResults.size() - 1; i++) {
            Double miceCoef = miceResults.get(new MatrixValue.CellIndex(i, 1));
            Double meanCoef = meanResults.get(new MatrixValue.CellIndex(i, 1));

            Double miceSE = Math.abs(miceCoef / Math.sqrt(n));
            Double meanSE = Math.abs(meanCoef / Math.sqrt(n));

            Double micePValue = 2 * (1 - cdf(Math.abs(miceCoef / miceSE)));
            Double meanPValue = 2 * (1 - cdf(Math.abs(meanCoef / meanSE)));

            System.out.printf("%d\t\t%.4f\t\t%.4f\t\t%.4f\t\t%.4f%n",
                    i, miceCoef, meanCoef, micePValue, meanPValue);
            
            totalDifference += Math.abs(miceCoef - meanCoef);
        }
        
        double averageDifference = totalDifference / coeffCount;
        double threshold = 0.1;

        if (averageDifference < threshold) {
            System.out.println("Data type: MCAR (Missing Completely At Random)");
        } else {
            Double miceRSquared = miceResults.get(new MatrixValue.CellIndex(miceResults.size(), 1));
            Double meanRSquared = meanResults.get(new MatrixValue.CellIndex(meanResults.size(), 1));

            System.out.println("MICE R-squared: " + miceRSquared);
            System.out.println("Mean Imputation R-squared: " + meanRSquared);

            double rSquaredDifference = miceRSquared - meanRSquared;
            double rSquaredThreshold = 0.05;

            if (rSquaredDifference > rSquaredThreshold) {
                System.out.println("Data type: MAR (Missing At Random)");
            } else {
                System.out.println("Data type: NMAR (Not Missing At Random)");
            }
        }
        
    }

    private double cdf(double x) {
        return 0.5 * (1 + erf(x / Math.sqrt(2)));
    }

    private double erf(double z) {
        double t = 1.0 / (1.0 + 0.5 * Math.abs(z));
        double ans = 1 - t * Math.exp(-z * z - 1.26551223 +
                t * (1.00002368 +
                        t * (0.37409196 +
                                t * (0.09678418 +
                                        t * (-0.18628806 +
                                                t * (0.27886807 +
                                                        t * (-1.13520398 +
                                                                t * (1.48851587 +
                                                                        t * (-0.82215223 +
                                                                                t * 0.17087277)))))))));
        return z >= 0 ? ans : -ans;
    }
}
