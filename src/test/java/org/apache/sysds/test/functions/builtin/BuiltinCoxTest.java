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

package org.apache.sysds.test.functions.builtin;

import org.apache.sysds.common.Types;
import org.apache.sysds.test.AutomatedTestBase;
import org.apache.sysds.test.TestConfiguration;
import org.junit.Test;

import static org.apache.sysds.test.TestUtils.*;

public class BuiltinCoxTest extends AutomatedTestBase
{
	private final static String TEST_NAME = "cox";
	private final static String TEST_DIR = "functions/builtin/";
	private final static String TEST_CLASS_DIR = TEST_DIR + BuiltinCoxTest.class.getSimpleName() + "/";

	private final static double spDense = 0.99;

	@Override
	public void setUp() {
		addTestConfiguration(TEST_NAME, new TestConfiguration(TEST_CLASS_DIR, TEST_NAME,new String[]{"B"}));
	}

	@Test
	public void testFunction() {
		runCoxTest(1000, 2.0, 1.5, 0.8, 100, 0.05, 100, 0);
	}
	
	public void runCoxTest(int numRecords, double scaleWeibull, double shapeWeibull, double prob,
						   int numFeatures, double alpha, int moi, int mii) {
		Types.ExecMode platformOld = setExecMode(Types.ExecMode.SPARK);
		loadTestConfiguration(getTestConfiguration(TEST_NAME));
		String HOME = SCRIPT_DIR + TEST_DIR;
		fullDMLScriptName = HOME + TEST_NAME + ".dml";
		int seed = 11;

		programArgs = new String[]{
				"-nvargs", "X=" + input("X"), "TE=" + input("TE"), "F=" + input("F"),
				"M=" + output("M"), "S=" + output("S"), "T=" + output("T"),
				"COV=" + output("COV"), "RT=" + output("RT"), "XO=" + output("XO"), "MF=" + output("MF"),
				"alpha=" + alpha, "moi=" + moi, "mii=" + mii};

		double[][] X = getRandomMatrix(numRecords, numFeatures, 1, 5, spDense, seed);
		writeInputMatrixWithMTD("X", X, false);

		double[][] B = getRandomMatrix (numFeatures, 1, -1.0, 1.0, spDense, seed);
		double[][] U = getRandomMatrix(numRecords, 1, 0.000000001, 1, spDense, seed);
		double[][] TE = new double[numRecords][2];

		double probCensor = 1 - prob;
		double[][] event = floor(getRandomMatrix(numRecords, 1, 1 - probCensor, 1 + prob, spDense, seed));
		double nTime = sum(event, numRecords, 1);

		for(int i = 0; i < numRecords; i++) {
			TE[i][1] = event[i][0];
		}

		double[][] T = new double[numRecords][1];
		double max_T = 0;
		double min_T = 0;

		double[][] multipliedMatrices = performMatrixMultiplication(X, B);
		for(int i = 0; i < numRecords; i++) {
			max_T = Double.MIN_VALUE;
			min_T = Double.MAX_VALUE;

			T[i][0] = Math.pow(- Math.log(U[i][0])/ (scaleWeibull * Math.exp(multipliedMatrices[i][0])), (1/shapeWeibull));

			if (T[i][0] > max_T) {
				max_T = T[i][0];
			}
			if (T[i][0] < min_T) {
				min_T = T[i][0];
			}
		}

		double len = max_T - min_T;
		double numBins = len / nTime;
		for (int i = 0; i < numRecords; i++) {
			T[i][0] = T[i][0] / numBins;
		}
		ceil(T);

		for(int i = 0; i < numRecords; i++) {
			TE[i][0] = T[i][0];
		}

		writeInputMatrixWithMTD("TE", TE, false);

		if (numFeatures > 0) {
			double[][] F = new double[numFeatures][1];

			for (int i = 0; i < numFeatures; i++) {
				F[i][0] = i+1;
			}

			writeInputMatrixWithMTD("F", F, false);
		}

		runTest(true, false, null, -1);

	}
}
