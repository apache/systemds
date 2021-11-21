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

import org.apache.sysds.runtime.matrix.data.MatrixValue.CellIndex;
import org.apache.sysds.test.AutomatedTestBase;
import org.apache.sysds.test.TestConfiguration;
import org.apache.sysds.test.TestUtils;
import org.junit.Test;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;

public class BuiltinNaiveBayesPredictTest extends AutomatedTestBase {
	private final static String TEST_NAME = "NaiveBayesPredict";
	private final static String TEST_DIR = "functions/builtin/";
	private final static String TEST_CLASS_DIR = TEST_DIR + BuiltinNaiveBayesPredictTest.class.getSimpleName() + "/";
	private final static int numClasses = 10;

	public double eps = 1e-7;

	@Override
	public void setUp() {
		TestUtils.clearAssertionInformation();
		addTestConfiguration(TEST_NAME, new TestConfiguration(TEST_CLASS_DIR, TEST_NAME, new String[] {"YRaw", "Y"}));
	}

	@Test
	public void testSmallDense() {
		testNaiveBayesPredict(100, 50, 0.7);
	}

	@Test
	public void testLargeDense() {
		testNaiveBayesPredict(10000, 750, 0.7);
	}

	@Test
	public void testSmallSparse() {
		testNaiveBayesPredict(100, 50, 0.01);
	}

	@Test
	public void testLargeSparse() {
		testNaiveBayesPredict(10000, 750, 0.01);
	}

	public void testNaiveBayesPredict(int rows, int cols, double sparsity) {
		loadTestConfiguration(getTestConfiguration(TEST_NAME));
		String HOME = SCRIPT_DIR + TEST_DIR;
		fullDMLScriptName = HOME + TEST_NAME + ".dml";

		int classes = numClasses;
		double laplace = 1;

		List<String> proArgs = new ArrayList<>();
		proArgs.add("-args");
		proArgs.add(input("D"));
		proArgs.add(input("C"));
		proArgs.add(String.valueOf(classes));
		proArgs.add(String.valueOf(laplace));
		proArgs.add(output("YRaw"));
		proArgs.add(output("Y"));
		programArgs = proArgs.toArray(new String[proArgs.size()]);

		rCmd = getRCmd(inputDir(), Integer.toString(classes), Double.toString(laplace), expectedDir());

		double[][] D = getRandomMatrix(rows, cols, 0, 1, sparsity, -1);
		double[][] C = getRandomMatrix(rows, 1, 0, 1, 1, -1);
		for(int i = 0; i < rows; i++) {
			C[i][0] = (int) (C[i][0] * classes) + 1;
			C[i][0] = (C[i][0] > classes) ? classes : C[i][0];
		}

		writeInputMatrixWithMTD("D", D, true);
		writeInputMatrixWithMTD("C", C, true);

		runTest(true, EXCEPTION_NOT_EXPECTED, null, -1);

		runRScript(true);

		HashMap<CellIndex, Double> YRawR = readRMatrixFromExpectedDir("YRaw");
		HashMap<CellIndex, Double> YR = readRMatrixFromExpectedDir("Y");
		HashMap<CellIndex, Double> YRawSYSTEMDS = readDMLMatrixFromOutputDir("YRaw");
		HashMap<CellIndex, Double> YSYSTEMDS = readDMLMatrixFromOutputDir("Y");
		TestUtils.compareMatrices(YRawR, YRawSYSTEMDS, eps, "YRawR", "YRawSYSTEMDS");
		TestUtils.compareMatrices(YR, YSYSTEMDS, eps, "YR", "YSYSTEMDS");
	}
}
