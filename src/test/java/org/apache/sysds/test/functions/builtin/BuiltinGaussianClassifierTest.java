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

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;

import org.apache.sysds.runtime.matrix.data.MatrixValue.CellIndex;
import org.apache.sysds.test.AutomatedTestBase;
import org.apache.sysds.test.TestConfiguration;
import org.apache.sysds.test.TestUtils;
import org.junit.Test;

public class BuiltinGaussianClassifierTest extends AutomatedTestBase
{
	private final static String TEST_NAME = "GaussianClassifier";
	private final static String TEST_DIR = "functions/builtin/";
	private final static String TEST_CLASS_DIR = TEST_DIR + BuiltinGaussianClassifierTest.class.getSimpleName() + "/";

	private final static String DATASET = SCRIPT_DIR + "functions/transform/input/iris/iris.csv";


	@Override
	public void setUp() {
		addTestConfiguration(TEST_NAME,new TestConfiguration(TEST_CLASS_DIR, TEST_NAME,new String[]{"B"})); 
	}

	@Test
	public void testSmallDenseFiveClasses() {
		testGaussianClassifier(80, 10, 0.9, 5);
	}

	@Test
	public void testSmallDenseTenClasses() {
		testGaussianClassifier(80, 30, 0.9, 10);
	}

	@Test
	public void testBiggerDenseFiveClasses() { testGaussianClassifier(200, 50, 0.9, 5);}

	@Test
	public void testBiggerDenseTenClasses() {
		testGaussianClassifier(200, 50, 0.9, 10);
	}

	@Test
	public void testBiggerSparseFiveClasses() {
		testGaussianClassifier(200, 50, 0.3, 5);
	}

	@Test
	public void testBiggerSparseTenClasses() {
		testGaussianClassifier(200, 50, 0.3, 10);
	}

	@Test
	public void testSmallSparseFiveClasses() {
		testGaussianClassifier(80, 30, 0.3, 5);
	}

	@Test
	public void testSmallSparseTenClasses() {
		testGaussianClassifier(80, 30, 0.3, 10);
	}

	@SuppressWarnings("unused")
	public void testGaussianClassifier(int rows, int cols, double sparsity, int classes)
	{
		loadTestConfiguration(getTestConfiguration(TEST_NAME));
		String HOME = SCRIPT_DIR + TEST_DIR;
		fullDMLScriptName = HOME + TEST_NAME + ".dml";

		double varSmoothing = 1e-9;

		List<String> proArgs = new ArrayList<>();
		proArgs.add("-args");
		proArgs.add(input("X"));
		proArgs.add(input("Y"));
		proArgs.add(String.valueOf(varSmoothing));
		proArgs.add(output("priors"));
		proArgs.add(output("means"));
		proArgs.add(output("determinants"));
		proArgs.add(output("invcovs"));

		programArgs = proArgs.toArray(new String[proArgs.size()]);

		rCmd = getRCmd(inputDir(), Double.toString(varSmoothing), expectedDir());
		
		double[][] X = getRandomMatrix(rows, cols, 0, 100, sparsity, -1);
		double[][] Y = getRandomMatrix(rows, 1, 0, 1, 1, -1);
		for(int i=0; i<rows; i++){
			Y[i][0] = (int)(Y[i][0]*classes) + 1;
			Y[i][0] = (Y[i][0] > classes) ? classes : Y[i][0];
		}

		writeInputMatrixWithMTD("X", X, true);
		writeInputMatrixWithMTD("Y", Y, true);

		runTest(true, EXCEPTION_NOT_EXPECTED, null, -1);

		runRScript(true);

		HashMap<CellIndex, Double> priorR = readRMatrixFromExpectedDir("priors");
		HashMap<CellIndex, Double> priorSYSTEMDS= readDMLMatrixFromOutputDir("priors");
		HashMap<CellIndex, Double> meansRtemp = readRMatrixFromExpectedDir("means");
		HashMap<CellIndex, Double> meansSYSTEMDStemp = readDMLMatrixFromOutputDir("means");
		HashMap<CellIndex, Double> determinantsRtemp = readRMatrixFromExpectedDir("determinants");
		HashMap<CellIndex, Double> determinantsSYSTEMDStemp = readDMLMatrixFromOutputDir("determinants");
		HashMap<CellIndex, Double> invcovsRtemp = readRMatrixFromExpectedDir("invcovs");
		HashMap<CellIndex, Double> invcovsSYSTEMDStemp = readDMLMatrixFromOutputDir("invcovs");

		double[][] meansR = TestUtils.convertHashMapToDoubleArray(meansRtemp);
		double[][] meansSYSTEMDS = TestUtils.convertHashMapToDoubleArray(meansSYSTEMDStemp);
		double[][] determinantsR = TestUtils.convertHashMapToDoubleArray(determinantsRtemp);
		double[][] determinantsSYSTEMDS = TestUtils.convertHashMapToDoubleArray(determinantsSYSTEMDStemp);
		double[][] invcovsR = TestUtils.convertHashMapToDoubleArray(invcovsRtemp);
		double[][] invcovsSYSTEMDS = TestUtils.convertHashMapToDoubleArray(invcovsSYSTEMDStemp);

		TestUtils.compareMatrices(priorR, priorSYSTEMDS, Math.pow(10, -5.0), "priorR", "priorSYSTEMDS");
//		TODO: stable the following comparision
//		TestUtils.compareMatricesBitAvgDistance(meansR, meansSYSTEMDS, 10L,10L, this.toString());
//		TestUtils.compareMatricesBitAvgDistance(determinantsR, determinantsSYSTEMDS, (long)2E+12,(long)2E+12, this.toString());
//		TestUtils.compareMatricesBitAvgDistance(invcovsR, invcovsSYSTEMDS, (long)2E+20,(long)2E+20, this.toString());
	}

	@Test
	public void testIrisPrediction()
	{
		loadTestConfiguration(getTestConfiguration(TEST_NAME));
		String HOME = SCRIPT_DIR + TEST_DIR;
		fullDMLScriptName = HOME + TEST_NAME + "Prediction.dml";

		double varSmoothing = 1e-9;

		List<String> proArgs = new ArrayList<>();
		proArgs.add("-args");
		proArgs.add(DATASET);
		proArgs.add(String.valueOf(varSmoothing));
		proArgs.add(output("trueLabels"));
		proArgs.add(output("predLabels"));

		programArgs = proArgs.toArray(new String[proArgs.size()]);
		runTest(true, EXCEPTION_NOT_EXPECTED, null, -1);

		HashMap<CellIndex, Double> trueLabels = readDMLMatrixFromOutputDir("trueLabels");
		HashMap<CellIndex, Double> predLabels = readDMLMatrixFromOutputDir("predLabels");

		TestUtils.compareMatrices(trueLabels, predLabels, 0, "trueLabels", "predLabels");
	}
}
