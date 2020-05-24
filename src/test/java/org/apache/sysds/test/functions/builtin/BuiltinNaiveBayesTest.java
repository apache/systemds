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

import org.junit.Test;
import org.apache.sysds.runtime.matrix.data.MatrixValue.CellIndex;
import org.apache.sysds.test.AutomatedTestBase;
import org.apache.sysds.test.TestConfiguration;
import org.apache.sysds.test.TestUtils;

public class BuiltinNaiveBayesTest extends AutomatedTestBase 
{
	private final static String TEST_NAME = "NaiveBayes";
	private final static String TEST_DIR = "functions/builtin/";
	private final static String TEST_CLASS_DIR = TEST_DIR + BuiltinNaiveBayesTest.class.getSimpleName() + "/";

	private final static int numClasses = 10;

	@Override
	public void setUp() {
		addTestConfiguration(TEST_NAME,new TestConfiguration(TEST_CLASS_DIR, TEST_NAME,new String[]{"B"})); 
	}
	
	@Test
	public void testSmallDense() {
		testNaiveBayes(100, 50, 0.7);
	}

	@Test
	public void testLargeDense() {
		testNaiveBayes(10000, 750, 0.7);
	}

	@Test
	public void testSmallSparse() {
		testNaiveBayes(100, 50, 0.01);
	}

	@Test
	public void testLargeSparse() {
		testNaiveBayes(10000, 750, 0.01);
	}
	
	public void testNaiveBayes(int rows, int cols, double sparsity)
	{
		loadTestConfiguration(getTestConfiguration(TEST_NAME));
		String HOME = SCRIPT_DIR + TEST_DIR;
		fullDMLScriptName = HOME + TEST_NAME + ".dml";
		
		int classes = numClasses;
		double laplace_correction = 1;

		List<String> proArgs = new ArrayList<>();
		proArgs.add("-args");
		proArgs.add(input("X"));
		proArgs.add(input("Y"));
		proArgs.add(String.valueOf(classes));
		proArgs.add(String.valueOf(laplace_correction));
		proArgs.add(output("prior"));
		proArgs.add(output("conditionals"));
		programArgs = proArgs.toArray(new String[proArgs.size()]);

		rCmd = getRCmd(inputDir(), Integer.toString(classes), Double.toString(laplace_correction), expectedDir());
		
		double[][] X = getRandomMatrix(rows, cols, 0, 1, sparsity, -1);
		double[][] Y = getRandomMatrix(rows, 1, 0, 1, 1, -1);
		for(int i=0; i<rows; i++){
			Y[i][0] = (int)(Y[i][0]*classes) + 1;
			Y[i][0] = (Y[i][0] > classes) ? classes : Y[i][0];
		}

		writeInputMatrixWithMTD("X", X, true);
		writeInputMatrixWithMTD("Y", Y, true);

		runTest(true, EXCEPTION_NOT_EXPECTED, null, -1);

		runRScript(true);

		HashMap<CellIndex, Double> priorR = readRMatrixFromFS("prior");
		HashMap<CellIndex, Double> priorSYSTEMDS= readDMLMatrixFromHDFS("prior");
		HashMap<CellIndex, Double> conditionalsR = readRMatrixFromFS("conditionals");
		HashMap<CellIndex, Double> conditionalsSYSTEMDS = readDMLMatrixFromHDFS("conditionals");
		TestUtils.compareMatrices(priorR, priorSYSTEMDS, Math.pow(10, -12), "priorR", "priorSYSTEMDS");
		TestUtils.compareMatrices(conditionalsR, conditionalsSYSTEMDS, Math.pow(10.0, -12.0), "conditionalsR", "conditionalsSYSTEMDS");
	}
}
