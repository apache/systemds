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


import org.apache.sysds.runtime.matrix.data.MatrixValue;
import org.apache.sysds.test.AutomatedTestBase;
import org.apache.sysds.test.TestUtils;
import org.junit.Test;

import java.util.HashMap;


public class BuiltinAutoencoder2LayerTest extends AutomatedTestBase
{
	private final static String TEST_NAME = "autoencoder_2layer";
	private final static String TEST_DIR = "functions/builtin/";
	private static final String TEST_CLASS_DIR = TEST_DIR + BuiltinAutoencoder2LayerTest.class.getSimpleName() + "/";

	private final static int rows = 1058;
	private final static int cols = 784;
	private final static double sparse = 0.1;
	private final static double dense = 0.7;
	private final static double tolerance = 2e-3;

	private static int batchSize = 256;
	private static double step = 1e-5;
	private static double decay = 0.95;
	private static double momentum = 0.9;
	private static boolean obj = false;

	@Override
	public void setUp() {
		TestUtils.clearAssertionInformation();
		addTestConfiguration(TEST_CLASS_DIR, TEST_NAME);
	}

	public void restoreDefaultParams(){
		batchSize = 256;
		step = 1e-5;
		decay = 0.95;
		momentum = 0.9;
		obj = false;
	}

	@Test
	public void testAutoencoderSparse80To5() {
		runAutoencoderTest(80, 5, 2, sparse);
	}

	@Test
	public void testAutoencoderDense80To5() {
		runAutoencoderTest(80, 5, 2, dense);
	}
	
	@Test
	public void testAutoencoderSparse10To4() {
		runAutoencoderTest(10, 4, 2, sparse);
	}

	@Test
	public void testAutoencoderDense10To4() {
		runAutoencoderTest(10, 4, 2, dense);
	}

	@Test
	public void testAutoencoderSparse200To20FullObj() {
		obj = true;
		runAutoencoderTest(200, 20, 2, sparse);
		restoreDefaultParams();
	}

	@Test
	public void testAutoencoderDense120To10Batch512() {
		batchSize = 512;
		runAutoencoderTest(120, 10, 2, dense);
		restoreDefaultParams();
	}

	@Test
	public void testAutoencoderSparse200To12DecMomentum() {
		momentum = 0.8;
		runAutoencoderTest(200, 12, 2, sparse);
		restoreDefaultParams();
	}

	@Test
	public void testAutoencoderSparse200To12IncMomentum() {
		momentum = 0.95;
		runAutoencoderTest(200, 12, 2, sparse);
		restoreDefaultParams();
	}

	@Test
	public void testAutoencoderDense20To3DecDecay() {
		decay = 0.85;
		runAutoencoderTest(20, 3, 2, dense);
		restoreDefaultParams();
	}

	@Test
	public void testAutoencoderDense500To3FullObjBatch512IncStep() {
		obj = true;
		batchSize = 512;
		step = 1e-4;
		runAutoencoderTest(500, 3, 2, dense);
		restoreDefaultParams();
	}

	@Test
	public void testAutoencoderSparse354To7FullObjBatch512DecStepIncMomentumDecDecay() {
		obj = true;
		batchSize = 512;
		step = 1e-6;
		momentum = 0.95;
		decay = 0.90;
		runAutoencoderTest(354, 7, 2, sparse);
		restoreDefaultParams();
	}

	private void runAutoencoderTest(int numHidden1, int numHidden2, int maxEpochs, double sparsity) {
		loadTestConfiguration(getTestConfiguration(TEST_NAME));

		String HOME = SCRIPT_DIR + TEST_DIR;
		fullDMLScriptName = HOME + TEST_NAME + ".dml";
		String fullObj = obj ? "TRUE" : "FALSE";
		programArgs = new String[]{ "-stats", "-nvargs", "X="+input("X"),
			"H1="+numHidden1, "H2="+numHidden2, "EPOCH="+maxEpochs, "BATCH="+batchSize,
			"STEP="+step, "DECAY="+decay, "MOMENTUM="+momentum, "OBJ="+fullObj,
			"W1_rand="+input("W1_rand"),"W2_rand="+input("W2_rand"),
			"W3_rand="+input("W3_rand"), "W4_rand="+input("W4_rand"),
			"order_rand="+input("order_rand"),
			"W1_out="+output("W1_out"), "b1_out="+output("b1_out"),
			"W2_out="+output("W2_out"), "b2_out="+output("b2_out"),
			"W3_out="+output("W3_out"), "b3_out="+output("b3_out"),
			"W4_out="+output("W4_out"), "b4_out="+output("b4_out"),
			"hidden_out="+output("hidden_out")};
		fullRScriptName = HOME + TEST_NAME + ".R";
		rCmd = getRCmd(inputDir(), String.valueOf(numHidden1), String.valueOf(numHidden2), String.valueOf(maxEpochs),
			String.valueOf(batchSize), String.valueOf(momentum), String.valueOf(step), String.valueOf(decay), fullObj, expectedDir());

		double[][] X = getRandomMatrix(rows, cols, 0, 1, sparsity, 27);
		writeInputMatrixWithMTD("X", X, true);

		//generate rand matrices for W1, W2, W3, W4 here itself for passing onto both DML and R scripts
		double[][] W1_rand = getRandomMatrix(numHidden1, cols, 0, 1, sparsity, 800);
		writeInputMatrixWithMTD("W1_rand", W1_rand, true);
		double[][] W2_rand = getRandomMatrix(numHidden2, numHidden1, 0, 1, sparsity, 900);
		writeInputMatrixWithMTD("W2_rand", W2_rand, true);
		double[][] W3_rand = getRandomMatrix(numHidden1, numHidden2, 0, 1, sparsity, 589);
		writeInputMatrixWithMTD("W3_rand", W3_rand, true);
		double[][] W4_rand = getRandomMatrix(cols, numHidden1, 0, 1, sparsity, 150);
		writeInputMatrixWithMTD("W4_rand", W4_rand, true);
		double[][] order_rand = getRandomMatrix(rows, 1, 0, 1, sparsity, 143);
		writeInputMatrixWithMTD("order_rand", order_rand, true); //for the permut operation on input X
		runTest(true, false, null, -1);

		runRScript(true);

		//compare matrices
		HashMap<MatrixValue.CellIndex, Double> w1OutDML = readDMLMatrixFromOutputDir("W1_out");
		HashMap<MatrixValue.CellIndex, Double> b1OutDML = readDMLMatrixFromOutputDir("b1_out");
		HashMap<MatrixValue.CellIndex, Double> w2OutDML = readDMLMatrixFromOutputDir("W2_out");
		HashMap<MatrixValue.CellIndex, Double> b2OutDML = readDMLMatrixFromOutputDir("b2_out");
		HashMap<MatrixValue.CellIndex, Double> w3OutDML = readDMLMatrixFromOutputDir("W3_out");
		HashMap<MatrixValue.CellIndex, Double> b3OutDML = readDMLMatrixFromOutputDir("b3_out");
		HashMap<MatrixValue.CellIndex, Double> w4OutDML = readDMLMatrixFromOutputDir("W4_out");
		HashMap<MatrixValue.CellIndex, Double> b4OutDML = readDMLMatrixFromOutputDir("b4_out");

		HashMap<MatrixValue.CellIndex, Double> w1OutR = readRMatrixFromExpectedDir("W1_out");
		HashMap<MatrixValue.CellIndex, Double> b1OutR = readRMatrixFromExpectedDir("b1_out");
		HashMap<MatrixValue.CellIndex, Double> w2OutR = readRMatrixFromExpectedDir("W2_out");
		HashMap<MatrixValue.CellIndex, Double> b2OutR = readRMatrixFromExpectedDir("b2_out");
		HashMap<MatrixValue.CellIndex, Double> w3OutR = readRMatrixFromExpectedDir("W3_out");
		HashMap<MatrixValue.CellIndex, Double> b3OutR = readRMatrixFromExpectedDir("b3_out");
		HashMap<MatrixValue.CellIndex, Double> w4OutR = readRMatrixFromExpectedDir("W4_out");
		HashMap<MatrixValue.CellIndex, Double> b4OutR = readRMatrixFromExpectedDir("b4_out");

		TestUtils.compareMatrices(w1OutDML, w1OutR, tolerance, "W1-DML", "W1-R");
		TestUtils.compareMatrices(b1OutDML, b1OutR, tolerance, "b1-DML", "b1-R");
		TestUtils.compareMatrices(w2OutDML, w2OutR, tolerance, "W2-DML", "W2-R");
		TestUtils.compareMatrices(b2OutDML, b2OutR, tolerance, "b2-DML", "b2-R");
		TestUtils.compareMatrices(w3OutDML, w3OutR, tolerance, "W3-DML", "W3-R");
		TestUtils.compareMatrices(b3OutDML, b3OutR, tolerance, "b3-DML", "b3-R");
		TestUtils.compareMatrices(w4OutDML, w4OutR, tolerance, "W4-DML", "W4-R");
		TestUtils.compareMatrices(b4OutDML, b4OutR, tolerance, "b4-DML", "b4-R");
	}
}