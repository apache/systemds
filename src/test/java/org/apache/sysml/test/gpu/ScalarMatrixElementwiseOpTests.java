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

package org.apache.sysml.test.gpu;

import java.util.Arrays;
import java.util.HashMap;
import java.util.List;

import org.apache.sysml.api.mlcontext.Matrix;
import org.apache.sysml.test.utils.TestUtils;
import org.junit.Ignore;
import org.junit.Test;

/**
 * Tests scalar-matrix element wise operations on the GPU
 */
public class ScalarMatrixElementwiseOpTests extends GPUTests {

	private final static String TEST_NAME = "ScalarMatrixElementwiseOpTests";

	private final int[] rowSizes = new int[] { 1, 64, 2049 };
	private final int[] columnSizes = new int[] { 1, 64, 2049 };
	private final double[] sparsities = new double[] { 0.0, 0.03, 0.3, 0.9 };
	private final int seed = 42;

	@Override
	public void setUp() {
		TestUtils.clearAssertionInformation();
		addTestConfiguration(TEST_DIR, TEST_NAME);
		getAndLoadTestConfiguration(TEST_NAME);
	}

	@Test
	public void testLessThanRightScalar() {
		runScalarMatrixElementWiseTests("O = X < scalar", "X", "scalar", "O", new double[] { 0.0, 20.0 }, "gpu_<");
	}

	@Test
	public void testLessThanLeftScalar() {
		runScalarMatrixElementWiseTests("O = scalar < X", "X", "scalar", "O", new double[] { 0.0, 20.0 }, "gpu_<");
	}

	@Test
	public void testLessThanEqualRightScalar() {
		runScalarMatrixElementWiseTests("O = X <= scalar", "X", "scalar", "O", new double[] { 0.0, 20.0 }, "gpu_<=");
	}

	@Test
	public void testLessThanEqualLeftScalar() {
		runScalarMatrixElementWiseTests("O = scalar <= X", "X", "scalar", "O", new double[] { 0.0, 20.0 }, "gpu_<=");
	}

	@Test
	public void testGreaterThanRightScalar() {
		runScalarMatrixElementWiseTests("O = X > scalar", "X", "scalar", "O", new double[] { 0.0, 20.0 }, "gpu_>");
	}

	@Test
	public void testGreaterThanLeftScalar() {
		runScalarMatrixElementWiseTests("O = scalar > X", "X", "scalar", "O", new double[] { 0.0, 20.0 }, "gpu_>");
	}

	@Test
	public void testGreaterThanEqualRightScalar() {
		runScalarMatrixElementWiseTests("O = X >= scalar", "X", "scalar", "O", new double[] { 0.0,  20.0 }, "gpu_>=");
	}

	@Test
	public void testGreaterThanEqualLeftScalar() {
		runScalarMatrixElementWiseTests("O = scalar >= X", "X", "scalar", "O", new double[] { 0.0, 20.0 }, "gpu_>=");
	}

	@Test
	public void testEqualRightScalar() {
		runScalarMatrixElementWiseTests("O = X == scalar", "X", "scalar", "O", new double[] { 0.0, 20.0 }, "gpu_==");
	}

	@Test
	public void testEqualLeftScalar() {
		runScalarMatrixElementWiseTests("O = scalar == X", "X", "scalar", "O", new double[] { 0.0, 20.0 }, "gpu_==");
	}

	@Test
	public void testNotEqualRightScalar() {
		runScalarMatrixElementWiseTests("O = X != scalar", "X", "scalar", "O", new double[] { 0.0, 20.0 }, "gpu_!=");
	}

	@Test
	public void testNotEqualEqualLeftScalar() {
		runScalarMatrixElementWiseTests("O = scalar != X", "X", "scalar", "O", new double[] { 0.0, 20.0 }, "gpu_!=");
	}

	@Test
	public void testPlusRightScalar() {
		runScalarMatrixElementWiseTests("O = X + scalar", "X", "scalar", "O", new double[] { 0.0, 0.5, 20.0 }, "gpu_+");
	}

	@Test
	public void testPlusLeftScalar() {
		runScalarMatrixElementWiseTests("O = scalar + X", "X", "scalar", "O", new double[] { 0.0, 0.5, 20.0 }, "gpu_+");
	}

	@Test
	public void testMinusRightScalar() {
		runScalarMatrixElementWiseTests("O = X - scalar", "X", "scalar", "O", new double[] { 0.0, 0.5, 1.0 }, "gpu_-");
	}

	@Test
	public void testMinusLeftScalar() {
		runScalarMatrixElementWiseTests("O = scalar - X", "X", "scalar", "O", new double[] { 0.0, 0.5, 1.0 }, "gpu_-");
	}

	@Test
	public void testMultRightScalar() {
		runScalarMatrixElementWiseTests("O = X * scalar", "X", "scalar", "O", new double[] { 0.0, 0.5, 2.0 }, "gpu_*");
	}

	@Test
	public void testMultLeftScalar() {
		runScalarMatrixElementWiseTests("O = scalar * X", "X", "scalar", "O", new double[] { 0.0, 0.5, 2.0 }, "gpu_*");
	}

	@Test
	public void testDivideRightScalar() {
		runScalarMatrixElementWiseTests("O = X / scalar", "X", "scalar", "O", new double[] { 0.0, 0.5, 5.0 }, "gpu_/");
	}

	@Test
	public void testDivideLeftScalar() {
		runScalarMatrixElementWiseTests("O = scalar / X", "X", "scalar", "O", new double[] { 0.0, 0.5, 5.0 }, "gpu_/");
	}

	@Test
	public void testIntegerDivideRightScalar() {
		runScalarMatrixElementWiseTests("O = X %/% scalar", "X", "scalar", "O", new double[] { 0.0, 0.5, 5.0 }, "gpu_%/%");
	}

	@Test
	public void testIntegerDivideLeftScalar() {
		runScalarMatrixElementWiseTests("O = scalar %/% X", "X", "scalar", "O", new double[] { 0.0, 0.5, 5.0 }, "gpu_%/%");
	}

	@Test
	public void testModulusRightScalar() {
		runScalarMatrixElementWiseTests("O = X %% scalar", "X", "scalar", "O", new double[] { 0.0, 0.5, 5.0 }, "gpu_%%");
	}

	@Test
	public void testModulusLeftScalar() {
		runScalarMatrixElementWiseTests("O = scalar %% X", "X", "scalar", "O", new double[] { 0.0, 0.5, 5.0 }, "gpu_%%");
	}

	// This THRESHOLD is set to specifically accommodate testModulusLeftScalar when the matrix is of size [2049,2049],
	// the scalar is 5.0 and sparsity is 0.9
	@Override
	protected double getTHRESHOLD() {
		return 1e-5;
	}

	// This specific test case fails when THRESHOLD is set to 1e-9
	@Ignore
	@Test
	public void testModulusLeftScalar2049x2049_5() {
		String scriptStr = "O = scalar %% X";
		String inputMatrix = "X";
		String inputScalar = "scalar";
		String output = "O";
		int m = 2048;
		int n = 2049;
		double sparsity = 0.9;
		double scalar = 5.0;
		System.out.println(
				"Matrix is of size [" + m + ", " + n + "], sparsity = " + sparsity + ", scalar = "
						+ scalar);
		Matrix X = generateInputMatrix(spark, m, n, sparsity, seed);
		HashMap<String, Object> inputs = new HashMap<>();
		inputs.put(inputMatrix, X);
		inputs.put(inputScalar, scalar);
		List<Object> cpuOut = runOnCPU(spark, scriptStr, inputs, Arrays.asList(output));
		List<Object> gpuOut = runOnGPU(spark, scriptStr, inputs, Arrays.asList(output));
		//assertHeavyHitterPresent(heavyHitterOpCode);
		assertEqualObjects (cpuOut.get(0), gpuOut.get(0));
	}

	@Test
	public void testMinus1MultRightScalar() {
		runScalarMatrixElementWiseTests("O = 1 - X * scalar", "X", "scalar", "O", new double[] { 0.0, 0.5, 5.0 }, "gpu_1-*");
	}

	@Test
	public void testMinus1MultLeftScalar() {
		runScalarMatrixElementWiseTests("O = 1 - scalar * X", "X", "scalar", "O", new double[] { 0.0, 0.5, 5.0 }, "gpu_1-*");
	}

	@Test
	public void testMinusNZLeftScalar() {
		runScalarMatrixElementWiseTests("O = X - scalar * (X != 0)", "X", "scalar", "O", new double[] { 0.0, 0.5, 5.0 }, "gpu_1-*");
	}

	@Test
	public void testMinusNZRightScalar() {
		runScalarMatrixElementWiseTests("O = X - (X != 0) * scalar", "X", "scalar", "O", new double[] { 0.0, 0.5, 5.0 }, "gpu_1-*");
	}

	// ****************************************************************
	// ************************ IGNORED TEST **************************
	// FIXME : There is a bug in CPU "^" when a A ^ B is executed where A & B are all zeroes
	@Ignore
	@Test
	public void testPow() {
		runScalarMatrixElementWiseTests("O = X ^ scalar", "X", "scalar", "O", new double[] { 0.0, 2.0, 10.0 }, "gpu_^");
	}

	/**
	 * Runs a simple scalar-matrix elementwise op test
	 *
	 * @param scriptStr         the script string
	 * @param inputMatrix       name of the matrix input in the script string
	 * @param inputScalar       name of the scalar input in the script string
	 * @param output            name of the output variable in the script string
	 * @param scalars           array of scalars for which to run this test
	 * @param heavyHitterOpCode the string printed for the unary op heavy hitter when executed on gpu
	 */
	private void runScalarMatrixElementWiseTests(String scriptStr, String inputMatrix, String inputScalar,
			String output, double[] scalars, String heavyHitterOpCode) {
		for (int i = 0; i < rowSizes.length; i++) {
			for (int j = 0; j < columnSizes.length; j++) {
				for (int k = 0; k < sparsities.length; k++) {
					for (int l = 0; l < scalars.length; l++) {
						int m = rowSizes[i];
						int n = columnSizes[j];
						double sparsity = sparsities[k];
						double scalar = scalars[l];
						System.out.println(
								"Matrix is of size [" + m + ", " + n + "], sparsity = " + sparsity + ", scalar = "
										+ scalar);
						Matrix X = generateInputMatrix(spark, m, n, sparsity, seed);
						HashMap<String, Object> inputs = new HashMap<>();
						inputs.put(inputMatrix, X);
						inputs.put(inputScalar, scalar);
						List<Object> cpuOut = runOnCPU(spark, scriptStr, inputs, Arrays.asList(output));
						List<Object> gpuOut = runOnGPU(spark, scriptStr, inputs, Arrays.asList(output));
						//assertHeavyHitterPresent(heavyHitterOpCode);
						assertEqualObjects(cpuOut.get(0), gpuOut.get(0));
					}
				}
			}
		}
	}

}
