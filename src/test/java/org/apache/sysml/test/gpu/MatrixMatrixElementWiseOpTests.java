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
 * Test Elementwise operations on the GPU
 */
public class MatrixMatrixElementWiseOpTests extends GPUTests {
	private final static String TEST_NAME = "MatrixMatrixElementWiseOpTests";

	private final int[] rowSizes = new int[] { 1, 64, 130, 1024, 2049 };
	private final int[] columnSizes = new int[] { 1, 64, 130, 1024, 2049 };
	private final double[] sparsities = new double[] { 0.0, 0.03, 0.3, 0.9 };
	private final double[] scalars = new double[] { 0.0, 0.5, 2.0 };
	private final int seed = 42;

	@Override
	public void setUp() {
		TestUtils.clearAssertionInformation();
		addTestConfiguration(TEST_DIR, TEST_NAME);
		getAndLoadTestConfiguration(TEST_NAME);
	}

	@Test
	public void testAxpy() {
		runAxpyTest("O = a*X + Y", "X", "Y", "a", "O", "gpu_-*");
	}

	@Test
	public void testAxmy() {
		runAxpyTest("O = X - a*Y", "X", "Y", "a", "O", "gpu_+*");
	}

	@Test
	public void testAdd() {
		runMatrixMatrixElementwiseTest("O = X + Y", "X", "Y", "O", "gpu_+");
	}

	@Test
	public void testMatrixColumnVectorAdd() {
		runMatrixColumnVectorTest("O = X + Y", "X", "Y", "O", "gpu_+");
	}

	@Test
	public void testMatrixRowVectorAdd() {
		runMatrixRowVectorTest("O = X + Y", "X", "Y", "O", "gpu_+");
	}

	@Test
	public void testSubtract() {
		runMatrixMatrixElementwiseTest("O = X - Y", "X", "Y", "O", "gpu_-");
	}

	@Test
	public void testMatrixColumnVectorSubtract() {
		runMatrixColumnVectorTest("O = X - Y", "X", "Y", "O", "gpu_-");
	}

	@Test
	public void testMatrixRowVectorSubtract() {
		runMatrixRowVectorTest("O = X - Y", "X", "Y", "O", "gpu_-");
	}

	@Test
	public void testMultiply() {
		runMatrixMatrixElementwiseTest("O = X * Y", "X", "Y", "O", "gpu_*");
	}

	@Test
	public void testMatrixColumnVectorMultiply() {
		runMatrixColumnVectorTest("O = X * Y", "X", "Y", "O", "gpu_*");
	}

	@Test
	public void testMatrixRowVectorMultiply() {
		runMatrixRowVectorTest("O = X * Y", "X", "Y", "O", "gpu_*");
	}

	@Test
	public void testDivide() {
		runMatrixMatrixElementwiseTest("O = X / Y", "X", "Y", "O", "gpu_/");
	}

	@Test
	public void testMatrixColumnVectorDivide() {
		runMatrixColumnVectorTest("O = X / Y", "X", "Y", "O", "gpu_/");
	}

	@Test
	public void testMatrixRowVectorDivide() {
		runMatrixRowVectorTest("O = X / Y", "X", "Y", "O", "gpu_/");
	}

	// ****************************************************************
	// ************************ IGNORED TEST **************************
	// FIXME : There is a bug in CPU "^" when a A ^ B is executed where A & B are all zeroes
	@Ignore
	@Test
	public void testPower() {
		runMatrixMatrixElementwiseTest("O = X ^ Y", "X", "Y", "O", "gpu_%");
	}

	/**
	 * Runs a simple matrix-matrix elementwise op test
	 *
	 * @param scriptStr         the script string
	 * @param input1            name of the first input variable in the script string
	 * @param input2            name of the second input variable in the script string
	 * @param output            name of the output variable in the script string
	 * @param heavyHitterOpcode the string printed for the unary op heavy hitter when executed on gpu
	 */
	private void runMatrixMatrixElementwiseTest(String scriptStr, String input1, String input2, String output,
			String heavyHitterOpcode) {
		for (int i = 0; i < rowSizes.length; i++) {
			for (int j = 0; j < columnSizes.length; j++) {
				for (int k = 0; k < sparsities.length; k++) {
					int m = rowSizes[i];
					int n = columnSizes[j];
					double sparsity = sparsities[k];
					Matrix X = generateInputMatrix(spark, m, n, sparsity, seed);
					Matrix Y = generateInputMatrix(spark, m, n, sparsity, seed);
					HashMap<String, Object> inputs = new HashMap<>();
					inputs.put(input1, X);
					inputs.put(input2, Y);
					List<Object> cpuOut = runOnCPU(spark, scriptStr, inputs, Arrays.asList(output));
					List<Object> gpuOut = runOnGPU(spark, scriptStr, inputs, Arrays.asList(output));
					//assertHeavyHitterPresent(heavyHitterOpcode);
					assertEqualObjects(cpuOut.get(0), gpuOut.get(0));
				}
			}
		}
	}

	/**
	 * Run O = aX +/- Y type operations test
	 *
	 * @param scriptStr         the script string
	 * @param input1            name of the first matrix input variable in the script string
	 * @param input2            name of the second matrix input variable in the script string
	 * @param scalarInput       name of the scalar which is multiplied with the first or second matrix
	 * @param output            name of the output variable in the script string
	 * @param heavyHitterOpcode the string printed for the unary op heavy hitter when executed on gpu
	 */
	private void runAxpyTest(String scriptStr, String input1, String input2, String scalarInput, String output,
			String heavyHitterOpcode) {
		for (int i = 0; i < rowSizes.length; i++) {
			for (int j = 0; j < columnSizes.length; j++) {
				for (int k = 0; k < sparsities.length; k++) {
					for (int l = 0; l < scalars.length; l++) {
						int m = rowSizes[i];
						int n = columnSizes[j];
						double scalar = scalars[l];
						double sparsity = sparsities[k];
						Matrix X = generateInputMatrix(spark, m, n, sparsity, seed);
						Matrix Y = generateInputMatrix(spark, m, n, sparsity, seed);
						HashMap<String, Object> inputs = new HashMap<>();
						inputs.put(input1, X);
						inputs.put(input2, Y);
						inputs.put(scalarInput, scalar);

						// Test O = aX + Y
						List<Object> cpuOut = runOnCPU(spark, scriptStr, inputs, Arrays.asList(output));
						List<Object> gpuOut = runOnGPU(spark, scriptStr, inputs, Arrays.asList(output));
						//assertHeavyHitterPresent(heavyHitterOpcode);
						assertEqualObjects(cpuOut.get(0), gpuOut.get(0));
					}
				}
			}
		}
	}

	/**
	 * Run O = X op Y where X is a matrix, Y is a column vector
	 *
	 * @param scriptStr         the script string
	 * @param matrixInput       name of the matrix input variable in the script string
	 * @param vectorInput       name of the vector input variable in the script string
	 * @param output            name of the output variable in the script string
	 * @param heavyHitterOpcode the string printed for the unary op heavy hitter when executed on gpu
	 */
	private void runMatrixColumnVectorTest(String scriptStr, String matrixInput, String vectorInput, String output,
			String heavyHitterOpcode) {
		int[] rows = new int[] { 64, 130, 1024, 2049 };
		int[] cols = new int[] { 64, 130, 1024, 2049 };

		for (int i = 0; i < rows.length; i++) {
			for (int j = 0; j < cols.length; j++) {
				for (int k = 0; k < sparsities.length; k++) {
					int m = rows[i];
					int n = cols[j];
					double sparsity = sparsities[k];
					Matrix X = generateInputMatrix(spark, m, n, sparsity, seed);
					Matrix Y = generateInputMatrix(spark, m, 1, sparsity, seed);
					HashMap<String, Object> inputs = new HashMap<>();
					inputs.put(matrixInput, X);
					inputs.put(vectorInput, Y);

					System.out.println("Vector[" + m + ", 1] op Matrix[" + m + ", " + n + "], sparsity = " + sparsity);
					List<Object> cpuOut = runOnCPU(spark, scriptStr, inputs, Arrays.asList(output));
					List<Object> gpuOut = runOnGPU(spark, scriptStr, inputs, Arrays.asList(output));
					//assertHeavyHitterPresent(heavyHitterOpcode);
					assertEqualObjects(cpuOut.get(0), gpuOut.get(0));

				}
			}
		}
	}

	/**
	 * Run O = X op Y where X is a matrix, Y is a row vector
	 *
	 * @param scriptStr         the script string
	 * @param matrixInput       name of the matrix input variable in the script string
	 * @param vectorInput       name of the vector input variable in the script string
	 * @param output            name of the output variable in the script string
	 * @param heavyHitterOpcode the string printed for the unary op heavy hitter when executed on gpu
	 */
	private void runMatrixRowVectorTest(String scriptStr, String matrixInput, String vectorInput, String output,
			String heavyHitterOpcode) {
		int[] rows = new int[] { 64, 130, 1024, 2049 };
		int[] cols = new int[] { 64, 130, 1024, 2049 };

		for (int i = 0; i < rows.length; i++) {
			for (int j = 0; j < cols.length; j++) {
				for (int k = 0; k < sparsities.length; k++) {
					int m = rows[i];
					int n = cols[j];
					double sparsity = sparsities[k];
					Matrix X = generateInputMatrix(spark, m, n, sparsity, seed);
					Matrix Y = generateInputMatrix(spark, 1, n, sparsity, seed);
					HashMap<String, Object> inputs = new HashMap<>();
					inputs.put(matrixInput, X);
					inputs.put(vectorInput, Y);

					System.out.println("Vector[" + m + ", 1] op Matrix[" + m + ", " + n + "], sparsity = " + sparsity);
					List<Object> cpuOut = runOnCPU(spark, scriptStr, inputs, Arrays.asList(output));
					List<Object> gpuOut = runOnGPU(spark, scriptStr, inputs, Arrays.asList(output));
					//assertHeavyHitterPresent(heavyHitterOpcode);
					assertEqualObjects(cpuOut.get(0), gpuOut.get(0));
				}
			}
		}
	}

}
