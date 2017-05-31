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
import org.junit.Test;

/**
 * Tests matrix multiplication on the GPU
 */
public class MatrixMultiplicationOpTest extends GPUTests {
	private final static String TEST_NAME = "MatrixMultiplicationOpTest";
	private final int seed = 42;

	@Override
	public void setUp() {
		TestUtils.clearAssertionInformation();
		addTestConfiguration(TEST_DIR, TEST_NAME);
		getAndLoadTestConfiguration(TEST_NAME);
	}

	@Override
	public double getTHRESHOLD() {
		return 1e-5;
	}

	@Test
	public void matrixMatrixTest1() {
		String scriptStr = "O = X %*% Y";

		int[] X1 = { 1, 128, 513, 1024 };
		int[] X2 = { 128, 512, 1024 };
		int[] Y2 = { 1, 128, 513, 1024 };
		double[] SX = { 0.0, 0.03, 0.3, 0.9 };
		double[] SY = { 0.0, 0.03, 0.3, 0.9 };

		for (int x1 = 0; x1 < X1.length; x1++) {
			for (int x2 = 0; x2 < X2.length; x2++) {
				int y1 = x2;
				for (int y2 = 0; y2 < Y2.length; y2++) {
					for (int sx = 0; sx < SX.length; sx++) {
						for (int sy = 0; sy < SY.length; sy++) {
							assertMatrixMultiplication(scriptStr, X1[x1], X2[x2], X2[y1], Y2[y2], SX[sx], SY[sy]);
						}
					}
				}
			}
		}
	}

	@Test
	public void matrixMatrixTest2() {
		String scriptStr = "O = X %*% t(Y)";

		int[] X1 = { 1, 128, 513, 1024 };
		int[] X2 = { 128, 512, 1024 };
		int[] Y1 = { 1, 128, 513, 1024 };
		double[] SX = { 0.0, 0.03, 0.3, 0.9 };
		double[] SY = { 0.0, 0.03, 0.3, 0.9 };

		for (int x1 = 0; x1 < X1.length; x1++) {
			for (int x2 = 0; x2 < X2.length; x2++) {
				int y2 = x2;
				for (int y1 = 0; y1 < Y1.length; y1++) {
					for (int sx = 0; sx < SX.length; sx++) {
						for (int sy = 0; sy < SY.length; sy++) {
							assertMatrixMultiplication(scriptStr, X1[x1], X2[x2], Y1[x2], X2[y2], SX[sx], SY[sy]);
						}
					}
				}
			}
		}
	}

	@Test
	public void matrixMatrixTest3() {
		String scriptStr = "O = t(X) %*% Y";

		int[] X1 = { 1, 128, 513, 1024 };
		int[] X2 = { 128, 512, 1024 };
		int[] Y2 = { 1, 128, 513, 1024 };
		double[] SX = { 0.0, 0.03, 0.3, 0.9 };
		double[] SY = { 0.0, 0.03, 0.3, 0.9 };

		for (int x1 = 0; x1 < X1.length; x1++) {
			int y1 = x1;
			for (int x2 = 0; x2 < X2.length; x2++) {
				for (int y2 = 0; y2 < Y2.length; y2++) {
					for (int sx = 0; sx < SX.length; sx++) {
						for (int sy = 0; sy < SY.length; sy++) {
							assertMatrixMultiplication(scriptStr, X1[x1], X2[x2], X1[y1], Y2[y2], SX[sx], SY[sy]);
						}
					}
				}
			}
		}
	}

	@Test
	public void matrixMatrixTest4() {
		String scriptStr = "O = t(X) %*% t(Y)";

		int[] X1 = { 1, 128, 513, 1024 };
		int[] X2 = { 128, 512, 1024 };
		int[] Y1 = { 1, 128, 513, 1024 };
		double[] SX = { 0.0, 0.03, 0.3, 0.9 };
		double[] SY = { 0.0, 0.03, 0.3, 0.9 };

		for (int x1 = 0; x1 < X1.length; x1++) {
			int y2 = x1;
			for (int x2 = 0; x2 < X2.length; x2++) {
				for (int y1 = 0; y1 < Y1.length; y1++) {
					for (int sx = 0; sx < SX.length; sx++) {
						for (int sy = 0; sy < SY.length; sy++) {
							assertMatrixMultiplication(scriptStr, X1[x1], X2[x2], Y1[y1], X1[y2], SX[sx], SY[sy]);
						}
					}
				}
			}
		}
	}

	@Test
	public void transposeSelfMatrixMultiply() {
		String scriptStr = "O = t(X) %*% X";

		int[] sizes = { 1, 128, 512, 1024, 2049 };
		double[] sparsities = { 0.0, 0.03, 0.3, 0.9 };

		for (int i = 0; i < sizes.length; i++) {
			for (int j = 0; j < sparsities.length; j++) {
				int side = sizes[i];
				double sparsity = sparsities[j];
				Matrix X = generateInputMatrix(spark, side, side, sparsity, seed);
				HashMap<String, Object> inputs = new HashMap<>();
				inputs.put("X", X);
				List<Object> cpuOuts = runOnCPU(spark, scriptStr, inputs, Arrays.asList("O"));
				List<Object> gpuOuts = runOnGPU(spark, scriptStr, inputs, Arrays.asList("O"));
				//assertHeavyHitterPresent("gpu_tsmm'");
				assertEqualObjects(cpuOuts.get(0), gpuOuts.get(0));
			}
		}
	}

	/**
	 * Assert that matrix multiplication is the same on gpu and cpu
	 *
	 * @param scriptStr script string that has matrix multiplication (eg : O = X %*% Y)
	 * @param rows1     rows in X
	 * @param cols1     cols in X
	 * @param rows2     rows in Y
	 * @param cols2     cols in Y
	 * @param sparsity1 sparsity for X
	 * @param sparsity2 sparsity for Y
	 */
	private void assertMatrixMultiplication(String scriptStr, int rows1, int cols1, int rows2, int cols2,
			double sparsity1, double sparsity2) {
		HashMap<String, Object> inputs = new HashMap<>();
		Matrix X = generateInputMatrix(spark, rows1, cols1, sparsity1, seed);
		Matrix Y = generateInputMatrix(spark, rows2, cols2, sparsity2, seed);
		inputs.put("X", X);
		inputs.put("Y", Y);
		List<Object> cpuOuts = runOnCPU(spark, scriptStr, inputs, Arrays.asList("O"));
		List<Object> gpuOuts = runOnGPU(spark, scriptStr, inputs, Arrays.asList("O"));
		//assertHeavyHitterPresent("gpu_ba+*'");
		assertEqualObjects(cpuOuts.get(0), gpuOuts.get(0));
	}
}
