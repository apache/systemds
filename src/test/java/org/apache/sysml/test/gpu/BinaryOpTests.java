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
 * Tests builtin binary ops on GPU
 */
public class BinaryOpTests extends GPUTests {

	private final static String TEST_NAME = "BinaryOpTests";
	private final int seed = 42;

	@Override
	public void setUp() {
		TestUtils.clearAssertionInformation();
		addTestConfiguration(TEST_DIR, TEST_NAME);
		getAndLoadTestConfiguration(TEST_NAME);
	}

	@Test
	public void testSolve() {
		// Test Ax = b
		// Dimensions of A = m * n
		// Dimensions of x = n * 1
		// Dimensions of b = m * 1

		double sparsity = 1.0; // Only dense matrices supported by "solve"
		final int[] sides = { 32, 33, 128, 256, 513, 2049 };
		for (int i = 0; i < sides.length; i++) {
			for (int j = i; j < sides.length; j++) {
				int m = sides[j];
				int n = sides[i];
				runSolveTest(sparsity, m, n);
			}
		}

	}

	/**
	 * Runs the test for solve (Ax = b) for input with given dimensions and sparsities
	 * A can be overdetermined (rows in A > columns in A)
	 *
	 * @param sparsity sparsity for the block A and b
	 * @param m        rows in A
	 * @param n        columns in A
	 */
	protected void runSolveTest(double sparsity, int m, int n) {
		String scriptStr = "x = solve(A, b)";
		System.out.println("In solve, A[" + m + ", " + n + "], b[" + m + ", 1]");
		Matrix A = generateInputMatrix(spark, m, n, sparsity, seed);
		Matrix b = generateInputMatrix(spark, m, 1, sparsity, seed);
		HashMap<String, Object> inputs = new HashMap<>();
		inputs.put("A", A);
		inputs.put("b", b);
		List<Object> outCPU = runOnCPU(spark, scriptStr, inputs, Arrays.asList("x"));
		List<Object> outGPU = runOnGPU(spark, scriptStr, inputs, Arrays.asList("x"));
		assertHeavyHitterPresent("gpu_solve");
		assertEqualObjects(outCPU.get(0), outGPU.get(0));
	}
}
