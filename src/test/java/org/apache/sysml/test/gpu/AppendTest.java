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
 * Tests rbind & cbind
 */
public class AppendTest extends GPUTests {

	private final static String TEST_NAME = "BinaryOpTests";
	private final int seed = 42;

	private final int[] rowSizes = new int[] { 1, 64, 2049 };
	private final int[] columnSizes = new int[] { 1, 64, 2049 };
	private final double[] sparsities = new double[] { 0.0, 0.3, 0.9 };

	@Override
	public void setUp() {
		TestUtils.clearAssertionInformation();
		addTestConfiguration(TEST_DIR, TEST_NAME);
		getAndLoadTestConfiguration(TEST_NAME);
	}

	@Test
	public void testRbind() {
		String scriptStr = "C = rbind(A, B)";

		for (int i = 0; i < rowSizes.length; i++) {
			for (int j = 0; j < columnSizes.length; j++) {
				for (int k = 0; k < rowSizes.length; k++) {
					for (int s = 0; s < sparsities.length; s++) {
						int m1 = rowSizes[i];
						int n1 = columnSizes[j];
						int m2 = rowSizes[k];
						int n2 = n1; // for rbind number of columns in both matrices need to be the same
						double sparsity = sparsities[s];

						System.out.println("In rbind, A[" + m1 + ", " + n1 + "], B[" + m2 + "," + n2 + "]");
						Matrix A = generateInputMatrix(spark, m1, n1, sparsity, seed);
						Matrix B = generateInputMatrix(spark, m2, n2, sparsity, seed);
						HashMap<String, Object> inputs = new HashMap<>();
						inputs.put("A", A);
						inputs.put("B", B);
						List<Object> outCPU = runOnCPU(spark, scriptStr, inputs, Arrays.asList("C"));
						List<Object> outGPU = runOnGPU(spark, scriptStr, inputs, Arrays.asList("C"));
						assertHeavyHitterPresent("gpu_append");
						assertEqualObjects(outCPU.get(0), outGPU.get(0));
					}
				}
			}
		}
	}

	@Test
	public void testCbind() {
		String scriptStr = "C = cbind(A, B)";

		for (int i = 0; i < rowSizes.length; i++) {
			for (int j = 0; j < columnSizes.length; j++) {
				for (int k = 0; k < columnSizes.length; k++) {
					for (int s = 0; s < sparsities.length; s++) {
						int m1 = rowSizes[i];
						int n1 = columnSizes[j];
						int m2 = m1; // for cbind number of rows in both matrices need to be the same
						int n2 = columnSizes[k];
						double sparsity = sparsities[s];

						System.out.println("In cbind, A[" + m1 + ", " + n1 + "], B[" + m2 + "," + n2 + "]");
						Matrix A = generateInputMatrix(spark, m1, n1, sparsity, seed);
						Matrix B = generateInputMatrix(spark, m2, n2, sparsity, seed);
						HashMap<String, Object> inputs = new HashMap<>();
						inputs.put("A", A);
						inputs.put("B", B);
						List<Object> outCPU = runOnCPU(spark, scriptStr, inputs, Arrays.asList("C"));
						List<Object> outGPU = runOnGPU(spark, scriptStr, inputs, Arrays.asList("C"));
						assertHeavyHitterPresent("gpu_append");
						assertEqualObjects(outCPU.get(0), outGPU.get(0));
					}
				}
			}
		}
	}
}
