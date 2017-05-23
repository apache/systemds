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

import org.apache.sysml.api.mlcontext.Matrix;
import org.apache.sysml.test.utils.TestUtils;
import org.junit.Test;

import java.util.Arrays;
import java.util.HashMap;
import java.util.List;

/** Test Elementwise operations on the GPU */
public class ElementWiseOpTests extends GPUTests {
	private final static String TEST_NAME = "ElementWiseOpTests";

	private final int[] rowSizes = new int[] { 1, 64, 130, 1024, 2049 };
	private final int[] columnSizes = new int[] { 1, 64, 130, 1024, 2049 };
	private final double[] sparsities = new double[] { 0.0, 0.03, 0.3, 0.9 };
	private final int seed = 42;

	@Override public void setUp() {
		TestUtils.clearAssertionInformation();
		addTestConfiguration(TEST_DIR, TEST_NAME);
		getAndLoadTestConfiguration(TEST_NAME);
	}

	@Test public void testAdd() {
		String scriptStr = "O = X + Y";
		String input1 = "X";
		String input2 = "Y";
		String output = "O";
		for (int i=0; i<rowSizes.length; i++){
			for (int j=0; j<columnSizes.length; j++){
				for (int k=0; k<sparsities.length; k++){
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
					assertEqualObjects(cpuOut.get(0), gpuOut.get(0));
				}
			}
		}
	}
}
