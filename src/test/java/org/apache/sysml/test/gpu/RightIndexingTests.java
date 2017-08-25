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
 * This test uses the script: O = X[rl:ru,cl:cu]
 */
public class RightIndexingTests extends GPUTests {
	
	private final static String TEST_NAME = "RightIndexingTests";
	private final int [] indexes1 = new int[] {1, 5, 10, 100};
	private final int [] indexes2 = new int[] {1, 5, 10, 100};
	private final double[] sparsities = new double[] { 0.0, 0.03, 0.3, 0.9 };
	private final int seed = 42;

	@Override
	public void setUp() {
		TestUtils.clearAssertionInformation();
		addTestConfiguration(TEST_DIR, TEST_NAME);
		getAndLoadTestConfiguration(TEST_NAME);
	}

	@Test
	public void runRightIndexingTest() {
		int dim1 = Arrays.stream(indexes1).max().getAsInt();
		int dim2 = Arrays.stream(indexes2).max().getAsInt();
		for(int i1 = 0; i1 < indexes1.length; i1++) {
			for(int i2 = i1; i2 < indexes1.length; i2++) {
				for(int j1 = 0; j1 < indexes2.length; j1++) {
					for(int j2 = j1; j2 < indexes2.length; j2++) {
						int rl = indexes1[i1]; int ru = indexes1[i2];
						int cl = indexes2[j1]; int cu = indexes2[j2];
						for (int k = 0; k < sparsities.length; k++) {
							double sparsity = sparsities[k];
							Matrix X = generateInputMatrix(spark, dim1, dim2, sparsity, seed);
							//FIXME Matrix Y = generateInputMatrix(spark, dim1, dim2, sparsity, seed);
							HashMap<String, Object> inputs = new HashMap<>();
							inputs.put("X", X);
							String scriptStr = "O = X[" + rl + ":" + ru + "," +  cl + ":" + cu + "];";
							System.out.println("Executing the script: " + scriptStr);
							List<Object> cpuOut = runOnCPU(spark, scriptStr, inputs, Arrays.asList("O"));
							List<Object> gpuOut = runOnGPU(spark, scriptStr, inputs, Arrays.asList("O"));
							assertEqualObjects(cpuOut.get(0), gpuOut.get(0));
						}
					}
				}
			}
		}
	}
}
