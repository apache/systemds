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

/**
 * Test neural network operations on the GPU
 */
public class NeuralNetworkOpTests extends GPUTests {

	private final static String TEST_NAME = "NeuralNetworkOpTests";
	private final int seed = 42;
	private final double MAX_OP_SIZE = 0.1 * 1024 * 1024 * 1024; // 0.25 GB


	private final List<Integer> Nlst = Arrays.asList(128, 64, 32, 16);
	private final List<Integer> Clst = Arrays.asList(32, 10, 3, 1);
	private final List<Integer> Hlst = Arrays.asList(512, 256, 128, 64, 32);
	private final List<Integer> Wlst = Arrays.asList(512, 256, 128, 64, 32);
	private final List<Integer> Klst = Arrays.asList(40, 30, 20, 10);
	private final List<Integer> Rlst = Arrays.asList(256, 128, 64, 32, 16, 8, 4, 2);
	private final List<Integer> Slst = Arrays.asList(256, 128, 64, 32, 16, 8, 4, 2);
	private final List<Integer> strideXlst = Arrays.asList(4, 2, 1);
	private final List<Integer> strideYlst = Arrays.asList(4, 2, 1);
	private final List<Integer> padXlst = Arrays.asList(4, 2, 1);
	private final List<Integer> padYlst = Arrays.asList(4, 2, 1);
	private final List<Double> sparsitylst = Arrays.asList(1.0);    // Only test for dense


	@Override public void setUp() {
		TestUtils.clearAssertionInformation();
		addTestConfiguration(TEST_DIR, TEST_NAME);
		getAndLoadTestConfiguration(TEST_NAME);
	}

	@Override public double getTHRESHOLD() {
		return 1e-5;
	}

	@Test public void testConv2d() {
		String scriptStr = "O = conv2d(image, filter, padding=[padX, padY], stride=[strideX, strideY], input_shape=[N,C,H,W], filter_shape=[K,C,R,S])";

		for (int N : Nlst) {
			for (int C : Clst) {
				for (int H : Hlst) {
					for (int W : Wlst) {
						for (int K : Klst) {
							for (int R : Rlst) {
								for (int S : Slst) {
									for (int strideX : strideXlst) {
										for (int strideY : strideYlst) {
											for (int padX : padXlst) {
												for (int padY : padYlst) {
													for (double sparsity : sparsitylst) {

														// Make sure ops fit in GPU memory and within constraints of cudnn
														long imageSize = N * C * H * W * 8l;
														if (imageSize > MAX_OP_SIZE)  // image size
															continue;
														long filterSize = K * C * R * S * 8l;
														if (filterSize > MAX_OP_SIZE)  // filter size
															continue;
														if (R > (H + padX) || S > (W
																+ padY)) // filter is smaller than image + padding
															continue;

														double imageSizeInGB = imageSize / (1024.0 * 1024.0);
														double filterSizeInGB = filterSize / (1024.0 * 1024.0);
														System.out
																.format("conv2d, image[%d,%d,%d,%d](%.1fMB), filter[%d,%d,%d,%d](%.1fMB), stride[%d,%d], padding[%d,%d]",
																		N, C, H, W, imageSizeInGB, K, C, R, S,
																		filterSizeInGB, strideX, strideY, padX, padY);
														Matrix image = generateInputMatrix(spark, N, C * H * W, 0.3,
																seed);
														Matrix filter = generateInputMatrix(spark, K, C * R * S,
																sparsity, seed);
														HashMap<String, Object> inputs = new HashMap<>();
														inputs.put("N", N);
														inputs.put("C", C);
														inputs.put("H", H);
														inputs.put("W", W);
														inputs.put("K", K);
														inputs.put("R", R);
														inputs.put("S", S);
														inputs.put("strideX", strideX);
														inputs.put("strideY", strideY);
														inputs.put("padX", padX);
														inputs.put("padY", padY);
														inputs.put("image", image);
														inputs.put("filter", filter);
														List<Object> outCPU = runOnCPU(spark, scriptStr, inputs,
																Arrays.asList("O"));
														List<Object> outGPU = runOnGPU(spark, scriptStr, inputs,
																Arrays.asList("O"));
														assertHeavyHitterPresent("gpu_conv2d");
														assertEqualObjects(outCPU.get(0), outGPU.get(0));
														clearGPUMemory();
													}
												}
											}
										}
									}
								}
							}
						}
					}
				}
			}
		}
	}
}
