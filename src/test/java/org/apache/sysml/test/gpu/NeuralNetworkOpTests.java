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
import org.apache.sysml.runtime.DMLRuntimeException;
import org.apache.sysml.runtime.instructions.gpu.context.GPUContext;
import org.apache.sysml.runtime.instructions.gpu.context.GPUContextPool;
import org.apache.sysml.runtime.util.ConvolutionUtils;
import org.apache.sysml.test.utils.TestUtils;
import org.junit.Assert;
import org.junit.Ignore;
import org.junit.Test;

/**
 * Test neural network operations on the GPU
 * Because of the large number of cases that each test deals with, this class takes
 * very long to run. (It took about 9 hours to run the testMaxPoolBackward() to completion.
 * The recommended course of action before a release is
 * 1. Remove the @Ignore annotations
 * 2. Run just these test on a machine with CUDA 8 installed.
 * Only this class can be run like so:
 * <code>
 * mvn -Dit.test=org.apache.sysml.test.gpu.NeuralNetworkOpTests verify -PgpuTests
 * </code>
 */
public class NeuralNetworkOpTests extends GPUTests {

	private final static String TEST_NAME = "NeuralNetworkOpTests";
	// The MAX_OP_SIZE is to take into consideration the memory available on the GPU as well as
	// limits set by cudnn (operands need to be less than 2GB)
	private static final double MAX_OP_SIZE;

	static {
		double MAX = 0.5 * 1024 * 1024 * 1024; // 0.5 GB (this HAS to be less than 2GB)
		try {
			// Cap the maximum allowed operand size to 1/3rd of the usable GPU memory or MAX, whichever is lesser
			List<GPUContext> gCtxs = GPUContextPool.reserveAllGPUContexts();
			long availableMemory = gCtxs.get(0).getAvailableMemory();
			double averageMemoryPerOperand = availableMemory / 3.0;
			MAX_OP_SIZE = Math.min(averageMemoryPerOperand, MAX);
			GPUContextPool.freeAllGPUContexts();
		} catch (DMLRuntimeException e) {
			throw new RuntimeException(e);
		}

	}

	private final int seed = 42;

	// More comprehensive but time consuming tests
	/*
	private final List<Integer> Nlst = Arrays.asList(128, 64, 32);
    private final List<Integer> Clst = Arrays.asList(30, 20, 3);
    private final List<Integer> Hlst = Arrays.asList(400, 128, 32);
    private final List<Integer> Wlst = Arrays.asList(400, 128, 32);
    private final List<Integer> Klst = Arrays.asList(30, 20, 10);
    private final List<Integer> Rlst = Arrays.asList(128, 63, 4);
    private final List<Integer> Slst = Arrays.asList(128, 63, 4);
    private final List<Integer> strideHeightLst = Arrays.asList(9, 3);
    private final List<Integer> strideWidthLst = Arrays.asList(9, 3);
    private final List<Integer> padHeightLst = Arrays.asList(3, 1);
    private final List<Integer> padWidthLst = Arrays.asList(3, 1);
    private final List<Double> sparsitylst = Arrays.asList(1.0);    // Only test for dense
    */
	private final List<Integer> Nlst = Arrays.asList(128, 64);
	private final List<Integer> Clst = Arrays.asList(30, 3);
	private final List<Integer> Hlst = Arrays.asList(256, 64);
	private final List<Integer> Wlst = Arrays.asList(256, 64);
	private final List<Integer> Klst = Arrays.asList(30, 20);
	private final List<Integer> Rlst = Arrays.asList(128, 3);
	private final List<Integer> Slst = Arrays.asList(128, 3);
	private final List<Integer> strideHeightLst = Arrays.asList(9, 1);
	private final List<Integer> strideWidthLst = Arrays.asList(9, 1);
	private final List<Integer> padHeightLst = Arrays.asList(3, 1);
	private final List<Integer> padWidthLst = Arrays.asList(3, 1);
	private final List<Double> sparsitylst = Arrays.asList(1.0);    // Only test for dense

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

	@Ignore
	@Test
	public void testConv2d() {
		String scriptStr = "O = conv2d(image, filter, padding=[padH, padW], stride=[strideH, strideW], input_shape=[N,C,H,W], filter_shape=[K,C,R,S])";

		for (long N : Nlst) {
			for (long C : Clst) {
				for (long H : Hlst) {
					for (long W : Wlst) {
						for (long K : Klst) {
							for (long R : Rlst) {
								for (long S : Slst) {
									for (long strideH : strideHeightLst) {
										for (long strideW : strideWidthLst) {
											for (long padH : padHeightLst) {
												for (long padW : padWidthLst) {
													for (double sparsity : sparsitylst) {

														// Make sure ops fit in GPU memory and within constraints of cudnn
														long imageSize = N * C * H * W * 8l;
														if (imageSize > MAX_OP_SIZE)  // image size
															continue;
														long filterSize = K * C * R * S * 8l;
														if (filterSize > MAX_OP_SIZE)  // filter size
															continue;
														// filter is smaller than image + padding
														if (R > (H + padH) || S > (W + padW))
															continue;

														int P = (int) ConvolutionUtils.getP(H, R, strideH, padH);
														int Q = (int) ConvolutionUtils.getQ(W, S, strideW, padW);

														long doutSize = N * K * P * Q * 8l;
														if (doutSize > MAX_OP_SIZE) // dout/output size
															continue;

														double imageSizeInMB = imageSize / (1024.0 * 1024.0);
														double filterSizeInMB = filterSize / (1024.0 * 1024.0);
														double doutSizeInMB = doutSize / (1024.0 * 1024.0);
														System.out
																.format("conv2d, image[%d,%d,%d,%d](%.1fMB), filter[%d,%d,%d,%d](%.1f), dout[%d,%d,%d,%d](%.1fMB), stride[%d,%d], padding[%d,%d]",
																		N, C, H, W, imageSizeInMB, N, C, R, S,
																		filterSizeInMB, N, K, P, Q, doutSizeInMB,
																		strideH, strideW, padH, padW);
														Matrix image = generateInputMatrix(spark, (int) N,
																(int) (C * H * W), -127, 127, sparsity, seed);
														Matrix filter = generateInputMatrix(spark, (int) K,
																(int) (C * R * S), -127, 127, sparsity, seed);
														HashMap<String, Object> inputs = new HashMap<>();
														inputs.put("N", N);
														inputs.put("C", C);
														inputs.put("H", H);
														inputs.put("W", W);
														inputs.put("K", K);
														inputs.put("R", R);
														inputs.put("S", S);
														inputs.put("strideH", strideH);
														inputs.put("strideW", strideW);
														inputs.put("padH", padH);
														inputs.put("padW", padW);
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

	@Ignore
	@Test
	/**
	 * Ignored test to iron out issues
	 */ public void testConv2dOneCase() {
		String scriptStr = "O = conv2d(image, filter, padding=[padH, padW], stride=[strideH, strideW], input_shape=[N,C,H,W], filter_shape=[K,C,R,S]); print(toString(O, decimal=10, rows=1000, cols=1000))";

		long N = 32;
		long C = 3;
		long H = 128;
		long W = 128;

		long K = 30;
		long R = 64;
		long S = 64;

		long padH = 9;
		long padW = 9;
		long strideH = 3;
		long strideW = 3;

		double sparsity = 1.0;

		// Make sure ops fit in GPU memory and within constraints of cudnn
		long imageSize = N * C * H * W * 8l;
		if (imageSize > MAX_OP_SIZE)  // image size
			Assert.fail();
		long filterSize = K * C * R * S * 8l;
		if (filterSize > MAX_OP_SIZE)  // filter size
			Assert.fail();
		// filter is smaller than image + padding
		if (R > (H + padH) || S > (W + padW))
			Assert.fail();

		int P = (int) ConvolutionUtils.getP(H, R, strideH, padH);
		int Q = (int) ConvolutionUtils.getQ(W, S, strideW, padW);

		long doutSize = N * K * P * Q * 8l;
		if (doutSize > MAX_OP_SIZE) // dout/output size
			Assert.fail();

		double imageSizeInMB = imageSize / (1024.0 * 1024.0);
		double filterSizeInMB = filterSize / (1024.0 * 1024.0);
		double doutSizeInMB = doutSize / (1024.0 * 1024.0);
		System.out
				.format("conv2d, image[%d,%d,%d,%d](%.1fMB), filter[%d,%d,%d,%d](%.1f), dout[%d,%d,%d,%d](%.1fMB), stride[%d,%d], padding[%d,%d]",
						N, C, H, W, imageSizeInMB, N, C, R, S, filterSizeInMB, N, K, P, Q, doutSizeInMB, strideH,
						strideW, padH, padW);
		Matrix image = generateInputMatrix(spark, (int) N, (int) (C * H * W), -1, 1, sparsity, seed);
		Matrix filter = generateInputMatrix(spark, (int) K, (int) (C * R * S), -1, 1.0, sparsity, seed);
		HashMap<String, Object> inputs = new HashMap<>();
		inputs.put("N", N);
		inputs.put("C", C);
		inputs.put("H", H);
		inputs.put("W", W);
		inputs.put("K", K);
		inputs.put("R", R);
		inputs.put("S", S);
		inputs.put("strideH", strideH);
		inputs.put("strideW", strideW);
		inputs.put("padH", padH);
		inputs.put("padW", padW);
		inputs.put("image", image);
		inputs.put("filter", filter);
		List<Object> outCPU = runOnCPU(spark, scriptStr, inputs, Arrays.asList("O"));
		List<Object> outGPU = runOnGPU(spark, scriptStr, inputs, Arrays.asList("O"));
		assertHeavyHitterPresent("gpu_conv2d");
		assertEqualObjects(outCPU.get(0), outGPU.get(0));
		clearGPUMemory();
	}

	@Ignore
	@Test
	public void testConv2dBackwardFilter() {
		String scriptStr = "O = conv2d_backward_filter(image, dout, padding=[padH, padW], stride=[strideH, strideW], input_shape=[N,C,H,W], filter_shape=[K,C,R,S])";

		for (long N : Nlst) {
			for (long C : Clst) {
				for (long H : Hlst) {
					for (long W : Wlst) {
						for (long K : Klst) {
							for (long R : Rlst) {
								for (long S : Slst) {
									for (long strideH : strideHeightLst) {
										for (long strideW : strideWidthLst) {
											for (long padH : padHeightLst) {
												for (long padW : padWidthLst) {
													for (double sparsity : sparsitylst) {

														// filter is smaller than image + padding
														if (R > (H + padH) || S > (W + padW))
															continue;

														// Make sure ops fit in GPU memory and within constraints of cudnn
														long imageSize = N * C * H * W * 8l;
														if (imageSize > MAX_OP_SIZE)  // image size
															continue;
														long filterSize = K * C * R * S * 8l;
														if (filterSize > MAX_OP_SIZE)  // filter size
															continue;

														int P = (int) ConvolutionUtils.getP(H, R, strideH, padH);
														int Q = (int) ConvolutionUtils.getQ(W, S, strideW, padW);

														long doutSize = N * K * P * Q * 8l;
														if (doutSize > MAX_OP_SIZE) // dout/output size
															continue;

														double imageSizeInMB = imageSize / (1024.0 * 1024.0);
														double filterSizeInMB = filterSize / (1024.0 * 1024.0);
														double doutSizeInMB = doutSize / (1024.0 * 1024.0);
														System.out
																.format("conv2d_backward_filter, image[%d,%d,%d,%d](%.1fMB), filter[%d,%d,%d,%d](%.1f), dout[%d,%d,%d,%d](%.1fMB), stride[%d,%d], padding[%d,%d]",
																		N, C, H, W, imageSizeInMB, N, C, R, S,
																		filterSizeInMB, N, K, P, Q, doutSizeInMB,
																		strideH, strideW, padH, padW);
														Matrix image = generateInputMatrix(spark, (int) N,
																(int) (C * H * W), -127.0, 127, sparsity, seed);
														Matrix dout = generateInputMatrix(spark, (int) N,
																(int) (K * P * Q), -127.0, 127, sparsity, seed);
														HashMap<String, Object> inputs = new HashMap<>();
														inputs.put("N", N);
														inputs.put("C", C);
														inputs.put("H", H);
														inputs.put("W", W);
														inputs.put("K", K);
														inputs.put("R", R);
														inputs.put("S", S);
														inputs.put("strideH", strideH);
														inputs.put("strideW", strideW);
														inputs.put("padH", padH);
														inputs.put("padW", padW);
														inputs.put("image", image);
														inputs.put("dout", dout);
														List<Object> outCPU = runOnCPU(spark, scriptStr, inputs,
																Arrays.asList("O"));
														List<Object> outGPU = runOnGPU(spark, scriptStr, inputs,
																Arrays.asList("O"));
														assertHeavyHitterPresent("gpu_conv2d_backward_filter");
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

	@Ignore
	@Test
	public void testConv2dBackwardData() {
		String scriptStr = "O = conv2d_backward_data(filter, dout, padding=[padH, padW], stride=[strideH, strideW], input_shape=[N,C,H,W], filter_shape=[K,C,R,S])";

		for (long N : Nlst) {
			for (long C : Clst) {
				for (long H : Hlst) {
					for (long W : Wlst) {
						for (long K : Klst) {
							for (long R : Rlst) {
								for (long S : Slst) {
									for (long strideH : strideHeightLst) {
										for (long strideW : strideWidthLst) {
											for (long padH : padHeightLst) {
												for (long padW : padWidthLst) {
													for (double sparsity : sparsitylst) {

														// filter is smaller than image + padding
														if (R > (H + padH) || S > (W + padW))
															continue;

														// Make sure ops fit in GPU memory and within constraints of cudnn
														long imageSize = N * C * H * W * 8l;
														if (imageSize > MAX_OP_SIZE)  // image size
															continue;
														long filterSize = K * C * R * S * 8l;
														if (filterSize > MAX_OP_SIZE)  // filter size
															continue;

														int P = (int) ConvolutionUtils.getP(H, R, strideH, padH);
														int Q = (int) ConvolutionUtils.getQ(W, S, strideW, padW);

														long doutSize = N * K * P * Q * 8l;
														if (doutSize > MAX_OP_SIZE) // dout/output size
															continue;

														double imageSizeInMB = imageSize / (1024.0 * 1024.0);
														double filterSizeInMB = filterSize / (1024.0 * 1024.0);
														double doutSizeInMB = doutSize / (1024.0 * 1024.0);
														System.out
																.format("conv2d_backward_data, image[%d,%d,%d,%d](%.1fMB), filter[%d,%d,%d,%d](%.1f), dout[%d,%d,%d,%d](%.1fMB), stride[%d,%d], padding[%d,%d]",
																		N, C, H, W, imageSizeInMB, N, C, R, S,
																		filterSizeInMB, N, K, P, Q, doutSizeInMB,
																		strideH, strideW, padH, padW);

														Matrix filter = generateInputMatrix(spark, (int) K,
																(int) (C * R * S), -127.0, 127, sparsity, seed);
														Matrix dout = generateInputMatrix(spark, (int) N,
																(int) (K * P * Q), -127.0, 127, sparsity, seed);
														HashMap<String, Object> inputs = new HashMap<>();
														inputs.put("N", N);
														inputs.put("C", C);
														inputs.put("H", H);
														inputs.put("W", W);
														inputs.put("K", K);
														inputs.put("R", R);
														inputs.put("S", S);
														inputs.put("strideH", strideH);
														inputs.put("strideW", strideW);
														inputs.put("padH", padH);
														inputs.put("padW", padW);
														inputs.put("filter", filter);
														inputs.put("dout", dout);
														List<Object> outCPU = runOnCPU(spark, scriptStr, inputs,
																Arrays.asList("O"));
														List<Object> outGPU = runOnGPU(spark, scriptStr, inputs,
																Arrays.asList("O"));
														assertHeavyHitterPresent("gpu_conv2d_backward_data");
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

	@Ignore
	@Test
	public void testMaxPool() {
		String scriptStr = "O = max_pool(image, padding=[padH, padW], stride=[strideH, strideW], input_shape=[N,C,H,W], pool_size=[R,S])";

		for (long N : Nlst) {
			for (long C : Clst) {
				for (long H : Hlst) {
					for (long W : Wlst) {
						for (long R : Rlst) {
							for (long S : Slst) {
								for (long strideH : strideHeightLst) {
									for (long strideW : strideWidthLst) {
										for (long padH : padHeightLst) {
											for (long padW : padWidthLst) {
												for (double sparsity : sparsitylst) {

													// pool is smaller than image + padding
													if (R > (H + padH) || S > (W + padW))
														continue;

													// Make sure ops fit in GPU memory and within constraints of cudnn
													long imageSize = N * C * H * W * 8l;
													if (imageSize > MAX_OP_SIZE)  // image size
														continue;
													long poolSize = R * S * 8l;
													if (poolSize > MAX_OP_SIZE)  // filter size
														continue;

													int P = (int) ConvolutionUtils.getP(H, R, strideH, padH);
													int Q = (int) ConvolutionUtils.getQ(W, S, strideW, padW);

													long doutSize = N * C * P * Q * 8l;
													if (doutSize > MAX_OP_SIZE) // dout/output size
														continue;

													double imageSizeInMB = imageSize / (1024.0 * 1024.0);
													double poolSizeInMB = poolSize / (1024.0 * 1024.0);
													double doutSizeInMB = doutSize / (1024.0 * 1024.0);
													System.out
															.format("max_pool, image[%d,%d,%d,%d](%.1fMB), pool[%d,%d](%.1f), dout[%d,%d,%d,%d](%.1fMB), stride[%d,%d], padding[%d,%d]",
																	N, C, H, W, imageSizeInMB, R, S, poolSizeInMB, N, C,
																	P, Q, doutSizeInMB, strideH, strideW, padH, padW);

													Matrix image = generateInputMatrix(spark, (int) N,
															(int) (C * H * W), -127.0, 127, sparsity, seed);
													HashMap<String, Object> inputs = new HashMap<>();
													inputs.put("N", N);
													inputs.put("C", C);
													inputs.put("H", H);
													inputs.put("W", W);
													inputs.put("R", R);
													inputs.put("S", S);
													inputs.put("strideH", strideH);
													inputs.put("strideW", strideW);
													inputs.put("padH", padH);
													inputs.put("padW", padW);
													inputs.put("image", image);
													List<Object> outCPU = runOnCPU(spark, scriptStr, inputs,
															Arrays.asList("O"));
													List<Object> outGPU = runOnGPU(spark, scriptStr, inputs,
															Arrays.asList("O"));
													assertHeavyHitterPresent("gpu_maxpooling");
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

	@Ignore
	@Test
	public void testMaxPoolBackward() {
		String scriptStr = "O = max_pool_backward(image, dout, padding=[padH, padW], stride=[strideH, strideW], input_shape=[N,C,H,W], pool_size=[R,S])";

		for (long N : Nlst) {
			for (long C : Clst) {
				for (long H : Hlst) {
					for (long W : Wlst) {
						for (long R : Rlst) {
							for (long S : Slst) {
								for (long strideH : strideHeightLst) {
									for (long strideW : strideWidthLst) {
										for (long padH : padHeightLst) {
											for (long padW : padWidthLst) {
												for (double sparsity : sparsitylst) {

													// pool is smaller than image + padding
													if (R > (H + padH) || S > (W + padW))
														continue;

													// Make sure ops fit in GPU memory and within constraints of cudnn
													long imageSize = N * C * H * W * 8l;
													if (imageSize > MAX_OP_SIZE)  // image size
														continue;
													long poolSize = R * S * 8l;
													if (poolSize > MAX_OP_SIZE)  // filter size
														continue;

													int P = (int) ConvolutionUtils.getP(H, R, strideH, padH);
													int Q = (int) ConvolutionUtils.getQ(W, S, strideW, padW);

													long doutSize = N * C * P * Q * 8l;
													if (doutSize > MAX_OP_SIZE) // dout/output size
														continue;

													double imageSizeInMB = imageSize / (1024.0 * 1024.0);
													double poolSizeInMB = poolSize / (1024.0 * 1024.0);
													double doutSizeInMB = doutSize / (1024.0 * 1024.0);
													System.out
															.format("max_pool_backward, image[%d,%d,%d,%d](%.1fMB), pool[%d,%d](%.1f), dout[%d,%d,%d,%d](%.1fMB), stride[%d,%d], padding[%d,%d]",
																	N, C, H, W, imageSizeInMB, R, S, poolSizeInMB, N, C,
																	P, Q, doutSizeInMB, strideH, strideW, padH, padW);

													Matrix image = generateInputMatrix(spark, (int) N,
															(int) (C * H * W), -127.0, 127, sparsity, seed);
													Matrix dout = generateInputMatrix(spark, (int) N, (int) (C * P * Q),
															-127.0, 127, sparsity, seed);
													HashMap<String, Object> inputs = new HashMap<>();
													inputs.put("N", N);
													inputs.put("C", C);
													inputs.put("H", H);
													inputs.put("W", W);
													inputs.put("R", R);
													inputs.put("S", S);
													inputs.put("strideH", strideH);
													inputs.put("strideW", strideW);
													inputs.put("padH", padH);
													inputs.put("padW", padW);
													inputs.put("image", image);
													inputs.put("dout", dout);
													List<Object> outCPU = runOnCPU(spark, scriptStr, inputs,
															Arrays.asList("O"));
													List<Object> outGPU = runOnGPU(spark, scriptStr, inputs,
															Arrays.asList("O"));
													assertHeavyHitterPresent("gpu_maxpooling_backward");
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
