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
import org.apache.sysml.runtime.util.ConvolutionUtils;
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
    private final double MAX_OP_SIZE = 1.8 * 1024 * 1024 * 1024; // 0.25 GB

    private final List<Integer> Nlst = Arrays.asList(128, 64, 32);
    private final List<Integer> Clst = Arrays.asList(40, 20, 3);
    private final List<Integer> Hlst = Arrays.asList(1024, 512, 32);
    private final List<Integer> Wlst = Arrays.asList(1024, 512, 32);
    private final List<Integer> Klst = Arrays.asList(40, 20, 10);
    private final List<Integer> Rlst = Arrays.asList(32, 8);
    private final List<Integer> Slst = Arrays.asList(32, 8);
    private final List<Integer> strideXlst = Arrays.asList(9, 2);
    private final List<Integer> strideYlst = Arrays.asList(9, 2);
    private final List<Integer> padXlst = Arrays.asList(4, 1);
    private final List<Integer> padYlst = Arrays.asList(4, 1);
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

    @Test
    public void testConv2d() {
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
                                                        // filter is smaller than image + padding
                                                        if (R > (H + padX) || S > (W + padY))
                                                            continue;

                                                        int P = (int) ConvolutionUtils.getP(H, R, strideY, padY);
                                                        int Q = (int) ConvolutionUtils.getQ(W, S, strideX, padX);

                                                        long doutSize = N * K * P * Q * 8l;
                                                        if (doutSize > MAX_OP_SIZE) // dout/output size
                                                            continue;

                                                        double imageSizeInGB = imageSize / (1024.0 * 1024.0);
                                                        double filterSizeInGB = filterSize / (1024.0 * 1024.0);
                                                        double doutSizeInGB = doutSize / (1024.0 * 1024.0);
                                                        System.out.format(
                                                            "conv2d_backward_filter, image[%d,%d,%d,%d](%.1fMB), filter[%d,%d,%d,%d](%.1f), dout[%d,%d,%d,%d](%.1fMB), stride[%d,%d], padding[%d,%d]",
                                                            N, C, H, W, imageSizeInGB, N, C, R, S, filterSizeInGB, N, K,
                                                            P, Q, doutSizeInGB, strideX, strideY, padX, padY);
                                                        Matrix image = generateInputMatrix(spark, N, C * H * W,
                                                            sparsity, seed);
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

    @Test
    public void testConv2dBackwardFilter() {
        String scriptStr = "O = conv2d_backward_filter(image, dout, padding=[padX, padY], stride=[strideX, strideY], input_shape=[N,C,H,W], filter_shape=[K,C,R,S])";

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

                                                        // filter is smaller than image + padding
                                                        if (R > (H + padX) || S > (W + padY))
                                                            continue;

                                                        // Make sure ops fit in GPU memory and within constraints of cudnn
                                                        long imageSize = N * C * H * W * 8l;
                                                        if (imageSize > MAX_OP_SIZE)  // image size
                                                            continue;
                                                        long filterSize = K * C * R * S * 8l;
                                                        if (filterSize > MAX_OP_SIZE)  // filter size
                                                            continue;

                                                        int P = (int) ConvolutionUtils.getP(H, R, strideY, padY);
                                                        int Q = (int) ConvolutionUtils.getQ(W, S, strideX, padX);

                                                        long doutSize = N * K * P * Q * 8l;
                                                        if (doutSize > MAX_OP_SIZE) // dout/output size
                                                            continue;

                                                        double imageSizeInGB = imageSize / (1024.0 * 1024.0);
                                                        double filterSizeInGB = filterSize / (1024.0 * 1024.0);
                                                        double doutSizeInGB = doutSize / (1024.0 * 1024.0);
                                                        System.out.format(
                                                            "conv2d_backward_filter, image[%d,%d,%d,%d](%.1fMB), filter[%d,%d,%d,%d](%.1f), dout[%d,%d,%d,%d](%.1fMB), stride[%d,%d], padding[%d,%d]",
                                                            N, C, H, W, imageSizeInGB, N, C, R, S, filterSizeInGB, N, K,
                                                            P, Q, doutSizeInGB, strideX, strideY, padX, padY);
                                                        Matrix image = generateInputMatrix(spark, N, C * H * W,
                                                            sparsity, seed);
                                                        Matrix dout = generateInputMatrix(spark, N, K * P * Q, sparsity,
                                                            seed);
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

    @Test
    public void testConv2dBackwardData() {
        String scriptStr = "O = conv2d_backward_data(filter, dout, padding=[padX, padY], stride=[strideX, strideY], input_shape=[N,C,H,W], filter_shape=[K,C,R,S])";

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

                                                        // filter is smaller than image + padding
                                                        if (R > (H + padX) || S > (W + padY))
                                                            continue;

                                                        // Make sure ops fit in GPU memory and within constraints of cudnn
                                                        long imageSize = N * C * H * W * 8l;
                                                        if (imageSize > MAX_OP_SIZE)  // image size
                                                            continue;
                                                        long filterSize = K * C * R * S * 8l;
                                                        if (filterSize > MAX_OP_SIZE)  // filter size
                                                            continue;

                                                        int P = (int) ConvolutionUtils.getP(H, R, strideY, padY);
                                                        int Q = (int) ConvolutionUtils.getQ(W, S, strideX, padX);

                                                        long doutSize = N * K * P * Q * 8l;
                                                        if (doutSize > MAX_OP_SIZE) // dout/output size
                                                            continue;

                                                        double imageSizeInGB = imageSize / (1024.0 * 1024.0);
                                                        double filterSizeInGB = filterSize / (1024.0 * 1024.0);
                                                        double doutSizeInGB = doutSize / (1024.0 * 1024.0);
                                                        System.out.format(
                                                            "conv2d_backward_data, image[%d,%d,%d,%d](%.1fMB), filter[%d,%d,%d,%d](%.1f), dout[%d,%d,%d,%d](%.1fMB), stride[%d,%d], padding[%d,%d]",
                                                            N, C, H, W, imageSizeInGB, N, C, R, S, filterSizeInGB, N, K,
                                                            P, Q, doutSizeInGB, strideX, strideY, padX, padY);

                                                        Matrix filter = generateInputMatrix(spark, N, C * R * S,
                                                            sparsity, seed);
                                                        Matrix dout = generateInputMatrix(spark, N, K * P * Q, sparsity,
                                                            seed);
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

    @Test
    public void testMaxPool() {
        String scriptStr = "O = max_pool(image, padding=[padX, padY], stride=[strideX, strideY], input_shape=[N,C,H,W], pool_size=[R,S])";

        for (int N : Nlst) {
            for (int C : Clst) {
                for (int H : Hlst) {
                    for (int W : Wlst) {
                        for (int R : Rlst) {
                            for (int S : Slst) {
                                for (int strideX : strideXlst) {
                                    for (int strideY : strideYlst) {
                                        for (int padX : padXlst) {
                                            for (int padY : padYlst) {
                                                for (double sparsity : sparsitylst) {

                                                    // pool is smaller than image + padding
                                                    if (R > (H + padX) || S > (W + padY))
                                                        continue;

                                                    // Make sure ops fit in GPU memory and within constraints of cudnn
                                                    long imageSize = N * C * H * W * 8l;
                                                    if (imageSize > MAX_OP_SIZE)  // image size
                                                        continue;
                                                    long poolSize = R * S * 8l;
                                                    if (poolSize > MAX_OP_SIZE)  // filter size
                                                        continue;

                                                    int P = (int) ConvolutionUtils.getP(H, R, strideY, padY);
                                                    int Q = (int) ConvolutionUtils.getQ(W, S, strideX, padX);

                                                    long doutSize = N * C * P * Q * 8l;
                                                    if (doutSize > MAX_OP_SIZE) // dout/output size
                                                        continue;

                                                    double imageSizeInGB = imageSize / (1024.0 * 1024.0);
                                                    double poolSizeInGB = poolSize / (1024.0 * 1024.0);
                                                    double doutSizeInGB = doutSize / (1024.0 * 1024.0);
                                                    System.out.format(
                                                        "max_pool, image[%d,%d,%d,%d](%.1fMB), pool[%d,%d](%.1f), dout[%d,%d,%d,%d](%.1fMB), stride[%d,%d], padding[%d,%d]",
                                                        N, C, H, W, imageSizeInGB, R, S, poolSizeInGB, N, C, P, Q,
                                                        doutSizeInGB, strideX, strideY, padX, padY);

                                                    Matrix image = generateInputMatrix(spark, N, C * H * W, sparsity,
                                                        seed);
                                                    HashMap<String, Object> inputs = new HashMap<>();
                                                    inputs.put("N", N);
                                                    inputs.put("C", C);
                                                    inputs.put("H", H);
                                                    inputs.put("W", W);
                                                    inputs.put("R", R);
                                                    inputs.put("S", S);
                                                    inputs.put("strideX", strideX);
                                                    inputs.put("strideY", strideY);
                                                    inputs.put("padX", padX);
                                                    inputs.put("padY", padY);
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

    @Test
    public void testMaxPoolBackward() {
        String scriptStr = "O = max_pool_backward(image, dout, padding=[padX, padY], stride=[strideX, strideY], input_shape=[N,C,H,W], pool_size=[R,S])";

        for (int N : Nlst) {
            for (int C : Clst) {
                for (int H : Hlst) {
                    for (int W : Wlst) {
                        for (int R : Rlst) {
                            for (int S : Slst) {
                                for (int strideX : strideXlst) {
                                    for (int strideY : strideYlst) {
                                        for (int padX : padXlst) {
                                            for (int padY : padYlst) {
                                                for (double sparsity : sparsitylst) {

                                                    // pool is smaller than image + padding
                                                    if (R > (H + padX) || S > (W + padY))
                                                        continue;

                                                    // Make sure ops fit in GPU memory and within constraints of cudnn
                                                    long imageSize = N * C * H * W * 8l;
                                                    if (imageSize > MAX_OP_SIZE)  // image size
                                                        continue;
                                                    long poolSize = R * S * 8l;
                                                    if (poolSize > MAX_OP_SIZE)  // filter size
                                                        continue;

                                                    int P = (int) ConvolutionUtils.getP(H, R, strideY, padY);
                                                    int Q = (int) ConvolutionUtils.getQ(W, S, strideX, padX);

                                                    long doutSize = N * C * P * Q * 8l;
                                                    if (doutSize > MAX_OP_SIZE) // dout/output size
                                                        continue;

                                                    double imageSizeInGB = imageSize / (1024.0 * 1024.0);
                                                    double poolSizeInGB = poolSize / (1024.0 * 1024.0);
                                                    double doutSizeInGB = doutSize / (1024.0 * 1024.0);
                                                    System.out.format(
                                                        "max_pool_backward, image[%d,%d,%d,%d](%.1fMB), pool[%d,%d](%.1f), dout[%d,%d,%d,%d](%.1fMB), stride[%d,%d], padding[%d,%d]",
                                                        N, C, H, W, imageSizeInGB, R, S, poolSizeInGB, N, C, P, Q,
                                                        doutSizeInGB, strideX, strideY, padX, padY);

                                                    Matrix image = generateInputMatrix(spark, N, C * H * W, sparsity,
                                                        seed);
                                                    Matrix dout = generateInputMatrix(spark, N, C * P * Q, sparsity,
                                                        seed);
                                                    HashMap<String, Object> inputs = new HashMap<>();
                                                    inputs.put("N", N);
                                                    inputs.put("C", C);
                                                    inputs.put("H", H);
                                                    inputs.put("W", W);
                                                    inputs.put("R", R);
                                                    inputs.put("S", S);
                                                    inputs.put("strideX", strideX);
                                                    inputs.put("strideY", strideY);
                                                    inputs.put("padX", padX);
                                                    inputs.put("padY", padY);
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
