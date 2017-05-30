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
import org.apache.sysml.runtime.DMLRuntimeException;
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
    private final double MAX_OP_SIZE = 1.5 * 1024 * 1024 * 1024; // 1 GB

    /*
    private final List<Integer> Nlst = Arrays.asList(64, 32, 16);
    private final List<Integer> Clst = Arrays.asList(30, 10, 3);
    private final List<Integer> Hlst = Arrays.asList(400, 200, 32);
    private final List<Integer> Wlst = Arrays.asList(400, 200, 32);
    private final List<Integer> Klst = Arrays.asList(30, 20, 10);
    private final List<Integer> Rlst = Arrays.asList(127, 61, 10);
    private final List<Integer> Slst = Arrays.asList(127, 61, 10);
    private final List<Integer> strideXlst = Arrays.asList(9, 3);
    private final List<Integer> strideYlst = Arrays.asList(9, 3);
    private final List<Integer> padXlst = Arrays.asList(5, 1);
    private final List<Integer> padYlst = Arrays.asList(5, 1);
    private final List<Double> sparsitylst = Arrays.asList(1.0);    // Only test for dense
    */

    private final List<Integer> Nlst = Arrays.asList(128, 64, 32);
    private final List<Integer> Clst = Arrays.asList(30, 20, 3);
    private final List<Integer> Hlst = Arrays.asList(400, 128, 32);
    private final List<Integer> Wlst = Arrays.asList(400, 128, 32);
    private final List<Integer> Klst = Arrays.asList(30, 20, 10);
    private final List<Integer> Rlst = Arrays.asList(128, 63, 4);
    private final List<Integer> Slst = Arrays.asList(128, 63, 4);
    private final List<Integer> strideXlst = Arrays.asList(9, 3);
    private final List<Integer> strideYlst = Arrays.asList(9, 3);
    private final List<Integer> padXlst = Arrays.asList(3, 1);
    private final List<Integer> padYlst = Arrays.asList(3, 1);
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

        for (long N : Nlst) {
            for (long C : Clst) {
                for (long H : Hlst) {
                    for (long W : Wlst) {
                        for (long K : Klst) {
                            for (long R : Rlst) {
                                for (long S : Slst) {
                                    for (long strideX : strideXlst) {
                                        for (long strideY : strideYlst) {
                                            for (long padX : padXlst) {
                                                for (long padY : padYlst) {
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

                                                        int P = (int) ConvolutionUtils.getP(H, R, strideX, padX);
                                                        int Q = (int) ConvolutionUtils.getQ(W, S, strideY, padY);

                                                        long doutSize = N * K * P * Q * 8l;
                                                        if (doutSize > MAX_OP_SIZE) // dout/output size
                                                            continue;

                                                        double imageSizeInMB = imageSize / (1024.0 * 1024.0);
                                                        double filterSizeInMB = filterSize / (1024.0 * 1024.0);
                                                        double doutSizeInMB = doutSize / (1024.0 * 1024.0);
                                                        System.out.format(
                                                            "conv2d_backward_filter, image[%d,%d,%d,%d](%.1fMB), filter[%d,%d,%d,%d](%.1f), dout[%d,%d,%d,%d](%.1fMB), stride[%d,%d], padding[%d,%d]",
                                                            N, C, H, W, imageSizeInMB, N, C, R, S, filterSizeInMB, N, K,
                                                            P, Q, doutSizeInMB, strideX, strideY, padX, padY);
                                                        Matrix image = generateInputMatrix(spark, (int)N, (int) (C * H * W),
                                                            sparsity, seed);
                                                        Matrix filter = generateInputMatrix(spark, (int)K, (int) (C * R * S),
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

        for (long N : Nlst) {
            for (long C : Clst) {
                for (long H : Hlst) {
                    for (long W : Wlst) {
                        for (long K : Klst) {
                            for (long R : Rlst) {
                                for (long S : Slst) {
                                    for (long strideX : strideXlst) {
                                        for (long strideY : strideYlst) {
                                            for (long padX : padXlst) {
                                                for (long padY : padYlst) {
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

                                                        int P = (int) ConvolutionUtils.getP(H, R, strideX, padX);
                                                        int Q = (int) ConvolutionUtils.getQ(W, S, strideY, padY);

                                                        long doutSize = N * K * P * Q * 8l;
                                                        if (doutSize > MAX_OP_SIZE) // dout/output size
                                                            continue;

                                                        double imageSizeInMB = imageSize / (1024.0 * 1024.0);
                                                        double filterSizeInMB = filterSize / (1024.0 * 1024.0);
                                                        double doutSizeInMB = doutSize / (1024.0 * 1024.0);
                                                        System.out.format(
                                                            "conv2d_backward_filter, image[%d,%d,%d,%d](%.1fMB), filter[%d,%d,%d,%d](%.1f), dout[%d,%d,%d,%d](%.1fMB), stride[%d,%d], padding[%d,%d]",
                                                            N, C, H, W, imageSizeInMB, N, C, R, S, filterSizeInMB, N, K,
                                                            P, Q, doutSizeInMB, strideX, strideY, padX, padY);
                                                        Matrix image = generateInputMatrix(spark, (int)N, (int)(C * H * W),
                                                            sparsity, seed);
                                                        Matrix dout = generateInputMatrix(spark, (int)N, (int)(K * P * Q), sparsity,
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

        for (long N : Nlst) {
            for (long C : Clst) {
                for (long H : Hlst) {
                    for (long W : Wlst) {
                        for (long K : Klst) {
                            for (long R : Rlst) {
                                for (long S : Slst) {
                                    for (long strideX : strideXlst) {
                                        for (long strideY : strideYlst) {
                                            for (long padX : padXlst) {
                                                for (long padY : padYlst) {
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

                                                        int P = (int) ConvolutionUtils.getP(H, R, strideX, padX);
                                                        int Q = (int) ConvolutionUtils.getQ(W, S, strideY, padY);

                                                        long doutSize = N * K * P * Q * 8l;
                                                        if (doutSize > MAX_OP_SIZE) // dout/output size
                                                            continue;

                                                        double imageSizeInMB = imageSize / (1024.0 * 1024.0);
                                                        double filterSizeInMB = filterSize / (1024.0 * 1024.0);
                                                        double doutSizeInMB = doutSize / (1024.0 * 1024.0);
                                                        System.out.format(
                                                            "conv2d_backward_data, image[%d,%d,%d,%d](%.1fMB), filter[%d,%d,%d,%d](%.1f), dout[%d,%d,%d,%d](%.1fMB), stride[%d,%d], padding[%d,%d]",
                                                            N, C, H, W, imageSizeInMB, N, C, R, S, filterSizeInMB, N, K,
                                                            P, Q, doutSizeInMB, strideX, strideY, padX, padY);

                                                        Matrix filter = generateInputMatrix(spark, (int)K, (int)(C * R * S),
                                                            sparsity, seed);
                                                        Matrix dout = generateInputMatrix(spark, (int)N, (int)(K * P * Q), sparsity,
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

        for (long N : Nlst) {
            for (long C : Clst) {
                for (long H : Hlst) {
                    for (long W : Wlst) {
                        for (long R : Rlst) {
                            for (long S : Slst) {
                                for (long strideX : strideXlst) {
                                    for (long strideY : strideYlst) {
                                        for (long padX : padXlst) {
                                            for (long padY : padYlst) {
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

                                                    int P = (int) ConvolutionUtils.getP(H, R, strideX, padX);
                                                    int Q = (int) ConvolutionUtils.getQ(W, S, strideY, padY);

                                                    long doutSize = N * C * P * Q * 8l;
                                                    if (doutSize > MAX_OP_SIZE) // dout/output size
                                                        continue;

                                                    double imageSizeInMB = imageSize / (1024.0 * 1024.0);
                                                    double poolSizeInMB = poolSize / (1024.0 * 1024.0);
                                                    double doutSizeInMB = doutSize / (1024.0 * 1024.0);
                                                    System.out.format(
                                                        "max_pool, image[%d,%d,%d,%d](%.1fMB), pool[%d,%d](%.1f), dout[%d,%d,%d,%d](%.1fMB), stride[%d,%d], padding[%d,%d]",
                                                        N, C, H, W, imageSizeInMB, R, S, poolSizeInMB, N, C, P, Q,
                                                        doutSizeInMB, strideX, strideY, padX, padY);

                                                    Matrix image = generateInputMatrix(spark, (int)N, (int)(C * H * W), sparsity,
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

        for (long N : Nlst) {
            for (long C : Clst) {
                for (long H : Hlst) {
                    for (long W : Wlst) {
                        for (long R : Rlst) {
                            for (long S : Slst) {
                                for (long strideX : strideXlst) {
                                    for (long strideY : strideYlst) {
                                        for (long padX : padXlst) {
                                            for (long padY : padYlst) {
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

                                                    int P = (int) ConvolutionUtils.getP(H, R, strideX, padX);
                                                    int Q = (int) ConvolutionUtils.getQ(W, S, strideY, padY);

                                                    long doutSize = N * C * P * Q * 8l;
                                                    if (doutSize > MAX_OP_SIZE) // dout/output size
                                                        continue;

                                                    double imageSizeInMB = imageSize / (1024.0 * 1024.0);
                                                    double poolSizeInMB = poolSize / (1024.0 * 1024.0);
                                                    double doutSizeInMB = doutSize / (1024.0 * 1024.0);
                                                    System.out.format(
                                                        "max_pool_backward, image[%d,%d,%d,%d](%.1fMB), pool[%d,%d](%.1f), dout[%d,%d,%d,%d](%.1fMB), stride[%d,%d], padding[%d,%d]",
                                                        N, C, H, W, imageSizeInMB, R, S, poolSizeInMB, N, C, P, Q,
                                                        doutSizeInMB, strideX, strideY, padX, padY);

                                                    Matrix image = generateInputMatrix(spark, (int)N, (int)(C * H * W), sparsity,
                                                        seed);
                                                    Matrix dout = generateInputMatrix(spark, (int)N, (int)(C * P * Q), sparsity,
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
