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

import jcuda.*;
import jcuda.driver.*;
import jcuda.nvrtc.*;
import jcuda.runtime.JCuda;

import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Paths;

import static jcuda.driver.JCudaDriver.cuCtxCreate;

public class PhiloxJNvrtcExample {

    public static void main(String[] args) {
        // Enable exceptions and omit error checks
        JCuda.setExceptionsEnabled(true);
        JCudaDriver.setExceptionsEnabled(true);
        JNvrtc.setExceptionsEnabled(true);

        String ptx = "";
        try {
            ptx = new String(Files.readAllBytes(Paths.get("philox_kernel.ptx")));
        } catch (IOException e) {
            System.out.println(e.getMessage());
        }

        // Print the PTX for debugging
        //System.out.println("Generated PTX:");
        // System.out.println(ptx);

        // Initialize the driver API and create a context
        JCudaDriver.cuInit(0);
        CUdevice device = new CUdevice();
        JCudaDriver.cuDeviceGet(device, 0);
        CUcontext context = new CUcontext();
        cuCtxCreate(context, 0, device);

        CUmodule module = new CUmodule();
        JCudaDriver.cuModuleLoadData(module, ptx);

        // Get a function pointer to the kernel
        CUfunction function = new CUfunction();
        JCudaDriver.cuModuleGetFunction(function, module, "philox_4_64");

        // Prepare data
        int n = 1000; // Number of random numbers to generate
        long[] hostOut = new long[n];
        CUdeviceptr deviceOut = new CUdeviceptr();
        JCudaDriver.cuMemAlloc(deviceOut, n * Sizeof.LONG);

        // Direkte Werte für seed und startingCounter
        long seed = 0L;        // Fester Seed-Wert
        long startingCounter = 0L;               // Startwert für Counter

        Pointer kernelParameters = Pointer.to(
                Pointer.to(deviceOut),           // ulong* output
                Pointer.to(new long[]{seed}),    // uint64_t seed
                Pointer.to(new long[]{startingCounter}), // uint64_t startingCounter
                Pointer.to(new long[]{n})        // size_t numElements
        );

        // Launch the kernel
        int blockSizeX = 128;
        int gridSizeX = (int) Math.ceil((double)n / blockSizeX);
        JCudaDriver.cuLaunchKernel(
                function,
                gridSizeX, 1, 1,      // Grid dimension
                blockSizeX, 1, 1,     // Block dimension
                0, null,              // Shared memory size and stream
                kernelParameters, null // Kernel- und extra parameters
        );
        JCudaDriver.cuCtxSynchronize();

        // Copy result back
        JCudaDriver.cuMemcpyDtoH(Pointer.to(hostOut), deviceOut, n * Sizeof.LONG);

        // Print results
        System.out.println("Generated random numbers with seed=" +
                          String.format("0x%016X", seed) +
                          " and startingCounter=" + startingCounter);
        for (int i = 0; i < Math.min(10, n); i++) {
            System.out.printf("hostOut[%d] = 0x%016X\n", i, hostOut[i]);
        }

        // Cleanup
        JCudaDriver.cuMemFree(deviceOut);
        JCudaDriver.cuCtxDestroy(context);
    }
}
