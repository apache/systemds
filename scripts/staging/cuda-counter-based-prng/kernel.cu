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

#include <cuda_runtime.h>
#include <iostream>
#include <iomanip>
#include <chrono>
#include <math.h>
#include <Random123/philox.h>
#include <Random123/uniform.hpp>

// CUDA kernel to generate random doubles between 0 and 1 using all 4 integers from Philox
extern "C" __global__ void philox_4_64_uniform(double* output, uint64_t originalKey, r123::Philox4x64::ctr_type startingCounter, size_t numElements) {
    // Calculate the thread's unique index
    uint64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    double UINT_TO_ZERO_ONE = 1.0 / LONG_MAX;

    // Ensure the thread index is within bounds
    if (idx * 4 < numElements) {
        // Initialize the Philox generator with a unique counter and key
        r123::Philox4x64 rng;
        r123::Philox4x64::ctr_type ctr;
        uint64_t sum0 = startingCounter[0] + idx;
        uint64_t sum1 = startingCounter[1] + (sum0 < startingCounter[0] ? 1 : 0); // Carry-Bit

        ctr[0] = sum0;
        ctr[1] = sum1;
        ctr[2] = startingCounter[2];
        ctr[3] = startingCounter[3];
        r123::Philox4x64::key_type key = {{originalKey}};                          // Key (seed)

        // Generate 4 random integers
        r123::Philox4x64::ctr_type result = rng(ctr, key);

        // Convert each 64-bit integer to a double in [-1, 1]
        for (int i = 0; i < 4; ++i) {
            double randomDouble = static_cast<double>((long)result[i]) * UINT_TO_ZERO_ONE;
            size_t outputIdx = idx * 4 + i;

            // Ensure we don't exceed the output array bounds
            if (outputIdx < numElements) {
                output[outputIdx] = randomDouble;
            }
        }
    }
}

// CUDA kernel to generate random doubles between 0 and 1 using all 4 integers from Philox
extern "C" __global__ void philox_4_64_standard(double* output, uint64_t originalKey, r123::Philox4x64::ctr_type startingCounter, size_t numElements) {
    // Calculate the thread's unique index
    uint64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    uint64_t idx2 = idx + numElements;
    double UINT_TO_ZERO_ONE = 1.0 / LONG_MAX;

    // Ensure the thread index is within bounds
    if (idx * 4 < numElements) {
        // Initialize the Philox generator with a unique counter and key
        r123::Philox4x64 rng;
        r123::Philox4x64::ctr_type ctr1;
        uint64_t sum0 = startingCounter[0] + idx;
        uint64_t sum1 = startingCounter[1] + (sum0 < startingCounter[0] ? 1 : 0); // Carry-Bit

        ctr1[0] = sum0;
        ctr1[1] = sum1;
        ctr1[2] = startingCounter[2];
        ctr1[3] = startingCounter[3];
        r123::Philox4x64::ctr_type ctr2;
        sum0 = startingCounter[0] + idx2;
        sum1 = startingCounter[1] + (sum0 < startingCounter[0] ? 1 : 0); // Carry-Bit

        ctr2[0] = sum0;
        ctr2[1] = sum1;
        ctr2[2] = startingCounter[2];
        ctr2[3] = startingCounter[3];

        r123::Philox4x64::key_type key1 = {{originalKey}};
        r123::Philox4x64::key_type key2 = {{originalKey}};

        // Generate 4 random integers
        r123::Philox4x64::ctr_type result1 = rng(ctr1, key1);
        r123::Philox4x64::ctr_type result2 = rng(ctr2, key2);

        // Convert each 64-bit integer to a double in [-1, 1]
        for (int i = 0; i < 4; ++i) {
            double randomDouble1 = static_cast<double>((long)result1[i]) * UINT_TO_ZERO_ONE;
            double randomDouble2 = static_cast<double>((long)result2[i]) * UINT_TO_ZERO_ONE;

            size_t outputIdx = idx * 4 + i;

            // Ensure we don't exceed the output array bounds
            if (outputIdx < numElements) {
                output[outputIdx] = (randomDouble1 + randomDouble2) / 2;
            }
        }
    }
}


// CUDA kernel to generate random integers from Philox
extern "C" __global__ void philox_4_32(uint* output, uint32_t seed, uint32_t startingCounter, size_t numElements) {
    // Calculate the thread's unique index
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Ensure the thread index is within bounds
    if (idx * 4 < numElements) {
        // Initialize the Philox generator with a unique counter and key
        r123::Philox4x32 rng;
        r123::Philox4x32::ctr_type ctr = {{startingCounter + idx, 0, 0, 0}}; // Counter (startingCounter + thread index)
        r123::Philox4x32::key_type key = {{seed}};                          // Key (seed)

        // Generate 4 random integers
        r123::Philox4x32::ctr_type result = rng(ctr, key);

        for (int i = 0; i < 4; ++i) {
            size_t outputIdx = idx * 4 + i;

            // Ensure we don't exceed the output array bounds
            if (outputIdx < numElements) {
                output[outputIdx] = result[i];
            }
        }
    }
}


// CUDA kernel to generate random longs from Philox
extern "C" __global__ void philox_4_64(ulong* output, uint64_t seed, uint64_t startingCounter, size_t numElements) {
    // Calculate the thread's unique index
    uint64_t idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Ensure the thread index is within bounds
    if (idx * 4 < numElements) {
        // Initialize the Philox generator with a unique counter and key
        r123::Philox4x64 rng;
        r123::Philox4x64::ctr_type ctr = {{startingCounter + idx, 0, 0, 0}}; // Counter (startingCounter + thread index)
        r123::Philox4x64::key_type key = {{seed}};                          // Key (seed)

        // Generate 4 random integers
        r123::Philox4x64::ctr_type result = rng(ctr, key);

        for (int i = 0; i < 4; ++i) {
            size_t outputIdx = idx * 4 + i;

            // Ensure we don't exceed the output array bounds
            if (outputIdx < numElements) {
                output[outputIdx] = result[i];
            }
        }
    }
}


int main(int argc, char** argv) {
    // Check command-line arguments
    if (argc != 4) {
        std::cerr << "Usage: " << argv[0] << " <seed> <startingCounter> <numElements>\n";
        return 1;
    }

    // Parse command-line arguments
    uint64_t seed = std::stoull(argv[1]);          // Seed (key)
    uint64_t startingCounter = std::stoull(argv[2]); // Starting counter
    size_t numElements = std::stoul(argv[3]);      // Number of random numbers to generate

    // Allocate host and device memory
    double* h_output = new double[numElements];
    double* d_output;
    cudaMalloc(&d_output, numElements * sizeof(double));

    // Launch the CUDA kernel
    const int blockSize = 512;
    const int gridSize = (numElements + blockSize * 4 - 1) / (blockSize * 4); // Adjust grid size for 4 outputs per thread
    r123::Philox4x64::ctr_type uniformCounter = {{startingCounter, 0, 0, 0}};

    auto start = std::chrono::high_resolution_clock::now();
    philox_4_64_standard<<<gridSize, blockSize>>>(d_output, seed, uniformCounter, numElements);
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);

    std::cout << "Time: " << duration.count() << " microseconds" << std::endl;

    // Copy the results back to the host
    cudaMemcpy(h_output, d_output, numElements * sizeof(double), cudaMemcpyDeviceToHost);

    // Print the first 10 random doubles
    std::cout << "First 10 random doubles:\n";
    for (int i = 0; i < 10; ++i) {
        std::cout << h_output[i] << "\n";
    }

    double avg = 0.0;
    for (int i = 0; i < numElements; i++) {
        avg += (double)h_output[i] / numElements;
    }
    printf("Average: %f\n", avg);
    double standardDeviation = 0.0;
    for (int i = 0; i < numElements; i++) {
        standardDeviation += std::pow((double)h_output[i] - avg, 2);
    }
    standardDeviation = sqrt(standardDeviation / numElements);
    printf("standardDeviation: %f\n", standardDeviation);


    // Free memory
    delete[] h_output;
    cudaFree(d_output);

    // --------------------------------------------------------------------------------

    seed = std::stoull(argv[1]);          // Seed (key)
    startingCounter = std::stoull(argv[2]); // Starting counter
    numElements = std::stoul(argv[3]);      // Number of random numbers to generate

    // Allocate host and device memory
    uint* h_output_int = new uint[numElements];
    uint* d_output_int;
    cudaMalloc(&d_output_int, numElements * sizeof(uint));

    // Launch the CUDA kernel
    philox_4_32<<<gridSize, blockSize>>>(d_output_int, seed, startingCounter, numElements);

    // Copy the results back to the host
    cudaMemcpy(h_output_int, d_output_int, numElements * sizeof(uint), cudaMemcpyDeviceToHost);

    // Print the first 10 random doubles
    std::cout << "First 10 random doubles:\n";
    for (int i = 0; i < 10; ++i) {
        std::cout << std::hex << h_output_int[i] << " " << r123::uneg11<float>(h_output_int[i]) << "\n";
    }

    // Free memory
    delete[] h_output_int;
    cudaFree(d_output_int);

    // --------------------------------------------------------------------------------

    seed = std::stoull(argv[1]);          // Seed (key)
    startingCounter = std::stoull(argv[2]); // Starting counter
    numElements = std::stoul(argv[3]);      // Number of random numbers to generate

    // Allocate host and device memory
    ulong* h_output_long = new ulong[numElements];
    ulong* d_output_long;
    cudaMalloc(&d_output_long, numElements * sizeof(ulong));

    // Launch the CUDA kernel
    philox_4_64<<<gridSize, blockSize>>>(d_output_long, seed, startingCounter, numElements);

    // Copy the results back to the host
    cudaMemcpy(h_output_long, d_output_long, numElements * sizeof(ulong), cudaMemcpyDeviceToHost);

    // Print the first 10 random doubles
    std::cout << "First 10 random doubles:\n";
    for (int i = 0; i < 10; ++i) {
        std::cout << std::setprecision(17) << std::hex << h_output_long[i] << " " << (static_cast<double>((long)h_output_long[i]) / LONG_MAX) << "\n";
    }

    // Free memory
    delete[] h_output_long;
    cudaFree(d_output_long);

    return 0;
}
