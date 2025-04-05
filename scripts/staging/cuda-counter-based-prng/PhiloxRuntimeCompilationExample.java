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

import java.io.BufferedReader;
import java.io.File;
import java.io.FileWriter;
import java.io.InputStreamReader;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;

import static java.nio.file.Files.readAllBytes;
import static jcuda.driver.JCudaDriver.*;

public class PhiloxRuntimeCompilationExample implements AutoCloseable {
	private static String philox4x64KernelSource = "#include <cuda_runtime.h>\n" +
		"#include <Random123/philox.h>\n" +
		"extern \"C\" __global__ void philox_4_64(ulong* output, uint64_t startingCounter, uint64_t seed, size_t numElements) {\n"
		+
		"    uint64_t idx = blockIdx.x * blockDim.x + threadIdx.x;\n" +
		"    if (idx * 4 < numElements) {\n" +
		"        r123::Philox4x64 rng;\n" +
		"        r123::Philox4x64::ctr_type ctr = {{startingCounter + idx, 0, 0, 0}};\n" +
		"        r123::Philox4x64::key_type key = {{seed}};\n" +
		"        r123::Philox4x64::ctr_type result = rng(ctr, key);\n" +
		"        for (int i = 0; i < 4; ++i) {\n" +
		"            size_t outputIdx = idx * 4 + i;\n" +
		"            if (outputIdx < numElements) {\n" +
		"                output[outputIdx] = result[i];\n" +
		"            }\n" +
		"        }\n" +
		"    }\n" +
		"}\n";

	private final CUcontext context;
	private final CUmodule module;
	private final CUfunction function;
	private final int blockSize;

	public PhiloxRuntimeCompilationExample() {
		JCudaDriver.setExceptionsEnabled(true);
		// Initialize CUDA
		cuInit(0);
		CUdevice device = new CUdevice();
		cuDeviceGet(device, 0);
		context = new CUcontext();
		int result = cuCtxCreate(context, 0, device);
		if (result != CUresult.CUDA_SUCCESS) {
			throw new RuntimeException(
				"Kontext-Erstellung fehlgeschlagen: " + result + ", " + CUresult.stringFor(result));
		}

		// Compile to PTX
		String ptx = compileToTPX(philox4x64KernelSource);

		// Load the PTX
		module = new CUmodule();
		cuModuleLoadData(module, ptx);
		function = new CUfunction();
		cuModuleGetFunction(function, module, "philox_4_64");

		// Set block size based on device capabilities
		blockSize = 64; // Can be adjusted based on device properties
	}

	private String compileToTPX(String source) {
		try {
			// Temporäre Dateien erstellen
			File sourceFile = File.createTempFile("philox_kernel", ".cu");
			File outputFile = File.createTempFile("philox_kernel", ".ptx");

			// CUDA-Quellcode in temporäre Datei schreiben
			try (FileWriter writer = new FileWriter(sourceFile)) {
				writer.write(philox4x64KernelSource);
			}

			// nvcc Kommando zusammenbauen
			List<String> command = new ArrayList<>();
			command.add("/usr/local/cuda/bin/nvcc");
			command.add("-ccbin");
			command.add("gcc-8");
			command.add("--ptx"); // PTX-Output generieren
			command.add("-o");
			command.add(outputFile.getAbsolutePath());
			command.add("-I");
			command.add("./lib/random123/include");
			command.add(sourceFile.getAbsolutePath());

			// Prozess erstellen und ausführen
			ProcessBuilder pb = new ProcessBuilder(command);
			pb.redirectErrorStream(true);
			Process process = pb.start();

			// Output des Kompilers lesen
			try (BufferedReader reader = new BufferedReader(
				new InputStreamReader(process.getInputStream()))) {
				String line;
				StringBuilder output = new StringBuilder();
				while ((line = reader.readLine()) != null) {
					output.append(line).append("\n");
				}
				System.out.println("Compiler Output: " + output.toString());
			}

			// Auf Prozessende warten
			int exitCode = process.waitFor();
			if (exitCode != 0) {
				throw new RuntimeException("nvcc Kompilierung fehlgeschlagen mit Exit-Code: " + exitCode);
			}

			// PTX-Datei einlesen
			String ptxCode = new String(readAllBytes(outputFile.toPath()));

			// Aufräumen
			sourceFile.delete();
			outputFile.delete();

			return ptxCode;

		} catch (Exception e) {
			throw new RuntimeException("Fehler bei der CUDA-Kompilierung: " + e.getMessage(), e);
		}
	}

	/**
	 * Generates random numbers using the Philox4x64 algorithm
	 *
	 * @param startingCounter Initial counter value
	 * @param seed            Random seed
	 * @param numElements     Number of random numbers to generate
	 * @return Array of random numbers
	 */
	public CUdeviceptr Philox4x64(long startingCounter, long seed, int numElements) {
		// Allocate host memory for results
		// long[] hostOutput = new long[numElements];

		// Allocate device memory
		CUdeviceptr deviceOutput = new CUdeviceptr();
		cuMemAlloc(deviceOutput, (long) numElements * Sizeof.LONG);

		try {
			// Set up kernel parameters mit Debugging
			System.out.printf("numElements: %d, seed: %d, startingCounter: %d%n",
				numElements, seed, startingCounter);

			Pointer kernelParams = Pointer.to(
				Pointer.to(deviceOutput),
				Pointer.to(new long[] { startingCounter }),
				Pointer.to(new long[] { seed }),
				Pointer.to(new long[] { numElements }));

			// Calculate grid size
			int gridSize = (numElements + (blockSize * 4) - 1) / (blockSize * 4);

			// Launch kernel mit Fehlerprüfung
			int kernelResult = cuLaunchKernel(function,
				gridSize, 1, 1, // Grid dimension
				blockSize, 1, 1, // Block dimension
				0, null, // Shared memory size and stream
				kernelParams, null // Kernel parameters and extra parameters
			);
			if (kernelResult != CUresult.CUDA_SUCCESS) {
				throw new RuntimeException(
					"Kernel-Launch fehlgeschlagen: " + kernelResult + ", " + CUresult.stringFor(kernelResult));
			}

			// Copy results back to host
			// cuMemcpyDtoH(Pointer.to(hostOutput), deviceOutput, (long) numElements *
			// Sizeof.LONG);
		} finally {
			// Free device memory
			// cuMemFree(deviceOutput);
		}

		// return hostOutput;
		return deviceOutput;
	}

	/**
	 * Cleans up CUDA resources
	 */
	public void close() {
		cuModuleUnload(module);
		cuCtxDestroy(context);
	}

	// Example usage
	public static void main(String[] args) {
		try (PhiloxRuntimeCompilationExample generator = new PhiloxRuntimeCompilationExample()) {
			// Generate 1 million random numbers
			int numElements = 1_000_000;
			long seed = 0L;
			long startingCounter = 0L;

			CUdeviceptr randomNumbers = generator.Philox4x64(startingCounter, seed, numElements);

			long[] elements = new long[10];
			cuMemcpyDtoH(Pointer.to(elements), randomNumbers, 10L * Sizeof.LONG);
			cuMemFree(randomNumbers);

			// Print first few numbers
			System.out.println("First 10 random numbers:");
			for (int i = 0; i < 10; i++) {
				System.out.printf("%d: %x%n", i, elements[i]);
			}

			int size = 10_000_000;
			long start = System.currentTimeMillis();
			CUdeviceptr ptr = generator.Philox4x64(0L, 0L, size);
			long end = System.currentTimeMillis();
			System.out.println("philox4x64 speed test: " + (end - start) * 1000 + " microseconds");
			cuMemFree(ptr);
			Random r = new Random();
			long javaStart = System.currentTimeMillis();
			for (int i = 0; i < size; i++) {
				r.nextLong();
			}
			long javaEnd = System.currentTimeMillis();
			System.out.println("java speed test: " + (javaEnd - javaStart) * 1000 + " microseconds");
			System.out.println("philox4x64 is " + (double) (javaEnd - javaStart) / (double) (end - start)
				+ " times faster than java");

		}
	}
}
