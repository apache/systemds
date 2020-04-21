/*
 * Modifications Copyright 2019 Graz University of Technology
 *
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
 package org.tugraz.sysds.runtime.instructions.gpu.context;

import static jcuda.driver.JCudaDriver.cuLaunchKernel;
import static jcuda.driver.JCudaDriver.cuModuleGetFunction;
import static jcuda.driver.JCudaDriver.cuModuleLoadDataEx;

import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.util.HashMap;

import org.tugraz.sysds.api.DMLScript;
import org.tugraz.sysds.runtime.DMLRuntimeException;
import org.tugraz.sysds.runtime.io.IOUtilFunctions;
import org.tugraz.sysds.runtime.matrix.data.LibMatrixCUDA;

import jcuda.Pointer;
import jcuda.driver.CUfunction;
import jcuda.driver.CUmodule;
import jcuda.driver.CUresult;
import jcuda.runtime.JCuda;

public class JCudaKernels {

	private final static String ptxFileName = "/kernels/SystemDS.ptx";
	private HashMap<String, CUfunction> kernels = new HashMap<>();
	private CUmodule module;

	/**
	 * Loads the kernels in the file ptxFileName. Though cubin files are also supported, we will stick with
	 * ptx file as they are target-independent similar to Java's .class files.
	 */
	JCudaKernels() {
		module = new CUmodule();
		// Load the kernels specified in the ptxFileName file
		checkResult(cuModuleLoadDataEx(module, initKernels(ptxFileName), 0, new int[0], Pointer.to(new int[0])));
	}

	/**
	 * Setups the kernel parameters and launches the kernel using cuLaunchKernel API.
	 * This function currently supports two dimensional grid and blocks.
	 *
	 * @param name      name of the kernel
	 * @param config    execution configuration
	 * @param arguments can be of type Pointer, long, double, float and int
	 */
	public void launchKernel(String name, ExecutionConfig config, Object... arguments) {
		name = name + LibMatrixCUDA.customKernelSuffix;
		CUfunction function = kernels.get(name);
		
		if (function == null) {
			// caching functions into hashmap reduces the lookup overhead
			function = new CUfunction();
			try {
				checkResult(cuModuleGetFunction(function, module, name));
			} catch(jcuda.CudaException e) {
				throw new DMLRuntimeException("Error finding the custom kernel:" + name, e);
			}
		}

		// Setup parameters
		Pointer[] kernelParams = new Pointer[arguments.length];
		for (int i = 0; i < arguments.length; i++) {
			if (arguments[i] == null) {
				throw new DMLRuntimeException("The argument to the kernel cannot be null.");
			} else if (arguments[i] instanceof Pointer) {
				kernelParams[i] = Pointer.to((Pointer) arguments[i]);
			} else if (arguments[i] instanceof Integer) {
				kernelParams[i] = Pointer.to(new int[] { (Integer) arguments[i] });
			} else if (arguments[i] instanceof Double) {
				kernelParams[i] = Pointer.to(new double[] { (Double) arguments[i] });
			} else if (arguments[i] instanceof Long) {
				kernelParams[i] = Pointer.to(new long[] { (Long) arguments[i] });
			} else if (arguments[i] instanceof Float) {
				kernelParams[i] = Pointer.to(new float[] { (Float) arguments[i] });
			} else {
				throw new DMLRuntimeException("The argument of type " + arguments[i].getClass() + " is not supported.");
			}
		}

		// Launches the kernel using CUDA's driver API.
		checkResult(cuLaunchKernel(function, config.gridDimX, config.gridDimY, config.gridDimZ, config.blockDimX,
				config.blockDimY, config.blockDimZ, config.sharedMemBytes, config.stream, Pointer.to(kernelParams),
				null));
		if(DMLScript.SYNCHRONIZE_GPU)
			JCuda.cudaDeviceSynchronize();
	}

	public static void checkResult(int cuResult) {
		if (cuResult != CUresult.CUDA_SUCCESS) {
			throw new DMLRuntimeException(CUresult.stringFor(cuResult));
		}
	}

	/**
	 * Reads the ptx file from resource
	 *
	 * @param ptxFileName
	 * @return
	 */
	private static Pointer initKernels(String ptxFileName) {
		ByteArrayOutputStream out = null;
		try( InputStream in = JCudaKernels.class.getResourceAsStream(ptxFileName) ) {
			if (in != null) {
				out = new ByteArrayOutputStream();
				byte buffer[] = new byte[8192];
				while (true) {
					int read = in.read(buffer);
					if (read == -1)
						break;
					out.write(buffer, 0, read);
				}
				out.write('\0');
				out.flush();
				return Pointer.to(out.toByteArray());
			} else {
				throw new DMLRuntimeException("The input file " + ptxFileName
						+ " not found. (Hint: Please compile SystemDS using -DenableGPU=true flag. Example: mvn package -DenableGPU=true).");
			}
		} catch (IOException e) {
			throw new DMLRuntimeException("Could not initialize the kernels", e);
		} finally {
			IOUtilFunctions.closeSilently(out);
		}
	}
}
