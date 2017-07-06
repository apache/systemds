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
package org.apache.sysml.runtime.instructions.gpu.context;

import static jcuda.driver.JCudaDriver.cuLaunchKernel;
import static jcuda.driver.JCudaDriver.cuModuleGetFunction;
import static jcuda.driver.JCudaDriver.cuModuleLoadDataEx;

import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.util.HashMap;

import org.apache.sysml.runtime.DMLRuntimeException;
import org.apache.sysml.runtime.io.IOUtilFunctions;

import jcuda.Pointer;
import jcuda.driver.CUfunction;
import jcuda.driver.CUmodule;
import jcuda.driver.CUresult;
import jcuda.runtime.JCuda;

/**
 * Utility class that allows LibMatrixCUDA as well as GPUObject to invoke custom CUDA kernels.
 * <p>
 * The utility org.apache.sysml.runtime.instructions.gpu.context.JCudaKernels simplifies the launching of the kernels.
 * For example: to launch a kernel
 * {@code copyUpperToLowerTriangleDense<<1,1,32,32>>(jcudaDenseMatrixPtr, dim, dim*dim) }, the user has to call:
 * {@code kernels.launchKernel("copyUpperToLowerTriangleDense", new ExecutionConfig(1,1,32,32), jcudaDenseMatrixPtr, dim, dim*dim) }
 */
public class JCudaKernels {

	private final static String ptxFileName = "/kernels/SystemML.ptx";
	private HashMap<String, CUfunction> kernels = new HashMap<String, CUfunction>();
	private CUmodule module;
	//	private final int deviceNum;

	/**
	 * Loads the kernels in the file ptxFileName. Though cubin files are also supported, we will stick with
	 * ptx file as they are target-independent similar to Java's .class files.
	 *
	 * @param deviceNum the device number for which to initiate the driver API
	 * @throws DMLRuntimeException if DMLRuntimeException occurs
	 */
	JCudaKernels(int deviceNum) throws DMLRuntimeException {
		//		this.deviceNum = deviceNum;
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
	 * @throws DMLRuntimeException if DMLRuntimeException occurs
	 */
	public void launchKernel(String name, ExecutionConfig config, Object... arguments) throws DMLRuntimeException {
		CUfunction function = kernels.get(name);
		if (function == null) {
			// caching functions into hashmap reduces the lookup overhead
			function = new CUfunction();
			checkResult(cuModuleGetFunction(function, module, name));
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
		JCuda.cudaDeviceSynchronize();
	}

	public static void checkResult(int cuResult) throws DMLRuntimeException {
		if (cuResult != CUresult.CUDA_SUCCESS) {
			throw new DMLRuntimeException(CUresult.stringFor(cuResult));
		}
	}

	/**
	 * Reads the ptx file from resource
	 *
	 * @param ptxFileName
	 * @return
	 * @throws DMLRuntimeException
	 */
	private Pointer initKernels(String ptxFileName) throws DMLRuntimeException {
		InputStream in = null;
		ByteArrayOutputStream out = null;
		try {
			in = JCudaKernels.class.getResourceAsStream(ptxFileName);
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
						+ " not found. (Hint: Please compile SystemML using -DenableGPU=true flag. Example: mvn package -DenableGPU=true).");
			}
		} catch (IOException e) {
			throw new DMLRuntimeException("Could not initialize the kernels", e);
		} finally {
			IOUtilFunctions.closeSilently(out);
			IOUtilFunctions.closeSilently(in);
		}
	}
}
