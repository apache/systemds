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

package org.apache.sysds.cujava.driver;

import org.apache.sysds.cujava.CuJavaLibLoader;
import org.apache.sysds.cujava.CudaException;
import org.apache.sysds.cujava.Pointer;

public class CuJavaDriver {

	private static boolean exceptionsEnabled = false;

	private static final String LIB_BASE = "cujava_driver";

	private CuJavaDriver() {

	}

	static {
		CuJavaLibLoader.load(LIB_BASE);
	}

	private static int checkCudaResult(int result) {
		if(exceptionsEnabled && result != CUresult.CUDA_SUCCESS) {
			throw new CudaException(CUresult.resultString(result));
		}
		return result;
	}

	public static int cuCtxCreate(CUcontext pctx, int flags, CUdevice dev) {
		return checkCudaResult(cuCtxCreateNative(pctx, flags, dev));
	}

	private static native int cuCtxCreateNative(CUcontext pctx, int flags, CUdevice dev);

	public static int cuDeviceGet(CUdevice device, int ordinal) {
		return checkCudaResult(cuDeviceGetNative(device, ordinal));
	}

	private static native int cuDeviceGetNative(CUdevice device, int ordinal);

	public static int cuDeviceGetCount(int count[]) {
		return checkCudaResult(cuDeviceGetCountNative(count));
	}

	private static native int cuDeviceGetCountNative(int count[]);

	public static int cuInit(int flags) {
		return checkCudaResult(cuInitNative(flags));
	}

	private static native int cuInitNative(int flags);

	public static int cuLaunchKernel(CUfunction f, int gridDimX, int gridDimY, int gridDimZ, int blockDimX,
		int blockDimY, int blockDimZ, int sharedMemBytes, CUstream hStream, Pointer kernelParams, Pointer extra) {
		return checkCudaResult(
			cuLaunchKernelNative(f, gridDimX, gridDimY, gridDimZ, blockDimX, blockDimY, blockDimZ, sharedMemBytes,
				hStream, kernelParams, extra));
	}

	private static native int cuLaunchKernelNative(CUfunction f, int gridDimX, int gridDimY, int gridDimZ,
		int blockDimX, int blockDimY, int blockDimZ, int sharedMemBytes, CUstream hStream, Pointer kernelParams,
		Pointer extra);

	public static int cuModuleGetFunction(CUfunction hfunc, CUmodule hmod, String name) {
		return checkCudaResult(cuModuleGetFunctionNative(hfunc, hmod, name));
	}

	private static native int cuModuleGetFunctionNative(CUfunction hfunc, CUmodule hmod, String name);

	public static int cuModuleLoadDataEx(CUmodule phMod, Pointer p, int numOptions, int options[],
		Pointer optionValues) {
		if(numOptions == 0) {
			options = (options != null) ? options : new int[0];
			optionValues = (optionValues != null) ? optionValues : Pointer.to(new int[0]);
		}
		return checkCudaResult(cuModuleLoadDataExNative(phMod, p, numOptions, options, optionValues));
	}

	private static native int cuModuleLoadDataExNative(CUmodule phMod, Pointer p, int numOptions, int options[],
		Pointer optionValues);

	public static void setExceptionsEnabled(boolean enabled) {
		exceptionsEnabled = enabled;
	}

	public static int cuMemAlloc(CUdeviceptr dptr, long bytesize) {
		return checkCudaResult(cuMemAllocNative(dptr, bytesize));
	}

	private static native int cuMemAllocNative(CUdeviceptr dptr, long bytesize);

	public static int cuModuleUnload(CUmodule hmod) {
		return checkCudaResult(cuModuleUnloadNative(hmod));
	}

	private static native int cuModuleUnloadNative(CUmodule hmod);

	public static int cuCtxDestroy(CUcontext ctx) {
		return checkCudaResult(cuCtxDestroyNative(ctx));
	}

	private static native int cuCtxDestroyNative(CUcontext ctx);

	public static int cuMemFree(CUdeviceptr dptr) {
		return checkCudaResult(cuMemFreeNative(dptr));
	}

	private static native int cuMemFreeNative(CUdeviceptr dptr);


	public static int cuMemcpyDtoH(Pointer dstHost, CUdeviceptr srcDevice, long ByteCount) {
		return checkCudaResult(cuMemcpyDtoHNative(dstHost, srcDevice, ByteCount));
	}

	private static native int cuMemcpyDtoHNative(Pointer dstHost, CUdeviceptr srcDevice, long ByteCount);

	public static int cuCtxSynchronize() {
		return checkCudaResult(cuCtxSynchronizeNative());
	}

	private static native int cuCtxSynchronizeNative();

	public static int cuDeviceGetAttribute(int pi[], int attrib, CUdevice dev) {
		return checkCudaResult(cuDeviceGetAttributeNative(pi, attrib, dev));
	}

	private static native int cuDeviceGetAttributeNative(int pi[], int attrib, CUdevice dev);
}
