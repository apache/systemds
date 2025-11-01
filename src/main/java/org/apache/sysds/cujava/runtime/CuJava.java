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

package org.apache.sysds.cujava.runtime;

import org.apache.sysds.cujava.CuJavaLibLoader;
import org.apache.sysds.cujava.CudaException;
import org.apache.sysds.cujava.Pointer;

public class CuJava {

	private static boolean exceptionsEnabled = false;

	private static final String LIB_BASE = "cujava_runtime";

	public static final int cudaMemAttachGlobal = 1;

	public static final int cudaDeviceScheduleBlockingSync = 4;

	private CuJava() {

	}

	static {
		CuJavaLibLoader.load(LIB_BASE);
	}

	private static int checkCudaError(int result) {
		if(exceptionsEnabled && result != CudaError.cudaSuccess) {
			throw new CudaException(CudaError.errorString(result));
		}
		return result;
	}

	public static int cudaMemcpy(Pointer dst, Pointer src, long count, int cudaMemcpyKind_kind) {
		return checkCudaError(cudaMemcpyNative(dst, src, count, cudaMemcpyKind_kind));
	}

	private static native int cudaMemcpyNative(Pointer dst, Pointer src, long count, int cudaMemcpyKind_kind);

	public static int cudaMalloc(Pointer devPtr, long size) {
		return checkCudaError(cudaMallocNative(devPtr, size));
	}

	private static native int cudaMallocNative(Pointer devPtr, long size);

	public static int cudaFree(Pointer devPtr) {
		return checkCudaError(cudaFreeNative(devPtr));
	}

	private static native int cudaFreeNative(Pointer devPtr);

	public static int cudaMemset(Pointer mem, int c, long count) {
		return checkCudaError(cudaMemsetNative(mem, c, count));
	}

	private static native int cudaMemsetNative(Pointer mem, int c, long count);

	public static int cudaDeviceSynchronize() {
		return checkCudaError(cudaDeviceSynchronizeNative());
	}

	private static native int cudaDeviceSynchronizeNative();

	public static void setExceptionsEnabled(boolean enabled) {
		exceptionsEnabled = enabled;
	}

	public static int cudaMallocManaged(Pointer devPtr, long size, int flags) {
		return checkCudaError(cudaMallocManagedNative(devPtr, size, flags));
	}

	private static native int cudaMallocManagedNative(Pointer devPtr, long size, int flags);

	public static int cudaMemGetInfo(long free[], long total[]) {
		return checkCudaError(cudaMemGetInfoNative(free, total));
	}

	private static native int cudaMemGetInfoNative(long free[], long total[]);

	public static int cudaGetDeviceCount(int count[]) {
		return checkCudaError(cudaGetDeviceCountNative(count));
	}

	private static native int cudaGetDeviceCountNative(int count[]);

	public static int cudaSetDevice(int device) {
		return checkCudaError(cudaSetDeviceNative(device));
	}

	private static native int cudaSetDeviceNative(int device);

	public static int cudaSetDeviceFlags(int flags) {
		return checkCudaError(cudaSetDeviceFlagsNative(flags));
	}

	private static native int cudaSetDeviceFlagsNative(int flags);

	public static int cudaGetDevice(int device[]) {
		return checkCudaError(cudaGetDeviceNative(device));
	}

	private static native int cudaGetDeviceNative(int device[]);

	public static int cudaGetDeviceProperties(CudaDeviceProp prop, int device) {
		return checkCudaError(cudaGetDevicePropertiesNative(prop, device));
	}

	private static native int cudaGetDevicePropertiesNative(CudaDeviceProp prop, int device);

}
