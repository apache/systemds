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
import org.apache.sysds.cujava.Pointer;
import org.apache.sysds.cujava.CudaException;
import org.apache.sysds.cujava.runtime.CudaError;

public class CuJava {

	private static boolean exceptionsEnabled = true;

	private static final String LIB_BASE = "cujava_runtime";

	private CuJava(){

	}

	static {
		CuJavaLibLoader.load(LIB_BASE);
	}

	private static int checkCudaError(int result)
	{
		if (exceptionsEnabled && result != CudaError.cudaSuccess)
		{
			throw new CudaException(CudaError.errorString(result));
		}
		return result;
	}

	public static int cudaMemcpy(Pointer dst, Pointer src, long count, int cudaMemcpyKind_kind) {
		return checkCudaError(cudaMemcpyNative(dst, src, count, cudaMemcpyKind_kind));
	}
	private static native int cudaMemcpyNative(Pointer dst, Pointer src, long count, int cudaMemcpyKind_kind);


	public static int cudaMalloc(Pointer devPtr, long size)
	{
		return checkCudaError(cudaMallocNative(devPtr, size));
	}
	private static native int cudaMallocNative(Pointer devPtr, long size);

	public static int cudaFree(Pointer devPtr)
	{
		return checkCudaError(cudaFreeNative(devPtr));
	}
	private static native int cudaFreeNative(Pointer devPtr);


}
