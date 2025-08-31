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

package org.apache.sysds.cujava.cublas;

import org.apache.sysds.cujava.CuJavaLibLoader;
import org.apache.sysds.cujava.CudaException;
import org.apache.sysds.cujava.Pointer;

/**
 * The methods declared in this class refer to cublas v2. cublas v1 is deprecated in CUDA 12 and SystemDS does not
 * utilize v1 methods anymore.
 */

public class CuJavaCublas {

	private static boolean exceptionsEnabled = false;

	private static final String LIB_BASE = "cujava_cublas";

	private CuJavaCublas() {
		// prevent instantiation
	}

	static {
		CuJavaLibLoader.load(LIB_BASE);
	}

	private static int checkCublasStatus(int result) {
		if(exceptionsEnabled && result != cublasStatus.CUBLAS_STATUS_SUCCESS) {
			throw new CudaException(cublasStatus.statusString(result));
		}
		return result;
	}

	public static void setExceptionsEnabled(boolean enabled) {
		exceptionsEnabled = enabled;
	}

	public static int cublasCreate(cublasHandle handle) {
		return checkCublasStatus(cublasCreateNative(handle));
	}

	private static native int cublasCreateNative(cublasHandle handle);

	public static int cublasDestroy(cublasHandle handle) {
		return checkCublasStatus(cublasDestroyNative(handle));
	}

	private static native int cublasDestroyNative(cublasHandle handle);

	public static int cublasDgeam(cublasHandle handle, int transa, int transb, int m, int n, Pointer alpha, Pointer A,
		int lda, Pointer beta, Pointer B, int ldb, Pointer C, int ldc) {
		return checkCublasStatus(cublasDgeamNative(handle, transa, transb, m, n, alpha, A, lda, beta, B, ldb, C, ldc));
	}

	private static native int cublasDgeamNative(cublasHandle handle, int transa, int transb, int m, int n,
		Pointer alpha, Pointer A, int lda, Pointer beta, Pointer B, int ldb, Pointer C, int ldc);

	public static int cublasDdot(cublasHandle handle, int n, Pointer x, int incx, Pointer y, int incy, Pointer result) {
		return checkCublasStatus(cublasDdotNative(handle, n, x, incx, y, incy, result));
	}

	private static native int cublasDdotNative(cublasHandle handle, int n, Pointer x, int incx, Pointer y, int incy,
		Pointer result);
}
