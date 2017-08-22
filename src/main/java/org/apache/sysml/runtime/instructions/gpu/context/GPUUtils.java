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

import static jcuda.jcublas.cublasOperation.CUBLAS_OP_T;

import jcuda.Pointer;
import jcuda.jcublas.JCublas2;

import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;
import org.apache.sysml.runtime.DMLRuntimeException;
import org.apache.sysml.runtime.controlprogram.caching.MatrixObject;

public class GPUUtils {
	private static final Log LOG = LogFactory.getLog(GPUUtils.class.getName());
	
	/**
	 * Transposes a dense matrix on the GPU by calling the cublasDgeam operation
	 *
	 * @param gCtx     a valid {@link GPUContext}
	 * @param densePtr Pointer to dense matrix on the GPU
	 * @param m        rows in ouput matrix
	 * @param n        columns in output matrix
	 * @param lda      rows in input matrix
	 * @param ldc      columns in output matrix
	 * @throws DMLRuntimeException if operation failed
	 * @return transposed matrix
	 */
	public static Pointer transpose(GPUContext gCtx, Pointer densePtr, int m, int n, int lda, int ldc)
			throws DMLRuntimeException {
		LOG.trace("GPU : transpose of block of size [" + m + "," + n + "]" + ", GPUContext=" + gCtx);
		Pointer alpha = Pointer.to(new double[] { 1.0 });
		Pointer beta = Pointer.to(new double[] { 0.0 });
		Pointer A = densePtr;
		Pointer C = gCtx.allocate(((long) m) * getDoubleSizeOf(n));

		// Transpose the matrix to get a dense matrix
		JCublas2.cublasDgeam(gCtx.getCublasHandle(), CUBLAS_OP_T, CUBLAS_OP_T, m, n, alpha, A, lda, beta, new Pointer(),
				lda, C, ldc);
		return C;
	}
	
	public static long getDoubleSizeOf(long numElems) {
		return numElems * ((long) jcuda.Sizeof.DOUBLE);
	}
	
	public static long getIntSizeOf(long numElems) {
		return numElems * ((long) jcuda.Sizeof.INT);
	}
	
	/**
	 * Creates a new MatrixObject and a new GPUObject but uses the same GPUPointer.
	 * This method is useful in update in-place operations.
	 * 
	 * @param gCtx gpu context
	 * @param src input matrix object
	 * @param dest output matrix object
	 * @return shallow copied matrix object
	 * @throws DMLRuntimeException if error occurs
	 */
	public static MatrixObject shallowCopy(GPUContext gCtx, MatrixObject src, MatrixObject dest) throws DMLRuntimeException {
		dest.setMetaData(src.getMetaData());
		GPUObject gpuObj = src.getGPUObject(gCtx).shallowCopy(dest);
		gpuObj.resetLocks();
		dest.setGPUObject(gCtx, gpuObj);
		return dest;
	}
	
	/**
	 * Creates a new MatrixObject, a new GPUObject and a new GPUPointer.
	 * The content of old GPUPointer is copied to the new GPUPointer.
	 * 
	 * @param gCtx gpu context
	 * @param src input matrix object
	 * @param dest output matrix object
	 * @return shallow copied matrix object
	 * @throws DMLRuntimeException if error occurs
	 */
	public static MatrixObject deepCopy(GPUContext gCtx, MatrixObject src, MatrixObject dest) throws DMLRuntimeException {
		dest.setMetaData(src.getMetaData());
		GPUObject gpuObj = src.getGPUObject(gCtx).deepCopy(dest);
		gpuObj.resetLocks();
		dest.setGPUObject(gCtx, gpuObj);
		return dest;
	}
}
