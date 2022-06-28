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
package org.apache.sysds.runtime.matrix.data;

import static jcuda.runtime.JCuda.cudaMemcpy;
import static jcuda.runtime.cudaMemcpyKind.cudaMemcpyDeviceToHost;
import static jcuda.runtime.cudaMemcpyKind.cudaMemcpyHostToDevice;
import static org.apache.sysds.runtime.matrix.data.LibMatrixCUDA.getCusparseHandle;

import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;
import org.apache.sysds.runtime.DMLRuntimeException;
import org.apache.sysds.runtime.instructions.gpu.context.GPUContext;

import jcuda.Pointer;
import jcuda.Sizeof;
import jcuda.jcublas.JCublas2;
import jcuda.jcublas.cublasHandle;
import jcuda.jcusolver.JCusolverDn;
import jcuda.jcusolver.cusolverDnHandle;
import jcuda.jcusparse.JCusparse;
import jcuda.jcusparse.cusparseHandle;
import jcuda.jcusparse.cusparseMatDescr;

public class DoublePrecisionCudaSupportFunctions implements CudaSupportFunctions {

	private static final Log LOG = LogFactory.getLog(DoublePrecisionCudaSupportFunctions.class.getName());

	@Override
	public void cublasgeam(cublasHandle handle, int transa, int transb, int m, int n, Pointer alpha, Pointer A, int lda,
			Pointer beta, Pointer B, int ldb, Pointer C, int ldc) {
		JCublas2.cublasDgeam(handle, transa, transb, m, n, alpha, A, lda, beta, B, ldb, C, ldc);
	}
	
	@Override
	public void cublasdot(cublasHandle handle, int n, Pointer x, int incx, Pointer y, int incy, Pointer result) {
		JCublas2.cublasDdot(handle, n, x, incx, y, incy, result);
	}
	
	@Override
	public int cublasgemv(cublasHandle handle, int trans, int m, int n, Pointer alpha, Pointer A, int lda, Pointer x,
			int incx, Pointer beta, Pointer y, int incy) {
		return JCublas2.cublasDgemv(handle, trans, m, n, alpha, A, lda, x, incx, beta, y, incy);
	}
	
	@Override
	public int cublasgemm(cublasHandle handle, int transa, int transb, int m, int n, int k, Pointer alpha, Pointer A,
			int lda, Pointer B, int ldb, Pointer beta, Pointer C, int ldc) {
		return JCublas2.cublasDgemm(handle, transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
	}

	@Override
	public int cublassyrk(cublasHandle handle, int uplo, int trans, int n, int k, Pointer alpha, Pointer A, int lda,
			Pointer beta, Pointer C, int ldc) {
		return JCublas2.cublasDsyrk(handle, uplo, trans, n, k, alpha, A, lda, beta, C, ldc);
	}
	
	@Override
	public int cublasaxpy(cublasHandle handle, int n, Pointer alpha, Pointer x, int incx, Pointer y, int incy) {
		return JCublas2.cublasDaxpy(handle, n, alpha, x, incx, y, incy);
	}
	
	@Override
	public int cublastrsm(cublasHandle handle, int side, int uplo, int trans, int diag, int m, int n, Pointer alpha,
			Pointer A, int lda, Pointer B, int ldb) {
		return JCublas2.cublasDtrsm(handle, side, uplo, trans, diag, m, n, alpha, A, lda, B, ldb);
	}

	@Override
	public int cusolverDngeqrf_bufferSize(cusolverDnHandle handle, int m, int n, Pointer A, int lda, int[] Lwork) {
		return JCusolverDn.cusolverDnDgeqrf_bufferSize(handle, m, n, A, lda, Lwork);
	}
	
	@Override
	public int cusolverDngeqrf(cusolverDnHandle handle, int m, int n, Pointer A, int lda, Pointer TAU,
			Pointer Workspace, int Lwork, Pointer devInfo) {
		return JCusolverDn.cusolverDnDgeqrf(handle, m, n, A, lda, TAU, Workspace, Lwork, devInfo);
	}

	@Override
	public int cusolverDnormqr(cusolverDnHandle handle, int side, int trans, int m, int n, int k, Pointer A, int lda,
			Pointer tau, Pointer C, int ldc, Pointer work, int lwork, Pointer devInfo) {
		return JCusolverDn.cusolverDnDormqr(handle, side, trans, m, n, k, A, lda, tau, C, ldc, work, lwork, devInfo);
	}

	@Override
	public int cusparsecsrgeam(GPUContext gCtx, int m, int n, Pointer alpha, cusparseMatDescr descrA, int nnzA,
							   Pointer csrValA, Pointer csrRowPtrA, Pointer csrColIndA, Pointer beta, cusparseMatDescr descrB, int nnzB,
							   Pointer csrValB, Pointer csrRowPtrB, Pointer csrColIndB, cusparseMatDescr descrC, Pointer csrValC,
							   Pointer csrRowPtrC, Pointer csrColIndC) {
		long[] bufferSize = { -1 };

		JCusparse.cusparseDcsrgeam2_bufferSizeExt(getCusparseHandle(gCtx), m, n, alpha, descrA, nnzA,
				csrValA, csrRowPtrA, csrColIndA, beta, descrB, nnzB,
				csrValB, csrRowPtrB, csrColIndB, descrC, csrValC, csrRowPtrC, csrColIndC, bufferSize);

		Pointer buf1 = gCtx.allocate("", bufferSize[0]);

		return JCusparse.cusparseDcsrgeam2(getCusparseHandle(gCtx), m, n, alpha, descrA, nnzA,
				csrValA, csrRowPtrA, csrColIndA, beta, descrB, nnzB,
				csrValB, csrRowPtrB, csrColIndB, descrC, csrValC, csrRowPtrC, csrColIndC, buf1);
	}

	@Override
	public int cusparsecsr2dense(cusparseHandle handle, int m, int n, cusparseMatDescr descrA, Pointer csrValA,
			Pointer csrRowPtrA, Pointer csrColIndA, Pointer A, int lda) {
		return JCusparse.cusparseDcsr2dense(handle, m, n, descrA, csrValA, csrRowPtrA, csrColIndA, A, lda);
	}

	@Override
	public int cusparsedense2csr(cusparseHandle handle, int m, int n, cusparseMatDescr descrA, Pointer A, int lda,
			Pointer nnzPerRow, Pointer csrValA, Pointer csrRowPtrA, Pointer csrColIndA) {
		return JCusparse.cusparseDdense2csr(handle, m, n, descrA, A, lda, nnzPerRow, csrValA, csrRowPtrA, csrColIndA);
	}

	@Override
	public int cusparsennz(cusparseHandle handle, int dirA, int m, int n, cusparseMatDescr descrA, Pointer A, int lda,
			Pointer nnzPerRowCol, Pointer nnzTotalDevHostPtr) {
		return JCusparse.cusparseDnnz(handle, dirA, m, n, descrA, A, lda, nnzPerRowCol, nnzTotalDevHostPtr);
	}

	@Override
	public void deviceToHost(GPUContext gCtx, Pointer src, double[] dest, String instName, boolean isEviction) {
		// ToDo: stats
		if(src == null)
			throw new DMLRuntimeException("The source pointer in deviceToHost is null");
		if(dest == null)
			throw new DMLRuntimeException("The destination array in deviceToHost is null");
		if(LOG.isDebugEnabled()) {
			LOG.debug("deviceToHost: src of size " + gCtx.getMemoryManager().getSizeAllocatedGPUPointer(src) + " (in bytes) -> dest of size " + (dest.length*Double.BYTES)  + " (in bytes).");
		}
		cudaMemcpy(Pointer.to(dest), src, ((long)dest.length)*Sizeof.DOUBLE, cudaMemcpyDeviceToHost);
	}

	@Override
	public void hostToDevice(GPUContext gCtx, double[] src, Pointer dest, String instName) {
		// ToDo: stats
		cudaMemcpy(dest, Pointer.to(src), ((long)src.length)*Sizeof.DOUBLE, cudaMemcpyHostToDevice);
	}
}
