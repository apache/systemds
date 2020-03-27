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

import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.FloatBuffer;
import java.util.stream.IntStream;

import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;
import org.apache.sysds.api.DMLScript;
import org.apache.sysds.runtime.instructions.gpu.context.GPUContext;
import org.apache.sysds.utils.GPUStatistics;

import jcuda.Pointer;
import jcuda.Sizeof;
import jcuda.jcublas.JCublas2;
import jcuda.jcublas.cublasHandle;
import jcuda.jcusolver.JCusolverDn;
import jcuda.jcusolver.cusolverDnHandle;
import jcuda.jcusparse.JCusparse;
import jcuda.jcusparse.cusparseHandle;
import jcuda.jcusparse.cusparseMatDescr;

public class SinglePrecisionCudaSupportFunctions implements CudaSupportFunctions {
	
	private static final Log LOG = LogFactory.getLog(SinglePrecisionCudaSupportFunctions.class.getName());

	@Override
	public int cusparsecsrgemm(cusparseHandle handle, int transA, int transB, int m, int n, int k,
			cusparseMatDescr descrA, int nnzA, Pointer csrValA, Pointer csrRowPtrA, Pointer csrColIndA,
			cusparseMatDescr descrB, int nnzB, Pointer csrValB, Pointer csrRowPtrB, Pointer csrColIndB,
			cusparseMatDescr descrC, Pointer csrValC, Pointer csrRowPtrC, Pointer csrColIndC) {
		return JCusparse.cusparseScsrgemm(handle, transA,  transB,  m,  n,  k,
				 descrA,  nnzA,  csrValA,  csrRowPtrA,  csrColIndA,
				 descrB,  nnzB,  csrValB,  csrRowPtrB,  csrColIndB,
				 descrC,  csrValC,  csrRowPtrC,  csrColIndC);
	}

	@Override
	public int cublasgeam(cublasHandle handle, int transa, int transb, int m, int n, Pointer alpha, Pointer A, int lda,
			Pointer beta, Pointer B, int ldb, Pointer C, int ldc) {
		return JCublas2.cublasSgeam(handle, transa, transb, m, n, alpha, A, lda, beta, B, ldb, C, ldc);
	}

	@Override
	public int cusparsecsrmv(cusparseHandle handle, int transA, int m, int n, int nnz, Pointer alpha,
			cusparseMatDescr descrA, Pointer csrValA, Pointer csrRowPtrA, Pointer csrColIndA, Pointer x, Pointer beta,
			Pointer y) {
		return JCusparse.cusparseScsrmv(handle, transA, m, n, nnz, alpha, 
				descrA, csrValA, csrRowPtrA, csrColIndA, x, beta, y);
	}
	
	@Override
	public int	cusparsecsrmm2(cusparseHandle handle, int transa, int transb, int m, int n, int k, int nnz, jcuda.Pointer alpha, cusparseMatDescr descrA, 
			jcuda.Pointer csrValA, jcuda.Pointer csrRowPtrA, jcuda.Pointer csrColIndA, 
			jcuda.Pointer B, int ldb, jcuda.Pointer beta, jcuda.Pointer C, int ldc) {
		return JCusparse.cusparseScsrmm2(handle, transa, transb, m, n, k, nnz, alpha, descrA, csrValA, 
				csrRowPtrA, csrColIndA, B, ldb, beta, C, ldc);
	}

	@Override
	public int cublasdot(cublasHandle handle, int n, Pointer x, int incx, Pointer y, int incy, Pointer result) {
		return JCublas2.cublasSdot(handle, n, x, incx, y, incy, result);
	}

	@Override
	public int cublasgemv(cublasHandle handle, int trans, int m, int n, Pointer alpha, Pointer A, int lda, Pointer x,
			int incx, Pointer beta, Pointer y, int incy) {
		return JCublas2.cublasSgemv(handle, trans, m, n, alpha, A, lda, x, incx, beta, y, incy);
	}

	@Override
	public int cublasgemm(cublasHandle handle, int transa, int transb, int m, int n, int k, Pointer alpha, Pointer A,
			int lda, Pointer B, int ldb, Pointer beta, Pointer C, int ldc) {
		return JCublas2.cublasSgemm(handle, transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
	}

	@Override
	public int cusparsecsr2csc(cusparseHandle handle, int m, int n, int nnz, Pointer csrVal, Pointer csrRowPtr,
			Pointer csrColInd, Pointer cscVal, Pointer cscRowInd, Pointer cscColPtr, int copyValues, int idxBase) {
		return JCusparse.cusparseScsr2csc(handle, m, n, nnz, csrVal, csrRowPtr, csrColInd, cscVal, cscRowInd, cscColPtr, copyValues, idxBase);
	}

	@Override
	public int cublassyrk(cublasHandle handle, int uplo, int trans, int n, int k, Pointer alpha, Pointer A, int lda,
			Pointer beta, Pointer C, int ldc) {
		return JCublas2.cublasSsyrk(handle, uplo, trans, n, k, alpha, A, lda, beta, C, ldc);
	}

	@Override
	public int cublasaxpy(cublasHandle handle, int n, Pointer alpha, Pointer x, int incx, Pointer y, int incy) {
		return JCublas2.cublasSaxpy(handle, n, alpha, x, incx, y, incy);
	}

	@Override
	public int cublastrsm(cublasHandle handle, int side, int uplo, int trans, int diag, int m, int n, Pointer alpha,
			Pointer A, int lda, Pointer B, int ldb) {
		return JCublas2.cublasStrsm(handle, side, uplo, trans, diag, m, n, alpha, A, lda, B, ldb);
	}

	@Override
	public int cusolverDngeqrf_bufferSize(cusolverDnHandle handle, int m, int n, Pointer A, int lda, int[] Lwork) {
		return JCusolverDn.cusolverDnSgeqrf_bufferSize(handle, m, n, A, lda, Lwork);
	}

	@Override
	public int cusolverDngeqrf(cusolverDnHandle handle, int m, int n, Pointer A, int lda, Pointer TAU,
			Pointer Workspace, int Lwork, Pointer devInfo) {
		return JCusolverDn.cusolverDnSgeqrf(handle, m, n, A, lda, TAU, Workspace, Lwork, devInfo);
	}
	
	@Override
	public int cusolverDnormqr(cusolverDnHandle handle, int side, int trans, int m, int n, int k, Pointer A, int lda,
			Pointer tau, Pointer C, int ldc, Pointer work, int lwork, Pointer devInfo) {
		return JCusolverDn.cusolverDnSormqr(handle, side, trans, m, n, k, A, lda, tau, C, ldc, work, lwork, devInfo);
	}

	@Override
	public int cusparsecsrgeam(cusparseHandle handle, int m, int n, Pointer alpha, cusparseMatDescr descrA, int nnzA,
			Pointer csrValA, Pointer csrRowPtrA, Pointer csrColIndA, Pointer beta, cusparseMatDescr descrB, int nnzB,
			Pointer csrValB, Pointer csrRowPtrB, Pointer csrColIndB, cusparseMatDescr descrC, Pointer csrValC,
			Pointer csrRowPtrC, Pointer csrColIndC) {
		return JCusparse.cusparseScsrgeam(handle, m, n, alpha, descrA, nnzA, 
				csrValA, csrRowPtrA, csrColIndA, beta, descrB, nnzB, 
				csrValB, csrRowPtrB, csrColIndB, descrC, csrValC, csrRowPtrC, csrColIndC);
	}

	@Override
	public int cusparsecsr2dense(cusparseHandle handle, int m, int n, cusparseMatDescr descrA, Pointer csrValA,
			Pointer csrRowPtrA, Pointer csrColIndA, Pointer A, int lda) {
		return JCusparse.cusparseScsr2dense(handle, m, n, descrA, csrValA, csrRowPtrA, csrColIndA, A, lda);
	}
	
	@Override
	public int cusparsedense2csr(cusparseHandle handle, int m, int n, cusparseMatDescr descrA, Pointer A, int lda,
			Pointer nnzPerRow, Pointer csrValA, Pointer csrRowPtrA, Pointer csrColIndA) {
		return JCusparse.cusparseSdense2csr(handle, m, n, descrA, A, lda, nnzPerRow, csrValA, csrRowPtrA, csrColIndA);
	}
	
	@Override
	public int cusparsennz(cusparseHandle handle, int dirA, int m, int n, cusparseMatDescr descrA, Pointer A, int lda,
			Pointer nnzPerRowCol, Pointer nnzTotalDevHostPtr) {
		return JCusparse.cusparseSnnz(handle, dirA, m, n, descrA, A, lda, nnzPerRowCol, nnzTotalDevHostPtr);
	}
	
	@Override
	public void deviceToHost(GPUContext gCtx, Pointer src, double[] dest, String instName, boolean isEviction) {
		long t0 = DMLScript.STATISTICS ? System.nanoTime() : 0;
		// We invoke transfer matrix from device to host in two cases:
		// 1. During eviction of unlocked matrices
		// 2. During acquireHostRead
		// 
		// If the single-precision support is enabled, then float-to-double conversion is required as CP expects the data to be in double format. 
		// This conversion can be done on host or on device. We typically prefer to do this conversion on device due to GPU's high-memory bandwidth. 
		// However, the conversion requires an additional space to be allocated for the conversion, which can lead to infinite recursion 
		// during eviction: `evict -> devictToHost -> float2double -> allocate -> ensureFreeSpace -> evict`. 
		// To avoid this recursion, it is necessary to perform this conversion in host.
		if(PERFORM_CONVERSION_ON_DEVICE && !isEviction) {
			Pointer deviceDoubleData = gCtx.allocate(instName, ((long)dest.length)*Sizeof.DOUBLE);
			LibMatrixCUDA.float2double(gCtx, src, deviceDoubleData, dest.length);
			cudaMemcpy(Pointer.to(dest), deviceDoubleData, ((long)dest.length)*Sizeof.DOUBLE, cudaMemcpyDeviceToHost);
			gCtx.cudaFreeHelper(instName, deviceDoubleData, DMLScript.EAGER_CUDA_FREE);
		}
		else {
			LOG.debug("Potential OOM: Allocated additional space on host in deviceToHost");
			FloatBuffer floatData = ByteBuffer.allocateDirect(Sizeof.FLOAT*dest.length).order(ByteOrder.nativeOrder()).asFloatBuffer();
			cudaMemcpy(Pointer.to(floatData), src, ((long)dest.length)*Sizeof.FLOAT, cudaMemcpyDeviceToHost);
			LibMatrixNative.fromFloatBuffer(floatData, dest);
		}
		if(DMLScript.STATISTICS) {
			long totalTime = System.nanoTime() - t0;
			GPUStatistics.cudaFloat2DoubleTime.add(totalTime);
			GPUStatistics.cudaFloat2DoubleCount.add(1);
		}
	}

	@Override
	public void hostToDevice(GPUContext gCtx, double[] src, Pointer dest, String instName) {
		LOG.debug("Potential OOM: Allocated additional space in hostToDevice");
		// TODO: Perform conversion on GPU using double2float and float2double kernels
		long t0 = DMLScript.STATISTICS ? System.nanoTime() : 0;
		if(PERFORM_CONVERSION_ON_DEVICE) {
			Pointer deviceDoubleData = gCtx.allocate(instName, ((long)src.length)*Sizeof.DOUBLE);
			cudaMemcpy(deviceDoubleData, Pointer.to(src), ((long)src.length)*Sizeof.DOUBLE, cudaMemcpyHostToDevice);
			LibMatrixCUDA.double2float(gCtx, deviceDoubleData, dest, src.length);
			gCtx.cudaFreeHelper(instName, deviceDoubleData, DMLScript.EAGER_CUDA_FREE);
		}
		else {
			FloatBuffer floatData = ByteBuffer.allocateDirect(Sizeof.FLOAT*src.length).order(ByteOrder.nativeOrder()).asFloatBuffer();
			IntStream.range(0, src.length).parallel().forEach(i -> floatData.put(i, (float)src[i]));
			cudaMemcpy(dest, Pointer.to(floatData), ((long)src.length)*Sizeof.FLOAT, cudaMemcpyHostToDevice);
		}
		
		if(DMLScript.STATISTICS) {
			long totalTime = System.nanoTime() - t0;
			GPUStatistics.cudaDouble2FloatTime.add(totalTime);
			GPUStatistics.cudaDouble2FloatCount.add(1);
		}
	}
}
