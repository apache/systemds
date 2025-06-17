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

import jcuda.jcublas.cublasHandle;
import jcuda.jcusolver.cusolverDnHandle;
import jcuda.jcusparse.cusparseHandle;
import jcuda.jcusparse.cusparseMatDescr;

import jcuda.jcusparse.cusparseSpGEMMDescr;
import jcuda.jcusparse.cusparseSpMatDescr;
import org.apache.sysds.runtime.instructions.gpu.context.GPUContext;

import jcuda.Pointer;

/**
 * DESIGN DOCUMENTATION FOR SUPPORTING LOWER PRECISION:
 * 1. SystemDS.cu has been templatized in following way to support different datatype:
 * - Similar to CuBLAS and CuSPARSE, the global kernels have the datatype specification in their name (for example: f for float
 * and d for datatpe). But unlike CuBLAS and CuSPARSE, these are suffixes so as to simplify the engine.  
 * - The global kernels with datatype specification invoke a corresponding templatized kernel (without suffix) which contains the core logic.
 * - The suffixes are added in JCudaKernels's launchKernel method before invocation.
 * For example:
 * <code>
 * template &lt; typename T &gt;
 * __device__ void matrix_atan(T *A, T *C, unsigned int size) {
 *     int index = blockIdx.x * blockDim.x + threadIdx.x;
 *     if (index &lt; size){
 *         C[index] = atan(A[index]);
 *     }
 * }
 * extern "C" __global__ void matrix_atand(double *A, double *C, unsigned int size) {
 * 	matrix_atan(A, C, size);
 * }
 * extern "C" __global__ void matrix_atanf(float *A, float *C, unsigned int size) {
 * 	matrix_atan(A, C, size);
 * } 
 * </code>
 * 
 * 2. The CUDA library calls (such as CuBLAS, CuSPARSE, etc) go through this interface.
 * The naming and parameters of the methods in this class are consistent with that of CUDA library to simplify development.
 * 
 * 3. During SystemDS initialization, the appropriate class implementing CudaKernels interface is set based on the configuration property sysds.dataType.
 */
public interface CudaSupportFunctions {
	public static boolean PERFORM_CONVERSION_ON_DEVICE = true;

	int cusparsecsrgemm(cusparseHandle handle, int transA, int transB, int alg, cusparseSpMatDescr spMatDescrA,
		cusparseSpMatDescr spMatDescrB, cusparseSpMatDescr spMatDescrC, cusparseSpGEMMDescr spgemmDescr);

	int cublasgeam(cublasHandle handle, int transa, int transb, int m, int n, jcuda.Pointer alpha,
		jcuda.Pointer A, int lda, jcuda.Pointer beta, jcuda.Pointer B, int ldb, jcuda.Pointer C, int ldc);
	public int	cusparsecsrmv(cusparseHandle handle, int transA, int m, int n, int nnz, jcuda.Pointer alpha, cusparseMatDescr descrA, jcuda.Pointer csrValA, jcuda.Pointer csrRowPtrA, jcuda.Pointer csrColIndA, 
			jcuda.Pointer x, jcuda.Pointer beta, jcuda.Pointer y);
	public int	cusparsecsrmm2(cusparseHandle handle, int transa, int transb, int m, int n, int k, int nnz, jcuda.Pointer alpha, cusparseMatDescr descrA, jcuda.Pointer csrValA, jcuda.Pointer csrRowPtrA, jcuda.Pointer csrColIndA, 
			jcuda.Pointer B, int ldb, jcuda.Pointer beta, jcuda.Pointer C, int ldc);
	public int cublasdot(cublasHandle handle, int n, jcuda.Pointer x, int incx, jcuda.Pointer y, int incy, jcuda.Pointer result);
	public int cublasgemv(cublasHandle handle, int trans, int m, int n, jcuda.Pointer alpha, jcuda.Pointer A, int lda, jcuda.Pointer x, int incx, jcuda.Pointer beta, jcuda.Pointer y, int incy);
	public int cublasgemm(cublasHandle handle, int transa, int transb, int m, int n, int k, jcuda.Pointer alpha, jcuda.Pointer A, int lda, jcuda.Pointer B, int ldb, jcuda.Pointer beta, jcuda.Pointer C, int ldc);
	public int cusparsecsr2csc(cusparseHandle handle, int m, int n, int nnz, jcuda.Pointer csrVal, jcuda.Pointer csrRowPtr, jcuda.Pointer csrColInd, jcuda.Pointer cscVal, jcuda.Pointer cscRowInd, jcuda.Pointer cscColPtr, int copyValues, int idxBase);
	public int cublassyrk(cublasHandle handle, int uplo, int trans, int n, int k, jcuda.Pointer alpha, jcuda.Pointer A, int lda, jcuda.Pointer beta, jcuda.Pointer C, int ldc);
	public int cublasaxpy(cublasHandle handle, int n, jcuda.Pointer alpha, jcuda.Pointer x, int incx, jcuda.Pointer y, int incy);
	public int cublastrsm(cublasHandle handle, int side, int uplo, int trans, int diag, int m, int n, jcuda.Pointer alpha, jcuda.Pointer A, int lda, jcuda.Pointer B, int ldb);
	public int cusolverDngeqrf_bufferSize(cusolverDnHandle handle, int m, int n, Pointer A, int lda, int[] Lwork);
	public int cusolverDngeqrf(cusolverDnHandle handle, int m, int n, Pointer A, int lda, Pointer TAU, Pointer Workspace, int Lwork, Pointer devInfo);
	public int cusolverDnormqr(cusolverDnHandle handle, int side, int trans, int m, int n, int k, Pointer A, int lda, Pointer tau, Pointer C, int ldc, Pointer work, int lwork, Pointer devInfo);
	public int cusparsecsrgeam(cusparseHandle handle, int m, int n, jcuda.Pointer alpha, cusparseMatDescr descrA, int nnzA, jcuda.Pointer csrValA, jcuda.Pointer csrRowPtrA, jcuda.Pointer csrColIndA, jcuda.Pointer beta, cusparseMatDescr descrB, int nnzB, jcuda.Pointer csrValB, jcuda.Pointer csrRowPtrB, jcuda.Pointer csrColIndB, cusparseMatDescr descrC, jcuda.Pointer csrValC, jcuda.Pointer csrRowPtrC, jcuda.Pointer csrColIndC);
	public int cusparsecsr2dense(cusparseHandle handle, int m, int n, cusparseMatDescr descrA, jcuda.Pointer csrValA, jcuda.Pointer csrRowPtrA, jcuda.Pointer csrColIndA, jcuda.Pointer A, int lda, long nnz) ;
	public int cusparsedense2csr(cusparseHandle handle, int m, int n, cusparseMatDescr descrA, jcuda.Pointer A, int lda, jcuda.Pointer nnzPerRow, jcuda.Pointer csrValA, jcuda.Pointer csrRowPtrA, jcuda.Pointer csrColIndA);
	public int cusparsennz(cusparseHandle handle, int dirA, int m, int n, cusparseMatDescr descrA, jcuda.Pointer A, int lda, jcuda.Pointer nnzPerRowCol, jcuda.Pointer nnzTotalDevHostPtr);
	public void deviceToHost(GPUContext gCtx, Pointer src, double [] dest, String instName, boolean isEviction);
	public void hostToDevice(GPUContext gCtx, double [] src,  Pointer dest, String instName);
	
}
