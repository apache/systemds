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

import static jcuda.cudaDataType.CUDA_R_64F;
import static jcuda.jcusparse.JCusparse.*;
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
import jcuda.jcusparse.cusparseSpMatDescr;
import jcuda.jcusparse.cusparseSpGEMMDescr;

import static jcuda.jcusparse.cusparseIndexType.CUSPARSE_INDEX_32I;
import static jcuda.jcusparse.cusparseIndexBase.CUSPARSE_INDEX_BASE_ZERO;
import static jcuda.cudaDataType.CUDA_R_32F;
import static jcuda.jcusparse.cusparseSpGEMMAlg.CUSPARSE_SPGEMM_DEFAULT;
import static jcuda.jcusparse.cusparseStatus.CUSPARSE_STATUS_SUCCESS;
import static jcuda.jcusparse.cusparseSpMVAlg.CUSPARSE_SPMV_ALG_DEFAULT;
import static jcuda.jcusparse.cusparseOperation.CUSPARSE_OPERATION_NON_TRANSPOSE;
import static jcuda.jcusparse.cusparseOrder.CUSPARSE_ORDER_COL;
import static jcuda.jcusparse.cusparseSpMMAlg.CUSPARSE_SPMM_ALG_DEFAULT;
import static jcuda.jcusparse.cusparseCsr2CscAlg.CUSPARSE_CSR2CSC_ALG1;
import static jcuda.jcusparse.cusparseSparseToDenseAlg.CUSPARSE_SPARSETODENSE_ALG_DEFAULT;
import static jcuda.jcusparse.cusparseIndexBase.CUSPARSE_INDEX_BASE_ONE;
import static jcuda.jcusparse.cusparseDenseToSparseAlg.CUSPARSE_DENSETOSPARSE_ALG_DEFAULT;
import static jcuda.runtime.JCuda.cudaMalloc;
import static jcuda.runtime.JCuda.cudaFree;

import jcuda.jcusparse.cusparseDnVecDescr;
import jcuda.jcusparse.cusparseDnMatDescr;

public class SinglePrecisionCudaSupportFunctions implements CudaSupportFunctions {

	private static final Log LOG = LogFactory.getLog(SinglePrecisionCudaSupportFunctions.class.getName());

	@Override
	public int cusparsecsrgemm(cusparseHandle handle, int transA, int transB, int alg, cusparseSpMatDescr spMatDescrA,
		cusparseSpMatDescr spMatDescrB, cusparseSpMatDescr spMatDescrC, cusparseSpGEMMDescr spgemmDescr) {
		double[] alpha = {1.0}, beta = {0.0};
		Pointer alphaPtr = Pointer.to(alpha), betaPtr = Pointer.to(beta);
		return cusparseSpGEMM_copy(handle, transA, transB, alphaPtr, spMatDescrA.asConst(), spMatDescrB.asConst(),
			betaPtr, spMatDescrC, CUDA_R_32F, alg, spgemmDescr);
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
		/* ------------------------------------------------------------------ */
		/* Descriptors and workspace                                          */
		/* ------------------------------------------------------------------ */
		cusparseSpMatDescr matA = new cusparseSpMatDescr();
		cusparseDnVecDescr vecX = new cusparseDnVecDescr();
		cusparseDnVecDescr vecY = new cusparseDnVecDescr();
		Pointer dBuf = null;
		int status;

		try {
			/* 1. CSR matrix A (FP32) -------------------------------------- */
			JCusparse.cusparseCreateCsr(matA, m, n, nnz, csrRowPtrA, csrColIndA, csrValA, CUSPARSE_INDEX_32I,
				CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F);

			/* 2. Dense vectors X and Y (FP32) ------------------------------ */
			JCusparse.cusparseCreateDnVec(vecX, n, x, CUDA_R_32F);
			JCusparse.cusparseCreateDnVec(vecY, m, y, CUDA_R_32F);

			/* 3. Query workspace size ------------------------------------- */
			long[] bufSize = {0};
			status = JCusparse.cusparseSpMV_bufferSize(handle, transA, alpha, matA.asConst(), vecX.asConst(), beta,
				vecY, CUDA_R_32F, CUSPARSE_SPMV_ALG_DEFAULT, bufSize);
			if(status != CUSPARSE_STATUS_SUCCESS)
				return status;

			if(bufSize[0] > 0) {
				dBuf = new Pointer();
				cudaMalloc(dBuf, bufSize[0]);
			}

			/* 4. Perform SpMV -------------------------------------------- */
			status = JCusparse.cusparseSpMV(handle, transA, alpha, matA.asConst(), vecX.asConst(), beta, vecY,
				CUDA_R_32F, CUSPARSE_SPMV_ALG_DEFAULT, dBuf);

			return status;
		}
		finally {
			if(dBuf != null)
				cudaFree(dBuf);
			JCusparse.cusparseDestroyDnVec(vecX.asConst());
			JCusparse.cusparseDestroyDnVec(vecY.asConst());
			JCusparse.cusparseDestroySpMat(matA.asConst());
		}
	}

	@Override
	public int cusparsecsrmm2(cusparseHandle handle, int transA, int transB, int m, int n, int k, int nnz,
		Pointer alpha, cusparseMatDescr descrA, Pointer csrValA, Pointer csrRowPtrA, Pointer csrColIndA, Pointer B,
		int ldb, Pointer beta, Pointer C, int ldc) {
		/* ------------------------------------------------------------------ */
		/* Descriptors and workspace                                          */
		/* ------------------------------------------------------------------ */
		cusparseSpMatDescr matA = new cusparseSpMatDescr();
		cusparseDnMatDescr matB = new cusparseDnMatDescr();
		cusparseDnMatDescr matC = new cusparseDnMatDescr();
		Pointer dBuf = null;
		int status;

		try {
			/* 1. CSR matrix A (FP32) -------------------------------------- */
			JCusparse.cusparseCreateCsr(matA, m, k, nnz, csrRowPtrA, csrColIndA, csrValA, CUSPARSE_INDEX_32I,
				CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F);

			/* 2. Dense matrix B (column-major) ---------------------------- */
			int rowsB = (transB == CUSPARSE_OPERATION_NON_TRANSPOSE) ? k : n;
			int colsB = (transB == CUSPARSE_OPERATION_NON_TRANSPOSE) ? n : k;
			JCusparse.cusparseCreateDnMat(matB, rowsB, colsB, ldb, B, CUDA_R_32F, CUSPARSE_ORDER_COL);

			/* 3. Dense matrix C (output) ---------------------------------- */
			int rowsC = (transA == CUSPARSE_OPERATION_NON_TRANSPOSE) ? m : k;
			int colsC = colsB;   // always equals n
			JCusparse.cusparseCreateDnMat(matC, rowsC, colsC, ldc, C, CUDA_R_32F, CUSPARSE_ORDER_COL);

			/* 4. Query workspace size ------------------------------------- */
			long[] bufSize = {0};
			status = JCusparse.cusparseSpMM_bufferSize(handle, transA, transB, alpha, matA.asConst(), matB.asConst(),
				beta, matC, CUDA_R_32F, CUSPARSE_SPMM_ALG_DEFAULT, bufSize);
			if(status != CUSPARSE_STATUS_SUCCESS)
				return status;

			if(bufSize[0] > 0) {
				dBuf = new Pointer();
				cudaMalloc(dBuf, bufSize[0]);
			}

			/* 5. Execute SpMM -------------------------------------------- */
			status = JCusparse.cusparseSpMM(handle, transA, transB, alpha, matA.asConst(), matB.asConst(), beta, matC,
				CUDA_R_32F, CUSPARSE_SPMM_ALG_DEFAULT, dBuf);

			return status;
		}
		finally {
			if(dBuf != null)
				cudaFree(dBuf);
			JCusparse.cusparseDestroyDnMat(matB.asConst());
			JCusparse.cusparseDestroyDnMat(matC.asConst());
			JCusparse.cusparseDestroySpMat(matA.asConst());
		}
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
		final int alg = CUSPARSE_CSR2CSC_ALG1;		// Algorithm 1 is universally supported
		final int valType = CUDA_R_32F;				// single-precision

		/* ------------------------------------------------------------------ */
		/* 1. Query required workspace size                                   */
		/* ------------------------------------------------------------------ */
		long[] bufSize = {0};
		int status = JCusparse.cusparseCsr2cscEx2_bufferSize(handle, m, n, nnz, csrVal, csrRowPtr, csrColInd, cscVal,
			cscColPtr, cscRowInd, valType, copyValues, idxBase, alg, bufSize);
		if(status != CUSPARSE_STATUS_SUCCESS)
			return status;

		/* ------------------------------------------------------------------ */
		/* 2. Allocate workspace (if needed)                                  */
		/* ------------------------------------------------------------------ */
		Pointer buffer = null;
		if(bufSize[0] > 0) {
			buffer = new Pointer();
			cudaMalloc(buffer, bufSize[0]);
		}

		try {
			/* -------------------------------------------------------------- */
			/* 3. Perform CSR -> CSC conversion                                */
			/* -------------------------------------------------------------- */
			status = JCusparse.cusparseCsr2cscEx2(handle, m, n, nnz, csrVal, csrRowPtr, csrColInd, cscVal, cscColPtr,
				cscRowInd, valType, copyValues, idxBase, alg, buffer);

			return status;
		}
		finally {
			if(buffer != null)
				cudaFree(buffer);
		}
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
		/* ------------------------------------------------------------------ */
		/* 1. Query temporary-buffer size                                     */
		/* ------------------------------------------------------------------ */
		long[] bufSize = {0};

		int status = JCusparse.cusparseScsrgeam2_bufferSizeExt(handle, m, n, alpha, descrA, nnzA, csrValA, csrRowPtrA,
			csrColIndA, beta, descrB, nnzB, csrValB, csrRowPtrB, csrColIndB, descrC, csrValC, csrRowPtrC, csrColIndC,
			bufSize);
		if(status != CUSPARSE_STATUS_SUCCESS)
			return status;

		/* ------------------------------------------------------------------ */
		/* 2. Allocate workspace (if needed)                                  */
		/* ------------------------------------------------------------------ */
		Pointer buffer = null;
		if(bufSize[0] > 0) {
			buffer = new Pointer();
			cudaMalloc(buffer, bufSize[0]);
		}

		try {
			/* -------------------------------------------------------------- */
			/* 3. Perform C = α*A  +  β*B                                     */
			/* -------------------------------------------------------------- */
			status = JCusparse.cusparseScsrgeam2(handle, m, n, alpha, descrA, nnzA, csrValA, csrRowPtrA, csrColIndA,
				beta, descrB, nnzB, csrValB, csrRowPtrB, csrColIndB, descrC, csrValC, csrRowPtrC, csrColIndC, buffer);

			return status;   // propagate cuSPARSE return code
		}
		finally {
			/* -------------------------------------------------------------- */
			/* 4. Free workspace                                              */
			/* -------------------------------------------------------------- */
			if(buffer != null)
				cudaFree(buffer);
		}
	}

	@Override
	public int cusparsecsr2dense(cusparseHandle handle, int m, int n, cusparseMatDescr descrA, Pointer csrValA,
		Pointer csrRowPtrA, Pointer csrColIndA, Pointer A, int lda, long nnz) {

		// Get index base from legacy descriptor -> 0 or 1
		int idxBase = JCusparse.cusparseGetMatIndexBase(descrA);

		// Create generric sparse-matrix descriptor required by CUDA 12
		cusparseSpMatDescr spMatA = new cusparseSpMatDescr();

		// Build CSR descriptor
		cusparseCreateCsr(spMatA, m, n, nnz, csrRowPtrA, csrColIndA, csrValA, CUSPARSE_INDEX_32I,
			CUSPARSE_INDEX_32I, idxBase, CUDA_R_32F);

		// Build dense descriptor
		cusparseDnMatDescr matB = new cusparseDnMatDescr();
		cusparseCreateDnMat(matB, m, n, lda, A, CUDA_R_32F, CUSPARSE_ORDER_COL);

		// Determine buffer size
		int alg = CUSPARSE_SPARSETODENSE_ALG_DEFAULT;
		long[] bufSize = {0};
		cusparseSparseToDense_bufferSize(handle, spMatA.asConst(), matB, alg,
			bufSize);    //bufSize[0] now holds the exact byte count

		// Allocate scratch space of the requested size
		Pointer dBuffer = new Pointer();
		if(bufSize[0] > 0) {
			cudaMalloc(dBuffer, bufSize[0]);
		}
		try {
			// Write dense matrix
			int algSparseToDense = CUSPARSE_SPARSETODENSE_ALG_DEFAULT;
			return cusparseSparseToDense(handle, spMatA.asConst(), matB, algSparseToDense, dBuffer);
		}
		finally {
			if(bufSize[0] > 0)
				cudaFree(dBuffer);
			cusparseDestroyDnMat(matB.asConst());
			cusparseDestroySpMat(spMatA.asConst());
		}

	}

	@Override
	public int cusparsedense2csr(cusparseHandle handle, int m, int n, cusparseMatDescr descrA, Pointer A, int lda,
		Pointer nnzPerRow, Pointer csrValA, Pointer csrRowPtrA, Pointer csrColIndA) {
		/* ------------------------------------------------------------------ */
		/* 0.  Index base (0 or 1) comes from the descriptor                  */
		/* ------------------------------------------------------------------ */
		int idxBase = JCusparse.cusparseGetMatIndexBase(descrA);

		/* ------------------------------------------------------------------ */
		/* 1.  Create dense-matrix and CSR descriptors (FP32)                 */
		/* ------------------------------------------------------------------ */
		cusparseDnMatDescr matDense = new cusparseDnMatDescr();
		JCusparse.cusparseCreateDnMat(matDense, m, n, lda, A, CUDA_R_32F, CUSPARSE_ORDER_COL);

		cusparseSpMatDescr matCsr = new cusparseSpMatDescr();
		/* nnz initially 0 – cuSPARSE will fill it during analysis           */
		JCusparse.cusparseCreateCsr(matCsr, m, n, 0L, csrRowPtrA, csrColIndA, csrValA, CUSPARSE_INDEX_32I,
			CUSPARSE_INDEX_32I, idxBase, CUDA_R_32F);

		/* ------------------------------------------------------------------ */
		/* 2.  Query workspace size                                           */
		/* ------------------------------------------------------------------ */
		long[] bufSize = {0};
		int alg = CUSPARSE_DENSETOSPARSE_ALG_DEFAULT;

		int status = JCusparse.cusparseDenseToSparse_bufferSize(handle, matDense.asConst(), matCsr, alg, bufSize);
		if(status != CUSPARSE_STATUS_SUCCESS) {
			JCusparse.cusparseDestroySpMat(matCsr.asConst());
			JCusparse.cusparseDestroyDnMat(matDense.asConst());
			return status;
		}

		/* ------------------------------------------------------------------ */
		/* 3.  Allocate workspace (if required)                               */
		/* ------------------------------------------------------------------ */
		Pointer buffer = null;
		if(bufSize[0] > 0) {
			buffer = new Pointer();
			cudaMalloc(buffer, bufSize[0]);
		}

		try {
			/* -------------------------------------------------------------- */
			/* 4.  Phase-1: symbolic pass                                     */
			/* -------------------------------------------------------------- */
			status = JCusparse.cusparseDenseToSparse_analysis(handle, matDense.asConst(), matCsr, alg, buffer);
			if(status != CUSPARSE_STATUS_SUCCESS)
				return status;

			/* -------------------------------------------------------------- */
			/* 5.  Phase-2: numeric conversion                                */
			/* -------------------------------------------------------------- */
			status = JCusparse.cusparseDenseToSparse_convert(handle, matDense.asConst(), matCsr, alg, buffer);
			if(status != CUSPARSE_STATUS_SUCCESS)
				return status;

			return status;   // success
		}
		finally {
			/* -------------------------------------------------------------- */
			/* 7.  Cleanup                                                    */
			/* -------------------------------------------------------------- */
			if(buffer != null)
				cudaFree(buffer);
			JCusparse.cusparseDestroySpMat(matCsr.asConst());
			JCusparse.cusparseDestroyDnMat(matDense.asConst());
		}
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
			Pointer deviceDoubleData = gCtx.allocate(instName, ((long)dest.length)*Sizeof.DOUBLE, false);
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
			Pointer deviceDoubleData = gCtx.allocate(instName, ((long)src.length)*Sizeof.DOUBLE, false);
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
