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

import static jcuda.jcusparse.JCusparse.*;
import static jcuda.jcusparse.JCusparse.cusparseGetMatIndexBase;
import static jcuda.runtime.JCuda.*;
import static jcuda.runtime.cudaMemcpyKind.cudaMemcpyDeviceToHost;
import static jcuda.runtime.cudaMemcpyKind.cudaMemcpyHostToDevice;

import jcuda.jcusparse.cusparseSpMatDescr;
import jcuda.jcusparse.cusparseSpGEMMDescr;
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
import jcuda.jcusparse.cusparseDnVecDescr;
import jcuda.jcusparse.cusparseDnMatDescr;

import static jcuda.jcusparse.cusparseIndexType.CUSPARSE_INDEX_32I;
import static jcuda.jcusparse.cusparseIndexBase.CUSPARSE_INDEX_BASE_ZERO;
import static jcuda.cudaDataType.CUDA_R_64F;
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

public class DoublePrecisionCudaSupportFunctions implements CudaSupportFunctions {

	private static final Log LOG = LogFactory.getLog(DoublePrecisionCudaSupportFunctions.class.getName());

	@Override
	public int cusparsecsrgemm(cusparseHandle handle, int transA, int transB, int alg, cusparseSpMatDescr spMatDescrA,
		cusparseSpMatDescr spMatDescrB, cusparseSpMatDescr spMatDescrC, cusparseSpGEMMDescr spgemmDescr) {
		double[] alpha = {1.0}, beta = {0.0};
		Pointer alphaPtr = Pointer.to(alpha), betaPtr = Pointer.to(beta);
		return cusparseSpGEMM_copy(handle, transA, transB, alphaPtr, spMatDescrA.asConst(), spMatDescrB.asConst(),
			betaPtr, spMatDescrC, CUDA_R_64F, alg, spgemmDescr);
	}

	@Override
	public int cublasgeam(cublasHandle handle, int transa, int transb, int m, int n, Pointer alpha, Pointer A, int lda,
		Pointer beta, Pointer B, int ldb, Pointer C, int ldc) {
		return JCublas2.cublasDgeam(handle, transa, transb, m, n, alpha, A, lda, beta, B, ldb, C, ldc);
	}

	@Override
	public int cusparsecsrmv(cusparseHandle handle, int transA, int m, int n, int nnz, Pointer alpha,
		cusparseSpMatDescr spMatDescrA, cusparseMatDescr descrA, Pointer csrValA, Pointer csrRowPtrA,
		Pointer csrColIndA, Pointer x, Pointer beta, Pointer y) {
		// Create sparse matrix A in CSR format
		int idxBase = cusparseGetMatIndexBase(descrA);
		int dataType = CUDA_R_64F;
		cusparseCreateCsr(spMatDescrA, m, n, nnz, csrRowPtrA, csrColIndA, csrValA, CUSPARSE_INDEX_32I,
			CUSPARSE_INDEX_32I, idxBase, dataType);
		// Create dense vectors vecX and vecY
		cusparseDnVecDescr vecX = new cusparseDnVecDescr();
		cusparseDnVecDescr vecY = new cusparseDnVecDescr();
		cusparseCreateDnVec(vecX, n, x, dataType);
		cusparseCreateDnVec(vecY, m, y, dataType);
		// allocate an external buffer if needed
		long[] bufferSize = {0};
		int alg = CUSPARSE_SPMV_ALG_DEFAULT;
		cusparseSpMV_bufferSize(handle, transA, alpha, spMatDescrA.asConst(), vecX.asConst(), beta, vecY, dataType, alg,
			bufferSize);
		// execute SpMV
		Pointer dBuffer = new Pointer();
		if(bufferSize[0] > 0)
			cudaMalloc(dBuffer, bufferSize[0]);
		try {
			return cusparseSpMV(handle, transA, alpha, spMatDescrA.asConst(), vecX.asConst(), beta, vecY, dataType, alg,
				dBuffer);
		}
		finally {
			if(bufferSize[0] > 0)
				cudaFree(dBuffer);
			cusparseDestroyDnVec(vecX.asConst());
			cusparseDestroyDnVec(vecY.asConst());
		}
	}

	@Override
	public int cusparsecsrmm2(cusparseHandle handle, int transA, int transB, int m, int n, int k, int nnz,
		Pointer alpha, cusparseMatDescr descrA, Pointer csrValA, Pointer csrRowPtrA, Pointer csrColIndA, Pointer B,
		int ldb, Pointer beta, Pointer C, int ldc) {
		/* Descriptors and workspace -------------------------------------- */
		cusparseSpMatDescr matA = new cusparseSpMatDescr();
		cusparseDnMatDescr matB = new cusparseDnMatDescr();
		cusparseDnMatDescr matC = new cusparseDnMatDescr();
		Pointer dBuf = new Pointer();
		long dBufBytes = 0;
		int status;

		try {
			/* 1. CSR matrix A -------------------------------------------- */
			cusparseCreateCsr(matA, m, k, nnz, csrRowPtrA, csrColIndA, csrValA, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
				CUSPARSE_INDEX_BASE_ZERO, CUDA_R_64F);

			/* 2. Dense matrix B  (col-major layout) ---------------------- */
			int rowsB = (transB == CUSPARSE_OPERATION_NON_TRANSPOSE) ? k : n;
			int colsB = (transB == CUSPARSE_OPERATION_NON_TRANSPOSE) ? n : k;
			cusparseCreateDnMat(matB, rowsB, colsB, ldb, B, CUDA_R_64F, CUSPARSE_ORDER_COL);

			/* 3. Dense matrix C  (output) -------------------------------- */
			int rowsC = (transA == CUSPARSE_OPERATION_NON_TRANSPOSE) ? m : k;
			int colsC = colsB;                       // always equals n
			cusparseCreateDnMat(matC, rowsC, colsC, ldc, C, CUDA_R_64F, CUSPARSE_ORDER_COL);

			/* 4. Query workspace size ------------------------------------ */
			long[] bufSize = {0};
			status = JCusparse.cusparseSpMM_bufferSize(handle, transA, transB, alpha, matA.asConst(), matB.asConst(),
				beta, matC, CUDA_R_64F, CUSPARSE_SPMM_ALG_DEFAULT, bufSize);
			if(status != CUSPARSE_STATUS_SUCCESS)
				return status;

			dBufBytes = bufSize[0];
			if(dBufBytes > 0)
				cudaMalloc(dBuf, dBufBytes);

			/* 5. Execute SpMM ------------------------------------------- */
			status = JCusparse.cusparseSpMM(handle, transA, transB, alpha, matA.asConst(), matB.asConst(), beta, matC,
				CUDA_R_64F, CUSPARSE_SPMM_ALG_DEFAULT, dBuf);

			return status;
		}
		finally {
			/* Cleanup ---------------------------------------------------- */
			if(dBufBytes > 0)
				cudaFree(dBuf);
			JCusparse.cusparseDestroyDnMat(matB.asConst());
			JCusparse.cusparseDestroyDnMat(matC.asConst());
			JCusparse.cusparseDestroySpMat(matA.asConst());
		}
	}

	@Override
	public int cublasdot(cublasHandle handle, int n, Pointer x, int incx, Pointer y, int incy, Pointer result) {
		return JCublas2.cublasDdot(handle, n, x, incx, y, incy, result);
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
	public int cusparsecsr2csc(cusparseHandle handle, int m, int n, int nnz, Pointer csrVal, Pointer csrRowPtr,
		Pointer csrColInd, Pointer cscVal, Pointer cscRowInd, Pointer cscColPtr, int copyValues, int idxBase) {

		int valType = CUDA_R_64F;            // double precision
		int alg = CUSPARSE_CSR2CSC_ALG1;     // always supported

		long[] bufferSize = {0};
		cusparseCsr2cscEx2_bufferSize(handle, m, n, nnz, csrVal, csrRowPtr, csrColInd, cscVal, cscColPtr, cscRowInd,
			valType, copyValues, idxBase, alg, bufferSize);

		Pointer buffer = new Pointer();
		if(bufferSize[0] > 0)
			cudaMalloc(buffer, bufferSize[0]);
		try {
			return cusparseCsr2cscEx2(handle, m, n, nnz, csrVal, csrRowPtr, csrColInd, cscVal, cscColPtr, cscRowInd,
				valType, copyValues, idxBase, alg, buffer);
		}
		finally {
			if(bufferSize[0] > 0)
				cudaFree(buffer);
		}
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
	public int cusparsecsrgeam(cusparseHandle handle, int m, int n, Pointer alpha, cusparseMatDescr descrA, int nnzA,
		Pointer csrValA, Pointer csrRowPtrA, Pointer csrColIndA, Pointer beta, cusparseMatDescr descrB, int nnzB,
		Pointer csrValB, Pointer csrRowPtrB, Pointer csrColIndB, cusparseMatDescr descrC, Pointer csrValC,
		Pointer csrRowPtrC, Pointer csrColIndC) {

		long[] pBufferSizeInBytes = {0};

		int status = JCusparse.cusparseDcsrgeam2_bufferSizeExt(handle, m, n, alpha, descrA, nnzA, csrValA, csrRowPtrA,
			csrColIndA, beta, descrB, nnzB, csrValB, csrRowPtrB, csrColIndB, descrC, csrValC, csrRowPtrC, csrColIndC,
			pBufferSizeInBytes);
		if(status != CUSPARSE_STATUS_SUCCESS)
			return status;

		Pointer buffer = new Pointer();
		if(pBufferSizeInBytes[0] > 0)
			cudaMalloc(buffer, pBufferSizeInBytes[0]);

		try {
			// C = α*A + β*B
			return JCusparse.cusparseDcsrgeam2(handle, m, n, alpha, descrA, nnzA, csrValA, csrRowPtrA, csrColIndA, beta,
				descrB, nnzB, csrValB, csrRowPtrB, csrColIndB, descrC, csrValC, csrRowPtrC, csrColIndC, buffer);
		}
		finally {
			if(pBufferSizeInBytes[0] > 0)
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
			CUSPARSE_INDEX_32I, idxBase, CUDA_R_64F);

		// Build dense descriptor
		cusparseDnMatDescr matB = new cusparseDnMatDescr();
		cusparseCreateDnMat(matB, m, n, lda, A, CUDA_R_64F, CUSPARSE_ORDER_COL);

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
		/* 1. Determine index base and wrap the input/output in descriptors   */
		/* ------------------------------------------------------------------ */
		int idxBase = JCusparse.cusparseGetMatIndexBase(descrA);

		cusparseDnMatDescr matDense = new cusparseDnMatDescr();
		JCusparse.cusparseCreateDnMat(matDense, m, n, lda, A, CUDA_R_64F, CUSPARSE_ORDER_COL);

		cusparseSpMatDescr matCsr = new cusparseSpMatDescr();
		/* nnz initially 0 – cuSPARSE fills it during analysis phase */
		JCusparse.cusparseCreateCsr(matCsr, m, n, 0L, csrRowPtrA, csrColIndA, csrValA, CUSPARSE_INDEX_32I,
			CUSPARSE_INDEX_32I, idxBase, CUDA_R_64F);

		/* ------------------------------------------------------------------ */
		/* 2. Query temporary buffer size                                     */
		/* ------------------------------------------------------------------ */
		long[] bufSz = {0};
		int alg = CUSPARSE_DENSETOSPARSE_ALG_DEFAULT;

		int status = JCusparse.cusparseDenseToSparse_bufferSize(handle, matDense.asConst(), matCsr, alg, bufSz);
		if(status != CUSPARSE_STATUS_SUCCESS) {
			JCusparse.cusparseDestroySpMat(matCsr.asConst());
			JCusparse.cusparseDestroyDnMat(matDense.asConst());
			return status;
		}

		Pointer buffer = new Pointer();
		if(bufSz[0] > 0)
			cudaMalloc(buffer, bufSz[0]);

		try {
			/* -------------------------------------------------------------- */
			/* 3. Symbolic pass: decide sparsity pattern, fill csrRowPtrA     */
			/* -------------------------------------------------------------- */
			status = JCusparse.cusparseDenseToSparse_analysis(handle, matDense.asConst(), matCsr, alg, buffer);
			if(status != CUSPARSE_STATUS_SUCCESS)
				return status;

			/* -------------------------------------------------------------- */
			/* 4. Numeric conversion: fill csrColIndA and csrValA             */
			/* -------------------------------------------------------------- */
			status = JCusparse.cusparseDenseToSparse_convert(handle, matDense.asConst(), matCsr, alg, buffer);
			if(status != CUSPARSE_STATUS_SUCCESS)
				return status;

			return status;
		}
		finally {
			if(bufSz[0] > 0)
				cudaFree(buffer);
			JCusparse.cusparseDestroySpMat(matCsr.asConst());
			JCusparse.cusparseDestroyDnMat(matDense.asConst());
		}
	}

	@Override
	public int cusparsennz(cusparseHandle handle, int dirA, int m, int n, cusparseMatDescr descrA, Pointer A, int lda,
		Pointer nnzPerRowCol, Pointer nnzTotalDevHostPtr) {
		return JCusparse.cusparseDnnz(handle, dirA, m, n, descrA, A, lda, nnzPerRowCol, nnzTotalDevHostPtr);
	}

	@Override
	public void deviceToHost(GPUContext gCtx, Pointer src, double[] dest, String instName, boolean isEviction) {
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
		cudaMemcpy(dest, Pointer.to(src), ((long)src.length)*Sizeof.DOUBLE, cudaMemcpyHostToDevice);
	}
}
