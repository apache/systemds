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

package org.apache.sysds.runtime.instructions.gpu.context;

import jcuda.Pointer;
import jcuda.Sizeof;
import jcuda.cudaDataType;
import jcuda.jcublas.cublasHandle;
import jcuda.jcusparse.*;
import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;
import org.apache.sysds.api.DMLScript;
import org.apache.sysds.runtime.DMLRuntimeException;
import org.apache.sysds.runtime.matrix.data.LibMatrixCUDA;
import org.apache.sysds.utils.Statistics;

import static jcuda.jcusparse.JCusparse.*;
import static jcuda.jcusparse.cusparseIndexBase.CUSPARSE_INDEX_BASE_ZERO;
import static jcuda.jcusparse.cusparseMatrixType.CUSPARSE_MATRIX_TYPE_GENERAL;
import static jcuda.runtime.JCuda.*;
import static jcuda.runtime.cudaMemcpyKind.*;

/**
 * Compressed Sparse Row (CSR) format for CUDA
 * Generalized matrix multiply is implemented for CSR format in the cuSparse library among other operations
 * 
 * Since we assume that the matrix is stored with zero-based indexing (i.e. CUSPARSE_INDEX_BASE_ZERO),
 * the matrix
 * 1.0 4.0 0.0 0.0 0.0 
 * 0.0 2.0 3.0 0.0 0.0 
 * 5.0 0.0 0.0 7.0 8.0 
 * 0.0 0.0 9.0 0.0 6.0
 * 
 * is stored as
 * val = 1.0 4.0 2.0 3.0 5.0 7.0 8.0 9.0 6.0 
 * rowPtr = 0.0 2.0 4.0 7.0 9.0 
 * colInd = 0.0 1.0 1.0 2.0 0.0 3.0 4.0 2.0 4.0
 */
public class CSRPointer {

	private static final Log LOG = LogFactory.getLog(CSRPointer.class.getName());

	private static final double ULTRA_SPARSITY_TURN_POINT = 0.00004;
	public static cusparseMatDescr matrixDescriptor;
	/**
	 * {@link GPUContext} instance to track the GPU to do work on
	 */
	private final GPUContext gpuContext;
	/**
	 * Number of non zeroes
	 */
	public long nnz;

	/**
	 * double array of non zero values
	 */
	public Pointer val;

	/**
	 * integer array of start of all rows and end of last row + 1
	 */
	public Pointer rowPtr;

	/**
	 * integer array of nnz values' column indices
	 */
	public Pointer colInd;

	/**
	 * descriptor of matrix, only CUSPARSE_MATRIX_TYPE_GENERAL supported
	 */
	public cusparseMatDescr descr;

	/**
	 * Default constructor to help with Factory method {@link #allocateEmpty(GPUContext, long, long)}
	 *
	 * @param gCtx a valid {@link GPUContext}
	 */
	private CSRPointer(GPUContext gCtx) {
		gpuContext = gCtx;
		val = new Pointer();
		rowPtr = new Pointer();
		colInd = new Pointer();
		allocateMatDescrPointer();
	}

	private static long getDataTypeSizeOf(long numElems) {
		return numElems * LibMatrixCUDA.sizeOfDataType;
	}

	private static long getIntSizeOf(long numElems) {
		return numElems * Sizeof.INT;
	}

	public static int toIntExact(long l) {
		if (l < Integer.MIN_VALUE || l > Integer.MAX_VALUE) {
			throw new DMLRuntimeException("Cannot be cast to int:" + l);
		}
		return (int) l;
	}

	/**
	 * @return Singleton default matrix descriptor object
	 * (set with CUSPARSE_MATRIX_TYPE_GENERAL, CUSPARSE_INDEX_BASE_ZERO)
	 */
	public static cusparseMatDescr getDefaultCuSparseMatrixDescriptor() {
		if (matrixDescriptor == null) {
			// Code from JCuda Samples - http://www.jcuda.org/samples/JCusparseSample.java
			matrixDescriptor = new cusparseMatDescr();
			cusparseCreateMatDescr(matrixDescriptor);
			cusparseSetMatType(matrixDescriptor, CUSPARSE_MATRIX_TYPE_GENERAL);
			cusparseSetMatIndexBase(matrixDescriptor, CUSPARSE_INDEX_BASE_ZERO);
		}
		return matrixDescriptor;
	}

	/**
	 * Estimate the size of a CSR matrix in GPU memory
	 * Size of pointers is not needed and is not added in
	 *
	 * @param nnz2 number of non zeroes
	 * @param rows number of rows
	 * @return size estimate
	 */
	public static long estimateSize(long nnz2, long rows) {
		long sizeofValArray = getDataTypeSizeOf(nnz2);
		long sizeofRowPtrArray = getIntSizeOf(rows + 1);
		long sizeofColIndArray = getIntSizeOf(nnz2);
		long sizeofDescr = getIntSizeOf(4);
		// From the CUSPARSE documentation, the cusparseMatDescr in native code is represented as:
		// typedef struct {
		// 	cusparseMatrixType_t MatrixType;
		//	cusparseFillMode_t FillMode;
		//	cusparseDiagType_t DiagType;
		// 	cusparseIndexBase_t IndexBase;
		// } cusparseMatDescr_t;
		long tot = sizeofValArray + sizeofRowPtrArray + sizeofColIndArray + sizeofDescr;
		return tot;
	}

	/**
	 * Static method to copy a CSR sparse matrix from Host to Device
	 *
	 * @param gCtx GPUContext
	 * @param dest   [input] destination location (on GPU)
	 * @param rows   number of rows
	 * @param nnz    number of non-zeroes
	 * @param rowPtr integer array of row pointers
	 * @param colInd integer array of column indices
	 * @param values double array of non zero values
	 */
	public static void copyToDevice(GPUContext gCtx, CSRPointer dest, int rows, long nnz, int[] rowPtr, int[] colInd, double[] values) {
		CSRPointer r = dest;
		r.nnz = nnz;
		if(rows < 0) throw new DMLRuntimeException("Incorrect input parameter: rows=" + rows);
		if(nnz < 0) throw new DMLRuntimeException("Incorrect input parameter: nnz=" + nnz);
		if(rowPtr.length < rows + 1) throw new DMLRuntimeException("The length of rowPtr needs to be greater than or equal to " + (rows + 1));
		if(colInd.length < nnz) throw new DMLRuntimeException("The length of colInd needs to be greater than or equal to " + nnz);
		if(values.length < nnz) throw new DMLRuntimeException("The length of values needs to be greater than or equal to " + nnz);
		LibMatrixCUDA.cudaSupportFunctions.hostToDevice(gCtx, values, r.val, null);
		cudaMemcpy(r.rowPtr, Pointer.to(rowPtr), getIntSizeOf(rows + 1), cudaMemcpyHostToDevice);
		cudaMemcpy(r.colInd, Pointer.to(colInd), getIntSizeOf(nnz), cudaMemcpyHostToDevice);
		//if (DMLScript.STATISTICS)
		//	GPUStatistics.cudaToDevTime.add(System.nanoTime() - t0);
		//if (DMLScript.STATISTICS)
		//	GPUStatistics.cudaToDevCount.add(3);
	}
	
	/**
	 * Static method to copy a CSR sparse matrix from Device to host
	 *
	 * @param src    [input] source location (on GPU)
	 * @param rows   [input] number of rows
	 * @param nnz    [input] number of non-zeroes
	 * @param rowPtr [output] pre-allocated integer array of row pointers of size (rows+1)
	 * @param colInd [output] pre-allocated integer array of column indices of size nnz
	 */
	public static void copyPtrToHost(CSRPointer src, int rows, long nnz, int[] rowPtr, int[] colInd) {
		CSRPointer r = src;
		cudaMemcpy(Pointer.to(rowPtr), r.rowPtr, getIntSizeOf(rows + 1), cudaMemcpyDeviceToHost);
		cudaMemcpy(Pointer.to(colInd), r.colInd, getIntSizeOf(nnz), cudaMemcpyDeviceToHost);
	}

	/**
	 * Estimates the number of non zero elements from the results of a sparse cusparseDgeam operation
	 * C = a op(A) + b op(B)
	 *
	 * @param gCtx   a valid {@link GPUContext}
	 * @param handle a valid {@link cusparseHandle}
	 * @param A      Sparse Matrix A on GPU
	 * @param B      Sparse Matrix B on GPU
	 * @param m      Rows in A
	 * @param n      Columns in Bs
	 * @return CSR (compressed sparse row) pointer
	 */
	public static CSRPointer allocateForDgeam(GPUContext gCtx, cusparseHandle handle, CSRPointer A, CSRPointer B, int m, int n) {
		if (A.nnz >= Integer.MAX_VALUE || B.nnz >= Integer.MAX_VALUE)
			throw new DMLRuntimeException("Number of non zeroes is larger than supported by cuSparse");
		CSRPointer C = new CSRPointer(gCtx);
		step1AllocateRowPointers(gCtx, handle, C, m);
		step2GatherNNZGeam(gCtx, handle, A, B, C, m, n);
		step3AllocateValNInd(gCtx, handle, C);
		return C;
	}

	/**
	 * Estimates the number of non-zero elements from the result of a sparse matrix multiplication C = A * B
	 * and returns the {@link CSRPointer} to C with the appropriate GPU memory.
	 *
	 * @param gCtx   a valid {@link GPUContext}
	 * @param handle a valid {@link cusparseHandle}
	 * @param A      Sparse Matrix A on GPU
	 * @param transA 'T' if A is to be transposed, 'N' otherwise
	 * @param B      Sparse Matrix B on GPU
	 * @param transB 'T' if B is to be transposed, 'N' otherwise
	 * @param m      Rows in A
	 * @param n      Columns in B
	 * @param k      Columns in A / Rows in B
	 * @return a {@link CSRPointer} instance that encapsulates the CSR matrix on GPU
	 */
	public static CSRPointer allocateForMatrixMultiply(GPUContext gCtx, cusparseHandle handle, CSRPointer A, int transA,
			CSRPointer B, int transB, int m, int n, int k) {
		// Following the code example at http://docs.nvidia.com/cuda/cusparse/#cusparse-lt-t-gt-csrgemm and at
		// https://github.com/jcuda/jcuda-matrix-utils/blob/master/JCudaMatrixUtils/src/test/java/org/jcuda/matrix/samples/JCusparseSampleDgemm.java
		CSRPointer C = new CSRPointer(gCtx);
		step1AllocateRowPointers(gCtx, handle, C, m);
		step2GatherNNZGemm(gCtx, handle, A, transA, B, transB, C, m, n, k);
		step3AllocateValNInd(gCtx, handle, C);
		return C;
	}

	/**
	 * Factory method to allocate an empty CSR Sparse matrix on the GPU
	 *
	 * @param gCtx a valid {@link GPUContext}
	 * @param nnz2 number of non-zeroes
	 * @param rows number of rows
	 * @param initialize memset to zero?
	 * @return a {@link CSRPointer} instance that encapsulates the CSR matrix on GPU
	 */
	public static CSRPointer allocateEmpty(GPUContext gCtx, long nnz2, long rows, boolean initialize) {
		LOG.trace("GPU : allocateEmpty from CSRPointer with nnz=" + nnz2 + " and rows=" + rows + ", GPUContext=" + gCtx);
		if(nnz2 < 0) throw new DMLRuntimeException("Incorrect usage of internal API, number of non zeroes is less " +
			"than 0 when trying to allocate sparse data on GPU");
		if(rows <= 0) throw new DMLRuntimeException("Incorrect usage of internal API, number of rows is less than or " +
			"equal to 0 when trying to allocate sparse data on GPU");
		CSRPointer r = new CSRPointer(gCtx);
		r.nnz = nnz2;
		if (nnz2 == 0) {
			// The convention for an empty sparse matrix is to just have an instance of the CSRPointer object
			// with no memory allocated on the GPU.
			return r;
		}
		// increment the cudaCount by 1 for the allocation of all 3 arrays
		r.val = gCtx.allocate(null, getDataTypeSizeOf(nnz2), initialize);
		r.rowPtr = gCtx.allocate(null, getIntSizeOf(rows + 1), initialize);
		r.colInd = gCtx.allocate(null, getIntSizeOf(nnz2), initialize);
		return r;
	}

	public static CSRPointer allocateEmpty(GPUContext gCtx, long nnz2, long rows) {
		return allocateEmpty(gCtx, nnz2, rows, true);
	}

	/**
	 * Allocate row pointers of m+1 elements
	 *
	 * @param gCtx   a valid {@link GPUContext}
	 * @param handle a valid {@link cusparseHandle}
	 * @param C      Output matrix
	 * @param rowsC  number of rows in C
	 */
	private static void step1AllocateRowPointers(GPUContext gCtx, cusparseHandle handle, CSRPointer C, int rowsC) {
		LOG.trace("GPU : step1AllocateRowPointers" + ", GPUContext=" + gCtx);
		cusparseSetPointerMode(handle, cusparsePointerMode.CUSPARSE_POINTER_MODE_HOST);

		// Do not increment the cudaCount of allocations on GPU
		C.rowPtr = gCtx.allocate(null, getIntSizeOf((long) rowsC + 1), true);
	}

	/**
	 * Determine total number of nonzero element for the cusparseDgeam  operation.
	 * This is done from either (nnzC=*nnzTotalDevHostPtr) or (nnzC=csrRowPtrC(m)-csrRowPtrC(0))
	 *
	 * @param gCtx   a valid {@link GPUContext}
	 * @param handle a valid {@link cusparseHandle}
	 * @param A      Sparse Matrix A on GPU
	 * @param B      Sparse Matrix B on GPU
	 * @param C      Output Sparse Matrix C on GPU
	 * @param m      Rows in C
	 * @param n      Columns in C
	 */

	private static void step2GatherNNZGeam(GPUContext gCtx, cusparseHandle handle, CSRPointer A, CSRPointer B, CSRPointer C, int m, int n) {
		LOG.trace("GPU : step2GatherNNZGeam for DGEAM" + ", GPUContext=" + gCtx);
		long[] pBufferSizeInBytes = {0};
		cusparseDcsrgeam2_bufferSizeExt(handle, m, n, Pointer.to(new double[]{1.0}), A.descr, toIntExact(A.nnz), A.val, A.rowPtr, A.colInd,
			Pointer.to(new double[]{1.0}), B.descr, toIntExact(B.nnz), B.val, B.rowPtr, B.colInd, C.descr, C.val, C.rowPtr, C.colInd, pBufferSizeInBytes);
		Pointer buffer = new Pointer();
		cudaMalloc(buffer, pBufferSizeInBytes[0]);
		int[] CnnzArray = {-1};
		cusparseXcsrgeam2Nnz(handle, m, n, A.descr, toIntExact(A.nnz), A.rowPtr, A.colInd, B.descr, toIntExact(B.nnz), B.rowPtr, B.colInd,
			C.descr, C.rowPtr, Pointer.to(CnnzArray) ,buffer);
		if(CnnzArray[0] != -1) {
			C.nnz = CnnzArray[0];
		}
		else {                            // fall-back (rare older devices)
			int[] baseArray = {0};
			cudaMemcpy(Pointer.to(CnnzArray),
				C.rowPtr.withByteOffset((long)m * Sizeof.INT),
				Sizeof.INT, cudaMemcpyDeviceToHost);
			cudaMemcpy(Pointer.to(baseArray),
				C.rowPtr, Sizeof.INT, cudaMemcpyDeviceToHost);
			C.nnz = CnnzArray[0] - baseArray[0];
		}
		cudaFree(buffer);
	}

	/**
	 * Determine total number of nonzero element for the cusparseDgemm operation.
	 *
	 * @param gCtx   a valid {@link GPUContext}
	 * @param handle a valid {@link cusparseHandle}
	 * @param A      Sparse Matrix A on GPU
	 * @param transA op - whether A is transposed
	 * @param B      Sparse Matrix B on GPU
	 * @param transB op - whether B is transposed
	 * @param C      Output Sparse Matrix C on GPU
	 * @param m      Number of rows of sparse matrix op ( A ) and C
	 * @param n      Number of columns of sparse matrix op ( B ) and C
	 * @param k      Number of columns/rows of sparse matrix op ( A ) / op ( B )
	 */

	private static void step2GatherNNZGemm(GPUContext gCtx, cusparseHandle handle, CSRPointer A, int transA,
		CSRPointer B, int transB, CSRPointer C, int m, int n, int k)            // C = op(A)·op(B)  (m×k)·(k×n)
	{
		LOG.trace("GPU : step2GatherNNZGemm (SpGEMM), GPUContext=" + gCtx);

		/* ---------- quick guard ---------------------------------------- */
		if(A.nnz >= Integer.MAX_VALUE || B.nnz >= Integer.MAX_VALUE)
			throw new DMLRuntimeException("Number of non-zeros exceeds cuSPARSE 32-bit limit");

		/* ---------- 1. CSR descriptors for A, B, C --------------------- */
		cusparseSpMatDescr matA = new cusparseSpMatDescr();
		cusparseSpMatDescr matB = new cusparseSpMatDescr();
		cusparseSpMatDescr matC = new cusparseSpMatDescr();

		cusparseCreateCsr(matA, m, k, A.nnz, A.rowPtr, A.colInd, A.val, cusparseIndexType.CUSPARSE_INDEX_32I,
			cusparseIndexType.CUSPARSE_INDEX_32I, cusparseIndexBase.CUSPARSE_INDEX_BASE_ZERO, cudaDataType.CUDA_R_64F);

		cusparseCreateCsr(matB, k, n, B.nnz, B.rowPtr, B.colInd, B.val, cusparseIndexType.CUSPARSE_INDEX_32I,
			cusparseIndexType.CUSPARSE_INDEX_32I, cusparseIndexBase.CUSPARSE_INDEX_BASE_ZERO, cudaDataType.CUDA_R_64F);

		cusparseCreateCsr(matC, m, n, 0L,                 // nnz(C) unknown
			C.rowPtr, Pointer.to(new int[] {0}), Pointer.to(new double[] {0}), cusparseIndexType.CUSPARSE_INDEX_32I,
			cusparseIndexType.CUSPARSE_INDEX_32I, cusparseIndexBase.CUSPARSE_INDEX_BASE_ZERO, cudaDataType.CUDA_R_64F);

		/* ---------- 2. SpGEMM descriptor ------------------------------- */
		cusparseSpGEMMDescr spgemmDesc = new cusparseSpGEMMDescr();
		cusparseSpGEMM_createDescr(spgemmDesc);

		Pointer alpha = Pointer.to(new double[] {1.0});
		Pointer beta = Pointer.to(new double[] {0.0});
		int alg = cusparseSpGEMMAlg.CUSPARSE_SPGEMM_DEFAULT;

		/* ---------- 3. Phase-1 : work-estimation ----------------------- */
		long[] bufSize1 = {0};
		cusparseSpGEMM_workEstimation(handle, transA, transB, alpha, matA.asConst(), matB.asConst(), beta, matC,
			cudaDataType.CUDA_R_64F, alg, spgemmDesc, bufSize1, null);                               // first query

		Pointer dBuf1 = new Pointer();
		if(bufSize1[0] > 0)
			cudaMalloc(dBuf1, bufSize1[0]);

		cusparseSpGEMM_workEstimation(handle, transA, transB, alpha, matA.asConst(), matB.asConst(), beta, matC,
			cudaDataType.CUDA_R_64F, alg, spgemmDesc, bufSize1, dBuf1);                              // real run

		/* ---------- 4. Phase-2 : compute structure / nnz --------------- */
		long[] bufSize2 = {0};
		cusparseSpGEMM_compute(                           // size query
			handle, transA, transB, alpha, matA.asConst(), matB.asConst(), beta, matC, cudaDataType.CUDA_R_64F, alg,
			spgemmDesc, bufSize2, null);                              // ← 13 args

		Pointer dBuf2 = new Pointer();
		if(bufSize2[0] > 0)
			cudaMalloc(dBuf2, bufSize2[0]);

		cusparseSpGEMM_compute(                           // actual compute
			handle, transA, transB, alpha, matA.asConst(), matB.asConst(), beta, matC, cudaDataType.CUDA_R_64F, alg,
			spgemmDesc, bufSize2, dBuf2);

		/* ---------- 5. read nnz(C) ------------------------------------- */
		long[] rows = {0}, cols = {0}, nnz = {0};
		cusparseSpMatGetSize(matC.asConst(), rows, cols, nnz);
		C.nnz = (int) nnz[0];

		/* ---------- 6. temp col/val arrays so COPY can write them ------ */
		Pointer dCcol = new Pointer();
		Pointer dCval = new Pointer();
		if(C.nnz > 0) {
			cudaMalloc(dCcol, C.nnz * Sizeof.INT);
			cudaMalloc(dCval, C.nnz * Sizeof.DOUBLE);
		}
		cusparseCsrSetPointers(matC, C.rowPtr, dCcol, dCval);

		/* ---------- 7. Phase-3 : copy final CSR into user arrays ------- */
		cusparseSpGEMM_copy(                              // ← 11 args
			handle, transA, transB, alpha, matA.asConst(), matB.asConst(), beta, matC, cudaDataType.CUDA_R_64F, alg,
			spgemmDesc);

		/* ---------- 8. clean-up --------------------------------------- */
		cudaFree(dCcol);
		cudaFree(dCval);
		cudaFree(dBuf1);
		cudaFree(dBuf2);

		cusparseSpGEMM_destroyDescr(spgemmDesc);
		cusparseDestroySpMat(matA.asConst());
		cusparseDestroySpMat(matB.asConst());
		cusparseDestroySpMat(matC.asConst());
	}

	/**
	 * Allocate val and index pointers.
	 *
	 * @param gCtx   a valid {@link GPUContext}
	 * @param handle a valid {@link cusparseHandle}
	 * @param C      Output sparse matrix on GPU
	 */
	private static void step3AllocateValNInd(GPUContext gCtx, cusparseHandle handle, CSRPointer C) {
		LOG.trace("GPU : step3AllocateValNInd" + ", GPUContext=" + gCtx);
		// Increment cudaCount by one when all three arrays of CSR sparse array are allocated

		C.val = gCtx.allocate(null, getDataTypeSizeOf(C.nnz), false);
		C.colInd = gCtx.allocate(null, getIntSizeOf(C.nnz), false);
	}

	// ==============================================================================================

	// The following methods estimate the memory needed for sparse matrices that are
	// results of operations on other sparse matrices using the cuSparse Library.
	// The operation is C = op(A) binaryOperation op(B), C is the output and A & B are the inputs
	// op = whether to transpose or not
	// binaryOperation = For cuSparse, +, - are *(matmul) are supported

	// From CuSparse Manual,
	// Since A and B have different sparsity patterns, cuSPARSE adopts a two-step approach
	// to complete sparse matrix C. In the first step, the user allocates csrRowPtrC of m+1
	// elements and uses function cusparseXcsrgeamNnz() to determine csrRowPtrC
	// and the total number of nonzero elements. In the second step, the user gathers nnzC
	//(number of nonzero elements of matrix C) from either (nnzC=*nnzTotalDevHostPtr)
	// or (nnzC=csrRowPtrC(m)-csrRowPtrC(0)) and allocates csrValC, csrColIndC of
	// nnzC elements respectively, then finally calls function cusparse[S|D|C|Z]csrgeam()
	// to complete matrix C.

	public CSRPointer clone(int rows) {
		CSRPointer me = this;
		CSRPointer that = new CSRPointer(me.getGPUContext());
		that.allocateMatDescrPointer();
		that.nnz = me.nnz;
		that.val = allocate(that.nnz * LibMatrixCUDA.sizeOfDataType, false);
		that.rowPtr = allocate(rows * Sizeof.INT, false);
		that.colInd = allocate(that.nnz * Sizeof.INT, false);
		cudaMemcpy(that.val, me.val, that.nnz * LibMatrixCUDA.sizeOfDataType, cudaMemcpyDeviceToDevice);
		cudaMemcpy(that.rowPtr, me.rowPtr, rows * Sizeof.INT, cudaMemcpyDeviceToDevice);
		cudaMemcpy(that.colInd, me.colInd, that.nnz * Sizeof.INT, cudaMemcpyDeviceToDevice);
		return that;
	}

	private Pointer allocate(long size, boolean initialize) {
		return getGPUContext().allocate(null, size, initialize);
	}

	private GPUContext getGPUContext() {
		return gpuContext;
	}

	// ==============================================================================================

	/**
	 * Check for ultra sparsity
	 *
	 * @param rows number of rows
	 * @param cols number of columns
	 * @return true if ultra sparse
	 */
	public boolean isUltraSparse(int rows, int cols) {
		double sp = ((double) nnz / rows / cols);
		return sp < ULTRA_SPARSITY_TURN_POINT;
	}

	/**
	 * Initializes {@link #descr} to CUSPARSE_MATRIX_TYPE_GENERAL,
	 * the default that works for DGEMM.
	 */
	private void allocateMatDescrPointer() {
		this.descr = getDefaultCuSparseMatrixDescriptor();
	}

	/**
	 * Copies this CSR matrix on the GPU to a dense column-major matrix
	 * on the GPU. This is a temporary matrix for operations such as
	 * cusparseDcsrmv.
	 * Since the allocated matrix is temporary, bookkeeping is not updated.
	 * The caller is responsible for calling "free" on the returned Pointer object
	 *
	 * @param cusparseHandle a valid {@link cusparseHandle}
	 * @param cublasHandle   a valid {@link cublasHandle}
	 * @param rows           number of rows in this CSR matrix
	 * @param cols           number of columns in this CSR matrix
	 * @param instName          name of the invoking instruction to record{@link Statistics}.
	 * @return A {@link Pointer} to the allocated dense matrix (in column-major format)
	 */
	public Pointer toColumnMajorDenseMatrix(cusparseHandle cusparseHandle, cublasHandle cublasHandle, int rows,
			int cols, String instName) {
		LOG.trace("GPU : sparse -> column major dense (inside CSRPointer) on " + this + ", GPUContext="
				+ getGPUContext());
		long size = rows * getDataTypeSizeOf(cols);
		Pointer A = allocate(size, false);
		// If this sparse block is empty, the allocated dense matrix, initialized to zeroes, will be returned.
		if (val != null && rowPtr != null && colInd != null && nnz > 0) {
			// Note: cusparseDcsr2dense method cannot handle empty blocks
			LibMatrixCUDA.cudaSupportFunctions.cusparsecsr2dense(cusparseHandle, rows, cols, descr, val, rowPtr, colInd, A, rows);
			//cudaDeviceSynchronize;
		} else {
			LOG.debug("in CSRPointer, the values array, row pointers array or column indices array was null");
		}
		return A;
	}

	/**
	 * Calls cudaFree lazily on the allocated {@link Pointer} instances
	 *
	 */
	public void deallocate() {
		deallocate(DMLScript.EAGER_CUDA_FREE);
	}

	/**
	 * Calls cudaFree lazily or eagerly on the allocated {@link Pointer} instances
	 *
	 * @param eager whether to do eager or lazy cudaFrees
	 */
	public void deallocate(boolean eager) {
		if (nnz > 0) {
			if (val != null)
				getGPUContext().cudaFreeHelper(null, val, eager);
			if (rowPtr != null)
				getGPUContext().cudaFreeHelper(null, rowPtr, eager);
			if (colInd != null)
				getGPUContext().cudaFreeHelper(null, colInd, eager);
		}
		val = null;
		rowPtr = null;
		colInd = null;
	}

	@Override
	public String toString() {
		return "CSRPointer{" + "nnz=" + nnz + '}';
	}
}
