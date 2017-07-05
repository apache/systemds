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

import static jcuda.jcusparse.JCusparse.cusparseCreateMatDescr;
import static jcuda.jcusparse.JCusparse.cusparseDcsr2dense;
import static jcuda.jcusparse.JCusparse.cusparseSetMatIndexBase;
import static jcuda.jcusparse.JCusparse.cusparseSetMatType;
import static jcuda.jcusparse.JCusparse.cusparseSetPointerMode;
import static jcuda.jcusparse.JCusparse.cusparseXcsrgeamNnz;
import static jcuda.jcusparse.JCusparse.cusparseXcsrgemmNnz;
import static jcuda.jcusparse.cusparseIndexBase.CUSPARSE_INDEX_BASE_ZERO;
import static jcuda.jcusparse.cusparseMatrixType.CUSPARSE_MATRIX_TYPE_GENERAL;
import static jcuda.runtime.JCuda.cudaMemcpy;
import static jcuda.runtime.cudaMemcpyKind.cudaMemcpyDeviceToDevice;
import static jcuda.runtime.cudaMemcpyKind.cudaMemcpyDeviceToHost;
import static jcuda.runtime.cudaMemcpyKind.cudaMemcpyHostToDevice;

import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;
import org.apache.sysml.api.DMLScript;
import org.apache.sysml.runtime.DMLRuntimeException;
import org.apache.sysml.utils.GPUStatistics;

import jcuda.Pointer;
import jcuda.Sizeof;
import jcuda.jcublas.cublasHandle;
import jcuda.jcusparse.cusparseHandle;
import jcuda.jcusparse.cusparseMatDescr;
import jcuda.jcusparse.cusparsePointerMode;

/**
 * Compressed Sparse Row (CSR) format for CUDA
 * Generalized matrix multiply is implemented for CSR format in the cuSparse library among other operations
 */
public class CSRPointer {

	private static final Log LOG = LogFactory.getLog(CSRPointer.class.getName());

	private static final double ULTRA_SPARSITY_TURN_POINT = 0.0004;
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

	private static long getDoubleSizeOf(long numElems) {
		return numElems * ((long) jcuda.Sizeof.DOUBLE);
	}

	//  private Pointer allocate(String instName, long size) throws DMLRuntimeException {
	//    return getGPUContext().allocate(instName, size);
	//  }

	private static long getIntSizeOf(long numElems) {
		return numElems * ((long) jcuda.Sizeof.INT);
	}

	//  private void cudaFreeHelper(Pointer toFree) throws DMLRuntimeException {
	//    getGPUContext().cudaFreeHelper(toFree);
	//  }

	public static int toIntExact(long l) throws DMLRuntimeException {
		if (l < Integer.MIN_VALUE || l > Integer.MAX_VALUE) {
			throw new DMLRuntimeException("Cannot be cast to int:" + l);
		}
		return (int) l;
	}

	//  private void cudaFreeHelper(String instName, Pointer toFree, boolean eager) throws DMLRuntimeException {
	//    getGPUContext().cudaFreeHelper(instName, toFree, eager);
	//  }

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
		long sizeofValArray = getDoubleSizeOf(nnz2);
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
	 * @param dest   [input] destination location (on GPU)
	 * @param rows   number of rows
	 * @param nnz    number of non-zeroes
	 * @param rowPtr integer array of row pointers
	 * @param colInd integer array of column indices
	 * @param values double array of non zero values
	 */
	public static void copyToDevice(CSRPointer dest, int rows, long nnz, int[] rowPtr, int[] colInd, double[] values) {
		CSRPointer r = dest;
		long t0 = 0;
		if (DMLScript.STATISTICS)
			t0 = System.nanoTime();
		r.nnz = nnz;
		cudaMemcpy(r.rowPtr, Pointer.to(rowPtr), getIntSizeOf(rows + 1), cudaMemcpyHostToDevice);
		cudaMemcpy(r.colInd, Pointer.to(colInd), getIntSizeOf(nnz), cudaMemcpyHostToDevice);
		cudaMemcpy(r.val, Pointer.to(values), getDoubleSizeOf(nnz), cudaMemcpyHostToDevice);
		if (DMLScript.STATISTICS)
			GPUStatistics.cudaToDevTime.addAndGet(System.nanoTime() - t0);
		if (DMLScript.STATISTICS)
			GPUStatistics.cudaToDevCount.addAndGet(3);
	}

	/**
	 * Static method to copy a CSR sparse matrix from Device to host
	 *
	 * @param src    [input] source location (on GPU)
	 * @param rows   [input] number of rows
	 * @param nnz    [input] number of non-zeroes
	 * @param rowPtr [output] pre-allocated integer array of row pointers of size (rows+1)
	 * @param colInd [output] pre-allocated integer array of column indices of size nnz
	 * @param values [output] pre-allocated double array of values of size nnz
	 */
	public static void copyToHost(CSRPointer src, int rows, long nnz, int[] rowPtr, int[] colInd, double[] values) {
		CSRPointer r = src;
		long t0 = 0;
		if (DMLScript.STATISTICS)
			t0 = System.nanoTime();
		cudaMemcpy(Pointer.to(rowPtr), r.rowPtr, getIntSizeOf(rows + 1), cudaMemcpyDeviceToHost);
		cudaMemcpy(Pointer.to(colInd), r.colInd, getIntSizeOf(nnz), cudaMemcpyDeviceToHost);
		cudaMemcpy(Pointer.to(values), r.val, getDoubleSizeOf(nnz), cudaMemcpyDeviceToHost);
		if (DMLScript.STATISTICS)
			GPUStatistics.cudaFromDevTime.addAndGet(System.nanoTime() - t0);
		if (DMLScript.STATISTICS)
			GPUStatistics.cudaFromDevCount.addAndGet(3);
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
	 * @throws DMLRuntimeException if DMLRuntimeException occurs
	 */
	public static CSRPointer allocateForDgeam(GPUContext gCtx, cusparseHandle handle, CSRPointer A, CSRPointer B, int m,
			int n) throws DMLRuntimeException {
		if (A.nnz >= Integer.MAX_VALUE || B.nnz >= Integer.MAX_VALUE) {
			throw new DMLRuntimeException("Number of non zeroes is larger than supported by cuSparse");
		}
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
	 * @param gCtx   ?
	 * @param handle a valid {@link cusparseHandle}
	 * @param A      Sparse Matrix A on GPU
	 * @param transA 'T' if A is to be transposed, 'N' otherwise
	 * @param B      Sparse Matrix B on GPU
	 * @param transB 'T' if B is to be transposed, 'N' otherwise
	 * @param m      Rows in A
	 * @param n      Columns in B
	 * @param k      Columns in A / Rows in B
	 * @return a {@link CSRPointer} instance that encapsulates the CSR matrix on GPU
	 * @throws DMLRuntimeException if DMLRuntimeException occurs
	 */
	public static CSRPointer allocateForMatrixMultiply(GPUContext gCtx, cusparseHandle handle, CSRPointer A, int transA,
			CSRPointer B, int transB, int m, int n, int k) throws DMLRuntimeException {
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
	 * @param gCtx ?
	 * @param nnz2 number of non-zeroes
	 * @param rows number of rows
	 * @return a {@link CSRPointer} instance that encapsulates the CSR matrix on GPU
	 * @throws DMLRuntimeException if DMLRuntimeException occurs
	 */
	public static CSRPointer allocateEmpty(GPUContext gCtx, long nnz2, long rows) throws DMLRuntimeException {
		LOG.trace("GPU : allocateEmpty from CSRPointer with nnz=" + nnz2 + " and rows=" + rows + ", GPUContext=" + gCtx);
		assert nnz2 > -1 : "Incorrect usage of internal API, number of non zeroes is less than 0 when trying to allocate sparse data on GPU";
		CSRPointer r = new CSRPointer(gCtx);
		r.nnz = nnz2;
		if (nnz2 == 0) {
			// The convention for an empty sparse matrix is to just have an instance of the CSRPointer object
			// with no memory allocated on the GPU.
			return r;
		}
		gCtx.ensureFreeSpace(getDoubleSizeOf(nnz2) + getIntSizeOf(rows + 1) + getIntSizeOf(nnz2));
		// increment the cudaCount by 1 for the allocation of all 3 arrays
		r.val = gCtx.allocate(null, getDoubleSizeOf(nnz2));
		r.rowPtr = gCtx.allocate(null, getIntSizeOf(rows + 1));
		r.colInd = gCtx.allocate(null, getIntSizeOf(nnz2));
		return r;
	}

	/**
	 * Allocate row pointers of m+1 elements
	 *
	 * @param gCtx   a valid {@link GPUContext}
	 * @param handle a valid {@link cusparseHandle}
	 * @param C      Output matrix
	 * @param rowsC  number of rows in C
	 * @throws DMLRuntimeException ?
	 */
	private static void step1AllocateRowPointers(GPUContext gCtx, cusparseHandle handle, CSRPointer C, int rowsC)
			throws DMLRuntimeException {
		LOG.trace("GPU : step1AllocateRowPointers" + ", GPUContext=" + gCtx);
		cusparseSetPointerMode(handle, cusparsePointerMode.CUSPARSE_POINTER_MODE_HOST);
		//cudaDeviceSynchronize;
		// Do not increment the cudaCount of allocations on GPU
		C.rowPtr = gCtx.allocate(getIntSizeOf((long) rowsC + 1));
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
	 * @throws DMLRuntimeException ?
	 */
	private static void step2GatherNNZGeam(GPUContext gCtx, cusparseHandle handle, CSRPointer A, CSRPointer B,
			CSRPointer C, int m, int n) throws DMLRuntimeException {
		LOG.trace("GPU : step2GatherNNZGeam for DGEAM" + ", GPUContext=" + gCtx);
		int[] CnnzArray = { -1 };
		cusparseXcsrgeamNnz(handle, m, n, A.descr, toIntExact(A.nnz), A.rowPtr, A.colInd, B.descr, toIntExact(B.nnz),
				B.rowPtr, B.colInd, C.descr, C.rowPtr, Pointer.to(CnnzArray));
		//cudaDeviceSynchronize;
		if (CnnzArray[0] != -1) {
			C.nnz = CnnzArray[0];
		} else {
			int baseArray[] = { 0 };
			cudaMemcpy(Pointer.to(CnnzArray), C.rowPtr.withByteOffset(getIntSizeOf(m)), getIntSizeOf(1),
					cudaMemcpyDeviceToHost);
			cudaMemcpy(Pointer.to(baseArray), C.rowPtr, getIntSizeOf(1), cudaMemcpyDeviceToHost);
			C.nnz = CnnzArray[0] - baseArray[0];
		}
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
	 * @throws DMLRuntimeException ?
	 */
	private static void step2GatherNNZGemm(GPUContext gCtx, cusparseHandle handle, CSRPointer A, int transA,
			CSRPointer B, int transB, CSRPointer C, int m, int n, int k) throws DMLRuntimeException {
		LOG.trace("GPU : step2GatherNNZGemm for DGEMM" + ", GPUContext=" + gCtx);
		int[] CnnzArray = { -1 };
		if (A.nnz >= Integer.MAX_VALUE || B.nnz >= Integer.MAX_VALUE) {
			throw new DMLRuntimeException("Number of non zeroes is larger than supported by cuSparse");
		}
		cusparseXcsrgemmNnz(handle, transA, transB, m, n, k, A.descr, toIntExact(A.nnz), A.rowPtr, A.colInd, B.descr,
				toIntExact(B.nnz), B.rowPtr, B.colInd, C.descr, C.rowPtr, Pointer.to(CnnzArray));
		//cudaDeviceSynchronize;
		if (CnnzArray[0] != -1) {
			C.nnz = CnnzArray[0];
		} else {
			int baseArray[] = { 0 };
			cudaMemcpy(Pointer.to(CnnzArray), C.rowPtr.withByteOffset(getIntSizeOf(m)), getIntSizeOf(1),
					cudaMemcpyDeviceToHost);
			cudaMemcpy(Pointer.to(baseArray), C.rowPtr, getIntSizeOf(1), cudaMemcpyDeviceToHost);
			C.nnz = CnnzArray[0] - baseArray[0];
		}
	}

	/**
	 * Allocate val and index pointers.
	 *
	 * @param gCtx   a valid {@link GPUContext}
	 * @param handle a valid {@link cusparseHandle}
	 * @param C      Output sparse matrix on GPU
	 * @throws DMLRuntimeException ?
	 */
	private static void step3AllocateValNInd(GPUContext gCtx, cusparseHandle handle, CSRPointer C)
			throws DMLRuntimeException {
		LOG.trace("GPU : step3AllocateValNInd" + ", GPUContext=" + gCtx);
		// Increment cudaCount by one when all three arrays of CSR sparse array are allocated
		C.val = gCtx.allocate(null, getDoubleSizeOf(C.nnz));
		C.colInd = gCtx.allocate(null, getIntSizeOf(C.nnz));
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

	public CSRPointer clone(int rows) throws DMLRuntimeException {
		CSRPointer me = this;
		CSRPointer that = new CSRPointer(me.getGPUContext());

		that.allocateMatDescrPointer();
		long totalSize = estimateSize(me.nnz, rows);
		that.gpuContext.ensureFreeSpace(totalSize);

		that.nnz = me.nnz;
		that.val = allocate(that.nnz * Sizeof.DOUBLE);
		that.rowPtr = allocate(rows * Sizeof.DOUBLE);
		that.colInd = allocate(that.nnz * Sizeof.DOUBLE);

		cudaMemcpy(that.val, me.val, that.nnz * Sizeof.DOUBLE, cudaMemcpyDeviceToDevice);
		cudaMemcpy(that.rowPtr, me.rowPtr, rows * Sizeof.DOUBLE, cudaMemcpyDeviceToDevice);
		cudaMemcpy(that.colInd, me.colInd, that.nnz * Sizeof.DOUBLE, cudaMemcpyDeviceToDevice);

		return that;
	}

	private Pointer allocate(long size) throws DMLRuntimeException {
		return getGPUContext().allocate(size);
	}

	private void cudaFreeHelper(Pointer toFree, boolean eager) throws DMLRuntimeException {
		getGPUContext().cudaFreeHelper(toFree, eager);
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
	 * @return A {@link Pointer} to the allocated dense matrix (in column-major format)
	 * @throws DMLRuntimeException if DMLRuntimeException occurs
	 */
	public Pointer toColumnMajorDenseMatrix(cusparseHandle cusparseHandle, cublasHandle cublasHandle, int rows,
			int cols) throws DMLRuntimeException {
		LOG.trace("GPU : sparse -> column major dense (inside CSRPointer) on " + this + ", GPUContext="
				+ getGPUContext());
		long size = ((long) rows) * getDoubleSizeOf((long) cols);
		Pointer A = allocate(size);
		// If this sparse block is empty, the allocated dense matrix, initialized to zeroes, will be returned.
		if (val != null && rowPtr != null && colInd != null && nnz > 0) {
			// Note: cusparseDcsr2dense method cannot handle empty blocks
			cusparseDcsr2dense(cusparseHandle, rows, cols, descr, val, rowPtr, colInd, A, rows);
			//cudaDeviceSynchronize;
		} else {
			LOG.debug("in CSRPointer, the values array, row pointers array or column indices array was null");
		}
		return A;
	}

	/**
	 * Calls cudaFree lazily on the allocated {@link Pointer} instances
	 *
	 * @throws DMLRuntimeException ?
	 */
	public void deallocate() throws DMLRuntimeException {
		deallocate(false);
	}

	/**
	 * Calls cudaFree lazily or eagerly on the allocated {@link Pointer} instances
	 *
	 * @param eager whether to do eager or lazy cudaFrees
	 * @throws DMLRuntimeException ?
	 */
	public void deallocate(boolean eager) throws DMLRuntimeException {
		if (nnz > 0) {
			cudaFreeHelper(val, eager);
			cudaFreeHelper(rowPtr, eager);
			cudaFreeHelper(colInd, eager);
		}
	}

	@Override
	public String toString() {
		return "CSRPointer{" + "nnz=" + nnz + '}';
	}
}
