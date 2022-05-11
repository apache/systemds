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

import static jcuda.jcusparse.JCusparse.cusparseCreateMatDescr;
import static jcuda.jcusparse.JCusparse.cusparseSetMatIndexBase;
import static jcuda.jcusparse.JCusparse.cusparseSetMatType;
import static jcuda.jcusparse.JCusparse.cusparseSetPointerMode;
import static jcuda.jcusparse.JCusparse.cusparseXcsrgeam2Nnz;
import static jcuda.jcusparse.JCusparse.cusparseDcsrgeam2_bufferSizeExt;
import static jcuda.jcusparse.cusparseIndexBase.CUSPARSE_INDEX_BASE_ZERO;
import static jcuda.jcusparse.cusparseMatrixType.CUSPARSE_MATRIX_TYPE_GENERAL;
import static jcuda.runtime.JCuda.cudaMemcpy;
import static jcuda.runtime.cudaMemcpyKind.cudaMemcpyDeviceToDevice;
import static jcuda.runtime.cudaMemcpyKind.cudaMemcpyDeviceToHost;
import static jcuda.runtime.cudaMemcpyKind.cudaMemcpyHostToDevice;

import jcuda.jcusparse.JCusparse;
import jcuda.jcusparse.cusparseIndexBase;
import jcuda.jcusparse.cusparseIndexType;
import jcuda.jcusparse.cusparseSpMatDescr;
import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;
import org.apache.sysds.api.DMLScript;
import org.apache.sysds.runtime.DMLRuntimeException;
import org.apache.sysds.runtime.data.SparseBlockCSR;
import org.apache.sysds.runtime.matrix.data.LibMatrixCUDA;
import org.apache.sysds.utils.GPUStatistics;
import org.apache.sysds.utils.Statistics;

import jcuda.Pointer;
import jcuda.Sizeof;
import jcuda.jcusparse.cusparseHandle;
import jcuda.jcusparse.cusparseMatDescr;
import jcuda.jcusparse.cusparsePointerMode;

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
	@Deprecated
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

	public cusparseSpMatDescr createSpMatDescr(long rows, long cols) {
		return createSpMatDescr(rows, cols, this);
	}

	public static cusparseSpMatDescr createSpMatDescr(long rows, long cols, CSRPointer ptr) {
		cusparseSpMatDescr descr = new cusparseSpMatDescr();
		if(ptr != null) {
			JCusparse.cusparseCreateCsr(descr, rows, cols, ptr.nnz, ptr.rowPtr, ptr.colInd, ptr.val, cusparseIndexType.CUSPARSE_INDEX_32I,
					cusparseIndexType.CUSPARSE_INDEX_32I, cusparseIndexBase.CUSPARSE_INDEX_BASE_ZERO, LibMatrixCUDA.CUDA_DATA_TYPE);
		}
		else {
			JCusparse.cusparseCreateCsr(descr, rows, cols, 0, null, null, null, cusparseIndexType.CUSPARSE_INDEX_32I,
					cusparseIndexType.CUSPARSE_INDEX_32I, cusparseIndexBase.CUSPARSE_INDEX_BASE_ZERO, LibMatrixCUDA.CUDA_DATA_TYPE);
		}
		return descr;
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
		return sizeofValArray + sizeofRowPtrArray + sizeofColIndArray + sizeofDescr;
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
		long t0 = 0;
		if (DMLScript.STATISTICS)
			t0 = System.nanoTime();
		dest.nnz = nnz;
		if(rows < 0) throw new DMLRuntimeException("Incorrect input parameter: rows=" + rows);
		if(nnz < 0) throw new DMLRuntimeException("Incorrect input parameter: nnz=" + nnz);
		if(rowPtr.length < rows + 1) throw new DMLRuntimeException("The length of rowPtr needs to be greater than or equal to " + (rows + 1));
		if(colInd.length < nnz) throw new DMLRuntimeException("The length of colInd needs to be greater than or equal to " + nnz);
		if(values.length < nnz) throw new DMLRuntimeException("The length of values needs to be greater than or equal to " + nnz);
		LibMatrixCUDA.cudaSupportFunctions.hostToDevice(gCtx, values, dest.val, null);
		cudaMemcpy(dest.rowPtr, Pointer.to(rowPtr), getIntSizeOf(rows + 1), cudaMemcpyHostToDevice);
		cudaMemcpy(dest.colInd, Pointer.to(colInd), getIntSizeOf(nnz), cudaMemcpyHostToDevice);
		if (DMLScript.STATISTICS)
			GPUStatistics.cudaToDevTime.add(System.nanoTime() - t0);
		if (DMLScript.STATISTICS)
			GPUStatistics.cudaToDevCount.add(3);
	}
	
	/**
	 * Static method to copy a CSR sparse matrix from Device to host
	 *
	 * @param src    [input] source location (on GPU)
	 * @param rows   [input] number of rows
	 */
	public static SparseBlockCSR copyPtrToHost(GPUContext gCtx, CSRPointer src, int rows, String instName) {
		int nnz = toIntExact(src.nnz);
		double[] values = new double[nnz];
		int[] rowPtr = new int[rows + 1];
		int[] colInd = new int[nnz];

		long t0 = DMLScript.STATISTICS ? System.nanoTime() : 0;
		cudaMemcpy(Pointer.to(rowPtr), src.rowPtr, getIntSizeOf(rows + 1), cudaMemcpyDeviceToHost);
		cudaMemcpy(Pointer.to(colInd), src.colInd, getIntSizeOf(nnz), cudaMemcpyDeviceToHost);
		LibMatrixCUDA.cudaSupportFunctions.deviceToHost(gCtx, src.val, values, instName, true);

		if(DMLScript.STATISTICS) {
			long totalTime = System.nanoTime() - t0;
			GPUStatistics.cudaFromDevTime.add(totalTime);
			GPUStatistics.cudaFromDevCount.add(3);
		}
		return new SparseBlockCSR(rowPtr, colInd, values, nnz);
	}

	/**
	 * Estimates the number of non-zero elements from the results of a sparse cusparseDgeam operation
	 * C = a*op(A) + b*op(B)
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
		C.val = gCtx.allocate("cusparseXgeam", getDataTypeSizeOf(C.nnz));
		C.colInd = gCtx.allocate("cusparseXgeam", getIntSizeOf(C.nnz));
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
		int[] CnnzArray = {-1};

		double[] alpha = {0};
		double[] beta = {0};
		long[] pBufferSizeInBytes = {-1};
		cusparseDcsrgeam2_bufferSizeExt(handle, m, n, Pointer.to(alpha), A.descr, toIntExact(A.nnz), A.val, A.rowPtr,
			A.colInd, Pointer.to(beta), B.descr, toIntExact(B.nnz), B.val, B.rowPtr, B.colInd, C.descr, C.val, C.rowPtr,
			Pointer.to(CnnzArray), pBufferSizeInBytes);

		Pointer buf = gCtx.allocate(null, pBufferSizeInBytes[0]);

		cusparseXcsrgeam2Nnz(handle, m, n, A.descr, toIntExact(A.nnz), A.rowPtr, A.colInd, B.descr, toIntExact(B.nnz),
			B.rowPtr, B.colInd, C.descr, C.rowPtr, Pointer.to(CnnzArray), buf);

		if(CnnzArray[0] != -1) {
			C.nnz = CnnzArray[0];
		}
		else {
			int[] baseArray = {0};
			cudaMemcpy(Pointer.to(CnnzArray), C.rowPtr.withByteOffset(getIntSizeOf(m)), getIntSizeOf(1),
				cudaMemcpyDeviceToHost);
			cudaMemcpy(Pointer.to(baseArray), C.rowPtr, getIntSizeOf(1), cudaMemcpyDeviceToHost);
			C.nnz = CnnzArray[0] - baseArray[0];
		}
	}

	// ==============================================================================================

	// The following methods estimate the memory needed for sparse matrices that are
	// results of operations on other sparse matrices using the cuSparse Library.
	// The operation is C = op(A) binaryOperation op(B), C is the output and A & B are the inputs.
	// op = whether to transpose or not
	// binaryOperation = For cuSparse, +, - and *(matmul) are supported

	public CSRPointer clone(int rows) {
		CSRPointer me = this;
		CSRPointer that = new CSRPointer(me.getGPUContext());
		that.allocateMatDescrPointer();
		that.nnz = me.nnz;
		that.val = getGPUContext().allocate("CSRPointer clone values", that.nnz * LibMatrixCUDA.sizeOfDataType, false);
		that.rowPtr = getGPUContext().allocate("CSRPointer clone rowPtrs", (long) rows * Sizeof.INT, false);
		that.colInd = getGPUContext().allocate("CSRPointer clone colInd", that.nnz * Sizeof.INT, false);
		cudaMemcpy(that.val, me.val, that.nnz * LibMatrixCUDA.sizeOfDataType, cudaMemcpyDeviceToDevice);
		cudaMemcpy(that.rowPtr, me.rowPtr, (long) rows * Sizeof.INT, cudaMemcpyDeviceToDevice);
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
	@Deprecated
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
	 * @param rows           number of rows in this CSR matrix
	 * @param cols           number of columns in this CSR matrix
	 * @param instName          name of the invoking instruction to record{@link Statistics}.
	 * @return A {@link Pointer} to the allocated dense matrix (in column-major format)
	 */
	public Pointer toColumnMajorDenseMatrix(cusparseHandle cusparseHandle, int rows, int cols, String instName) {
		LOG.trace("GPU : sparse -> column major dense (inside CSRPointer) on " + this + ", GPUContext="
				+ getGPUContext());
		long size = rows * getDataTypeSizeOf(cols);
		Pointer A = getGPUContext().allocate(instName, size, false);
		// If this sparse block is empty, the allocated dense matrix, initialized to zero, will be returned.
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
