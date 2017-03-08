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

import jcuda.Pointer;
import jcuda.jcublas.JCublas2;
import jcuda.jcublas.cublasHandle;
import jcuda.jcusparse.JCusparse;
import jcuda.jcusparse.*;
import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;
import org.apache.sysml.runtime.DMLRuntimeException;
import org.apache.sysml.runtime.controlprogram.caching.CacheException;
import org.apache.sysml.runtime.controlprogram.caching.MatrixObject;
import org.apache.sysml.runtime.instructions.gpu.GPUInstruction;
import org.apache.sysml.runtime.matrix.data.*;
import org.apache.sysml.utils.GPUStatistics;
import org.apache.sysml.utils.LRUCacheMap;

import java.util.HashMap;

import static jcuda.jcublas.cublasOperation.CUBLAS_OP_T;
import static jcuda.jcusparse.JCusparse.*;
import static jcuda.jcusparse.cusparseIndexBase.CUSPARSE_INDEX_BASE_ZERO;
import static jcuda.jcusparse.cusparseMatrixType.CUSPARSE_MATRIX_TYPE_GENERAL;
import static jcuda.runtime.JCuda.*;
import static jcuda.runtime.cudaMemcpyKind.cudaMemcpyDeviceToHost;
import static jcuda.runtime.cudaMemcpyKind.cudaMemcpyHostToDevice;

/**
 * Handle to a matrix block on the GPU
 */
public class JCudaObject extends GPUObject {

	private static final Log LOG = LogFactory.getLog(JCudaObject.class.getName());

	/**
	 * Compressed Sparse Row (CSR) format for CUDA
	 * Generalized matrix multiply is implemented for CSR format in the cuSparse library among other operations
	 */
	public static class CSRPointer {
		
		public static cusparseMatDescr matrixDescriptor;
		
		/**
		 * @return Singleton default matrix descriptor object 
		 * 			(set with CUSPARSE_MATRIX_TYPE_GENERAL, CUSPARSE_INDEX_BASE_ZERO)
		 */
		public static cusparseMatDescr getDefaultCuSparseMatrixDescriptor(){
			if (matrixDescriptor == null){
				// Code from JCuda Samples - http://www.jcuda.org/samples/JCusparseSample.java
				matrixDescriptor = new cusparseMatDescr();
				cusparseCreateMatDescr(matrixDescriptor);
				cusparseSetMatType(matrixDescriptor, CUSPARSE_MATRIX_TYPE_GENERAL);
				cusparseSetMatIndexBase(matrixDescriptor, CUSPARSE_INDEX_BASE_ZERO);
			}
			return matrixDescriptor;
		}
		
		private static final double ULTRA_SPARSITY_TURN_POINT = 0.0004;

		/**
		 * Default constructor to help with Factory method {@link #allocateEmpty(long, long)}
		 */
		private CSRPointer() {
			val = new Pointer();
			rowPtr = new Pointer();
			colInd = new Pointer();
			allocateMatDescrPointer();
		}
		
		/** Number of non zeroes	 									*/
		public long nnz;
		/** double array of non zero values 							*/
		public Pointer val;
		/** integer array of start of all rows and end of last row + 1 	*/
		public Pointer rowPtr;
		/** integer array of nnz values' column indices					*/
		public Pointer colInd;
		/** descriptor of matrix, only CUSPARSE_MATRIX_TYPE_GENERAL supported	*/
		public cusparseMatDescr descr;
		
		/** 
		 * Check for ultra sparsity
		 * 
		 * @param rows number of rows
		 * @param cols number of columns
		 * @return true if ultra sparse
		 */
		public boolean isUltraSparse(int rows, int cols) {
			double sp = ((double)nnz/rows/cols);
			return sp<ULTRA_SPARSITY_TURN_POINT;
		}
		
		/**
		 * Initializes {@link #descr} to CUSPARSE_MATRIX_TYPE_GENERAL,
		 * the default that works for DGEMM.
		 */
		private void allocateMatDescrPointer() {			
			this.descr = getDefaultCuSparseMatrixDescriptor();
		}
		
		/**
		 * Estimate the size of a CSR matrix in GPU memory
		 * Size of pointers is not needed and is not added in
		 * @param nnz2	number of non zeroes
		 * @param rows	number of rows 
		 * @return size estimate
		 */
		public static long estimateSize(long nnz2, long rows) {
			long sizeofValArray = getDoubleSizeOf(nnz2);
			long sizeofRowPtrArray  = getIntSizeOf(rows + 1);
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
		 * Factory method to allocate an empty CSR Sparse matrix on the GPU
		 * @param nnz2	number of non-zeroes
		 * @param rows 	number of rows
		 * @return a {@link CSRPointer} instance that encapsulates the CSR matrix on GPU
		 * @throws DMLRuntimeException if DMLRuntimeException occurs
		 */
		public static CSRPointer allocateEmpty(long nnz2, long rows) throws DMLRuntimeException {
			assert nnz2 > -1 : "Incorrect usage of internal API, number of non zeroes is less than 0 when trying to allocate sparse data on GPU";
			CSRPointer r = new CSRPointer();
			r.nnz = nnz2;
			if(nnz2 == 0) {
				// The convention for an empty sparse matrix is to just have an instance of the CSRPointer object
				// with no memory allocated on the GPU.
				return r;
			}
			ensureFreeSpace(getDoubleSizeOf(nnz2) + getIntSizeOf(rows + 1) + getIntSizeOf(nnz2));
			// increment the cudaCount by 1 for the allocation of all 3 arrays
			r.val = allocate(null, getDoubleSizeOf(nnz2), 0);
			r.rowPtr = allocate(null, getIntSizeOf(rows + 1), 0);
			r.colInd = allocate(null, getIntSizeOf(nnz2), 1);
			return r;
		}
		
		/**
		 * Static method to copy a CSR sparse matrix from Host to Device
		 * @param dest	[input] destination location (on GPU)
		 * @param rows	number of rows
		 * @param nnz 	number of non-zeroes
		 * @param rowPtr	integer array of row pointers
		 * @param colInd	integer array of column indices
		 * @param values	double array of non zero values
		 */
		public static void copyToDevice(CSRPointer dest, int rows, long nnz, int[] rowPtr, int[] colInd, double[] values) {
			CSRPointer r = dest;
			long t0 = System.nanoTime();
			r.nnz = nnz;
			cudaMemcpy(r.rowPtr, Pointer.to(rowPtr), getIntSizeOf(rows + 1), cudaMemcpyHostToDevice);
			cudaMemcpy(r.colInd, Pointer.to(colInd), getIntSizeOf(nnz), cudaMemcpyHostToDevice);
			cudaMemcpy(r.val, Pointer.to(values), getDoubleSizeOf(nnz), cudaMemcpyHostToDevice);
			GPUStatistics.cudaToDevTime.addAndGet(System.nanoTime()-t0);
			GPUStatistics.cudaToDevCount.addAndGet(3);
		}
		
		/**
		 * Static method to copy a CSR sparse matrix from Device to host
		 * @param src	[input] source location (on GPU)
		 * @param rows	[input] number of rows
		 * @param nnz	[input] number of non-zeroes
		 * @param rowPtr	[output] pre-allocated integer array of row pointers of size (rows+1)
		 * @param colInd	[output] pre-allocated integer array of column indices of size nnz
		 * @param values	[output] pre-allocated double array of values of size nnz
		 */
		public static void copyToHost(CSRPointer src, int rows, long nnz, int[] rowPtr, int[] colInd, double[] values){
			CSRPointer r = src;
			long t0 = System.nanoTime();
			cudaMemcpy(Pointer.to(rowPtr), r.rowPtr, getIntSizeOf(rows + 1), cudaMemcpyDeviceToHost);
			cudaMemcpy(Pointer.to(colInd), r.colInd, getIntSizeOf(nnz), cudaMemcpyDeviceToHost);
			cudaMemcpy(Pointer.to(values), r.val, getDoubleSizeOf(nnz), cudaMemcpyDeviceToHost);
			GPUStatistics.cudaFromDevTime.addAndGet(System.nanoTime()-t0);
			GPUStatistics.cudaFromDevCount.addAndGet(3);
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

		/**
		 * Allocate row pointers of m+1 elements
		 * 
		 * @param handle	a valid {@link cusparseHandle}
		 * @param C			Output matrix
		 * @param rowsC			number of rows in C
		 * @throws DMLRuntimeException ?
		 */
		private static void step1AllocateRowPointers(cusparseHandle handle, CSRPointer C, int rowsC) throws DMLRuntimeException {
			cusparseSetPointerMode(handle, cusparsePointerMode.CUSPARSE_POINTER_MODE_HOST);
            cudaDeviceSynchronize();
			// Do not increment the cudaCount of allocations on GPU
			C.rowPtr = allocate(null, getIntSizeOf((long)rowsC+1), 0);
		}
		
		/**
		 * Determine total number of nonzero element for the cusparseDgeam  operation.
		 * This is done from either (nnzC=*nnzTotalDevHostPtr) or (nnzC=csrRowPtrC(m)-csrRowPtrC(0))
		 * 
		 * @param handle	a valid {@link cusparseHandle}
		 * @param A			Sparse Matrix A on GPU
		 * @param B			Sparse Matrix B on GPU
		 * @param C			Output Sparse Matrix C on GPU
		 * @param m			Rows in C
		 * @param n			Columns in C
		 * @throws DMLRuntimeException ?
		 */
		private static void step2GatherNNZGeam(cusparseHandle handle, CSRPointer A, CSRPointer B, CSRPointer C, int m, int n) throws DMLRuntimeException {
			int[] CnnzArray = { -1 };
			cusparseXcsrgeamNnz(handle, m, n,  
					A.descr, toIntExact(A.nnz), A.rowPtr, A.colInd, 
					B.descr, toIntExact(B.nnz), B.rowPtr, B.colInd, 
					C.descr, C.rowPtr, Pointer.to(CnnzArray));
            cudaDeviceSynchronize();
			if (CnnzArray[0] != -1){
				C.nnz = CnnzArray[0];
			}
			else {
		        int baseArray[] = { 0 };
		        cudaMemcpy(Pointer.to(CnnzArray), C.rowPtr.withByteOffset(getIntSizeOf(m)), getIntSizeOf(1), cudaMemcpyDeviceToHost);
	            cudaMemcpy(Pointer.to(baseArray), C.rowPtr,								   getIntSizeOf(1), cudaMemcpyDeviceToHost);
	            C.nnz = CnnzArray[0] - baseArray[0];
			}
		}

		/**
		 *	Determine total number of nonzero element for the cusparseDgemm operation.
		 *
		 * @param handle	a valid {@link cusparseHandle}
		 * @param A			Sparse Matrix A on GPU
		 * @param transA	op - whether A is transposed
		 * @param B			Sparse Matrix B on GPU
		 * @param transB	op - whether B is transposed
		 * @param C			Output Sparse Matrix C on GPU
		 * @param m			Number of rows of sparse matrix op ( A ) and C
		 * @param n			Number of columns of sparse matrix op ( B ) and C
		 * @param k			Number of columns/rows of sparse matrix op ( A ) / op ( B )
		 * @throws DMLRuntimeException ?
		 */
		private static void step2GatherNNZGemm(cusparseHandle handle, CSRPointer A, int transA, CSRPointer B, int transB, CSRPointer C, int m, int n, int k) throws DMLRuntimeException {
			int[] CnnzArray = { -1 };
			if (A.nnz >= Integer.MAX_VALUE || B.nnz >= Integer.MAX_VALUE) { 
				throw new DMLRuntimeException("Number of non zeroes is larger than supported by cuSparse"); 
			}
			cusparseXcsrgemmNnz(handle, transA, transB, m, n, k, 
					A.descr, toIntExact(A.nnz), A.rowPtr, A.colInd, 
					B.descr, toIntExact(B.nnz), B.rowPtr, B.colInd, 
					C.descr, C.rowPtr, Pointer.to(CnnzArray));
            cudaDeviceSynchronize();
			if (CnnzArray[0] != -1){
				C.nnz = CnnzArray[0];
			}
			else {
		        int baseArray[] = { 0 };
		        cudaMemcpy(Pointer.to(CnnzArray), C.rowPtr.withByteOffset(getIntSizeOf(m)), getIntSizeOf(1), cudaMemcpyDeviceToHost);
	            cudaMemcpy(Pointer.to(baseArray), C.rowPtr,								   getIntSizeOf(1), cudaMemcpyDeviceToHost);
	            C.nnz = CnnzArray[0] - baseArray[0];
			}
		}

		/**
		 * Allocate val and index pointers.
		 * 
		 * @param handle	a valid {@link cusparseHandle}
		 * @param C			Output sparse matrix on GPU
		 * @throws DMLRuntimeException ?
		 */
		private static void step3AllocateValNInd(cusparseHandle handle, CSRPointer C) throws DMLRuntimeException {
			// Increment cudaCount by one when all three arrays of CSR sparse array are allocated
			C.val = allocate(null, getDoubleSizeOf(C.nnz), 0);
			C.colInd = allocate(null, getIntSizeOf(C.nnz), 1);
		}

		// ==============================================================================================


		/**
		 * Estimates the number of non zero elements from the results of a sparse cusparseDgeam operation
		 * C = a op(A) + b op(B)
		 * @param handle 	a valid {@link cusparseHandle}
		 * @param A			Sparse Matrix A on GPU
		 * @param B			Sparse Matrix B on GPU
		 * @param m			Rows in A
		 * @param n			Columns in Bs
		 * @return CSR (compressed sparse row) pointer
		 * @throws DMLRuntimeException if DMLRuntimeException occurs
		 */
		public static CSRPointer allocateForDgeam(cusparseHandle handle, CSRPointer A, CSRPointer B, int m, int n) 
				throws DMLRuntimeException{
			if (A.nnz >= Integer.MAX_VALUE || B.nnz >= Integer.MAX_VALUE) { 
				throw new DMLRuntimeException("Number of non zeroes is larger than supported by cuSparse"); 
			}
			CSRPointer C = new CSRPointer();
			step1AllocateRowPointers(handle, C, m);
			step2GatherNNZGeam(handle, A, B, C, m, n);
			step3AllocateValNInd(handle, C);
			return C;
		}
		
		/**
		 * Estimates the number of non-zero elements from the result of a sparse matrix multiplication C = A * B
		 * and returns the {@link CSRPointer} to C with the appropriate GPU memory.
		 * @param handle	a valid {@link cusparseHandle}
		 * @param A			Sparse Matrix A on GPU
		 * @param transA	'T' if A is to be transposed, 'N' otherwise
		 * @param B			Sparse Matrix B on GPU
		 * @param transB	'T' if B is to be transposed, 'N' otherwise
		 * @param m			Rows in A
		 * @param n			Columns in B
		 * @param k			Columns in A / Rows in B
		 * @return a {@link CSRPointer} instance that encapsulates the CSR matrix on GPU
		 * @throws DMLRuntimeException if DMLRuntimeException occurs
		 */
		public static CSRPointer allocateForMatrixMultiply(cusparseHandle handle, CSRPointer A, int transA, CSRPointer B, int transB, int m, int n, int k) 
				throws DMLRuntimeException{
			// Following the code example at http://docs.nvidia.com/cuda/cusparse/#cusparse-lt-t-gt-csrgemm and at
			// https://github.com/jcuda/jcuda-matrix-utils/blob/master/JCudaMatrixUtils/src/test/java/org/jcuda/matrix/samples/JCusparseSampleDgemm.java
			CSRPointer C = new CSRPointer();
			step1AllocateRowPointers(handle, C, m);
			step2GatherNNZGemm(handle, A, transA, B, transB, C, m, n, k);
			step3AllocateValNInd(handle, C);
			return C;
		}
		
		/**
		 * Copies this CSR matrix on the GPU to a dense column-major matrix
		 * on the GPU. This is a temporary matrix for operations such as 
		 * cusparseDcsrmv.
		 * Since the allocated matrix is temporary, bookkeeping is not updated.
		 * The caller is responsible for calling "free" on the returned Pointer object
		 * @param cusparseHandle	a valid {@link cusparseHandle}
		 * @param cublasHandle 		a valid {@link cublasHandle}
		 * @param rows		number of rows in this CSR matrix
		 * @param cols		number of columns in this CSR matrix
		 * @return			A {@link Pointer} to the allocated dense matrix (in column-major format)
		 * @throws DMLRuntimeException if DMLRuntimeException occurs
		 */
		public Pointer toColumnMajorDenseMatrix(cusparseHandle cusparseHandle, cublasHandle cublasHandle, int rows, int cols) throws DMLRuntimeException {
			long size = ((long)rows) * getDoubleSizeOf((long)cols);
			Pointer A = JCudaObject.allocate(size);
			// If this sparse block is empty, the allocated dense matrix, initialized to zeroes, will be returned.
			if (val != null && rowPtr != null && colInd != null && nnz > 0) {
				// Note: cusparseDcsr2dense method cannot handle empty blocks
				cusparseDcsr2dense(cusparseHandle, rows, cols, descr, val, rowPtr, colInd, A, rows);
                cudaDeviceSynchronize();
			} else {
				LOG.warn("in CSRPointer, the values array, row pointers array or column indices array was null");
			}
			return A;
		}
		
		/**
		 * Calls cudaFree lazily on the allocated {@link Pointer} instances
		 */
		public void deallocate() {
			deallocate(false);
		}

		/**
		 * Calls cudaFree lazily or eagerly on the allocated {@link Pointer} instances
		 * @param eager whether to do eager or lazy cudaFrees
		 */
		public void deallocate(boolean eager){
			if (nnz > 0) {
				cudaFreeHelper(val, eager);
				cudaFreeHelper(rowPtr, eager);
				cudaFreeHelper(colInd, eager);
			}
		}
	};
	
	private static long getDoubleSizeOf(long numElems) {
		return numElems * ((long)jcuda.Sizeof.DOUBLE);
	}
	
	private static long getIntSizeOf(long numElems) {
		return numElems * ((long)jcuda.Sizeof.INT);
	}
	
	public synchronized boolean isAllocated() {
		return (jcudaDenseMatrixPtr != null || jcudaSparseMatrixPtr != null);
	}
	
	/** Pointer to dense matrix */
	public Pointer jcudaDenseMatrixPtr = null;
	/** Pointer to sparse matrix */
	public CSRPointer jcudaSparseMatrixPtr = null;

	public long numBytes;							/** Number of bytes occupied by this block on GPU */

	/**
	 * Initializes this JCudaObject with a {@link MatrixObject} instance which will contain metadata about the enclosing matrix block
	 * @param m
	 */
	JCudaObject(MatrixObject m) {
		super(m);
	}

	/**
	 * Allocates a sparse and empty {@link JCudaObject}
	 * This is the result of operations that are both non zero matrices.
	 * 
	 * @throws DMLRuntimeException if DMLRuntimeException occurs
	 */
	public void allocateSparseAndEmpty() throws DMLRuntimeException{
		setSparseMatrixCudaPointer(CSRPointer.allocateEmpty(0, mat.getNumRows()));
		setDeviceModify(0);
	}


	/**
	 * Allocates a dense matrix of size obtained from the attached matrix metadata
	 * and fills it up with a single value
	 * 
	 * @param v value to fill up the dense matrix
	 * @throws DMLRuntimeException if DMLRuntimeException occurs
	 */
	public void allocateAndFillDense(double v) throws DMLRuntimeException {
		long rows = mat.getNumRows();
		long cols = mat.getNumColumns();
		int numElems = toIntExact(rows * cols);
		long size = getDoubleSizeOf(numElems);
		setDenseMatrixCudaPointer(allocate(size));
		setDeviceModify(size);
		// The "fill" kernel is called which treats the matrix "jcudaDensePtr" like a vector and fills it with value "v"
		LibMatrixCUDA.kernels.launchKernel("fill", ExecutionConfig.getConfigForSimpleVectorOperations(numElems), jcudaDenseMatrixPtr, v, numElems);
	}

	/**
	 * If this {@link JCudaObject} is sparse and empty
	 * Being allocated is a prerequisite to being sparse and empty.
	 * 
	 * @return true if sparse and empty
	 */
	public boolean isSparseAndEmpty() {
		boolean isSparseAndAllocated = isAllocated()&& LibMatrixCUDA.isInSparseFormat(mat);
		boolean isEmptyAndSparseAndAllocated = isSparseAndAllocated && jcudaSparseMatrixPtr.nnz == 0;
		return isEmptyAndSparseAndAllocated;
	}

	@Override
	public synchronized boolean acquireDeviceRead() throws DMLRuntimeException {
		boolean transferred = false;
		if(!isAllocated()) {
			copyFromHostToDevice();
			transferred = true;
		} else {
			numLocks.addAndGet(1);
		}
		if(!isAllocated())
			throw new DMLRuntimeException("Expected device data to be allocated");
		return transferred;
	}
	
	@Override
	public synchronized boolean acquireDeviceModifyDense() throws DMLRuntimeException {
		boolean allocated = false;
		if(!isAllocated()) {
			mat.setDirty(true);
			// Dense block, size = numRows * numCols
			allocateDenseMatrixOnDevice();
			allocated = true;
			synchronized(evictionLock) {
				JCudaContext.allocatedPointers.add(this);
			}
		}
		isDeviceCopyModified = true;
		if(!isAllocated()) 
			throw new DMLRuntimeException("Expected device data to be allocated");
		return allocated;
	}
	
	@Override
	public synchronized boolean acquireDeviceModifySparse() throws DMLRuntimeException {
		boolean allocated = false;
		isInSparseFormat = true;
		if(!isAllocated()) {
			mat.setDirty(true);
			allocateSparseMatrixOnDevice();
			allocated = true;
			synchronized(evictionLock) {
				JCudaContext.allocatedPointers.add(this);
			}
		}
		isDeviceCopyModified = true;
		if(!isAllocated()) 
			throw new DMLRuntimeException("Expected device data to be allocated");
		return allocated;
	}
	
	@Override
	public synchronized boolean acquireHostRead() throws CacheException {
		boolean copied = false;
		if(isAllocated()) {
			try {
				if(isDeviceCopyModified) {
					copyFromDeviceToHost();
					copied = true;
				}
			} catch (DMLRuntimeException e) {
				throw new CacheException(e);
			}
		}
		else {
			throw new CacheException("Cannot perform acquireHostRead as the GPU data is not allocated:" + mat.getVarName());
		}
		return copied;
	}
	
	/**
	 * updates the locks depending on the eviction policy selected
	 * @throws CacheException if there is no locked GPU Object
	 */
	private void updateReleaseLocks() throws CacheException {
		if(numLocks.addAndGet(-1) < 0) {
            throw new CacheException("Redundant release of GPU object");
		}
		if(evictionPolicy == EvictionPolicy.LRU) {
            timestamp.set(System.nanoTime());
		}
		else if(evictionPolicy == EvictionPolicy.LFU) {
            timestamp.addAndGet(1);
		}
		else if(evictionPolicy == EvictionPolicy.MIN_EVICT) {
            // Do Nothing
		}
		else {
            throw new CacheException("The eviction policy is not supported:" + evictionPolicy.name());
		}
	}
	
	/**
	 * releases input allocated on GPU
	 * @throws CacheException if data is not allocated
	 */
	public synchronized void releaseInput() throws CacheException {
		updateReleaseLocks();
		if(!isAllocated())
			throw new CacheException("Attempting to release an input before allocating it");
	}

	/**
	@Override
	void allocateMemoryOnDevice(long numElemToAllocate) throws DMLRuntimeException {
		if(!isAllocated()) {
			long start = System.nanoTime();
			if(numElemToAllocate == -1 && LibMatrixCUDA.isInSparseFormat(mat)) {
				setSparseMatrixCudaPointer(CSRPointer.allocateEmpty(mat.getNnz(), mat.getNumRows()));
				numBytes = CSRPointer.estimateSize(mat.getNnz(), mat.getNumRows());
				JCudaContext.deviceMemBytes.addAndGet(-numBytes);
				isInSparseFormat = true;
				//throw new DMLRuntimeException("Sparse format not implemented");
			} else if(numElemToAllocate == -1) {
				// Called for dense input
				setDenseMatrixCudaPointer(new Pointer());
				numBytes = mat.getNumRows()*getDoubleSizeOf(mat.getNumColumns());
				cudaMalloc(jcudaDenseMatrixPtr, numBytes);
				JCudaContext.deviceMemBytes.addAndGet(-numBytes);
			}
			else {
				// Called for dense output
				setDenseMatrixCudaPointer(new Pointer());
				numBytes = getDoubleSizeOf(numElemToAllocate);
				if(numElemToAllocate <= 0 || numBytes <= 0)
					throw new DMLRuntimeException("Cannot allocate dense matrix object with " + numElemToAllocate + " elements and size " + numBytes);
				cudaMalloc(jcudaDenseMatrixPtr,  numBytes);
				JCudaContext.deviceMemBytes.addAndGet(-numBytes);
			}

			GPUStatistics.cudaAllocTime.addAndGet(System.nanoTime()-start);
			GPUStatistics.cudaAllocCount.addAndGet(1);

		}
	}
	 */

	@Override
	void allocateDenseMatrixOnDevice() throws DMLRuntimeException {
		assert !isAllocated() : "Internal error - trying to allocated dense matrix to a JCudaObject that is already allocated";
		long rows = mat.getNumRows();
		long cols = mat.getNumColumns();
		assert rows > 0 : "Internal error - invalid number of rows when allocating dense matrix";
		assert cols > 0 : "Internal error - invalid number of columns when allocating dense matrix;";
        long size = getDoubleSizeOf(rows * cols);
		Pointer tmp = allocate(size);
		setDenseMatrixCudaPointer(tmp);
		setDeviceModify(size);
	}

	@Override
	void allocateSparseMatrixOnDevice() throws DMLRuntimeException {
		assert !isAllocated() : "Internal error = trying to allocated sparse matrix to a JCudaObject that is already allocated";
		long rows = mat.getNumRows();
		long nnz = mat.getNnz();
		assert rows > 0 : "Internal error - invalid number of rows when allocating a sparse matrix";
		assert nnz > 0 : "Internal error - invalid number of non zeroes when allocating a sparse matrix";
		CSRPointer tmp = CSRPointer.allocateEmpty(nnz, rows);
		setSparseMatrixCudaPointer(tmp);
		long size = CSRPointer.estimateSize(nnz, rows);
		setDeviceModify(size);
	}

	/**
	 * releases output allocated on GPU
	 * @throws CacheException if data is not allocated
	 */
    @Override
	public synchronized void releaseOutput() throws CacheException {
		updateReleaseLocks();
		isDeviceCopyModified = true;
		if(!isAllocated())
			throw new CacheException("Attempting to release an output before allocating it");
	}

	@Override
	public void setDeviceModify(long numBytes) {
		this.numLocks.addAndGet(1);
		this.numBytes = numBytes;
		((JCudaContext)GPUContext.currContext).getAndAddAvailableMemory(-numBytes);
	}

	@Override
	void deallocateMemoryOnDevice(boolean eager) {
		if(jcudaDenseMatrixPtr != null) {
			long start = System.nanoTime();
			cudaFreeHelper(null, jcudaDenseMatrixPtr, eager);
			((JCudaContext)GPUContext.currContext).getAndAddAvailableMemory(numBytes);
			GPUStatistics.cudaDeAllocTime.addAndGet(System.nanoTime()-start);
			GPUStatistics.cudaDeAllocCount.addAndGet(1);
		}
		if (jcudaSparseMatrixPtr != null) {
			long start = System.nanoTime();
			jcudaSparseMatrixPtr.deallocate(eager);
			((JCudaContext)GPUContext.currContext).getAndAddAvailableMemory(numBytes);
			GPUStatistics.cudaDeAllocTime.addAndGet(System.nanoTime()-start);
			GPUStatistics.cudaDeAllocCount.addAndGet(1);
		}
		jcudaDenseMatrixPtr = null;
		jcudaSparseMatrixPtr = null;
		numLocks.set(0);
	}
	
	/** 
	 * Thin wrapper over {@link #evict(long)}
	 * @param size size to check
	 * @throws DMLRuntimeException if DMLRuntimeException occurs
	 */
	static void ensureFreeSpace(long size) throws DMLRuntimeException {
		ensureFreeSpace(null, size);
	}

	/**
	 * Thin wrapper over {@link #evict(long)}
	 * @param instructionName instructionName name of the instruction for which performance measurements are made
	 * @param size size to check
	 * @throws DMLRuntimeException if DMLRuntimeException occurs
	 */
	static void ensureFreeSpace(String instructionName, long size) throws DMLRuntimeException {
		if(size >= getAvailableMemory()) {
			evict(instructionName, size);
		}
	}
	
	@Override
	void copyFromHostToDevice() 
		throws DMLRuntimeException 
	{
		printCaller();
		long start = System.nanoTime();
		
		MatrixBlock tmp = mat.acquireRead();
		if(tmp.isInSparseFormat()) {
			
			int rowPtr[] = null;
			int colInd[] = null;
			double[] values = null;
			
			tmp.recomputeNonZeros();
			long nnz = tmp.getNonZeros();
			mat.getMatrixCharacteristics().setNonZeros(nnz);
			
			SparseBlock block = tmp.getSparseBlock();
			boolean copyToDevice = true;
			if(block == null && tmp.getNonZeros() == 0) {
//				// Allocate empty block --> not necessary
//				// To reproduce this, see org.apache.sysml.test.integration.applications.dml.ID3DMLTest
//				rowPtr = new int[0];
//				colInd = new int[0];
//				values = new double[0];
				copyToDevice = false;
			}
			else if(block == null && tmp.getNonZeros() != 0) {
				throw new DMLRuntimeException("Expected CP sparse block to be not null.");
			}
			else {
				// CSR is the preferred format for cuSparse GEMM
				// Converts MCSR and COO to CSR
				SparseBlockCSR csrBlock = null;
				if (block instanceof SparseBlockCSR){ 
					csrBlock = (SparseBlockCSR)block;
				} else if (block instanceof SparseBlockCOO) {
					// TODO - should we do this on the GPU using cusparse<t>coo2csr() ?
					long t0 = System.nanoTime();
					SparseBlockCOO cooBlock = (SparseBlockCOO)block;
					csrBlock = new SparseBlockCSR(toIntExact(mat.getNumRows()), cooBlock.rowIndexes(), cooBlock.indexes(), cooBlock.values());
					GPUStatistics.cudaSparseConversionTime.addAndGet(System.nanoTime() - t0);
					GPUStatistics.cudaSparseConversionCount.incrementAndGet();
				} else if (block instanceof SparseBlockMCSR) {
					long t0 = System.nanoTime();
					SparseBlockMCSR mcsrBlock = (SparseBlockMCSR)block;
					csrBlock = new SparseBlockCSR(mcsrBlock.getRows(), toIntExact(mcsrBlock.size()));
					GPUStatistics.cudaSparseConversionTime.addAndGet(System.nanoTime() - t0);
					GPUStatistics.cudaSparseConversionCount.incrementAndGet();
				} else {
					throw new DMLRuntimeException("Unsupported sparse matrix format for CUDA operations");
				}
				rowPtr = csrBlock.rowPointers();
				colInd = csrBlock.indexes();
				values = csrBlock.values();	
			}
			allocateSparseMatrixOnDevice();
			synchronized(evictionLock) {
				JCudaContext.allocatedPointers.add(this);
			}
			if(copyToDevice) {
				CSRPointer.copyToDevice(jcudaSparseMatrixPtr, tmp.getNumRows(), tmp.getNonZeros(), rowPtr, colInd, values);
			}
			// throw new DMLRuntimeException("Sparse matrix is not implemented");
			// tmp.sparseToDense();
		}
		else {
			double[] data = tmp.getDenseBlock();
			
			if( data == null && tmp.getSparseBlock() != null )
				throw new DMLRuntimeException("Incorrect sparsity calculation");
			else if( data==null && tmp.getNonZeros() != 0 )
				throw new DMLRuntimeException("MatrixBlock is not allocated");
			else if( tmp.getNonZeros() == 0 )
				data = new double[tmp.getNumRows()*tmp.getNumColumns()];
			
			// Copy dense block
			allocateDenseMatrixOnDevice();
			synchronized(evictionLock) {
				JCudaContext.allocatedPointers.add(this);
			}
			cudaMemcpy(jcudaDenseMatrixPtr, Pointer.to(data), getDoubleSizeOf(mat.getNumRows()*mat.getNumColumns()), cudaMemcpyHostToDevice);
		}
		
		mat.release();
		
		GPUStatistics.cudaToDevTime.addAndGet(System.nanoTime()-start);
		GPUStatistics.cudaToDevCount.addAndGet(1);
	}
	
	public static int toIntExact(long l) throws DMLRuntimeException {
	    if (l < Integer.MIN_VALUE || l > Integer.MAX_VALUE) {
	        throw new DMLRuntimeException("Cannot be cast to int:" + l);
	    }
	    return (int) l;
	}

	@Override
	protected void copyFromDeviceToHost() throws DMLRuntimeException {
		if (jcudaDenseMatrixPtr != null && jcudaSparseMatrixPtr != null){
			throw new DMLRuntimeException("Invalid state : JCuda dense/sparse pointer are both allocated");
		}

		if(jcudaDenseMatrixPtr != null) {
			printCaller();
			long start = System.nanoTime();
			MatrixBlock tmp = new MatrixBlock(toIntExact(mat.getNumRows()), toIntExact(mat.getNumColumns()), false);
			tmp.allocateDenseBlock();
			double [] data = tmp.getDenseBlock();
			
			cudaMemcpy(Pointer.to(data), jcudaDenseMatrixPtr, getDoubleSizeOf(data.length), cudaMemcpyDeviceToHost);
			
			tmp.recomputeNonZeros();
			mat.acquireModify(tmp);
			mat.release();
			
			GPUStatistics.cudaFromDevTime.addAndGet(System.nanoTime()-start);
			GPUStatistics.cudaFromDevCount.addAndGet(1);
		}
		else if (jcudaSparseMatrixPtr != null){
			printCaller();
			if(!LibMatrixCUDA.isInSparseFormat(mat))
				throw new DMLRuntimeException("Block not in sparse format on host yet the device sparse matrix pointer is not null");

			if(this.isSparseAndEmpty()){
				MatrixBlock tmp = new MatrixBlock();	// Empty Block
				mat.acquireModify(tmp);
				mat.release();
			} else {
				long start = System.nanoTime();

				int rows = toIntExact(mat.getNumRows());
				int cols = toIntExact(mat.getNumColumns());
				int nnz = toIntExact(jcudaSparseMatrixPtr.nnz);
				int[] rowPtr = new int[rows + 1];
				int[] colInd = new int[nnz];
				double[] values = new double[nnz];
				CSRPointer.copyToHost(jcudaSparseMatrixPtr, rows, nnz, rowPtr, colInd, values);

				SparseBlockCSR sparseBlock = new SparseBlockCSR(rowPtr, colInd, values, nnz);
				MatrixBlock tmp = new MatrixBlock(rows, cols, nnz, sparseBlock);
				mat.acquireModify(tmp);
				mat.release();
				GPUStatistics.cudaFromDevTime.addAndGet(System.nanoTime() - start);
				GPUStatistics.cudaFromDevCount.addAndGet(1);
			}
		}
		else {
			throw new DMLRuntimeException("Cannot copy from device to host as JCuda dense/sparse pointer is not allocated");
		}
		isDeviceCopyModified = false;
	}

	@Override
	protected long getSizeOnDevice() throws DMLRuntimeException {
		long GPUSize = 0;
		long rlen = mat.getNumRows();
		long clen = mat.getNumColumns();
		long nnz = mat.getNnz();

		if(LibMatrixCUDA.isInSparseFormat(mat)) {
			GPUSize = CSRPointer.estimateSize(nnz, rlen);
		}
		else {
			GPUSize = getDoubleSizeOf(rlen * clen);
		}
		return GPUSize;
	}
	
	private String getClassAndMethod(StackTraceElement st) {
		String [] str = st.getClassName().split("\\.");
		return str[str.length - 1] + "." + st.getMethodName();
	}
	
	/**
	 * Convenience debugging method.
	 * Checks {@link JCudaContext#DEBUG} flag before printing to System.out
	 */
	private void printCaller() {
		if(JCudaContext.DEBUG) {
			StackTraceElement[] st = Thread.currentThread().getStackTrace();
			String ret = getClassAndMethod(st[1]);
			for (int i = 2; i < st.length && i < 7; i++) {
				ret += "->" + getClassAndMethod(st[i]);
			}
			System.out.println("CALL_STACK:" + ret);
		}
	}
	
	/**
	 * Convenience method to directly examine the Sparse matrix on GPU
	 * 
	 * @return CSR (compressed sparse row) pointer
	 */
	public CSRPointer getSparseMatrixCudaPointer() {
		return jcudaSparseMatrixPtr;
	}
	
	/**
	 * Convenience method to directly set the sparse matrix on GPU
	 * Make sure to call {@link #setDeviceModify(long)} after this to set appropriate state, if you are not sure what you are doing.
	 * Needed for operations like {@link JCusparse#cusparseDcsrgemm(cusparseHandle, int, int, int, int, int, cusparseMatDescr, int, Pointer, Pointer, Pointer, cusparseMatDescr, int, Pointer, Pointer, Pointer, cusparseMatDescr, Pointer, Pointer, Pointer)}
	 * @param sparseMatrixPtr CSR (compressed sparse row) pointer
	 */
	public synchronized void setSparseMatrixCudaPointer(CSRPointer sparseMatrixPtr) {
		this.jcudaSparseMatrixPtr = sparseMatrixPtr;
		this.isInSparseFormat = true;
		if(jcudaDenseMatrixPtr != null) {
			cudaFreeHelper(jcudaDenseMatrixPtr);
			jcudaDenseMatrixPtr = null;
		}
	}

	/**
	 * Convenience method to directly set the dense matrix pointer on GPU
	 * Make sure to call {@link #setDeviceModify(long)} after this to set appropriate state, if you are not sure what you are doing.
	 * 
	 * @param densePtr dense pointer
	 */
	public synchronized void setDenseMatrixCudaPointer(Pointer densePtr){
		this.jcudaDenseMatrixPtr = densePtr;
		this.isInSparseFormat = false;
		if(jcudaSparseMatrixPtr != null) {
			jcudaSparseMatrixPtr.deallocate();
			jcudaSparseMatrixPtr = null;
		}
	}
	
	/**
	 * Converts this JCudaObject from dense to sparse format.
	 * 
	 * @throws DMLRuntimeException if DMLRuntimeException occurs
	 */
	public void denseToSparse() throws DMLRuntimeException {
		long t0 = System.nanoTime();
		cusparseHandle cusparseHandle = LibMatrixCUDA.cusparseHandle;
		if(cusparseHandle == null)
			throw new DMLRuntimeException("Expected cusparse to be initialized");
		int rows = toIntExact(mat.getNumRows());
		int cols = toIntExact(mat.getNumColumns());
		
		if(jcudaDenseMatrixPtr == null || !isAllocated())
			throw new DMLRuntimeException("Expected allocated dense matrix before denseToSparse() call");

		convertDensePtrFromRowMajorToColumnMajor();
		setSparseMatrixCudaPointer(columnMajorDenseToRowMajorSparse(cusparseHandle, rows, cols, jcudaDenseMatrixPtr));
		// TODO: What if mat.getNnz() is -1 ?
		numBytes = CSRPointer.estimateSize(mat.getNnz(), rows);
		GPUStatistics.cudaDenseToSparseTime.addAndGet(System.nanoTime() - t0);
		GPUStatistics.cudaDenseToSparseCount.addAndGet(1);
	}

	/**
	 * Transposes a dense matrix on the GPU by calling the cublasDgeam operation
	 * @param densePtr	Pointer to dense matrix on the GPU
	 * @param m			rows in ouput matrix
	 * @param n			columns in output matrix
	 * @param lda		rows in input matrix
	 * @param ldc		columns in output matrix
	 * @return			transposed matrix
	 * @throws DMLRuntimeException if operation failed
	 */
	public static Pointer transpose(Pointer densePtr, int m, int n, int lda, int ldc) throws DMLRuntimeException {
		Pointer alpha = LibMatrixCUDA.pointerTo(1.0);
		Pointer beta = LibMatrixCUDA.pointerTo(0.0);
		Pointer A = densePtr;
		Pointer C = JCudaObject.allocate(((long)m)*getDoubleSizeOf(n));

		// Transpose the matrix to get a dense matrix
		JCublas2.cublasDgeam(LibMatrixCUDA.cublasHandle, CUBLAS_OP_T, CUBLAS_OP_T, m, n, alpha, A, lda, beta, new Pointer(), lda, C, ldc);
		return C;
	}

	/**
	 * Convenience method. Converts Row Major Dense Matrix --> Column Major Dense Matrix
	 * @throws DMLRuntimeException if DMLRuntimeException occurs
	 */
	private void convertDensePtrFromRowMajorToColumnMajor() throws DMLRuntimeException {
		int m = toIntExact(mat.getNumRows());
		int n = toIntExact(mat.getNumColumns());
		int lda = n;
		int ldc = m;
		if(!isAllocated()) {
			throw new DMLRuntimeException("Error in converting row major to column major : data is not allocated");
		}

		Pointer tmp = transpose(jcudaDenseMatrixPtr, m, n, lda, ldc);
		cudaFreeHelper(jcudaDenseMatrixPtr);
		setDenseMatrixCudaPointer(tmp);
	}

	private void convertDensePtrFromColMajorToRowMajor() throws DMLRuntimeException {
		int n = toIntExact(mat.getNumRows());
		int m = toIntExact(mat.getNumColumns());
		int lda = n;
	    int ldc = m;
		if(!isAllocated()) {
			throw new DMLRuntimeException("Error in converting column major to row major : data is not allocated");
		}

		Pointer tmp = transpose(jcudaDenseMatrixPtr, m, n, lda, ldc);
		cudaFreeHelper(jcudaDenseMatrixPtr);
		setDenseMatrixCudaPointer(tmp);
	}
	
	/**
	 * Convert sparse to dense (Performs transpose, use sparseToColumnMajorDense if the kernel can deal with column major format)
	 * 
	 * @throws DMLRuntimeException if DMLRuntimeException occurs
	 */
	public void sparseToDense() throws DMLRuntimeException {
		sparseToDense(null);
	}

	/**
	 * Convert sparse to dense (Performs transpose, use sparseToColumnMajorDense if the kernel can deal with column major format)
	 * Also records per instruction invokation of sparseToDense.
	 * @param instructionName	Name of the instruction for which statistics are recorded in {@link GPUStatistics}
	 * @throws DMLRuntimeException ?
	 */
	public void sparseToDense(String instructionName) throws DMLRuntimeException {
		long start = System.nanoTime();
		if(jcudaSparseMatrixPtr == null || !isAllocated())
			throw new DMLRuntimeException("Expected allocated sparse matrix before sparseToDense() call");

		sparseToColumnMajorDense();
		convertDensePtrFromColMajorToRowMajor();
		long end = System.nanoTime();
		if (instructionName != null && GPUStatistics.DISPLAY_STATISTICS) GPUStatistics.maintainCPMiscTimes(instructionName, GPUInstruction.MISC_TIMER_SPARSE_TO_DENSE, end - start);
		GPUStatistics.cudaSparseToDenseTime.addAndGet(end - start);
		GPUStatistics.cudaSparseToDenseCount.addAndGet(1);
	}
	
	/**
	 * More efficient method to convert sparse to dense but returns dense in column major format
	 * 
	 * @throws DMLRuntimeException if DMLRuntimeException occurs
	 */
	public void sparseToColumnMajorDense() throws DMLRuntimeException {
		if(jcudaSparseMatrixPtr == null || !isAllocated())
			throw new DMLRuntimeException("Expected allocated sparse matrix before sparseToDense() call");
		
		cusparseHandle cusparseHandle = LibMatrixCUDA.cusparseHandle;
		if(cusparseHandle == null)
			throw new DMLRuntimeException("Expected cusparse to be initialized");
		int rows = toIntExact(mat.getNumRows());
		int cols = toIntExact(mat.getNumColumns());
		setDenseMatrixCudaPointer(jcudaSparseMatrixPtr.toColumnMajorDenseMatrix(cusparseHandle, null, rows, cols));
		numBytes = ((long)rows)*getDoubleSizeOf(cols);
	}
	
	/**
	 * Convenience method to convert a CSR matrix to a dense matrix on the GPU
	 * Since the allocated matrix is temporary, bookkeeping is not updated.
	 * Also note that the input dense matrix is expected to be in COLUMN MAJOR FORMAT
	 * Caller is responsible for deallocating memory on GPU.
	 * 
	 * @param cusparseHandle handle to cusparse library
	 * @param rows number of rows
	 * @param cols number of columns
	 * @param densePtr [in] dense matrix pointer on the GPU in row major
	 * @return CSR (compressed sparse row) pointer
	 * @throws DMLRuntimeException if DMLRuntimeException occurs
	 */
	public static CSRPointer columnMajorDenseToRowMajorSparse(cusparseHandle cusparseHandle, int rows, int cols, Pointer densePtr) throws DMLRuntimeException {
		cusparseMatDescr matDescr = CSRPointer.getDefaultCuSparseMatrixDescriptor();
		Pointer nnzPerRowPtr = new Pointer();
		Pointer nnzTotalDevHostPtr = new Pointer();
		
		ensureFreeSpace(getIntSizeOf(rows + 1));
		
		long t1 = System.nanoTime();
		nnzPerRowPtr = allocate(getIntSizeOf(rows));
		nnzTotalDevHostPtr = allocate(getIntSizeOf(1));
		GPUStatistics.cudaAllocTime.addAndGet(System.nanoTime() - t1);
		GPUStatistics.cudaAllocCount.addAndGet(2);
		
		// Output is in dense vector format, convert it to CSR
		cusparseDnnz(cusparseHandle, cusparseDirection.CUSPARSE_DIRECTION_ROW, rows, cols, matDescr, densePtr, rows, nnzPerRowPtr, nnzTotalDevHostPtr);
		cudaDeviceSynchronize();
		int[] nnzC = {-1};
		
		long t2 = System.nanoTime();
		cudaMemcpy(Pointer.to(nnzC), nnzTotalDevHostPtr, getIntSizeOf(1), cudaMemcpyDeviceToHost);
		GPUStatistics.cudaFromDevTime.addAndGet(System.nanoTime() - t2);
		GPUStatistics.cudaFromDevCount.addAndGet(2);
		
		if (nnzC[0] == -1){
			throw new DMLRuntimeException("cusparseDnnz did not calculate the correct number of nnz from the sparse-matrix vector mulitply on the GPU");
		}
		
		CSRPointer C = CSRPointer.allocateEmpty(nnzC[0], rows);		
		cusparseDdense2csr(cusparseHandle, rows, cols, matDescr, densePtr, rows, nnzPerRowPtr, C.val, C.rowPtr, C.colInd);
		cudaDeviceSynchronize();

		cudaFreeHelper(nnzPerRowPtr);
		cudaFreeHelper(nnzTotalDevHostPtr);
		
		return C;
	}


	/** Map of free blocks allocate on GPU. maps size_of_block -> pointer on GPU */
	static LRUCacheMap<Long, Pointer> freeCUDASpaceMap = new LRUCacheMap<Long, Pointer>();
	/** To record size of allocated blocks */
	static HashMap<Pointer, Long> cudaBlockSizeMap = new HashMap<Pointer, Long>();


	/**
	 * Convenience method for {@link #allocate(String, long, int)}, defaults statsCount to 1.
	 * @param size size of data (in bytes) to allocate
	 * @return jcuda pointer
	 * @throws DMLRuntimeException if DMLRuntimeException occurs
	 */
	public static Pointer allocate(long size) throws DMLRuntimeException {
		return allocate(null, size, 1);
	}

	/**
	 * Convenience method for {@link #allocate(String, long, int)}, defaults statsCount to 1.
	 * @param instructionName name of instruction for which to record per instruction performance statistics, null if don't want to record
	 * @param size size of data (in bytes) to allocate
	 * @return jcuda pointer
	 * @throws DMLRuntimeException if DMLRuntimeException occurs
	 */
	public static Pointer allocate(String instructionName, long size) throws DMLRuntimeException {
		return allocate(instructionName, size, 1);
	}

	/**
	 * Allocates temporary space on the device.
	 * Does not update bookkeeping.
	 * The caller is responsible for freeing up after usage.
	 * @param instructionName name of instruction for which to record per instruction performance statistics, null if don't want to record
	 * @param size   			Size of data (in bytes) to allocate
	 * @param statsCount	amount to increment the cudaAllocCount by
	 * @return jcuda Pointer
	 * @throws DMLRuntimeException if DMLRuntimeException occurs
	 */
	public static Pointer allocate(String instructionName, long size, int statsCount) throws DMLRuntimeException{
		synchronized (JCudaContext.syncObj) {
			Pointer A;
			if (freeCUDASpaceMap.containsKey(size)) {
				A = freeCUDASpaceMap.get(size);
				freeCUDASpaceMap.remove(size);
			} else {
				long t0 = System.nanoTime();
				ensureFreeSpace(instructionName, size);
				A = new Pointer();
				cudaMalloc(A, size);
				((JCudaContext)(JCudaContext.currContext)).deviceMemBytes.addAndGet(size);
				GPUStatistics.cudaAllocTime.getAndAdd(System.nanoTime() - t0);
				GPUStatistics.cudaAllocCount.getAndAdd(statsCount);
				if (instructionName != null && GPUStatistics.DISPLAY_STATISTICS) GPUStatistics.maintainCPMiscTimes(instructionName, GPUInstruction.MISC_TIMER_ALLOCATE, System.nanoTime() - t0);
			}
			// Set all elements to 0 since newly allocated space will contain garbage
			cudaMemset(A, 0, size);
			cudaBlockSizeMap.put(A, size);
			return A;
		}
	}

	/**
	 * Does lazy cudaFree calls
	 * @param toFree {@link Pointer} instance to be freed
	 */
	public static void cudaFreeHelper(final Pointer toFree) {
		cudaFreeHelper(null, toFree, false);
	}

	/**
	 * does lazy/eager cudaFree calls
	 * @param toFree {@link Pointer} instance to be freed
	 * @param eager true if to be done eagerly
	 * @throws DMLRuntimeException
	 */
	public static void cudaFreeHelper(final Pointer toFree, boolean eager) {
		cudaFreeHelper(null, toFree, eager);
	}

	/**
	 * Does lazy cudaFree calls
	 * @param instructionName name of the instruction for which to record per instruction free time, null if do not want to record
	 * @param toFree {@link Pointer} instance to be freed
	 */
	public static void cudaFreeHelper(String instructionName, final Pointer toFree) {
		cudaFreeHelper(instructionName, toFree, false);
	}

	/**
	 * Does cudaFree calls, lazily
	 * @param instructionName name of the instruction for which to record per instruction free time, null if do not want to record
	 * @param toFree {@link Pointer} instance to be freed
	 * @param eager true if to be done eagerly
	 */
	@SuppressWarnings("rawtypes")
	public static void cudaFreeHelper(String instructionName, final Pointer toFree, boolean eager){
		long t0 = 0;
		assert cudaBlockSizeMap.containsKey(toFree) : "ERROR : Internal state corrupted, cache block size map is not aware of a block it trying to free up";
		long size = cudaBlockSizeMap.get(toFree);
		if (eager) {
			if (instructionName != null) t0 = System.nanoTime();
			((JCudaContext)(JCudaContext.currContext)).deviceMemBytes.addAndGet(-size);
			cudaFree(toFree);
			cudaBlockSizeMap.remove(toFree);
			if (instructionName != null && GPUStatistics.DISPLAY_STATISTICS) GPUStatistics.maintainCPMiscTimes(instructionName, GPUInstruction.MISC_TIMER_CUDA_FREE, System.nanoTime() - t0);
		} else {
			freeCUDASpaceMap.put(size, toFree);
		}
	}


	/**
	 * Gets the double array from GPU memory onto host memory and returns string.
	 * @param A Pointer to memory on device (GPU), assumed to point to a double array
	 * @param rows rows in matrix A
     * @param cols columns in matrix A
	 * @return the debug string
	 * @throws DMLRuntimeException  if DMLRuntimeException occurs
	 */
	public static String debugString(Pointer A, long rows, long cols) throws DMLRuntimeException {
		StringBuffer sb = new StringBuffer();
        int len = toIntExact(rows * cols);
		double[] tmp = new double[len];
		cudaMemcpy(Pointer.to(tmp), A, getDoubleSizeOf(len), cudaMemcpyDeviceToHost);
        int k = 0;
		for (int i=0; i<rows; i++){
            for (int j=0; j<cols; j++){
			   sb.append(tmp[k]).append(' ');
               k++;
            }
            sb.append('\n');
		}
		return sb.toString();
	}
}
