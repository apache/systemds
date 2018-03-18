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
import static jcuda.runtime.JCuda.cudaMemcpy;
import static jcuda.runtime.JCuda.cudaMemset;
import static jcuda.runtime.cudaMemcpyKind.cudaMemcpyDeviceToDevice;
import static jcuda.runtime.cudaMemcpyKind.cudaMemcpyDeviceToHost;
import java.util.concurrent.atomic.AtomicLong;
import java.util.concurrent.atomic.LongAdder;

import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;
import org.apache.sysml.api.DMLScript;
import org.apache.sysml.runtime.DMLRuntimeException;
import org.apache.sysml.runtime.controlprogram.caching.CacheException;
import org.apache.sysml.runtime.controlprogram.caching.MatrixObject;
import org.apache.sysml.runtime.instructions.cp.CPInstruction;
import org.apache.sysml.runtime.instructions.gpu.GPUInstruction;
import org.apache.sysml.runtime.matrix.data.LibMatrixCUDA;
import org.apache.sysml.runtime.matrix.data.MatrixBlock;
import org.apache.sysml.runtime.matrix.data.SparseBlock;
import org.apache.sysml.runtime.matrix.data.SparseBlockCOO;
import org.apache.sysml.runtime.matrix.data.SparseBlockCSR;
import org.apache.sysml.runtime.matrix.data.SparseBlockMCSR;
import org.apache.sysml.utils.GPUStatistics;

import jcuda.Pointer;
import jcuda.jcusparse.cusparseDirection;
import jcuda.jcusparse.cusparseHandle;
import jcuda.jcusparse.cusparseMatDescr;

/**
 * Handle to a matrix block on the GPU
 */
public class GPUObject {

	private static final Log LOG = LogFactory.getLog(GPUObject.class.getName());

	/**
	 * GPUContext that owns this GPUObject
	 */
	private final GPUContext gpuContext;

	/**
	 * Pointer to the underlying dense matrix block on GPU
	 */
	private Pointer jcudaDenseMatrixPtr = null;

	/**
	 * Pointer to the underlying sparse matrix block on GPU
	 */
	private CSRPointer jcudaSparseMatrixPtr = null;

	/**
	 * whether the block attached to this {@link GPUContext} is dirty on the device and needs to be copied back to host
	 */
	protected boolean dirty = false;

	/**
	 * number of read locks on this object (this GPUObject is being used in a current instruction)
	 */
	protected LongAdder readLocks = new LongAdder();
	
	/**
	 * whether write lock on this object (this GPUObject is being used in a current instruction)
	 */
	protected boolean writeLock = false;

	/**
	 * Timestamp, needed by {@link GPUContext#evict(long)}
	 */
	AtomicLong timestamp = new AtomicLong();

	/**
	 * Whether this block is in sparse format
	 */
	protected boolean isSparse = false;

	/**
	 * Enclosing {@link MatrixObject} instance
	 */
	protected MatrixObject mat = null;

	//	private Pointer allocate(String instName, long size) throws DMLRuntimeException {
	//		return getGPUContext().allocate(instName, size);
	//	}

	@Override
	public Object clone() {
		GPUObject me = this;
		GPUObject that = new GPUObject(me.gpuContext, me.mat);
		that.dirty = me.dirty;
		// The only place clone is getting called: LibMatrixCUDA's solve
		that.readLocks.reset();
		that.writeLock = false;
		
		that.timestamp = new AtomicLong(me.timestamp.get());
		that.isSparse = me.isSparse;

		try {
			if (me.jcudaDenseMatrixPtr != null) {
				long rows = me.mat.getNumRows();
				long cols = me.mat.getNumColumns();
				long size = rows * cols * LibMatrixCUDA.sizeOfDataType;
				that.jcudaDenseMatrixPtr = allocate(size);
				cudaMemcpy(that.jcudaDenseMatrixPtr, me.jcudaDenseMatrixPtr, size, cudaMemcpyDeviceToDevice);
			}

			if (me.getJcudaSparseMatrixPtr() != null) {
				long rows = mat.getNumRows();
				that.jcudaSparseMatrixPtr = me.jcudaSparseMatrixPtr.clone((int) rows);
			}

		} catch (DMLRuntimeException e) {
			throw new RuntimeException(e);
		}

		return that;
	}

	private Pointer allocate(long size) throws DMLRuntimeException {
		return getGPUContext().allocate(size);
	}

	private void cudaFreeHelper(Pointer toFree) throws DMLRuntimeException {
		getGPUContext().cudaFreeHelper(toFree);
	}

	//	private void cudaFreeHelper(Pointer toFree, boolean eager) throws DMLRuntimeException {
	//		getGPUContext().cudaFreeHelper(toFree, eager);
	//	}

	private void cudaFreeHelper(String instName, Pointer toFree, boolean eager) throws DMLRuntimeException {
		getGPUContext().cudaFreeHelper(instName, toFree, eager);
	}

	private GPUContext getGPUContext() {
		return gpuContext;
	}

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
		if(LOG.isTraceEnabled()) {
			LOG.trace("GPU : transpose of block of size [" + m + "," + n + "]" + ", GPUContext=" + gCtx);
		}
		Pointer alpha = LibMatrixCUDA.one();
		Pointer beta = LibMatrixCUDA.zero();
		Pointer A = densePtr;
		Pointer C = gCtx.allocate(((long) m) * getDatatypeSizeOf(n));

		// Transpose the matrix to get a dense matrix
		LibMatrixCUDA.cudaSupportFunctions.cublasgeam(gCtx.getCublasHandle(), CUBLAS_OP_T, CUBLAS_OP_T, m, n, alpha, A, lda, beta, new Pointer(),
				lda, C, ldc);
		return C;
	}

	/**
	 * Convenience method to convert a CSR matrix to a dense matrix on the GPU
	 * Since the allocated matrix is temporary, bookkeeping is not updated.
	 * Also note that the input dense matrix is expected to be in COLUMN MAJOR FORMAT
	 * Caller is responsible for deallocating memory on GPU.
	 *
	 * @param gCtx           a valid {@link GPUContext}
	 * @param cusparseHandle handle to cusparse library
	 * @param densePtr       [in] dense matrix pointer on the GPU in row major
	 * @param rows           number of rows
	 * @param cols           number of columns
	 * @return CSR (compressed sparse row) pointer
	 * @throws DMLRuntimeException if DMLRuntimeException occurs
	 */
	public static CSRPointer columnMajorDenseToRowMajorSparse(GPUContext gCtx, cusparseHandle cusparseHandle,
			Pointer densePtr, int rows, int cols) throws DMLRuntimeException {
		cusparseMatDescr matDescr = CSRPointer.getDefaultCuSparseMatrixDescriptor();
		Pointer nnzPerRowPtr = null;
		Pointer nnzTotalDevHostPtr = null;

		nnzPerRowPtr = gCtx.allocate(getIntSizeOf(rows));
		nnzTotalDevHostPtr = gCtx.allocate(getIntSizeOf(1));

		// Output is in dense vector format, convert it to CSR
		LibMatrixCUDA.cudaSupportFunctions.cusparsennz(cusparseHandle, cusparseDirection.CUSPARSE_DIRECTION_ROW, rows, cols, matDescr, densePtr, rows,
				nnzPerRowPtr, nnzTotalDevHostPtr);
		//cudaDeviceSynchronize();
		int[] nnzC = { -1 };

		long t2 = 0;
		if (DMLScript.STATISTICS)
			t2 = System.nanoTime();
		cudaMemcpy(Pointer.to(nnzC), nnzTotalDevHostPtr, getIntSizeOf(1), cudaMemcpyDeviceToHost);
		if (DMLScript.STATISTICS)
			GPUStatistics.cudaFromDevTime.add(System.nanoTime() - t2);
		if (DMLScript.STATISTICS)
			GPUStatistics.cudaFromDevCount.add(1);

		if (nnzC[0] == -1) {
			throw new DMLRuntimeException(
					"cusparseDnnz did not calculate the correct number of nnz from the sparse-matrix vector mulitply on the GPU");
		}
		if(LOG.isTraceEnabled()) {
			LOG.trace("GPU : col-major dense size[" + rows + "," + cols + "] to row-major sparse of with nnz = " + nnzC[0]
				+ ", GPUContext=" + gCtx);
		}

		CSRPointer C = CSRPointer.allocateEmpty(gCtx, nnzC[0], rows);
		LibMatrixCUDA.cudaSupportFunctions.cusparsedense2csr(cusparseHandle, rows, cols, matDescr, densePtr, rows, nnzPerRowPtr, C.val, C.rowPtr,
				C.colInd);
		//cudaDeviceSynchronize();

		gCtx.cudaFreeHelper(nnzPerRowPtr);
		gCtx.cudaFreeHelper(nnzTotalDevHostPtr);

		return C;
	}

	/**
	 * Convenience method to directly examine the Sparse matrix on GPU
	 *
	 * @return CSR (compressed sparse row) pointer
	 */
	public CSRPointer getSparseMatrixCudaPointer() {
		return getJcudaSparseMatrixPtr();
	}

	/**
	 * Convenience method to directly set the sparse matrix on GPU
	 * Needed for operations like cusparseDcsrgemm(cusparseHandle, int, int, int, int, int, cusparseMatDescr, int, Pointer, Pointer, Pointer, cusparseMatDescr, int, Pointer, Pointer, Pointer, cusparseMatDescr, Pointer, Pointer, Pointer)
	 *
	 * @param sparseMatrixPtr CSR (compressed sparse row) pointer
	 * @throws DMLRuntimeException ?
	 */
	public void setSparseMatrixCudaPointer(CSRPointer sparseMatrixPtr) throws DMLRuntimeException {
		if (this.jcudaSparseMatrixPtr != null) {
			throw new DMLRuntimeException("jcudaSparseMatrixPtr was already allocated for " + this + ", this will cause a memory leak on the GPU");
		}
		this.jcudaSparseMatrixPtr = sparseMatrixPtr;
		this.isSparse = true;
		if (getJcudaDenseMatrixPtr() != null) {
			cudaFreeHelper(getJcudaDenseMatrixPtr());
			jcudaDenseMatrixPtr = null;
		}
	}

	/**
	 * Convenience method to directly set the dense matrix pointer on GPU
	 *
	 * @param densePtr dense pointer
	 * @throws DMLRuntimeException ?
	 */
	public void setDenseMatrixCudaPointer(Pointer densePtr) throws DMLRuntimeException {
		if (this.jcudaDenseMatrixPtr != null) {
			throw new DMLRuntimeException("jcudaDenseMatrixPtr was already allocated for " + this + ", this will cause a memory leak on the GPU");
		}
		this.jcudaDenseMatrixPtr = densePtr;
		this.isSparse = false;
		if(LOG.isDebugEnabled()) {
			LOG.debug("Setting dense pointer of size " + getGPUContext().getMemoryManager().getSizeAllocatedGPUPointer(densePtr));
		}
		if (getJcudaSparseMatrixPtr() != null) {
			getJcudaSparseMatrixPtr().deallocate();
			jcudaSparseMatrixPtr = null;
		}
	}

	/**
	 * Converts this GPUObject from dense to sparse format.
	 *
	 * @throws DMLRuntimeException if DMLRuntimeException occurs
	 */
	public void denseToSparse() throws DMLRuntimeException {
		if(LOG.isTraceEnabled()) {
			LOG.trace("GPU : dense -> sparse on " + this + ", GPUContext=" + getGPUContext());
		}
		long t0 = 0;
		if (DMLScript.STATISTICS)
			t0 = System.nanoTime();
		cusparseHandle cusparseHandle = getGPUContext().getCusparseHandle();
		if (cusparseHandle == null)
			throw new DMLRuntimeException("Expected cusparse to be initialized");
		int rows = toIntExact(mat.getNumRows());
		int cols = toIntExact(mat.getNumColumns());

		if (getJcudaDenseMatrixPtr() == null || !isAllocated())
			throw new DMLRuntimeException("Expected allocated dense matrix before denseToSparse() call");

		denseRowMajorToColumnMajor();
		setSparseMatrixCudaPointer(
				columnMajorDenseToRowMajorSparse(getGPUContext(), cusparseHandle, getJcudaDenseMatrixPtr(), rows,
						cols));
		// TODO: What if mat.getNnz() is -1 ?
		if (DMLScript.STATISTICS)
			GPUStatistics.cudaDenseToSparseTime.add(System.nanoTime() - t0);
		if (DMLScript.STATISTICS)
			GPUStatistics.cudaDenseToSparseCount.add(1);
	}

	/**
	 * Convenience method. Converts Row Major Dense Matrix to Column Major Dense Matrix
	 *
	 * @throws DMLRuntimeException if DMLRuntimeException occurs
	 */
	public void denseRowMajorToColumnMajor() throws DMLRuntimeException {
		if(LOG.isTraceEnabled()) {
			LOG.trace("GPU : dense Ptr row-major -> col-major on " + this + ", GPUContext=" + getGPUContext());
		}
		int m = toIntExact(mat.getNumRows());
		int n = toIntExact(mat.getNumColumns());
		int lda = n;
		int ldc = m;
		if (!isAllocated()) {
			throw new DMLRuntimeException("Error in converting row major to column major : data is not allocated");
		}

		Pointer tmp = transpose(getGPUContext(), getJcudaDenseMatrixPtr(), m, n, lda, ldc);
		cudaFreeHelper(getJcudaDenseMatrixPtr());
		jcudaDenseMatrixPtr = null;
		setDenseMatrixCudaPointer(tmp);
	}

	/**
	 * Convenience method. Converts Column Major Dense Matrix to Row Major Dense Matrix
	 *
	 * @throws DMLRuntimeException if error
	 */
	public void denseColumnMajorToRowMajor() throws DMLRuntimeException {
		if(LOG.isTraceEnabled()) {
			LOG.trace("GPU : dense Ptr row-major -> col-major on " + this + ", GPUContext=" + getGPUContext());
		}

		int n = toIntExact(mat.getNumRows());
		int m = toIntExact(mat.getNumColumns());
		int lda = n;
		int ldc = m;
		if (!isAllocated()) {
			throw new DMLRuntimeException("Error in converting column major to row major : data is not allocated");
		}

		Pointer tmp = transpose(getGPUContext(), getJcudaDenseMatrixPtr(), m, n, lda, ldc);
		cudaFreeHelper(getJcudaDenseMatrixPtr());
		jcudaDenseMatrixPtr = null;
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
	 *
	 * @param instructionName Name of the instruction for which statistics are recorded in {@link GPUStatistics}
	 * @throws DMLRuntimeException ?
	 */
	public void sparseToDense(String instructionName) throws DMLRuntimeException {
		if(LOG.isTraceEnabled()) {
			LOG.trace("GPU : sparse -> dense on " + this + ", GPUContext=" + getGPUContext());
		}
		long start = 0, end = 0;
		if (DMLScript.STATISTICS)
			start = System.nanoTime();
		if (getJcudaSparseMatrixPtr() == null || !isAllocated())
			throw new DMLRuntimeException("Expected allocated sparse matrix before sparseToDense() call");

		sparseToColumnMajorDense();
		denseColumnMajorToRowMajor();
		if (DMLScript.STATISTICS)
			end = System.nanoTime();
		if (instructionName != null && DMLScript.FINEGRAINED_STATISTICS)
			GPUStatistics.maintainCPMiscTimes(instructionName, GPUInstruction.MISC_TIMER_SPARSE_TO_DENSE, end - start);
		if (DMLScript.STATISTICS)
			GPUStatistics.cudaSparseToDenseTime.add(end - start);
		if (DMLScript.STATISTICS)
			GPUStatistics.cudaSparseToDenseCount.add(1);
	}

	/**
	 * More efficient method to convert sparse to dense but returns dense in column major format
	 *
	 * @throws DMLRuntimeException if DMLRuntimeException occurs
	 */
	public void sparseToColumnMajorDense() throws DMLRuntimeException {
		if(LOG.isTraceEnabled()) {
			LOG.trace("GPU : sparse -> col-major dense on " + this + ", GPUContext=" + getGPUContext());
		}
		if (getJcudaSparseMatrixPtr() == null || !isAllocated())
			throw new DMLRuntimeException("Expected allocated sparse matrix before sparseToDense() call");

		cusparseHandle cusparseHandle = getGPUContext().getCusparseHandle();
		if (cusparseHandle == null)
			throw new DMLRuntimeException("Expected cusparse to be initialized");
		int rows = toIntExact(mat.getNumRows());
		int cols = toIntExact(mat.getNumColumns());
		setDenseMatrixCudaPointer(getJcudaSparseMatrixPtr().toColumnMajorDenseMatrix(cusparseHandle, null, rows, cols, null));
	}

	/**
	 * Initializes this GPUObject with a {@link MatrixObject} instance which will contain metadata about the enclosing matrix block
	 *
	 * @param mat2 the matrix block that owns this {@link GPUObject}
	 */
	GPUObject(GPUContext gCtx, MatrixObject mat2) {
		gpuContext = gCtx;
		this.mat = mat2;
	}

	public boolean isSparse() {
		return isSparse;
	}
	
	private static long getDatatypeSizeOf(long numElems) {
		return numElems * LibMatrixCUDA.sizeOfDataType;
	}

	private static long getIntSizeOf(long numElems) {
		return numElems * ((long) jcuda.Sizeof.INT);
	}

	public boolean isAllocated() {
		boolean eitherAllocated = (getJcudaDenseMatrixPtr() != null || getJcudaSparseMatrixPtr() != null);
		return eitherAllocated;
	}

	/**
	 * Allocates a sparse and empty {@link GPUObject}
	 * This is the result of operations that are both non zero matrices.
	 *
	 * @throws DMLRuntimeException if DMLRuntimeException occurs
	 */
	public void allocateSparseAndEmpty() throws DMLRuntimeException {
		if(LOG.isTraceEnabled()) {
			LOG.trace("GPU : allocate sparse and empty block on " + this + ", GPUContext=" + getGPUContext());
		}
		setSparseMatrixCudaPointer(CSRPointer.allocateEmpty(getGPUContext(), 0, mat.getNumRows()));
	}

	/**
	 * Allocates a dense matrix of size obtained from the attached matrix metadata
	 * and fills it up with a single value
	 *
	 * @param v value to fill up the dense matrix
	 * @throws DMLRuntimeException if DMLRuntimeException occurs
	 */
	public void allocateAndFillDense(double v) throws DMLRuntimeException {
		if(LOG.isTraceEnabled()) {
			LOG.trace("GPU : allocate and fill dense with value " + v + " on " + this + ", GPUContext=" + getGPUContext());
		}
		long rows = mat.getNumRows();
		long cols = mat.getNumColumns();
		int numElems = toIntExact(rows * cols);
		long size = getDatatypeSizeOf(numElems);
		setDenseMatrixCudaPointer(allocate(size));
		// The "fill" kernel is called which treats the matrix "jcudaDensePtr" like a vector and fills it with value "v"
		// If the fill value is 0, no need to call the special kernel, the allocate memsets the allocated region to 0
		if (v != 0)
			getGPUContext().getKernels()
			.launchKernel("fill", ExecutionConfig.getConfigForSimpleVectorOperations(numElems),
					getJcudaDenseMatrixPtr(), v, numElems);
	}

	/**
	 * If this {@link GPUObject} is sparse and empty
	 * Being allocated is a prerequisite to being sparse and empty.
	 *
	 * @return true if sparse and empty
	 * @throws DMLRuntimeException if error
	 */
	public boolean isSparseAndEmpty() throws DMLRuntimeException {
		boolean isSparseAndAllocated = isAllocated() && LibMatrixCUDA.isInSparseFormat(getGPUContext(), mat);
		boolean isEmptyAndSparseAndAllocated = isSparseAndAllocated && getJcudaSparseMatrixPtr().nnz == 0;
		return isEmptyAndSparseAndAllocated;
	}
		
	/**
	 * Being allocated is a prerequisite for computing nnz.
	 * Note: if the matrix is in dense format, it explicitly re-computes the number of nonzeros.
	 *
	 * @param instName instruction name
	 * @param recomputeDenseNNZ recompute NNZ if dense
	 * @return the number of nonzeroes
	 * @throws DMLRuntimeException if error
	 */
	public long getNnz(String instName, boolean recomputeDenseNNZ) throws DMLRuntimeException {
		if(isAllocated()) {
			if(LibMatrixCUDA.isInSparseFormat(getGPUContext(), mat)) {
				return getJcudaSparseMatrixPtr().nnz;
			}
			else {
				if(!recomputeDenseNNZ)
					return -1;
				
				long t1 = DMLScript.FINEGRAINED_STATISTICS ? System.nanoTime() : 0;
				GPUContext gCtx = getGPUContext();
				cusparseHandle cusparseHandle = gCtx.getCusparseHandle();
				cusparseMatDescr matDescr = CSRPointer.getDefaultCuSparseMatrixDescriptor();
				if (cusparseHandle == null)
					throw new DMLRuntimeException("Expected cusparse to be initialized");
				int rows = toIntExact(mat.getNumRows());
				int cols = toIntExact(mat.getNumColumns());
				Pointer nnzPerRowPtr = null;
				Pointer nnzTotalDevHostPtr = null;
				nnzPerRowPtr = gCtx.allocate(getIntSizeOf(rows));
				nnzTotalDevHostPtr = gCtx.allocate(getIntSizeOf(1));
				LibMatrixCUDA.cudaSupportFunctions.cusparsennz(cusparseHandle, cusparseDirection.CUSPARSE_DIRECTION_ROW, rows, cols, matDescr, getJcudaDenseMatrixPtr(), rows,
						nnzPerRowPtr, nnzTotalDevHostPtr);
				int[] nnzC = { -1 };
				cudaMemcpy(Pointer.to(nnzC), nnzTotalDevHostPtr, getIntSizeOf(1), cudaMemcpyDeviceToHost);
				if (nnzC[0] == -1) {
					throw new DMLRuntimeException(
							"cusparseDnnz did not calculate the correct number of nnz on the GPU");
				}
				gCtx.cudaFreeHelper(nnzPerRowPtr);
				gCtx.cudaFreeHelper(nnzTotalDevHostPtr);
				if(DMLScript.FINEGRAINED_STATISTICS) {
					GPUStatistics.maintainCPMiscTimes(instName, CPInstruction.MISC_TIMER_RECOMPUTE_NNZ, System.nanoTime()-t1);
			}
				return nnzC[0];
			}
		}
		else 
			throw new DMLRuntimeException("Expected the GPU object to be allocated");
	}

	public boolean acquireDeviceRead(String opcode) throws DMLRuntimeException {
		if(LOG.isTraceEnabled()) {
			LOG.trace("GPU : acquireDeviceRead on " + this);
		}
		boolean transferred = false;
		if (!isAllocated()) {
			if(LOG.isTraceEnabled()) {
				LOG.trace("GPU : in acquireDeviceRead, data is not allocated, copying from host, on " + this + ", GPUContext="
							+ getGPUContext());
			}
			copyFromHostToDevice(opcode);
			transferred = true;
		}
		addReadLock();
		if (!isAllocated())
			throw new DMLRuntimeException("Expected device data to be allocated");
		return transferred;
	}

	public boolean acquireDeviceModifyDense() throws DMLRuntimeException {
		if(LOG.isTraceEnabled()) {
			LOG.trace("GPU : acquireDeviceModifyDense on " + this + ", GPUContext=" + getGPUContext());
		}
		boolean allocated = false;
		if (!isAllocated()) {
			mat.setDirty(true);
			if(LOG.isTraceEnabled()) {
				LOG.trace("GPU : data is not allocated, allocating a dense block, on " + this);
			}
			// Dense block, size = numRows * numCols
			allocateDenseMatrixOnDevice();
			allocated = true;
		}
		dirty = true;
		if (!isAllocated())
			throw new DMLRuntimeException("Expected device data to be allocated");
		return allocated;
	}

	public boolean acquireDeviceModifySparse() throws DMLRuntimeException {
		if(LOG.isTraceEnabled()) {
			LOG.trace("GPU : acquireDeviceModifySparse on " + this + ", GPUContext=" + getGPUContext());
		}
		boolean allocated = false;
		isSparse = true;
		if (!isAllocated()) {
			if(LOG.isTraceEnabled()) {
				LOG.trace("GPU : data is not allocated, allocating a sparse block, on " + this);
			}
			mat.setDirty(true);
			allocateSparseMatrixOnDevice();
			allocated = true;
		}
		dirty = true;
		if (!isAllocated())
			throw new DMLRuntimeException("Expected device data to be allocated");
		return allocated;
	}

	/**
	 * if the data is allocated on the GPU and is dirty, it is copied back to the host memory
	 *
	 * @param instName name of the instruction
	 * @return true if a copy to host happened, false otherwise
	 * @throws CacheException ?
	 */
	public boolean acquireHostRead(String instName) throws CacheException {
		boolean copied = false;
		try {
			if(LOG.isTraceEnabled()) {
				LOG.trace("GPU : acquireDeviceModifySparse on " + this + ", GPUContext=" + getGPUContext());
			}
			if (isAllocated() && dirty) {
				if(LOG.isTraceEnabled()) {
					LOG.trace("GPU : data is dirty on device, copying to host, on " + this + ", GPUContext="
						+ getGPUContext());
				}
				copyFromDeviceToHost(instName, false);
				copied = true;
			}
		} catch (DMLRuntimeException e) {
			throw new CacheException(e);
		}
		return copied;
	}
	
	public boolean isLocked() {
		return writeLock || readLocks.longValue() > 0;
	}
	
	public void addReadLock() throws DMLRuntimeException {
		if(writeLock)
			throw new DMLRuntimeException("Attempting to add a read lock when writeLock="+ writeLock);
		else
			readLocks.increment();
	}
	
	public void addWriteLock() throws DMLRuntimeException {
		if(readLocks.longValue() > 0)
			throw new DMLRuntimeException("Attempting to add a write lock when readLocks="+ readLocks.longValue());
		else if(writeLock)
			throw new DMLRuntimeException("Attempting to add a write lock when writeLock="+ writeLock);
		else
			writeLock = true;
	}
	
	public void releaseReadLock() throws DMLRuntimeException {
		readLocks.decrement();
		if(readLocks.longValue() < 0)
			throw new DMLRuntimeException("Attempting to release a read lock when readLocks="+ readLocks.longValue());
	}
	
	public void releaseWriteLock() throws DMLRuntimeException {
		if(writeLock)
			writeLock = false;
		else
			throw new DMLRuntimeException("Internal state error : Attempting to release write lock on a GPUObject, which was already released");
	}
	
	public void resetReadWriteLock() {
		readLocks.reset();
		writeLock = false;
	}

	/**
	 * Updates the locks depending on the eviction policy selected
	 *
	 * @throws DMLRuntimeException if there is no locked GPU Object or if could not obtain a {@link GPUContext}
	 */
	private void updateReleaseLocks() throws DMLRuntimeException {
		DMLScript.EvictionPolicy evictionPolicy = DMLScript.GPU_EVICTION_POLICY;
		switch (evictionPolicy) {
			case LRU:
				timestamp.set(System.nanoTime());
				break;
			case LFU:
				timestamp.addAndGet(1);
				break;
			case MIN_EVICT: /* Do Nothing */
				break;
			case MRU:
				timestamp.set(-System.nanoTime());
				break;
			default:
				throw new CacheException("The eviction policy is not supported:" + evictionPolicy.name());
		}
	}

	/**
	 * Releases input allocated on GPU
	 *
	 * @throws DMLRuntimeException if data is not allocated or if there is no locked GPU Object or if could not obtain a {@link GPUContext}
	 */
	public void releaseInput() throws DMLRuntimeException {
		releaseReadLock();
		updateReleaseLocks();
		if (!isAllocated())
			throw new CacheException("Attempting to release an input before allocating it");
	}

	/**
	 * releases output allocated on GPU
	 *
	 * @throws DMLRuntimeException if data is not allocated or if there is no locked GPU Object or if could not obtain a {@link GPUContext}
	 */
	public void releaseOutput() throws DMLRuntimeException {
		releaseWriteLock();
		updateReleaseLocks();
		// Currently, there is no convenient way to acquireDeviceModify independently of dense/sparse format. 
		// Hence, allowing resetting releaseOutput again. 
		// Ideally, we would want to throw CacheException("Attempting to release an output that was not acquired via acquireDeviceModify") if !isDirty()
		dirty = true;
		if (!isAllocated())
			throw new CacheException("Attempting to release an output before allocating it");
	}

	void allocateDenseMatrixOnDevice() throws DMLRuntimeException {
		if(LOG.isTraceEnabled()) {
			LOG.trace("GPU : allocateDenseMatrixOnDevice, on " + this + ", GPUContext=" + getGPUContext());
		}
		if(isAllocated()) 
			throw new DMLRuntimeException("Internal error - trying to allocated dense matrix to a GPUObject that is already allocated");
		long rows = mat.getNumRows();
		long cols = mat.getNumColumns();
		if(rows <= 0)
			throw new DMLRuntimeException("Internal error - invalid number of rows when allocating dense matrix");
		if(cols <= 0)
			throw new DMLRuntimeException("Internal error - invalid number of columns when allocating dense matrix;");
		long size = getDatatypeSizeOf(rows * cols);
		Pointer tmp = allocate(size);
		setDenseMatrixCudaPointer(tmp);
	}

	void allocateSparseMatrixOnDevice() throws DMLRuntimeException {
		if(LOG.isTraceEnabled()) {
			LOG.trace("GPU : allocateSparseMatrixOnDevice, on " + this + ", GPUContext=" + getGPUContext());
		}
		if(isAllocated()) 
			throw new DMLRuntimeException("Internal error - trying to allocated sparse matrix to a GPUObject that is already allocated");
		long rows = mat.getNumRows();
		long nnz = mat.getNnz();
		if(rows <= 0)
			throw new DMLRuntimeException("Internal error - invalid number of rows when allocating sparse matrix");
		if(nnz < 0)
			throw new DMLRuntimeException("Internal error - invalid number of non zeroes when allocating a sparse matrix");
		CSRPointer tmp = CSRPointer.allocateEmpty(getGPUContext(), nnz, rows);
		setSparseMatrixCudaPointer(tmp);
	}

	void deallocateMemoryOnDevice(boolean eager) throws DMLRuntimeException {
		if(LOG.isTraceEnabled()) {
			LOG.trace("GPU : deallocateMemoryOnDevice, on " + this + ", GPUContext=" + getGPUContext());
		}
		if (getJcudaDenseMatrixPtr() != null) {
			cudaFreeHelper(null, getJcudaDenseMatrixPtr(), eager);
		}
		if (getJcudaSparseMatrixPtr() != null) {
			getJcudaSparseMatrixPtr().deallocate(eager);
		}
		jcudaDenseMatrixPtr = null;
		jcudaSparseMatrixPtr = null;
		resetReadWriteLock();
	}

	protected long getSizeOnDevice() throws DMLRuntimeException {
		long GPUSize = 0;
		long rlen = mat.getNumRows();
		long clen = mat.getNumColumns();
		long nnz = mat.getNnz();

		if (LibMatrixCUDA.isInSparseFormat(getGPUContext(), mat)) {
			GPUSize = CSRPointer.estimateSize(nnz, rlen);
		} else {
			GPUSize = getDatatypeSizeOf(rlen * clen);
		}
		return GPUSize;
	}

	void copyFromHostToDevice(String opcode) throws DMLRuntimeException {
		if(LOG.isTraceEnabled()) {
			LOG.trace("GPU : copyFromHostToDevice, on " + this + ", GPUContext=" + getGPUContext());
		}
		long start = 0;
		if (DMLScript.STATISTICS)
			start = System.nanoTime();

		long acqrTime = DMLScript.FINEGRAINED_STATISTICS ? System.nanoTime() : 0;
		MatrixBlock tmp = mat.acquireRead();
		if(DMLScript.FINEGRAINED_STATISTICS) {
			if(tmp.isInSparseFormat())
				GPUStatistics.maintainCPMiscTimes(opcode, CPInstruction.MISC_TIMER_GET_SPARSE_MB, System.nanoTime()-acqrTime);
			else
				GPUStatistics.maintainCPMiscTimes(opcode, CPInstruction.MISC_TIMER_GET_DENSE_MB, System.nanoTime()-acqrTime);
		}
		
		if (tmp.isInSparseFormat()) {
			int rowPtr[] = null;
			int colInd[] = null;
			double[] values = null;
			
			// Only recompute non-zero if unknown, else this will incur huge penalty !!
			if(tmp.getNonZeros() < 0) {
				tmp.recomputeNonZeros();
			}
			long nnz = tmp.getNonZeros();
			mat.getMatrixCharacteristics().setNonZeros(nnz);

			SparseBlock block = tmp.getSparseBlock();
			boolean copyToDevice = true;
			if (block == null && tmp.getNonZeros() == 0) {
				//				// Allocate empty block --> not necessary
				//				// To reproduce this, see org.apache.sysml.test.integration.applications.dml.ID3DMLTest
				//				rowPtr = new int[0];
				//				colInd = new int[0];
				//				values = new double[0];
				copyToDevice = false;
			} else if (block == null && tmp.getNonZeros() != 0) {
				throw new DMLRuntimeException("Expected CP sparse block to be not null.");
			} else {
				// CSR is the preferred format for cuSparse GEMM
				// Converts MCSR and COO to CSR
				SparseBlockCSR csrBlock = null;
				long t0 = 0;
				if (block instanceof SparseBlockCSR) {
					csrBlock = (SparseBlockCSR) block;
				} else if (block instanceof SparseBlockCOO) {
					// TODO - should we do this on the GPU using cusparse<t>coo2csr() ?
					if (DMLScript.STATISTICS)
						t0 = System.nanoTime();
					SparseBlockCOO cooBlock = (SparseBlockCOO) block;
					csrBlock = new SparseBlockCSR(toIntExact(mat.getNumRows()), cooBlock.rowIndexes(),
							cooBlock.indexes(), cooBlock.values());
					if (DMLScript.STATISTICS)
						GPUStatistics.cudaSparseConversionTime.add(System.nanoTime() - t0);
					if (DMLScript.STATISTICS)
						GPUStatistics.cudaSparseConversionCount.increment();
				} else if (block instanceof SparseBlockMCSR) {
					if (DMLScript.STATISTICS)
						t0 = System.nanoTime();
					SparseBlockMCSR mcsrBlock = (SparseBlockMCSR) block;
					csrBlock = new SparseBlockCSR(mcsrBlock.getRows(), toIntExact(mcsrBlock.size()));
					if (DMLScript.STATISTICS)
						GPUStatistics.cudaSparseConversionTime.add(System.nanoTime() - t0);
					if (DMLScript.STATISTICS)
						GPUStatistics.cudaSparseConversionCount.increment();
				} else {
					throw new DMLRuntimeException("Unsupported sparse matrix format for CUDA operations");
				}
				rowPtr = csrBlock.rowPointers();
				colInd = csrBlock.indexes();
				values = csrBlock.values();
			}

			allocateSparseMatrixOnDevice();

			if (copyToDevice) {
				long t1 = DMLScript.FINEGRAINED_STATISTICS ? System.nanoTime() : 0;
				CSRPointer.copyToDevice(getGPUContext(), getJcudaSparseMatrixPtr(), tmp.getNumRows(), tmp.getNonZeros(), rowPtr, colInd,
						values);
				if(DMLScript.FINEGRAINED_STATISTICS) 
					GPUStatistics.maintainCPMiscTimes(opcode, GPUInstruction.MISC_TIMER_HOST_TO_DEVICE, System.nanoTime() - t1);
			}
		} else {
			double[] data = tmp.getDenseBlockValues();

			if (data == null && tmp.getSparseBlock() != null)
				throw new DMLRuntimeException("Incorrect sparsity calculation");
			else if (data == null && tmp.getNonZeros() != 0)
				throw new DMLRuntimeException("MatrixBlock is not allocated");
			
			allocateDenseMatrixOnDevice();
			
			if (tmp.getNonZeros() == 0) {
				// Minor optimization: No need to allocate empty error for CPU 
				// data = new double[tmp.getNumRows() * tmp.getNumColumns()];
				long t1 = DMLScript.FINEGRAINED_STATISTICS ? System.nanoTime() : 0;
				cudaMemset(getJcudaDenseMatrixPtr(), 0, getDatatypeSizeOf(mat.getNumRows() * mat.getNumColumns()));
				if(DMLScript.FINEGRAINED_STATISTICS) 
					GPUStatistics.maintainCPMiscTimes(opcode, GPUInstruction.MISC_TIMER_SET_ZERO, System.nanoTime() - t1);
			}
			else {
				// Copy dense block
				// H2D now only measures the time taken to do 
				LibMatrixCUDA.cudaSupportFunctions.hostToDevice(getGPUContext(), data, getJcudaDenseMatrixPtr(), opcode);
			}
		}

		mat.release();

		if (DMLScript.STATISTICS)
			GPUStatistics.cudaToDevTime.add(System.nanoTime() - start);
		if (DMLScript.STATISTICS)
			GPUStatistics.cudaToDevCount.add(1);
	}

	public static int toIntExact(long l) throws DMLRuntimeException {
		if (l < Integer.MIN_VALUE || l > Integer.MAX_VALUE) {
			throw new DMLRuntimeException("Cannot be cast to int:" + l);
		}
		return (int) l;
	}

	protected void copyFromDeviceToHost(String instName, boolean isEviction) throws DMLRuntimeException {
		if(LOG.isTraceEnabled()) {
			LOG.trace("GPU : copyFromDeviceToHost, on " + this + ", GPUContext=" + getGPUContext());
		}
		if (getJcudaDenseMatrixPtr() != null && getJcudaSparseMatrixPtr() != null) {
			throw new DMLRuntimeException("Invalid state : JCuda dense/sparse pointer are both allocated");
		}

		if (getJcudaDenseMatrixPtr() != null) {
			long start = 0;
			if (DMLScript.STATISTICS)
				start = System.nanoTime();
			MatrixBlock tmp = new MatrixBlock(toIntExact(mat.getNumRows()), toIntExact(mat.getNumColumns()), false);
			tmp.allocateDenseBlock();
			LibMatrixCUDA.cudaSupportFunctions.deviceToHost(getGPUContext(),
						getJcudaDenseMatrixPtr(), tmp.getDenseBlockValues(), instName, isEviction);
			
			tmp.recomputeNonZeros();
			mat.acquireModify(tmp);
			mat.release();

			if (DMLScript.STATISTICS)
				GPUStatistics.cudaFromDevTime.add(System.nanoTime() - start);
			if (DMLScript.STATISTICS)
				GPUStatistics.cudaFromDevCount.add(1);
		} else if (getJcudaSparseMatrixPtr() != null) {
			if (!LibMatrixCUDA.isInSparseFormat(getGPUContext(), mat))
				throw new DMLRuntimeException(
						"Block not in sparse format on host yet the device sparse matrix pointer is not null");

			if (this.isSparseAndEmpty()) {
				MatrixBlock tmp = new MatrixBlock((int)mat.getNumRows(), (int)mat.getNumColumns(), 0l);    // Empty Block
				mat.acquireModify(tmp);
				mat.release();
			} else {
				long start = 0;
				if (DMLScript.STATISTICS)
					start = System.nanoTime();

				int rows = toIntExact(mat.getNumRows());
				int cols = toIntExact(mat.getNumColumns());
				int nnz = toIntExact(getJcudaSparseMatrixPtr().nnz);
				double[] values = new double[nnz];
				LibMatrixCUDA.cudaSupportFunctions.deviceToHost(getGPUContext(), getJcudaSparseMatrixPtr().val, values, instName, isEviction);
				int[] rowPtr = new int[rows + 1];
				int[] colInd = new int[nnz];
				long t0 = DMLScript.STATISTICS ? System.nanoTime() : 0;
				CSRPointer.copyPtrToHost(getJcudaSparseMatrixPtr(), rows, nnz, rowPtr, colInd);
				if (DMLScript.STATISTICS)
					GPUStatistics.cudaFromDevTime.add(System.nanoTime() - t0);
				if (DMLScript.STATISTICS)
					GPUStatistics.cudaFromDevCount.add(3);

				SparseBlockCSR sparseBlock = new SparseBlockCSR(rowPtr, colInd, values, nnz);
				MatrixBlock tmp = new MatrixBlock(rows, cols, nnz, sparseBlock);
				mat.acquireModify(tmp);
				mat.release();
				if (DMLScript.STATISTICS)
					GPUStatistics.cudaFromDevTime.add(System.nanoTime() - start);
				if (DMLScript.STATISTICS)
					GPUStatistics.cudaFromDevCount.add(1);
			}
		} else {
			throw new DMLRuntimeException(
					"Cannot copy from device to host as JCuda dense/sparse pointer is not allocated");
		}
		dirty = false;
	}

	/**
	 * lazily clears the data associated with this {@link GPUObject} instance
	 *
	 * @throws CacheException ?
	 */
	public void clearData() throws DMLRuntimeException {
		clearData(DMLScript.EAGER_CUDA_FREE);
	}

	/**
	 * Clears the data associated with this {@link GPUObject} instance
	 *
	 * @param eager whether to be done synchronously or asynchronously
	 * @throws CacheException ?
	 */
	public void clearData(boolean eager) throws DMLRuntimeException {
		deallocateMemoryOnDevice(eager);
		getGPUContext().getMemoryManager().removeGPUObject(this);
	}

	/**
	 * Pointer to dense matrix
	 *
	 * @return ?
	 */
	public Pointer getJcudaDenseMatrixPtr() {
		return jcudaDenseMatrixPtr;
	}

	/**
	 * Pointer to sparse matrix
	 *
	 * @return ?
	 */
	public CSRPointer getJcudaSparseMatrixPtr() {
		return jcudaSparseMatrixPtr;
	}

	/**
	 * Whether this block is dirty on the GPU
	 *
	 * @return ?
	 */
	public boolean isDirty() {
		return dirty;
	}

	@Override
	public String toString() {
		final StringBuilder sb = new StringBuilder("GPUObject{");
		sb.append(", dirty=").append(dirty);
		sb.append(", readLocks=").append(readLocks.longValue());
		sb.append(", writeLock=").append(writeLock);
		sb.append(", sparse? ").append(isSparse);
		sb.append(", dims=[").append(mat.getNumRows()).append(",").append(mat.getNumColumns()).append("]");
		if(jcudaDenseMatrixPtr != null)
			sb.append(", densePtr=").append(jcudaDenseMatrixPtr);
		if(jcudaSparseMatrixPtr != null)
			sb.append(", sparsePtr=").append(jcudaSparseMatrixPtr);
		sb.append('}');
		return sb.toString();
	}

}
