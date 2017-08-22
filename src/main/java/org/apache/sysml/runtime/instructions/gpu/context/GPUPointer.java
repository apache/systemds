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

import static jcuda.jcudnn.JCudnn.cudnnCreateTensorDescriptor;
import static jcuda.jcudnn.JCudnn.cudnnDestroyTensorDescriptor;
import static jcuda.jcudnn.JCudnn.cudnnSetTensor4dDescriptor;
import static jcuda.jcudnn.cudnnDataType.CUDNN_DATA_DOUBLE;
import static jcuda.jcudnn.cudnnTensorFormat.CUDNN_TENSOR_NCHW;
import static jcuda.runtime.JCuda.cudaMemcpy;
import static jcuda.runtime.cudaMemcpyKind.cudaMemcpyDeviceToDevice;

import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;
import org.apache.sysml.runtime.DMLRuntimeException;
import jcuda.Pointer;
import jcuda.Sizeof;
import jcuda.jcudnn.cudnnTensorDescriptor;
import jcuda.jcusparse.JCusparse;
import jcuda.jcusparse.cusparseHandle;
import jcuda.jcusparse.cusparseMatDescr;

/**
 * This class serves as a wrapper class to hold dense and sparse pointers.
 * It allows for shallow copies of GPUObjects (i.e. multiple GPUObjects hold references
 * to the same GPUPointer) and handles the deallocation seamlessly.
 * This logic is necessary in update in-place logic.
 * 
 * We recommend that you use GPUUtils's shallowCopy and deepCopy methods to create a copy of matrix object backed by gpu object
 * and GPUObject's shallowCopy and clone methods to create a copy of gpu object.
 */
public class GPUPointer {
	
	private static final Log LOG = LogFactory.getLog(GPUPointer.class.getName());

	/**
	 * Pointer to the underlying dense matrix block on GPU
	 */
	private Pointer jcudaDenseMatrixPtr = null;

	/**
	 * Pointer to the underlying sparse matrix block on GPU
	 */
	private CSRPointer jcudaSparseMatrixPtr = null;
	
	/**
	 * GPUContext that owns this GPUPointer
	 */
	private final GPUContext gpuContext;
	
	/**
	 * Number of rows
	 */
	private long rows;
	
	/**
	 * Number of columns
	 */
	private long cols;
	
	/**
	 * Whether this block is in sparse format
	 */
	protected boolean isSparse = false;
	
	GPUPointer(GPUContext gCtx, long rows, long cols) {
		gpuContext = gCtx;
		this.rows = rows;
		this.cols = cols;
	}
	
	/**
	 * Number of references to the device pointers
	 */
	private int backReferences = 1;
	
	/**
	 * An optional tensor descriptor (and shape) that can be set by a tensor instruction such as convolution,
	 * maxpooling and exploited by a subsequent non-tensor instruction such as relu
	 */
	private cudnnTensorDescriptor tensorDescriptor = null;

	/**
	 * the shape of this tensor, if in fact this is a tensor
	 */
	private int[] tensorShape = null;
	
	/**
	 * Increments the backreferences
	 */
	public void incrementBackReferences() {
		backReferences++;
	}
	
	/**
	 * Pointer to dense matrix
	 *
	 * @return pointer to dense array on device in row-major format
	 */
	public Pointer getJcudaDenseMatrixPtr() {
		return jcudaDenseMatrixPtr;
	}

	/**
	 * Pointer to sparse matrix
	 *
	 * @return pointer to sparse CSRPointer on device
	 */
	public CSRPointer getJcudaSparseMatrixPtr() {
		return jcudaSparseMatrixPtr;
	}
	
	private GPUContext getGPUContext() {
		return gpuContext;
	}
	
	/**
	 * Deallocate if there are no backreferences to the pointers.
	 * 
	 * @param eager should deallocated eagerly
	 * @return true if the pointers are deallocated
	 * @throws DMLRuntimeException if error occurs
	 */
	public boolean deallocate(boolean eager) throws DMLRuntimeException {
		backReferences--;
		if(backReferences < 0) {
			// throw new DMLRuntimeException("Attempting to deallocate GPUPointer which has already been deallocated");
			return false;
		}
		else if(backReferences > 0) {
			return false;
		}
		else {
			// Eager deallocation is performed when one destroys a GPUContext and during eviction.
			// For consistency, we donot eagerly delete when backReferences > 0
			// Only the deallocate of last matrix object performs eager deallocation.
			LOG.trace("GPU : deallocateMemoryOnDevice, on " + this + ", GPUContext=" + getGPUContext());
			if (getJcudaDenseMatrixPtr() != null) {
				cudaFreeHelper(null, getJcudaDenseMatrixPtr(), eager);
			}
			if (getJcudaSparseMatrixPtr() != null) {
				getJcudaSparseMatrixPtr().deallocate(eager);
			}
			jcudaDenseMatrixPtr = null;
			jcudaSparseMatrixPtr = null;
			if (tensorDescriptor != null) {
				cudnnDestroyTensorDescriptor(tensorDescriptor);
				tensorDescriptor = null;
			}
			return true;
		}
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
	
	/**
	 * Returns a previously allocated or allocates and returns a tensor descriptor
	 *
	 * @param N number of images
	 * @param C number of channels
	 * @param H height
	 * @param W width
	 * @return cudnn tensor descriptor
	 */
	public cudnnTensorDescriptor allocateTensorDescriptor(int N, int C, int H, int W) {
		LOG.trace("GPU : allocateTensorDescriptor with [N=" + N + ",C=" + C + ",H=" + H + ",W=" + W + "] on " + this);
		if (tensorDescriptor == null) {
			tensorDescriptor = new cudnnTensorDescriptor();
			cudnnCreateTensorDescriptor(tensorDescriptor);
			cudnnSetTensor4dDescriptor(tensorDescriptor, CUDNN_TENSOR_NCHW, CUDNN_DATA_DOUBLE, N, C, H, W);
			tensorShape = new int[4];
			tensorShape[0] = N;
			tensorShape[1] = C;
			tensorShape[2] = H;
			tensorShape[3] = W;
		}
		return tensorDescriptor;
	}
	
	/**
	 * Returns a previously allocated tensor descriptor or null
	 *
	 * @return cudnn tensor descriptor
	 */
	public cudnnTensorDescriptor getTensorDescriptor() {
		return tensorDescriptor;
	}
	
	/**
	 * Returns a previously allocated tensor shape or null
	 *
	 * @return int array of four elements or null
	 */
	public int[] getTensorShape() {
		return tensorShape;
	}
	
	/**
	 * Get number of rows
	 * @return number of rows
	 */
	public long getNumRows() {
		return rows;
	}
	
	/**
	 * Get number of columns
	 * @return number of columns
	 */
	public long getNumColumns() {
		return cols;
	}
	
	
	@Override
	public Object clone() {
		GPUPointer me = this;
		GPUPointer that = new GPUPointer(me.gpuContext, me.rows, me.cols);
		if (me.tensorShape != null) {
			that.tensorShape = new int[me.tensorShape.length];
			System.arraycopy(me.tensorShape, 0, that.tensorShape, 0, me.tensorShape.length);
			that.allocateTensorDescriptor(me.tensorShape[0], me.tensorShape[1], me.tensorShape[2], me.tensorShape[3]);
		}
		that.isSparse = me.isSparse;

		try {
			if (me.jcudaDenseMatrixPtr != null) {
				long rows = me.getNumRows();
				long cols = me.getNumColumns();
				long size = rows * cols * Sizeof.DOUBLE;
				me.gpuContext.ensureFreeSpace((int) size);
				that.jcudaDenseMatrixPtr = allocate(size);
				cudaMemcpy(that.jcudaDenseMatrixPtr, me.jcudaDenseMatrixPtr, size, cudaMemcpyDeviceToDevice);
			}

			if (me.getJcudaSparseMatrixPtr() != null) {
				long rows = getNumRows();
				that.jcudaSparseMatrixPtr = me.jcudaSparseMatrixPtr.clone((int) rows);
			}

		} catch (DMLRuntimeException e) {
			throw new RuntimeException(e);
		}

		return that;
	}
	
	/**
	 * Convenience method. Converts Column Major Dense Matrix to Row Major Dense Matrix
	 *
	 * @throws DMLRuntimeException if error
	 */
	public void denseColumnMajorToRowMajor() throws DMLRuntimeException {
		int n = toIntExact(getNumRows());
		int m = toIntExact(getNumColumns());
		int lda = n;
		int ldc = m;
		Pointer tmp = GPUUtils.transpose(getGPUContext(), getJcudaDenseMatrixPtr(), m, n, lda, ldc);
		cudaFreeHelper(getJcudaDenseMatrixPtr());
		setDenseMatrixCudaPointer(tmp);
	}
	
	/**
	 * Convenience method. Converts Row Major Dense Matrix to Column Major Dense Matrix
	 *
	 * @throws DMLRuntimeException if DMLRuntimeException occurs
	 */
	public void denseRowMajorToColumnMajor() throws DMLRuntimeException {
		int m = toIntExact(getNumRows());
		int n = toIntExact(getNumColumns());
		int lda = n;
		int ldc = m;
		Pointer tmp = GPUUtils.transpose(getGPUContext(), getJcudaDenseMatrixPtr(), m, n, lda, ldc);
		cudaFreeHelper(getJcudaDenseMatrixPtr());
		setDenseMatrixCudaPointer(tmp);
	}
	
	public static int toIntExact(long l) throws DMLRuntimeException {
		if (l < Integer.MIN_VALUE || l > Integer.MAX_VALUE) {
			throw new DMLRuntimeException("Cannot be cast to int:" + l);
		}
		return (int) l;
	}
	
	/**
	 * Convenience method to directly set the dense matrix pointer on GPU
	 *
	 * @param densePtr dense pointer
	 * @throws DMLRuntimeException ?
	 */
	public void setDenseMatrixCudaPointer(Pointer densePtr) throws DMLRuntimeException {
		this.jcudaDenseMatrixPtr = densePtr;
		this.isSparse = false;
		if (getJcudaSparseMatrixPtr() != null) {
			getJcudaSparseMatrixPtr().deallocate();
			jcudaSparseMatrixPtr = null;
		}
	}
	
	/**
	 * Convenience method to directly set the sparse matrix on GPU
	 * Needed for operations like {@link JCusparse#cusparseDcsrgemm(cusparseHandle, int, int, int, int, int, cusparseMatDescr, int, Pointer, Pointer, Pointer, cusparseMatDescr, int, Pointer, Pointer, Pointer, cusparseMatDescr, Pointer, Pointer, Pointer)}
	 *
	 * @param sparseMatrixPtr CSR (compressed sparse row) pointer
	 * @throws DMLRuntimeException ?
	 */
	public void setSparseMatrixCudaPointer(CSRPointer sparseMatrixPtr) throws DMLRuntimeException {
		this.jcudaSparseMatrixPtr = sparseMatrixPtr;
		this.isSparse = true;
		if (getJcudaDenseMatrixPtr() != null) {
			cudaFreeHelper(getJcudaDenseMatrixPtr());
			jcudaDenseMatrixPtr = null;
		}
	}
	
	/**
	 * @return true if the GPUPointer is allocated in sparse format 
	 */
	public boolean isInSparseFormat() {
		return isSparse;
	}
	
	private Pointer allocate(long size) throws DMLRuntimeException {
		return getGPUContext().allocate(size);
	}
}
