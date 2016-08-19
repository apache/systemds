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

package org.apache.sysml.runtime.matrix.data;

import static jcuda.jcudnn.JCudnn.cudnnConvolutionBackwardData;
import static jcuda.jcudnn.JCudnn.cudnnConvolutionBackwardFilter;
import static jcuda.jcudnn.JCudnn.cudnnConvolutionForward;
import static jcuda.jcudnn.JCudnn.cudnnCreateConvolutionDescriptor;
import static jcuda.jcudnn.JCudnn.cudnnCreateFilterDescriptor;
import static jcuda.jcudnn.JCudnn.cudnnCreatePoolingDescriptor;
import static jcuda.jcudnn.JCudnn.cudnnCreateTensorDescriptor;
import static jcuda.jcudnn.JCudnn.cudnnDestroyConvolutionDescriptor;
import static jcuda.jcudnn.JCudnn.cudnnDestroyFilterDescriptor;
import static jcuda.jcudnn.JCudnn.cudnnDestroyPoolingDescriptor;
import static jcuda.jcudnn.JCudnn.cudnnDestroyTensorDescriptor;
import static jcuda.jcudnn.JCudnn.cudnnGetConvolutionBackwardDataWorkspaceSize;
import static jcuda.jcudnn.JCudnn.cudnnGetConvolutionBackwardFilterWorkspaceSize;
import static jcuda.jcudnn.JCudnn.cudnnGetConvolutionForwardWorkspaceSize;
import static jcuda.jcudnn.JCudnn.cudnnPoolingBackward;
import static jcuda.jcudnn.JCudnn.cudnnPoolingForward;
import static jcuda.jcudnn.JCudnn.cudnnSetConvolution2dDescriptor;
import static jcuda.jcudnn.JCudnn.cudnnSetFilter4dDescriptor;
import static jcuda.jcudnn.JCudnn.cudnnSetPooling2dDescriptor;
import static jcuda.jcudnn.JCudnn.cudnnSetTensor4dDescriptor;
import static jcuda.jcudnn.cudnnConvolutionMode.CUDNN_CROSS_CORRELATION;
import static jcuda.jcudnn.cudnnDataType.CUDNN_DATA_DOUBLE;
import static jcuda.jcudnn.cudnnPoolingMode.CUDNN_POOLING_MAX;
import static jcuda.jcudnn.cudnnTensorFormat.CUDNN_TENSOR_NCHW;
import static jcuda.jcusparse.JCusparse.cusparseDcsrgemm;
import static jcuda.jcusparse.JCusparse.cusparseDcsrmv;
import static jcuda.jcusparse.JCusparse.cusparseDdense2csr;
import static jcuda.jcusparse.JCusparse.cusparseDnnz;
import static jcuda.jcusparse.cusparseOperation.CUSPARSE_OPERATION_NON_TRANSPOSE;
import static jcuda.jcusparse.cusparseOperation.CUSPARSE_OPERATION_TRANSPOSE;
import static jcuda.runtime.JCuda.cudaFree;
import static jcuda.runtime.JCuda.cudaMalloc;

import static jcuda.jcusparse.cusparseOperation.CUSPARSE_OPERATION_NON_TRANSPOSE;
import static jcuda.jcusparse.cusparseOperation.CUSPARSE_OPERATION_TRANSPOSE;
import static jcuda.runtime.JCuda.cudaFree;
import static jcuda.runtime.JCuda.cudaMemcpy;
import static jcuda.runtime.cudaMemcpyKind.cudaMemcpyDeviceToHost;
import static jcuda.runtime.cudaMemcpyKind.cudaMemcpyHostToDevice;


import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;
import org.apache.sysml.runtime.DMLRuntimeException;
import org.apache.sysml.runtime.controlprogram.caching.MatrixObject;
import org.apache.sysml.runtime.controlprogram.context.ExecutionContext;
import org.apache.sysml.runtime.instructions.gpu.context.JCudaObject;
import org.apache.sysml.runtime.instructions.gpu.context.JCudaObject.CSRPointer;
import org.apache.sysml.utils.Statistics;

import jcuda.Pointer;
import jcuda.Sizeof;
import jcuda.jcublas.JCublas2;
import jcuda.jcublas.cublasFillMode;
import jcuda.jcublas.cublasHandle;
import jcuda.jcublas.cublasOperation;
import jcuda.jcudnn.cudnnConvolutionDescriptor;
import jcuda.jcudnn.cudnnConvolutionFwdPreference;
import jcuda.jcudnn.cudnnFilterDescriptor;
import jcuda.jcudnn.cudnnHandle;
import jcuda.jcudnn.cudnnPoolingDescriptor;
import jcuda.jcudnn.cudnnTensorDescriptor;
import jcuda.jcusparse.cusparseHandle;
import java.util.Arrays;

//FIXME move could to respective instructions, this is not a block library
public class LibMatrixCUDA {
	
	public static cudnnHandle cudnnHandle;
	public static cublasHandle cublasHandle;
	public static cusparseHandle cusparseHandle;

    private static final Log LOG = LogFactory.getLog(LibMatrixCUDA.class.getName());

	private static int CONVOLUTION_PREFERENCE = cudnnConvolutionFwdPreference.CUDNN_CONVOLUTION_FWD_NO_WORKSPACE;
	
	public static void conv2d(MatrixObject image, MatrixObject filter, MatrixObject outputBlock, int N, int C, int H, int W,
			int K, int R, int S, int pad_h, int pad_w, int stride_h, int stride_w, int P, int Q)
			throws DMLRuntimeException {
		cudnnTensorDescriptor srcTensorDesc = null;
		cudnnTensorDescriptor dstTensorDesc = null;
		cudnnFilterDescriptor filterDesc = null;
		cudnnConvolutionDescriptor convDesc = null;
		Pointer workSpace = null;
		long sizeInBytes = 0;
		Pointer alpha = null;
		Pointer beta = null;
		try {
			// Allocate descriptors
			srcTensorDesc = allocateTensorDescriptor(N, C, H, W);
			dstTensorDesc = allocateTensorDescriptor(N, K, P, Q);
			filterDesc = allocateFilterDescriptor(K, C, R, S);
			
			// Allocate data
			// (Pointer) gpuCtx.prepare(image, true, true);
			// (Pointer) gpuCtx.prepare(filter, true, true);
			
			Pointer imagePointer = ((JCudaObject)image.getGPUObject()).jcudaDenseMatrixPtr; 
			Pointer filterPointer = ((JCudaObject)filter.getGPUObject()).jcudaDenseMatrixPtr; 
			Pointer dstPointer = ((JCudaObject)outputBlock.getGPUObject()).jcudaDenseMatrixPtr; 
			
			int padding [] = { pad_h, pad_w }; 
			int strides [] = { stride_h, stride_w };
			convDesc = allocateConvolutionDescriptor(padding, strides);
			
			// Select the best algorithm depending on the data and supported CUDA
			
			int algo = -1; 
			workSpace = new Pointer();
			
			if(CONVOLUTION_PREFERENCE == cudnnConvolutionFwdPreference.CUDNN_CONVOLUTION_FWD_NO_WORKSPACE) {
				algo = jcuda.jcudnn.cudnnConvolutionFwdAlgo.CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM;
			}
			else if(CONVOLUTION_PREFERENCE == cudnnConvolutionFwdPreference.CUDNN_CONVOLUTION_FWD_PREFER_FASTEST) {
				int [] algos = {
	            		jcuda.jcudnn.cudnnConvolutionFwdAlgo.CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM,
	            		jcuda.jcudnn.cudnnConvolutionFwdAlgo.CUDNN_CONVOLUTION_FWD_ALGO_GEMM,
	            		jcuda.jcudnn.cudnnConvolutionFwdAlgo.CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM
	            };
				// TODO: Look into FFt, Winograd, etc
				// Also ensure that GPU has enough memory to allocate memory
				long sizeInBytesArray[] = { 0 };
	            algo = jcuda.jcudnn.JCudnn.cudnnGetConvolutionForwardAlgorithm(cudnnHandle, srcTensorDesc, filterDesc, convDesc, dstTensorDesc,
	            		CONVOLUTION_PREFERENCE, sizeInBytesArray[0], algos);
	            cudnnGetConvolutionForwardWorkspaceSize(cudnnHandle, srcTensorDesc, filterDesc, convDesc, dstTensorDesc, algo, sizeInBytesArray);
	            if(sizeInBytesArray[0] != 0)
	            	jcuda.runtime.JCuda.cudaMalloc(workSpace, sizeInBytesArray[0]);
	            sizeInBytes = sizeInBytesArray[0];
			}
			else if(CONVOLUTION_PREFERENCE == cudnnConvolutionFwdPreference.CUDNN_CONVOLUTION_FWD_SPECIFY_WORKSPACE_LIMIT) {
				throw new DMLRuntimeException("CUDNN_CONVOLUTION_FWD_SPECIFY_WORKSPACE_LIMIT is not implemented");
			}
			else {
				throw new DMLRuntimeException("Unsupported preference criteria for convolution");
			}
			
			alpha = pointerTo(1.0);
			beta = pointerTo(0.0f);
			int status = cudnnConvolutionForward(cudnnHandle, alpha, 
					srcTensorDesc, imagePointer, 
					filterDesc, filterPointer,
					convDesc, algo, workSpace, sizeInBytes, beta,
					dstTensorDesc, dstPointer);
			if(status != jcuda.jcudnn.cudnnStatus.CUDNN_STATUS_SUCCESS) {
				throw new DMLRuntimeException("Could not executed cudnnConvolutionForward: " + jcuda.jcudnn.cudnnStatus.stringFor(status));
			}
		}
		finally {
			
			if(alpha != null)
				cudaFree(alpha);
			if(beta != null)
				cudaFree(beta);
			
			if(srcTensorDesc != null)
				cudnnDestroyTensorDescriptor(srcTensorDesc);
			if(dstTensorDesc != null)
				cudnnDestroyTensorDescriptor(dstTensorDesc);
			if(filterDesc != null)
				cudnnDestroyFilterDescriptor(filterDesc);
			if(convDesc != null)
				cudnnDestroyConvolutionDescriptor(convDesc);
			if(workSpace != null && sizeInBytes != 0)
				cudaFree(workSpace);
		}
	}	
	
	private static cudnnConvolutionDescriptor allocateConvolutionDescriptor(int padding [], int strides []) {
		cudnnConvolutionDescriptor convDesc = new cudnnConvolutionDescriptor();
		cudnnCreateConvolutionDescriptor(convDesc);
		cudnnSetConvolution2dDescriptor(convDesc, padding[0], padding[1], strides[0], strides[1], 1, 1, CUDNN_CROSS_CORRELATION);		
		return convDesc;
	}
	
	private static  Pointer pointerTo(double value) {
        return Pointer.to(new double[] { value });
    }
	
	private static  cudnnTensorDescriptor allocateTensorDescriptor(int N, int C, int H, int W) {
		cudnnTensorDescriptor ret = new cudnnTensorDescriptor();
		cudnnCreateTensorDescriptor(ret);
		cudnnSetTensor4dDescriptor(ret, CUDNN_TENSOR_NCHW, CUDNN_DATA_DOUBLE, N, C, H, W);
		return ret;
	}
	
	private static cudnnFilterDescriptor allocateFilterDescriptor(int K, int C, int R, int S) {
		cudnnFilterDescriptor filterDesc = new cudnnFilterDescriptor();
		cudnnCreateFilterDescriptor(filterDesc);
		cudnnSetFilter4dDescriptor(filterDesc, CUDNN_DATA_DOUBLE, K, C, R, S);
		return filterDesc;
	}
	
	/**
	 * allocates pooling descriptor, used in poolingForward and poolingBackward
	 * @param R			pooling window height
	 * @param S			pooling window width
	 * @param pad_h		vertical padding
	 * @param pad_w		horizontal padding
	 * @param stride_h	pooling vertical stride
	 * @param stride_w	pooling horizontal stride
	 * @return
	 */
	private static cudnnPoolingDescriptor allocatePoolingDescriptor(int R, int S, int pad_h, int pad_w, int stride_h, int stride_w) {
		cudnnPoolingDescriptor poolingDesc = new cudnnPoolingDescriptor();
		cudnnCreatePoolingDescriptor(poolingDesc);
		cudnnSetPooling2dDescriptor(poolingDesc, CUDNN_POOLING_MAX, R, S, pad_h, pad_w, stride_h, stride_w);
		return poolingDesc;
	}
	
	public static void conv2d_backward_filter(MatrixObject image, MatrixObject dout,
			MatrixObject outputBlock, int N, int C, int H, int W, int K, int R,
			int S, int pad_h, int pad_w, int stride_h, int stride_w, int P,
			int Q) throws DMLRuntimeException {
		Pointer alpha = null;
		Pointer beta = null;
		cudnnTensorDescriptor xTensorDesc = null;
		cudnnTensorDescriptor doutTensorDesc = null;
		cudnnFilterDescriptor dwDesc = null;
		cudnnConvolutionDescriptor convDesc = null;
		
		Pointer workSpace = null;
		long sizeInBytes = 0;
		try {
			// Allocate descriptors
			xTensorDesc = allocateTensorDescriptor(N, C, H, W);
			doutTensorDesc = allocateTensorDescriptor(N, K, P, Q);
			dwDesc = allocateFilterDescriptor(K, C, R, S);
			
			// Allocate data
			Pointer imagePointer = ((JCudaObject)image.getGPUObject()).jcudaDenseMatrixPtr; 
			Pointer doutPointer = ((JCudaObject)dout.getGPUObject()).jcudaDenseMatrixPtr; 
			Pointer dwPointer = ((JCudaObject)outputBlock.getGPUObject()).jcudaDenseMatrixPtr; 
			
			alpha = pointerTo(1.0); // TODO
			beta = pointerTo(0.0f);
			
			int padding [] = { pad_h, pad_w }; 
			int strides [] = { stride_h, stride_w };
			convDesc = allocateConvolutionDescriptor(padding, strides);
			long sizeInBytesArray[] = { 0 };
			
			// TODO: Select the best algorithm depending on the data and supported CUDA
			int algo = jcuda.jcudnn.cudnnConvolutionBwdFilterAlgo.CUDNN_CONVOLUTION_BWD_FILTER_ALGO_0;
			workSpace = new Pointer();
			cudnnGetConvolutionBackwardFilterWorkspaceSize(cudnnHandle,
					xTensorDesc, doutTensorDesc, convDesc, dwDesc, algo, sizeInBytesArray);
			
			int status = cudnnConvolutionBackwardFilter(cudnnHandle, alpha, xTensorDesc, imagePointer, 
					doutTensorDesc, doutPointer, convDesc, algo, workSpace, sizeInBytes, beta, dwDesc, dwPointer);
			if(status != jcuda.jcudnn.cudnnStatus.CUDNN_STATUS_SUCCESS) {
				throw new DMLRuntimeException("Could not executed cudnnConvolutionBackwardFilter: " + jcuda.jcudnn.cudnnStatus.stringFor(status));
			}
		}
		finally {
			if(alpha != null)
				cudaFree(alpha);
			if(beta != null)
				cudaFree(beta);
			if(xTensorDesc != null)
				cudnnDestroyTensorDescriptor(xTensorDesc);
			if(doutTensorDesc != null)
				cudnnDestroyTensorDescriptor(doutTensorDesc);
			if(dwDesc != null)
				cudnnDestroyFilterDescriptor(dwDesc);
			
			if(convDesc != null)
				cudnnDestroyConvolutionDescriptor(convDesc);
			
			if(workSpace != null && sizeInBytes != 0)
				cudaFree(workSpace);
		}
		
	}

	/**
	 * Performs tsmm, A %*% A' or A' %*% A, on GPU by exploiting cublasDsyrk(...)
	 * @param left	input matrix, as in a tsmm expression like A %*% A' or A' %*% A, we just need to check whether the left one is transposed or not, I named it 'left'
	 * @param output
	 * @param isLeftTransposed
	 * @throws DMLRuntimeException
	 */
	public static void matmultTSMM(MatrixObject left, MatrixObject output,
            boolean isLeftTransposed) throws DMLRuntimeException {
	    if(isInSparseFormat(left)) {
	            throw new DMLRuntimeException("Sparse GPU TSMM is not implemented");
	    }
	
	    // Since CuBLAS expects inputs in column-major format,
	    // reverse the order of matrix-multiplication and take care of dimension mismatch.      
	    int transa = isLeftTransposed ? cublasOperation.CUBLAS_OP_N : cublasOperation.CUBLAS_OP_T;
	    // Note: the dimensions are swapped
	    int m = (int) (isLeftTransposed ? left.getNumColumns() : left.getNumRows());
	    int k = (int) (isLeftTransposed ? left.getNumRows() : left.getNumColumns());
	
	    if(m == -1)
	            throw new DMLRuntimeException("Incorrect dimensions");
	
	    double[] alpha = {1.0d};
	    double[] beta = {0.0d};
	
	    int lda = (int) (isLeftTransposed ? m : k);
	    int ldc = m;
	
	    if(!left.getGPUObject().isAllocated)
	            throw new DMLRuntimeException("Input is not allocated:" + left.getGPUObject().isAllocated);
	    if(!output.getGPUObject().isAllocated)
	            throw new DMLRuntimeException("Output is not allocated:" + output.getGPUObject().isAllocated);
	
	    Pointer A = ((JCudaObject)left.getGPUObject()).jcudaDenseMatrixPtr;
	    Pointer C = ((JCudaObject)output.getGPUObject()).jcudaDenseMatrixPtr;
	    
	    //TODO: Fix it if there is a cuBLAS API to do flipping
	    
	    JCublas2.cublasDsyrk(cublasHandle, cublasFillMode.CUBLAS_FILL_MODE_UPPER,transa, m, k, Pointer.to(alpha), A, lda, Pointer.to(beta), C, ldc);
	    JCublas2.cublasDsyrk(cublasHandle, cublasFillMode.CUBLAS_FILL_MODE_LOWER,transa, m, k, Pointer.to(alpha), A, lda, Pointer.to(beta), C, ldc);
	}
	
	/**
	 * Matrix multiply on GPU
	 * Examines sparsity and shapes and routes call to appropriate method
	 * from cuBLAS or cuSparse
	 * C = op(A) x op(B)
	 * @param ec					Current {@link ExecutionContext} instance
	 * @param left1					Matrix A
	 * @param right1				Matrix B
	 * @param outputName			Name of the output matrix C (in code generated after LOP layer)
	 * @param isLeftTransposed1		op for A, transposed or not
	 * @param isRightTransposed1	op for B, tranposed or not
	 * @return	output of matrix multiply
	 * @throws DMLRuntimeException
	 */
	public static MatrixObject matmult(ExecutionContext ec, MatrixObject left1, MatrixObject right1, String outputName,
			boolean isLeftTransposed1, boolean isRightTransposed1) throws DMLRuntimeException {
		
		if(!left1.getGPUObject().isAllocated() || !right1.getGPUObject().isAllocated())
			throw new DMLRuntimeException("One of input is not allocated:" + left1.getGPUObject().isAllocated() + " " + right1.getGPUObject().isAllocated());
		
		boolean bothDense = !left1.getGPUObject().isInSparseFormat() && !right1.getGPUObject().isInSparseFormat();
		boolean bothSparse = left1.getGPUObject().isInSparseFormat() && right1.getGPUObject().isInSparseFormat();
		
		MatrixObject output = ec.getMatrixObject(outputName);

		if (bothDense) {		// Dense C = Dense A * Dense B
			// For both dense, do cuBLAS
			ec.getDenseMatrixOutputForGPUInstruction(outputName);	// Allocated the dense output matrix
			denseDenseMatmult(output, left1, right1, isLeftTransposed1, isRightTransposed1);
		}
		else if (bothSparse){	// Sparse C = Sparse A * Sparse B
			ec.allocateGPUMatrixObject(outputName);
			bothSparseMatmult(output, left1, right1, isLeftTransposed1, isRightTransposed1);
		}
		else {	// Either of A or B is sparse, Sparse C = Sparse/Dense A * Dense/Sparse B
				// Convert the dense to sparse and use the cusparseDcsrgemm routine
			ec.allocateGPUMatrixObject(outputName);
			eitherSparseMatmult(output, left1, right1, isLeftTransposed1, isRightTransposed1);
		}
		
		return output;
	}
	
	/**
	 * One of the matrices is sparse, the other dense
	 * C = op(A) x op(B)
	 * @param output				allocated output object for C on host to which GPU output will be attached
	 * @param left					Matrix A on host
	 * @param right					Matrix B on host
	 * @param isLeftTransposed		op for A, tranposed or not
	 * @param isRightTransposed		op for B, transposed or not
	 * @throws DMLRuntimeException
	 */
	protected static void eitherSparseMatmult(MatrixObject output, MatrixObject left, MatrixObject right,
			boolean isLeftTransposed, boolean isRightTransposed) throws DMLRuntimeException {
		
		int transA = isLeftTransposed ? CUSPARSE_OPERATION_TRANSPOSE : CUSPARSE_OPERATION_NON_TRANSPOSE;
		int transB = isRightTransposed ? CUSPARSE_OPERATION_TRANSPOSE : CUSPARSE_OPERATION_NON_TRANSPOSE;
		
		int m = (int) (isLeftTransposed ? left.getNumColumns() : left.getNumRows()) ;
		int n = (int) (isRightTransposed ? right.getNumRows() : right.getNumColumns());
		int k = (int) (isLeftTransposed ? left.getNumRows() :  left.getNumColumns());
		int k1 = (int) (isRightTransposed ? right.getNumColumns() : right.getNumRows());
		if(k != k1) 
			throw new DMLRuntimeException("Dimension mismatch: " + k + " != " + k1);
		
		if(m == -1 || n == -1 || k == -1)
			throw new DMLRuntimeException("Incorrect dimensions");
		
		
		if (left.getGPUObject().isInSparseFormat()) {	
			// Left sparse, right dense
			sparseDenseMatmult(output, left, right, isLeftTransposed, isRightTransposed, transA, transB, m, n, k);
		} else {
			// Left dense, right sparse
			denseSparseMatmult(output, right, left, isLeftTransposed, isRightTransposed, transA, transB, m, n, k);
		}
	}
	
	/**
	 * C = op(A) * op(B) where A is dense and B is sparse
	 * If B is ultrasparse, A is converted to a sparse matrix and {@link #sparseSparseMatmult(MatrixObject, int, int, int, int, int, CSRPointer, CSRPointer)} is invoked
	 * otherwise B is converted to a dense matrix and {@link #denseDenseMatmult(MatrixObject, int, int, int, int, boolean, boolean, Pointer, Pointer)} is invoked.
	 * @param output
	 * @param right
	 * @param left
	 * @param isLeftTransposed
	 * @param isRightTransposed
	 * @param transA
	 * @param transB
	 * @param m
	 * @param n
	 * @param k
	 * @throws DMLRuntimeException
	 */
	protected static void denseSparseMatmult(MatrixObject output, MatrixObject right, MatrixObject left,
			boolean isLeftTransposed, boolean isRightTransposed, int transA, int transB, int m, int n, int k)
			throws DMLRuntimeException {
		// right sparse, left dense
		CSRPointer B = ((JCudaObject)right.getGPUObject()).jcudaSparseMatrixPtr;
		Pointer ADense = ((JCudaObject)left.getGPUObject()).jcudaDenseMatrixPtr;
		if (B.isUltraSparse(k, n)){
			LOG.info(" GPU Dense-Sparse Matrix Multiplication (Converted to Sparse-Sparse)");
			// Convert left to CSR and do cuSparse matmul
			long t0 = System.nanoTime();
			CSRPointer A = JCudaObject.denseToSparse(cusparseHandle, (int)left.getNumRows(), (int)right.getNumColumns(), ADense);
			Statistics.cudaConversionTime.addAndGet(System.nanoTime() - t0);
			Statistics.cudaConversionCount.addAndGet(1);
			sparseSparseMatmult(output, transA, transB, m, n, k, A, B);
			A.deallocate();
		} else {
			LOG.info(" GPU Dense-Sparse Matrix Multiplication (Converted to Dense-Dense)");
			// Convert right to dense and do a cuBlas matmul
			Pointer BDenseTransposed = B.toDenseMatrix(cusparseHandle, cublasHandle, (int)right.getNumRows(), (int)right.getNumColumns());
			output.getGPUObject().acquireDeviceModifyDense();	// To allocate the dense matrix
			Pointer C = ((JCudaObject)output.getGPUObject()).jcudaDenseMatrixPtr;		
			denseDenseMatmult(C, 
					(int) left.getNumRows(), (int) left.getNumColumns(),
					(int) right.getNumColumns(), (int) right.getNumRows(), 
					isLeftTransposed, !isRightTransposed,
					ADense, BDenseTransposed);
			cudaFree(BDenseTransposed);
		}
	}

	/**
	 * * C = op(A) * op(B) where A is sparse and B is dense
	 * If A is ultrasparse, B is converted to a sparse matrix and {@link #sparseSparseMatmult(MatrixObject, int, int, int, int, int, CSRPointer, CSRPointer)} is invoked
	 * otherwise A is converted to a dense matrix and {@link #denseDenseMatmult(MatrixObject, int, int, int, int, boolean, boolean, Pointer, Pointer)} is invoked.
	 * @param output
	 * @param left
	 * @param right
	 * @param isLeftTransposed
	 * @param isRightTransposed
	 * @param transA
	 * @param transB
	 * @param m
	 * @param n
	 * @param k
	 * @throws DMLRuntimeException
	 */
	protected static void sparseDenseMatmult(MatrixObject output, MatrixObject left, MatrixObject right,
			boolean isLeftTransposed, boolean isRightTransposed, int transA, int transB, int m, int n, int k)
			throws DMLRuntimeException {
		CSRPointer A = ((JCudaObject)left.getGPUObject()).jcudaSparseMatrixPtr;
		Pointer BDense = ((JCudaObject)right.getGPUObject()).jcudaDenseMatrixPtr;
		
		if (n == 1){	
			// Sparse Matrix - Dense Vector multiply
			LOG.info(" GPU Sparse Matrix - Dense Vector Mutliply");
			sparseMatrixDenseVectorMult(output, A, BDense, transA, (int)left.getNumRows(), (int)right.getNumColumns(), (int)left.getNumColumns());
			
		} else {
			// Sparse Matrix Dense Matrix multiply
			if (A.isUltraSparse(m, k)){	
				LOG.info(" GPU Sparse-Dense Matrix Multiplication (Converted to Sparse-Sparse)");
				// Convert right to CSR and do cuSparse matmul
				long t0 = System.nanoTime();
				CSRPointer B = JCudaObject.denseToSparse(cusparseHandle, (int)right.getNumRows(), (int)right.getNumColumns(), BDense);
				Statistics.cudaConversionTime.addAndGet(System.nanoTime() - t0);
				Statistics.cudaConversionCount.addAndGet(1);
				sparseSparseMatmult(output, transA, transB, m, n, k, A, B);
				B.deallocate();
			} else {					
				LOG.info(" GPU Sparse-Dense Matrix Multiplication (Converted to Dense-Dense)");
				// Convert left to dense and do a cuBlas matmul
				Pointer ADenseTransposed = A.toDenseMatrix(cusparseHandle, cublasHandle, (int)left.getNumRows(), (int)left.getNumColumns());
				output.getGPUObject().acquireDeviceModifyDense();	// To allocate the dense matrix
				Pointer C = ((JCudaObject)output.getGPUObject()).jcudaDenseMatrixPtr;		
				denseDenseMatmult(C, 
						(int) left.getNumColumns(), (int) left.getNumRows(),
						(int) right.getNumRows(), (int) right.getNumColumns(), 
						!isLeftTransposed, isRightTransposed,
						ADenseTransposed, BDense);
				cudaFree(ADenseTransposed);
			}
		}
	}

	/**
	 * C = op(A) x B
	 * A is a sparse matrix, B is a dense vector
	 * @param output	allocated output on the host, to which the GPU output C will be attached
	 * @param A			sparse matrix A on the GPU
	 * @param B_dense	dense matrix/vector B on the GPU
	 * @param transA	op for A, tranposed or not
	 * @param m			number of rows in A (not op(A))
	 * @param n			number of cols in B (not op(B))
	 * @param k			number of cols in A or number of rows in B (not op(A) or op(B))
	 * @throws DMLRuntimeException
	 */
	protected static void sparseMatrixDenseVectorMult(MatrixObject output, CSRPointer A, Pointer B_dense, int transA,
			int m, int n, int k) throws DMLRuntimeException {
		long size = m * n * Sizeof.DOUBLE;
		Pointer C_dense = JCudaObject.allocate((int)size);
		double[] alpha = { 1 };
		double[] beta = { 0 };
		cusparseDcsrmv(cusparseHandle, transA, m, k, (int)A.nnz, Pointer.to(alpha), A.descr, A.val, A.rowPtr, A.colInd, B_dense, Pointer.to(beta), C_dense);
		
//		long t0 = System.nanoTime();
//		CSRPointer C = JCudaObject.denseToSparse(cusparseHandle, m, n, C_dense);
//		Statistics.cudaConversionTime.addAndGet(System.nanoTime() - t0);
//		Statistics.cudaConversionCount.addAndGet(1);
		
//		((JCudaObject)output.getGPUObject()).setSparseMatrixCudaPointer(C);
//		long sizeOfC = CSRPointer.estimateSize(C.nnz, output.getNumRows());
//		output.getGPUObject().setDeviceModify(sizeOfC);
//		cudaFree(C_dense);
		((JCudaObject)(output.getGPUObject())).setDenseMatrixCudaPointer(C_dense);
		output.getGPUObject().setDeviceModify(size);
	}

	/**
	 * Sparse C = Sparse op(A) * Sparse op(B)
	 * Reroutes call to sparse matrix-vector mult if needed
	 * @param output
	 * @param left
	 * @param right
	 * @param isLeftTransposed
	 * @param isRightTransposed
	 * @throws DMLRuntimeException
	 */
	protected static void bothSparseMatmult(MatrixObject output, MatrixObject left, MatrixObject right,
			boolean isLeftTransposed, boolean isRightTransposed) throws DMLRuntimeException {
		
		int transA = isLeftTransposed ? CUSPARSE_OPERATION_TRANSPOSE : CUSPARSE_OPERATION_NON_TRANSPOSE;
		int transB = isRightTransposed ? CUSPARSE_OPERATION_TRANSPOSE : CUSPARSE_OPERATION_NON_TRANSPOSE;
		
		int m = (int) (isLeftTransposed ? left.getNumColumns() : left.getNumRows()) ;
		int n = (int) (isRightTransposed ? right.getNumRows() : right.getNumColumns());
		int k = (int) (isLeftTransposed ? left.getNumRows() :  left.getNumColumns());
		int k1 = (int) (isRightTransposed ? right.getNumColumns() : right.getNumRows());
		if(k != k1) 
			throw new DMLRuntimeException("Dimension mismatch: " + k + " != " + k1);
		
		if(m == -1 || n == -1 || k == -1)
			throw new DMLRuntimeException("Incorrect dimensions");
			
		CSRPointer A = ((JCudaObject)left.getGPUObject()).jcudaSparseMatrixPtr;
		CSRPointer B = ((JCudaObject)right.getGPUObject()).jcudaSparseMatrixPtr;
		
		// TODO if (m == 1) {	// Vector-matrix multiplication
		
		if (!isRightTransposed && right.getNumColumns() == 1){ 	// Matrix-Vector multiplication
			sparseMatrixVectorMult(output, transA, (int)left.getNumRows(), (int)left.getNumColumns(), (int)right.getNumRows(), A, B);
		} else {												// Matrix-Matrix multiplication
			sparseSparseMatmult(output, transA, transB, m, n, k, A, B);
		}
	}

	/**
	 * Does a sparse matrix-vector multiply.
	 * C = op(A) x B, A is a sparse matrix, B is a sparse vector with numCols = 1.
	 * @param output	allocated output object C to which the GPU output matrix will be attached
	 * @param transA	if A is to be transposed or not (the op in op(A))
	 * @param m			number of rows in A (not op(A))
	 * @param n			number of cols in A (not op(A))
	 * @param k			number of rows in B, (cols in B is assumed to be 1)		
	 * @param A			left sparse matrix on GPU
	 * @param B			right sparse vector on GPU
	 * @throws DMLRuntimeException
	 */
	protected static void sparseMatrixVectorMult(MatrixObject output, int transA, int m, int n, int k,
			CSRPointer A, CSRPointer B) throws DMLRuntimeException {
		LOG.info(" GPU Sparse Matrix Sparse Vector Multiply (Converted to Sparse Matrix Dense Vector Multiply)");
		Pointer BDenseVector = B.toDenseMatrix(cusparseHandle, cublasHandle, k, 1);
		sparseMatrixDenseVectorMult(output, A, BDenseVector, transA, m, n, k);
	}

	/**
	 * Does a sparse-sparse Matrix multiply
	 * C = op(A) x op(B), A, B are sparse matrices
	 * @param output	allocated output object on host to which the GPU output matrix will be attached
	 * @param transA	op for A - to be transposed or not
	 * @param transB	op for B
	 * @param m			number of rows in op(A)
	 * @param n			number of cols in op(B)
	 * @param k			number of cols in op(A) or rows in op(B)
	 * @param A			left sparse matrix on GPU
	 * @param B			right sparse matrix on GPU
	 * @throws DMLRuntimeException
	 */
	protected static void sparseSparseMatmult(MatrixObject output, int transA, int transB, int m, int n, int k,
			CSRPointer A, CSRPointer B) throws DMLRuntimeException {
		LOG.info(" GPU Sparse-Sparse Matrix Multiply ");

		CSRPointer C = CSRPointer.allocateForMatrixMultiply(cusparseHandle, A, transA, B, transB, m, n, k);
		((JCudaObject)output.getGPUObject()).setSparseMatrixCudaPointer(C);
		long sizeOfC = CSRPointer.estimateSize(C.nnz, output.getNumRows());
		output.getGPUObject().setDeviceModify(sizeOfC);
		
		cusparseDcsrgemm(cusparseHandle, transA, transB, m, n, k,
				A.descr, (int)A.nnz, A.val, A.rowPtr, A.colInd,
				B.descr, (int)B.nnz, B.val, B.rowPtr, B.colInd,
				C.descr, C.val, C.rowPtr, C.colInd);
	}

	/**
	 * Dense dense matrix multiply
	 * C = op(A) * op(B), A and B are dense matrices
	 * @param output				output object C on host with GPU data allocated				
	 * @param left1					left matrix A on host (in row-major order)
	 * @param right1				right matrix B on host (in row-major order)
	 * @param isLeftTransposed1 	op for A, transposed or not
	 * @param isRightTransposed1	op for B, transposed or not
	 * @return
	 * @throws DMLRuntimeException
	 */
	protected static void denseDenseMatmult(MatrixObject output, MatrixObject left1, MatrixObject right1,
			boolean isLeftTransposed1, boolean isRightTransposed1) throws DMLRuntimeException {
		
		Pointer leftPtr = ((JCudaObject)left1.getGPUObject()).jcudaDenseMatrixPtr;
		Pointer rightPtr = ((JCudaObject)right1.getGPUObject()).jcudaDenseMatrixPtr;
		
		int leftRows = (int) left1.getNumRows();
		int leftCols = (int) left1.getNumColumns();
		int rightRows = (int) right1.getNumRows();
		int rightCols = (int) right1.getNumColumns();
		Pointer C = ((JCudaObject)output.getGPUObject()).jcudaDenseMatrixPtr;		
		denseDenseMatmult(C, leftRows, leftCols, rightRows, rightCols, isLeftTransposed1, isRightTransposed1,
				leftPtr, rightPtr);
	}

	/**
	 * Dense-dense matrix multiply
	 * C = op(A) * op(B), A and B are dense matrices
	 * On the host, the matrices are in row-major format; cuBLAS expects them in column-major format.
	 * What we have as input is t(A) and t(B), t(X) = transpose of X.
	 * We do t(B) %*% t(A) to get t(C); 
	 * If we were to calculate t(t(C), we would get the resultant matrix C, but this would be in column-major format.
	 * What we really want is t(C). This we already have as the result of t(B) %*% t(A).
	 * @param output			output allocated on GPU in column major format
	 * @param leftRows1			number of rows in A
	 * @param leftCols1			number of cols in A
	 * @param rightRows1		number of rows in B
	 * @param rightCols1		number of cols in B
	 * @param isLeftTransposed1		op for A, transposed or not
	 * @param isRightTransposed1	op for B, transposed or not
	 * @param leftPtr			A allocated on the GPU in row-major format
	 * @param rightPtr			B allocated on the GPU in row-major format
	 * @throws DMLRuntimeException
	 */
	public static void denseDenseMatmult(Pointer output, int leftRows1, int leftCols1, int rightRows1,
			int rightCols1, boolean isLeftTransposed1, boolean isRightTransposed1, Pointer leftPtr, Pointer rightPtr)
			throws DMLRuntimeException {
		
		Pointer A = rightPtr;
		Pointer B = leftPtr;
		
		int leftRows = rightCols1;
		int leftCols = rightRows1;
		int rightRows = leftCols1;
		int rightCols = leftRows1;
		
		boolean isLeftTransposed = isRightTransposed1; 
		boolean isRightTransposed = isLeftTransposed1; 
		
		// Note: the dimensions are swapped
		int m = (int) (isLeftTransposed ? leftCols : leftRows) ;
		int n = (int) (isRightTransposed ? rightRows : rightCols);
		int k = (int) (isLeftTransposed ?  leftRows : leftCols);
		int k1 = (int) (isRightTransposed ?  rightCols : rightRows);
		if(k != k1) 
			throw new DMLRuntimeException("Dimension mismatch: " + k + " != " + k1);
		
		if(m == -1 || n == -1 || k == -1)
			throw new DMLRuntimeException("Incorrect dimensions");
		
		double[] one = { 1 };
		double[] zero = { 0 };
		
		int lda = leftRows;
		int ldb = leftCols;
		int ldc = m;
		
		int transa = isLeftTransposed ? cublasOperation.CUBLAS_OP_T : cublasOperation.CUBLAS_OP_N;
		int transb = isRightTransposed ? cublasOperation.CUBLAS_OP_T : cublasOperation.CUBLAS_OP_N;

		Pointer C = output;
		if (m == 1 && n == 1){ 
			// Vector product
			LOG.info(" GPU Dense-dense Vector Product");
			double[] result = {0};
			JCublas2.cublasDdot(cublasHandle, k, A, 1, B, 1, Pointer.to(result));
			// By default in CuBlas V2, cublas pointer mode is set to CUBLAS_POINTER_MODE_HOST.
			// This means that scalar values passed are on host (as opposed to on device).
			// The result is copied from the host back to the device so that the rest of 
			// infrastructure can treat it uniformly.
			cudaMemcpy(C, Pointer.to(result), 1 * Sizeof.DOUBLE, cudaMemcpyHostToDevice);
		} else if (m == 1) {
			// Vector-matrix multiply
			LOG.info(" GPU Dense Vector-Matrix Multiply");
			transb = isRightTransposed ? cublasOperation.CUBLAS_OP_N : cublasOperation.CUBLAS_OP_T;
			JCublas2.cublasDgemv(cublasHandle, transb, k, n, Pointer.to(one), B, ldb, A, 1, Pointer.to(zero), C, 1);
		} else if (n == 1){
			// Matrix-vector multiply
			LOG.info(" GPU Dense Matrix-Vector Multiply");
			JCublas2.cublasDgemv(cublasHandle, transa, m, k, Pointer.to(one), A, lda, B, 1, Pointer.to(zero), C, 1);
		} else {
			LOG.info(" GPU Dense-Dense Matrix Multiply ");
			JCublas2.cublasDgemm(cublasHandle, transa, transb, m, n, k, Pointer.to(one), A, lda, B, ldb, Pointer.to(zero), C, ldc);
		}
	}

	public static void conv2d_backward_data(MatrixObject filter, MatrixObject dout,
			MatrixObject output, int N, int C, int H, int W, int K, int R,
			int S, int pad_h, int pad_w, int stride_h, int stride_w, int P,
			int Q) throws DMLRuntimeException {
		Pointer alpha = null;
		Pointer beta = null;
		cudnnTensorDescriptor dyDesc = null;
		cudnnTensorDescriptor dxDesc = null;
		cudnnFilterDescriptor wDesc = null;
		cudnnConvolutionDescriptor convDesc = null;
		
		Pointer workSpace = null;
		long sizeInBytes = 0;
		try {
			// Allocate descriptors
			wDesc = allocateFilterDescriptor(K, C, R, S);
			dyDesc = allocateTensorDescriptor(N, K, P, Q);
			dxDesc = allocateTensorDescriptor(N, C, H, W);
			
			// Allocate data
			Pointer w = ((JCudaObject)filter.getGPUObject()).jcudaDenseMatrixPtr; 
			Pointer dy = ((JCudaObject)dout.getGPUObject()).jcudaDenseMatrixPtr; 
			Pointer dx = ((JCudaObject)output.getGPUObject()).jcudaDenseMatrixPtr; 

			alpha = pointerTo(1.0); // TODO
			beta = pointerTo(0.0f);
			
			int padding [] = { pad_h, pad_w }; 
			int strides [] = { stride_h, stride_w };
			convDesc = allocateConvolutionDescriptor(padding, strides);
			long sizeInBytesArray[] = { 0 };
			
			// TODO: Select the best algorithm depending on the data and supported CUDA
			int algo = jcuda.jcudnn.cudnnConvolutionBwdDataAlgo.CUDNN_CONVOLUTION_BWD_DATA_ALGO_0;
			workSpace = new Pointer();
			cudnnGetConvolutionBackwardDataWorkspaceSize(cudnnHandle,
					wDesc, dyDesc, convDesc, dxDesc, algo, sizeInBytesArray);
			
			int status = cudnnConvolutionBackwardData(cudnnHandle, alpha, wDesc, w, 
					dyDesc, dy, convDesc, algo, workSpace, sizeInBytes, beta, dxDesc, dx);
			if(status != jcuda.jcudnn.cudnnStatus.CUDNN_STATUS_SUCCESS) {
				throw new DMLRuntimeException("Could not executed cudnnConvolutionBackwardData: " + jcuda.jcudnn.cudnnStatus.stringFor(status));
			}
		}
		finally {
			if(alpha != null)
				cudaFree(alpha);
			if(beta != null)
				cudaFree(beta);
			if(dyDesc != null)
				cudnnDestroyTensorDescriptor(dyDesc);
			if(dxDesc != null)
				cudnnDestroyTensorDescriptor(dxDesc);
			if(wDesc != null)
				cudnnDestroyFilterDescriptor(wDesc);
			
			if(convDesc != null)
				cudnnDestroyConvolutionDescriptor(convDesc);
			
			if(workSpace != null && sizeInBytes != 0)
				cudaFree(workSpace);
		}
	}
	
	/**
	 * performs maxpooling on GPU by exploiting cudnnPoolingForward(...)
	 * @param image
	 * @param outputBlock
	 * @param N				batch size
	 * @param C				number of channels
	 * @param H				height of image
	 * @param W				width of image
	 * @param K				number of filters
	 * @param R				height of filter
	 * @param S				width of filter
	 * @param pad_h			vertical padding
	 * @param pad_w			horizontal padding
	 * @param stride_h		horizontal stride
	 * @param stride_w		vertical stride
	 * @param P				(H - R + 1 + 2*pad_h)/stride_h
	 * @param Q				(W - S + 1 + 2*pad_w)/stride_w
	 * @throws DMLRuntimeException
	 */
	public static void maxpooling(MatrixObject image,
			MatrixObject outputBlock, int N, int C, int H, int W, int K, int R,
			int S, int pad_h, int pad_w, int stride_h, int stride_w, int P,
			int Q) throws DMLRuntimeException {
		Pointer alpha = null;
		Pointer beta = null;
		cudnnTensorDescriptor xDesc = null;
		cudnnTensorDescriptor yDesc = null;
		cudnnPoolingDescriptor poolingDesc = null;

		try {
			// Allocate descriptors
			yDesc = allocateTensorDescriptor(N, C, P, Q);
			xDesc = allocateTensorDescriptor(N, C, H, W);
			poolingDesc = allocatePoolingDescriptor(R, S, pad_h, pad_w, stride_h, stride_w);
			
			// Allocate data
			Pointer x = ((JCudaObject)image.getGPUObject()).jcudaDenseMatrixPtr; 
			Pointer y = ((JCudaObject)outputBlock.getGPUObject()).jcudaDenseMatrixPtr; 
			
			alpha = pointerTo(1.0);
			beta = pointerTo(0.0f);
			
			int status = cudnnPoolingForward(cudnnHandle, poolingDesc, alpha, xDesc, x, beta, yDesc, y);
			
			if(status != jcuda.jcudnn.cudnnStatus.CUDNN_STATUS_SUCCESS) {
				throw new DMLRuntimeException("Could not executed cudnnPoolingForward: " + jcuda.jcudnn.cudnnStatus.stringFor(status));
			}
		}
		finally {
			if(alpha != null)
				cudaFree(alpha);
			if(beta != null)
				cudaFree(beta);
			if(yDesc != null)
				cudnnDestroyTensorDescriptor(yDesc);
			if(xDesc != null)
				cudnnDestroyTensorDescriptor(xDesc);
			if(poolingDesc != null)
				cudnnDestroyPoolingDescriptor(poolingDesc);
		}
	}
	
	/**
	 * performs maxpoolingBackward on GPU by exploiting cudnnPoolingBackward(...)
	 * @param image
	 * @param dout			delta matrix, output of previous layer
	 * @param outputBlock
	 * @param N				batch size
	 * @param C				number of channels
	 * @param H				height of image
	 * @param W				width of image
	 * @param K				number of filters
	 * @param R				height of filter
	 * @param S				width of filter
	 * @param pad_h			vertical padding
	 * @param pad_w			horizontal padding
	 * @param stride_h		horizontal stride
	 * @param stride_w		vertical stride
	 * @param P				(H - R + 1 + 2*pad_h)/stride_h
	 * @param Q				(W - S + 1 + 2*pad_w)/stride_w
	 * @throws DMLRuntimeException
	 */
	public static void maxpooling_backward(MatrixObject image, MatrixObject dout,
			MatrixObject outputBlock, int N, int C, int H, int W, int K, int R,
			int S, int pad_h, int pad_w, int stride_h, int stride_w, int P,
			int Q) throws DMLRuntimeException {
		Pointer alpha = null;
		Pointer beta = null;
		cudnnTensorDescriptor xDesc = null;
		cudnnTensorDescriptor yDesc = null;
		cudnnTensorDescriptor dyDesc = null;
		cudnnTensorDescriptor dxDesc = null;
		cudnnPoolingDescriptor poolingDesc = null;

		try {
			// Allocate descriptors
			xDesc = allocateTensorDescriptor(N, C, H, W);
			yDesc = allocateTensorDescriptor(N, C, P, Q);
			dxDesc = allocateTensorDescriptor(N, C, H, W);
			dyDesc = allocateTensorDescriptor(N, C, P, Q);
			
			poolingDesc = allocatePoolingDescriptor(R, S, pad_h, pad_w, stride_h, stride_w);
			
			// Calling PoolForward first, y is one of the inputs for poolBackward
			// TODO: Remove calling poolForward after necessary changes at language level for poolBackward
			Pointer y = new Pointer();
			long numBytes = N*C*P*Q*Sizeof.DOUBLE;
			cudaMalloc(y, numBytes);
			
			// Allocate data
			Pointer x = ((JCudaObject)image.getGPUObject()).jcudaDenseMatrixPtr;
			Pointer dx = ((JCudaObject)outputBlock.getGPUObject()).jcudaDenseMatrixPtr;
			Pointer dy = ((JCudaObject)dout.getGPUObject()).jcudaDenseMatrixPtr;
			
			alpha = pointerTo(1.0);
			beta = pointerTo(0.0f);
			
			int status = cudnnPoolingForward(cudnnHandle, poolingDesc, alpha, xDesc, x, beta, yDesc, y);
			if(status != jcuda.jcudnn.cudnnStatus.CUDNN_STATUS_SUCCESS) {
				throw new DMLRuntimeException("Could not executed cudnnPoolingForward before cudnnPoolingBackward: " + jcuda.jcudnn.cudnnStatus.stringFor(status));
			}
			
			status = cudnnPoolingBackward(cudnnHandle, poolingDesc, alpha, yDesc, y, dyDesc, dy, xDesc, x, beta, dxDesc, dx);
			
			if(status != jcuda.jcudnn.cudnnStatus.CUDNN_STATUS_SUCCESS) {
				throw new DMLRuntimeException("Could not executed cudnnPoolingBackward: " + jcuda.jcudnn.cudnnStatus.stringFor(status));
			}
			
			cudaFree(y);
		}
		finally {
			if(alpha != null)
				cudaFree(alpha);
			if(beta != null)
				cudaFree(beta);
			if(yDesc != null)
				cudnnDestroyTensorDescriptor(yDesc);
			if(xDesc != null)
				cudnnDestroyTensorDescriptor(xDesc);
			if(dyDesc != null)
				cudnnDestroyTensorDescriptor(dyDesc);
			if(dxDesc != null)
				cudnnDestroyTensorDescriptor(dxDesc);
			if(poolingDesc != null)
				cudnnDestroyPoolingDescriptor(poolingDesc);	
		}	
	}
	public static boolean isInSparseFormat(MatrixObject mo) {
		if(mo.getGPUObject() != null && mo.getGPUObject().isAllocated())
			return mo.getGPUObject().isInSparseFormat();
		return MatrixBlock.evalSparseFormatInMemory(mo.getNumRows(), mo.getNumColumns(), mo.getNnz());
	}
}
