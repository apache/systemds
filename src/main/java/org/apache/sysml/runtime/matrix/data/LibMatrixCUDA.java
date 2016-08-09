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
import static jcuda.jcudnn.JCudnn.cudnnPoolingForward;
import static jcuda.jcudnn.JCudnn.cudnnPoolingBackward;
import static jcuda.jcudnn.JCudnn.cudnnCreateConvolutionDescriptor;
import static jcuda.jcudnn.JCudnn.cudnnCreateFilterDescriptor;
import static jcuda.jcudnn.JCudnn.cudnnCreateTensorDescriptor;
import static jcuda.jcudnn.JCudnn.cudnnCreatePoolingDescriptor;
import static jcuda.jcudnn.JCudnn.cudnnDestroyConvolutionDescriptor;
import static jcuda.jcudnn.JCudnn.cudnnDestroyFilterDescriptor;
import static jcuda.jcudnn.JCudnn.cudnnDestroyTensorDescriptor;
import static jcuda.jcudnn.JCudnn.cudnnDestroyPoolingDescriptor;
import static jcuda.jcudnn.JCudnn.cudnnGetConvolutionBackwardDataWorkspaceSize;
import static jcuda.jcudnn.JCudnn.cudnnGetConvolutionBackwardFilterWorkspaceSize;
import static jcuda.jcudnn.JCudnn.cudnnGetConvolutionForwardWorkspaceSize;
import static jcuda.jcudnn.JCudnn.cudnnSetConvolution2dDescriptor;
import static jcuda.jcudnn.JCudnn.cudnnSetFilter4dDescriptor;
import static jcuda.jcudnn.JCudnn.cudnnSetTensor4dDescriptor;
import static jcuda.jcudnn.JCudnn.cudnnSetPooling2dDescriptor;
import static jcuda.jcudnn.cudnnConvolutionMode.CUDNN_CROSS_CORRELATION;
import static jcuda.jcudnn.cudnnDataType.CUDNN_DATA_DOUBLE;
import static jcuda.jcudnn.cudnnTensorFormat.CUDNN_TENSOR_NCHW;
import static jcuda.jcudnn.cudnnPoolingMode.CUDNN_POOLING_MAX;
import jcuda.jcudnn.cudnnConvolutionFwdPreference;
import static jcuda.runtime.JCuda.cudaMalloc;
import static jcuda.runtime.JCuda.cudaFree;
import jcuda.Pointer;
import jcuda.Sizeof;
import jcuda.jcublas.JCublas;
import jcuda.jcublas.JCublas2;
import jcuda.jcublas.cublasHandle;
import jcuda.jcudnn.cudnnConvolutionDescriptor;
import jcuda.jcudnn.cudnnFilterDescriptor;
import jcuda.jcudnn.cudnnHandle;
import jcuda.jcudnn.cudnnPoolingDescriptor;
import jcuda.jcudnn.cudnnTensorDescriptor;

import org.apache.sysml.runtime.DMLRuntimeException;
import org.apache.sysml.runtime.controlprogram.caching.MatrixObject;
import org.apache.sysml.runtime.instructions.gpu.context.JCudaObject;

//FIXME move could to respective instructions, this is not a block library
public class LibMatrixCUDA {
	
	public static cudnnHandle cudnnHandle;
	public static cublasHandle cublasHandle;
	
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
			
			Pointer imagePointer = ((JCudaObject)image.getGPUObject()).jcudaPointer; 
			Pointer filterPointer = ((JCudaObject)filter.getGPUObject()).jcudaPointer; 
			Pointer dstPointer = ((JCudaObject)outputBlock.getGPUObject()).jcudaPointer; 
			
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
			Pointer imagePointer = ((JCudaObject)image.getGPUObject()).jcudaPointer; 
			Pointer doutPointer = ((JCudaObject)dout.getGPUObject()).jcudaPointer; 
			Pointer dwPointer = ((JCudaObject)outputBlock.getGPUObject()).jcudaPointer; 
			
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

	public static void matmultTSMM(MatrixObject left, MatrixObject output,
            boolean isLeftTransposed) throws DMLRuntimeException {
	    if(isInSparseFormat(left)) {
	            throw new DMLRuntimeException("Sparse GPU TSMM is not implemented");
	    }
	
	    // Since CuBLAS expects inputs in column-major format,
	    // reverse the order of matrix-multiplication and take care of dimension mismatch.      
	    char transa = isLeftTransposed ? 'N' : 'T';
	    // Note: the dimensions are swapped
	    int m = (int) (isLeftTransposed ? left.getNumColumns() : left.getNumRows());
	    int k = (int) (isLeftTransposed ? left.getNumRows() : left.getNumColumns());
	
	    if(m == -1)
	            throw new DMLRuntimeException("Incorrect dimensions");
	
	    double alpha = 1.0d;
	    double beta = 0.0d;
	
	    int lda = (int) (isLeftTransposed ? m : k);
	    int ldc = m;
	
	    if(!left.getGPUObject().isAllocated)
	            throw new DMLRuntimeException("Input is not allocated:" + left.getGPUObject().isAllocated);
	    if(!output.getGPUObject().isAllocated)
	            throw new DMLRuntimeException("Output is not allocated:" + output.getGPUObject().isAllocated);
	
	    Pointer A = ((JCudaObject)left.getGPUObject()).jcudaPointer;
	    Pointer C = ((JCudaObject)output.getGPUObject()).jcudaPointer;
	    
	    //TODO: Fix it if there is a cuBLAS API to do flipping
	    JCublas.cublasDsyrk('U',transa, m, k, alpha, A, lda, beta, C, ldc);
	    JCublas.cublasDsyrk('L',transa, m, k, alpha, A, lda, beta, C, ldc);
	}
	
	public static void matmult(MatrixObject left1, MatrixObject right1, MatrixObject output, 
			boolean isLeftTransposed1, boolean isRightTransposed1) throws DMLRuntimeException {
		if(isInSparseFormat(left1) || isInSparseFormat(right1)) {
			throw new DMLRuntimeException("Sparse GPU matrix multiplication is not implemented");
		}
		
		// Since CuBLAS expects inputs in column-major format,
		// reverse the order of matrix-multiplication and take care of dimension mismatch.
		MatrixObject left = right1; 
		MatrixObject right = left1;
		boolean isLeftTransposed = isRightTransposed1; 
		boolean isRightTransposed = isLeftTransposed1; 
		
		char transa = isLeftTransposed ? 'T' : 'N';
		char transb = isRightTransposed ? 'T' : 'N';
		// Note: the dimensions are swapped
		int m = (int) (isLeftTransposed ? left.getNumRows() : left.getNumColumns()) ;
		int n = (int) (isRightTransposed ? right.getNumColumns() : right.getNumRows());
		int k = (int) (isLeftTransposed ?  left.getNumColumns() : left.getNumRows());
		int k1 = (int) (isRightTransposed ?  right.getNumRows() : right.getNumColumns());
		if(k != k1) 
			throw new DMLRuntimeException("Dimension mismatch: " + k + " != " + k1);
		
		if(m == -1 || n == -1 || k == -1)
			throw new DMLRuntimeException("Incorrect dimensions");
		
		double alpha = 1;
		double beta = 0;
		
		int lda = isLeftTransposed ?  k : m;
		int ldb = isRightTransposed ? n : k;
		int ldc = m;
		
		if(!left.getGPUObject().isAllocated() || !right.getGPUObject().isAllocated())
			throw new DMLRuntimeException("One of input is not allocated:" + left.getGPUObject().isAllocated() + " " + right.getGPUObject().isAllocated());
		if(!output.getGPUObject().isAllocated())
			throw new DMLRuntimeException("Output is not allocated:" + output.getGPUObject().isAllocated());
		
		Pointer A = ((JCudaObject)left.getGPUObject()).jcudaPointer;
		Pointer B = ((JCudaObject)right.getGPUObject()).jcudaPointer;
		Pointer C = ((JCudaObject)output.getGPUObject()).jcudaPointer;
		
		JCublas.cublasDgemm(transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
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
			Pointer w = ((JCudaObject)filter.getGPUObject()).jcudaPointer; 
			Pointer dy = ((JCudaObject)dout.getGPUObject()).jcudaPointer; 
			Pointer dx = ((JCudaObject)output.getGPUObject()).jcudaPointer; 
			
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
			Pointer x = ((JCudaObject)image.getGPUObject()).jcudaPointer; 
			Pointer y = ((JCudaObject)outputBlock.getGPUObject()).jcudaPointer; 
			
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
			Pointer x = ((JCudaObject)image.getGPUObject()).jcudaPointer; 
			Pointer dx = ((JCudaObject)outputBlock.getGPUObject()).jcudaPointer;
			Pointer dy = ((JCudaObject)dout.getGPUObject()).jcudaPointer;
			
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
