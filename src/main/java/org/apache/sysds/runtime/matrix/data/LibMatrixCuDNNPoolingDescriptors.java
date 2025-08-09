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

import static jcuda.jcudnn.JCudnn.cudnnCreatePoolingDescriptor;
import static jcuda.jcudnn.JCudnn.cudnnCreateTensorDescriptor;
import static jcuda.jcudnn.JCudnn.cudnnDestroyTensorDescriptor;
import static jcuda.jcudnn.JCudnn.cudnnSetPooling2dDescriptor;
import static jcuda.jcudnn.JCudnn.cudnnSetTensor4dDescriptor;
import static jcuda.jcudnn.cudnnNanPropagation.CUDNN_PROPAGATE_NAN;
import static jcuda.jcudnn.cudnnPoolingMode.CUDNN_POOLING_MAX;
import static jcuda.jcudnn.cudnnPoolingMode.CUDNN_POOLING_AVERAGE_COUNT_INCLUDE_PADDING;
import static jcuda.jcudnn.cudnnTensorFormat.CUDNN_TENSOR_NCHW;

import org.apache.sysds.runtime.instructions.gpu.context.GPUContext;
import org.apache.sysds.runtime.matrix.data.LibMatrixDNN.PoolingType;

import jcuda.jcudnn.cudnnPoolingDescriptor;
import jcuda.jcudnn.cudnnTensorDescriptor;

/**
 * This class is a wrapper that contain necessary data structures to invoke 
 * a cudnn convolution* functions (such as cudnnConvolutionForward, etc)
 * 
 * It implements autocloseable to simplify the LibMatrixCuDNN code and also avoids potential memory leaks.
 */
public class LibMatrixCuDNNPoolingDescriptors implements java.lang.AutoCloseable {

	public cudnnTensorDescriptor xDesc; 
	public cudnnTensorDescriptor yDesc; 
	public cudnnTensorDescriptor dxDesc; 
	public cudnnTensorDescriptor dyDesc; 
	public cudnnPoolingDescriptor poolingDesc;
	
	@SuppressWarnings("deprecation")
	@Override
	public void close() {
		if(xDesc != null) 
			cudnnDestroyTensorDescriptor(xDesc);
		if(yDesc != null) 
			cudnnDestroyTensorDescriptor(yDesc);
		if(dxDesc != null) 
			cudnnDestroyTensorDescriptor(dxDesc);
		if(dyDesc != null) 
			cudnnDestroyTensorDescriptor(dyDesc);
		if(poolingDesc != null)
			jcuda.jcudnn.JCudnn.cudnnDestroyPoolingDescriptor(poolingDesc);
	}
	
	/**
	 * Get descriptors for maxpooling backward operation
	 * 
	 * @param gCtx gpu context
	 * @param instName instruction name
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
	 * @param poolingType	type of pooling
	 * @return decriptor wrapper
	 */
	public static LibMatrixCuDNNPoolingDescriptors cudnnPoolingBackwardDescriptors(GPUContext gCtx, 
			String instName, int N, int C, int H, int W, int K, int R,
			int S, int pad_h, int pad_w, int stride_h, int stride_w, int P,
			int Q, PoolingType poolingType) {
		LibMatrixCuDNNPoolingDescriptors ret = new LibMatrixCuDNNPoolingDescriptors();
		ret.xDesc = allocateTensorDescriptor(N, C, H, W);
		ret.yDesc = allocateTensorDescriptor(N, C, P, Q);
		ret.dxDesc = allocateTensorDescriptor(N, C, H, W);
		ret.dyDesc = allocateTensorDescriptor(N, C, P, Q);
		ret.poolingDesc = allocatePoolingDescriptor(R, S, pad_h, pad_w, stride_h, stride_w, poolingType);
		return ret;
	}
	
	/**
	 * Get descriptors for maxpooling operation
	 * 
	 * @param gCtx gpu context
	 * @param instName instruction name
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
	 * @param poolingType 	type of pooling
	 * @return decriptor wrapper
	 */
	public static LibMatrixCuDNNPoolingDescriptors cudnnPoolingDescriptors(GPUContext gCtx, 
			String instName, int N, int C, int H, int W, int K, int R,
			int S, int pad_h, int pad_w, int stride_h, int stride_w, int P,
			int Q, PoolingType poolingType) {
		LibMatrixCuDNNPoolingDescriptors ret = new LibMatrixCuDNNPoolingDescriptors();
		ret.xDesc = allocateTensorDescriptor(N, C, H, W);
		ret.yDesc = allocateTensorDescriptor(N, C, P, Q);
		ret.poolingDesc = allocatePoolingDescriptor(R, S, pad_h, pad_w, stride_h, stride_w, poolingType);
		return ret;
	}

	/**
	 * Convenience method to get tensor descriptor
	 * @param N number of images
	 * @param C number of channels
	 * @param H height
	 * @param W width
	 * @return cudnn tensor descriptor
	 */
	private static cudnnTensorDescriptor allocateTensorDescriptor(int N, int C, int H, int W) {
		cudnnTensorDescriptor tensorDescriptor = new cudnnTensorDescriptor();
		cudnnCreateTensorDescriptor(tensorDescriptor);
		cudnnSetTensor4dDescriptor(tensorDescriptor, CUDNN_TENSOR_NCHW, LibMatrixCUDA.CUDNN_DATA_TYPE, N, C, H, W);
		return tensorDescriptor;
	}
	
	/**
	 * allocates pooling descriptor, used in poolingForward and poolingBackward
	 * @param R			pooling window height
	 * @param S			pooling window width
	 * @param pad_h		vertical padding
	 * @param pad_w		horizontal padding
	 * @param stride_h	pooling vertical stride
	 * @param stride_w	pooling horizontal stride
	 * @param poolingType type of pooling
	 * @return cudnn pooling descriptor
	 */
	@SuppressWarnings("deprecation")
	private static cudnnPoolingDescriptor allocatePoolingDescriptor(int R, int S, int pad_h, int pad_w, int stride_h, int stride_w, PoolingType poolingType) {
		cudnnPoolingDescriptor poolingDesc = new cudnnPoolingDescriptor();
		cudnnCreatePoolingDescriptor(poolingDesc);
		int CUDNN_POOLING = (poolingType == PoolingType.MAX) ? CUDNN_POOLING_MAX : CUDNN_POOLING_AVERAGE_COUNT_INCLUDE_PADDING;
		cudnnSetPooling2dDescriptor(poolingDesc, CUDNN_POOLING, CUDNN_PROPAGATE_NAN, R, S, pad_h, pad_w, stride_h, stride_w);
		return poolingDesc;
	}
}
