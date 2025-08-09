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

import jcuda.Pointer;
import jcuda.jcudnn.cudnnConvolutionBwdDataAlgo;
import jcuda.jcudnn.cudnnConvolutionBwdDataAlgoPerf;
import jcuda.jcudnn.cudnnConvolutionBwdFilterAlgo;
import jcuda.jcudnn.cudnnConvolutionBwdFilterAlgoPerf;
import jcuda.jcudnn.cudnnConvolutionDescriptor;
import jcuda.jcudnn.cudnnConvolutionFwdAlgo;
import jcuda.jcudnn.cudnnFilterDescriptor;
import jcuda.jcudnn.cudnnTensorDescriptor;

import static jcuda.jcudnn.JCudnn.cudnnCreateConvolutionDescriptor;
import static jcuda.jcudnn.JCudnn.cudnnCreateFilterDescriptor;
import static jcuda.jcudnn.JCudnn.cudnnCreateTensorDescriptor;
import static jcuda.jcudnn.JCudnn.cudnnDestroyConvolutionDescriptor;
import static jcuda.jcudnn.JCudnn.cudnnDestroyFilterDescriptor;
import static jcuda.jcudnn.JCudnn.cudnnDestroyTensorDescriptor;
import static jcuda.jcudnn.JCudnn.cudnnSetConvolution2dDescriptor;
import static jcuda.jcudnn.JCudnn.cudnnSetFilter4dDescriptor;
import static jcuda.jcudnn.JCudnn.cudnnSetTensor4dDescriptor;
import static jcuda.jcudnn.cudnnConvolutionMode.CUDNN_CROSS_CORRELATION;
import static jcuda.jcudnn.cudnnTensorFormat.CUDNN_TENSOR_NCHW;

import org.apache.sysds.api.DMLScript;
import org.apache.sysds.runtime.DMLRuntimeException;
import org.apache.sysds.runtime.instructions.gpu.context.GPUContext;

/**
 * This class is a wrapper that contain necessary data structures to invoke 
 * a cudnn convolution* functions (such as cudnnConvolutionForward, etc)
 * 
 * It implements autocloseable to simplify the LibMatrixCuDNN code and also avoids potential memory leaks.
 * 
 * The caller has to use the factory methods: cudnnGetConvolutionForwardAlgorithm, 
 * cudnnGetConvolutionBackwardFilterAlgorithm and cudnnGetConvolutionBackwardDataAlgorithm
 * to get the LibMatrixCuDNNConvolutionAlgorithm object.
 * The naming of this methods is consistent with that of CuDNN library.
 *  
 */
public class LibMatrixCuDNNConvolutionAlgorithm implements java.lang.AutoCloseable {
	// Limit the workspace available to cudnn convolution operation to 1 GB
	static long MAX_WORKSPACE_LIMIT_BYTES = (long) 1e+9;
	
	public int algo = -1;
	public Pointer workSpace = new Pointer();
	public long sizeInBytes = 0;
	cudnnTensorDescriptor nchwTensorDesc = null;
	cudnnTensorDescriptor nkpqTensorDesc = null;
	cudnnFilterDescriptor filterDesc = null;
	cudnnConvolutionDescriptor convDesc = null;
	GPUContext gCtx = null; String instName = null;
	
	private LibMatrixCuDNNConvolutionAlgorithm(GPUContext gCtx, String instName, int N, int C, int H, int W, int K, int R, int S, 
			int pad_h, int pad_w, int stride_h, int stride_w, int P, int Q) {
		int padding[] = {pad_h, pad_w};
		int strides[] = {stride_h, stride_w};
		convDesc = allocateConvolutionDescriptor(padding, strides);
		this.gCtx = gCtx;
		this.instName = instName;
		nchwTensorDesc = allocateTensorDescriptor(N, C, H, W);
		nkpqTensorDesc = allocateTensorDescriptor(N, K, P, Q);
		filterDesc = allocateFilterDescriptor(K, C, R, S);
	}
	
	/**
	 * Deallocates the tensor and filter descriptors as well as allocated workspace
	 */
	@SuppressWarnings("deprecation")
	@Override
	public void close() {
		if(nchwTensorDesc != null)
			cudnnDestroyTensorDescriptor(nchwTensorDesc);
		if(nkpqTensorDesc != null)
			cudnnDestroyTensorDescriptor(nkpqTensorDesc);
		if(filterDesc != null)
			cudnnDestroyFilterDescriptor(filterDesc);
		if(convDesc != null)
			cudnnDestroyConvolutionDescriptor(convDesc);
		if(sizeInBytes != 0) {
			try {
				gCtx.cudaFreeHelper(instName, workSpace, DMLScript.EAGER_CUDA_FREE);
			} catch (DMLRuntimeException e) {
				throw new RuntimeException(e);
			}
		}
	}
	
	/**
	 * Factory method to get the algorithm wrapper for convolution forward
	 * 
	 * @param gCtx     a valid {@link GPUContext}
	 * @param instName the invoking instruction's name for record {@link org.apache.sysds.utils.Statistics}.
	 * @param N        number of input images
	 * @param C        number of channels
	 * @param H        height of each image
	 * @param W        width of each image
	 * @param K        number of output "channels"
	 * @param R        height of filter
	 * @param S        width of filter
	 * @param pad_h    padding height
	 * @param pad_w    padding width
	 * @param stride_h stride height
	 * @param stride_w string width
	 * @param P        output height
	 * @param Q        output width
	 * @param workspaceLimit maximum intermediate memory to use
	 * @return algorithm wrapper
	 */
	@SuppressWarnings("deprecation")
	public static LibMatrixCuDNNConvolutionAlgorithm cudnnGetConvolutionForwardAlgorithm(
			GPUContext gCtx, String instName, int N, int C, int H, int W, int K, int R, int S, 
			int pad_h, int pad_w, int stride_h, int stride_w, int P, int Q, long workspaceLimit) {
		LibMatrixCuDNNConvolutionAlgorithm ret = new LibMatrixCuDNNConvolutionAlgorithm(gCtx, instName, N, C, H, W, K, R, S, 
				pad_h, pad_w, stride_h, stride_w, P, Q);
		//int[] algos = {-1};
		long sizeInBytesArray[] = {Math.min(workspaceLimit, MAX_WORKSPACE_LIMIT_BYTES)};
		/*jcuda.jcudnn.JCudnn.cudnnGetConvolutionForwardAlgorithm(LibMatrixCuDNN.getCudnnHandle(gCtx),
				ret.nchwTensorDesc, ret.filterDesc, ret.convDesc, ret.nkpqTensorDesc,
				cudnnConvolutionFwdPreference.CUDNN_CONVOLUTION_FWD_SPECIFY_WORKSPACE_LIMIT, sizeInBytesArray[0], algos);*/
		// FIXME: cudnnGetConvolutionForwardAlgorithm method returns CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM
		// 	as the best suited conv2d algo for these inputs, however applying that algo returns zero values.
		ret.algo = cudnnConvolutionFwdAlgo.CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM; //force this algo
		jcuda.jcudnn.JCudnn.cudnnGetConvolutionForwardWorkspaceSize(LibMatrixCuDNN.getCudnnHandle(gCtx),
			ret.nchwTensorDesc, ret.filterDesc, ret.convDesc, ret.nkpqTensorDesc, ret.algo, sizeInBytesArray);
		if (sizeInBytesArray[0] != 0)
			ret.workSpace = gCtx.allocate(instName, sizeInBytesArray[0], false);
		ret.sizeInBytes = sizeInBytesArray[0];
		//ret.algo = algos[0];
		return ret;
	}
	
	/**
	 * Factory method to get the algorithm wrapper for convolution backward filter
	 * 
	 * @param gCtx     a valid {@link GPUContext}
	 * @param instName the invoking instruction's name for record {@link org.apache.sysds.utils.Statistics}.
	 * @param N        number of input images
	 * @param C        number of channels
	 * @param H        height of each image
	 * @param W        width of each image
	 * @param K        number of output "channels"
	 * @param R        height of filter
	 * @param S        width of filter
	 * @param pad_h    padding height
	 * @param pad_w    padding width
	 * @param stride_h stride height
	 * @param stride_w string width
	 * @param P        output height
	 * @param Q        output width
	 * @param workspaceLimit maximum intermediate memory to use
	 * @return algorithm wrapper
	 */
	@SuppressWarnings("deprecation")
	public static LibMatrixCuDNNConvolutionAlgorithm cudnnGetConvolutionBackwardFilterAlgorithm(
			GPUContext gCtx, String instName, int N, int C, int H, int W, int K, int R, int S, 
			int pad_h, int pad_w, int stride_h, int stride_w, int P, int Q, long workspaceLimit) {
		LibMatrixCuDNNConvolutionAlgorithm ret = new LibMatrixCuDNNConvolutionAlgorithm(gCtx, instName, N, C, H, W, K,
			R, S, pad_h, pad_w, stride_h, stride_w, P, Q);

		final int maxAlgos = cudnnConvolutionBwdFilterAlgo.CUDNN_CONVOLUTION_BWD_FILTER_ALGO_COUNT;
		cudnnConvolutionBwdFilterAlgoPerf[] perf = new cudnnConvolutionBwdFilterAlgoPerf[maxAlgos];
		for(int i = 0; i < maxAlgos; ++i)
			perf[i] = new cudnnConvolutionBwdFilterAlgoPerf();
		int[] returnedAlgoCount = {0};
		jcuda.jcudnn.JCudnn.cudnnGetConvolutionBackwardFilterAlgorithm_v7(LibMatrixCuDNN.getCudnnHandle(gCtx),
			ret.nchwTensorDesc, ret.nkpqTensorDesc, ret.convDesc, ret.filterDesc, maxAlgos, returnedAlgoCount, perf);

		long workspaceCap = Math.min(workspaceLimit, MAX_WORKSPACE_LIMIT_BYTES);
		int chosenAlgo = cudnnConvolutionBwdFilterAlgo.CUDNN_CONVOLUTION_BWD_FILTER_ALGO_0;
		long chosenWs = 0;

		for(int i = 0; i < returnedAlgoCount[0]; ++i) {
			if(perf[i].memory <= workspaceCap) {
				chosenAlgo = perf[i].algo;
				chosenWs = perf[i].memory;
				break;
			}
		}

		long[] sizeInBytesArray = {chosenWs};

		jcuda.jcudnn.JCudnn.cudnnGetConvolutionBackwardFilterWorkspaceSize(LibMatrixCuDNN.getCudnnHandle(gCtx),
			ret.nchwTensorDesc, ret.nkpqTensorDesc, ret.convDesc, ret.filterDesc, chosenAlgo, sizeInBytesArray);
		if(sizeInBytesArray[0] != 0)
			ret.workSpace = gCtx.allocate(instName, sizeInBytesArray[0], false);
		ret.sizeInBytes = sizeInBytesArray[0];
		ret.algo = chosenAlgo;

		return ret;
	}
	
	/**
	 * Factory method to get the algorithm wrapper for convolution backward data
	 * 
	 * @param gCtx     a valid {@link GPUContext}
	 * @param instName the invoking instruction's name for record {@link org.apache.sysds.utils.Statistics}.
	 * @param N        number of input images
	 * @param C        number of channels
	 * @param H        height of each image
	 * @param W        width of each image
	 * @param K        number of output "channels"
	 * @param R        height of filter
	 * @param S        width of filter
	 * @param pad_h    padding height
	 * @param pad_w    padding width
	 * @param stride_h stride height
	 * @param stride_w string width
	 * @param P        output height
	 * @param Q        output width
	 * @param workspaceLimit maximum intermediate memory to use
	 * @return algorithm wrapper
	 */
	@SuppressWarnings("deprecation")
	public static LibMatrixCuDNNConvolutionAlgorithm cudnnGetConvolutionBackwardDataAlgorithm(
			GPUContext gCtx, String instName, int N, int C, int H, int W, int K, int R, int S, 
			int pad_h, int pad_w, int stride_h, int stride_w, int P, int Q, long workspaceLimit) {
		LibMatrixCuDNNConvolutionAlgorithm ret = new LibMatrixCuDNNConvolutionAlgorithm(gCtx, instName, N, C, H, W, K, R, S, 
				pad_h, pad_w, stride_h, stride_w, P, Q);
		if(H == R || W == S) {
			// CuDNN's cudnnGetConvolutionBackwardDataAlgorithm returns CUDNN_CONVOLUTION_BWD_DATA_ALGO_1 for atleast one scenario 
			// for sentence CNN (N=1, C=1, H=2060, W=300, F=500, Hf=5, Wf=300, sparsity=0.1).
			// This causes more than 100x slowdown when compared with CUDNN_CONVOLUTION_BWD_DATA_ALGO_0.
			// To keep things simple for now, we will always prefer to use memory-less operator for conv1d: CUDNN_CONVOLUTION_BWD_DATA_ALGO_0
			ret.algo = jcuda.jcudnn.cudnnConvolutionBwdDataAlgo.CUDNN_CONVOLUTION_BWD_DATA_ALGO_0;
		}
		else {
			final int max = cudnnConvolutionBwdDataAlgo.CUDNN_CONVOLUTION_BWD_DATA_ALGO_COUNT;
			cudnnConvolutionBwdDataAlgoPerf[] perf = new cudnnConvolutionBwdDataAlgoPerf[max];
			for(int i = 0; i < max; ++i)
				perf[i] = new cudnnConvolutionBwdDataAlgoPerf();
			int[] nReturned = {0};

			jcuda.jcudnn.JCudnn.cudnnGetConvolutionBackwardDataAlgorithm_v7(LibMatrixCuDNN.getCudnnHandle(gCtx),
				ret.filterDesc, ret.nkpqTensorDesc, ret.convDesc, ret.nchwTensorDesc, max, nReturned, perf);

			long cap = Math.min(workspaceLimit, MAX_WORKSPACE_LIMIT_BYTES);
			int chosenAlgo = cudnnConvolutionBwdDataAlgo.CUDNN_CONVOLUTION_BWD_DATA_ALGO_0;
			long chosenWs = 0;

			for(int i = 0; i < nReturned[0]; ++i) {
				if(perf[i].memory <= cap) {
					chosenAlgo = perf[i].algo;
					chosenWs = perf[i].memory;
					break;
				}
			}

			long[] sizeInBytesArray = {chosenWs};
			jcuda.jcudnn.JCudnn.cudnnGetConvolutionBackwardDataWorkspaceSize(LibMatrixCuDNN.getCudnnHandle(gCtx),
				ret.filterDesc, ret.nkpqTensorDesc, ret.convDesc, ret.nchwTensorDesc, chosenAlgo, sizeInBytesArray);

			if(sizeInBytesArray[0] != 0)
				ret.workSpace = gCtx.allocate(instName, sizeInBytesArray[0], false);

			ret.sizeInBytes = sizeInBytesArray[0];
			ret.algo = chosenAlgo;
		}
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
	
	@SuppressWarnings("deprecation")
	private static cudnnFilterDescriptor allocateFilterDescriptor(int K, int C, int R, int S) {
		cudnnFilterDescriptor filterDesc = new cudnnFilterDescriptor();
		cudnnCreateFilterDescriptor(filterDesc);
		cudnnSetFilter4dDescriptor(filterDesc, LibMatrixCUDA.CUDNN_DATA_TYPE, CUDNN_TENSOR_NCHW, K, C, R, S);
		return filterDesc;
	}
	
	@SuppressWarnings("deprecation")
	private static cudnnConvolutionDescriptor allocateConvolutionDescriptor(int padding [], int strides []) {
		cudnnConvolutionDescriptor convDesc = new cudnnConvolutionDescriptor();
		cudnnCreateConvolutionDescriptor(convDesc);
		cudnnSetConvolution2dDescriptor(convDesc, padding[0], padding[1], strides[0], strides[1], 1, 1, CUDNN_CROSS_CORRELATION, LibMatrixCUDA.CUDNN_DATA_TYPE);
		return convDesc;
	}
}
