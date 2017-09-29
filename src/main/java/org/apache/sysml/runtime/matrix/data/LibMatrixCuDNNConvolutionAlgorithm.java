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

import org.apache.sysml.runtime.DMLRuntimeException;
import org.apache.sysml.runtime.instructions.gpu.GPUInstruction;
import org.apache.sysml.runtime.instructions.gpu.context.GPUContext;
import org.apache.sysml.utils.GPUStatistics;

import jcuda.Pointer;
import jcuda.jcudnn.cudnnConvolutionBwdDataPreference;
import jcuda.jcudnn.cudnnConvolutionBwdFilterPreference;
import jcuda.jcudnn.cudnnConvolutionDescriptor;
import jcuda.jcudnn.cudnnConvolutionFwdPreference;
import jcuda.jcudnn.cudnnFilterDescriptor;
import jcuda.jcudnn.cudnnTensorDescriptor;
import static jcuda.jcudnn.JCudnn.cudnnDestroyConvolutionDescriptor;
import static jcuda.jcudnn.JCudnn.cudnnDestroyFilterDescriptor;
import static jcuda.jcudnn.JCudnn.cudnnDestroyTensorDescriptor;

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
	public int algo = -1;
	public Pointer workSpace = new Pointer();
	public long sizeInBytes = 0;
	cudnnTensorDescriptor nchwTensorDesc = null;
	cudnnTensorDescriptor nkpqTensorDesc = null;
	cudnnFilterDescriptor filterDesc = null;
	cudnnConvolutionDescriptor convDesc = null;
	GPUContext gCtx = null; String instName = null;
	
	private LibMatrixCuDNNConvolutionAlgorithm(GPUContext gCtx, String instName, int N, int C, int H, int W, int K, int R, int S, 
			int pad_h, int pad_w, int stride_h, int stride_w, int P, int Q) throws DMLRuntimeException {
		int padding[] = {pad_h, pad_w};
		int strides[] = {stride_h, stride_w};
		convDesc = LibMatrixCuDNN.allocateConvolutionDescriptor(padding, strides);
		this.gCtx = gCtx;
		this.instName = instName;
		nchwTensorDesc = LibMatrixCuDNN.allocateTensorDescriptor(N, C, H, W);
		nkpqTensorDesc = LibMatrixCuDNN.allocateTensorDescriptor(N, K, P, Q);
		filterDesc = LibMatrixCuDNN.allocateFilterDescriptor(K, C, R, S);
	}
	
	/**
	 * Deallocates the tensor and filter descriptors as well as allocated workspace
	 */
	@Override
	public void close() {
		long t3 = 0;
		if (GPUStatistics.DISPLAY_STATISTICS) t3 = System.nanoTime();
		if(nchwTensorDesc != null)
			cudnnDestroyTensorDescriptor(nchwTensorDesc);
		if(nkpqTensorDesc != null)
			cudnnDestroyTensorDescriptor(nkpqTensorDesc);
		if(filterDesc != null)
			cudnnDestroyFilterDescriptor(filterDesc);
		if(convDesc != null)
			cudnnDestroyConvolutionDescriptor(convDesc);
		if(sizeInBytes != 0)
			gCtx.cudaFreeHelper(instName, workSpace);
		if(GPUStatistics.DISPLAY_STATISTICS)
			GPUStatistics.maintainCPMiscTimes(instName, GPUInstruction.MISC_TIMER_CUDNN_CLEANUP, System.nanoTime() - t3);
	}
	
	/**
	 * Factory method to get the algorithm wrapper for convolution forward
	 * 
	 * @param gCtx     a valid {@link GPUContext}
	 * @param instName the invoking instruction's name for record {@link org.apache.sysml.utils.Statistics}.
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
	 * @throws DMLRuntimeException if error occurs
	 */
	public static LibMatrixCuDNNConvolutionAlgorithm cudnnGetConvolutionForwardAlgorithm(
			GPUContext gCtx, String instName, int N, int C, int H, int W, int K, int R, int S, 
			int pad_h, int pad_w, int stride_h, int stride_w, int P, int Q, long workspaceLimit) throws DMLRuntimeException {
		long t1 = GPUStatistics.DISPLAY_STATISTICS ? System.nanoTime() : 0;
		LibMatrixCuDNNConvolutionAlgorithm ret = new LibMatrixCuDNNConvolutionAlgorithm(gCtx, instName, N, C, H, W, K, R, S, 
				pad_h, pad_w, stride_h, stride_w, P, Q);
		if(workspaceLimit <= 0) {
			// If overhead is greater than intermediate allocated memory, prefer the cudnn operator with no memory requirement, 
			// i.e. CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM
			ret.algo = jcuda.jcudnn.cudnnConvolutionFwdAlgo.CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM;
		}
		else {
			int[] algos = {-1};
			long sizeInBytesArray[] = {workspaceLimit};
			jcuda.jcudnn.JCudnn.cudnnGetConvolutionForwardAlgorithm(LibMatrixCuDNN.getCudnnHandle(gCtx), 
					ret.nchwTensorDesc, ret.filterDesc, ret.convDesc, ret.nkpqTensorDesc,
					cudnnConvolutionFwdPreference.CUDNN_CONVOLUTION_FWD_SPECIFY_WORKSPACE_LIMIT, sizeInBytesArray[0], algos);
			jcuda.jcudnn.JCudnn.cudnnGetConvolutionForwardWorkspaceSize(LibMatrixCuDNN.getCudnnHandle(gCtx), 
					ret.nchwTensorDesc, ret.filterDesc, ret.convDesc, ret.nkpqTensorDesc, algos[0], sizeInBytesArray);
			if (sizeInBytesArray[0] != 0)
				ret.workSpace = gCtx.allocate(sizeInBytesArray[0]);
			ret.sizeInBytes = sizeInBytesArray[0];
			ret.algo = algos[0];
		}
		if (GPUStatistics.DISPLAY_STATISTICS)
			GPUStatistics.maintainCPMiscTimes(instName, GPUInstruction.MISC_TIMER_CUDNN_INIT, System.nanoTime() - t1);
		return ret;
	}
	
	/**
	 * Factory method to get the algorithm wrapper for convolution backward filter
	 * 
	 * @param gCtx     a valid {@link GPUContext}
	 * @param instName the invoking instruction's name for record {@link org.apache.sysml.utils.Statistics}.
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
	 * @throws DMLRuntimeException if error occurs
	 */
	public static LibMatrixCuDNNConvolutionAlgorithm cudnnGetConvolutionBackwardFilterAlgorithm(
			GPUContext gCtx, String instName, int N, int C, int H, int W, int K, int R, int S, 
			int pad_h, int pad_w, int stride_h, int stride_w, int P, int Q, long workspaceLimit) throws DMLRuntimeException {
		long t1 = GPUStatistics.DISPLAY_STATISTICS ? System.nanoTime() : 0;
		LibMatrixCuDNNConvolutionAlgorithm ret = new LibMatrixCuDNNConvolutionAlgorithm(gCtx, instName, N, C, H, W, K, R, S, 
				pad_h, pad_w, stride_h, stride_w, P, Q);
		
		if(workspaceLimit <= 0) {
			// If overhead is greater than intermediate allocated memory, prefer the cudnn operator with no memory requirement
			// i.e. CUDNN_CONVOLUTION_BWD_FILTER_ALGO_0
			ret.algo = jcuda.jcudnn.cudnnConvolutionBwdFilterAlgo.CUDNN_CONVOLUTION_BWD_FILTER_ALGO_0;
		}
		else {
			int[] algos = {-1};
			long sizeInBytesArray[] = {workspaceLimit};
			jcuda.jcudnn.JCudnn.cudnnGetConvolutionBackwardFilterAlgorithm(
					LibMatrixCuDNN.getCudnnHandle(gCtx), 
					ret.nchwTensorDesc, ret.nkpqTensorDesc, ret.convDesc, ret.filterDesc, 
					cudnnConvolutionBwdFilterPreference.CUDNN_CONVOLUTION_BWD_FILTER_SPECIFY_WORKSPACE_LIMIT, sizeInBytesArray[0], algos);
			jcuda.jcudnn.JCudnn.cudnnGetConvolutionBackwardFilterWorkspaceSize(LibMatrixCuDNN.getCudnnHandle(gCtx), 
					ret.nchwTensorDesc, ret.nkpqTensorDesc, ret.convDesc, ret.filterDesc, algos[0], sizeInBytesArray);
			if (sizeInBytesArray[0] != 0)
				ret.workSpace = gCtx.allocate(sizeInBytesArray[0]);
			ret.sizeInBytes = sizeInBytesArray[0];
			ret.algo = algos[0];
		}
		if (GPUStatistics.DISPLAY_STATISTICS)
			GPUStatistics.maintainCPMiscTimes(instName, GPUInstruction.MISC_TIMER_CUDNN_INIT, System.nanoTime() - t1);
		return ret;
	}
	
	/**
	 * Factory method to get the algorithm wrapper for convolution backward data
	 * 
	 * @param gCtx     a valid {@link GPUContext}
	 * @param instName the invoking instruction's name for record {@link org.apache.sysml.utils.Statistics}.
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
	 * @throws DMLRuntimeException if error occurs
	 */
	public static LibMatrixCuDNNConvolutionAlgorithm cudnnGetConvolutionBackwardDataAlgorithm(
			GPUContext gCtx, String instName, int N, int C, int H, int W, int K, int R, int S, 
			int pad_h, int pad_w, int stride_h, int stride_w, int P, int Q, long workspaceLimit) throws DMLRuntimeException {
		long t1 = GPUStatistics.DISPLAY_STATISTICS ? System.nanoTime() : 0;
		LibMatrixCuDNNConvolutionAlgorithm ret = new LibMatrixCuDNNConvolutionAlgorithm(gCtx, instName, N, C, H, W, K, R, S, 
				pad_h, pad_w, stride_h, stride_w, P, Q);
		
		if(workspaceLimit <= 0) {
			// If overhead is greater than intermediate allocated memory, prefer the cudnn operator with no memory requirement
			// i.e. CUDNN_CONVOLUTION_BWD_DATA_ALGO_0
			ret.algo = jcuda.jcudnn.cudnnConvolutionBwdDataAlgo.CUDNN_CONVOLUTION_BWD_DATA_ALGO_0;
		}
		else {
			int[] algos = {-1};
			long sizeInBytesArray[] = {workspaceLimit};
			jcuda.jcudnn.JCudnn.cudnnGetConvolutionBackwardDataAlgorithm(
					LibMatrixCuDNN.getCudnnHandle(gCtx), 
					ret.filterDesc, ret.nkpqTensorDesc, ret.convDesc, ret.nchwTensorDesc,
					cudnnConvolutionBwdDataPreference.CUDNN_CONVOLUTION_BWD_DATA_SPECIFY_WORKSPACE_LIMIT, sizeInBytesArray[0], algos);
			jcuda.jcudnn.JCudnn.cudnnGetConvolutionBackwardDataWorkspaceSize(LibMatrixCuDNN.getCudnnHandle(gCtx), 
					ret.filterDesc, ret.nkpqTensorDesc, ret.convDesc, ret.nchwTensorDesc, algos[0], sizeInBytesArray);
			if (sizeInBytesArray[0] != 0)
				ret.workSpace = gCtx.allocate(sizeInBytesArray[0]);
			ret.sizeInBytes = sizeInBytesArray[0];
			ret.algo = algos[0];
		}
		if (GPUStatistics.DISPLAY_STATISTICS)
			GPUStatistics.maintainCPMiscTimes(instName, GPUInstruction.MISC_TIMER_CUDNN_INIT, System.nanoTime() - t1);
		return ret;
	}
}
