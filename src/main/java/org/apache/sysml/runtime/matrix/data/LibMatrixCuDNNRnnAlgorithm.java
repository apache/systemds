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

import static jcuda.jcudnn.JCudnn.cudnnCreateFilterDescriptor;
import static jcuda.jcudnn.JCudnn.cudnnCreateTensorDescriptor;
import static jcuda.jcudnn.JCudnn.cudnnDestroyFilterDescriptor;
import static jcuda.jcudnn.JCudnn.cudnnDestroyTensorDescriptor;
import static jcuda.jcudnn.JCudnn.cudnnSetTensorNdDescriptor;
import static jcuda.jcudnn.JCudnn.cudnnDestroyDropoutDescriptor;
import static jcuda.jcudnn.JCudnn.cudnnDestroyRNNDescriptor;
import static jcuda.jcudnn.cudnnTensorFormat.CUDNN_TENSOR_NCHW;
import static jcuda.jcudnn.JCudnn.cudnnCreateRNNDescriptor;
import static jcuda.jcudnn.cudnnRNNInputMode.CUDNN_LINEAR_INPUT;
import static jcuda.jcudnn.cudnnDirectionMode.CUDNN_UNIDIRECTIONAL;
import static jcuda.jcudnn.cudnnRNNAlgo.CUDNN_RNN_ALGO_STANDARD;

import org.apache.sysml.api.DMLScript;
import org.apache.sysml.runtime.DMLRuntimeException;
import org.apache.sysml.runtime.controlprogram.context.ExecutionContext;
import org.apache.sysml.runtime.instructions.gpu.context.GPUContext;

import jcuda.Pointer;
import jcuda.jcudnn.JCudnn;
import jcuda.jcudnn.cudnnDropoutDescriptor;
import jcuda.jcudnn.cudnnFilterDescriptor;
import jcuda.jcudnn.cudnnRNNDescriptor;
import jcuda.jcudnn.cudnnTensorDescriptor;

public class LibMatrixCuDNNRnnAlgorithm implements java.lang.AutoCloseable {
	GPUContext gCtx;
	String instName;
	cudnnDropoutDescriptor dropoutDesc;
	cudnnRNNDescriptor rnnDesc;
	cudnnTensorDescriptor[] xDesc, yDesc; // of length T
	cudnnTensorDescriptor hxDesc, cxDesc, hyDesc, cyDesc; 
	cudnnFilterDescriptor wDesc;
	long sizeInBytes; Pointer workSpace;
	long reserveSpaceSizeInBytes; Pointer reserveSpace;
	public LibMatrixCuDNNRnnAlgorithm(ExecutionContext ec, GPUContext gCtx, String instName, 
			String rnnMode, int N, int T, int M, int D, boolean isTraining, Pointer w, String reserveSpaceName) throws DMLRuntimeException {
		this.gCtx = gCtx;
		this.instName = instName;
		
		// Allocate input/output descriptors
		xDesc = new cudnnTensorDescriptor[T];
		yDesc = new cudnnTensorDescriptor[T];
		for(int t = 0; t < T; t++) {
			xDesc[t] = allocateTensorDescriptorWithStride(N, D, 1);
			yDesc[t] = allocateTensorDescriptorWithStride(N, M, 1);
		}
		hxDesc = allocateTensorDescriptorWithStride(1, N, M); 
		cxDesc = allocateTensorDescriptorWithStride(1, N, M);
		hyDesc = allocateTensorDescriptorWithStride(1, N, M);
		cyDesc = allocateTensorDescriptorWithStride(1, N, M);
		
		// Initial dropout descriptor
		dropoutDesc = new cudnnDropoutDescriptor();
		JCudnn.cudnnCreateDropoutDescriptor(dropoutDesc);
		long [] dropOutSizeInBytes = {-1};
		JCudnn.cudnnDropoutGetStatesSize(gCtx.getCudnnHandle(), dropOutSizeInBytes);
		Pointer dropOutStateSpace = new Pointer();
		if (dropOutSizeInBytes[0] != 0)
			dropOutStateSpace = gCtx.allocate(dropOutSizeInBytes[0]);
		JCudnn.cudnnSetDropoutDescriptor(dropoutDesc, gCtx.getCudnnHandle(), 0, dropOutStateSpace, dropOutSizeInBytes[0], 12345);
		
		// Initialize RNN descriptor
		rnnDesc = new cudnnRNNDescriptor();
		cudnnCreateRNNDescriptor(rnnDesc);
		JCudnn.cudnnSetRNNDescriptor_v6(gCtx.getCudnnHandle(), rnnDesc, M, 1, dropoutDesc, 
				CUDNN_LINEAR_INPUT, CUDNN_UNIDIRECTIONAL, 
				getCuDNNRnnMode(rnnMode), CUDNN_RNN_ALGO_STANDARD, LibMatrixCUDA.CUDNN_DATA_TYPE);
		
		// Allocate filter descriptor
		int expectedNumWeights = getExpectedNumWeights();
		if(rnnMode.equalsIgnoreCase("lstm") && (D+M+2)*4*M != expectedNumWeights) {
			throw new DMLRuntimeException("Incorrect number of RNN parameters " +  (D+M+2)*4*M + " != " +  expectedNumWeights + ", where numFeatures=" + D + ", hiddenSize=" + M);
		}
		wDesc = allocateFilterDescriptor(expectedNumWeights);
		
		// Setup workspace
		workSpace = new Pointer(); reserveSpace = new Pointer();
		sizeInBytes = getWorkspaceSize(T);
		if(sizeInBytes != 0)
			workSpace = gCtx.allocate(sizeInBytes);
		reserveSpaceSizeInBytes = 0;
		if(isTraining) {
			reserveSpaceSizeInBytes = getReservespaceSize(T);
			if (reserveSpaceSizeInBytes != 0) {
				int numCols =  (int) Math.ceil(((double)reserveSpaceSizeInBytes) / LibMatrixCUDA.sizeOfDataType);
				reserveSpace = LibMatrixCuDNN.getDenseOutputPointer(ec, gCtx, instName, reserveSpaceName, 1, numCols);
			}
		}
		if (reserveSpaceSizeInBytes == 0) {
			reserveSpace = LibMatrixCuDNN.getDenseOutputPointer(ec, gCtx, instName, reserveSpaceName, 1, 1);
		}
		
		/*
		int numLinearLayers = getNumLinearLayers(rnnMode); 
		for(int i = 0; i < numLinearLayers; i++) {
			cudnnFilterDescriptor  linLayerMatDesc = new cudnnFilterDescriptor();
			cudnnCreateFilterDescriptor(linLayerMatDesc);
			Pointer linLayerMat = new Pointer();
			JCudnn.cudnnGetRNNLinLayerMatrixParams(gCtx.getCudnnHandle(), rnnDesc, 0, 
					xDesc[0], wDesc, w, i, linLayerMatDesc, linLayerMat);
			int[] dataType = new int[] {-1};
			int[] format = new int[] {-1};
			int[] nbDims = new int[] {-1};
			int[] filterDimA = new int[3];
			JCudnn.cudnnGetFilterNdDescriptor(linLayerMatDesc, 3, dataType, format, nbDims, filterDimA);
			
			int filterDims = filterDimA[0] * filterDimA[1] * filterDimA[2];
			double [] tmp = new double[filterDims];
			LibMatrixCUDA.cudaSupportFunctions.deviceToHost(gCtx, linLayerMat, tmp, instName, false);
			System.out.println();
			for(int j = 0 ; j < tmp.length; j++) {
				System.out.print(" " + tmp[j]);
			}
			System.out.println();
			LibMatrixCUDA.getCudaKernels(gCtx).launchKernel("fill", 
					org.apache.sysml.runtime.instructions.gpu.context.ExecutionConfig.getConfigForSimpleVectorOperations(filterDims), 
					linLayerMat, Math.pow(filterDims, -1), filterDims);
			JCudnn.cudnnDestroyFilterDescriptor(linLayerMatDesc);
			
			cudnnFilterDescriptor  linLayerBiasDesc = new cudnnFilterDescriptor();
			cudnnCreateFilterDescriptor(linLayerBiasDesc);
			Pointer linLayerBias = new Pointer();
			JCudnn.cudnnGetRNNLinLayerBiasParams(gCtx.getCudnnHandle(), rnnDesc, 0,
					xDesc[0], wDesc, w, i, linLayerBiasDesc, linLayerBias);
			JCudnn.cudnnGetFilterNdDescriptor(linLayerBiasDesc, 3, dataType, format, nbDims, filterDimA);
			filterDims = filterDimA[0] * filterDimA[1] * filterDimA[2];
			LibMatrixCUDA.getCudaKernels(gCtx).launchKernel("fill", 
					org.apache.sysml.runtime.instructions.gpu.context.ExecutionConfig.getConfigForSimpleVectorOperations(filterDims), 
					linLayerBias, Math.pow(filterDims, -1), filterDims);
			JCudnn.cudnnDestroyFilterDescriptor(linLayerBiasDesc);
		}
		*/
	}
	
	@SuppressWarnings("unused")
	private int getNumLinearLayers(String rnnMode) throws DMLRuntimeException {
		int ret = 0;
		if(rnnMode.equalsIgnoreCase("rnn_relu") || rnnMode.equalsIgnoreCase("rnn_tanh")) {
			ret = 2;
		}
		else if(rnnMode.equalsIgnoreCase("lstm")) {
			ret = 8;
		}
		else if(rnnMode.equalsIgnoreCase("gru")) {
			ret = 6;
		}
		else {
			throw new DMLRuntimeException("Unsupported rnn mode:" + rnnMode);
		}
		return ret;
	}
	
	private long getWorkspaceSize(int seqLength) {
		long [] sizeInBytesArray = new long[1];
		JCudnn.cudnnGetRNNWorkspaceSize(gCtx.getCudnnHandle(), rnnDesc, seqLength, xDesc, sizeInBytesArray);
		return sizeInBytesArray[0];
	}
	
	private long getReservespaceSize(int seqLength) {
		long [] sizeInBytesArray = new long[1];
		JCudnn.cudnnGetRNNTrainingReserveSize(gCtx.getCudnnHandle(), rnnDesc, seqLength, xDesc, sizeInBytesArray);
		return sizeInBytesArray[0];
	}
	
	private int getCuDNNRnnMode(String rnnMode) throws DMLRuntimeException {
		int rnnModeVal = -1;
		if(rnnMode.equalsIgnoreCase("rnn_relu")) {
			rnnModeVal = jcuda.jcudnn.cudnnRNNMode.CUDNN_RNN_RELU;
		}
		else if(rnnMode.equalsIgnoreCase("rnn_tanh")) {
			rnnModeVal = jcuda.jcudnn.cudnnRNNMode.CUDNN_RNN_TANH;
		}
		else if(rnnMode.equalsIgnoreCase("lstm")) {
			rnnModeVal = jcuda.jcudnn.cudnnRNNMode.CUDNN_LSTM;
		}
		else if(rnnMode.equalsIgnoreCase("gru")) {
			rnnModeVal = jcuda.jcudnn.cudnnRNNMode.CUDNN_GRU;
		}
		else {
			throw new DMLRuntimeException("Unsupported rnn mode:" + rnnMode);
		}
		return rnnModeVal;
	}
	
	private int getExpectedNumWeights() throws DMLRuntimeException {
		long [] weightSizeInBytesArray = {-1}; // (D+M+2)*4*M
		JCudnn.cudnnGetRNNParamsSize(gCtx.getCudnnHandle(), rnnDesc, xDesc[0], weightSizeInBytesArray, LibMatrixCUDA.CUDNN_DATA_TYPE);
		// check if (D+M+2)*4M == weightsSize / sizeof(dataType) where weightsSize is given by 'cudnnGetRNNParamsSize'.
		return LibMatrixCUDA.toInt(weightSizeInBytesArray[0]/LibMatrixCUDA.sizeOfDataType);
	}
	
	private cudnnFilterDescriptor allocateFilterDescriptor(int numWeights) {
		cudnnFilterDescriptor filterDesc = new cudnnFilterDescriptor();
		cudnnCreateFilterDescriptor(filterDesc);
		JCudnn.cudnnSetFilterNdDescriptor(filterDesc, LibMatrixCUDA.CUDNN_DATA_TYPE, CUDNN_TENSOR_NCHW, 3, new int[] {numWeights, 1, 1});
		return filterDesc;
	}
	
	
	
	private static cudnnTensorDescriptor allocateTensorDescriptorWithStride(int firstDim, int secondDim, int thirdDim) throws DMLRuntimeException {
		cudnnTensorDescriptor tensorDescriptor = new cudnnTensorDescriptor();
		cudnnCreateTensorDescriptor(tensorDescriptor);
		int [] dimA = new int[] {firstDim, secondDim, thirdDim};
		int [] strideA = new int[] {dimA[2] * dimA[1], dimA[2], 1};
		cudnnSetTensorNdDescriptor(tensorDescriptor, LibMatrixCUDA.CUDNN_DATA_TYPE, 3, dimA, strideA);
		return tensorDescriptor;
	}


	@Override
	public void close() {
		if(dropoutDesc != null)
			cudnnDestroyDropoutDescriptor(dropoutDesc);
		dropoutDesc = null;
		if(rnnDesc != null)
			cudnnDestroyRNNDescriptor(rnnDesc);
		rnnDesc = null;
		if(hxDesc != null)
			cudnnDestroyTensorDescriptor(hxDesc);
		hxDesc = null;
		if(hyDesc != null)
			cudnnDestroyTensorDescriptor(hyDesc);
		hyDesc = null;
		if(cxDesc != null)
			cudnnDestroyTensorDescriptor(cxDesc);
		cxDesc = null;
		if(cyDesc != null)
			cudnnDestroyTensorDescriptor(cyDesc);
		cyDesc = null;
		if(wDesc != null)
			cudnnDestroyFilterDescriptor(wDesc);
		wDesc = null;
		if(xDesc != null) {
			for(cudnnTensorDescriptor dsc : xDesc) {
				cudnnDestroyTensorDescriptor(dsc);
			}
			xDesc = null;
		}
		if(yDesc != null) {
			for(cudnnTensorDescriptor dsc : yDesc) {
				cudnnDestroyTensorDescriptor(dsc);
			}
			yDesc = null;
		}
		if(sizeInBytes != 0) {
			try {
				gCtx.cudaFreeHelper(instName, workSpace, DMLScript.EAGER_CUDA_FREE);
			} catch (DMLRuntimeException e) {
				throw new RuntimeException(e);
			}
		}
		if(reserveSpaceSizeInBytes != 0) {
			try {
				gCtx.cudaFreeHelper(instName, reserveSpace, DMLScript.EAGER_CUDA_FREE);
			} catch (DMLRuntimeException e) {
				throw new RuntimeException(e);
			}
		}	
	}
}
