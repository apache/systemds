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

import static jcuda.jcudnn.JCudnn.cudnnCreateFilterDescriptor;
import static jcuda.jcudnn.JCudnn.cudnnCreateTensorDescriptor;
import static jcuda.jcudnn.JCudnn.cudnnDestroyFilterDescriptor;
import static jcuda.jcudnn.JCudnn.cudnnDestroyTensorDescriptor;
import static jcuda.jcudnn.JCudnn.cudnnSetTensorNdDescriptor;
import static jcuda.jcudnn.JCudnn.cudnnDestroyDropoutDescriptor;
import static jcuda.jcudnn.JCudnn.cudnnDestroyRNNDescriptor;
import static jcuda.jcudnn.JCudnn.cudnnGetRNNWeightSpaceSize;
import static jcuda.jcudnn.cudnnTensorFormat.CUDNN_TENSOR_NCHW;
import static jcuda.jcudnn.cudnnDataType.CUDNN_DATA_HALF;
import static jcuda.jcudnn.cudnnDataType.CUDNN_DATA_BFLOAT16;
import static jcuda.jcudnn.cudnnMathType.CUDNN_DEFAULT_MATH;
import static jcuda.jcudnn.cudnnMathType.CUDNN_TENSOR_OP_MATH;
import static jcuda.jcudnn.cudnnRNNAlgo.CUDNN_RNN_ALGO_STANDARD;
import static jcuda.jcudnn.cudnnRNNBiasMode.CUDNN_RNN_DOUBLE_BIAS;
import static jcuda.jcudnn.cudnnDirectionMode.CUDNN_UNIDIRECTIONAL;
import static jcuda.jcudnn.cudnnRNNInputMode.CUDNN_LINEAR_INPUT;
import static jcuda.jcudnn.cudnnRNNDataLayout.CUDNN_RNN_DATA_LAYOUT_SEQ_MAJOR_PACKED;
import static jcuda.jcudnn.cudnnForwardMode.CUDNN_FWD_MODE_TRAINING;

import org.apache.sysds.api.DMLScript;
import org.apache.sysds.common.Opcodes;
import org.apache.sysds.runtime.DMLRuntimeException;
import org.apache.sysds.runtime.controlprogram.context.ExecutionContext;
import org.apache.sysds.runtime.instructions.gpu.context.GPUContext;

import static jcuda.jcudnn.JCudnn.cudnnCreateRNNDescriptor;

import jcuda.Pointer;
import jcuda.jcudnn.JCudnn;
import jcuda.jcudnn.cudnnDropoutDescriptor;
import jcuda.jcudnn.cudnnFilterDescriptor;
import jcuda.jcudnn.cudnnRNNDescriptor;
import jcuda.jcudnn.cudnnTensorDescriptor;
import jcuda.jcudnn.cudnnRNNDataDescriptor;

public class LibMatrixCuDNNRnnAlgorithm implements java.lang.AutoCloseable {
	GPUContext gCtx;
	String instName;
	cudnnDropoutDescriptor dropoutDesc;
	cudnnRNNDescriptor rnnDesc;
	cudnnTensorDescriptor[] xDesc, dxDesc, yDesc, dyDesc; // of length T
	cudnnRNNDataDescriptor xDataDesc;
	cudnnTensorDescriptor hxDesc, cxDesc, hyDesc, cyDesc, dhxDesc, dcxDesc, dhyDesc, dcyDesc; 
	cudnnFilterDescriptor wDesc;
	cudnnFilterDescriptor dwDesc;
	long sizeInBytes; Pointer workSpace;
	long reserveSpaceSizeInBytes; Pointer reserveSpace;
	long dropOutSizeInBytes; Pointer dropOutStateSpace;
	public LibMatrixCuDNNRnnAlgorithm(ExecutionContext ec, GPUContext gCtx, String instName, 
			String rnnMode, int N, int T, int M, int D, boolean isTraining, Pointer w) throws DMLRuntimeException {
		this.gCtx = gCtx;
		this.instName = instName;
		
		// Allocate input/output descriptors
		xDesc = new cudnnTensorDescriptor[T];
		dxDesc = new cudnnTensorDescriptor[T];
		yDesc = new cudnnTensorDescriptor[T];
		dyDesc = new cudnnTensorDescriptor[T];
		for(int t = 0; t < T; t++) {
			xDesc[t] = allocateTensorDescriptorWithStride(N, D, 1);
			dxDesc[t] = allocateTensorDescriptorWithStride(N, D, 1);
			yDesc[t] = allocateTensorDescriptorWithStride(N, M, 1);
			dyDesc[t] = allocateTensorDescriptorWithStride(N, M, 1);
		}
		hxDesc = allocateTensorDescriptorWithStride(1, N, M); 
		dhxDesc = allocateTensorDescriptorWithStride(1, N, M);
		cxDesc = allocateTensorDescriptorWithStride(1, N, M);
		dcxDesc = allocateTensorDescriptorWithStride(1, N, M);
		hyDesc = allocateTensorDescriptorWithStride(1, N, M);
		dhyDesc = allocateTensorDescriptorWithStride(1, N, M);
		cyDesc = allocateTensorDescriptorWithStride(1, N, M);
		dcyDesc = allocateTensorDescriptorWithStride(1, N, M);
		
		// Initial dropout descriptor
		dropoutDesc = new cudnnDropoutDescriptor();
		JCudnn.cudnnCreateDropoutDescriptor(dropoutDesc);
		long [] _dropOutSizeInBytes = {-1};
		JCudnn.cudnnDropoutGetStatesSize(gCtx.getCudnnHandle(), _dropOutSizeInBytes);
		dropOutSizeInBytes = _dropOutSizeInBytes[0];
		dropOutStateSpace = new Pointer();
		if(dropOutSizeInBytes != 0)
			dropOutStateSpace = gCtx.allocate(instName, dropOutSizeInBytes, false);
		JCudnn.cudnnSetDropoutDescriptor(dropoutDesc, gCtx.getCudnnHandle(), 0, dropOutStateSpace, dropOutSizeInBytes,
			12345);

		// Initialize RNN descriptor
		rnnDesc = new cudnnRNNDescriptor();
		cudnnCreateRNNDescriptor(rnnDesc);
		int mathType = (LibMatrixCUDA.CUDNN_DATA_TYPE == CUDNN_DATA_HALF ||
			LibMatrixCUDA.CUDNN_DATA_TYPE == CUDNN_DATA_BFLOAT16) ? CUDNN_TENSOR_OP_MATH : CUDNN_DEFAULT_MATH;
		int mathPrec = LibMatrixCUDA.CUDNN_DATA_TYPE;
		JCudnn.cudnnSetRNNDescriptor_v8(rnnDesc, CUDNN_RNN_ALGO_STANDARD, getCuDNNRnnMode(rnnMode),
			CUDNN_RNN_DOUBLE_BIAS, CUDNN_UNIDIRECTIONAL, CUDNN_LINEAR_INPUT, LibMatrixCUDA.CUDNN_DATA_TYPE, mathPrec,
			mathType, D, M, 0, 1, dropoutDesc, 0);

		// ── inside the constructor, after rnnDesc has been configured ──
		xDataDesc = new cudnnRNNDataDescriptor();
		JCudnn.cudnnCreateRNNDataDescriptor(xDataDesc);
		JCudnn.cudnnSetRNNDataDescriptor(xDataDesc, LibMatrixCUDA.CUDNN_DATA_TYPE,
			CUDNN_RNN_DATA_LAYOUT_SEQ_MAJOR_PACKED, T, N, D, null, null);

		// Allocate filter descriptor
		int expectedNumWeights = getExpectedNumWeights();
		if(rnnMode.equalsIgnoreCase(Opcodes.LSTM.toString()) && (D + M + 2) * 4 * M != expectedNumWeights) {
			throw new DMLRuntimeException(
				"Incorrect number of RNN parameters " + (D + M + 2) * 4 * M + " != " + expectedNumWeights +
					", where numFeatures=" + D + ", hiddenSize=" + M);
		}
		wDesc = allocateFilterDescriptor(expectedNumWeights);
		dwDesc = allocateFilterDescriptor(expectedNumWeights);

		// Setup workspace
		workSpace = new Pointer(); reserveSpace = new Pointer();
		sizeInBytes = getWorkspaceSize(T);
		if(sizeInBytes != 0)
			workSpace = gCtx.allocate(instName, sizeInBytes, false);
		reserveSpaceSizeInBytes = 0;
		if(isTraining) {
			reserveSpaceSizeInBytes = getReservespaceSize(T);
			if (reserveSpaceSizeInBytes != 0) {
				reserveSpace = gCtx.allocate(instName, reserveSpaceSizeInBytes, false);
			}
		}
	}
	
	@SuppressWarnings("unused")
	private static int getNumLinearLayers(String rnnMode) throws DMLRuntimeException {
		int ret = 0;
		if(rnnMode.equalsIgnoreCase("rnn_relu") || rnnMode.equalsIgnoreCase("rnn_tanh")) {
			ret = 2;
		}
		else if(rnnMode.equalsIgnoreCase(Opcodes.LSTM.toString())) {
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
		long[] workSize = {0};
		long[] reserveSize = {0};          // v9 returns both sizes in one call

		JCudnn.cudnnGetRNNTempSpaceSizes(gCtx.getCudnnHandle(), rnnDesc, CUDNN_FWD_MODE_TRAINING, xDataDesc, workSize,
			reserveSize);

		// keep reserveSpaceSizeInBytes in sync with the new API
		reserveSpaceSizeInBytes = reserveSize[0];
		return workSize[0];
	}
	
	private long getReservespaceSize(int seqLength) {
		long[] workSize = {0};
		long[] reserveSize = {0};

		JCudnn.cudnnGetRNNTempSpaceSizes(gCtx.getCudnnHandle(), rnnDesc, CUDNN_FWD_MODE_TRAINING, xDataDesc, workSize,
			reserveSize);

		return reserveSize[0];
	}
	
	private static int getCuDNNRnnMode(String rnnMode) throws DMLRuntimeException {
		int rnnModeVal = -1;
		if(rnnMode.equalsIgnoreCase("rnn_relu")) {
			rnnModeVal = jcuda.jcudnn.cudnnRNNMode.CUDNN_RNN_RELU;
		}
		else if(rnnMode.equalsIgnoreCase("rnn_tanh")) {
			rnnModeVal = jcuda.jcudnn.cudnnRNNMode.CUDNN_RNN_TANH;
		}
		else if(rnnMode.equalsIgnoreCase(Opcodes.LSTM.toString())) {
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
		long[] weightSpaceBytes = {-1};

		// v9 API: returns the size (in bytes) of the packed “weight space”
		cudnnGetRNNWeightSpaceSize(gCtx.getCudnnHandle(), rnnDesc, weightSpaceBytes);

		if(weightSpaceBytes[0] < 0)
			throw new DMLRuntimeException("cuDNN returned a negative weight-space size");

		// convert from bytes to number of scalars
		long numScalars = weightSpaceBytes[0] / LibMatrixCUDA.sizeOfDataType;
		return LibMatrixCUDA.toInt(numScalars);
	}
	
	@SuppressWarnings("deprecation")
	private static cudnnFilterDescriptor allocateFilterDescriptor(int numWeights) {
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


	@SuppressWarnings("deprecation")
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
		if(dhxDesc != null)
			cudnnDestroyTensorDescriptor(dhxDesc);
		dhxDesc = null;
		if(hyDesc != null)
			cudnnDestroyTensorDescriptor(hyDesc);
		hyDesc = null;
		if(dhyDesc != null)
			cudnnDestroyTensorDescriptor(dhyDesc);
		dhyDesc = null;
		if(cxDesc != null)
			cudnnDestroyTensorDescriptor(cxDesc);
		cxDesc = null;
		if(dcxDesc != null)
			cudnnDestroyTensorDescriptor(dcxDesc);
		dcxDesc = null;
		if(cyDesc != null)
			cudnnDestroyTensorDescriptor(cyDesc);
		cyDesc = null;
		if(dcyDesc != null)
			cudnnDestroyTensorDescriptor(dcyDesc);
		dcyDesc = null;
		if(wDesc != null)
			cudnnDestroyFilterDescriptor(wDesc);
		wDesc = null;
		if(dwDesc != null)
			cudnnDestroyFilterDescriptor(dwDesc);
		dwDesc = null;
		if(xDesc != null) {
			for(cudnnTensorDescriptor dsc : xDesc) {
				cudnnDestroyTensorDescriptor(dsc);
			}
			xDesc = null;
		}
		if(dxDesc != null) {
			for(cudnnTensorDescriptor dsc : dxDesc) {
				cudnnDestroyTensorDescriptor(dsc);
			}
			dxDesc = null;
		}
		if(yDesc != null) {
			for(cudnnTensorDescriptor dsc : yDesc) {
				cudnnDestroyTensorDescriptor(dsc);
			}
			yDesc = null;
		}
		if(dyDesc != null) {
			for(cudnnTensorDescriptor dsc : dyDesc) {
				cudnnDestroyTensorDescriptor(dsc);
			}
			dyDesc = null;
		}
		if(sizeInBytes != 0) {
			try {
				gCtx.cudaFreeHelper(instName, workSpace, DMLScript.EAGER_CUDA_FREE);
			} catch (DMLRuntimeException e) {
				throw new RuntimeException(e);
			}
		}
		workSpace = null;
		if(reserveSpaceSizeInBytes != 0) {
			try {
				gCtx.cudaFreeHelper(instName, reserveSpace, DMLScript.EAGER_CUDA_FREE);
			} catch (DMLRuntimeException e) {
				throw new RuntimeException(e);
			}
		}	
		reserveSpace = null;
		if(dropOutSizeInBytes != 0) {
			try {
				gCtx.cudaFreeHelper(instName, dropOutStateSpace, DMLScript.EAGER_CUDA_FREE);
			} catch (DMLRuntimeException e) {
				throw new RuntimeException(e);
			}
		}
	}
}
