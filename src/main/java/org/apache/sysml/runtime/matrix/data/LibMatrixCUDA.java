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

import jcuda.Pointer;
import jcuda.Sizeof;
import jcuda.jcublas.JCublas2;
import jcuda.jcublas.cublasFillMode;
import jcuda.jcublas.cublasHandle;
import jcuda.jcublas.cublasOperation;
import jcuda.jcudnn.cudnnActivationDescriptor;
import jcuda.jcudnn.cudnnConvolutionDescriptor;
import jcuda.jcudnn.cudnnConvolutionFwdPreference;
import jcuda.jcudnn.cudnnFilterDescriptor;
import jcuda.jcudnn.cudnnHandle;
import jcuda.jcudnn.cudnnPoolingDescriptor;
import jcuda.jcudnn.cudnnTensorDescriptor;
import jcuda.jcusparse.JCusparse;
import jcuda.jcusparse.cusparseHandle;

import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;
import org.apache.sysml.api.DMLScript;
import org.apache.sysml.runtime.DMLRuntimeException;
import org.apache.sysml.runtime.controlprogram.caching.MatrixObject;
import org.apache.sysml.runtime.controlprogram.context.ExecutionContext;
import org.apache.sysml.runtime.functionobjects.And;
import org.apache.sysml.runtime.functionobjects.Builtin;
import org.apache.sysml.runtime.functionobjects.CM;
import org.apache.sysml.runtime.functionobjects.Divide;
import org.apache.sysml.runtime.functionobjects.Equals;
import org.apache.sysml.runtime.functionobjects.GreaterThan;
import org.apache.sysml.runtime.functionobjects.GreaterThanEquals;
import org.apache.sysml.runtime.functionobjects.IndexFunction;
import org.apache.sysml.runtime.functionobjects.KahanPlus;
import org.apache.sysml.runtime.functionobjects.KahanPlusSq;
import org.apache.sysml.runtime.functionobjects.LessThan;
import org.apache.sysml.runtime.functionobjects.LessThanEquals;
import org.apache.sysml.runtime.functionobjects.Mean;
import org.apache.sysml.runtime.functionobjects.Minus;
import org.apache.sysml.runtime.functionobjects.Multiply;
import org.apache.sysml.runtime.functionobjects.Multiply2;
import org.apache.sysml.runtime.functionobjects.NotEquals;
import org.apache.sysml.runtime.functionobjects.Or;
import org.apache.sysml.runtime.functionobjects.Plus;
import org.apache.sysml.runtime.functionobjects.Power;
import org.apache.sysml.runtime.functionobjects.Power2;
import org.apache.sysml.runtime.functionobjects.ReduceAll;
import org.apache.sysml.runtime.functionobjects.ReduceCol;
import org.apache.sysml.runtime.functionobjects.ReduceDiag;
import org.apache.sysml.runtime.functionobjects.ReduceRow;
import org.apache.sysml.runtime.functionobjects.ValueFunction;
import org.apache.sysml.runtime.instructions.cp.DoubleObject;
import org.apache.sysml.runtime.instructions.gpu.GPUInstruction;
import org.apache.sysml.runtime.instructions.gpu.context.ExecutionConfig;
import org.apache.sysml.runtime.instructions.gpu.context.GPUContext;
import org.apache.sysml.runtime.instructions.gpu.context.JCudaContext;
import org.apache.sysml.runtime.instructions.gpu.context.JCudaKernels;
import org.apache.sysml.runtime.instructions.gpu.context.JCudaObject;
import org.apache.sysml.runtime.instructions.gpu.context.JCudaObject.CSRPointer;
import org.apache.sysml.runtime.matrix.operators.AggregateOperator;
import org.apache.sysml.runtime.matrix.operators.AggregateUnaryOperator;
import org.apache.sysml.runtime.matrix.operators.BinaryOperator;
import org.apache.sysml.runtime.matrix.operators.CMOperator;
import org.apache.sysml.runtime.matrix.operators.LeftScalarOperator;
import org.apache.sysml.runtime.matrix.operators.RightScalarOperator;
import org.apache.sysml.runtime.matrix.operators.ScalarOperator;
import org.apache.sysml.utils.GPUStatistics;
import org.apache.sysml.utils.Statistics;

import static jcuda.jcublas.cublasOperation.CUBLAS_OP_N;
import static jcuda.jcublas.cublasOperation.CUBLAS_OP_T;
import static jcuda.jcudnn.JCudnn.cudnnActivationForward;
import static jcuda.jcudnn.JCudnn.cudnnConvolutionBackwardData;
import static jcuda.jcudnn.JCudnn.cudnnConvolutionBackwardFilter;
import static jcuda.jcudnn.JCudnn.cudnnConvolutionForward;
import static jcuda.jcudnn.JCudnn.cudnnCreateActivationDescriptor;
import static jcuda.jcudnn.JCudnn.cudnnCreateConvolutionDescriptor;
import static jcuda.jcudnn.JCudnn.cudnnCreateFilterDescriptor;
import static jcuda.jcudnn.JCudnn.cudnnCreatePoolingDescriptor;
import static jcuda.jcudnn.JCudnn.cudnnCreateTensorDescriptor;
import static jcuda.jcudnn.JCudnn.cudnnDestroyConvolutionDescriptor;
import static jcuda.jcudnn.JCudnn.cudnnDestroyFilterDescriptor;
import static jcuda.jcudnn.JCudnn.cudnnDestroyPoolingDescriptor;
import static jcuda.jcudnn.JCudnn.cudnnGetConvolutionBackwardDataWorkspaceSize;
import static jcuda.jcudnn.JCudnn.cudnnGetConvolutionBackwardFilterWorkspaceSize;
import static jcuda.jcudnn.JCudnn.cudnnGetConvolutionForwardWorkspaceSize;
import static jcuda.jcudnn.JCudnn.cudnnPoolingBackward;
import static jcuda.jcudnn.JCudnn.cudnnPoolingForward;
import static jcuda.jcudnn.JCudnn.cudnnSetActivationDescriptor;
import static jcuda.jcudnn.JCudnn.cudnnSetConvolution2dDescriptor;
import static jcuda.jcudnn.JCudnn.cudnnSetFilter4dDescriptor;
import static jcuda.jcudnn.JCudnn.cudnnSetPooling2dDescriptor;
import static jcuda.jcudnn.JCudnn.cudnnSetTensor4dDescriptor;
import static jcuda.jcudnn.cudnnActivationMode.CUDNN_ACTIVATION_RELU;
import static jcuda.jcudnn.cudnnConvolutionMode.CUDNN_CROSS_CORRELATION;
import static jcuda.jcudnn.cudnnDataType.CUDNN_DATA_DOUBLE;
import static jcuda.jcudnn.cudnnNanPropagation.CUDNN_PROPAGATE_NAN;
import static jcuda.jcudnn.cudnnPoolingMode.CUDNN_POOLING_MAX;
import static jcuda.jcudnn.cudnnTensorFormat.CUDNN_TENSOR_NCHW;
import static jcuda.jcusparse.JCusparse.cusparseDcsrgemm;
import static jcuda.jcusparse.JCusparse.cusparseDcsrmv;
import static jcuda.jcusparse.cusparseOperation.CUSPARSE_OPERATION_NON_TRANSPOSE;
import static jcuda.jcusparse.cusparseOperation.CUSPARSE_OPERATION_TRANSPOSE;
import static jcuda.runtime.JCuda.cudaDeviceSynchronize;
import static jcuda.runtime.JCuda.cudaMemcpy;
import static jcuda.runtime.cudaMemcpyKind.cudaMemcpyDeviceToDevice;
import static jcuda.runtime.cudaMemcpyKind.cudaMemcpyDeviceToHost;
import static jcuda.runtime.cudaMemcpyKind.cudaMemcpyHostToDevice;
import static org.apache.sysml.runtime.instructions.gpu.context.JCudaObject.allocate;
import static org.apache.sysml.runtime.instructions.gpu.context.JCudaObject.cudaFreeHelper;
import jcuda.jcudnn.cudnnBatchNormMode;
import jcuda.jcudnn.cudnnStatus;
import static jcuda.jcudnn.JCudnn.cudnnBatchNormalizationForwardInference;
import static jcuda.jcudnn.JCudnn.cudnnBatchNormalizationForwardTraining;
import static jcuda.jcudnn.JCudnn.cudnnBatchNormalizationBackward;

//FIXME move could to respective instructions, this is not a block library
public class LibMatrixCUDA {

	// Assume Compute Capability 3.0
	// MAX BLOCKS is 2^31 - 1 For compute capability > 3.0
	// MAX_THREADS is 1024 For compute capability > 3.0
	private static int _MAX_THREADS = -1;
	private static int _MAX_BLOCKS  = -1;
	private static int _WARP_SIZE 	= -1;

	/**
	 * Utility function to get maximum number of threads supported by the active CUDA device.
	 * This is put into a singleton style method because the GPUContext is not fully initialized when
	 * the LibMatrixCUDA class is loaded. If the {@link GPUContext#getGPUContext()} is invoked in a
	 * static block in this class, it will access the {@link JCudaContext} instance when it is not
	 * properly initialized. Due to the proper checks in place, a deadlock occurs.
	 * @return max threads
	 * @throws DMLRuntimeException if exception occurs
	 */
	static int getMaxThreads() throws DMLRuntimeException{
		if (_MAX_THREADS == -1){
			_MAX_THREADS = JCudaContext.getMaxThreadsPerBlock();
		}
		return _MAX_THREADS;
	}

	/**
	 * Utility function to get maximum number of blocks supported by the active CUDA device.
	 * This is put into a singleton style method because the GPUContext is not fully initialized when
	 * the LibMatrixCUDA class is loaded. If the {@link GPUContext#getGPUContext()} is invoked in a
	 * static block in this class, it will access the {@link JCudaContext} instance when it is not
	 * properly initialized. Due to the proper checks in place, a deadlock occurs.
	 * @return max blocks
	 * @throws DMLRuntimeException if exception occurs
	 */
	static int getMaxBlocks() throws DMLRuntimeException{
		if (_MAX_BLOCKS == -1){
			_MAX_BLOCKS = JCudaContext.getMaxBlocks();
		}
		return _MAX_BLOCKS;
	}

	/**
	 * Utility function to get the warp size supported by the active CUDA device.
	 * This is put into a singleton style method because the GPUContext is not fully initialized when
	 * the LibMatrixCUDA class is loaded. If the {@link GPUContext#getGPUContext()} is invoked in a
	 * static block in this class, it will access the {@link JCudaContext} instance when it is not
	 * properly initialized. Due to the proper checks in place, a deadlock occurs.
	 * @return warp size
	 * @throws DMLRuntimeException if exception occurs
	 */
	static int getWarpSize() throws DMLRuntimeException {
		if (_WARP_SIZE == -1) {
			_WARP_SIZE = JCudaContext.getWarpSize();
		}
		return _WARP_SIZE;
	}

	public static boolean isInSparseFormat(MatrixObject mo) {
		if(mo.getGPUObject() != null && mo.getGPUObject().isAllocated())
			return mo.getGPUObject().isInSparseFormat();
		return MatrixBlock.evalSparseFormatInMemory(mo.getNumRows(), mo.getNumColumns(), mo.getNnz());
	}


	//********************************************************************/
	//***************** DEEP LEARNING Operators **************************/
	//********************************************************************/


	public static cudnnHandle cudnnHandle;
	public static cublasHandle cublasHandle;
	public static cusparseHandle cusparseHandle;
	public static JCudaKernels kernels; // Used to launch custom kernels

	private static final Log LOG = LogFactory.getLog(LibMatrixCUDA.class.getName());

	private static int CONVOLUTION_PREFERENCE = cudnnConvolutionFwdPreference.CUDNN_CONVOLUTION_FWD_NO_WORKSPACE;
	
	private static Pointer _one; 
	private static Pointer _zero;
	/**
	 * Convenience method to get a pointer to value '1.0' on device. Instead of allocating and deallocating it for every kernel invocation.
	 * @return jcuda pointer
	 */
	private static Pointer one() {
		if(_one == null) {
			_one = pointerTo(1.0);
		}
		return _one;
	}
	/**
	 * Convenience method to get a pointer to value '0.0f' on device. Instead of allocating and deallocating it for every kernel invocation.
	 * @return jcuda pointer
	 */
	private static Pointer zero() {
		if(_zero == null) {
			_zero = pointerTo(0.0f);
		}
		return _zero;
	}
	
	/**
	 * Convenience method to get tensor descriptor from underlying JCudaObject
	 * @param mat matrix object
	 * @param N number of images
	 * @param C number of channels
	 * @param H height
	 * @param W width
	 * @return cudnn tensor descriptor
	 * @throws DMLRuntimeException if the input descriptor and matrix dimensions don't match
	 */
	private static cudnnTensorDescriptor allocateTensorDescriptor(MatrixObject mat, int N, int C, int H, int W) throws DMLRuntimeException {
		if(mat.getNumRows() != N || mat.getNumColumns() != C*H*W) {
			throw new DMLRuntimeException("Mismatch descriptor-matrix dimensions:" + mat.getNumRows() + " != " + N 
					+ " || " + mat.getNumColumns() + " != " + (C*H*W));
		}
		return ((JCudaObject)mat.getGPUObject()).allocateTensorDescriptor(N, C, H, W);
	}
	
	/**
	 * Convenience method to get jcudaDenseMatrixPtr. This method explicitly converts sparse to dense format, so use it judiciously.
	 * @param image input matrix object
	 * @param isForCuDNN true if the dense pointer is to be used by a CuDNN kernel
	 * @return jcuda pointer
	 * @throws DMLRuntimeException if error occurs while sparse to dense conversion
	 */
	private static Pointer getDensePointer(MatrixObject image, boolean isForCuDNN, String instName) throws DMLRuntimeException {
		if(isForCuDNN && image.getNumRows()*image.getNumColumns() > numDoublesIn2GB) {
			throw new DMLRuntimeException("CuDNN restriction: the size of input tensor cannot be greater than 2GB. Hint: try reducing the mini-batch size.");
		}
		return getDensePointer(image, instName);
	}
	
	/**
	 * Convenience method to get jcudaDenseMatrixPtr. This method explicitly converts sparse to dense format, so use it judiciously.
	 * @param image input matrix object
	 * @return jcuda pointer
	 * @throws DMLRuntimeException if error occurs while sparse to dense conversion
	 */
	private static Pointer getDensePointer(MatrixObject image, String instName) throws DMLRuntimeException {
		if(isInSparseFormat(image)) {
			((JCudaObject)image.getGPUObject()).sparseToDense(instName);
		}
		return ((JCudaObject)image.getGPUObject()).jcudaDenseMatrixPtr;
	}
	
	/**
	 * Convenience method for checking the status of CuDNN kernel.
	 * 
	 * @param status status returned by CuDNN
	 * @throws DMLRuntimeException if status is not CUDNN_STATUS_SUCCESS
	 */
	private static void checkStatus(int status) throws DMLRuntimeException {
		if(status != cudnnStatus.CUDNN_STATUS_SUCCESS) 
			throw new DMLRuntimeException("Error status returned by CuDNN:" + jcuda.jcudnn.cudnnStatus.stringFor(status));
	}

	public static void conv2dBiasAdd(String instName, MatrixObject image, MatrixObject bias, MatrixObject filter, MatrixObject outputBlock, int N, int C, int H, int W,
			int K, int R, int S, int pad_h, int pad_w, int stride_h, int stride_w, int P, int Q)
					throws DMLRuntimeException {
		conv2d(instName, image, filter, outputBlock, N, C, H, W, K, R, S, pad_h, pad_w, stride_h, stride_w, P, Q);
		biasAdd(instName, outputBlock, bias, outputBlock);
	}
	
	public static void conv2d(String instName, MatrixObject image, MatrixObject filter, MatrixObject outputBlock, int N, int C, int H, int W,
														int K, int R, int S, int pad_h, int pad_w, int stride_h, int stride_w, int P, int Q)
					throws DMLRuntimeException {
		cudnnFilterDescriptor filterDesc = null;
		cudnnConvolutionDescriptor convDesc = null;
		Pointer workSpace = null;
		long sizeInBytes = 0;
		try {
			long t1=0, t2=0;
			// Allocate descriptors
			if (GPUStatistics.DISPLAY_STATISTICS) t1 = System.nanoTime();
			cudnnTensorDescriptor srcTensorDesc = allocateTensorDescriptor(image, N, C, H, W);
			cudnnTensorDescriptor dstTensorDesc = allocateTensorDescriptor(outputBlock, N, K, P, Q);
			filterDesc = allocateFilterDescriptor(K, C, R, S);

			Pointer imagePointer = getDensePointer(image, true, instName);
			Pointer filterPointer = getDensePointer(filter, true, instName);
			Pointer dstPointer = getDensePointer(outputBlock, true, instName);

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
					workSpace = allocate(sizeInBytesArray[0]);
				sizeInBytes = sizeInBytesArray[0];
			}
			else if(CONVOLUTION_PREFERENCE == cudnnConvolutionFwdPreference.CUDNN_CONVOLUTION_FWD_SPECIFY_WORKSPACE_LIMIT) {
				throw new DMLRuntimeException("CUDNN_CONVOLUTION_FWD_SPECIFY_WORKSPACE_LIMIT is not implemented");
			}
			else {
				throw new DMLRuntimeException("Unsupported preference criteria for convolution");
			}
			if (GPUStatistics.DISPLAY_STATISTICS) GPUStatistics.maintainCPMiscTimes(instName, GPUInstruction.MISC_TIMER_CUDNN_INIT, System.nanoTime() - t1);
			if (GPUStatistics.DISPLAY_STATISTICS) t2 = System.nanoTime();
			int status = cudnnConvolutionForward(cudnnHandle, one(),
							srcTensorDesc, imagePointer,
							filterDesc, filterPointer,
							convDesc, algo, workSpace, sizeInBytes, zero(),
							dstTensorDesc, dstPointer);
			if (GPUStatistics.DISPLAY_STATISTICS) GPUStatistics.maintainCPMiscTimes(instName, GPUInstruction.MISC_TIMER_CONVOLUTION_FORWARD_LIB, System.nanoTime() - t2);
			if(status != jcuda.jcudnn.cudnnStatus.CUDNN_STATUS_SUCCESS) {
				throw new DMLRuntimeException("Could not executed cudnnConvolutionForward: " + jcuda.jcudnn.cudnnStatus.stringFor(status));
			}
		}
		finally {
			long t3=0;
			if (GPUStatistics.DISPLAY_STATISTICS) t3 = System.nanoTime();
			if(filterDesc != null)
				cudnnDestroyFilterDescriptor(filterDesc);
			if(convDesc != null)
				cudnnDestroyConvolutionDescriptor(convDesc);
			if(workSpace != null && sizeInBytes != 0)
				cudaFreeHelper(instName, workSpace);
			if (GPUStatistics.DISPLAY_STATISTICS) GPUStatistics.maintainCPMiscTimes(instName, GPUInstruction.MISC_TIMER_CUDNN_CLEANUP, System.nanoTime() - t3);
		}
	}

	private static cudnnConvolutionDescriptor allocateConvolutionDescriptor(int padding [], int strides []) {
		cudnnConvolutionDescriptor convDesc = new cudnnConvolutionDescriptor();
		cudnnCreateConvolutionDescriptor(convDesc);
		cudnnSetConvolution2dDescriptor(convDesc, padding[0], padding[1], strides[0], strides[1], 1, 1, CUDNN_CROSS_CORRELATION);
		return convDesc;
	}

	public static  Pointer pointerTo(double value) {
		return Pointer.to(new double[] { value });
	}

	private static cudnnFilterDescriptor allocateFilterDescriptor(int K, int C, int R, int S) {
		cudnnFilterDescriptor filterDesc = new cudnnFilterDescriptor();
		cudnnCreateFilterDescriptor(filterDesc);
		cudnnSetFilter4dDescriptor(filterDesc, CUDNN_DATA_DOUBLE, CUDNN_TENSOR_NCHW, K, C, R, S);
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
	 * @return cudnn pooling descriptor
	 */
	private static cudnnPoolingDescriptor allocatePoolingDescriptor(int R, int S, int pad_h, int pad_w, int stride_h, int stride_w) {
		cudnnPoolingDescriptor poolingDesc = new cudnnPoolingDescriptor();
		cudnnCreatePoolingDescriptor(poolingDesc);
		cudnnSetPooling2dDescriptor(poolingDesc, CUDNN_POOLING_MAX, CUDNN_PROPAGATE_NAN, R, S, pad_h, pad_w, stride_h, stride_w);
		return poolingDesc;
	}

	/**
	 * This method computes the backpropagation errors for previous layer of relu operation
	 * @param instName the invoking instruction's name for record {@link Statistics}.
	 * @param input input image
	 * @param dout  next layer error propogation
	 * @param outputBlock output
	 * @throws DMLRuntimeException if DMLRuntimeException occurs
	 */
	public static void reluBackward(String instName, MatrixObject input, MatrixObject dout, MatrixObject outputBlock) throws DMLRuntimeException {
		long rows = input.getNumRows();
		long cols = input.getNumColumns();
		Pointer imagePointer = getDensePointer(input, instName);
		Pointer doutPointer = getDensePointer(dout, instName);
		Pointer outputPointer = getDensePointer(outputBlock, instName);

		long t1=0;
		if (GPUStatistics.DISPLAY_STATISTICS) t1 = System.nanoTime();
		kernels.launchKernel("relu_backward",
						ExecutionConfig.getConfigForSimpleMatrixOperations((int)rows, (int)cols),
						imagePointer, doutPointer, outputPointer, (int)rows, (int)cols);
		if (GPUStatistics.DISPLAY_STATISTICS) GPUStatistics.maintainCPMiscTimes(instName, GPUInstruction.MISC_TIMER_BIAS_ADD_LIB, System.nanoTime() - t1);

	}

	/**
	 * Performs the operation corresponding to the DML script:
	 * ones = matrix(1, rows=1, cols=Hout*Wout)
	 * output = input + matrix(bias %*% ones, rows=1, cols=F*Hout*Wout)
	 * This operation is often followed by conv2d and hence we have introduced bias_add(input, bias) built-in function
	 * @param instName the invoking instruction's name for record {@link Statistics}.
	 * @param input input image
	 * @param bias bias
	 * @param outputBlock output
	 * @throws DMLRuntimeException if DMLRuntimeException occurs
	 */
	public static void biasAdd(String instName, MatrixObject input, MatrixObject bias, MatrixObject outputBlock) throws DMLRuntimeException {
		long rows = input.getNumRows();
		long cols = input.getNumColumns();
		long K = bias.getNumRows();
		long PQ = cols / K;
		if(bias.getNumColumns() != 1 || cols % K != 0) {
			throw new DMLRuntimeException("Incorrect inputs for bias_add: input[" + rows + " X " + cols + "] and bias[" + K + " X " + bias.getNumColumns() + "]");
		}
		Pointer imagePointer = getDensePointer(input, instName);
		Pointer biasPointer = getDensePointer(bias, instName);
		Pointer outputPointer = getDensePointer(outputBlock, instName);
		long t1 = 0;
		if (GPUStatistics.DISPLAY_STATISTICS) t1 = System.nanoTime();
		kernels.launchKernel("bias_add",
						ExecutionConfig.getConfigForSimpleMatrixOperations((int)rows, (int)cols),
						imagePointer, biasPointer, outputPointer, (int)rows, (int)cols, (int) PQ);
		if (GPUStatistics.DISPLAY_STATISTICS) GPUStatistics.maintainCPMiscTimes(instName, GPUInstruction.MISC_TIMER_RELU_BACKWARD_KERNEL, System.nanoTime() - t1);

	}
	
	private static void validateBatchNormalizationDimensions(MatrixObject scale, MatrixObject bias, MatrixObject runningMean, MatrixObject runningVar, int C) throws DMLRuntimeException {
		if(scale.getNumRows() != 1 || scale.getNumColumns() != C) {
			throw new DMLRuntimeException("Incorrect dimensions for scale");
		}
		if(bias.getNumRows() != 1 || bias.getNumColumns() != C) {
			throw new DMLRuntimeException("Incorrect dimensions for bias");
		}
		if(runningMean.getNumRows() != 1 || runningMean.getNumColumns() != C) {
			throw new DMLRuntimeException("Incorrect dimensions for running mean");
		}
		if(runningVar.getNumRows() != 1 || runningVar.getNumColumns() != C) {
			throw new DMLRuntimeException("Incorrect dimensions for running variance");
		}
	}
	
	/**
	 * Performs the forward BatchNormalization layer computation for inference
	 * 
	 * @param instName name of the instruction
	 * @param image input image
	 * @param scale scale (as per CuDNN) and gamma as per original paper: shape [1, C, 1, 1]
	 * @param bias bias (as per CuDNN) and beta as per original paper: shape [1, C, 1, 1]
	 * @param runningMean running mean accumulated during training phase: shape [1, C, 1, 1]
	 * @param runningVar running variance accumulated during training phase: shape [1, C, 1, 1]
	 * @param ret normalized input
	 * @param epsilon epsilon value used in the batch normalization formula
	 * @throws DMLRuntimeException if error occurs
	 */
	public static void batchNormalizationForwardInference(String instName, MatrixObject image, 
			MatrixObject scale, MatrixObject bias, MatrixObject runningMean, MatrixObject runningVar, 
			MatrixObject ret, double epsilon) throws DMLRuntimeException {
		int mode = cudnnBatchNormMode.CUDNN_BATCHNORM_SPATIAL;
		
		int N = (int) image.getNumRows();
		int C = (int) scale.getNumColumns();
		long CHW = image.getNumColumns();
		validateBatchNormalizationDimensions(scale, bias, runningMean, runningVar, C);
		
		// Allocate descriptors
		cudnnTensorDescriptor nCHWDescriptor = allocateNCHWDescriptors(N, C, CHW, 
				new MatrixObject[] {image},  new MatrixObject[] {ret});
		cudnnTensorDescriptor scaleTensorDesc = allocateTensorDescriptor(scale, 1, C, 1, 1);
		
		// Get underlying dense pointer
		Pointer imagePtr = getDensePointer(image, true, instName);
		Pointer retPtr = getDensePointer(ret, true, instName);
		Pointer biasPtr = getDensePointer(bias, true, instName);
		Pointer scalePtr = getDensePointer(scale, true, instName);
		Pointer runningMeanPtr = getDensePointer(runningMean, true, instName);
		Pointer runningVarPtr = getDensePointer(runningVar, true, instName);
		
		checkStatus(cudnnBatchNormalizationForwardInference(cudnnHandle, mode, one(), zero(),
				nCHWDescriptor, imagePtr, nCHWDescriptor, retPtr,
			scaleTensorDesc, scalePtr, biasPtr,
			runningMeanPtr, runningVarPtr, epsilon));
	}
	
	/**
	 * Performs the forward BatchNormalization layer computation for training
	 * 
	 * @param instName name of the instruction
	 * @param image input image
	 * @param scale scale (as per CuDNN) and gamma as per original paper: shape [1, C, 1, 1]
	 * @param bias bias (as per CuDNN) and beta as per original paper: shape [1, C, 1, 1]
	 * @param runningMean running mean accumulated during training phase: shape [1, C, 1, 1]
	 * @param runningVar running variance accumulated during training phase: shape [1, C, 1, 1]
	 * @param ret (output) normalized input
	 * @param retRunningMean (output) running mean accumulated during training phase: shape [1, C, 1, 1]
	 * @param retRunningVar (output) running variance accumulated during training phase: shape [1, C, 1, 1]
	 * @param epsilon epsilon value used in the batch normalization formula
	 * @param exponentialAverageFactor factor used in the moving average computation
	 * @throws DMLRuntimeException if error occurs
	 */
	public static void batchNormalizationForwardTraining(String instName, MatrixObject image, 
			MatrixObject scale,  MatrixObject bias, MatrixObject runningMean, MatrixObject runningVar, 
			MatrixObject ret, MatrixObject retRunningMean, MatrixObject retRunningVar, double epsilon, double exponentialAverageFactor) throws DMLRuntimeException {
		int mode = cudnnBatchNormMode.CUDNN_BATCHNORM_SPATIAL;
		
		int N = (int) image.getNumRows();
		int C = (int) scale.getNumColumns();
		long CHW = image.getNumColumns();
		validateBatchNormalizationDimensions(scale, bias, runningMean, runningVar, C);
		
		// Allocate descriptors
		cudnnTensorDescriptor nCHWDescriptor = allocateNCHWDescriptors(N, C, CHW, 
				new MatrixObject[] {image},  new MatrixObject[] {ret});
		cudnnTensorDescriptor scaleTensorDesc = allocateTensorDescriptor(scale, 1, C, 1, 1);
		
		// Get underlying dense pointer
		Pointer imagePtr = getDensePointer(image, true, instName);
		Pointer retPtr = getDensePointer(ret, true, instName);
		Pointer biasPtr = getDensePointer(bias, true, instName);
		Pointer scalePtr = getDensePointer(scale, true, instName);
		Pointer runningMeanPtr = getDensePointer(runningMean, true, instName);
		Pointer runningVarPtr = getDensePointer(runningVar, true, instName);
		
		// To allow for copy-on-write
		Pointer retRunningMeanPtr = getDensePointer(retRunningMean, true, instName);
		Pointer retRunningVarPtr = getDensePointer(retRunningVar, true, instName);
		cudaMemcpy(retRunningMeanPtr, runningMeanPtr, C * Sizeof.DOUBLE, cudaMemcpyDeviceToDevice);
		cudaMemcpy(retRunningVarPtr, runningVarPtr, C * Sizeof.DOUBLE, cudaMemcpyDeviceToDevice);
		
		// ignoring resultSaveMean and resultSaveVariance as it requires state management
		checkStatus(cudnnBatchNormalizationForwardTraining(cudnnHandle, mode, one(), zero(),
				nCHWDescriptor, imagePtr, nCHWDescriptor, retPtr,
			scaleTensorDesc, scalePtr, biasPtr, exponentialAverageFactor,
			retRunningMeanPtr, retRunningVarPtr, epsilon, new Pointer(), new Pointer()));
	}
	
	/**
	 * Convenient utility for batch normalization that returns a NCHW descriptor
	 * 
	 * @param N number of images
	 * @param C number of channels
	 * @param CHW channels*height*width
	 * @param input input matrix objects
	 * @param output output matrix objects
	 * @return one of the NCHW descriptor
	 * @throws DMLRuntimeException if error occurs
	 */
	private static cudnnTensorDescriptor allocateNCHWDescriptors(int N, int C, long CHW, MatrixObject [] input, MatrixObject [] output) throws DMLRuntimeException {
		cudnnTensorDescriptor ret  = null; // Return any one
		if(CHW > ((long)Integer.MAX_VALUE)*C) {
			throw new DMLRuntimeException("image size (height*width) should be less than " + Integer.MAX_VALUE);
		}
		cudnnTensorDescriptor knownNCHWdescriptor = null;
		int H = -1; int W = -1;
		for(int i = 0; i < input.length; i++) {
			knownNCHWdescriptor = ((JCudaObject)input[i].getGPUObject()).getTensorDescriptor();
			if(knownNCHWdescriptor != null) {
				int [] shape = ((JCudaObject)input[i].getGPUObject()).getTensorShape();
				if(shape[0] != N || shape[1] != C) {
					throw new DMLRuntimeException("Incorrect N and C:" + shape[0]  + " != " + N + " || " + shape[1]  + " != " +  C);
				}
				H = shape[2];
				W = shape[3];
				break;
			}
		}
		if(knownNCHWdescriptor != null) {
			// We precisely know N, C, H, W
			for(int i = 0; i < input.length; i++) {
				ret = allocateTensorDescriptor(input[i], N, C, H, W);
			}
			for(int i = 0; i < output.length; i++) {
				ret = allocateTensorDescriptor(output[i], N, C, H, W);
			}
		}
		else {
			int HW = (int) (CHW / C);
			H = HW; W = 1; // If not known
			double potentialH = Math.sqrt(HW);
			if(potentialH == ((int) potentialH)) {
				H = (int) potentialH; 
				W = H; 
			}
			// We are not sure about H and W, hence don't allocate them.
			ret = new cudnnTensorDescriptor();
			cudnnCreateTensorDescriptor(ret);
			cudnnSetTensor4dDescriptor(ret, CUDNN_TENSOR_NCHW, CUDNN_DATA_DOUBLE, N, C, H, W);
		}
		return ret;
	}
	
	/**
	 * This method computes the backpropagation errors for image, scale and bias of batch normalization layer
	 * 
	 * @param instName name of the instruction
	 * @param image input image
	 * @param dout input errors of shape C, H, W
	 * @param scale scale (as per CuDNN) and gamma as per original paper: shape [1, C, 1, 1]
	 * @param ret (output) backpropagation errors for previous layer
	 * @param retScale backpropagation error for scale
	 * @param retBias backpropagation error for bias
	 * @param epsilon epsilon value used in the batch normalization formula
	 * @throws DMLRuntimeException if error occurs
	 */
	public static void batchNormalizationBackward(String instName, MatrixObject image, MatrixObject dout, 
			MatrixObject scale, MatrixObject ret, MatrixObject retScale, MatrixObject retBias,
			double epsilon) throws DMLRuntimeException {
		int mode = cudnnBatchNormMode.CUDNN_BATCHNORM_SPATIAL;
		
		int N = (int) image.getNumRows();
		int C = (int) scale.getNumColumns();
		long CHW = image.getNumColumns();
		
		// Allocate descriptors
		cudnnTensorDescriptor nCHWDescriptor = allocateNCHWDescriptors(N, C, CHW, 
				new MatrixObject[] {image, dout},  new MatrixObject[] {ret});
		cudnnTensorDescriptor scaleTensorDesc = allocateTensorDescriptor(scale, 1, C, 1, 1);
		
		// Get underlying dense pointer
		Pointer imagePtr = getDensePointer(image, true, instName);
		Pointer doutPtr = getDensePointer(dout, true, instName);
		Pointer scalePtr = getDensePointer(scale, true, instName);
		Pointer retPtr = getDensePointer(ret, true, instName);
		Pointer retScalePtr = getDensePointer(retScale, true, instName);
		Pointer retBiasPtr = getDensePointer(retBias, true, instName);
		
		// ignoring resultSaveMean and resultSaveVariance as it requires state management
		checkStatus(cudnnBatchNormalizationBackward(cudnnHandle, mode,  one(), zero(), one(), zero(),
				nCHWDescriptor,  imagePtr, nCHWDescriptor, doutPtr, nCHWDescriptor, retPtr,
				scaleTensorDesc, scalePtr, retScalePtr, retBiasPtr, epsilon, new Pointer(), new Pointer()));
	}


	/**
	 * This method computes the backpropogation errors for filter of convolution operation
	 * 
	 * @param instName the invoking instruction's name for record {@link Statistics}.
	 * @param image input image
	 * @param dout errors from next layer
	 * @param outputBlock  output errors
	 * @param N number of images
	 * @param C number of channels
	 * @param H height
	 * @param W width
	 * @param K number of filters
	 * @param R filter height
	 * @param S filter width
	 * @param pad_h pad height
	 * @param pad_w pad width
	 * @param stride_h stride height
	 * @param stride_w stride width
	 * @param P output activation height
	 * @param Q output activation width
	 * @throws DMLRuntimeException if DMLRuntimeException occurs
	 */
	public static void conv2dBackwardFilter(String instName, MatrixObject image, MatrixObject dout,
																					MatrixObject outputBlock, int N, int C, int H, int W, int K, int R,
																					int S, int pad_h, int pad_w, int stride_h, int stride_w, int P,
																					int Q) throws DMLRuntimeException {
		cudnnFilterDescriptor dwDesc = null;
		cudnnConvolutionDescriptor convDesc = null;

		Pointer workSpace = null;
		long sizeInBytes = 0;
		try {

			long t1=0, t2=0;
			if (GPUStatistics.DISPLAY_STATISTICS) t1 = System.nanoTime();
			// Allocate descriptors
			cudnnTensorDescriptor xTensorDesc = allocateTensorDescriptor(image, N, C, H, W);
			cudnnTensorDescriptor doutTensorDesc = allocateTensorDescriptor(dout, N, K, P, Q);
			dwDesc = allocateFilterDescriptor(K, C, R, S);

			// Allocate data
			Pointer imagePointer = getDensePointer(image, true, instName);
			Pointer doutPointer = getDensePointer(dout, true, instName);
			Pointer dwPointer = getDensePointer(outputBlock, true, instName);
			int padding [] = { pad_h, pad_w };
			int strides [] = { stride_h, stride_w };
			convDesc = allocateConvolutionDescriptor(padding, strides);
			long sizeInBytesArray[] = { 0 };

			// TODO: Select the best algorithm depending on the data and supported CUDA
			int algo = jcuda.jcudnn.cudnnConvolutionBwdFilterAlgo.CUDNN_CONVOLUTION_BWD_FILTER_ALGO_0;

			workSpace = new Pointer();
			cudnnGetConvolutionBackwardFilterWorkspaceSize(cudnnHandle,
							xTensorDesc, doutTensorDesc, convDesc, dwDesc, algo, sizeInBytesArray);
			if (GPUStatistics.DISPLAY_STATISTICS)GPUStatistics.maintainCPMiscTimes(instName, GPUInstruction.MISC_TIMER_CUDNN_INIT, System.nanoTime() - t1);

			if (GPUStatistics.DISPLAY_STATISTICS) t2 = System.nanoTime();
			int status = cudnnConvolutionBackwardFilter(cudnnHandle, one(), xTensorDesc, imagePointer,
							doutTensorDesc, doutPointer, convDesc, algo, workSpace, sizeInBytes, zero(), dwDesc, dwPointer);
			if (GPUStatistics.DISPLAY_STATISTICS)GPUStatistics.maintainCPMiscTimes(instName, GPUInstruction.MISC_TIMER_CONVOLUTION_BACKWARD_FILTER_LIB, System.nanoTime() - t2);

			if(status != jcuda.jcudnn.cudnnStatus.CUDNN_STATUS_SUCCESS) {
				throw new DMLRuntimeException("Could not executed cudnnConvolutionBackwardFilter: " + jcuda.jcudnn.cudnnStatus.stringFor(status));
			}
		}
		finally {
			long t3=0;
			if (GPUStatistics.DISPLAY_STATISTICS) t3 = System.nanoTime();

			if(workSpace != null && sizeInBytes != 0)
				cudaFreeHelper(instName, workSpace);
			if(dwDesc != null)
				cudnnDestroyFilterDescriptor(dwDesc);

			if(convDesc != null)
				cudnnDestroyConvolutionDescriptor(convDesc);
			if (GPUStatistics.DISPLAY_STATISTICS) GPUStatistics.maintainCPMiscTimes(instName, GPUInstruction.MISC_TIMER_CUDNN_CLEANUP, System.nanoTime() - t3);
		}
	}

	private static long numDoublesIn2GB = 125000000;

	/**
	 * This method computes the backpropogation errors for previous layer of convolution operation
	 * @param instName the invoking instruction's name for record {@link Statistics}.
	 * @param filter filter used in conv2d
	 * @param dout errors from next layer
	 * @param output  output errors
	 * @param N number of images
	 * @param C number of channels
	 * @param H height
	 * @param W width
	 * @param K number of filters
	 * @param R filter height
	 * @param S filter width
	 * @param pad_h pad height
	 * @param pad_w pad width
	 * @param stride_h stride height
	 * @param stride_w stride width
	 * @param P output activation height
	 * @param Q output activation width
	 * @throws DMLRuntimeException if DMLRuntimeException occurs
	 */
	public static void conv2dBackwardData(String instName, MatrixObject filter, MatrixObject dout,
																				MatrixObject output, int N, int C, int H, int W, int K, int R,
																				int S, int pad_h, int pad_w, int stride_h, int stride_w, int P,
																				int Q) throws DMLRuntimeException {
		cudnnFilterDescriptor wDesc = null;
		cudnnConvolutionDescriptor convDesc = null;

		Pointer workSpace = null;
		long sizeInBytes = 0;
		try {
			long t1=0, t2=0;
			if (GPUStatistics.DISPLAY_STATISTICS) t1 = System.nanoTime();
			// Allocate descriptors
			wDesc = allocateFilterDescriptor(K, C, R, S);
			cudnnTensorDescriptor dyDesc = allocateTensorDescriptor(dout, N, K, P, Q);
			cudnnTensorDescriptor dxDesc = allocateTensorDescriptor(output, N, C, H, W);

			// Allocate data
			Pointer w = getDensePointer(filter, true, instName);
			Pointer dy = getDensePointer(dout, true, instName);
			Pointer dx = getDensePointer(output, true, instName);
			
			int padding [] = { pad_h, pad_w };
			int strides [] = { stride_h, stride_w };
			convDesc = allocateConvolutionDescriptor(padding, strides);
			long sizeInBytesArray[] = { 0 };

			// TODO: Select the best algorithm depending on the data and supported CUDA
			int algo = jcuda.jcudnn.cudnnConvolutionBwdDataAlgo.CUDNN_CONVOLUTION_BWD_DATA_ALGO_0;
			workSpace = new Pointer();
			cudnnGetConvolutionBackwardDataWorkspaceSize(cudnnHandle,
							wDesc, dyDesc, convDesc, dxDesc, algo, sizeInBytesArray);
			if (GPUStatistics.DISPLAY_STATISTICS) GPUStatistics.maintainCPMiscTimes(instName, GPUInstruction.MISC_TIMER_CUDNN_INIT, System.nanoTime() - t1);

			if (GPUStatistics.DISPLAY_STATISTICS) t2 = System.nanoTime();
			int status = cudnnConvolutionBackwardData(cudnnHandle, one(), wDesc, w,
							dyDesc, dy, convDesc, algo, workSpace, sizeInBytes, zero(), dxDesc, dx);
			if (GPUStatistics.DISPLAY_STATISTICS) GPUStatistics.maintainCPMiscTimes(instName, GPUInstruction.MISC_TIMER_CONVOLUTION_BACKWARD_DATA_LIB, System.nanoTime() - t2);

			if(status != jcuda.jcudnn.cudnnStatus.CUDNN_STATUS_SUCCESS) {
				throw new DMLRuntimeException("Could not executed cudnnConvolutionBackwardData: " + jcuda.jcudnn.cudnnStatus.stringFor(status));
			}
		}
		finally {
			long t3=0;
			if (GPUStatistics.DISPLAY_STATISTICS) t3 = System.nanoTime();

			if(workSpace != null && sizeInBytes != 0)
				cudaFreeHelper(instName, workSpace);
			if(wDesc != null)
				cudnnDestroyFilterDescriptor(wDesc);
			if(convDesc != null)
				cudnnDestroyConvolutionDescriptor(convDesc);

			if (GPUStatistics.DISPLAY_STATISTICS) GPUStatistics.maintainCPMiscTimes(instName, GPUInstruction.MISC_TIMER_CUDNN_CLEANUP, System.nanoTime() - t3);
		}
	}
	
	/**
	 * performs maxpooling on GPU by exploiting cudnnPoolingForward(...)
	 * @param instName the invoking instruction's name for record {@link Statistics}.
	 * @param image image as matrix object
	 * @param outputBlock output matrix
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
	 * @throws DMLRuntimeException if DMLRuntimeException occurs
	 */
	public static void maxpooling(String instName, MatrixObject image,
			MatrixObject outputBlock, int N, int C, int H, int W, int K, int R,
			int S, int pad_h, int pad_w, int stride_h, int stride_w, int P,
			int Q) throws DMLRuntimeException {
		Pointer x = getDensePointer(image, true, instName);
		cudnnTensorDescriptor xDesc = allocateTensorDescriptor(image, N, C, H, W);
		performMaxpooling(instName, x, xDesc, outputBlock, N, C, H, W, K, R, S, pad_h, pad_w, stride_h, stride_w, P, Q);
	}
	
	public static void performMaxpooling(String instName, Pointer x, cudnnTensorDescriptor xDesc,
			MatrixObject outputBlock, int N, int C, int H, int W, int K, int R,
			int S, int pad_h, int pad_w, int stride_h, int stride_w, int P,
			int Q) throws DMLRuntimeException {
		Pointer y = getDensePointer(outputBlock, true, instName);
		cudnnPoolingDescriptor poolingDesc = null;

		try {
			long t1=0,t2=0;
			if (GPUStatistics.DISPLAY_STATISTICS) t1 = System.nanoTime();
			// Allocate descriptors
			cudnnTensorDescriptor yDesc = allocateTensorDescriptor(outputBlock, N, C, P, Q);
			poolingDesc = allocatePoolingDescriptor(R, S, pad_h, pad_w, stride_h, stride_w);
			if (GPUStatistics.DISPLAY_STATISTICS) GPUStatistics.maintainCPMiscTimes(instName, GPUInstruction.MISC_TIMER_CUDNN_INIT, System.nanoTime() - t1);

			if (GPUStatistics.DISPLAY_STATISTICS) t2 = System.nanoTime();
			int status = cudnnPoolingForward(cudnnHandle, poolingDesc, one(), xDesc, x, zero(), yDesc, y);
			if (GPUStatistics.DISPLAY_STATISTICS) GPUStatistics.maintainCPMiscTimes(instName, GPUInstruction.MISC_TIMER_MAXPOOLING_FORWARD_LIB, System.nanoTime() - t2);

			if(status != jcuda.jcudnn.cudnnStatus.CUDNN_STATUS_SUCCESS) {
				throw new DMLRuntimeException("Could not executed cudnnPoolingForward: " + jcuda.jcudnn.cudnnStatus.stringFor(status));
			}
		}
		finally {
			long t3=0;
			if (GPUStatistics.DISPLAY_STATISTICS) t3 = System.nanoTime();
			if(poolingDesc != null)
				cudnnDestroyPoolingDescriptor(poolingDesc);
			if (GPUStatistics.DISPLAY_STATISTICS) GPUStatistics.maintainCPMiscTimes(instName, GPUInstruction.MISC_TIMER_CUDNN_CLEANUP, System.nanoTime() - t3);
		}
	}
	
	/**
	 * performs relu followed by maxpooling on GPU by exploiting cudnnPoolingForward(...)
	 * @param instName the invoking instruction's name for record {@link Statistics}.
	 * @param image image as matrix object
	 * @param outputBlock output matrix
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
	 * @throws DMLRuntimeException if DMLRuntimeException occurs
	 */
	public static void reluMaxpooling(String instName, MatrixObject image,
			MatrixObject outputBlock, int N, int C, int H, int W, int K, int R,
			int S, int pad_h, int pad_w, int stride_h, int stride_w, int P,
			int Q) throws DMLRuntimeException {
		cudnnTensorDescriptor srcTensorDesc = allocateTensorDescriptor(image, N, C, H, W);
		long size  = image.getNumRows() * image.getNumColumns() * Sizeof.DOUBLE;
		Pointer tmp = allocate(size);
		performCuDNNReLU(instName, image, tmp, srcTensorDesc);
		cudaDeviceSynchronize(); // It seemed like the cudnn operation in performCuDNNReLU was being done aysnchronously, this adds the neccesary sync
		performMaxpooling(instName, tmp, srcTensorDesc, outputBlock, N, C, H, W, K, R, S, pad_h, pad_w, stride_h, stride_w, P, Q);
		cudaFreeHelper(tmp);
	}

	/**
	 * Performs maxpoolingBackward on GPU by exploiting cudnnPoolingBackward(...)
	 * This method computes the backpropogation errors for previous layer of maxpooling operation
	 * @param instName the invoking instruction's name for record {@link Statistics}.
	 * @param image image as matrix object
	 * @param dout			delta matrix, output of previous layer
	 * @param outputBlock output matrix
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
	 * @throws DMLRuntimeException if DMLRuntimeException occurs
	 */
	public static void maxpoolingBackward(String instName, MatrixObject image, MatrixObject dout,
																				MatrixObject outputBlock, int N, int C, int H, int W, int K, int R,
																				int S, int pad_h, int pad_w, int stride_h, int stride_w, int P,
																				int Q) throws DMLRuntimeException {
		Pointer y = null;
		cudnnPoolingDescriptor poolingDesc = null;

		try {
			long t1=0, t2=0, t3=0;
			if (GPUStatistics.DISPLAY_STATISTICS) t1 = System.nanoTime();
			// Allocate descriptors
			cudnnTensorDescriptor xDesc = allocateTensorDescriptor(image, N, C, H, W);
			cudnnTensorDescriptor yDesc = allocateTensorDescriptor(dout, N, C, P, Q);
			cudnnTensorDescriptor dxDesc = allocateTensorDescriptor(outputBlock, N, C, H, W);
			cudnnTensorDescriptor dyDesc = allocateTensorDescriptor(dout, N, C, P, Q);

			poolingDesc = allocatePoolingDescriptor(R, S, pad_h, pad_w, stride_h, stride_w);

			// Calling PoolForward first, y is one of the inputs for poolBackward
			// TODO: Remove calling poolForward after necessary changes at language level for poolBackward
			long numBytes = N*C*P*Q*Sizeof.DOUBLE;
			y = allocate(numBytes);

			// Allocate data
			Pointer x = getDensePointer(image, true, instName);
			Pointer dx = getDensePointer(outputBlock, true, instName);
			Pointer dy = getDensePointer(dout, true, instName);

			if (GPUStatistics.DISPLAY_STATISTICS) GPUStatistics.maintainCPMiscTimes(instName, GPUInstruction.MISC_TIMER_CUDNN_INIT, System.nanoTime() - t1);

			if (GPUStatistics.DISPLAY_STATISTICS) t2 = System.nanoTime();
			int status = cudnnPoolingForward(cudnnHandle, poolingDesc, one(), xDesc, x, zero(), yDesc, y);
			if (GPUStatistics.DISPLAY_STATISTICS) GPUStatistics.maintainCPMiscTimes(instName, GPUInstruction.MISC_TIMER_MAXPOOLING_FORWARD_LIB, System.nanoTime() - t2);

			if(status != jcuda.jcudnn.cudnnStatus.CUDNN_STATUS_SUCCESS) {
				throw new DMLRuntimeException("Could not executed cudnnPoolingForward before cudnnPoolingBackward: " + jcuda.jcudnn.cudnnStatus.stringFor(status));
			}

			if (GPUStatistics.DISPLAY_STATISTICS) t3 = System.nanoTime();
			status = cudnnPoolingBackward(cudnnHandle, poolingDesc, one(), yDesc, y, dyDesc, dy, xDesc, x, zero(), dxDesc, dx);
			if (GPUStatistics.DISPLAY_STATISTICS) GPUStatistics.maintainCPMiscTimes(instName, GPUInstruction.MISC_TIMER_MAXPOOLING_BACKWARD_LIB, System.nanoTime() - t3);

			if(status != jcuda.jcudnn.cudnnStatus.CUDNN_STATUS_SUCCESS) {
				throw new DMLRuntimeException("Could not executed cudnnPoolingBackward: " + jcuda.jcudnn.cudnnStatus.stringFor(status));
			}
		}
		finally {
			long t4=0;
			if (GPUStatistics.DISPLAY_STATISTICS) t4 = System.nanoTime();

			if(y != null)
				cudaFreeHelper(instName, y);
			if(poolingDesc != null)
				cudnnDestroyPoolingDescriptor(poolingDesc);

			if (GPUStatistics.DISPLAY_STATISTICS) GPUStatistics.maintainCPMiscTimes(instName, GPUInstruction.MISC_TIMER_CUDNN_CLEANUP, System.nanoTime() - t4);
		}
	}
	
	private static void performCuDNNReLU(String instName, MatrixObject in, Pointer dstData, cudnnTensorDescriptor srcTensorDesc) throws DMLRuntimeException {
		long t0=0;
		try {
			cudnnTensorDescriptor dstTensorDesc = srcTensorDesc;
			
			Pointer srcData = getDensePointer(in, true, instName);
			cudnnActivationDescriptor activationDescriptor = new cudnnActivationDescriptor();
			cudnnCreateActivationDescriptor(activationDescriptor);
			double dummy = -1;
			cudnnSetActivationDescriptor(activationDescriptor, CUDNN_ACTIVATION_RELU, CUDNN_PROPAGATE_NAN, dummy);
			if (GPUStatistics.DISPLAY_STATISTICS) t0 = System.nanoTime();
			cudnnActivationForward(cudnnHandle, activationDescriptor,
							one(), srcTensorDesc, srcData,
							zero(), dstTensorDesc, dstData);
			if (GPUStatistics.DISPLAY_STATISTICS) GPUStatistics.maintainCPMiscTimes(instName, GPUInstruction.MISC_TIMER_ACTIVATION_FORWARD_LIB, System.nanoTime() - t0);
		}
		finally {
			long t1=0;
			if (GPUStatistics.DISPLAY_STATISTICS) t1 = System.nanoTime();
			if (GPUStatistics.DISPLAY_STATISTICS) GPUStatistics.maintainCPMiscTimes(instName, GPUInstruction.MISC_TIMER_CUDNN_CLEANUP, System.nanoTime() - t1);
		}
	}
	

	/**
	 * Performs the relu operation on the GPU.
	 * @param ec currently active {@link ExecutionContext}
	 * @param instName the invoking instruction's name for record {@link Statistics}.
	 * @param in input matrix
	 * @param outputName	name of the output matrix
	 * @throws DMLRuntimeException	if an error occurs
	 */
	public static void relu(ExecutionContext ec, String instName, MatrixObject in, String outputName) throws DMLRuntimeException {
		long N = in.getNumRows();
		long CHW = in.getNumColumns();
		MatrixObject output = ec.getMatrixObject(outputName);
		getDenseMatrixOutputForGPUInstruction(ec, instName, outputName);	// Allocated the dense output matrix
		long t0=0;
		cudnnTensorDescriptor srcTensorDesc = ((JCudaObject)in.getGPUObject()).getTensorDescriptor();
		if(N*CHW >= numDoublesIn2GB ||  srcTensorDesc == null) {
			// Invokes relu(double* A,  double* ret, int rlen, int clen)
			if (GPUStatistics.DISPLAY_STATISTICS) t0 = System.nanoTime();
			Pointer dstData = getDensePointer(output, instName);
			Pointer srcData = getDensePointer(in, instName); // TODO: FIXME: Add sparse kernel support for relu
			kernels.launchKernel("relu",
							ExecutionConfig.getConfigForSimpleMatrixOperations((int)N, (int)CHW),
							srcData, dstData, (int)N, (int) CHW);
			if (GPUStatistics.DISPLAY_STATISTICS) GPUStatistics.maintainCPMiscTimes(instName, GPUInstruction.MISC_TIMER_RELU_KERNEL, System.nanoTime() - t0);
		}
		else {
			performCuDNNReLU(instName, in, getDensePointer(output, true, instName), srcTensorDesc);
		}
	}



	//********************************************************************/
	//************* End of DEEP LEARNING Operators ***********************/
	//********************************************************************/



	//********************************************************************/
	//********** TRANSPOSE SELF MATRIX MULTIPLY Functions ****************/
	//********************************************************************/

	/**
	 * Performs tsmm, A %*% A' or A' %*% A, on GPU by exploiting cublasDsyrk(...)
	 *
	 * @param ec execution context
	 * @param instName the invoking instruction's name for record {@link Statistics}.
	 * @param left input matrix, as in a tsmm expression like A %*% A' or A' %*% A, we just need to check whether the left one is transposed or not, I named it 'left'
	 * @param outputName output matrix name
	 * @param isLeftTransposed if true, left transposed
	 * @throws DMLRuntimeException if DMLRuntimeException occurs
	 */
	public static void matmultTSMM(ExecutionContext ec, String instName, MatrixObject left, String outputName,
																 boolean isLeftTransposed) throws DMLRuntimeException {
		if(isInSparseFormat(left)) {
			// For sparse TSMM, invoke matmult (TODO: possible performance improvement)
			matmult(ec, instName, left, left, outputName, isLeftTransposed, !isLeftTransposed);
			return;
		}

		// For dense TSMM, exploit cublasDsyrk(...) and call custom kernel to flip the matrix
		MatrixObject output = ec.getMatrixObject(outputName);
		getDenseMatrixOutputForGPUInstruction(ec, instName, outputName);	// Allocated the dense output matrix

		// Since CuBLAS expects inputs in column-major format,
		// reverse the order of matrix-multiplication and take care of dimension mismatch.
		int transa = isLeftTransposed ? cublasOperation.CUBLAS_OP_N : cublasOperation.CUBLAS_OP_T;
		// Note: the dimensions are swapped
		int m = (int) (isLeftTransposed ? left.getNumColumns() : left.getNumRows());
		int k = (int) (isLeftTransposed ? left.getNumRows() : left.getNumColumns());

		if(m == -1)
			throw new DMLRuntimeException("Incorrect dimensions");

		int lda = (int) (isLeftTransposed ? m : k);
		int ldc = m;

		if(!left.getGPUObject().isAllocated())
			throw new DMLRuntimeException("Input is not allocated:" + left.getGPUObject().isAllocated());
		if(!output.getGPUObject().isAllocated())
			throw new DMLRuntimeException("Output is not allocated:" + output.getGPUObject().isAllocated());

		Pointer A = getDensePointer(left, instName);
		Pointer C = getDensePointer(output, instName);

		long t0=0, t1=0;

		if (GPUStatistics.DISPLAY_STATISTICS) t0 = System.nanoTime();
		JCublas2.cublasDsyrk(cublasHandle, cublasFillMode.CUBLAS_FILL_MODE_LOWER,transa, m, k, one(), A, lda, zero(), C, ldc);
		if (GPUStatistics.DISPLAY_STATISTICS) GPUStatistics.maintainCPMiscTimes(instName, GPUInstruction.MISC_TIMER_SYRK_LIB, System.nanoTime() - t0);

		if (GPUStatistics.DISPLAY_STATISTICS) t1 = System.nanoTime();
		copyUpperToLowerTriangle(instName, output);
		if (GPUStatistics.DISPLAY_STATISTICS) GPUStatistics.maintainCPMiscTimes(instName, GPUInstruction.MISC_TIMER_UPPER_TO_LOWER_TRIANGLE_KERNEL, System.nanoTime() - t1);
	}

	/**
	 * Used for all version of TSMM where the result is known to be symmetric.
	 * Hence, we compute only the upper triangular matrix and copy this partial
	 * result down to lower triangular matrix once.
	 *
	 * @param instName instruction name
	 * @param ret upper triangular matrix
	 * @throws DMLRuntimeException if DMLRuntimeException occurs
	 */
	private static void copyUpperToLowerTriangle(String instName, MatrixObject ret) throws DMLRuntimeException {
		if(isInSparseFormat(ret)) {
			throw new DMLRuntimeException("Sparse GPU copyUpperToLowerTriangle is not implemented");
		}
		if(ret.getNumRows() != ret.getNumColumns()) {
			throw new DMLRuntimeException("Only square matrix kernel is implemented for copyUpperToLowerTriangle");
		}
		int dim = (int) ret.getNumRows();
		kernels.launchKernel("copy_u2l_dense",
						ExecutionConfig.getConfigForSimpleMatrixOperations(dim, dim),
						getDensePointer(ret, instName), dim, dim*dim);
	}



	//********************************************************************/
	//******** End of TRANSPOSE SELF MATRIX MULTIPLY Functions ***********/
	//********************************************************************/

	//********************************************************************/
	//***************** MATRIX MULTIPLY Functions ************************/
	//********************************************************************/

	/**
	 * Matrix multiply on GPU
	 * Examines sparsity and shapes and routes call to appropriate method
	 * from cuBLAS or cuSparse
	 * C = op(A) x op(B)
	 * @param ec									Current {@link ExecutionContext} instance
	 * @param instName name of the invoking instruction to record{@link Statistics}.
	 * @param left1								Matrix A
	 * @param right1							Matrix B
	 * @param outputName					Name of the output matrix C (in code generated after LOP layer)
	 * @param isLeftTransposed1		op for A, transposed or not
	 * @param isRightTransposed1	op for B, tranposed or not
	 * @return	output of matrix multiply
	 * @throws DMLRuntimeException if DMLRuntimeException occurs
	 */
	public static MatrixObject matmult(ExecutionContext ec, String instName, MatrixObject left1, MatrixObject right1, String outputName,
																		 boolean isLeftTransposed1, boolean isRightTransposed1) throws DMLRuntimeException {

		if(!left1.getGPUObject().isAllocated() || !right1.getGPUObject().isAllocated())
			throw new DMLRuntimeException("One of input is not allocated:" + left1.getGPUObject().isAllocated() + " " + right1.getGPUObject().isAllocated());

		boolean bothDense = !left1.getGPUObject().isInSparseFormat() && !right1.getGPUObject().isInSparseFormat();
		boolean bothSparse = left1.getGPUObject().isInSparseFormat() && right1.getGPUObject().isInSparseFormat();

		MatrixObject output = ec.getMatrixObject(outputName);

		if (bothDense) {		// Dense C = Dense A * Dense B
			// For both dense, do cuBLAS
			getDenseMatrixOutputForGPUInstruction(ec, instName, outputName);	// Allocated the dense output matrix
			denseDenseMatmult(instName, output, left1, right1, isLeftTransposed1, isRightTransposed1);
		}
		else if (bothSparse){	// Sparse C = Sparse A * Sparse B
			ec.allocateGPUMatrixObject(outputName);
			bothSparseMatmult(instName, output, left1, right1, isLeftTransposed1, isRightTransposed1);
		}
		else {	// Either of A or B is sparse, Sparse C = Sparse/Dense A * Dense/Sparse B
			// Convert the dense to sparse and use the cusparseDcsrgemm routine
			ec.allocateGPUMatrixObject(outputName);
			eitherSparseMatmult(instName, output, left1, right1, isLeftTransposed1, isRightTransposed1);
		}

		return output;
	}

	/**
	 * One of the matrices is sparse, the other dense
	 * C = op(A) x op(B)
	 * @param instName the invoking instruction's name for record {@link Statistics}.
	 * @param output				allocated output object for C on host to which GPU output will be attached
	 * @param left					Matrix A on host
	 * @param right					Matrix B on host
	 * @param isLeftTransposed		op for A, tranposed or not
	 * @param isRightTransposed		op for B, transposed or not
	 * @throws DMLRuntimeException if DMLRuntimeException occurs
	 */
	protected static void eitherSparseMatmult(String instName, MatrixObject output, MatrixObject left, MatrixObject right,
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
			sparseDenseMatmult(instName, output, left, right, isLeftTransposed, isRightTransposed, transA, transB, m, n, k);
		} else {
			// Left dense, right sparse
			denseSparseMatmult(instName, output, right, left, isLeftTransposed, isRightTransposed, transA, transB, m, n, k);
		}
	}

	/**
	 * C = op(A) * op(B) where A is dense and B is sparse
	 * If B is ultrasparse, A is converted to a sparse matrix and {@code sparseSparseMatmult(MatrixObject, int, int, int, int, int, CSRPointer, CSRPointer)} is invoked
	 * otherwise B is converted to a dense matrix and {@code denseDenseMatmult(Pointer, int, int, int, int, boolean, boolean, Pointer, Pointer)} is invoked.
	 * @param instName the invoking instruction's name for record {@link Statistics}.
	 * @param output ?
	 * @param right ?
	 * @param left ?
	 * @param isLeftTransposed ?
	 * @param isRightTransposed ?
	 * @param transA ?
	 * @param transB ?
	 * @param m ?
	 * @param n ?
	 * @param k ?
	 * @throws DMLRuntimeException if DMLRuntimeException occurs
	 */
	protected static void denseSparseMatmult(String instName, MatrixObject output, MatrixObject right, MatrixObject left,
																					 boolean isLeftTransposed, boolean isRightTransposed, int transA, int transB, int m, int n, int k)
					throws DMLRuntimeException {
		// right sparse, left dense
		CSRPointer B = ((JCudaObject)right.getGPUObject()).jcudaSparseMatrixPtr;
		Pointer ADense = getDensePointer(left, instName);
		if (B.isUltraSparse(k, n)){
			LOG.debug(" GPU Dense-Sparse Matrix Multiplication (Converted to Sparse-Sparse)");
			// Convert left to CSR and do cuSparse matmul
			int rowsA = (int)left.getNumRows();
			int colsA = (int)left.getNumColumns();


			long t0=0,t1=0, t2=0;
			if (DMLScript.STATISTICS) t0 = System.nanoTime();
			Pointer AT = JCudaObject.transpose(ADense, rowsA, colsA, colsA, rowsA);
			if (GPUStatistics.DISPLAY_STATISTICS) GPUStatistics.maintainCPMiscTimes(instName, GPUInstruction.MISC_TIMER_TRANSPOSE_LIB, System.nanoTime() - t0);

			if (GPUStatistics.DISPLAY_STATISTICS) t1 = System.nanoTime();
			CSRPointer A = JCudaObject.columnMajorDenseToRowMajorSparse(cusparseHandle, rowsA, colsA, AT);
			if (GPUStatistics.DISPLAY_STATISTICS) GPUStatistics.maintainCPMiscTimes(instName, GPUInstruction.MISC_TIMER_DENSE_TO_SPARSE, System.nanoTime() - t1);

			if (DMLScript.STATISTICS) GPUStatistics.cudaDenseToSparseTime.getAndAdd(System.nanoTime() - t0);
			if (DMLScript.STATISTICS) GPUStatistics.cudaDenseToSparseCount.getAndAdd(1);
			sparseSparseMatmult(instName, output, transA, transB, m, n, k, A, B);

			if (GPUStatistics.DISPLAY_STATISTICS) t2 = System.nanoTime();
			A.deallocate();
			cudaFreeHelper(AT);
			if (GPUStatistics.DISPLAY_STATISTICS) GPUStatistics.maintainCPMiscTimes(instName, GPUInstruction.MISC_TIMER_CUDA_FREE, System.nanoTime() - t2, 2);

		} else {
			LOG.debug(" GPU Dense-Sparse Matrix Multiplication (Converted to Dense-Dense)");
			// Convert right to dense and do a cuBlas matmul
			// BDenseTransposed is a column major matrix
			// Note the arguments to denseDenseMatmult to accommodate for this.
			long t0=0, t1=0;
			if (DMLScript.STATISTICS) t0 = System.nanoTime();
			Pointer BDenseTransposed = B.toColumnMajorDenseMatrix(cusparseHandle, cublasHandle, (int)right.getNumRows(), (int)right.getNumColumns());
			if (GPUStatistics.DISPLAY_STATISTICS) GPUStatistics.maintainCPMiscTimes(instName, GPUInstruction.MISC_TIMER_SPARSE_TO_DENSE, System.nanoTime() - t0);
			if (DMLScript.STATISTICS) GPUStatistics.cudaSparseToDenseTime.getAndAdd(System.nanoTime() - t0);
			if (DMLScript.STATISTICS) GPUStatistics.cudaSparseToDenseCount.getAndAdd(System.nanoTime() - t0);

			if (GPUStatistics.DISPLAY_STATISTICS) t1 = System.nanoTime();
			boolean allocated = output.getGPUObject().acquireDeviceModifyDense();	// To allocate the dense matrix
			if (GPUStatistics.DISPLAY_STATISTICS && allocated) GPUStatistics.maintainCPMiscTimes(instName, GPUInstruction.MISC_TIMER_ALLOCATE_DENSE_OUTPUT, System.nanoTime() - t1);
			Pointer C = getDensePointer(output, instName);
			denseDenseMatmult(instName, C,
							(int) left.getNumRows(), (int) left.getNumColumns(),
							(int) right.getNumColumns(), (int) right.getNumRows(),
							isLeftTransposed, !isRightTransposed,
							ADense, BDenseTransposed);

			cudaFreeHelper(instName, BDenseTransposed);
		}
	}

	/**
	 * * C = op(A) * op(B) where A is sparse and B is dense
	 * If A is ultrasparse, B is converted to a sparse matrix and {@code sparseSparseMatmult(MatrixObject, int, int, int, int, int, CSRPointer, CSRPointer)} is invoked
	 * otherwise A is converted to a dense matrix and {@code denseDenseMatmult(Pointer, int, int, int, int, boolean, boolean, Pointer, Pointer)} is invoked.
	 * @param instName the invoking instruction's name for record {@link Statistics}.
	 * @param output ?
	 * @param left ?
	 * @param right ?
	 * @param isLeftTransposed ?
	 * @param isRightTransposed ?
	 * @param transA ?
	 * @param transB ?
	 * @param m ?
	 * @param n ?
	 * @param k ?
	 * @throws DMLRuntimeException if DMLRuntimeException occurs
	 */
	protected static void sparseDenseMatmult(String instName, MatrixObject output, MatrixObject left, MatrixObject right,
																					 boolean isLeftTransposed, boolean isRightTransposed, int transA, int transB, int m, int n, int k)
					throws DMLRuntimeException {
		CSRPointer A = ((JCudaObject)left.getGPUObject()).jcudaSparseMatrixPtr;
		Pointer BDense = getDensePointer(right, instName);

		if (n == 1){
			// Sparse Matrix - Dense Vector multiply
			LOG.debug(" GPU Sparse Matrix - Dense Vector Mutliply");
			sparseMatrixDenseVectorMult(instName, output, A, BDense, transA, (int)left.getNumRows(), (int)left.getNumColumns());

		} else {

			long t0=0, t1=0, t2=0;
			// Sparse Matrix Dense Matrix multiply
			if (A.isUltraSparse(m, k)){
				LOG.debug(" GPU Sparse-Dense Matrix Multiplication (Converted to Sparse-Sparse)");
				// Convert right to CSR and do cuSparse matmul
				int rowsB = (int)right.getNumRows();
				int colsB = (int)right.getNumColumns();

				if (DMLScript.STATISTICS) t0 = System.nanoTime();
				Pointer BT = JCudaObject.transpose(BDense, rowsB, colsB, colsB, rowsB);
				if (GPUStatistics.DISPLAY_STATISTICS) GPUStatistics.maintainCPMiscTimes(instName, GPUInstruction.MISC_TIMER_TRANSPOSE_LIB, System.nanoTime() - t0);

				if (GPUStatistics.DISPLAY_STATISTICS) t1 = System.nanoTime();
				CSRPointer B = JCudaObject.columnMajorDenseToRowMajorSparse(cusparseHandle, rowsB, colsB, BT);
				if (GPUStatistics.DISPLAY_STATISTICS) GPUStatistics.maintainCPMiscTimes(instName, GPUInstruction.MISC_TIMER_DENSE_TO_SPARSE, System.nanoTime() - t1);

				if (DMLScript.STATISTICS) GPUStatistics.cudaDenseToSparseTime.getAndAdd(System.nanoTime() - t0);
				if (DMLScript.STATISTICS) GPUStatistics.cudaDenseToSparseCount.getAndAdd(1);

				sparseSparseMatmult(instName, output, transA, transB, m, n, k, A, B);

				if (GPUStatistics.DISPLAY_STATISTICS) t2 = System.nanoTime();
				B.deallocate();
				cudaFreeHelper(BT);
				if (GPUStatistics.DISPLAY_STATISTICS) GPUStatistics.maintainCPMiscTimes(instName, GPUInstruction.MISC_TIMER_CUDA_FREE, System.nanoTime() - t2, 2);

			} else {
				LOG.debug(" GPU Sparse-Dense Matrix Multiplication (Converted to Dense-Dense)");
				// Convert left to dense and do a cuBlas matmul
				// ADenseTransposed is a column major matrix
				// Note the arguments to denseDenseMatmult to accommodate for this.
				if (DMLScript.STATISTICS) t0 = System.nanoTime();
				Pointer ADenseTransposed = A.toColumnMajorDenseMatrix(cusparseHandle, cublasHandle, (int)left.getNumRows(), (int)left.getNumColumns());
				if (GPUStatistics.DISPLAY_STATISTICS) GPUStatistics.maintainCPMiscTimes(instName, GPUInstruction.MISC_TIMER_SPARSE_TO_DENSE, System.nanoTime() - t0);
				if (DMLScript.STATISTICS) GPUStatistics.cudaSparseToDenseTime.getAndAdd(System.nanoTime() - t0);
				if (DMLScript.STATISTICS) GPUStatistics.cudaSparseToDenseCount.getAndAdd(System.nanoTime() - t0);

				if (GPUStatistics.DISPLAY_STATISTICS) t1 = System.nanoTime();
				boolean allocated = output.getGPUObject().acquireDeviceModifyDense();	// To allocate the dense matrix
				if (allocated && GPUStatistics.DISPLAY_STATISTICS) GPUStatistics.maintainCPMiscTimes(instName, GPUInstruction.MISC_TIMER_ALLOCATE_DENSE_OUTPUT, System.nanoTime() - t1);

				Pointer C = getDensePointer(output, instName);
				denseDenseMatmult(instName, C,
								(int) left.getNumColumns(), (int) left.getNumRows(),
								(int) right.getNumRows(), (int) right.getNumColumns(),
								!isLeftTransposed, isRightTransposed,
								ADenseTransposed, BDense);

				cudaFreeHelper(instName, ADenseTransposed);
			}
		}
	}

	/**
	 * C = op(A) x B
	 * A is a sparse matrix, B is a dense vector
	 * @param instName the invoking instruction's name for record {@link Statistics}.
	 * @param output	allocated output on the host, to which the GPU output C will be attached
	 * @param A			sparse matrix A on the GPU
	 * @param B_dense	dense matrix/vector B on the GPU
	 * @param transA	op for A, tranposed or not
	 * @param m			number of rows in A (not op(A))
	 * @param k			number of cols in A or number of rows in B (not op(A) or op(B))
	 * @throws DMLRuntimeException if DMLRuntimeException occurs
	 */
	protected static void sparseMatrixDenseVectorMult(String instName, MatrixObject output, CSRPointer A, Pointer B_dense, int transA,
																										int m, int k) throws DMLRuntimeException {
		long size = m * Sizeof.DOUBLE;
		if (transA == CUSPARSE_OPERATION_TRANSPOSE){
			size = k * Sizeof.DOUBLE;
		}
		Pointer C_dense = JCudaObject.allocate(instName, (int)size);
		long t1=0;
		if (GPUStatistics.DISPLAY_STATISTICS) t1 = System.nanoTime();
		cusparseDcsrmv(cusparseHandle, transA, m, k, (int)A.nnz, one(), A.descr, A.val, A.rowPtr, A.colInd, B_dense, zero(), C_dense);
		cudaDeviceSynchronize(); 	// Since cusparseDcsrmv is asynchronously executed
		if (GPUStatistics.DISPLAY_STATISTICS) GPUStatistics.maintainCPMiscTimes(instName, GPUInstruction.MISC_TIMER_SPARSE_MATRIX_DENSE_VECTOR_LIB, System.nanoTime() - t1);

		((JCudaObject)(output.getGPUObject())).setDenseMatrixCudaPointer(C_dense);
		output.getGPUObject().setDeviceModify(size);
	}

	/**
	 * Sparse C = Sparse op(A) * Sparse op(B)
	 * Reroutes call to sparse matrix-vector mult if needed
	 * @param instName the invoking instruction's name for record {@link Statistics}.
	 * @param output ?
	 * @param instName name of the invoking instruction to record{@link Statistics}.
	 * @param left ?
	 * @param right ?
	 * @param isLeftTransposed ?
	 * @param isRightTransposed ?
	 * @throws DMLRuntimeException if DMLRuntimeException occurs
	 */
	protected static void bothSparseMatmult(String instName, MatrixObject output, MatrixObject left, MatrixObject right,
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
			sparseMatrixVectorMult(instName, output, transA, (int)left.getNumRows(), (int)left.getNumColumns(), (int)right.getNumRows(), A, B);
		} else {												// Matrix-Matrix multiplication
			sparseSparseMatmult(instName, output, transA, transB, m, n, k, A, B);
		}
	}

	/**
	 * Does a sparse matrix-vector multiply.
	 * C = op(A) x B, A is a sparse matrix, B is a sparse vector with numCols = 1.
	 * @param instName the invoking instruction's name for record {@link Statistics}.
	 * @param output	allocated output object C to which the GPU output matrix will be attached
	 * @param transA	if A is to be transposed or not (the op in op(A))
	 * @param m			number of rows in A (not op(A))
	 * @param n			number of cols in A (not op(A))
	 * @param k			number of rows in B, (cols in B is assumed to be 1)
	 * @param A			left sparse matrix on GPU
	 * @param B			right sparse vector on GPU
	 * @throws DMLRuntimeException if DMLRuntimeException occurs
	 */
	protected static void sparseMatrixVectorMult(String instName, MatrixObject output, int transA, int m, int n, int k,
																							 CSRPointer A, CSRPointer B) throws DMLRuntimeException {
		LOG.debug(" GPU Sparse Matrix Sparse Vector Multiply (Converted to Sparse Matrix Dense Vector Multiply)");
		long t0=0;
		if (GPUStatistics.DISPLAY_STATISTICS) t0 = System.nanoTime();
		Pointer BDenseVector = B.toColumnMajorDenseMatrix(cusparseHandle, cublasHandle, k, 1);
		if (GPUStatistics.DISPLAY_STATISTICS) GPUStatistics.maintainCPMiscTimes(instName, GPUInstruction.MISC_TIMER_SPARSE_TO_DENSE, System.nanoTime() - t0);
		sparseMatrixDenseVectorMult(instName, output, A, BDenseVector, transA, m, k);
	}

	/**
	 * Does a sparse-sparse Matrix multiply
	 * C = op(A) x op(B), A, B are sparse matrices
	 * @param instName the invoking instruction's name for record {@link Statistics}.
	 * @param output	allocated output object on host to which the GPU output matrix will be attached
	 * @param transA	op for A - to be transposed or not
	 * @param transB	op for B
	 * @param m			number of rows in op(A)
	 * @param n			number of cols in op(B)
	 * @param k			number of cols in op(A) or rows in op(B)
	 * @param A			left sparse matrix on GPU
	 * @param B			right sparse matrix on GPU
	 * @throws DMLRuntimeException if DMLRuntimeException occurs
	 */
	protected static void sparseSparseMatmult(String instName, MatrixObject output, int transA, int transB, int m, int n, int k,
																						CSRPointer A, CSRPointer B) throws DMLRuntimeException {
		LOG.debug(" GPU Sparse-Sparse Matrix Multiply ");

		long t0=0, t1=0;
		if (GPUStatistics.DISPLAY_STATISTICS) t0 = System.nanoTime();
		CSRPointer C = CSRPointer.allocateForMatrixMultiply(cusparseHandle, A, transA, B, transB, m, n, k);
		if (GPUStatistics.DISPLAY_STATISTICS) GPUStatistics.maintainCPMiscTimes(instName, GPUInstruction.MISC_TIMER_SPARSE_ALLOCATE_LIB, System.nanoTime() - t0);

		((JCudaObject)output.getGPUObject()).setSparseMatrixCudaPointer(C);
		long sizeOfC = CSRPointer.estimateSize(C.nnz, output.getNumRows());
		output.getGPUObject().setDeviceModify(sizeOfC);

		if (GPUStatistics.DISPLAY_STATISTICS) t1 = System.nanoTime();
		cusparseDcsrgemm(cusparseHandle, transA, transB, m, n, k,
						A.descr, (int)A.nnz, A.val, A.rowPtr, A.colInd,
						B.descr, (int)B.nnz, B.val, B.rowPtr, B.colInd,
						C.descr, C.val, C.rowPtr, C.colInd);
		cudaDeviceSynchronize();
		if (GPUStatistics.DISPLAY_STATISTICS) GPUStatistics.maintainCPMiscTimes(instName, GPUInstruction.MISC_TIMER_SPARSE_MATRIX_SPARSE_MATRIX_LIB, System.nanoTime() - t1);
	}

	/**
	 * Dense dense matrix multiply
	 * C = op(A) * op(B), A and B are dense matrices
	 * @param instName name of the invoking instruction to record{@link Statistics}.
	 * @param output				output object C on host with GPU data allocated
	 * @param left1					left matrix A on host (in row-major order)
	 * @param right1				right matrix B on host (in row-major order)
	 * @param isLeftTransposed1 	op for A, transposed or not
	 * @param isRightTransposed1	op for B, transposed or not
	 * @throws DMLRuntimeException if DMLRuntimeException occurs
	 */
	protected static void denseDenseMatmult(String instName, MatrixObject output, MatrixObject left1, MatrixObject right1,
																					boolean isLeftTransposed1, boolean isRightTransposed1) throws DMLRuntimeException {

		Pointer leftPtr = getDensePointer(left1, instName);
		Pointer rightPtr = getDensePointer(right1, instName);

		int leftRows = (int) left1.getNumRows();
		int leftCols = (int) left1.getNumColumns();
		int rightRows = (int) right1.getNumRows();
		int rightCols = (int) right1.getNumColumns();
		Pointer C = getDensePointer(output, instName);
		denseDenseMatmult(instName, C, leftRows, leftCols, rightRows, rightCols, isLeftTransposed1, isRightTransposed1,
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
	 * @param instName name of the invoking instruction to record{@link Statistics}.
	 * @param output			output allocated on GPU in column major format
	 * @param leftRows1			number of rows in A
	 * @param leftCols1			number of cols in A
	 * @param rightRows1		number of rows in B
	 * @param rightCols1		number of cols in B
	 * @param isLeftTransposed1		op for A, transposed or not
	 * @param isRightTransposed1	op for B, transposed or not
	 * @param leftPtr			A allocated on the GPU in row-major format
	 * @param rightPtr			B allocated on the GPU in row-major format
	 * @throws DMLRuntimeException if DMLRuntimeException occurs
	 */
	public static void denseDenseMatmult(String instName, Pointer output, int leftRows1, int leftCols1, int rightRows1,
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

		//int lda = leftRows;
		//int ldb = leftCols;
		int lda = isLeftTransposed ?  k : m;
		int ldb = isRightTransposed ? n : k;
		int ldc = m;

		int transa = isLeftTransposed ? cublasOperation.CUBLAS_OP_T : cublasOperation.CUBLAS_OP_N;
		int transb = isRightTransposed ? cublasOperation.CUBLAS_OP_T : cublasOperation.CUBLAS_OP_N;

		long t0=0;
		if (GPUStatistics.DISPLAY_STATISTICS) t0 = System.nanoTime();
		Pointer C = output;
		if (m == 1 && n == 1){
			// Vector product
			LOG.debug(" GPU Dense-dense Vector Product");
			double[] result = {0};
			JCublas2.cublasDdot(cublasHandle, k, A, 1, B, 1, Pointer.to(result));
			// By default in CuBlas V2, cublas pointer mode is set to CUBLAS_POINTER_MODE_HOST.
			// This means that scalar values passed are on host (as opposed to on device).
			// The result is copied from the host back to the device so that the rest of
			// infrastructure can treat it uniformly.
			cudaMemcpy(C, Pointer.to(result), 1 * Sizeof.DOUBLE, cudaMemcpyHostToDevice);
			if (GPUStatistics.DISPLAY_STATISTICS) GPUStatistics.maintainCPMiscTimes(instName, GPUInstruction.MISC_TIMER_DENSE_DOT_LIB, System.nanoTime() - t0);
		} else if (m == 1) {
			// Vector-matrix multiply
			LOG.debug(" GPU Dense Vector-Matrix Multiply");
			transb = isRightTransposed ? cublasOperation.CUBLAS_OP_N : cublasOperation.CUBLAS_OP_T;
			JCublas2.cublasDgemv(cublasHandle, transb, rightRows, rightCols, Pointer.to(one), B, ldb, A, 1, Pointer.to(zero), C, 1);
			if (GPUStatistics.DISPLAY_STATISTICS) GPUStatistics.maintainCPMiscTimes(instName, GPUInstruction.MISC_TIMER_DENSE_VECTOR_DENSE_MATRIX_LIB, System.nanoTime() - t0);
		} else if (n == 1){
			// Matrix-vector multiply
			LOG.debug(" GPU Dense Matrix-Vector Multiply");
			JCublas2.cublasDgemv(cublasHandle, transa, leftRows, leftCols, Pointer.to(one), A, lda, B, 1, Pointer.to(zero), C, 1);
			if (GPUStatistics.DISPLAY_STATISTICS) GPUStatistics.maintainCPMiscTimes(instName, GPUInstruction.MISC_TIMER_DENSE_MATRIX_DENSE_VECTOR_LIB, System.nanoTime() - t0);
		} else {
			LOG.debug(" GPU Dense-Dense Matrix Multiply ");
			JCublas2.cublasDgemm(cublasHandle, transa, transb, m, n, k, Pointer.to(one), A, lda, B, ldb, Pointer.to(zero), C, ldc);
			if (GPUStatistics.DISPLAY_STATISTICS) GPUStatistics.maintainCPMiscTimes(instName, GPUInstruction.MISC_TIMER_DENSE_MATRIX_DENSE_MATRIX_LIB, System.nanoTime() - t0);
		}
	}

	//********************************************************************/
	//***************** END OF MATRIX MULTIPLY Functions *****************/
	//********************************************************************/


	//********************************************************************/
	//****************  UNARY AGGREGATE Functions ************************/
	//********************************************************************/


	/**
	 * Entry point to perform Unary aggregate operations on the GPU.
	 * The execution context object is used to allocate memory for the GPU.
	 * @param ec			Instance of {@link ExecutionContext}, from which the output variable will be allocated
	 * @param instName name of the invoking instruction to record{@link Statistics}.
	 * @param in1			input matrix
	 * @param output	output matrix/scalar name
	 * @param op			Instance of {@link AggregateUnaryOperator} which encapsulates the direction of reduction/aggregation and the reduction operation.
	 * @throws DMLRuntimeException if {@link DMLRuntimeException} occurs
	 */
	public static void unaryAggregate(ExecutionContext ec, String instName, MatrixObject in1, String output, AggregateUnaryOperator op)
					throws DMLRuntimeException {

		final int REDUCTION_ALL = 1;
		final int REDUCTION_ROW = 2;
		final int REDUCTION_COL = 3;
		final int REDUCTION_DIAG = 4;

		// A kahan sum implemention is not provided. is a "uak+" or other kahan operator is encountered,
		// it just does regular summation reduction.
		final int OP_PLUS = 1;
		final int OP_PLUS_SQ = 2;
		final int OP_MEAN = 3;
		final int OP_VARIANCE = 4;
		final int OP_MULTIPLY = 5;
		final int OP_MAX = 6;
		final int OP_MIN = 7;
		final int OP_MAXINDEX = 8;
		final int OP_MININDEX = 9;


		// Sanity Checks
		if(!in1.getGPUObject().isAllocated())
			throw new DMLRuntimeException("Internal Error - The input is not allocated for a GPU Aggregate Unary:" + in1.getGPUObject().isAllocated());

		boolean isSparse = in1.getGPUObject().isInSparseFormat();
		IndexFunction indexFn = op.indexFn;
		AggregateOperator aggOp = op.aggOp;

		// Convert Reduction direction to a number to pass to CUDA kernel
		int reductionDirection = -1;
		if (indexFn instanceof ReduceAll){
			reductionDirection = REDUCTION_ALL;
		} else if (indexFn instanceof ReduceRow){
			reductionDirection = REDUCTION_ROW;
		} else if (indexFn instanceof ReduceCol){
			reductionDirection = REDUCTION_COL;
		} else if (indexFn instanceof ReduceDiag){
			reductionDirection = REDUCTION_DIAG;
		} else {
			throw new DMLRuntimeException("Internal Error - Invalid index function type, only reducing along rows, columns, diagonals or all elements is supported in Aggregate Unary operations");
		}
		assert reductionDirection !=-1 : "Internal Error - Incorrect type of reduction direction set for aggregate unary GPU instruction";

		// Convert function type to a number to pass to the CUDA Kernel
		int opIndex = -1;
		if (aggOp.increOp.fn instanceof KahanPlus) {
			opIndex = OP_PLUS;
		} else if (aggOp.increOp.fn instanceof KahanPlusSq) {
			opIndex = OP_PLUS_SQ;
		} else if (aggOp.increOp.fn instanceof Mean) {
			opIndex = OP_MEAN;
		} else if (aggOp.increOp.fn instanceof CM) {
			assert ((CM)aggOp.increOp.fn).getAggOpType() == CMOperator.AggregateOperationTypes.VARIANCE : "Internal Error - Invalid Type of CM operator for Aggregate Unary operation on GPU";
			opIndex = OP_VARIANCE;
		} else if (aggOp.increOp.fn instanceof Plus) {
			opIndex = OP_PLUS;
		} else if (aggOp.increOp.fn instanceof Multiply) {
			opIndex = OP_MULTIPLY;
		} else if (aggOp.increOp.fn instanceof Builtin) {
			Builtin b = (Builtin)aggOp.increOp.fn;
			switch(b.bFunc) {
				case MAX: opIndex = OP_MAX; break;
				case MIN: opIndex = OP_MIN; break;
				case MAXINDEX: opIndex = OP_MAXINDEX; break;
				case MININDEX: opIndex = OP_MININDEX;break;
				default:
					new DMLRuntimeException("Internal Error - Unsupported Builtin Function for Aggregate unary being done on GPU");
			}
		} else {
			throw new DMLRuntimeException("Internal Error - Aggregate operator has invalid Value function");
		}
		assert opIndex != -1 : "Internal Error - Incorrect type of operation set for aggregate unary GPU instruction";


		int rlen = (int)in1.getNumRows();
		int clen = (int)in1.getNumColumns();
		if (isSparse){
			// The strategy for the time being is to convert sparse to dense
			// until a sparse specific kernel is written.
			((JCudaObject)in1.getGPUObject()).sparseToDense(instName);
			// long nnz = in1.getNnz();
			// assert nnz > 0 : "Internal Error - number of non zeroes set to " + nnz + " in Aggregate Binary for GPU";
			// MatrixObject out = ec.getSparseMatrixOutputForGPUInstruction(output, nnz);
			// throw new DMLRuntimeException("Internal Error - Not implemented");

		}

		Pointer out = null;
		if (reductionDirection == REDUCTION_COL || reductionDirection == REDUCTION_ROW) {
			// Matrix output
			MatrixObject out1 = getDenseMatrixOutputForGPUInstruction(ec, instName, output);
			out = getDensePointer(out1, instName);
		}

		Pointer in = getDensePointer(in1, instName);
		int size = rlen * clen;

		// For scalars, set the scalar output in the Execution Context object
		switch (opIndex){
			case OP_PLUS: {
				switch(reductionDirection) {
					case REDUCTION_ALL : {
						double result = reduceAll(instName, "reduce_sum", in, size);
						ec.setScalarOutput(output, new DoubleObject(result));
						break;
					}
					case REDUCTION_COL : {	// The names are a bit misleading, REDUCTION_COL refers to the direction (reduce all elements in a column)
						reduceRow(instName, "reduce_row_sum", in, out, rlen, clen);
						break;
					}
					case REDUCTION_ROW : {
						reduceCol(instName, "reduce_col_sum", in, out, rlen, clen);
						break;
					}
					case REDUCTION_DIAG :
						throw new DMLRuntimeException("Internal Error - Row, Column and Diag summation not implemented yet");
				}
				break;
			}
			case OP_PLUS_SQ : {
				// Calculate the squares in a temporary object tmp
				Pointer tmp = JCudaObject.allocate(instName, size * Sizeof.DOUBLE);

				squareMatrix(instName, in, tmp, rlen, clen);
				// Then do the sum on the temporary object and free it
				switch(reductionDirection) {
					case REDUCTION_ALL : {
						double result = reduceAll(instName, "reduce_sum", tmp, size);
						ec.setScalarOutput(output, new DoubleObject(result));
						break;
					}
					case REDUCTION_COL : {	// The names are a bit misleading, REDUCTION_COL refers to the direction (reduce all elements in a column)
						reduceRow(instName, "reduce_row_sum", tmp, out, rlen, clen);
						break;
					}
					case REDUCTION_ROW : {
						reduceCol(instName, "reduce_col_sum", tmp, out, rlen, clen);
						break;
					}
					default:
						throw new DMLRuntimeException("Internal Error - Unsupported reduction direction for summation squared");
				}
				cudaFreeHelper(instName, tmp);
				break;
			}
			case OP_MEAN:{
				switch(reductionDirection) {
					case REDUCTION_ALL: {
						double result = reduceAll(instName, "reduce_sum", in, size);
						double mean = result / size;
						ec.setScalarOutput(output, new DoubleObject(mean));
						break;
					}
					case REDUCTION_COL: {
						reduceRow(instName, "reduce_row_mean", in, out, rlen, clen);
						break;
					}
					case REDUCTION_ROW: {
						reduceCol(instName, "reduce_col_mean", in, out, rlen, clen);
						break;
					}
					default:
						throw new DMLRuntimeException("Internal Error - Unsupported reduction direction for mean");
				}
				break;
			}
			case OP_MULTIPLY : {
				switch (reductionDirection) {
					case REDUCTION_ALL: {
						double result = reduceAll(instName, "reduce_prod", in, size);
						ec.setScalarOutput(output, new DoubleObject(result));
						break;
					}
					default:
						throw new DMLRuntimeException("Internal Error - Unsupported reduction direction for multiplication");
				}
				break;
			}
			case OP_MAX :{
				switch(reductionDirection) {
					case REDUCTION_ALL: {
						double result = reduceAll(instName, "reduce_max", in, size);
						ec.setScalarOutput(output, new DoubleObject(result));
						break;
					}
					case REDUCTION_COL: {
						reduceRow(instName, "reduce_row_max", in, out, rlen, clen);
						break;
					}
					case REDUCTION_ROW: {
						reduceCol(instName, "reduce_col_max", in, out, rlen, clen);
						break;
					}
					default:
						throw new DMLRuntimeException("Internal Error - Unsupported reduction direction for max");
				}
				break;
			}
			case OP_MIN :{
				switch(reductionDirection) {
					case REDUCTION_ALL: {
						double result = reduceAll(instName, "reduce_min", in, size);
						ec.setScalarOutput(output, new DoubleObject(result));
						break;
					}
					case REDUCTION_COL: {
						reduceRow(instName, "reduce_row_min", in, out, rlen, clen);
						break;
					}
					case REDUCTION_ROW: {
						reduceCol(instName, "reduce_col_min", in, out, rlen, clen);
						break;
					}
					default:
						throw new DMLRuntimeException("Internal Error - Unsupported reduction direction for min");
				}
				break;
			}
			case OP_VARIANCE : {
				// Temporary GPU array for
				Pointer tmp = JCudaObject.allocate(instName, size * Sizeof.DOUBLE);
				Pointer tmp2 = JCudaObject.allocate(instName, size * Sizeof.DOUBLE);

				switch(reductionDirection) {

					case REDUCTION_ALL: {
						double result = reduceAll(instName, "reduce_sum", in, size);
						double mean = result / size;

						// Subtract mean from every element in the matrix
						ScalarOperator minusOp = new RightScalarOperator(Minus.getMinusFnObject(), mean);
						matrixScalarOp(instName, in, mean, rlen, clen, tmp, minusOp);

						squareMatrix(instName, tmp, tmp2, rlen, clen);

						double result2 = reduceAll(instName, "reduce_sum", tmp2, size);
						double variance = result2 / (size - 1);
						ec.setScalarOutput(output, new DoubleObject(variance));

						break;
					}
					case REDUCTION_COL: {
						reduceRow(instName, "reduce_row_mean", in, out, rlen, clen);
						// Subtract the row-wise mean from every element in the matrix
						BinaryOperator minusOp = new BinaryOperator(Minus.getMinusFnObject());
						matrixMatrixOp(instName, in, out, rlen, clen, VectorShape.NONE.code(), VectorShape.COLUMN.code(), tmp, minusOp);

						squareMatrix(instName, tmp, tmp2, rlen, clen);

						Pointer tmpRow = JCudaObject.allocate(instName, rlen * Sizeof.DOUBLE);
						reduceRow(instName, "reduce_row_sum", tmp2, tmpRow, rlen, clen);

						ScalarOperator divideOp = new RightScalarOperator(Divide.getDivideFnObject(), clen - 1);
						matrixScalarOp(instName, tmpRow, clen - 1, rlen, clen, out, divideOp);

						cudaFreeHelper(instName, tmpRow);

						break;
					}
					case REDUCTION_ROW: {
						reduceCol(instName, "reduce_col_mean", in, out, rlen, clen);
						// Subtract the columns-wise mean from every element in the matrix
						BinaryOperator minusOp = new BinaryOperator(Minus.getMinusFnObject());
						matrixMatrixOp(instName, in, out, rlen, clen, VectorShape.NONE.code(), VectorShape.ROW.code(), tmp, minusOp);

						squareMatrix(instName, tmp, tmp2, rlen, clen);

						Pointer tmpCol = JCudaObject.allocate(instName, clen * Sizeof.DOUBLE);
						reduceCol(instName, "reduce_col_sum", tmp2, tmpCol, rlen, clen);

						ScalarOperator divideOp = new RightScalarOperator(Divide.getDivideFnObject(), rlen - 1);
						matrixScalarOp(instName, tmpCol, rlen - 1, rlen, clen, out, divideOp);

						cudaFreeHelper(instName, tmpCol);

						break;
					}
					default:
						throw new DMLRuntimeException("Internal Error - Unsupported reduction direction for variance");
				}
				cudaFreeHelper(instName, tmp);
				cudaFreeHelper(instName, tmp2);
				break;
			}
			case OP_MAXINDEX : {
				switch(reductionDirection) {
					case REDUCTION_COL:
						throw new DMLRuntimeException("Internal Error - Column maxindex of matrix not implemented yet for GPU ");
					default:
						throw new DMLRuntimeException("Internal Error - Unsupported reduction direction for maxindex");
				}
				// break;
			}
			case OP_MININDEX : {
				switch(reductionDirection) {
					case REDUCTION_COL:
						throw new DMLRuntimeException("Internal Error - Column minindex of matrix not implemented yet for GPU ");
					default:
						throw new DMLRuntimeException("Internal Error - Unsupported reduction direction for minindex");
				}
				// break;
			}
			default : throw new DMLRuntimeException("Internal Error - Invalid GPU Unary aggregate function!");
		}
	}

	/**
	 * Helper method to square a matrix in GPU memory
	 * @param instName the invoking instruction's name for record {@link Statistics}.
	 * @param in		input matrix on GPU
	 * @param out		output matrix on GPU
	 * @param rlen	row length
	 * @param clen	column length
	 * @throws DMLRuntimeException if error
	 */
	private static void squareMatrix(String instName, Pointer in, Pointer out, int rlen, int clen) throws DMLRuntimeException {
		ScalarOperator power2op = new RightScalarOperator(Power.getPowerFnObject(), 2);
		matrixScalarOp(instName, in, 2, rlen, clen, out, power2op);
	}

	/**
	 * Do a simple reduction, the output of which is a single value
	 * @param kernelFunction 	name of the kernel function to invoke
	 * @param in							{@link Pointer} to matrix in device memory
	 * @param n								size of array
	 * @return	the reduced value
	 * @throws DMLRuntimeException if DMLRuntimeException occurs
	 */
	private static double reduceAll(String instName, String kernelFunction, Pointer in, int n) throws DMLRuntimeException {
		int[] tmp = getKernelParamsForReduceAll(n);
		int blocks = tmp[0], threads = tmp[1], sharedMem = tmp[2];

		Pointer tempOut = JCudaObject.allocate(instName, n * Sizeof.DOUBLE);

		long t1=0,t2=0,t3=0;

		if (GPUStatistics.DISPLAY_STATISTICS) t1 = System.nanoTime();
		kernels.launchKernel(kernelFunction, new ExecutionConfig(blocks, threads, sharedMem), in, tempOut, n);
		cudaDeviceSynchronize();
		if (GPUStatistics.DISPLAY_STATISTICS) GPUStatistics.maintainCPMiscTimes(instName, GPUInstruction.MISC_TIMER_REDUCE_ALL_KERNEL, System.nanoTime() - t1);

		int s = blocks;
		while (s > 1) {
			tmp = getKernelParamsForReduceAll(s);
			blocks = tmp[0]; threads = tmp[1]; sharedMem = tmp[2];
			if (GPUStatistics.DISPLAY_STATISTICS) t2 = System.nanoTime();
			kernels.launchKernel(kernelFunction, new ExecutionConfig(blocks, threads, sharedMem),
							tempOut, tempOut, s);
			if (GPUStatistics.DISPLAY_STATISTICS) GPUStatistics.maintainCPMiscTimes(instName, GPUInstruction.MISC_TIMER_REDUCE_ALL_KERNEL, System.nanoTime() - t2);
			s = (s + (threads*2-1)) / (threads*2);
		}
		double[] result = {-1f};

		if (GPUStatistics.DISPLAY_STATISTICS) t3 = System.nanoTime();
		cudaMemcpy(Pointer.to(result), tempOut, Sizeof.DOUBLE, cudaMemcpyDeviceToHost);
		if (GPUStatistics.DISPLAY_STATISTICS) GPUStatistics.maintainCPMiscTimes(instName, GPUInstruction.MISC_TIMER_DEVICE_TO_HOST, System.nanoTime() - t3);

		cudaFreeHelper(instName, tempOut);
		return result[0];
	}

	/**
	 * Do a reduction by row. Data is reduced per row and the
	 * resulting vector is calculated.
	 * @param kernelFunction 	name of the kernel function to invoke
	 * @param in							{@link Pointer} to input matrix in device memory (size - rows * columns)
	 * @param out							{@link Pointer} to output matrix in device memory (size - rows * 1)
	 * @param rows						number of rows in input matrix
	 * @param cols						number of columns in input matrix
	 * @throws DMLRuntimeException if DMLRuntimeException occurs
	 */
	private static void reduceRow(String instName, String kernelFunction, Pointer in, Pointer out, int rows, int cols) throws DMLRuntimeException {
		int[] tmp = getKernelParamsForReduceByRow(rows, cols);
		int blocks = tmp[0], threads = tmp[1], sharedMem = tmp[2];

		long t0=0;
		if (GPUStatistics.DISPLAY_STATISTICS) t0 = System.nanoTime();
		kernels.launchKernel(kernelFunction, new ExecutionConfig(blocks, threads, sharedMem),
						in, out, rows, cols);
		cudaDeviceSynchronize();
		if (GPUStatistics.DISPLAY_STATISTICS) GPUStatistics.maintainCPMiscTimes(instName, GPUInstruction.MISC_TIMER_REDUCE_ROW_KERNEL, System.nanoTime() - t0);

	}

	/**
	 * Do a reduction by column. Data is reduced per column and the
	 * resulting vector is calculated.
	 * @param kernelFunction 	name of the kernel function to invoke
	 * @param in							{@link Pointer} to input matrix in device memory (size - rows * columns)
	 * @param out							{@link Pointer} to output matrix in device memory (size - 1 * cols)
	 * @param rows						number of rows in input matrix
	 * @param cols						number of columns in input matrix
	 * @throws DMLRuntimeException if DMLRuntimeException occurs
	 */
	private static void reduceCol(String instName, String kernelFunction, Pointer in, Pointer out, int rows, int cols) throws DMLRuntimeException {
		int[] tmp = getKernelParamsForReduceByCol(rows, cols);
		int blocks = tmp[0], threads = tmp[1], sharedMem = tmp[2];

		long t0=0;
		if (GPUStatistics.DISPLAY_STATISTICS) t0 = System.nanoTime();
		kernels.launchKernel(kernelFunction, new ExecutionConfig(blocks, threads, sharedMem),
						in, out, rows, cols);
		cudaDeviceSynchronize();
		if (GPUStatistics.DISPLAY_STATISTICS) GPUStatistics.maintainCPMiscTimes(instName, GPUInstruction.MISC_TIMER_REDUCE_COL_KERNEL, System.nanoTime() - t0);
	}

	/**
	 * Get threads, blocks and shared memory for a reduce all operation
	 * @param n size of input array
	 * @return integer array containing {blocks, threads, shared memory}
	 */
	private static int[] getKernelParamsForReduceAll(int n) throws DMLRuntimeException{
		final int MAX_THREADS = getMaxThreads();
		final int MAX_BLOCKS = getMaxBlocks();
		final int WARP_SIZE = getWarpSize();
		int threads = (n < MAX_THREADS *2) ? nextPow2((n + 1)/ 2) : MAX_THREADS;

		int blocks = (n + (threads * 2 - 1)) / (threads * 2);
		blocks = Math.min(MAX_BLOCKS, blocks);

		int sharedMemSize = threads * Sizeof.DOUBLE;
		if (threads <= WARP_SIZE){
			sharedMemSize *= 2;
		}
		return new int[] {blocks, threads, sharedMemSize};
	}

	/**
	 * Get threads, blocks and shared memory for a reduce by row operation
	 * @param rows number of rows in input matrix
	 * @param cols number of columns in input matrix
	 * @return integer array containing {blocks, threads, shared memory}
	 */
	private static int[] getKernelParamsForReduceByRow(int rows, int cols) throws DMLRuntimeException {
		final int WARP_SIZE = getWarpSize();
		final int MAX_THREADS = getMaxThreads();
		int threads = (cols < MAX_THREADS *2) ? nextPow2((cols + 1)/ 2) : MAX_THREADS;
		int blocks = rows;
		int sharedMemSize = threads * Sizeof.DOUBLE;
		if (threads <= WARP_SIZE){
			sharedMemSize *=2;
		}
		return new int[] {blocks, threads, sharedMemSize};
	}

	private static int[] getKernelParamsForReduceByCol(int rows, int cols) throws DMLRuntimeException {
		final int MAX_THREADS = getMaxThreads();
		final int MAX_BLOCKS = getMaxBlocks();
		final int WARP_SIZE = getWarpSize();
		int threads = Math.min(cols, MAX_THREADS);
		int blocks = Math.min(cols/MAX_THREADS, MAX_BLOCKS);
		if (cols % MAX_THREADS != 0) blocks++;
		int sharedMemSize = threads * Sizeof.DOUBLE;
		if (threads <= WARP_SIZE){
			sharedMemSize *=2;
		}
		return new int[] {blocks, threads, sharedMemSize};
	}


	private static int nextPow2(int x)
	{
		--x;
		x |= x >> 1;
		x |= x >> 2;
		x |= x >> 4;
		x |= x >> 8;
		x |= x >> 16;
		return ++x;
	}

	//********************************************************************/
	//****************  END OF UNARY AGGREGATE Functions *****************/
	//********************************************************************/


	//********************************************************************/
	//************ Matrix-Matrix & Matrix-Scalar Functions ***************/
	//********************************************************************/

	/**
	 * Entry point to perform elementwise matrix-scalar operation specified by op
	 *
	 * @param ec execution context
	 * @param instName the invoking instruction's name for record {@link Statistics}.
	 * @param in input matrix
	 * @param outputName output matrix name
	 * @param isInputTransposed true if input transposed
	 * @param op scalar operator
	 * @throws DMLRuntimeException if DMLRuntimeException occurs
	 */
	public static void matrixScalarArithmetic(ExecutionContext ec, String instName, MatrixObject in, String outputName, boolean isInputTransposed, ScalarOperator op) throws DMLRuntimeException {
		double constant = op.getConstant();
		boolean isCUDALibAvailable = (op.fn instanceof Multiply
						|| (op.fn instanceof Divide && op instanceof RightScalarOperator && constant != 0)) && !isSparseAndEmpty(in);
		if(!isCUDALibAvailable) {
			if(constant == 0) {
				if(op.fn instanceof Plus || (op.fn instanceof Minus && op instanceof RightScalarOperator) || op.fn instanceof Or) {
					deviceCopy(ec, instName, in, outputName, isInputTransposed);
				}
				else if(op.fn instanceof Multiply || op.fn instanceof And) {
					setOutputToConstant(ec, instName, 0.0, outputName);
				}
				else if(op.fn instanceof Power) {
					setOutputToConstant(ec, instName, 1.0, outputName);
				}
				else if(op.fn instanceof Divide && isSparseAndEmpty(in)) {
					setOutputToConstant(ec, instName, Double.NaN, outputName);
				}
				else if(op.fn instanceof Divide) {
					//For division, IEEE 754 defines x/0.0 as INFINITY and 0.0/0.0 as NaN.
					compareAndSet(ec, instName, in, outputName, 0.0, 1e-6, Double.NaN, Double.POSITIVE_INFINITY, Double.POSITIVE_INFINITY);
				}
				else {
					// TODO: Potential to optimize
					matrixScalarOp(ec, instName, in, outputName, isInputTransposed, op);
				}
			}
			else if(constant == 1.0 && op.fn instanceof Or) {
				setOutputToConstant(ec, instName, 1.0, outputName);
			}
			else if(constant == 1.0 && (op.fn instanceof And || op.fn instanceof Power)) {
				deviceCopy(ec, instName, in, outputName, isInputTransposed);
			}
			else {
				matrixScalarOp(ec, instName, in, outputName, isInputTransposed, op);
			}
		}
		else {
			double alpha = 0;
			if(op.fn instanceof Multiply) {
				alpha = op.getConstant();
			}
			else if(op.fn instanceof Divide && op instanceof RightScalarOperator) {
				alpha = Math.pow(op.getConstant(), -1.0);
			}
			else {
				throw new DMLRuntimeException("Unsupported op");
			}

			// TODO: Performance optimization: Call cublasDaxpy if(in.getNumRows() == 1 || in.getNumColumns() == 1)
			// C = alpha* op( A ) + beta* op ( B )
			dgeam(ec, instName, in, in, outputName, isInputTransposed, isInputTransposed, alpha, 0.0);
		}
	}

	/**
	 * Performs elementwise operation specified by op of two input matrices in1 and in2
	 *
	 * @param ec execution context
	 * @param instName the invoking instruction's name for record {@link Statistics}.
	 * @param in1 input matrix 1
	 * @param in2 input matrix 2
	 * @param outputName output matrix name
	 * @param isLeftTransposed true if left-transposed
	 * @param isRightTransposed true if right-transposed
	 * @param op binary operator
	 * @throws DMLRuntimeException if DMLRuntimeException occurs
	 */
	public static void matrixScalarArithmetic(ExecutionContext ec, String instName, MatrixObject in1, MatrixObject in2,
																						String outputName, boolean isLeftTransposed, boolean isRightTransposed, BinaryOperator op) throws DMLRuntimeException {
		boolean isCUDALibAvailable = (op.fn instanceof Plus || op.fn instanceof Minus) && !isSparseAndEmpty(in1) && !isSparseAndEmpty(in2) && !isVector(in1) && !isVector(in2);
		if(!isCUDALibAvailable) {
			matrixMatrixOp(ec, instName, in1, in2, outputName, isLeftTransposed, isRightTransposed, op);
		}
		else {
			double alpha;
			double beta;
			if(op.fn instanceof Plus) {
				alpha = 1.0;
				beta = 1.0;
			}
			else if(op.fn instanceof Minus) {
				alpha = 1.0;
				beta = -1.0;
			}
			else {
				throw new DMLRuntimeException("Unsupported op");
			}
			// C = alpha* op( A ) + beta* op ( B )
			dgeam(ec, instName, in1, in2, outputName, isLeftTransposed, isRightTransposed, alpha, beta);
		}
	}

	/**
	 * Utility to do matrix-scalar operation kernel
	 * @param instName the invoking instruction's name for record {@link Statistics}.
	 * @param ec execution context
	 * @param in input matrix
	 * @param outputName output variable name
	 * @param isInputTransposed true if input is transposed
	 * @param op operator
	 * @throws DMLRuntimeException if DMLRuntimeException occurs
	 */
	private static void matrixScalarOp(ExecutionContext ec, String instName, MatrixObject in, String outputName, boolean isInputTransposed,
																		 ScalarOperator op) throws DMLRuntimeException {
		if(isInputTransposed)
			throw new DMLRuntimeException("Transposing the input is not supported");

		int rlenA = (int) in.getNumRows();
		int clenA = (int) in.getNumColumns();
		Pointer A = getDensePointer(in, instName); // TODO: FIXME: Implement sparse binCellSparseScalarOp kernel
		double scalar = op.getConstant();
		MatrixObject out = ec.getMatrixObject(outputName);
		getDenseMatrixOutputForGPUInstruction(ec, instName, outputName);	// Allocated the dense output matrix
		Pointer C = getDensePointer(out, instName);
		matrixScalarOp(instName, A, scalar, rlenA, clenA, C, op);
	}

	/**
	 * Helper method to launch binary scalar-matrix arithmetic operations CUDA kernel.
	 * This method is isolated to be taken advatage of from other operations
	 * as it accepts JCuda {@link Pointer} instances instead of {@link MatrixObject} instances.
	 * @param instName the invoking instruction's name for record {@link Statistics}.
	 * @param a					the dense input matrix (allocated on GPU)
	 * @param scalar		the scalar value to do the op
	 * @param rlenA			row length of matrix a
	 * @param clenA			column lenght of matrix a
	 * @param c					the dense output matrix
	 * @param op				operation to perform
	 * @throws DMLRuntimeException throws runtime exception
	 */
	private static void matrixScalarOp(String instName, Pointer a, double scalar, int rlenA, int clenA, Pointer c, ScalarOperator op) throws DMLRuntimeException {
		int isLeftScalar = (op instanceof LeftScalarOperator) ? 1 : 0;
    int size = rlenA * clenA;
		long t0=0;
    if (GPUStatistics.DISPLAY_STATISTICS) t0 = System.nanoTime();
		kernels.launchKernel("matrix_scalar_op",
						ExecutionConfig.getConfigForSimpleVectorOperations(size),
						a, scalar, c, size, getBinaryOp(op.fn), isLeftScalar);
		if (GPUStatistics.DISPLAY_STATISTICS) GPUStatistics.maintainCPMiscTimes(instName, GPUInstruction.MISC_TIMER_MATRIX_SCALAR_OP_KERNEL, System.nanoTime() - t0);
	}

	/**
	 * Utility to launch binary cellwise matrix-matrix operations CUDA kernel
	 *
	 * @param ec execution context
	 * @param instName the invoking instruction's name for record {@link Statistics}.
	 * @param in1 left input matrix
	 * @param in2 right input matrix
	 * @param outputName output variable name
	 * @param isLeftTransposed true if left matrix is transposed
	 * @param isRightTransposed true if right matrix is transposed
	 * @param op operator
	 * @throws DMLRuntimeException if DMLRuntimeException occurs
	 */
	private static void matrixMatrixOp(ExecutionContext ec, String instName, MatrixObject in1, MatrixObject in2,
																		 String outputName, boolean isLeftTransposed, boolean isRightTransposed, BinaryOperator op) throws DMLRuntimeException {

		boolean isEmpty1 = isSparseAndEmpty(in1);
		boolean isEmpty2 = isSparseAndEmpty(in2);
		int rlenA = (int) in1.getNumRows();
		int rlenB = (int) in2.getNumRows();
		int clenA = (int) in1.getNumColumns();
		int clenB = (int) in2.getNumColumns();
		int vecStatusA = getVectorStatus(rlenA, clenA).code();
		int vecStatusB = getVectorStatus(rlenB, clenB).code();

		if (isEmpty1 && isEmpty2){
			MatrixObject out = ec.getMatrixObject(outputName);
			ec.allocateGPUMatrixObject(outputName);
			// When both inputs are empty, the output is empty too (except in the case of division)
			if (op.fn instanceof Divide) {
				((JCudaObject) out.getGPUObject()).allocateAndFillDense(Double.NaN);
			} else {
				((JCudaObject) out.getGPUObject()).allocateSparseAndEmpty();
			}
		}
		// Check for M1 * M2 when M1 is empty; if M2 is a vector then fallback to general case
		else if(isEmpty1 && clenB != 1 && rlenB != 1) {
			// C = empty_in1 op in2 ==> becomes ==> C = 0.0 op in2
			matrixScalarArithmetic(ec, instName, in2, outputName, isRightTransposed, new LeftScalarOperator(op.fn, 0.0));
		}
		// Check for M1 * M2 when M2 is empty; if M1 is a vector then fallback to general case
		else if(isEmpty2 && clenA != 1 && rlenA != 1) {
			// C = in1 op empty_in2 ==> becomes ==> C = in1 op 0.0
			matrixScalarArithmetic(ec, instName, in1, outputName, isLeftTransposed, new RightScalarOperator(op.fn, 0.0));
		}
		else {
			Pointer A = getDensePointer(in1, instName); // TODO: FIXME: Implement sparse binCellSparseOp kernel
			Pointer B = getDensePointer(in2, instName); // TODO: FIXME: Implement sparse binCellSparseOp kernel

			MatrixObject out = ec.getMatrixObject(outputName);
			getDenseMatrixOutputForGPUInstruction(ec, instName, outputName);	// Allocated the dense output matrix
			Pointer C = getDensePointer(out, instName);

			int maxRlen = Math.max(rlenA, rlenB);
			int maxClen = Math.max(clenA, clenB);

			matrixMatrixOp(instName, A, B, maxRlen, maxClen, vecStatusA, vecStatusB, C, op);
		}
	}

	/**
	 * Do an elementwise matrix-matrix arithmetic operation on the GPU
	 * c = a op b
	 * Either rows and cols in A are the same as in B or
	 * one of them is a vector or both are.
	 * @param instName the invoking instruction's name for record {@link Statistics}.
	 * @param a						The input matrix a allocated on the GPU
	 * @param b						The input matrix b allocated on the GPU
	 * @param maxRlen			the maximum of the row lengths between a & b
	 * @param maxClen			the maximum of the column lengths between a & b
	 * @param vecStatusA	if matrix A is a vector
	 * @param vecStatusB	if matrix B is a vector
	 * @param c						output matrix of size (maxRlen, maxClen) allocated on GPU
	 * @param op					the operation to perform
	 * @throws DMLRuntimeException
	 */
	private static void matrixMatrixOp(String instName, Pointer a, Pointer b, int maxRlen, int maxClen, int vecStatusA, int vecStatusB, Pointer c, BinaryOperator op) throws DMLRuntimeException {
		long t0=0;
		if (GPUStatistics.DISPLAY_STATISTICS) t0 = System.nanoTime();
		kernels.launchKernel("matrix_matrix_cellwise_op",
            ExecutionConfig.getConfigForSimpleMatrixOperations(maxRlen, maxClen),
						a, b, c, maxRlen, maxClen, vecStatusA, vecStatusB, getBinaryOp(op.fn));
		if (GPUStatistics.DISPLAY_STATISTICS) GPUStatistics.maintainCPMiscTimes(instName, GPUInstruction.MISC_TIMER_MATRIX_MATRIX_CELLWISE_OP_KERNEL, System.nanoTime() - t0);
	}

	/**
	 * This enum declares the different vector shapes
	 * as they recognized in the invoked CUDA kernel(s).
	 */
	enum VectorShape {
		COLUMN 	(1),
		ROW 		(2),
		NONE 		(0);
		private final int code;
		VectorShape(int code) {
			this.code = code;
		}
		int code() { return code; }
	}

	/**
	 * Given the number of rows and columns, returns
	 * whether this is a row vector, column vector or neither.
	 * @param rows
	 * @param cols
	 * @return 1 for column vector, 2 for row vector, 0 for neither
	 */
	private static VectorShape getVectorStatus(long rows, long cols) {
		if(cols == 1)
			return VectorShape.COLUMN;
		else if(rows == 1)
			return VectorShape.ROW;
		else
			return VectorShape.NONE;
	}

	private static boolean isVector(MatrixObject in) {
		return in.getNumRows() == 1 || in.getNumColumns() == 1;
	}

	private static boolean isSparseAndEmpty(MatrixObject in1) {
		boolean isSparse1 = isInSparseFormat(in1);
		boolean isEmpty1 = isSparse1 && (((JCudaObject)in1.getGPUObject()).jcudaSparseMatrixPtr.nnz == 0);
		return isEmpty1;
	}

	private static void deviceCopy(ExecutionContext ec, String instName, MatrixObject src, String outputName, boolean isInputTransposed) throws DMLRuntimeException {
		if(!isInputTransposed)
			deviceCopy(ec, instName, src, outputName);
		else
			transpose(ec, instName, src, outputName);
	}

	/**
	 * Performs a deep device copy of a matrix on the GPU
	 *
	 * @param ec execution context
	 * @param instName the invoking instruction's name for record {@link Statistics}.
	 * @param src source matrix
	 * @param outputName destination variable name
	 * @throws DMLRuntimeException if DMLRuntimeException occurs
	 */
	private static void deviceCopy(ExecutionContext ec, String instName, MatrixObject src, String outputName) throws DMLRuntimeException {
		Pointer srcPtr = getDensePointer(src, instName); // TODO: FIXME: Implement sparse kernel
		MatrixObject out = ec.getMatrixObject(outputName);
		getDenseMatrixOutputForGPUInstruction(ec, instName, outputName);	// Allocated the dense output matrix
		Pointer destPtr = getDensePointer(out, instName);
		deviceCopy(instName, srcPtr, destPtr, (int)src.getNumRows(), (int)src.getNumColumns());
	}

	private static void compareAndSet(ExecutionContext ec, String instName, MatrixObject in, String outputName, double compareVal,  double tolerance,
																		double ifEqualsVal, double ifLessThanVal, double ifGreaterThanVal) throws DMLRuntimeException {
		Pointer A = getDensePointer(in, instName); // TODO: FIXME: Implement sparse kernel
		MatrixObject out = ec.getMatrixObject(outputName);
		getDenseMatrixOutputForGPUInstruction(ec, instName, outputName);	// Allocated the dense output matrix
		Pointer ret = getDensePointer(out, instName);
		int rlen = (int) out.getNumRows();
		int clen = (int) out.getNumColumns();
		// out.getMatrixCharacteristics().setNonZeros(rlen*clen);
		// compareAndSet(double* A,  double* ret, int rlen, int clen, double compareVal, double ifEqualsVal, double ifNotEqualsVal)
		long t0=0;
		if (GPUStatistics.DISPLAY_STATISTICS) t0 = System.nanoTime();
		kernels.launchKernel("compare_and_set",
						ExecutionConfig.getConfigForSimpleMatrixOperations(rlen, clen),
						A, ret, rlen, clen, compareVal, tolerance, ifEqualsVal, ifLessThanVal, ifGreaterThanVal);
		if (GPUStatistics.DISPLAY_STATISTICS) GPUStatistics.maintainCPMiscTimes(instName, GPUInstruction.MISC_TIMER_COMPARE_AND_SET_KERNEL, System.nanoTime() - t0);
	}

	/**
	 * Fills an an array on the GPU with a given scalar value
	 * @param ec					currently active instance of the {@link ExecutionContext}
	 * @param instName name of the invoking instruction to record{@link Statistics}.
	 * @param constant		scalar value with which to fill the matrix
	 * @param outputName	(internal) name of the matrix that is to be filled
	 * @throws DMLRuntimeException if error
	 */
	private static void setOutputToConstant(ExecutionContext ec, String instName, double constant, String outputName) throws DMLRuntimeException {
		if(constant == 0) {
			// TODO: Create sparse empty block instead
		}
		MatrixObject out = ec.getMatrixObject(outputName);
		getDenseMatrixOutputForGPUInstruction(ec, instName, outputName);	// Allocated the dense output matrix
		Pointer A = getDensePointer(out, instName);
		int rlen = (int) out.getNumRows();
		int clen = (int) out.getNumColumns();
//	    if(constant == 0) {
//	    	out.getMatrixCharacteristics().setNonZeros(0);
//	    }
//	    else {
//	    	out.getMatrixCharacteristics().setNonZeros(rlen*clen);
//	    }
		// dense_matrix_set(double* A,  double scalar, int rlen, int clen)

		long t0=0;
		if (GPUStatistics.DISPLAY_STATISTICS) t0 = System.nanoTime();
		int size = rlen * clen;
		kernels.launchKernel("fill", ExecutionConfig.getConfigForSimpleVectorOperations(size),
						A, constant, size);
		//		kernels.launchKernel("dense_matrix_set",
		//						ExecutionConfig.getConfigForSimpleMatrixOperations(rlen, clen),
		//						A, constant, rlen, clen);
		if (GPUStatistics.DISPLAY_STATISTICS) GPUStatistics.maintainCPMiscTimes(instName, GPUInstruction.MISC_TIMER_FILL_KERNEL, System.nanoTime() - t0);
	}

	/**
	 * Performs a deep copy of input device double pointer corresponding to matrix
	 * @param instName the invoking instruction's name for record {@link Statistics}.
	 * @param src source matrix
	 * @param dest destination matrix
	 * @param rlen number of rows
	 * @param clen number of columns
	 * @throws DMLRuntimeException if DMLRuntimeException occurs
	 */
	private static void deviceCopy(String instName, Pointer src, Pointer dest, int rlen, int clen) throws DMLRuntimeException {
		long t0=0;
		if (GPUStatistics.DISPLAY_STATISTICS) t0 = System.nanoTime();
		//kernels.launchKernel("dense_matrix_copy",
		//				ExecutionConfig.getConfigForSimpleMatrixOperations(rlen, clen),
		//				src, dest, rlen, clen);
		int size = rlen * clen * Sizeof.DOUBLE;
		cudaMemcpy(dest, src, size, cudaMemcpyDeviceToDevice);
		if (GPUStatistics.DISPLAY_STATISTICS) GPUStatistics.maintainCPMiscTimes(instName, GPUInstruction.MISC_TIMER_DEVICE_TO_DEVICE, System.nanoTime() - t0);
	}

	/**
	 * Helper function to get numeric value for binary op.
	 * This number is passed down to the CUDA kernel
	 * and the appropriate binary operation is performed on the GPU.
	 * op = {0=plus, 1=minus, 2=multiply, 3=divide, 4=power,
	 * 5=less, 6=lessequal, 7=greater, 8=greaterequal, 9=equal, 10=notequal,
	 * 11=min, 12=max, 13=and, 14=or, 15=log}
	 */
	private static int getBinaryOp(ValueFunction fn) throws DMLRuntimeException {
		if(fn instanceof Plus) return 0;
		else if(fn instanceof Minus) return 1;
		else if(fn instanceof Multiply) return 2;
		else if(fn instanceof Divide) return 3;
		else if(fn instanceof Power) return 4;
		else if(fn instanceof LessThan) return 5;
		else if(fn instanceof LessThanEquals) return 6;
		else if(fn instanceof GreaterThan) return 7;
		else if(fn instanceof GreaterThanEquals) return 8;
		else if(fn instanceof Equals) return 9;
		else if(fn instanceof NotEquals) return 10;
		else if(fn instanceof And) return 13;
		else if(fn instanceof Or) return 14;
		else if(fn instanceof Multiply2) return 2;
		else if(fn instanceof Power2) return 4;

		throw new DMLRuntimeException("The given value function is not supported:" + fn.getClass().getName());
	}

	/**
	 * Performs sparse and dense dgeam given two input matrices
	 * C = alpha* op( A ) + beta* op ( B )
	 * where op = transpose or not (specified by isLeftTransposed and isRightTransposed).
	 *
	 * @param ec execution context
	 * @param instName the invoking instruction's name for record {@link Statistics}.
	 * @param in1 left input matrix
	 * @param in2 right input matrix
	 * @param outputName output variable name
	 * @param isLeftTransposed true if left matrix is transposed
	 * @param isRightTransposed true if right matrix is transposed
	 * @param alpha alpha
	 * @param beta beta
	 * @throws DMLRuntimeException if DMLRuntimeException occurs
	 */
	private static void dgeam(ExecutionContext ec, String instName, MatrixObject in1, MatrixObject in2, String outputName,
														boolean isLeftTransposed, boolean isRightTransposed, double alpha, double beta) throws DMLRuntimeException {
		Pointer alphaPtr = pointerTo(alpha);
		Pointer betaPtr = pointerTo(beta);
		int transa = isLeftTransposed ? CUBLAS_OP_T : CUBLAS_OP_N;
		int transb = isRightTransposed ? CUBLAS_OP_T : CUBLAS_OP_N;
		int m = (int) in1.getNumRows();
		int n = (int) in1.getNumColumns();
		if(!isLeftTransposed && isRightTransposed) {
			m = (int) in1.getNumColumns();
			n = (int) in1.getNumRows();
		}
		int lda = isLeftTransposed ? n : m;
		int ldb = isRightTransposed ? n : m;
		int ldc = m;

		MatrixObject out = ec.getMatrixObject(outputName);
		boolean isSparse1 = isInSparseFormat(in1);
//		boolean isEmpty1 = isSparse1 && (((JCudaObject)in1.getGPUObject()).jcudaSparseMatrixPtr.nnz == 0);
		boolean isSparse2 = isInSparseFormat(in2);
//		boolean isEmpty2 = isSparse2 && (((JCudaObject)in2.getGPUObject()).jcudaSparseMatrixPtr.nnz == 0);

		long t0=0,t1=0;
		// TODO: Implement sparse-dense matrix cublasDgeam kernel
		if(isSparse1 || isSparse2) {
			// Invoke cuSparse when either are in sparse format
			// Perform sparse-sparse dgeam
			if(!isInSparseFormat(in1)) {
				if (GPUStatistics.DISPLAY_STATISTICS) t0 = System.nanoTime();
				((JCudaObject)in1.getGPUObject()).denseToSparse();
				if (GPUStatistics.DISPLAY_STATISTICS) GPUStatistics.maintainCPMiscTimes(instName, GPUInstruction.MISC_TIMER_DENSE_TO_SPARSE, System.nanoTime() - t0);
			}
			CSRPointer A = ((JCudaObject)in1.getGPUObject()).jcudaSparseMatrixPtr;
			if(!isInSparseFormat(in2)) {
				if (GPUStatistics.DISPLAY_STATISTICS) t0 = System.nanoTime();
				((JCudaObject)in2.getGPUObject()).denseToSparse();
				if (GPUStatistics.DISPLAY_STATISTICS) GPUStatistics.maintainCPMiscTimes(instName, GPUInstruction.MISC_TIMER_DENSE_TO_SPARSE, System.nanoTime() - t0);
			}
			CSRPointer B = ((JCudaObject)in2.getGPUObject()).jcudaSparseMatrixPtr;

			ec.allocateGPUMatrixObject(outputName);

			if (GPUStatistics.DISPLAY_STATISTICS) t1 = System.nanoTime();
			CSRPointer C = CSRPointer.allocateForDgeam(cusparseHandle, A, B, m, n);
			if (GPUStatistics.DISPLAY_STATISTICS) GPUStatistics.maintainCPMiscTimes(instName, GPUInstruction.MISC_TIMER_SPARSE_ALLOCATE_LIB, System.nanoTime() - t1);

			((JCudaObject)out.getGPUObject()).setSparseMatrixCudaPointer(C);
			long sizeOfC = CSRPointer.estimateSize(C.nnz, out.getNumRows());
			out.getGPUObject().setDeviceModify(sizeOfC);
			if (GPUStatistics.DISPLAY_STATISTICS) t0 = System.nanoTime();
			JCusparse.cusparseDcsrgeam(cusparseHandle, m, n, alphaPtr, A.descr, (int)A.nnz, A.val, A.rowPtr, A.colInd, betaPtr,
							B.descr, (int)B.nnz, B.val, B.rowPtr, B.colInd,
							C.descr, C.val, C.rowPtr, C.colInd);
			cudaDeviceSynchronize();
			if (GPUStatistics.DISPLAY_STATISTICS) GPUStatistics.maintainCPMiscTimes(instName, GPUInstruction.MISC_TIMER_SPARSE_DGEAM_LIB, System.nanoTime() - t0);
		}
		else {
			// Dense-Dense dgeam
			Pointer A = getDensePointer(in1, instName);
			Pointer B = getDensePointer(in2, instName);
			getDenseMatrixOutputForGPUInstruction(ec, instName, outputName);	// Allocated the dense output matrix
			Pointer C = getDensePointer(out, instName);

			if (GPUStatistics.DISPLAY_STATISTICS) t0 = System.nanoTime();
			JCublas2.cublasDgeam(cublasHandle, transa, transb, m, n, alphaPtr, A, lda, betaPtr, B, ldb, C, ldc);
			if (GPUStatistics.DISPLAY_STATISTICS) GPUStatistics.maintainCPMiscTimes(instName, GPUInstruction.MISC_TIMER_DENSE_DGEAM_LIB, System.nanoTime() - t0);
		}
	}


	//********************************************************************/
	//****** End of Matrix-Matrix & Matrix-Scalar Functions **************/
	//********************************************************************/



	//********************************************************************/
	//************************ Re-org Functions **************************/
	//********************************************************************/

	/**
	 * Transposes the input matrix using cublasDgeam
	 *
	 * @param ec execution context
	 * @param instName the invoking instruction's name for record {@link Statistics}.
	 * @param in input matrix
	 * @param outputName output matrix name
	 * @throws DMLRuntimeException if DMLRuntimeException occurs
	 */
	public static void transpose(ExecutionContext ec, String instName, MatrixObject in, String outputName) throws DMLRuntimeException {
		// C = alpha* op( A ) + beta* op ( B )
		// = 1.0 * A^T + 0.0 * A^T
		dgeam(ec, instName, in, in, outputName, true, true, 1.0, 0.0);
	}

	//********************************************************************/
	//******************* End of Re-org Functions ************************/
	//********************************************************************/



	//********************************************************************/
	//************************ Builtin Functions *************************/
	//********************************************************************/

	/**
	 * Performs an "exp" operation on a matrix on the GPU
	 * @param ec	execution context
	 * @param instName the invoking instruction's name for record {@link Statistics}.
	 * @param in1	input matrix
	 * @param outputName	output matrix name
	 * @throws DMLRuntimeException	if DMLRuntimeException occurs
	 */
	public static void exp(ExecutionContext ec, String instName, MatrixObject in1, String outputName) throws DMLRuntimeException {
		JCudaObject in = ((JCudaObject)in1.getGPUObject());
		boolean isSparseAndEmpty = in.isSparseAndEmpty();
		long t1=0;
		if (isSparseAndEmpty) {
			// e^0 = 1, create a dense block full of 1s
			MatrixObject out = ec.getMatrixObject(outputName);
			ec.allocateGPUMatrixObject(outputName);
			((JCudaObject)(out.getGPUObject())).allocateAndFillDense(1);
		} else {
			// Dense
			MatrixObject out = getDenseMatrixOutputForGPUInstruction(ec, instName, outputName);
			Pointer output = getDensePointer(out, instName);
			// If the input is in sparse format, convert it to dense.
			// The output will always be dense, because for all x, exp(x) > 0
			Pointer input = getDensePointer(in1, instName);
			int size = (int)(in1.getNumColumns() * in1.getNumRows());
			if (GPUStatistics.DISPLAY_STATISTICS) t1 = System.nanoTime();
			kernels.launchKernel("matrix_exp", ExecutionConfig.getConfigForSimpleVectorOperations(size),
							input, output, size);
			if (GPUStatistics.DISPLAY_STATISTICS) GPUStatistics.maintainCPMiscTimes(instName, GPUInstruction.MISC_TIMER_EXP_KERNEL, System.nanoTime() - t1);
		}
	}

	/**
	 * Performs daxpy operation
	 *
	 * @param ec execution context
	 * @param instName the invoking instruction's name for record {@link Statistics}.
	 * @param in1 input matrix 1
	 * @param in2 input matrix 2
	 * @param outputName output matrix name
	 * @param constant pointer constant
	 * @throws DMLRuntimeException if DMLRuntimeException occurs
	 */
	public static void axpy(ExecutionContext ec, String instName, MatrixObject in1, MatrixObject in2,
													String outputName,  double constant) throws DMLRuntimeException {
		Pointer A = getDensePointer(in1, instName);
		Pointer B = getDensePointer(in2, instName);
		MatrixObject out = ec.getMatrixObject(outputName);
		getDenseMatrixOutputForGPUInstruction(ec, instName, outputName);	// Allocated the dense output matrix
		Pointer C = getDensePointer(out, instName);
		Pointer alphaPtr = pointerTo(constant);
		long n = (in1.getNumRows()*in1.getNumColumns());
		// C <- A + alpha*B
		// becomes
		// C <- A
		// C <- alpha*B + C
		long t1=0, t2=0;
		if (GPUStatistics.DISPLAY_STATISTICS) t1 = System.nanoTime();
		cudaMemcpy(C, A, n*((long)jcuda.Sizeof.DOUBLE), cudaMemcpyDeviceToDevice);
		if (GPUStatistics.DISPLAY_STATISTICS) GPUStatistics.maintainCPMiscTimes(instName, GPUInstruction.MISC_TIMER_DEVICE_TO_DEVICE, System.nanoTime() - t1);

		if (GPUStatistics.DISPLAY_STATISTICS) t2 = System.nanoTime();
		JCublas2.cublasDaxpy(cublasHandle, (int) n, alphaPtr, B, 1, C, 1);
		if (GPUStatistics.DISPLAY_STATISTICS) GPUStatistics.maintainCPMiscTimes(instName, GPUInstruction.MISC_TIMER_DAXPY_LIB, System.nanoTime() - t2);
	}

	//********************************************************************/
	//*****************  END OF Builtin Functions ************************/
	//********************************************************************/

	/**
	 * Convenience method for debugging matrices on the GPU.
	 * @param in		Pointer to a double array (matrix) on the GPU
	 * @param rlen	row length
	 * @param clen	column length
	 */
	@SuppressWarnings("unused")
	private static void debugPrintMatrix(Pointer in, int rlen, int clen){
		double[] data = new double[rlen * clen];
		cudaMemcpy(Pointer.to(data), in, rlen*clen*Sizeof.DOUBLE, cudaMemcpyDeviceToHost);
		int k=0;
		for (int i=0; i<rlen; ++i){
			for (int j=0; j<clen; ++j){
				System.out.print(data[k]);
				k++;
			}
			System.out.println();
		}
	}

	/**
	 * Helper method to get the output block (allocated on the GPU)
	 * Also records performance information into {@link Statistics}
	 * @param ec		active {@link ExecutionContext}
	 * @param instName the invoking instruction's name for record {@link Statistics}.
	 * @param name	name of input matrix (that the {@link ExecutionContext} is aware of)
	 * @return	the matrix object
	 * @throws DMLRuntimeException	if an error occurs
	 */
	private static MatrixObject getDenseMatrixOutputForGPUInstruction(ExecutionContext ec, String instName, String name) throws DMLRuntimeException {
		long t0=0;
		if (GPUStatistics.DISPLAY_STATISTICS) t0 = System.nanoTime();
		Pair<MatrixObject, Boolean> mb = ec.getDenseMatrixOutputForGPUInstruction(name);
		if (mb.getValue())
			if (GPUStatistics.DISPLAY_STATISTICS)
				GPUStatistics.maintainCPMiscTimes(instName, GPUInstruction.MISC_TIMER_ALLOCATE_DENSE_OUTPUT, System.nanoTime() - t0);
		return mb.getKey();
	}

}
