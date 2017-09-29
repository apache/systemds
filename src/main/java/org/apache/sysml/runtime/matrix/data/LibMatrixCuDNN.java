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

import static jcuda.jcudnn.JCudnn.cudnnActivationForward;
import static jcuda.jcudnn.JCudnn.cudnnBatchNormalizationBackward;
import static jcuda.jcudnn.JCudnn.cudnnBatchNormalizationForwardInference;
import static jcuda.jcudnn.JCudnn.cudnnBatchNormalizationForwardTraining;
import static jcuda.jcudnn.JCudnn.cudnnConvolutionBackwardData;
import static jcuda.jcudnn.JCudnn.cudnnConvolutionBackwardFilter;
import static jcuda.jcudnn.JCudnn.cudnnConvolutionForward;
import static jcuda.jcudnn.JCudnn.cudnnCreateActivationDescriptor;
import static jcuda.jcudnn.JCudnn.cudnnCreateConvolutionDescriptor;
import static jcuda.jcudnn.JCudnn.cudnnCreateFilterDescriptor;
import static jcuda.jcudnn.JCudnn.cudnnCreatePoolingDescriptor;
import static jcuda.jcudnn.JCudnn.cudnnCreateTensorDescriptor;
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
import static jcuda.runtime.JCuda.cudaMemcpy;
import static jcuda.runtime.JCuda.cudaMemset;
import static jcuda.runtime.cudaMemcpyKind.cudaMemcpyDeviceToDevice;
import jcuda.CudaException;
import jcuda.Pointer;
import jcuda.Sizeof;
import jcuda.jcudnn.cudnnActivationDescriptor;
import jcuda.jcudnn.cudnnBatchNormMode;
import jcuda.jcudnn.cudnnConvolutionDescriptor;
import jcuda.jcudnn.cudnnConvolutionFwdPreference;
import jcuda.jcudnn.cudnnFilterDescriptor;
import jcuda.jcudnn.cudnnHandle;
import jcuda.jcudnn.cudnnPoolingDescriptor;
import jcuda.jcudnn.cudnnStatus;
import jcuda.jcudnn.cudnnTensorDescriptor;

import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;
import org.apache.sysml.hops.OptimizerUtils;
import org.apache.sysml.runtime.DMLRuntimeException;
import org.apache.sysml.runtime.controlprogram.caching.MatrixObject;
import org.apache.sysml.runtime.controlprogram.context.ExecutionContext;
import org.apache.sysml.runtime.instructions.gpu.GPUInstruction;
import org.apache.sysml.runtime.instructions.gpu.context.ExecutionConfig;
import org.apache.sysml.runtime.instructions.gpu.context.GPUContext;
import org.apache.sysml.utils.GPUStatistics;
import org.apache.sysml.utils.Statistics;

/**
 * This class contains method that invoke CuDNN operations.
 */
public class LibMatrixCuDNN extends LibMatrixCUDA {

	protected static int CONVOLUTION_PREFERENCE = cudnnConvolutionFwdPreference.CUDNN_CONVOLUTION_FWD_NO_WORKSPACE;
	private static final Log LOG = LogFactory.getLog(LibMatrixCuDNN.class.getName());

	protected static cudnnHandle getCudnnHandle(GPUContext gCtx) throws DMLRuntimeException {
		return gCtx.getCudnnHandle();
	}

	/**
	 * Does a 2D convolution followed by a bias_add
	 *
	 * @param gCtx     a valid {@link GPUContext}
	 * @param instName the invoking instruction's name for record {@link Statistics}.
	 * @param image    input image matrix object
	 * @param bias     bias matrix object
	 * @param filter   filter matrix object
	 * @param output   output matrix object
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
	 * @param intermediateMemoryBudget intermediate memory budget
	 * @throws DMLRuntimeException if error
	 */
	public static void conv2dBiasAdd(GPUContext gCtx, String instName, MatrixObject image, MatrixObject bias, MatrixObject filter, MatrixObject output, int N, int C, int H, int W,
			int K, int R, int S, int pad_h, int pad_w, int stride_h, int stride_w, int P, int Q, double intermediateMemoryBudget)
					throws DMLRuntimeException {
		conv2d(gCtx, instName, image, filter, output, N, C, H, W, K, R, S, pad_h, pad_w, stride_h, stride_w, P, Q, intermediateMemoryBudget);
		//cudaDeviceSynchronize;
		biasAdd(gCtx, instName, output, bias, output);
	}

	/**
	 * Performs a 2D convolution
	 * 
	 * @param gCtx a valid {@link GPUContext}
	 * @param instName the invoking instruction's name for record {@link Statistics}.
	 * @param image input matrix object
	 * @param filter filter matrix object
	 * @param outputBlock output matrix object
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
	 * @param intermediateMemoryBudget intermediate memory budget
	 * @throws DMLRuntimeException if error
	 */
	public static void conv2d(GPUContext gCtx, String instName, MatrixObject image, MatrixObject filter, MatrixObject outputBlock, int N, int C, int H, int W,
			int K, int R, int S, int pad_h, int pad_w, int stride_h, int stride_w, int P, int Q, double intermediateMemoryBudget) throws DMLRuntimeException {

		long CHW = C*H*W; long KPQ = K*P*Q; long CRS = C*R*S; 
		long NCHW = N*CHW; long NKPQ = N*KPQ; long KCRS = K*CRS;

		if(NCHW < maxNumDoublesOfCuDNNTensor && NKPQ < maxNumDoublesOfCuDNNTensor && KCRS < maxNumDoublesOfCuDNNTensor) {
			// Filter and output are accounted as dense in the memory estimation for conv2d
			double overhead = isInSparseFormat(gCtx, filter) ? OptimizerUtils.estimateSizeExactSparsity(K, CRS, 1.0) : 0;
			overhead += isInSparseFormat(gCtx, image) ? OptimizerUtils.estimateSizeExactSparsity(N, CHW, 1.0) : 0;

			Pointer filterPointer = getDensePointerForCuDNN(gCtx, filter, instName);
			Pointer dstPointer = getDensePointerForCuDNN(gCtx, outputBlock, instName);
			
			// Required for LibMatrixCuDNNConvolutionAlgorithm
			long workspaceLimit = (long) (intermediateMemoryBudget-overhead);
			int localN = overhead <= intermediateMemoryBudget ? N : 1;
			
			try(LibMatrixCuDNNConvolutionAlgorithm algo = 
					LibMatrixCuDNNConvolutionAlgorithm.cudnnGetConvolutionForwardAlgorithm(gCtx, instName, 
					localN, C, H, W, K, R, S, pad_h, pad_w, stride_h, stride_w, P, Q, workspaceLimit)) {
				if(localN == N) {
					// Perform all-input all-channel conv2d
					Pointer imagePointer = getDensePointerForCuDNN(gCtx, image, instName);
					cudnnConv2d(gCtx, instName, imagePointer, filterPointer, dstPointer, algo);
				}
				else {
					try(LibMatrixCuDNNInputRowFetcher imgFetcher = new LibMatrixCuDNNInputRowFetcher(gCtx, instName, image)) {
						for(int n = 0; n < N; n++) {
							// Perform one-input all-channel conv2d
							cudnnConv2d(gCtx, instName, imgFetcher.getNthRow(n), filterPointer, dstPointer.withByteOffset(n*KPQ*Sizeof.DOUBLE), algo);
						}
					}
				}
			}
		}
		else {
			throwCuDNNDimensionError(N, CHW, K, CRS, N, KPQ);
		}
	}


	/**
	 * Throw an user-friendly error that shows limitation of invoking a cuDNN kernel
	 *  
	 * @param dim1 input1 number of rows
	 * @param dim2 input1 number of columns
	 * @param dim3 input2 number of rows
	 * @param dim4 input2 number of columns
	 * @param dim5 output number of rows
	 * @param dim6 output number of columns
	 * @throws DMLRuntimeException the exception with the appropriate message
	 */
	private static void throwCuDNNDimensionError(long dim1, long dim2, long dim3, long dim4) throws DMLRuntimeException {
		throw new DMLRuntimeException("The dimensions of input/output matrices is too large to execute a CuDNN kernel. "
				+ "Max CuDNN matrix size:" + maxNumDoublesOfCuDNNTensor + ". "
				+ "Given input matrix dimensions: [" + dim1 + "," + dim2 + "]. Output dimension:  [" + dim3 + "," + dim4 + "].");
	}

	/**
	 * Throw an user-friendly error that shows limitation of invoking a cuDNN kernel
	 *  
	 * @param dim1 input1 number of rows
	 * @param dim2 input1 number of columns
	 * @param dim3 input2 number of rows
	 * @param dim4 input2 number of columns
	 * @param dim5 output number of rows
	 * @param dim6 output number of columns
	 * @throws DMLRuntimeException the exception with the appropriate message
	 */
	private static void throwCuDNNDimensionError(long dim1, long dim2, long dim3, long dim4, long dim5, long dim6) throws DMLRuntimeException {
		throw new DMLRuntimeException("The dimensions of input/output matrices is too large to execute a CuDNN kernel. "
				+ "Max CuDNN matrix size:" + maxNumDoublesOfCuDNNTensor + ". "
				+ "Given input matrix dimensions: [" + dim1 + "," + dim2 + "], [" + dim3 + "," + dim4 + "]. Output dimension: [" + dim5 + "," + dim6 + "]");
	}

	/**
	 * Performs 2D convolution
	 * Takes up an insignificant amount of intermediate space when CONVOLUTION_PREFERENCE is set to CUDNN_CONVOLUTION_FWD_NO_WORKSPACE
	 * Intermediate space is required by the filter descriptor and convolution descriptor which are metadata structures and don't scale with the size of the input
	 *
	 * @param gCtx     a valid {@link GPUContext}
	 * @param instName the invoking instruction's name for record {@link Statistics}.
	 * @param image    the input matrix (or image) allocated on the GPU
	 * @param filter   the filter allocated on the GPU
	 * @param output   the output matrix allocated on the GPU
	 * @param algo     cudnn algorithm wrapper
	 * @throws DMLRuntimeException if error
	 */
	private static void cudnnConv2d(GPUContext gCtx, String instName, Pointer image, Pointer filter, Pointer output, 
			LibMatrixCuDNNConvolutionAlgorithm algo)
					throws DMLRuntimeException {
		if(LOG.isTraceEnabled()) {
			LOG.trace("GPU : conv2d" + ", GPUContext=" + gCtx);
		}
		try {
			long t1 = 0;
			if (GPUStatistics.DISPLAY_STATISTICS) t1 = System.nanoTime();
			int status = cudnnConvolutionForward(getCudnnHandle(gCtx), one(),
					algo.nchwTensorDesc, image,
					algo.filterDesc, filter,
					algo.convDesc, algo.algo, algo.workSpace, algo.sizeInBytes, zero(),
					algo.nkpqTensorDesc, output);
			if (GPUStatistics.DISPLAY_STATISTICS)
				GPUStatistics.maintainCPMiscTimes(instName, GPUInstruction.MISC_TIMER_CONVOLUTION_FORWARD_LIB, System.nanoTime() - t1);
			if (status != cudnnStatus.CUDNN_STATUS_SUCCESS) {
				throw new DMLRuntimeException("Could not executed cudnnConvolutionForward: " + cudnnStatus.stringFor(status));
			}
		} catch (CudaException e) {
			throw new DMLRuntimeException("Error in conv2d in GPUContext " + gCtx.toString() + " from Thread " + Thread.currentThread().toString(), e);
		} 
	}

	/**
	 * This method computes the backpropogation errors for filter of convolution operation
	 * 
	 * @param gCtx   a valid {@link GPUContext}
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
	 * @param intermediateMemoryBudget intermediate memory budget
	 * @throws DMLRuntimeException if DMLRuntimeException occurs
	 */
	public static void conv2dBackwardFilter(GPUContext gCtx, String instName, MatrixObject image, MatrixObject dout,
			MatrixObject outputBlock, int N, int C, int H, int W, int K, int R,
			int S, int pad_h, int pad_w, int stride_h, int stride_w, int P,
			int Q, double intermediateMemoryBudget) throws DMLRuntimeException {
		long CHW = C*H*W; long KPQ = K*P*Q; long CRS = C*R*S; 
		long NCHW = N*CHW; long NKPQ = N*KPQ; long KCRS = K*CRS;
		
		
		if(NCHW < maxNumDoublesOfCuDNNTensor && NKPQ < maxNumDoublesOfCuDNNTensor && KCRS < maxNumDoublesOfCuDNNTensor) {
			Pointer dwPointer = getDensePointerForCuDNN(gCtx, outputBlock, instName);
			double overhead = isInSparseFormat(gCtx, image) ? OptimizerUtils.estimateSizeExactSparsity(N, CHW, 1.0) : 0;
			overhead += isInSparseFormat(gCtx, dout) ? OptimizerUtils.estimateSizeExactSparsity(N, KPQ, 1.0) : 0;

			// Required for LibMatrixCuDNNConvolutionAlgorithm
			long workspaceLimit = (long) (intermediateMemoryBudget-overhead);
			int localN = overhead <= intermediateMemoryBudget ? N : 1;
			
			try(LibMatrixCuDNNConvolutionAlgorithm algo = 
					LibMatrixCuDNNConvolutionAlgorithm.cudnnGetConvolutionBackwardFilterAlgorithm(gCtx, instName, 
					localN, C, H, W, K, R, S, pad_h, pad_w, stride_h, stride_w, P, Q, workspaceLimit)) {
				if(localN == N) {
					// Perform all-input all-channel conv2dBackwardFilter
					Pointer imagePointer = getDensePointerForCuDNN(gCtx, image, instName);
					Pointer doutPointer = getDensePointerForCuDNN(gCtx, dout, instName);
					cudnnConv2dBackwardFilter(gCtx, instName, imagePointer, doutPointer, dwPointer, algo);
				}
				else {
					try(LibMatrixCuDNNInputRowFetcher imgFetcher = new LibMatrixCuDNNInputRowFetcher(gCtx, instName, image);
						LibMatrixCuDNNInputRowFetcher doutFetcher = new LibMatrixCuDNNInputRowFetcher(gCtx, instName, dout)) {
						// Perform one-input conv2dBackwardFilter
						Pointer tempdwPointer = gCtx.allocate(KCRS*Sizeof.DOUBLE);
						for(int n = 0; n < N; n++) {
							long t0 = GPUStatistics.DISPLAY_STATISTICS ? System.nanoTime() : 0;
							cudaMemset(tempdwPointer, 0, KCRS*Sizeof.DOUBLE);
							if(GPUStatistics.DISPLAY_STATISTICS) GPUStatistics.maintainCPMiscTimes(instName, GPUInstruction.MISC_TIMER_SET_ZERO, System.nanoTime() - t0);
							// Perform one-input conv2dBackwardFilter
							cudnnConv2dBackwardFilter(gCtx, instName, imgFetcher.getNthRow(n), doutFetcher.getNthRow(n), tempdwPointer, algo);
							getCudaKernels(gCtx).launchKernel("inplace_add",
									ExecutionConfig.getConfigForSimpleMatrixOperations(K, toInt(CRS)),
									tempdwPointer, dwPointer, K, toInt(CRS));

						}
						// Deallocate temporary array to hold one element of input
						gCtx.cudaFreeHelper(tempdwPointer, true);
					}
				}
			}
		}
		else {
			throwCuDNNDimensionError(N, CHW, N, KPQ, K, CRS);
		}
	}

	/**
	 * This method computes the backpropogation errors for filter of convolution operation
	 * 
	 * @param gCtx   a valid {@link GPUContext}
	 * @param instName the invoking instruction's name for record {@link Statistics}.
	 * @param imagePointer pointer to input image
	 * @param doutPointer pointer to errors from next layer
	 * @param dwPointer  output errors
	 * @param algo     cudnn algorithm wrapper
	 * @throws DMLRuntimeException if DMLRuntimeException occurs
	 */
	private static void cudnnConv2dBackwardFilter(GPUContext gCtx, String instName, Pointer imagePointer, Pointer doutPointer,
			Pointer dwPointer, LibMatrixCuDNNConvolutionAlgorithm algo) throws DMLRuntimeException {
		if(LOG.isTraceEnabled()) {
			LOG.trace("GPU : conv2dBackwardFilter" + ", GPUContext=" + gCtx);
		}
		try {
			long t1 = GPUStatistics.DISPLAY_STATISTICS ? System.nanoTime() : 0;
			int status = cudnnConvolutionBackwardFilter(getCudnnHandle(gCtx), one(), algo.nchwTensorDesc, imagePointer,
					algo.nkpqTensorDesc, doutPointer, algo.convDesc, algo.algo, algo.workSpace, algo.sizeInBytes, zero(), algo.filterDesc, dwPointer);
			if (GPUStatistics.DISPLAY_STATISTICS)
				GPUStatistics.maintainCPMiscTimes(instName, GPUInstruction.MISC_TIMER_CONVOLUTION_BACKWARD_FILTER_LIB, System.nanoTime() - t1);
			if (status != jcuda.jcudnn.cudnnStatus.CUDNN_STATUS_SUCCESS) {
				throw new DMLRuntimeException("Could not executed cudnnConvolutionBackwardFilter: " + jcuda.jcudnn.cudnnStatus.stringFor(status));
			}
		} catch (CudaException e) {
			throw new DMLRuntimeException("Error in conv2d in GPUContext " + gCtx.toString() + " from Thread " + Thread.currentThread().toString(), e);
		} 
	}

	/**
	 * This method computes the backpropogation errors for previous layer of convolution operation
	 * 
	 * @param gCtx   a valid {@link GPUContext}
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
	 * @param intermediateMemoryBudget intermediate memory budget
	 * @throws DMLRuntimeException if DMLRuntimeException occurs
	 */
	public static void conv2dBackwardData(GPUContext gCtx, String instName, MatrixObject filter, MatrixObject dout,
			MatrixObject output, int N, int C, int H, int W, int K, int R,
			int S, int pad_h, int pad_w, int stride_h, int stride_w, int P,
			int Q, double intermediateMemoryBudget) throws DMLRuntimeException {
		long CHW = C*H*W; long KPQ = K*P*Q; long CRS = C*R*S; 
		long NCHW = N*CHW; long NKPQ = N*KPQ; long KCRS = K*CRS;

		if(NCHW < maxNumDoublesOfCuDNNTensor && NKPQ < maxNumDoublesOfCuDNNTensor && KCRS < maxNumDoublesOfCuDNNTensor) {
			// Filter and output are accounted as dense in the memory estimation for conv2dBackwardData
			double overhead = isInSparseFormat(gCtx, filter) ? OptimizerUtils.estimateSizeExactSparsity(K, CRS, 1.0) : 0;
			overhead += isInSparseFormat(gCtx, dout) ? OptimizerUtils.estimateSizeExactSparsity(N, KPQ, 1.0) : 0;
			Pointer filterPointer = getDensePointerForCuDNN(gCtx, filter, instName);
			Pointer dstPointer = getDensePointerForCuDNN(gCtx, output, instName);
			
			// Required for LibMatrixCuDNNConvolutionAlgorithm
			long workspaceLimit = (long) (intermediateMemoryBudget-overhead);
			int localN = overhead <= intermediateMemoryBudget ? N : 1;
			
			try(LibMatrixCuDNNConvolutionAlgorithm algo = 
					LibMatrixCuDNNConvolutionAlgorithm.cudnnGetConvolutionBackwardDataAlgorithm(gCtx, instName, 
					localN, C, H, W, K, R, S, pad_h, pad_w, stride_h, stride_w, P, Q, workspaceLimit)) {
				if(localN == N) {
					// Perform all-input all-channel conv2dBackwardData
					Pointer doutPointer = getDensePointerForCuDNN(gCtx, dout, instName);
					cudnnConv2dBackwardData(gCtx, instName, filterPointer, doutPointer, dstPointer, algo);
				}
				else {
					try(LibMatrixCuDNNInputRowFetcher doutFetcher = new LibMatrixCuDNNInputRowFetcher(gCtx, instName, dout)) {
						for(int n = 0; n < N; n++) {
							cudnnConv2dBackwardData(gCtx, instName, doutFetcher.getNthRow(n), filterPointer, dstPointer.withByteOffset(n*CHW*Sizeof.DOUBLE), algo);
						}
					}
				}
			}
		}
		else {
			throwCuDNNDimensionError(N, CHW, N, KPQ, K, CRS);
		}
	}

	/**
	 * This method computes the backpropogation errors for previous layer of convolution operation
	 * 
	 * @param gCtx   a valid {@link GPUContext}
	 * @param instName the invoking instruction's name for record {@link Statistics}.
	 * @param w pointer to filter used in conv2d
	 * @param dy pointer to errors from next layer
	 * @param dx pointer to  output errors
	 * @param algo cudnn algorithm wrapper
	 * @throws DMLRuntimeException if DMLRuntimeException occurs
	 */
	private static void cudnnConv2dBackwardData(GPUContext gCtx, String instName, Pointer w, Pointer dy,
			Pointer dx, LibMatrixCuDNNConvolutionAlgorithm algo) throws DMLRuntimeException {
		if(LOG.isTraceEnabled()) {
			LOG.trace("GPU : conv2dBackwardData" + ", GPUContext=" + gCtx);
		}
		try {
			long t1 = GPUStatistics.DISPLAY_STATISTICS ? System.nanoTime() : 0;
			int status = cudnnConvolutionBackwardData(getCudnnHandle(gCtx), one(), algo.filterDesc, w,
					algo.nkpqTensorDesc, dy, algo.convDesc, algo.algo, algo.workSpace, algo.sizeInBytes, zero(), algo.nchwTensorDesc, dx);
			if (GPUStatistics.DISPLAY_STATISTICS) GPUStatistics.maintainCPMiscTimes(instName, GPUInstruction.MISC_TIMER_CONVOLUTION_BACKWARD_DATA_LIB, System.nanoTime() - t1);

			if(status != jcuda.jcudnn.cudnnStatus.CUDNN_STATUS_SUCCESS) {
				throw new DMLRuntimeException("Could not executed cudnnConvolutionBackwardData: " + jcuda.jcudnn.cudnnStatus.stringFor(status));
			}
		} catch (CudaException e) {
			throw new DMLRuntimeException("Error in conv2d in GPUContext " + gCtx.toString() + " from Thread " + Thread.currentThread().toString(), e);
		}	
	}

	/**
	 * performs maxpooling on GPU by exploiting cudnnPoolingForward(...)
	 * @param gCtx   a valid {@link GPUContext}
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
	 * @param intermediateMemoryBudget intermediate memory budget
	 * @throws DMLRuntimeException if DMLRuntimeException occurs
	 */
	public static void maxpooling(GPUContext gCtx, String instName, MatrixObject image,
			MatrixObject outputBlock, int N, int C, int H, int W, int K, int R,
			int S, int pad_h, int pad_w, int stride_h, int stride_w, int P,
			int Q, double intermediateMemoryBudget) throws DMLRuntimeException {
		long CHW = C*H*W; long CPQ = C*P*Q;  
		long NCHW = N*CHW; long NCPQ = N*CPQ; 

		if(NCHW < maxNumDoublesOfCuDNNTensor && NCPQ < maxNumDoublesOfCuDNNTensor) {
			// Filter and output are accounted as dense in the memory estimation for conv2dBackwardData
			long overhead = isInSparseFormat(gCtx, image) ? OptimizerUtils.estimateSizeExactSparsity(N, CHW, 1.0) : 0;
			Pointer y = getDensePointerForCuDNN(gCtx, outputBlock, instName);
			if(overhead <= intermediateMemoryBudget) {
				Pointer x = getDensePointerForCuDNN(gCtx, image, instName);
				cudnnTensorDescriptor xDesc = allocateTensorDescriptor(gCtx, image, N, C, H, W);
				cudnnMaxpooling(gCtx, instName, x, xDesc, y, N, C, H, W, K, R, S, pad_h, pad_w, stride_h, stride_w, P, Q);
			}
			else {
				LibMatrixCuDNNInputRowFetcher imgFetcher = new LibMatrixCuDNNInputRowFetcher(gCtx, instName, image);
				cudnnTensorDescriptor xDesc = allocateTensorDescriptor(gCtx, image, N, C, H, W);
				for(int n = 0; n < N; n++) {
					cudnnMaxpooling(gCtx, instName, imgFetcher.getNthRow(n), xDesc, y.withByteOffset(n*CPQ*Sizeof.DOUBLE), 1, C, H, W, K, R, S, pad_h, pad_w, stride_h, stride_w, P, Q);
				}
				imgFetcher.close();
			}
		}
		else {
			throwCuDNNDimensionError(N, CHW, N, CPQ);
		}
	}

	private static void cudnnMaxpooling(GPUContext gCtx, String instName, Pointer x, cudnnTensorDescriptor xDesc,
			Pointer y, int N, int C, int H, int W, int K, int R,
			int S, int pad_h, int pad_w, int stride_h, int stride_w, int P,
			int Q) throws DMLRuntimeException {
		if(LOG.isTraceEnabled()) {
			LOG.trace("GPU : performMaxpooling" + ", GPUContext=" + gCtx);
		}

		cudnnPoolingDescriptor poolingDesc = null;

		try {
			long t1=0,t2=0;
			if (GPUStatistics.DISPLAY_STATISTICS) t1 = System.nanoTime();
			// Allocate descriptors
			cudnnTensorDescriptor yDesc = allocateTensorDescriptor(N, C, P, Q);
			poolingDesc = allocatePoolingDescriptor(R, S, pad_h, pad_w, stride_h, stride_w);
			if (GPUStatistics.DISPLAY_STATISTICS) GPUStatistics.maintainCPMiscTimes(instName, GPUInstruction.MISC_TIMER_CUDNN_INIT, System.nanoTime() - t1);

			if (GPUStatistics.DISPLAY_STATISTICS) t2 = System.nanoTime();
			int status = cudnnPoolingForward(getCudnnHandle(gCtx), poolingDesc, one(), xDesc, x, zero(), yDesc, y);
			if (GPUStatistics.DISPLAY_STATISTICS) GPUStatistics.maintainCPMiscTimes(instName, GPUInstruction.MISC_TIMER_MAXPOOLING_FORWARD_LIB, System.nanoTime() - t2);

			if(status != jcuda.jcudnn.cudnnStatus.CUDNN_STATUS_SUCCESS) {
				throw new DMLRuntimeException("Could not executed cudnnPoolingForward: " + jcuda.jcudnn.cudnnStatus.stringFor(status));
			}
		} catch (CudaException e) {
			throw new DMLRuntimeException("Error in conv2d in GPUContext " + gCtx.toString() + " from Thread " + Thread.currentThread().toString(), e);
		}
		finally {
			long t3=0;
			if (GPUStatistics.DISPLAY_STATISTICS) t3 = System.nanoTime();
			if(poolingDesc != null)
				jcuda.jcudnn.JCudnn.cudnnDestroyPoolingDescriptor(poolingDesc);
			if (GPUStatistics.DISPLAY_STATISTICS) GPUStatistics.maintainCPMiscTimes(instName, GPUInstruction.MISC_TIMER_CUDNN_CLEANUP, System.nanoTime() - t3);
		}
	}

	/**
	 * Performs maxpoolingBackward on GPU by exploiting cudnnPoolingBackward(...)
	 * This method computes the backpropogation errors for previous layer of maxpooling operation
	 * @param gCtx   a valid {@link GPUContext}
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
	 * @param intermediateMemoryBudget intermediate memory budget
	 * @throws DMLRuntimeException if DMLRuntimeException occurs
	 */
	public static void maxpoolingBackward(GPUContext gCtx, String instName, MatrixObject image, MatrixObject dout,
			MatrixObject outputBlock, int N, int C, int H, int W, int K, int R,
			int S, int pad_h, int pad_w, int stride_h, int stride_w, int P,
			int Q, double intermediateMemoryBudget) throws DMLRuntimeException {
		long CHW = C*H*W; long CPQ = C*P*Q;  
		long NCHW = N*CHW; long NCPQ = N*CPQ; 

		if(NCHW < maxNumDoublesOfCuDNNTensor && NCPQ < maxNumDoublesOfCuDNNTensor) {
			// Filter and output are accounted as dense in the memory estimation for conv2dBackwardData
			long overhead = isInSparseFormat(gCtx, image) ? OptimizerUtils.estimateSizeExactSparsity(N, CHW, 1.0) : 0;
			overhead += isInSparseFormat(gCtx, dout) ? OptimizerUtils.estimateSizeExactSparsity(N, CPQ, 1.0) : 0;
			Pointer dx = getDensePointerForCuDNN(gCtx, outputBlock, instName);
			if(overhead <= intermediateMemoryBudget) {
				Pointer x = getDensePointerForCuDNN(gCtx, image, instName);
				Pointer dy = getDensePointerForCuDNN(gCtx, dout, instName);
				cudnnMaxpoolingBackward(gCtx, instName, x, dy, dx, N, C, H, W, K, R, S, pad_h, pad_w, stride_h, stride_w, P, Q);
			}
			else {
				LibMatrixCuDNNInputRowFetcher imgFetcher = new LibMatrixCuDNNInputRowFetcher(gCtx, instName, image);
				LibMatrixCuDNNInputRowFetcher doutFetcher = new LibMatrixCuDNNInputRowFetcher(gCtx, instName, dout);
				for(int n = 0; n < N; n++) {
					cudnnMaxpoolingBackward(gCtx, instName, imgFetcher.getNthRow(n), doutFetcher.getNthRow(n), 
							dx.withByteOffset(n*CHW*Sizeof.DOUBLE), 
							1, C, H, W, K, R, S, pad_h, pad_w, stride_h, stride_w, P, Q);
				}
				// Deallocate temporary array to hold one element of input
				imgFetcher.close();
				doutFetcher.close();
			}
		}
		else {
			throwCuDNNDimensionError(N, CHW, N, CPQ);
		}
	}
	
	private static void cudnnMaxpoolingBackward(GPUContext gCtx, String instName, 
			Pointer x, Pointer dy, Pointer dx, 
			int N, int C, int H, int W, int K, int R,
			int S, int pad_h, int pad_w, int stride_h, int stride_w, int P,
			int Q) throws DMLRuntimeException {
		if(LOG.isTraceEnabled()) {
			LOG.trace("GPU : maxpoolingBackward" + ", GPUContext=" + gCtx);
		}
		Pointer y = null;
		cudnnPoolingDescriptor poolingDesc = null;

		try {
			long t1=0, t2=0, t3=0;
			if (GPUStatistics.DISPLAY_STATISTICS) t1 = System.nanoTime();
			// Allocate descriptors
			cudnnTensorDescriptor xDesc = allocateTensorDescriptor(N, C, H, W);
			cudnnTensorDescriptor yDesc = allocateTensorDescriptor(N, C, P, Q);
			cudnnTensorDescriptor dxDesc = allocateTensorDescriptor(N, C, H, W);
			cudnnTensorDescriptor dyDesc = allocateTensorDescriptor(N, C, P, Q);

			poolingDesc = allocatePoolingDescriptor(R, S, pad_h, pad_w, stride_h, stride_w);

			// Calling PoolForward first, y is one of the inputs for poolBackward
			// TODO: Remove calling poolForward after necessary changes at language level for poolBackward
			long numBytes = N*C*P*Q*Sizeof.DOUBLE;
			y = gCtx.allocate(numBytes);
			
			if (GPUStatistics.DISPLAY_STATISTICS) GPUStatistics.maintainCPMiscTimes(instName, GPUInstruction.MISC_TIMER_CUDNN_INIT, System.nanoTime() - t1);

			if (GPUStatistics.DISPLAY_STATISTICS) t2 = System.nanoTime();
			int status = cudnnPoolingForward(getCudnnHandle(gCtx), poolingDesc, one(), xDesc, x, zero(), yDesc, y);
			if (GPUStatistics.DISPLAY_STATISTICS) GPUStatistics.maintainCPMiscTimes(instName, GPUInstruction.MISC_TIMER_MAXPOOLING_FORWARD_LIB, System.nanoTime() - t2);

			if(status != jcuda.jcudnn.cudnnStatus.CUDNN_STATUS_SUCCESS) {
				throw new DMLRuntimeException("Could not executed cudnnPoolingForward before cudnnPoolingBackward: " + jcuda.jcudnn.cudnnStatus.stringFor(status));
			}

			if (GPUStatistics.DISPLAY_STATISTICS) t3 = System.nanoTime();
			status = cudnnPoolingBackward(getCudnnHandle(gCtx), poolingDesc, one(), yDesc, y, dyDesc, dy, xDesc, x, zero(), dxDesc, dx);
			if (GPUStatistics.DISPLAY_STATISTICS) GPUStatistics.maintainCPMiscTimes(instName, GPUInstruction.MISC_TIMER_MAXPOOLING_BACKWARD_LIB, System.nanoTime() - t3);

			if(status != jcuda.jcudnn.cudnnStatus.CUDNN_STATUS_SUCCESS) {
				throw new DMLRuntimeException("Could not executed cudnnPoolingBackward: " + jcuda.jcudnn.cudnnStatus.stringFor(status));
			}
		} catch (CudaException e) {
			throw new DMLRuntimeException("Error in conv2d in GPUContext " + gCtx.toString() + " from Thread " + Thread.currentThread().toString(), e);
		}
		finally {
			long t4=0;
			if (GPUStatistics.DISPLAY_STATISTICS) t4 = System.nanoTime();

			if(y != null)
				gCtx.cudaFreeHelper(instName, y);
			if(poolingDesc != null)
				jcuda.jcudnn.JCudnn.cudnnDestroyPoolingDescriptor(poolingDesc);

			if (GPUStatistics.DISPLAY_STATISTICS) GPUStatistics.maintainCPMiscTimes(instName, GPUInstruction.MISC_TIMER_CUDNN_CLEANUP, System.nanoTime() - t4);
		}
	}

	static cudnnConvolutionDescriptor allocateConvolutionDescriptor(int padding [], int strides []) {
		cudnnConvolutionDescriptor convDesc = new cudnnConvolutionDescriptor();
		cudnnCreateConvolutionDescriptor(convDesc);
		cudnnSetConvolution2dDescriptor(convDesc, padding[0], padding[1], strides[0], strides[1], 1, 1, CUDNN_CROSS_CORRELATION);
		return convDesc;
	}

	protected static cudnnFilterDescriptor allocateFilterDescriptor(int K, int C, int R, int S) {
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
	 * Convenience method to get tensor descriptor
	 * @param N number of images
	 * @param C number of channels
	 * @param H height
	 * @param W width
	 * @return cudnn tensor descriptor
	 * @throws DMLRuntimeException if the input descriptor and matrix dimensions don't match
	 */
	static cudnnTensorDescriptor allocateTensorDescriptor(int N, int C, int H, int W) throws DMLRuntimeException {
		cudnnTensorDescriptor tensorDescriptor = new cudnnTensorDescriptor();
		cudnnCreateTensorDescriptor(tensorDescriptor);
		cudnnSetTensor4dDescriptor(tensorDescriptor, CUDNN_TENSOR_NCHW, CUDNN_DATA_DOUBLE, N, C, H, W);
		return tensorDescriptor;
	}

	/**
	 * Convenience method to get tensor descriptor from underlying GPUObject
	 * @param gCtx   a valid {@link GPUContext}
	 * @param mat matrix object
	 * @param N number of images
	 * @param C number of channels
	 * @param H height
	 * @param W width
	 * @return cudnn tensor descriptor
	 * @throws DMLRuntimeException if the input descriptor and matrix dimensions don't match
	 */
	private static cudnnTensorDescriptor allocateTensorDescriptor(GPUContext gCtx, MatrixObject mat, int N, int C, int H, int W) throws DMLRuntimeException {
		if(mat.getNumRows() != N || mat.getNumColumns() != C*H*W) {
			throw new DMLRuntimeException("Mismatch descriptor-matrix dimensions:" + mat.getNumRows() + " != " + N
					+ " || " + mat.getNumColumns() + " != " + (C*H*W));
		}
		return mat.getGPUObject(gCtx).allocateTensorDescriptor(N, C, H, W);
	}

	/**
	 * Performs the forward BatchNormalization layer computation for inference
	 * @param gCtx   a valid {@link GPUContext}
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
	public static void batchNormalizationForwardInference(GPUContext gCtx, String instName, MatrixObject image,
			MatrixObject scale, MatrixObject bias, MatrixObject runningMean, MatrixObject runningVar,
			MatrixObject ret, double epsilon) throws DMLRuntimeException {
		if(LOG.isTraceEnabled()) {
			LOG.trace("GPU : batchNormalizationForwardInference" + ", GPUContext=" + gCtx);
		}
		int mode = cudnnBatchNormMode.CUDNN_BATCHNORM_SPATIAL;

		int N = toInt(image.getNumRows());
		int C = toInt(scale.getNumColumns());
		long CHW = image.getNumColumns();
		validateBatchNormalizationDimensions(scale, bias, runningMean, runningVar, C);

		// Allocate descriptors
		cudnnTensorDescriptor nCHWDescriptor = allocateNCHWDescriptors(gCtx, N, C, CHW,
				new MatrixObject[] {image},  new MatrixObject[] {ret});
		cudnnTensorDescriptor scaleTensorDesc = allocateTensorDescriptor(gCtx, scale, 1, C, 1, 1);

		// Get underlying dense pointer
		Pointer imagePtr = getDensePointerForCuDNN(gCtx, image, instName);
		Pointer retPtr = getDensePointerForCuDNN(gCtx, ret, instName);
		Pointer biasPtr = getDensePointerForCuDNN(gCtx, bias, instName);
		Pointer scalePtr = getDensePointerForCuDNN(gCtx, scale, instName);
		Pointer runningMeanPtr = getDensePointerForCuDNN(gCtx, runningMean, instName);
		Pointer runningVarPtr = getDensePointerForCuDNN(gCtx, runningVar, instName);

		checkStatus(cudnnBatchNormalizationForwardInference(getCudnnHandle(gCtx), mode, one(), zero(),
				nCHWDescriptor, imagePtr, nCHWDescriptor, retPtr,
				scaleTensorDesc, scalePtr, biasPtr,
				runningMeanPtr, runningVarPtr, epsilon));
	}

	/**
	 * Performs the forward BatchNormalization layer computation for training
	 * @param gCtx   a valid {@link GPUContext}
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
	public static void batchNormalizationForwardTraining(GPUContext gCtx, String instName, MatrixObject image,
			MatrixObject scale,  MatrixObject bias, MatrixObject runningMean, MatrixObject runningVar,
			MatrixObject ret, MatrixObject retRunningMean, MatrixObject retRunningVar, double epsilon, double exponentialAverageFactor) throws DMLRuntimeException {
		if(LOG.isTraceEnabled()) {
			LOG.trace("GPU : batchNormalizationForwardTraining" + ", GPUContext=" + gCtx);
		}
		int mode = cudnnBatchNormMode.CUDNN_BATCHNORM_SPATIAL;

		int N = toInt(image.getNumRows());
		int C = toInt(scale.getNumColumns());
		long CHW = image.getNumColumns();
		validateBatchNormalizationDimensions(scale, bias, runningMean, runningVar, C);

		// Allocate descriptors
		cudnnTensorDescriptor nCHWDescriptor = allocateNCHWDescriptors(gCtx, N, C, CHW,
				new MatrixObject[] {image},  new MatrixObject[] {ret});
		cudnnTensorDescriptor scaleTensorDesc = allocateTensorDescriptor(gCtx, scale, 1, C, 1, 1);

		// Get underlying dense pointer
		Pointer imagePtr = getDensePointerForCuDNN(gCtx, image, instName);
		Pointer retPtr = getDensePointerForCuDNN(gCtx, ret, instName);
		Pointer biasPtr = getDensePointerForCuDNN(gCtx, bias, instName);
		Pointer scalePtr = getDensePointerForCuDNN(gCtx, scale, instName);
		Pointer runningMeanPtr = getDensePointerForCuDNN(gCtx, runningMean, instName);
		Pointer runningVarPtr = getDensePointerForCuDNN(gCtx, runningVar, instName);

		// To allow for copy-on-write
		Pointer retRunningMeanPtr = getDensePointerForCuDNN(gCtx, retRunningMean, instName);
		Pointer retRunningVarPtr = getDensePointerForCuDNN(gCtx, retRunningVar, instName);
		cudaMemcpy(retRunningMeanPtr, runningMeanPtr, C * Sizeof.DOUBLE, cudaMemcpyDeviceToDevice);
		cudaMemcpy(retRunningVarPtr, runningVarPtr, C * Sizeof.DOUBLE, cudaMemcpyDeviceToDevice);

		// ignoring resultSaveMean and resultSaveVariance as it requires state management
		checkStatus(cudnnBatchNormalizationForwardTraining(getCudnnHandle(gCtx), mode, one(), zero(),
				nCHWDescriptor, imagePtr, nCHWDescriptor, retPtr,
				scaleTensorDesc, scalePtr, biasPtr, exponentialAverageFactor,
				retRunningMeanPtr, retRunningVarPtr, epsilon, new Pointer(), new Pointer()));
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
	 * Convenient utility for batch normalization that returns a NCHW descriptor
	 * @param gCtx a valid {@link GPUContext}
	 * @param N number of images
	 * @param C number of channels
	 * @param CHW channels*height*width
	 * @param input input matrix objects
	 * @param output output matrix objects
	 * @return one of the NCHW descriptor
	 * @throws DMLRuntimeException if error occurs
	 */
	private static cudnnTensorDescriptor allocateNCHWDescriptors(GPUContext gCtx, int N, int C, long CHW, MatrixObject [] input, MatrixObject [] output) throws DMLRuntimeException {
		cudnnTensorDescriptor ret  = null; // Return any one
		if(CHW > ((long)Integer.MAX_VALUE)*C) {
			throw new DMLRuntimeException("image size (height*width) should be less than " + Integer.MAX_VALUE);
		}
		cudnnTensorDescriptor knownNCHWdescriptor = null;
		int H = -1; int W = -1;
		for(int i = 0; i < input.length; i++) {
			knownNCHWdescriptor = input[i].getGPUObject(gCtx).getTensorDescriptor();
			if(knownNCHWdescriptor != null) {
				int [] shape = input[i].getGPUObject(gCtx).getTensorShape();
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
				ret = allocateTensorDescriptor(gCtx, input[i], N, C, H, W);
			}
			for(int i = 0; i < output.length; i++) {
				ret = allocateTensorDescriptor(gCtx, output[i], N, C, H, W);
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
	 * @param gCtx   a valid {@link GPUContext}
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
	public static void batchNormalizationBackward(GPUContext gCtx, String instName, MatrixObject image, MatrixObject dout,
			MatrixObject scale, MatrixObject ret, MatrixObject retScale, MatrixObject retBias,
			double epsilon) throws DMLRuntimeException {
		if(LOG.isTraceEnabled()) {
			LOG.trace("GPU : batchNormalizationBackward" + ", GPUContext=" + gCtx);
		}
		int mode = cudnnBatchNormMode.CUDNN_BATCHNORM_SPATIAL;

		int N = toInt(image.getNumRows());
		int C = toInt(scale.getNumColumns());
		long CHW = image.getNumColumns();

		// Allocate descriptors
		cudnnTensorDescriptor nCHWDescriptor = allocateNCHWDescriptors(gCtx, N, C, CHW,
				new MatrixObject[] {image, dout},  new MatrixObject[] {ret});
		cudnnTensorDescriptor scaleTensorDesc = allocateTensorDescriptor(gCtx, scale, 1, C, 1, 1);

		// Get underlying dense pointer
		Pointer imagePtr = getDensePointerForCuDNN(gCtx, image, instName);
		Pointer doutPtr = getDensePointerForCuDNN(gCtx, dout, instName);
		Pointer scalePtr = getDensePointerForCuDNN(gCtx, scale, instName);
		Pointer retPtr = getDensePointerForCuDNN(gCtx, ret, instName);
		Pointer retScalePtr = getDensePointerForCuDNN(gCtx, retScale, instName);
		Pointer retBiasPtr = getDensePointerForCuDNN(gCtx, retBias, instName);

		// ignoring resultSaveMean and resultSaveVariance as it requires state management
		checkStatus(cudnnBatchNormalizationBackward(getCudnnHandle(gCtx), mode,  one(), zero(), one(), zero(),
				nCHWDescriptor,  imagePtr, nCHWDescriptor, doutPtr, nCHWDescriptor, retPtr,
				scaleTensorDesc, scalePtr, retScalePtr, retBiasPtr, epsilon, new Pointer(), new Pointer()));
	}


	private static void cudnnReLU(GPUContext gCtx, String instName, MatrixObject in, Pointer dstData, cudnnTensorDescriptor srcTensorDesc) throws DMLRuntimeException {
		long t0=0;
		try {
			if(LOG.isTraceEnabled()) {
				LOG.trace("GPU : performCuDNNReLU" + ", GPUContext=" + gCtx);
			}
			cudnnTensorDescriptor dstTensorDesc = srcTensorDesc;

			Pointer srcData = getDensePointerForCuDNN(gCtx, in, instName);
			cudnnActivationDescriptor activationDescriptor = new cudnnActivationDescriptor();
			cudnnCreateActivationDescriptor(activationDescriptor);
			double dummy = -1;
			cudnnSetActivationDescriptor(activationDescriptor, CUDNN_ACTIVATION_RELU, CUDNN_PROPAGATE_NAN, dummy);
			if (GPUStatistics.DISPLAY_STATISTICS) t0 = System.nanoTime();
			cudnnActivationForward(getCudnnHandle(gCtx), activationDescriptor,
					one(), srcTensorDesc, srcData,
					zero(), dstTensorDesc, dstData);
			if (GPUStatistics.DISPLAY_STATISTICS) GPUStatistics.maintainCPMiscTimes(instName, GPUInstruction.MISC_TIMER_ACTIVATION_FORWARD_LIB, System.nanoTime() - t0);
		} catch (CudaException e) {
			throw new DMLRuntimeException("Error in conv2d in GPUContext " + gCtx.toString() + " from Thread " + Thread.currentThread().toString(), e);
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
	 * @param gCtx   a valid {@link GPUContext}
	 * @param instName the invoking instruction's name for record {@link Statistics}.
	 * @param in input matrix
	 * @param outputName	name of the output matrix
	 * @throws DMLRuntimeException	if an error occurs
	 */
	public static void relu(ExecutionContext ec, GPUContext gCtx, String instName, MatrixObject in, String outputName) throws DMLRuntimeException {
		if (ec.getGPUContext(0) != gCtx)
			throw new DMLRuntimeException("GPU : Invalid internal state, the GPUContext set with the ExecutionContext is not the same used to run this LibMatrixCUDA function");
		long N = in.getNumRows();
		long CHW = in.getNumColumns();
		MatrixObject output = ec.getMatrixObject(outputName);
		getDenseMatrixOutputForGPUInstruction(ec, instName, outputName, in.getNumRows(), in.getNumColumns()); // Allocated the dense output matrix
		long t0=0;
		cudnnTensorDescriptor srcTensorDesc = in.getGPUObject(gCtx).getTensorDescriptor();
		if(N*CHW >= maxNumDoublesOfCuDNNTensor ||  srcTensorDesc == null) {
			if(LOG.isTraceEnabled()) {
				LOG.trace("GPU : relu custom kernel" + ", GPUContext=" + gCtx);
			}
			// Invokes relu(double* A,  double* ret, int rlen, int clen)
			if (GPUStatistics.DISPLAY_STATISTICS) t0 = System.nanoTime();
			Pointer dstData = getDensePointerForCuDNN(gCtx, output, instName);
			Pointer srcData = getDensePointerForCuDNN(gCtx, in, instName); // TODO: FIXME: Add sparse kernel support for relu
			getCudaKernels(gCtx).launchKernel("relu",
					ExecutionConfig.getConfigForSimpleMatrixOperations(toInt(N), toInt(CHW)),
					srcData, dstData, toInt(N), toInt(CHW));
			if (GPUStatistics.DISPLAY_STATISTICS) GPUStatistics.maintainCPMiscTimes(instName, GPUInstruction.MISC_TIMER_RELU_KERNEL, System.nanoTime() - t0);
		}
		else {
			cudnnReLU(gCtx, instName, in, getDensePointerForCuDNN(gCtx, output, instName), srcTensorDesc);
		}
	}

	/**
	 * Convenience method to get jcudaDenseMatrixPtr. This method explicitly converts sparse to dense format, so use it judiciously.
	 * 
	 * @param gCtx a valid {@link GPUContext}
	 * @param image input matrix object
	 * @param instName name of the instruction
	 * @return jcuda pointer
	 * @throws DMLRuntimeException if error occurs while sparse to dense conversion
	 */
	protected static Pointer getDensePointerForCuDNN(GPUContext gCtx, MatrixObject image, String instName) throws DMLRuntimeException {
		long numElems = image.getNumRows()*image.getNumColumns();
		if(numElems > maxNumDoublesOfCuDNNTensor) {
			throw new DMLRuntimeException("CuDNN restriction: the size of input tensor cannot have greater than 2 giga-elements, but has " + numElems + " (i.e. [" + image.getNumRows() + " X " + image.getNumColumns() + "]). Hint: try reducing the mini-batch size.");
		}
		return getDensePointer(gCtx, image, instName);
	}

	/**
	 * Convenience method for checking the status of CuDNN kernel.
	 *
	 * @param status status returned by CuDNN
	 * @throws DMLRuntimeException if status is not CUDNN_STATUS_SUCCESS
	 */
	protected static void checkStatus(int status) throws DMLRuntimeException {
		if(status != cudnnStatus.CUDNN_STATUS_SUCCESS)
			throw new DMLRuntimeException("Error status returned by CuDNN:" + jcuda.jcudnn.cudnnStatus.stringFor(status));
	}
}
