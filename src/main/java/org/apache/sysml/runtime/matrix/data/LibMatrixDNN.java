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

import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.Callable;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.concurrent.atomic.AtomicLong;

import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;
import org.apache.sysml.api.DMLScript;
import org.apache.sysml.hops.OptimizerUtils;
import org.apache.sysml.runtime.DMLRuntimeException;
import org.apache.sysml.runtime.instructions.InstructionUtils;
import org.apache.sysml.runtime.matrix.operators.BinaryOperator;
import org.apache.sysml.runtime.util.ConvolutionUtils;
import org.apache.sysml.utils.Statistics;

/*
 * This class allows users to invoke deep learning related operations 
 * (such as conv2d, conv2d_backward_data, conv2d_backward_filter, maxpooling, maxpooling_backward, bias_add)
 * using multiple threads.
 * 
 * The methods accept the input matrices as MatrixBlock and the parameters using ConvolutionParameters.
 * 
 * To run in single thread, please set ConvolutionParameters.numThreads to 1.
 * 
 * DESIGN:
 * 
 * 1. LibMatrixDNN contains the user-facing methods for deep learning related operations. 
 * 2. The deep learning tasks are executed in parallel using java's ExecutorService. The key pattern
 * followed by the above mentioned functions are as follows:
 *   execute(LibMatrixDNNHelper.get__Workers(params), params);
 * 3. LibMatrixDNN's execute() method ensures the creation and shutdown of the ExecutorService.
 * 4. LibMatrixDNNHelper.get__Workers creates appropriate workers based on the runtime characteristics of
 * the input data (for example: input activations, filter, dout, ...). For code maintenance, these workers
 * are placed in the respective LibMatrixDNN__Helper files.
 * 5. The above mentioned workers may also use additional workers such as im2col and rotate180.
 * We have created similar get__Workers methods to return the appropriate worker based on the
 * runtime characteristics.
 * 6. As opposed to earlier implementation, this design reduces branch misprediction as well 
 * as instruction cache misses. It also allows us to experiment with new operators for different
 * data characteristics without affecting the performance of other operators. 
 * 7. This class assumes that the caller (for CP ConvolutionCPInstruction) deals with the empty block cases.  
 * 
 */
public class LibMatrixDNN {
	
	protected static final Log LOG =  LogFactory.getLog(LibMatrixDNN.class.getName());
	
	//library configurations and external contracts
	public static boolean DISPLAY_STATISTICS = false; //conv2d summaries in stats output
	
	// ------------------------------------------------------------------------------------------------
	private static AtomicLong conv2dSparseCount = new AtomicLong(0);
	private static AtomicLong conv2dDenseCount = new AtomicLong(0);
	private static AtomicLong conv2dBwdFilterSparseCount = new AtomicLong(0);
	private static AtomicLong conv2dBwdFilterDenseCount = new AtomicLong(0);
	private static AtomicLong conv2dBwdDataSparseCount = new AtomicLong(0);
	private static AtomicLong conv2dBwdDataDenseCount = new AtomicLong(0);
	private static AtomicLong im2colSparseCount = new AtomicLong(0);
	private static AtomicLong im2colDenseCount = new AtomicLong(0);
	private static AtomicLong maxPoolBwdSparseCount = new AtomicLong(0);
	private static AtomicLong maxPoolBwdDenseCount = new AtomicLong(0);
	static AtomicLong loopedConvMatMultTime = new AtomicLong(0);
	static AtomicLong loopedConvIm2ColTime = new AtomicLong(0);
	static AtomicLong loopedConvBwdFilterMatMultTime = new AtomicLong(0);
	static AtomicLong loopedConvBwdFilterIm2ColTime = new AtomicLong(0);
	static AtomicLong loopedConvBwdDataMatMultTime = new AtomicLong(0);
	static AtomicLong loopedConvBwdDataCol2ImTime = new AtomicLong(0);
	
	public static void appendStatistics(StringBuilder sb) {
		if(DMLScript.STATISTICS && DISPLAY_STATISTICS) {
			sb.append("LibMatrixDNN dense count (conv/bwdF/bwdD/im2col/maxBwd):\t" 
					+ conv2dDenseCount.get() + "/"
					+ conv2dBwdFilterDenseCount.get() + "/"
					+ conv2dBwdDataDenseCount.get() + "/"
					+ im2colDenseCount.get() + "/"
					+ maxPoolBwdDenseCount.get() + ".\n");
			sb.append("LibMatrixDNN sparse count (conv/bwdF/bwdD/im2col/maxBwd):\t" 
					+ conv2dSparseCount.get() + "/"
					+ conv2dBwdFilterSparseCount.get() + "/"
					+ conv2dBwdDataSparseCount.get() + "/"
					+ im2colSparseCount.get() + "/"
					+ maxPoolBwdSparseCount.get() + ".\n");
			sb.append("LibMatrixDNN conv(im2col/matmult), bwdF (im2col/matmult), bwdD (col2im/matmult) time:\t" +
					String.format("%.3f", loopedConvIm2ColTime.get()*1e-9) + "/" +
					String.format("%.3f", loopedConvMatMultTime.get()*1e-9) + "/" + 
					String.format("%.3f", loopedConvBwdFilterIm2ColTime.get()*1e-9) + "/" +
					String.format("%.3f", loopedConvBwdFilterMatMultTime.get()*1e-9) + "/" +
					String.format("%.3f", loopedConvBwdDataCol2ImTime.get()*1e-9) + "/" +
					String.format("%.3f", loopedConvBwdDataMatMultTime.get()*1e-9) + " sec.\n");
		}
	}
	public static void resetStatistics() {
		conv2dDenseCount.set(0);
		conv2dBwdFilterDenseCount.set(0);
		conv2dBwdDataDenseCount.set(0);
		im2colDenseCount.set(0);
		maxPoolBwdDenseCount.set(0);
		
		conv2dSparseCount.set(0);
		conv2dBwdFilterSparseCount.set(0);
		conv2dBwdDataSparseCount.set(0);
		im2colSparseCount.set(0);
		maxPoolBwdSparseCount.set(0);
		
		loopedConvIm2ColTime.set(0);
		loopedConvMatMultTime.set(0);
		loopedConvBwdFilterMatMultTime.set(0);
		loopedConvBwdFilterIm2ColTime.set(0);
		loopedConvBwdDataMatMultTime.set(0);
		loopedConvBwdDataCol2ImTime.set(0);
	}
	
	// Commonly used operators
	static BinaryOperator _binaryElementWiseAddition = null;
	static BinaryOperator _binaryElementWiseMultiplication = null;
	static {
		try {
			_binaryElementWiseAddition = InstructionUtils.parseBinaryOperator("+");
			_binaryElementWiseMultiplication = InstructionUtils.parseBinaryOperator("*");
		} catch (DMLRuntimeException e) {
			throw new RuntimeException("ERROR initializing LibMatrixDNN", e);
		}
	}
	// ------------------------------------------------------------------------------------------------
	
	/**
	 * This method performs convolution (i.e. cross-correlation) operation on input
	 * 
	 * @param input input batch 
	 * @param filter filter
	 * @param outputBlock output of convolution
	 * @param params convolution parameters
	 * @throws DMLRuntimeException if DMLRuntimeException occurs
	 */
	public static void conv2d(MatrixBlock input, MatrixBlock filter, MatrixBlock outputBlock, ConvolutionParameters params) throws DMLRuntimeException {
		LibMatrixDNN.checkInputsConv2d(input, filter, outputBlock, params);
		
		if(params.bias != null && params.bias.isInSparseFormat())
			params.bias.sparseToDense(); // Since bias is extremely small array
		
		if(isEligibleForConv2dSparse(params))
			Statistics.numNativeSparseConv2dCalls.increment();
		
		execute(LibMatrixDNNHelper.getConv2dWorkers(params), params);
		
		//post-processing: maintain nnz
		outputBlock.recomputeNonZeros();
	}
	
	/**
	 * This method computes the backpropogation errors for previous layer of convolution operation
	 * 
	 * @param filter filter used in conv2d 
	 * @param dout errors from next layer
	 * @param outputBlock  output errors
	 * @param params convolution parameters
	 * @throws DMLRuntimeException if DMLRuntimeException occurs
	 */
	public static void conv2dBackwardData(MatrixBlock filter, MatrixBlock dout, MatrixBlock outputBlock, ConvolutionParameters params) throws DMLRuntimeException {
		checkInputsConv2dBackwardData(filter, dout, outputBlock, params);
		
		if(isEligibleForConv2dBackwardDataDense(params))
			Statistics.numNativeSparseConv2dBwdDataCalls.increment();
		
		execute(LibMatrixDNNHelper.getConv2dBackwardDataWorkers(params), params);
		
		//post-processing: maintain nnz
		outputBlock.recomputeNonZeros();
	}
	
	/**
	 * This method computes the backpropogation errors for filter of convolution operation
	 * 
	 * @param input input image 
	 * @param dout errors from next layer
	 * @param outputBlock  output errors
	 * @param params convolution parameters
	 * @throws DMLRuntimeException if DMLRuntimeException occurs
	 */
	public static void conv2dBackwardFilter(MatrixBlock input, MatrixBlock dout, MatrixBlock outputBlock, ConvolutionParameters params) throws DMLRuntimeException {
		checkInputsConv2dBackwardFilter(input, dout, outputBlock, params);
		
		if(isEligibleForConv2dBackwardFilterSparseDense(params))
			Statistics.numNativeSparseConv2dBwdFilterCalls.increment();
		
		execute(LibMatrixDNNHelper.getConv2dBackwardFilterWorkers(params), params);
		
		//post-processing: maintain nnz
		outputBlock.recomputeNonZeros();
	}
	
	
	private static void checkOrThrowException(String msg, long lhs, long rhs) throws DMLRuntimeException {
		if(lhs != rhs)
			throw new DMLRuntimeException(msg + ":" + lhs + " != " + rhs);
	}
	private static void checkOrThrowException(String msg, long lhs, long rhs1, long rhs2, long rhs3) throws DMLRuntimeException {
		if(lhs != (rhs1*rhs2*rhs3))
			throw new DMLRuntimeException(msg + ":" + lhs + " != (" + rhs1 + " * " + rhs2 + " * " + rhs3);
	}
	
	static void checkInputsConv2dBackwardData(MatrixBlock filter, MatrixBlock dout, MatrixBlock outputBlock, ConvolutionParameters params)  throws DMLRuntimeException {
		params.input1 = filter;
		params.input2 = dout;
		params.output = outputBlock;
		checkOrThrowException("Incorrect input to conv2d_backward_data: Number of rows of input filter != "
				+ "number of filters in filter_shape", filter.getNumRows(), params.K);
		checkOrThrowException("Incorrect input to conv2d_backward_data: Number of columns of input filter != "
				+ "channels*filter_height*filter_height in filter_shape", filter.getNumColumns(), params.C, params.R, params.S);
		checkOrThrowException("Incorrect input to conv2d_backward_data: Number of rows of input errors != "
				+ "batch size in input_shape", dout.getNumRows(), params.N);
		checkOrThrowException("Incorrect input to conv2d_backward_data: Number of columns of input errors != "
				+ "expected input error channels*height*width", dout.getNumColumns(), params.K, params.P, params.Q);
		if(params.stride_h <= 0 || params.stride_w <= 0) 
			throw new DMLRuntimeException("Only positive strides supported:" + params.stride_h + ", " + params.stride_w);
		
		if(DMLScript.STATISTICS && DISPLAY_STATISTICS) {
			if(filter.isInSparseFormat() || dout.isInSparseFormat()) {
				conv2dBwdDataSparseCount.addAndGet(1);
			}
			else {
				conv2dBwdDataDenseCount.addAndGet(1);
			}
		}
	}
	
	static void checkInputsConv2dBackwardFilter(MatrixBlock input, MatrixBlock dout, MatrixBlock outputBlock, ConvolutionParameters params)  throws DMLRuntimeException {
		params.input1 = input;
		params.input2 = dout;
		params.output = outputBlock;
		checkOrThrowException("Incorrect input to conv2d_backward_filter: Number of rows of input data != "
				+ "batch size in input_shape", input.getNumRows(), params.N);
		checkOrThrowException("Incorrect input to conv2d_backward_filter: Number of columns of input data != "
				+ "channels*input_height*input_height in input_shape", input.getNumColumns(), params.C, params.H, params.W);
		checkOrThrowException("Incorrect input to conv2d_backward_filter: Number of rows of input errors != "
				+ "batch size in input_shape", dout.getNumRows(), params.N);
		checkOrThrowException("Incorrect input to conv2d_backward_filter: Number of columns of input errors != "
				+ "expected input error channels*height*width", dout.getNumColumns(), params.K, params.P, params.Q);
		if(params.stride_h <= 0 || params.stride_w <= 0) 
			throw new DMLRuntimeException("Only positive strides supported:" + params.stride_h + ", " + params.stride_w);
		
		if(DMLScript.STATISTICS && DISPLAY_STATISTICS) {
			if(input.isInSparseFormat() || dout.isInSparseFormat()) {
				conv2dBwdFilterSparseCount.addAndGet(1);
			}
			else {
				conv2dBwdFilterDenseCount.addAndGet(1);
			}
		}
	}
	
	static void checkInputsConv2d(MatrixBlock input, MatrixBlock filter, MatrixBlock outputBlock, ConvolutionParameters params) throws DMLRuntimeException {
		params.input1 = input;
		params.input2 = filter;
		params.output = outputBlock;
		
		checkOrThrowException("Incorrect input to conv2d: Number of rows of input filter != "
				+ "number of filters in filter_shape", filter.getNumRows(), params.K);
		checkOrThrowException("Incorrect input to conv2d: Number of columns of input filter != "
				+ "channels*filter_height*filter_height in filter_shape", filter.getNumColumns(), params.C, params.R, params.S);
		checkOrThrowException("Incorrect input to conv2d: Number of rows of input data != "
				+ "batch size in input_shape", input.getNumRows(), params.N);
		checkOrThrowException("Incorrect input to conv2d: Number of columns of input data != "
				+ "channels*input_height*input_height in input_shape", input.getNumColumns(), params.C, params.H, params.W);
		if(params.stride_h <= 0 || params.stride_w <= 0) 
			throw new DMLRuntimeException("Only positive strides supported:" + params.stride_h + ", " + params.stride_w);
		
		if(DMLScript.STATISTICS && DISPLAY_STATISTICS) {
			if(input.isInSparseFormat() || filter.isInSparseFormat()) {
				conv2dSparseCount.addAndGet(1);
			}
			else {
				conv2dDenseCount.addAndGet(1);
			}
		}
	}
	
	/**
	 * This method computes the backpropogation errors for previous layer of maxpooling operation
	 * 
	 * @param input input matrix
	 * @param dout dout matrix
	 * @param outputBlock output matrix
	 * @param params convolution parameters
	 * @param performReluBackward perform ReLU backward
	 * @throws DMLRuntimeException if DMLRuntimeException occurs
	 */
	public static void maxpoolingBackward(MatrixBlock input, MatrixBlock dout, MatrixBlock outputBlock, 
			ConvolutionParameters params, boolean performReluBackward) throws DMLRuntimeException {
		params.input1 = input;
		params.input2 = dout;
		params.output = outputBlock;
		if(input.getNumColumns() != params.C*params.H*params.W || input.getNumRows() != params.N) {
			throw new DMLRuntimeException("Incorrect input dimensions in maxpooling_backward:" + input.getNumRows() + " " + input.getNumColumns() + " " + params.N + " " + params.K*params.P*params.Q);
		}

		if(dout.getNumColumns() != params.C*params.P*params.Q || dout.getNumRows() != params.N) {
			throw new DMLRuntimeException("Incorrect dout dimensions in maxpooling_backward:" + input.getNumRows() + " " + input.getNumColumns() + " " + params.N + " " + params.K*params.P*params.Q);
		}
		
		if(DMLScript.STATISTICS && DISPLAY_STATISTICS) {
			if(input.isInSparseFormat() || dout.isInSparseFormat()) {
				maxPoolBwdSparseCount.addAndGet(1);
			}
			else {
				maxPoolBwdDenseCount.addAndGet(1);
			}
		}
		
		if (params.output.isInSparseFormat())
			throw new DMLRuntimeException("Sparse maxpooling_backward is not supported");

		fillIndexesArray(params);
		
		execute(LibMatrixDNNHelper.getMaxPoolingBackwardWorkers(params, performReluBackward), params);
		
		//post-processing: maintain nnz 
		outputBlock.recomputeNonZeros();
	}
	
	/**
	 * This method computes start and end indexes required for max_pool and max_pool_backward operations.
	 * This speeds up the performance of max_pool and  max_pool_backward
	 * 
	 * @param params parameters required for max_pool and max_pool_backward operations
	 */
	private static void fillIndexesArray(ConvolutionParameters params) {
		params.start_indexes_h = new int[params.P];
		params.end_indexes_h = new int[params.P];
		params.start_indexes_w = new int[params.Q];
		params.end_indexes_w = new int[params.Q];
		for (int p = 0; p < params.P; p++) {
			int start_index_h = p * params.stride_h - params.pad_h;
			int end_index_h = start_index_h + params.R;
			// Note: We do not treat pad as zero
			params.start_indexes_h[p] = Math.max(start_index_h, 0);
			params.end_indexes_h[p] = Math.min(end_index_h, params.H);
		}
		for (int q = 0; q < params.Q; q++) {
			int start_index_w =  q * params.stride_w - params.pad_w;
			int end_index_w = start_index_w + params.S;
			// Note: We do not treat pad as zero
			params.start_indexes_w[q] = Math.max(start_index_w, 0);
			params.end_indexes_w[q] = Math.min(end_index_w, params.W);
		}
	}
	
	/**
	 * This method computes the backpropagation errors for previous layer of relu operation
	 * 
	 * @param input input matrix
	 * @param dout errors from next layer
	 * @param outputBlock output matrix
	 * @param numThreads number of threads
	 * @throws DMLRuntimeException if DMLRuntimeException occurs
	 */
	public static void reluBackward(MatrixBlock input, MatrixBlock dout, MatrixBlock outputBlock, int numThreads) throws DMLRuntimeException {
		int N = input.getNumRows();
		ConvolutionParameters params = new ConvolutionParameters(N, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, numThreads);
		params.input1 = input;
		params.input2 = dout;
		params.output = outputBlock;
		if(input.getNumRows() != dout.getNumRows() || input.getNumColumns() != dout.getNumColumns()) {
			throw new DMLRuntimeException("Incorrect dimensions for relu_backward:" + 
				input.getNumRows() + " != " + dout.getNumRows() + " || " + input.getNumColumns() + " != " + dout.getNumColumns());
		}
		
		execute(LibMatrixDNNHelper.getReluBackwardWorkers(params), params);
		
		// post-processing: maintain nnz
		outputBlock.recomputeNonZeros();
	}
	
	/**
	 * Performs the operation corresponding to the DML script:
	 * ones = matrix(1, rows=1, cols=Hout*Wout)		
	 * output = input + matrix(bias %*% ones, rows=1, cols=F*Hout*Wout)
	 * This operation is often followed by conv2d and hence we have introduced bias_add(input, bias) built-in function
	 * 
	 * @param input input matrix
	 * @param bias bias matrix
	 * @param outputBlock output matrix
	 * @param numThreads number of threads
	 * @throws DMLRuntimeException if DMLRuntimeException occurs
	 */
	public static void biasAdd(MatrixBlock input, MatrixBlock bias, MatrixBlock outputBlock, int numThreads) throws DMLRuntimeException {
		int N = input.getNumRows();
		int K = bias.getNumRows();
		int PQ = input.getNumColumns() / K;
		
		if(bias.getNumColumns() != 1 || input.getNumColumns() % K != 0) {
			throw new DMLRuntimeException("Incorrect inputs for bias_add: input[" + N + " X " + input.getNumColumns()  + "] and bias[" + K + " X " + bias.getNumColumns() + "]");
		}
		
		double [] outputArray = outputBlock.getDenseBlock();
		if(input.isEmptyBlock()) {
			for(int n = 0;  n < N; n++) 
				ConvolutionUtils.fillBias(bias, outputArray, n, n+1, N, K, PQ);
		}
		else {
			// Handles both dense and sparse inputs and copies it to dense output
			outputBlock.copy(input); 
			int index = 0;
			if(bias.isInSparseFormat())
				bias.sparseToDense(); // Since bias is extremely small array
			double [] biasArr = bias.getDenseBlock();
			for(int n = 0; n < N; n++) {
				for(int k = 0; k < K; k++) {
					for(int pq = 0; pq < PQ; pq++, index++) {
						outputArray[index] += biasArr[k];
					}
				}
			}
		}
		
		//post-processing: maintain nnz
		outputBlock.recomputeNonZeros();
	}
	
	
	/**
	 * Performs the operation corresponding to the DML script:
	 * ones = matrix(1, rows=1, cols=Hout*Wout)		
	 * output = input * matrix(bias %*% ones, rows=1, cols=F*Hout*Wout)
	 * This operation is often followed by conv2d and hence we have introduced bias_multiply(input, bias) built-in function
	 * 
	 * @param input input matrix
	 * @param bias bias matrix
	 * @param outputBlock output matrix
	 * @param numThreads number of threads
	 * @throws DMLRuntimeException if DMLRuntimeException occurs
	 */
	public static void biasMultiply(MatrixBlock input, MatrixBlock bias, MatrixBlock outputBlock, int numThreads) throws DMLRuntimeException {
		int N = input.getNumRows();
		int K = bias.getNumRows();
		int PQ = input.getNumColumns() / K;
		
		ConvolutionParameters params = new ConvolutionParameters(N, PQ, -1, -1, K, -1, -1, -1, -1, -1, -1, numThreads);
		params.input1 = input;
		params.input2 = bias;
		params.output = outputBlock;
		
		if(bias.getNumColumns() != 1 || input.getNumColumns() % K != 0) {
			throw new DMLRuntimeException("Incorrect inputs for bias_multiply: input[" + N + " X " + input.getNumColumns()  + "] and bias[" + K + " X " + bias.getNumColumns() + "]");
		}
		
		if(!input.isEmptyBlock() && !bias.isEmptyBlock()) {
			// Handles both dense and sparse inputs and copies it to dense output
			outputBlock.copy(input); 
			double [] outputArray = outputBlock.getDenseBlock();
			int index = 0;
			if(bias.isInSparseFormat())
				bias.sparseToDense(); // Since bias is extremely small array
			double [] biasArr = bias.getDenseBlock();
			for(int n = 0; n < N; n++) {
				for(int k = 0; k < K; k++) {
					for(int pq = 0; pq < PQ; pq++, index++) {
						outputArray[index] *= biasArr[k];
					}
				}
			}
			
			//post-processing: maintain nnz
			params.output.recomputeNonZeros();
		}
		else {
			params.output.setNonZeros(0);
		}
	}
	
	public static void maxpooling(MatrixBlock input, MatrixBlock outputBlock, ConvolutionParameters params) throws DMLRuntimeException {
		params.input1 = input;
		params.output = outputBlock;
		
		if(input.getNumColumns() != params.C*params.H*params.W || input.getNumRows() != params.N) {
			throw new DMLRuntimeException("Incorrect input dimensions in maxpooling:" + input.getNumRows() + " " + input.getNumColumns() + " " + params.N + " " + params.C*params.H*params.W);
		}
		
		fillIndexesArray(params);
		
		execute(LibMatrixDNNHelper.getMaxPoolingWorkers(params), params);
		
		// post-processing: maintain nnz
		outputBlock.recomputeNonZeros();
	}
	
	/**
	 * Executes the tasks in parallel using java's ExecutorService.
	 *  
	 * @param tasks deep learning related tasks
	 * @param params convolution parameters
	 * @throws DMLRuntimeException if the error occurs
	 */
	private static void execute(ArrayList<Callable<Long>> tasks, ConvolutionParameters params) throws DMLRuntimeException {
		int k = OptimizerUtils.getConstrainedNumThreads(params.numThreads);
		try {
			if(k == 1) {
				// Single-threaded execution when called in parfor
				// this avoid unnecessary creation of threadpool.
				for(Callable<Long> task : tasks) {
					task.call();
				}
			}
			else {
				ExecutorService pool = Executors.newFixedThreadPool( Math.min(k, params.N) );
				List<Future<Long>> taskret = pool.invokeAll(tasks);
				pool.shutdown();
				for( Future<Long> task : taskret )
					task.get();
			}
		} 
		catch (Exception e) {
			throw new DMLRuntimeException("Error while executing multi-threaded tasks", e);
		}
	}
	
	static boolean isEligibleForConv2dBackwardFilterSparseDense(ConvolutionParameters params) {
		// NativeHelper.conv2dBackwardFilterSparseDense only if input is sparse. 
		// dout converted to dense if sparse.
		return params.enableNative && params.input1.isInSparseFormat();
	}
	static boolean isEligibleForConv2dSparse(ConvolutionParameters params) {
		// NativeHelper.conv2dSparse only if filter is dense and input is sparse
		return params.enableNative && params.input1.isInSparseFormat() && !params.input2.isInSparseFormat();
	}
	static boolean isEligibleForConv2dBackwardDataDense(ConvolutionParameters params) {
		// NativeHelper.conv2dBackwardDataDense only if filter is dense. 
		// dout converted to dense if sparse.
		return params.enableNative && !params.input1.isInSparseFormat();
	}
}
