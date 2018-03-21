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
import java.util.concurrent.Future;
import java.util.concurrent.atomic.AtomicLong;

import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;
import org.apache.sysml.api.DMLScript;
import org.apache.sysml.hops.OptimizerUtils;
import org.apache.sysml.runtime.DMLRuntimeException;
import org.apache.sysml.runtime.util.CommonThreadPool;
import org.apache.sysml.runtime.util.ConvolutionUtils;

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
 * 4. LibMatrixDNN__.getWorkers creates appropriate workers based on the runtime characteristics of
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
	public static enum PoolingType {
		MAX, AVG
	}
	
	//library configurations and external contracts
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
		if(DMLScript.FINEGRAINED_STATISTICS) {
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
		
		long nnz = execute(LibMatrixDNNConv2d.getConv2dWorkers(params), params);
		
		//post-processing: maintain nnz
		outputBlock.setNonZeros(nnz);
		outputBlock.examSparsity();
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
		
		long nnz = execute(LibMatrixDNNConv2d.getConv2dBackwardDataWorkers(params), params);
		
		//post-processing: maintain nnz
		outputBlock.setNonZeros(nnz);
		outputBlock.examSparsity();
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
		
		execute(LibMatrixDNNConv2d.getConv2dBackwardFilterWorkers(params), params);
		
		//post-processing: maintain nnz
		outputBlock.recomputeNonZeros(); 
		outputBlock.examSparsity();
	}
	
	public static void pooling(MatrixBlock input, MatrixBlock output, ConvolutionParameters params, PoolingType poolType) throws DMLRuntimeException {
		params.input1 = input;
		params.output = output;
		
		if(input.getNumColumns() != params.C*params.H*params.W || input.getNumRows() != params.N) {
			throw new DMLRuntimeException("Incorrect input dimensions in maxpooling:" + input.getNumRows() + " " 
				+ input.getNumColumns() + " " + params.N + " " + params.C*params.H*params.W);
		}
		
		//materialize indexes unless basic case with stride=1 and pad=0
		if( !params.isStride1Pad0() || input.sparse )
			fillIndexesArray(params);
		
		long nnz = execute(LibMatrixDNNPooling.getPoolingWorkers(params, poolType), params);
		
		// post-processing: maintain nnz
		output.setNonZeros(nnz);
		output.examSparsity();
	}
	

	/**
	 * This method computes the backpropogation errors for previous layer of pooling operation
	 * 
	 * @param input input matrix
	 * @param dout dout matrix
	 * @param outputBlock output matrix
	 * @param params convolution parameters
	 * @param performReluBackward perform ReLU backward
	 * @param poolType type of pooling
	 * @throws DMLRuntimeException if DMLRuntimeException occurs
	 */
	public static void poolingBackward(MatrixBlock input, MatrixBlock dout, MatrixBlock outputBlock, 
			ConvolutionParameters params, boolean performReluBackward, PoolingType poolType) throws DMLRuntimeException {
		params.input1 = input;
		params.input2 = dout;
		params.output = outputBlock;
		
		if(poolType == PoolingType.MAX && (input.getNumColumns() != params.C*params.H*params.W || input.getNumRows() != params.N)) {
			throw new DMLRuntimeException("Incorrect input dimensions in maxpooling_backward:" + input.getNumRows() + " " + input.getNumColumns() + " " + params.N + " " + params.K*params.P*params.Q);
		}

		if(dout.getNumColumns() != params.C*params.P*params.Q || dout.getNumRows() != params.N) {
			throw new DMLRuntimeException("Incorrect dout dimensions in pooling_backward:" + input.getNumRows() + " " + input.getNumColumns() + " " + params.N + " " + params.K*params.P*params.Q);
		}
		
		if(DMLScript.FINEGRAINED_STATISTICS) {
			boolean isSparse = (poolType == PoolingType.MAX) ? (input.isInSparseFormat() || dout.isInSparseFormat()) : dout.isInSparseFormat();
			if(isSparse)
				maxPoolBwdSparseCount.addAndGet(1);
			else
				maxPoolBwdDenseCount.addAndGet(1);
		}
		
		if (params.output.isInSparseFormat())
			throw new DMLRuntimeException("Sparse pooling_backward is not supported");

		if(poolType == PoolingType.AVG) {
			fillIndexesArray(params); 
		}
		else {
			if( !(params.input1.isInSparseFormat() && !params.input2.isInSparseFormat()) )
				fillIndexesArray(params); //not needed for sparse-dense	 
		}
		long nnz = execute(LibMatrixDNNPooling.getPoolingBackwardWorkers(params, performReluBackward, poolType), params);
		//post-processing: maintain nnz 
		outputBlock.setNonZeros(nnz);
		outputBlock.examSparsity();
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
		
		long nnz = execute(LibMatrixDNNRelu.getReluBackwardWorkers(params), params);
		
		// post-processing: maintain nnz
		outputBlock.setNonZeros(nnz);
		outputBlock.examSparsity();
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
		
		double [] outputArray = outputBlock.getDenseBlockValues();
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
			double [] biasArr = bias.getDenseBlockValues();
			for(int n = 0; n < N; n++) {
				for(int k = 0; k < K; k++) {
					double biasVal = biasArr[k];
					for(int pq = 0; pq < PQ; pq++, index++) {
						outputArray[index] += biasVal;
					}
				}
			}
		}
		
		//post-processing: maintain nnz
		outputBlock.recomputeNonZeros(); 
		outputBlock.examSparsity();
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
			if(bias.isInSparseFormat())
				bias.sparseToDense(); // Since bias is extremely small array
			double [] biasArr = bias.getDenseBlockValues();
			if(!input.isInSparseFormat()) {
				double [] outputArray = outputBlock.getDenseBlockValues();
				int index = 0;
				for(int n = 0; n < N; n++) {
					for(int k = 0; k < K; k++) {
						double biasVal = biasArr[k];
						for(int pq = 0; pq < PQ; pq++, index++) {
							outputArray[index] *= biasVal;
						}
					}
				}
			}
			else {
				// First delete those elements which will become zero 
				for(int k = 0; k < K; k++) {
					if(biasArr[k] == 0) {
						for(int n = 0; n < N; n++) {
							outputBlock.sparseBlock.deleteIndexRange(n, k*PQ, (k+1)*PQ);
						}
					}
				}
				// Then perform bias_multiply for non-zero bias entries
				for(int n = 0; n < N; n++) {
					if( !outputBlock.sparseBlock.isEmpty(n) ) {
						int apos = outputBlock.sparseBlock.pos(n);
						int alen = outputBlock.sparseBlock.size(n);
						int[] aix = outputBlock.sparseBlock.indexes(n);
						double[] avals = outputBlock.sparseBlock.values(n);
						
						for(int j=apos; j<apos+alen; j++) {
							// Since aix[j] => KPQ
							int k = aix[j] % PQ;
							if(biasArr[k] != 0)
								avals[j] *= biasArr[k];
						}
					}
				}
			}
			
			//post-processing: maintain nnz
			params.output.recomputeNonZeros();
			params.output.examSparsity();
		}
		else {
			params.output.setNonZeros(0);
		}
	}
	
	/**
	 * Executes the tasks in parallel using java's ExecutorService.
	 *  
	 * @param tasks deep learning related tasks
	 * @param params convolution parameters
	 * @throws DMLRuntimeException if the error occurs
	 */
	private static long execute(ArrayList<Callable<Long>> tasks, ConvolutionParameters params) throws DMLRuntimeException {
		int k = OptimizerUtils.getConstrainedNumThreads(params.numThreads);
		long lnnz = 0;
		try {
			if(k == 1) {
				// Single-threaded execution when called in parfor
				// this avoid unnecessary creation of threadpool.
				for(Callable<Long> task : tasks) {
					lnnz += task.call();
				}
			}
			else {
				ExecutorService pool = CommonThreadPool.get( Math.min(k, params.N) );
				List<Future<Long>> taskret = pool.invokeAll(tasks);
				pool.shutdown();
				for( Future<Long> task : taskret )
					lnnz += task.get();
			}
		} 
		catch (Exception e) {
			throw new DMLRuntimeException("Error while executing multi-threaded tasks", e);
		}
		
		return lnnz;
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
		
		if(DMLScript.FINEGRAINED_STATISTICS) {
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
		
		if(DMLScript.FINEGRAINED_STATISTICS) {
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
		
		if(DMLScript.FINEGRAINED_STATISTICS) {
			if(input.isInSparseFormat() || filter.isInSparseFormat()) {
				conv2dSparseCount.addAndGet(1);
			}
			else {
				conv2dDenseCount.addAndGet(1);
			}
		}
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
		for( int p=0, ix=-params.pad_h; p < params.P; p++, ix+=params.stride_h ) {
			// Note: We do not treat pad as zero
			params.start_indexes_h[p] = Math.max(ix, 0);
			params.end_indexes_h[p] = Math.min(ix+params.R, params.H);
		}
		for( int q=0, ix=-params.pad_w; q < params.Q; q++, ix+=params.stride_w) {
			// Note: We do not treat pad as zero
			params.start_indexes_w[q] = Math.max(ix, 0);
			params.end_indexes_w[q] = Math.min(ix+params.S, params.W);
		}
	}
}
