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
import java.util.Arrays;
import java.util.List;
import java.util.concurrent.Callable;
import java.util.concurrent.ConcurrentLinkedQueue;
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

/**
 * This class allows users to invoke deep learning related operations 
 * (such as conv2d, conv2d_backward_data, conv2d_backward_filter, maxpooling, maxpooling_backward, bias_add)
 * using multiple threads.
 * 
 * The methods accept the input matrices as MatrixBlock and the parameters using ConvolutionParameters.
 * 
 * To run in single thread, please set ConvolutionParameters.numThreads to 1.
 */
public class LibMatrixDNN {
	
	protected static final Log LOG =  LogFactory.getLog(LibMatrixDNN.class.getName());
	
	//library configurations and external contracts
	public static final boolean SUPPORTS_SPARSE_OUTPUTS = false; //operations able to handle sparse outputs 
	private static final boolean ALLOW_MULTI_THREADED_OPS = true; //enable multi-threading in cp
	private static final int NUM_TASK_FACTOR = 2; //number of tasks is vcores scaled by this factor
	public static boolean DISPLAY_STATISTICS = false; //conv2d summaries in stats output

	private enum TaskType {
		MaxPooling_Forward, MaxPooling_Backward, 
		// Alternate approaches that we tried but the performance was unsatisfactory be included: direct, non-looped im2col
		LoopedIm2ColConv2d, LoopedIm2ColConv2dBwdFilter, LoopedIm2ColConv2dBwdData,
		BiasAdd, ReluBackward, BiasMultiply
	}
	
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
	private static AtomicLong loopedConvMatMultTime = new AtomicLong(0);
	private static AtomicLong loopedConvIm2ColTime = new AtomicLong(0);
	private static AtomicLong loopedConvBwdFilterMatMultTime = new AtomicLong(0);
	private static AtomicLong loopedConvBwdFilterIm2ColTime = new AtomicLong(0);
	private static AtomicLong loopedConvBwdDataMatMultTime = new AtomicLong(0);
	private static AtomicLong loopedConvBwdDataCol2ImTime = new AtomicLong(0);
	
	public static void appendStatistics(StringBuilder sb) {
		if(DMLScript.STATISTICS && DISPLAY_STATISTICS && (conv2dDenseCount.get() != 0 || conv2dSparseCount.get() != 0)) {
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
			if(loopedConvMatMultTime.get() != 0 || loopedConvIm2ColTime.get() != 0) {
				sb.append("LibMatrixDNN conv(im2col/matmult), bwdF (im2col/matmult), bwdD (col2im/matmult) time:\t" +
						String.format("%.3f", loopedConvIm2ColTime.get()*1e-9) + "/" +
						String.format("%.3f", loopedConvMatMultTime.get()*1e-9) + "/" + 
						String.format("%.3f", loopedConvBwdFilterIm2ColTime.get()*1e-9) + "/" +
						String.format("%.3f", loopedConvBwdFilterMatMultTime.get()*1e-9) + "/" +
						String.format("%.3f", loopedConvBwdDataCol2ImTime.get()*1e-9) + "/" +
						String.format("%.3f", loopedConvBwdDataMatMultTime.get()*1e-9) + " sec.\n");
			}
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
	private static BinaryOperator _binaryElementWiseAddition = null;
	private static BinaryOperator _binaryElementWiseMultiplication = null;
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
	 * This method computes the backpropogation errors for previous layer of convolution operation
	 * 
	 * @param filter filter used in conv2d 
	 * @param dout errors from next layer
	 * @param outputBlock  output errors
	 * @param params convolution parameters
	 * @throws DMLRuntimeException if DMLRuntimeException occurs
	 */
	public static void conv2dBackwardData(MatrixBlock filter, MatrixBlock dout, MatrixBlock outputBlock, ConvolutionParameters params) throws DMLRuntimeException {
		params.input1 = filter;
		params.input2 = dout;
		params.output = outputBlock;
		if(filter.getNumRows() != params.K || filter.getNumColumns() != params.C*params.R*params.S || 
				dout.getNumRows() != params.N || dout.getNumColumns() != params.K*params.P*params.Q) {
			throw new DMLRuntimeException("Incorrect input to conv2d_backward_filter");
		}
		if(params.stride_h <= 0 || params.stride_w <= 0) {
			throw new DMLRuntimeException("Only positive strides supported");
		}
		
		if(DMLScript.STATISTICS && DISPLAY_STATISTICS) {
			if(filter.isInSparseFormat() || dout.isInSparseFormat()) {
				conv2dBwdDataSparseCount.addAndGet(1);
			}
			else {
				conv2dBwdDataDenseCount.addAndGet(1);
			}
		}
		
		runConvTask(TaskType.LoopedIm2ColConv2dBwdData, params);
		
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
		params.input1 = input;
		params.input2 = dout;
		params.output = outputBlock;
		if(input.getNumRows() != params.N || input.getNumColumns() != params.C*params.H*params.W || 
				dout.getNumRows() != params.N || dout.getNumColumns() != params.K*params.P*params.Q) {
			throw new DMLRuntimeException("Incorrect input to conv2d_backward_filter");
		}
		if(params.stride_h <= 0 || params.stride_w <= 0) {
			throw new DMLRuntimeException("Only positive strides supported");
		}
		
		if(DMLScript.STATISTICS && DISPLAY_STATISTICS) {
			if(input.isInSparseFormat() || dout.isInSparseFormat()) {
				conv2dBwdFilterSparseCount.addAndGet(1);
			}
			else {
				conv2dBwdFilterDenseCount.addAndGet(1);
			}
		}
		
		runConvTask(TaskType.LoopedIm2ColConv2dBwdFilter, params);
		
		//post-processing: maintain nnz
		outputBlock.recomputeNonZeros();
	}
	
	/**
	 * Performs the operation for(e : elem) ret += t(e) in a cache-conscious manner
	 * by sequentially aggregating for(e : elem) tmp += e and finally transposing
	 * ret = t(tmp).
	 * 
	 * @param ret left and output matrix
	 * @param elem array of right untransposed matrices (expected in dense format)
	 * @param params convolution parameters
	 * @throws DMLRuntimeException in case of unsupported inputs or output
	 */
	private static void elementWiseInPlaceTransposedAddition(MatrixBlock ret, MatrixBlock[] elem) 
		throws DMLRuntimeException 
	{
		//sanity checks non-empty and dense inputs / dense output
		if( elem == null || elem.length==0 )
			throw new DMLRuntimeException("Empty input not supported.");
		for( MatrixBlock e : elem )
			if( e.isInSparseFormat() )
				throw new DMLRuntimeException("Sparse input format not supported.");
		if( ret.isInSparseFormat() )
			throw new DMLRuntimeException("Sparse output format not supported.");
				
		//Step 1: aggregate partial blocks without transpose
		MatrixBlock tmpAgg = elem[0]; 
		double[] tmp = tmpAgg.denseBlock;
		for( int k=1; k<elem.length; k++ ) {
			double[] tmp2 = elem[k].denseBlock;
			for( int i=0; i<tmp.length; i++ )
				tmp[i] += tmp2[i];
		}
		
		//Step 2: cache-conscious transpose to output
		tmpAgg.setNonZeros(-1); //avoid early abort
		LibMatrixReorg.transpose(tmpAgg, ret);
	}
	
	private static void doLoopedIm2ColConv2dBwdData(int n, MatrixBlock dout_reshaped, ConvolutionParameters params) throws DMLRuntimeException {
		MatrixBlock filter = params.input1;
		MatrixBlock dout = params.input2;
		doRotate180(n, 0, dout, dout_reshaped.denseBlock, params, true);
		dout_reshaped.recomputeNonZeros();
		
		MatrixBlock temp = new MatrixBlock(params.P*params.Q, params.C*params.R*params.S, false);
		long t1 = DMLScript.STATISTICS && DISPLAY_STATISTICS ? System.nanoTime() : 0;
		LibMatrixMult.matrixMult(dout_reshaped, filter, temp, false);
		long t2 = DMLScript.STATISTICS && DISPLAY_STATISTICS ? System.nanoTime() : 0 ;
		doCol2imOverSingleImage(n, temp, params);
		long t3 = DMLScript.STATISTICS && DISPLAY_STATISTICS ? System.nanoTime() : 0 ;
		if(DMLScript.STATISTICS && DISPLAY_STATISTICS) {
			loopedConvBwdDataMatMultTime.addAndGet(t2-t1);
			loopedConvBwdDataCol2ImTime.addAndGet(t3-t2);
		}
	}
	
	private static MatrixBlock doLoopedIm2ColConv2dBwdFilter(int n, 
			MatrixBlock im2ColOutBlock, MatrixBlock dout_reshaped, MatrixBlock partialRetBlock, ConvolutionParameters params, double []  tempIm2ColArr) throws DMLRuntimeException {
		long t1 = DMLScript.STATISTICS && DISPLAY_STATISTICS ? System.nanoTime() : 0;
		doIm2col(n, im2ColOutBlock, params, tempIm2ColArr);
		im2ColOutBlock.recomputeNonZeros();
		long t2 = DMLScript.STATISTICS && DISPLAY_STATISTICS ? System.nanoTime() : 0 ;
		
		doRotate180(n, 0, params.input2, dout_reshaped.denseBlock, params, true);
		dout_reshaped.recomputeNonZeros();
		
		MatrixBlock temp = new MatrixBlock(params.C*params.R*params.S, params.K, false);
		long t3 = DMLScript.STATISTICS && DISPLAY_STATISTICS ? System.nanoTime() : 0 ;
		LibMatrixMult.matrixMult(im2ColOutBlock, dout_reshaped, temp, false);
		long t4 = DMLScript.STATISTICS && DISPLAY_STATISTICS ? System.nanoTime() : 0 ;
		if(DMLScript.STATISTICS && DISPLAY_STATISTICS) {
			loopedConvBwdFilterMatMultTime.addAndGet(t4-t3);
			loopedConvBwdFilterIm2ColTime.addAndGet(t2-t1);
		}
		if(!temp.isEmptyBlock()) {
			// partialRetBlock is size: [params.C*params.R*params.S, params.K]
			ConvolutionUtils.binaryOperationInPlace(temp, partialRetBlock.getDenseBlock(), 0, params.K, 0, params.C*params.R*params.S, 
					_binaryElementWiseAddition);
		}
		return partialRetBlock;
	}
	
	private static void computeTensorIndexes(int j, int [] ret, int H, int W) throws DMLRuntimeException {
		ret[0] = j / (H*W);
		ret[1] = (j - ret[0]*(H*W))/W;
		ret[2] = j % W;
	}
	
	public static void conv2d(MatrixBlock input, MatrixBlock filter, MatrixBlock outputBlock, ConvolutionParameters params) throws DMLRuntimeException {
		params.input1 = input;
		params.input2 = filter;
		params.output = outputBlock;
		
		if(input.getNumRows() != params.N || input.getNumColumns() != params.C*params.H*params.W || 
				filter.getNumRows() != params.K || filter.getNumColumns() != params.C*params.R*params.S) {
			throw new DMLRuntimeException("Incorrect input to conv2d: " + input.getNumRows());
		}
		
		if(DMLScript.STATISTICS && DISPLAY_STATISTICS) {
			if(input.isInSparseFormat() || filter.isInSparseFormat()) {
				conv2dSparseCount.addAndGet(1);
			}
			else {
				conv2dDenseCount.addAndGet(1);
			}
		}
		
		runConvTask(TaskType.LoopedIm2ColConv2d, params);
		
		//post-processing: maintain nnz
		outputBlock.recomputeNonZeros();
	}
	
	private static void doLoopedIm2ColConv2d(int n, MatrixBlock im2ColOutBlock, ConvolutionParameters params, double []  temp) throws DMLRuntimeException {
		long t1 = DMLScript.STATISTICS && DISPLAY_STATISTICS ? System.nanoTime() : 0;
		doIm2col(n, im2ColOutBlock, params, temp);
		im2ColOutBlock.recomputeNonZeros();
		long t2 = DMLScript.STATISTICS && DISPLAY_STATISTICS ? System.nanoTime() : 0;
		
		MatrixBlock matMultOutBlock = new MatrixBlock(params.K, params.P*params.Q, false);
		LibMatrixMult.matrixMult(params.input2, im2ColOutBlock, matMultOutBlock, false);
		long t3 = DMLScript.STATISTICS && DISPLAY_STATISTICS ? System.nanoTime() : 0;
		
		if(DMLScript.STATISTICS && DISPLAY_STATISTICS) {
			loopedConvIm2ColTime.addAndGet(t2 - t1);
			loopedConvMatMultTime.addAndGet(t3 - t2);
		}
		
		// -----------------------------------------------------------------------------
		// Copying is required as LibMatrixMult.matrixMult (and/or Java) is not pointer aware.
		// This is not required in Native implementation
		int destPos = n*params.K*params.P*params.Q;
		int length = params.K*params.P*params.Q;
		if(!matMultOutBlock.isEmptyBlock()) {
			if(matMultOutBlock.isInSparseFormat()) {
				// Copy the sparse matrix matMultOutBlock of shape [K X PQ] to 
				// params.output.denseBlock + destPos
				final int outOffset = n*params.K*params.P*params.Q;
				final int PQ = params.P*params.Q;
				for(int k = 0; k < matMultOutBlock.getNumRows(); k++) {
					if( !matMultOutBlock.sparseBlock.isEmpty(k) ) {
						int apos = matMultOutBlock.sparseBlock.pos(k);
						int alen = matMultOutBlock.sparseBlock.size(k);
						int[] aix = matMultOutBlock.sparseBlock.indexes(k);
						double[] avals = matMultOutBlock.sparseBlock.values(k);
						for(int j = apos; j < apos+alen; j++) {
							int pqIndex = aix[j];
							params.output.denseBlock[outOffset + k*PQ + pqIndex ] = avals[j];
						}
					}
				}
			}
			else
				System.arraycopy(matMultOutBlock.denseBlock, 0, params.output.denseBlock, destPos, length);
		}
		// -----------------------------------------------------------------------------
		
		// Recomputing nnz is not required for each individual im2col as it is invoked by outer public methods (i.e. conv2d.
		//post-processing: maintain nnz
		// params.output.recomputeNonZeros(); 
	}
	
	
	/**
	 * This method computes the backpropogation errors for previous layer of maxpooling operation
	 * 
	 * @param input input matrix
	 * @param dout dout matrix
	 * @param outputBlock output matrix
	 * @param params convolution parameters
	 * @throws DMLRuntimeException if DMLRuntimeException occurs
	 */
	public static void maxpoolingBackward(MatrixBlock input, MatrixBlock dout, MatrixBlock outputBlock, ConvolutionParameters params) throws DMLRuntimeException {
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
		runConvTask(TaskType.MaxPooling_Backward, params);
		
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
	
	private static void doPoolingBackward(int n, ConvolutionParameters params) throws DMLRuntimeException {
		double [] inputArray = null;
		if (!params.input1.isInSparseFormat())
			inputArray = params.input1.getDenseBlock();
		double [] doutArray = null;
		if (!params.input2.isInSparseFormat())
			doutArray = params.input2.getDenseBlock();
		double [] outputArray = null;
		if (!params.output.isInSparseFormat())
			outputArray = params.output.getDenseBlock();
		else
			throw new DMLRuntimeException("Only dense output supported for pooling_backward");
			
		if(inputArray != null) {
			if(doutArray != null)
				doPoolingBackwardDenseDense(n, inputArray, doutArray, outputArray, params);
			else
				doPoolingBackwardDenseSparse(n, inputArray, params.input2, outputArray, params);
		}
		else {
			if(doutArray != null)
				doPoolingBackwardSparseDense(n, doutArray, outputArray, params);
			else
				doPoolingBackwardSparseSparse(n, outputArray, params);
		}
	}
	
	private static void doPoolingBackwardSparseDense(int n, double [] doutArray,  double [] outputArray, ConvolutionParameters params) throws DMLRuntimeException {
		if (!params.input1.isInSparseFormat())
			throw new DMLRuntimeException("Incorrect usage: Call optimized versions");
		
		for (int c = 0; c < params.C; c++) {
			for (int p = 0; p < params.P; p++) {
				for (int q = 0; q < params.Q; q++) {
					double inVal = doutArray[n*params.C*params.P*params.Q + c*params.P*params.Q +  p * params.Q + q];
					if(inVal != 0) {
						final int inputOffset = n*params.C*params.H*params.W + c*params.H*params.W;
						int maxIndex = getMaxIndexSparse(p, q, inputOffset, n, c, params.input1, params);
						if(maxIndex != -1)
							outputArray[maxIndex] += inVal;
					}
				}
			}
		}
	}
	
	private static void doPoolingBackwardSparseSparse(int n, double [] outputArray, ConvolutionParameters params) throws DMLRuntimeException {
		if (!params.input1.isInSparseFormat())
			throw new DMLRuntimeException("Incorrect usage: Call optimized versions");
		
		if( !params.input2.sparseBlock.isEmpty(n) ) {
			int [] tensorIndexes = new int[3];
			int apos = params.input2.sparseBlock.pos(n);
			int alen = params.input2.sparseBlock.size(n);
			int[] aix = params.input2.sparseBlock.indexes(n);
			double[] avals = params.input2.sparseBlock.values(n);
			for(int j = apos; j < apos+alen; j++) {
				computeTensorIndexes(aix[j], tensorIndexes, params.P, params.Q);
				int c = tensorIndexes[0];
				int p = tensorIndexes[1];
				int q = tensorIndexes[2];
				final int inputOffset = n*params.C*params.H*params.W + c*params.H*params.W;
				int maxIndex = getMaxIndexSparse(p, q, inputOffset, n, c, params.input1, params);
				if(maxIndex != -1)
					outputArray[maxIndex] += avals[j];
			}
		}
	}
	
	private static void doPoolingBackwardDenseSparse(int n, double [] inputArray, 
			MatrixBlock dout, double [] outputArray, ConvolutionParameters params) throws DMLRuntimeException {
		if( !dout.sparseBlock.isEmpty(n) ) {
			int [] tensorIndexes = new int[3];
			int apos = dout.sparseBlock.pos(n);
			int alen = dout.sparseBlock.size(n);
			int[] aix = dout.sparseBlock.indexes(n);
			double[] avals = dout.sparseBlock.values(n);
			for(int j = apos; j < apos+alen; j++) {
				computeTensorIndexes(aix[j], tensorIndexes, params.P, params.Q);
				int c = tensorIndexes[0];
				int p = tensorIndexes[1];
				int q = tensorIndexes[2];
				final int inputOffset = n*params.C*params.H*params.W + c*params.H*params.W;
				int maxIndex = getMaxIndex(p, q, inputOffset, inputArray, params);
				if(maxIndex != -1)
					outputArray[maxIndex] += avals[j];
			}
		}
	}
	
	private static void doPoolingBackwardDenseDense(int n, double [] inputArray, double [] doutArray, 
			double [] outputArray, ConvolutionParameters params) {
		for (int c = 0; c < params.C; c++) {
			final int inputOffset = n*params.C*params.H*params.W + c*params.H*params.W;
			final int outputOffset = n*params.C*params.P*params.Q + c*params.P*params.Q;
			
			for (int p = 0; p < params.P; p++) {
				for (int q = 0; q < params.Q; q++) {
					int maxIndex = getMaxIndex(p, q, inputOffset, inputArray, params);
					if(maxIndex != -1)
						outputArray[maxIndex] += doutArray[outputOffset +  p * params.Q + q];
				}
			}
		}
	}
	
	/**
	 * Returns the index of cell with maximum value. This method is optimized for sparse input
	 * 
	 * @param p output feature map height
	 * @param q output feature map width
	 * @param inputOffset offset to be used for input index
	 * @param n number of images
	 * @param c number of channels 
	 * @param input input matrix
	 * @param params convolution parameters
	 * @return index of the cell with maximum value
	 * @throws DMLRuntimeException if error occurs
	 */
	private static int getMaxIndexSparse(int p, int q, int inputOffset, int n, int c, MatrixBlock input, ConvolutionParameters params) throws DMLRuntimeException {
		if(!input.isInSparseFormat())
			throw new DMLRuntimeException("Incorrect usage: Only sparse format supported");
		
		int [] tensorIndexes = new int[3];
		
		int start_index_h = params.start_indexes_h[p];
		int end_index_h = params.end_indexes_h[p];
		int start_index_w = params.start_indexes_w[q];
		int end_index_w = params.end_indexes_w[q];
		
		int maxIndex = -1; 
		double maxVal = -Double.MAX_VALUE;
		
		// Note: We do not treat pad as zero and hence we don't do:  
		// maxVal = 0 
		// if start_index_h < 0 || start_index_w < 0 || end_index_h >= params.H || end_index_w >= params.W

		// input.isEmptyBlock() check is done by the caller
		if( !input.sparseBlock.isEmpty(n) ) {
			// Find maxIndex
			int apos = input.sparseBlock.pos(n);
			int alen = input.sparseBlock.size(n);
			int[] aix = input.sparseBlock.indexes(n);
			double[] avals = input.sparseBlock.values(n);
			for(int j=apos; j<apos+alen; j++) {
				computeTensorIndexes(aix[j], tensorIndexes, params.H, params.W);
				if(c != tensorIndexes[0])
					continue;
				int h = tensorIndexes[1];
				int w = tensorIndexes[2];
				if(h >= start_index_h && h < end_index_h && w >= start_index_w && w < end_index_w) {
					if(maxVal < avals[j]) {
						maxIndex = inputOffset +  h*params.W + w;
						maxVal = avals[j];
					}
				}
			}
		}
		else {
			maxIndex = inputOffset;
		}
		return maxIndex;
	}
	
	/**
	 * Returns the index of cell with maximum value. This method is optimized for dense input
	 * 
	 * @param p output feature map height
	 * @param q output feature map width
	 * @param inputOffset offset to be used for input index
	 * @param inputArray input array
	 * @param params convolution parameters
	 * @return index of cell with maximum value
	 */
	private static int getMaxIndex(int p, int q, int inputOffset, double [] inputArray, ConvolutionParameters params) {
		int start_index_h = params.start_indexes_h[p];
		int end_index_h = params.end_indexes_h[p];
		int start_index_w = params.start_indexes_w[q];
		int end_index_w = params.end_indexes_w[q];
		
		int maxIndex = -1; 
		double maxVal = -Double.MAX_VALUE;
		
		// Note: We do not treat pad as zero and hence we don't do:  
		// maxVal = 0 
		// if start_index_h < 0 || start_index_w < 0 || end_index_h >= params.H || end_index_w >= params.W
		
		// Find maxIndex
		double currDoutVal = -1;
		for (int h = start_index_h; h < end_index_h; h++) {
			for (int w = start_index_w; w < end_index_w; w++) {
				currDoutVal = inputArray[inputOffset +  h*params.W + w];
				if(maxVal < currDoutVal) {
					maxIndex = inputOffset +  h*params.W + w;
					maxVal = currDoutVal;
				}
			}
		}
		return maxIndex;
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
		
		runConvTask(TaskType.ReluBackward, params);
		
		//note: no post-processing as nnz maintained per task
	}
	
	private static long doReluBackward(ConvolutionParameters params, int rl, int ru) throws DMLRuntimeException {
		// (X > 0) * dout
		double [] outputArray = params.output.getDenseBlock();
		int numOutCols = params.input1.getNumColumns();
		
		if(!params.input1.isInSparseFormat() && !params.input2.isInSparseFormat()) {
			double [] inputArr = params.input1.getDenseBlock();
			double [] doutArr = params.input2.getDenseBlock();
			for(int i = rl*numOutCols; i < ru*numOutCols; i++) {
				outputArray[i] = inputArr[i] > 0 ? doutArr[i] : 0;
			}
		}
		else {
			// Perform (X > 0)
			ConvolutionUtils.scalarOperations(params.input1, outputArray, rl*numOutCols, numOutCols, rl, ru, 
					InstructionUtils.parseScalarBinaryOperator(">", false, 0));
			// Then perform (X > 0) * dout
			ConvolutionUtils.binaryOperationInPlace(params.input2, outputArray, rl*numOutCols, numOutCols, rl, ru, 
					_binaryElementWiseMultiplication);
		}
		
		//post-processing: maintain nnz
		return params.output.recomputeNonZeros(rl, ru-1, 0, numOutCols-1);
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
		runConvTask(TaskType.MaxPooling_Forward, params);
		
		//post-processing: maintain nnz
		outputBlock.recomputeNonZeros();
	}
	
	private static void doPooling(int n, ConvolutionParameters params) throws DMLRuntimeException {
		double [] inputArray = null;
		if (!params.input1.isInSparseFormat())
			inputArray = params.input1.getDenseBlock();
		double [] outputArray = null;
		if (!params.output.isInSparseFormat())
			outputArray = params.output.getDenseBlock();
		else
			throw new DMLRuntimeException("Expected the output to be allocated in dense format");
		
		final int inOffset = n*params.C*params.H*params.W;
		int out_index = n*params.C*params.P*params.Q;
		final int HW = params.H*params.W;
		
		if(inputArray != null) {
			for (int c = 0; c < params.C; c++) {
				final int inOffset1 = inOffset + c*HW;
				for (int p = 0; p < params.P; p++) {
					for (int q = 0; q < params.Q; q++, out_index++) {
						for (int h = params.start_indexes_h[p]; h < params.end_indexes_h[p]; h++) {
							for (int w = params.start_indexes_w[q]; w < params.end_indexes_w[q]; w++) {
								outputArray[out_index] = Math.max(outputArray[out_index], inputArray[inOffset1 +  h*params.W + w]);
							}
						}
					}
				}
			}
		}
		else {
			// TODO: Optimize sparse maxpooling
			// Low priority after adding fused relu_maxpooling operator as output of conv2d expected to be dense
			for (int c = 0; c < params.C; c++) {
				for (int p = 0; p < params.P; p++) {
					for (int q = 0; q < params.Q; q++, out_index++) {
						for (int h = params.start_indexes_h[p]; h < params.end_indexes_h[p]; h++) {
							for (int w = params.start_indexes_w[q]; w < params.end_indexes_w[q]; w++) {
								outputArray[out_index] = Math.max(outputArray[out_index], params.input1.quickGetValue(n, c*HW +  h*params.W + w));
							}
						}
					}
				}
			}
		}
	}
	
	private static void doRotate180(int inputN, int outputN, MatrixBlock input, 
			double [] outputArray,  ConvolutionParameters params, boolean zeroOutSparseOutput) throws DMLRuntimeException {
		double [] inputArray = null;
		if (!input.isInSparseFormat())
			inputArray = input.getDenseBlock();
		if(outputArray == null)
			throw new DMLRuntimeException("Sparse output is not supported for rotate180");
		
		int outputOffset = outputN*params.K*params.P*params.Q;
		if(inputArray != null) {
			for (int k = 0; k < params.K; k++) {
				for (int p = 0; p < params.P; p++) {
					for (int q = 0; q < params.Q; q++) {
						outputArray[outputOffset + p*params.Q*params.K + q*params.K + k] = inputArray[inputN*params.K*params.P*params.Q + k*params.P*params.Q + p*params.Q + q];
					}
				}
			}
		}
		else {
			if(zeroOutSparseOutput)
				Arrays.fill(outputArray, 0);
			
			if(!input.isEmptyBlock()) {
				if( !input.sparseBlock.isEmpty(inputN) ) {
					int [] tensorIndexes = new int[3];
					int apos = input.sparseBlock.pos(inputN);
					int alen = input.sparseBlock.size(inputN);
					int[] aix = input.sparseBlock.indexes(inputN);
					double[] avals = input.sparseBlock.values(inputN);
					for(int j = apos; j < apos+alen; j++) {
						computeTensorIndexes(aix[j], tensorIndexes, params.P, params.Q);
						int k = tensorIndexes[0];
						int p = tensorIndexes[1];
						int q = tensorIndexes[2];
						outputArray[outputOffset + p*params.Q*params.K + q*params.K + k] = avals[j];
					}
				}
			}
		}
	}
	
	// ----------------------------------------------------------------------------------------------------------------
	private static void addMatrixBlocks(int poolSize, TaskType type, ConvolutionParameters params, 
			ConcurrentLinkedQueue<MatrixBlock> im2ColOutBlocks, ConcurrentLinkedQueue<MatrixBlock> doutReshapedBlocks,
			ConcurrentLinkedQueue<MatrixBlock> partialRetBlocks) {
		for(int i = 0; i < poolSize; i++) {
			if(type == TaskType.LoopedIm2ColConv2d || type == TaskType.LoopedIm2ColConv2dBwdFilter) {
				MatrixBlock im2ColOutBlock = new MatrixBlock(params.C*params.R*params.S, params.P*params.Q, false);
				im2ColOutBlock.allocateDenseBlock();
				im2ColOutBlocks.add(im2ColOutBlock);
			}
			
			if(type == TaskType.LoopedIm2ColConv2dBwdFilter) {
				MatrixBlock partialRetBlock = new MatrixBlock(params.C*params.R*params.S, params.K, false);
				partialRetBlock.allocateDenseBlock();
				partialRetBlocks.add(partialRetBlock);
			}
			
			if(type == TaskType.LoopedIm2ColConv2dBwdData || type == TaskType.LoopedIm2ColConv2dBwdFilter) {
				MatrixBlock doutReshapedBlock = new MatrixBlock(params.P*params.Q, params.K, false);
				doutReshapedBlock.allocateDenseBlock();
				doutReshapedBlocks.add(doutReshapedBlock);
			}
		}
	}
	// Methods to execute convolution-related tasks using multiple threads.
	private static void runConvTask(TaskType type, ConvolutionParameters params) throws DMLRuntimeException {
		int k = OptimizerUtils.getConstrainedNumThreads(params.numThreads);
		ConcurrentLinkedQueue<MatrixBlock> im2ColOutBlocks = new ConcurrentLinkedQueue<MatrixBlock>();
		ConcurrentLinkedQueue<MatrixBlock> doutReshapedBlocks = new ConcurrentLinkedQueue<MatrixBlock>();
		ConcurrentLinkedQueue<MatrixBlock> partialRetBlocks = new ConcurrentLinkedQueue<MatrixBlock>();
		
		if (ALLOW_MULTI_THREADED_OPS && params.isOutputThreadSafe() && k > 1) {
			int poolSize = Math.min(k, params.N);
			addMatrixBlocks(poolSize, type, params, im2ColOutBlocks, doutReshapedBlocks, partialRetBlocks);
			
			ArrayList<ConvTask> tasks = new ArrayList<ConvTask>();
			int blklen = (int)(Math.ceil((double)params.N/poolSize/NUM_TASK_FACTOR));
			for( int i=0; i<poolSize*NUM_TASK_FACTOR && i*blklen<params.N; i++ )
				tasks.add(new ConvTask(i*blklen, Math.min((i+1)*blklen, params.N), 
						type, params, im2ColOutBlocks, doutReshapedBlocks, partialRetBlocks));
			
			try {
				ExecutorService pool = Executors.newFixedThreadPool( poolSize );
				List<Future<Long>> taskret = pool.invokeAll(tasks);
				pool.shutdown();
				for( Future<Long> task : taskret )
					params.output.nonZeros += task.get();
				if(type == TaskType.LoopedIm2ColConv2dBwdFilter) {
					elementWiseInPlaceTransposedAddition(params.output, partialRetBlocks.toArray(new MatrixBlock[0]));
				}
			} 
			catch (Exception e) {
				throw new DMLRuntimeException("Error while executing multi-threaded " + type.name(), e);
			}
		}
		else {
			addMatrixBlocks(1, type, params, im2ColOutBlocks, doutReshapedBlocks, partialRetBlocks);
			try {
				//execute single task and maintain nnz if supported
				params.output.setNonZeros(new ConvTask(0, params.N, type, params, im2ColOutBlocks, 
						doutReshapedBlocks, partialRetBlocks).call());
				
				if(type == TaskType.LoopedIm2ColConv2dBwdFilter) {
					elementWiseInPlaceTransposedAddition(params.output, partialRetBlocks.toArray(new MatrixBlock[0]));
				}
			} catch (Exception e) {
				throw new DMLRuntimeException("Error while executing single-threaded " + type.name(), e);
			}
		}
	}
	// ----------------------------------------------------------------------------------------------------------------
	
	/**
	 * The ConvTask allows the convolution operations (such s conv2d, conv2d_backward, maxpooling, etc)
	 * to be executed in multi-thread manner.
	 * 
	 */
	private static class ConvTask implements Callable<Long> 
	{
		public int _rl; 
		public int _ru; 
		private final ConvolutionParameters _params;
		private final TaskType _type;
		private final ConcurrentLinkedQueue<MatrixBlock> _im2ColOutBlocks;
		private final ConcurrentLinkedQueue<MatrixBlock> _partialRetBlocks;
		private final ConcurrentLinkedQueue<MatrixBlock> _doutReshapedBlocks;
		
		public ConvTask(int rl, int ru, TaskType type, ConvolutionParameters params, 
				ConcurrentLinkedQueue<MatrixBlock> im2ColOutBlocks,
				ConcurrentLinkedQueue<MatrixBlock> doutReshapedBlocks,
				ConcurrentLinkedQueue<MatrixBlock> partialRetBlocks) {
			_rl = rl;
			_ru = ru;
			_type = type;
			_params = params;
			_im2ColOutBlocks = im2ColOutBlocks;
			_partialRetBlocks = partialRetBlocks;
			_doutReshapedBlocks = doutReshapedBlocks;
		}
		
		@Override
		public Long call() throws DMLRuntimeException {
			long lnnz = 0; //nnz per partition
			
			switch(_type) {
				case MaxPooling_Forward:
					for(int n = _rl; n < _ru; n++)
						doPooling(n, _params);
					break;
				case MaxPooling_Backward:
					for(int n = _rl; n < _ru; n++) 
						doPoolingBackward(n, _params);
					break;
				case BiasAdd:
				{
					double [] dest = _params.output.getDenseBlock();
					ConvolutionUtils.binaryBiasOperations(_params.input1, _params.bias, dest, _params.K, _params.P*_params.Q, 
							_rl, _ru, _binaryElementWiseAddition);
					break;
				}
				case BiasMultiply:
				{
					double [] dest = _params.output.getDenseBlock();
					ConvolutionUtils.binaryBiasOperations(_params.input1, _params.bias, dest, _params.K, _params.P*_params.Q, 
							_rl, _ru, _binaryElementWiseMultiplication);
					break;
				}
				case ReluBackward:
					lnnz = doReluBackward(_params, _rl, _ru);
					break;
				case LoopedIm2ColConv2d:
				{	
					MatrixBlock im2ColOutBlock = _im2ColOutBlocks.remove();
					double [] temp = _params.input1.isInSparseFormat() ? new double[_params.input1.getNumColumns()] : null;
					for(int n = _rl; n < _ru; n++) 
						doLoopedIm2ColConv2d(n, im2ColOutBlock, _params, temp);
					_im2ColOutBlocks.add(im2ColOutBlock);
					if(_params.bias != null)
						ConvolutionUtils.binaryBiasOperationInPlace(_params.bias, _params.output.getDenseBlock(), _params.K, 
								_params.P*_params.Q, _rl, _ru, _binaryElementWiseAddition);
					break;
				}
				case LoopedIm2ColConv2dBwdFilter:
				{
					MatrixBlock im2ColOutBlock = _im2ColOutBlocks.remove();
					MatrixBlock partialRetBlock = _partialRetBlocks.remove();
					MatrixBlock doutReshapedBlock = _doutReshapedBlocks.remove();
					double [] temp = _params.input1.isInSparseFormat() ? new double[_params.input1.getNumColumns()] : null;
					for(int n = _rl; n < _ru; n++) 
						partialRetBlock = doLoopedIm2ColConv2dBwdFilter(n, im2ColOutBlock, doutReshapedBlock, partialRetBlock, _params, temp);
					_im2ColOutBlocks.add(im2ColOutBlock);
					_partialRetBlocks.add(partialRetBlock);
					_doutReshapedBlocks.add(doutReshapedBlock);
					break;
				}
				case LoopedIm2ColConv2dBwdData:
				{
					MatrixBlock doutReshapedBlock = _doutReshapedBlocks.remove();
					for(int n = _rl; n < _ru; n++) 
						doLoopedIm2ColConv2dBwdData(n, doutReshapedBlock, _params);
					_doutReshapedBlocks.add(doutReshapedBlock);
					break;
				}
				default:
					throw new DMLRuntimeException("Unsupported ConvTask:" + _type.name());
			}
			
			return lnnz;
		}
	}
		
	// Converts input: PQ X CRS matrix and writes to 1 X CHW
	private static void doCol2imOverSingleImage(int outputN, MatrixBlock input, ConvolutionParameters params) throws DMLRuntimeException {
		if(input.rlen != params.P*params.Q || input.clen != params.C*params.R*params.S) {
			throw new DMLRuntimeException("Incorrect input dimensions");
		}
		
		double [] outputArray = null;
		if (!params.output.isInSparseFormat())
			outputArray = params.output.getDenseBlock();
		else {
			throw new DMLRuntimeException("Only dense output is implemented");
		}
		
		if(!input.isInSparseFormat()) {
			double [] inputArray = input.getDenseBlock();
			doCol2IMDenseInput(0, outputN, inputArray, outputArray, params);
		}
		else {
			if(!input.isEmptyBlock()) {
				int [] tensorIndexes = new int[3];
				for(int i = 0; i < input.getNumRows(); i++) {
					if( !input.sparseBlock.isEmpty(i) ) {
						computeTensorIndexes(i, tensorIndexes, params.P, params.Q);
						int p = tensorIndexes[1];
						int q = tensorIndexes[2];
						if(tensorIndexes[0] != 0) 
							throw new DMLRuntimeException("Incorrect tensor indexes: " + tensorIndexes[0] + " != 0 <" + p + " " + q + " " + tensorIndexes[0] + params.P + " " + params.Q + ">");
						
						int apos = input.sparseBlock.pos(i);
						int alen = input.sparseBlock.size(i);
						int[] aix = input.sparseBlock.indexes(i);
						double[] avals = input.sparseBlock.values(i);
						for(int j = apos; j < apos+alen; j++) {
							computeTensorIndexes(aix[j], tensorIndexes, params.R, params.S);
							int c = tensorIndexes[0];
							int r = tensorIndexes[1];
							int s = tensorIndexes[2];
							int h = p*params.stride_h + r - params.pad_h;
							int w = q*params.stride_w + s - params.pad_w;
							if(h >= 0 && h < params.H && w >= 0 && w < params.W) {
								int outIndex = outputN*params.C*params.H*params.W + c*params.H*params.W + h*params.W + w;
								outputArray[outIndex] += avals[j];
							}
						}
					}
				}
			}
		}
	}
	
	// Converts input: PQ X CRS matrix and writes to 1 X CHW if inputN == 0
	// Or converts input: NPQ X CRS matrix and writes to N X CHW 
	private static void doCol2IMDenseInput(int inputN, int outputN, double [] inputArray, double [] outputArray, ConvolutionParameters params) throws DMLRuntimeException {
		final int outputNOffset = outputN*params.C*params.H*params.W;
		for (int p = 0; p < params.P; p++) {
			// h = p*params.stride_h + r - params.pad_h
			//   = r + hOffset
			// Based on restrictions: h >= 0 and r >= 0 and h < params.H and r < params.R, we get
			// max(0, - hOffset) <= r < min(params.R, params.H - hOffset)
			final int hOffset = p*params.stride_h - params.pad_h;
			final int rStart = Math.max(0, - hOffset);
			final int rEnd = Math.min(params.R, params.H - hOffset);
			for (int q = 0; q < params.Q; q++) {
				// Using the same logic as above on following:
				// w = q*params.stride_w + s - params.pad_w
				final int wOffset = q*params.stride_w - params.pad_w;
				final int sStart = Math.max(0, - wOffset);
				final int sEnd = Math.min(params.S, params.W - wOffset);
				final int tempOffset = (inputN*params.P*params.Q + p*params.Q + q)*params.C*params.R*params.S;
				for (int c = 0; c < params.C; c++) {
					final int outOffset = outputNOffset + c*params.H*params.W;
					final int inputOffset = tempOffset + c*params.R*params.S;
					for (int r = rStart; r < rEnd; r++) {
						for (int s = sStart; s < sEnd; s++) {
							int inputIndex = inputOffset + r*params.S + s;
							int outIndex = outOffset + (hOffset + r)*params.W + wOffset + s;
							outputArray[outIndex] += inputArray[inputIndex];
						}
					}
				}
			}
		}
	}
	
	private static void doIm2colDense(int n, double [] inputArray, double [] outputArray, ConvolutionParameters params) {
		int CRS = params.C * params.R * params.S;
		final int nOffset = n * params.C*params.H*params.W;
		if (params.stride_h == 1 && params.stride_w == 1 && params.pad_h == 0 && params.pad_w == 0) {
			for (int c = 0; c < CRS; ++c) {
				int wOffset = c % params.S;
				int hOffset = (c / params.S) % params.R;
				int cInput = c / params.R / params.S;
				for (int h = 0; h < params.P; ++h) {
					int hPadded = h + hOffset;
					int outOffset = (c * params.P + h) * params.Q;
					int inputOffset = nOffset + (cInput * params.H + hPadded) * params.W;
					System.arraycopy(inputArray, inputOffset + wOffset, outputArray, outOffset, params.Q);
					int w = params.Q - 1;
					int wPadded = w + wOffset;
					if (hPadded < params.H && wPadded < params.W)
						outputArray[outOffset + w] = inputArray[inputOffset + wPadded];
					else
						outputArray[outOffset + w] = 0;
				}
			}
		} else {
			for (int c = 0; c < CRS; ++c) {
				int wOffset = c % params.S;
				int hOffset = (c / params.S) % params.R;
				int cInput = c / params.R / params.S;
				for (int h = 0; h < params.P; ++h) {
					int outOffset = (c * params.P + h) * params.Q;
					int hPadded = h * params.stride_h - params.pad_h + hOffset;
					int inputOffset = nOffset + (cInput * params.H + hPadded) * params.W;
					if (hPadded < 0 || hPadded >= params.H) {
						Arrays.fill(outputArray, outOffset, outOffset+params.Q, 0);
					} else {
						for (int w = 0; w < params.Q; ++w) {
							int wPadded = w * params.stride_w - params.pad_w + wOffset;
							if (wPadded >= 0 && wPadded < params.W)
								outputArray[outOffset + w] = inputArray[inputOffset + wPadded];
							else
								outputArray[outOffset + w] = 0;
						}
					}
				}
			}
		}
	}
	
	// Returns the row of matrix in dense format
	private static double [] getRowInDenseFormat(MatrixBlock input, int n, double []  temp) {
		// Use temporary array to avoid binary search
		Arrays.fill(temp, 0);
		if( !input.sparseBlock.isEmpty(n) ) {
			int apos = input.sparseBlock.pos(n);
			int alen = input.sparseBlock.size(n);
			int[] aix = input.sparseBlock.indexes(n);
			double[] avals = input.sparseBlock.values(n);
			for(int j=apos; j<apos+alen; j++)
				temp[ aix[j] ] = avals[j];
		}
		return temp;
	}
	
	// Keeping this as a separate sparse method to allow for further dense optimizations
	private static void doIm2colSparse(int n, MatrixBlock input, double [] outputArray, ConvolutionParameters params, double []  temp) throws DMLRuntimeException {
		int CRS = params.C * params.R * params.S;
		
		// Using a temporary array improves performance by not requiring binary search for getValue
		// Since the access pattern depends on ConvolutionParameters, this serves as a temporary fix.
		temp = getRowInDenseFormat(input, n, temp);
		// final int nOffset = n * params.C*params.H*params.W;
		for (int c = 0; c < CRS; ++c) {
			int wOffset = c % params.S;
			int hOffset = (c / params.S) % params.R;
			int cInput = c / params.R / params.S;
			for (int h = 0; h < params.P; ++h) {
				int outOffset = (c * params.P + h) * params.Q;
				int hPadded = h * params.stride_h - params.pad_h + hOffset;
				int tempOffset = (cInput * params.H + hPadded) * params.W;
				// int inputOffset = nOffset + tempOffset;
				if (hPadded < 0 || hPadded >= params.H) {
					Arrays.fill(outputArray, outOffset, outOffset+params.Q, 0);
				} else {
					for (int w = 0; w < params.Q; ++w) {
						int wPadded = w * params.stride_w - params.pad_w + wOffset;
						if (wPadded >= 0 && wPadded < params.W) 
							outputArray[outOffset + w] = temp[tempOffset + wPadded];
						else
							outputArray[outOffset + w] = 0;
					}
				}
			}
		}
	}
	
	private static void doIm2col(int n, MatrixBlock output, ConvolutionParameters params, double []  temp) throws DMLRuntimeException {
		double [] inputArray = null;
		if (!params.input1.isInSparseFormat())
			inputArray = params.input1.getDenseBlock();
		double [] outputArray = null;
		if(!output.isInSparseFormat())
			outputArray = output.getDenseBlock();
		else 
			throw new DMLRuntimeException("Sparse output is not supported for im2col");
		
		if(inputArray != null)
			doIm2colDense(n, inputArray, outputArray, params);
		else
			doIm2colSparse(n, params.input1, outputArray, params, temp);
	}
}
