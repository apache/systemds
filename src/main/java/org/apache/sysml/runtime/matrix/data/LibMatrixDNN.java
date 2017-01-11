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
import java.util.Iterator;
import java.util.List;
import java.util.concurrent.Callable;
import java.util.concurrent.ConcurrentLinkedQueue;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.concurrent.atomic.AtomicLong;

import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;
import org.apache.sysml.api.DMLScript;
import org.apache.sysml.hops.OptimizerUtils;
import org.apache.sysml.runtime.DMLRuntimeException;

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
	// ------------------------------------------------------------------------------------------------
	// Useful flags for performance testing:
	private static boolean DISPLAY_STATISTICS = false;
	private static final boolean ALLOW_MULTI_THREADED_OPS = true;
	// ------------------------------------------------------------------------------------------------
	
	enum TaskType {
		MaxPooling_Forward, MaxPooling_Backward, 
		// Alternate approaches that we tried but the performance was unsatisfactory be included: direct, non-looped im2col
		LoopedIm2ColConv2d, LoopedIm2ColConv2dBwdFilter, LoopedIm2ColConv2dBwdData,
		BiasAdd, ReluBackward
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
	}
	
	/**
	 * Performs the operation: ret += elem
	 * @param ret left and output matrix
	 * @param elem right matrix
	 * @throws DMLRuntimeException if DMLRuntimeException occurs
	 */
	private static void elementWiseInPlaceAddition(MatrixBlock ret, MatrixBlock elem) throws DMLRuntimeException {
		if(ret.getNumRows() != elem.getNumRows() || ret.getNumColumns() != elem.getNumColumns()) {
			throw new DMLRuntimeException("Incorrect dimensions");
		}
		if(!ret.isInSparseFormat() && !elem.isInSparseFormat()) {
			for(int i = 0; i < ret.getNumRows()*ret.getNumColumns(); i++) {
				ret.denseBlock[i] += elem.denseBlock[i];
			}
		}
		else if(!ret.isInSparseFormat() && elem.isInSparseFormat()) {
			if(!elem.isEmptyBlock()) {
				Iterator<IJV> iter = elem.sparseBlock.getIterator();
				int numCol = ret.getNumColumns();
				while(iter.hasNext()) {
					IJV ijv = iter.next();
					int index = ijv.getI()*numCol + ijv.getJ();
					ret.denseBlock[index] += ijv.getV(); 
				}
			}
		}
		else {
			throw new DMLRuntimeException("Sparse return format not supported");
		}
	}
	
	/**
	 * Performs the operation: ret += t(elem)
	 * @param ret left and output matrix
	 * @param elem right untransposed matrix
	 * @param params convolution parameters
	 * @throws DMLRuntimeException if DMLRuntimeException occurs
	 */
	private static void elementWiseInPlaceTransposedAddition(MatrixBlock ret, MatrixBlock elem) throws DMLRuntimeException {
		if(ret.getNumRows() != elem.getNumColumns() || ret.getNumColumns() != elem.getNumRows()) {
			throw new DMLRuntimeException("Incorrect dimensions");
		}
		int numRow = ret.getNumColumns();
		if(!ret.isInSparseFormat() && !elem.isInSparseFormat()) {
			int iter = 0;
			for(int i = 0; i < elem.getNumRows(); i++) {
				for(int j = 0; j < elem.getNumColumns(); j++, iter++) {
					int index = j*numRow+i;
					ret.denseBlock[index] += elem.denseBlock[iter];
				}
			}
		}
		else if(!ret.isInSparseFormat() && elem.isInSparseFormat()) {
			if(!elem.isEmptyBlock()) {
				Iterator<IJV> iter = elem.sparseBlock.getIterator();
				while(iter.hasNext()) {
					IJV ijv = iter.next();
					int index = ijv.getJ()*numRow + ijv.getI();
					ret.denseBlock[index] += ijv.getV(); 
				}
			}
		}
		else {
			throw new DMLRuntimeException("Sparse return format not supported");
		}
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
			MatrixBlock im2ColOutBlock, MatrixBlock dout_reshaped, MatrixBlock partialRetBlock, ConvolutionParameters params) throws DMLRuntimeException {
		long t1 = DMLScript.STATISTICS && DISPLAY_STATISTICS ? System.nanoTime() : 0;
		doIm2col(n, im2ColOutBlock, params);
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
		if(!temp.isEmptyBlock())
			elementWiseInPlaceTransposedAddition(partialRetBlock, temp);
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
			throw new DMLRuntimeException("Incorrect input to conv2d");
		}
		
		if(DMLScript.STATISTICS && DISPLAY_STATISTICS) {
			if(input.isInSparseFormat() || filter.isInSparseFormat()) {
				conv2dSparseCount.addAndGet(1);
			}
			else {
				conv2dDenseCount.addAndGet(1);
			}
		}
		
		if(!input.isInSparseFormat() && TEST_SPARSE_INPUT) {
			input.denseToSparse();
		}
		if(!filter.isInSparseFormat() && TEST_SPARSE_FILTER) {
			filter.denseToSparse();
		}
		
		runConvTask(TaskType.LoopedIm2ColConv2d, params);
	}
	
	private static void doLoopedIm2ColConv2d(int n, MatrixBlock im2ColOutBlock, ConvolutionParameters params) throws DMLRuntimeException {
		long t1 = DMLScript.STATISTICS && DISPLAY_STATISTICS ? System.nanoTime() : 0;
		doIm2col(n, im2ColOutBlock, params);
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
				// NOTE: Potential bottlenc to copy sparse matmult back to dense output
				Iterator<IJV> iter = matMultOutBlock.sparseBlock.getIterator();
				final int outOffset = n*params.K*params.P*params.Q;
				while(iter.hasNext()) {
					IJV ijv = iter.next();
					int k = ijv.getI();
					int p = ijv.getJ() / params.Q;
					int q = ijv.getJ() % params.Q;
					params.output.denseBlock[outOffset + k*params.P*params.Q + p*params.Q + q] = ijv.getV();
				}
			}
			else
				System.arraycopy(matMultOutBlock.denseBlock, 0, params.output.denseBlock, destPos, length);
		}
		// -----------------------------------------------------------------------------
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
	}
	
	private static void fillIndexesArray(ConvolutionParameters params) {
		params.start_indexes_h = new int[params.P];
		params.end_indexes_h = new int[params.P];
		params.start_indexes_w = new int[params.Q];
		params.end_indexes_w = new int[params.Q];
		for (int p = 0; p < params.P; p++) {
			int start_index_h = p * params.stride_h - params.pad_h;
			final int end_index_h = Math.min(start_index_h + params.R, params.H);
			start_index_h = Math.max(start_index_h, 0);
			params.start_indexes_h[p] = start_index_h;
			params.end_indexes_h[p] = end_index_h;
		}
		for (int q = 0; q < params.Q; q++) {
			int start_index_w = Math.max(q * params.stride_w - params.pad_w, 0);
			int end_index_w = Math.min(start_index_w + params.S, params.W);
			start_index_w = Math.max(start_index_w, 0);
			params.start_indexes_w[q] = start_index_w;
			params.end_indexes_w[q] = end_index_w;
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
						outputArray[maxIndex] += inVal;
					}
				}
			}
		}
	}
	
	private static void doPoolingBackwardSparseSparse(int n, double [] outputArray, ConvolutionParameters params) throws DMLRuntimeException {
		if (!params.input1.isInSparseFormat())
			throw new DMLRuntimeException("Incorrect usage: Call optimized versions");
		
		// params.input2.isEmptyBlock() check is done by the caller
		Iterator<IJV> iter = params.input2.sparseBlock.getIterator(n, n+1);
		int [] tensorIndexes = new int[3];
		
		while(iter.hasNext()) {
			IJV ijv = iter.next();
			computeTensorIndexes(ijv.getJ(), tensorIndexes, params.P, params.Q);
			int c = tensorIndexes[0];
			int p = tensorIndexes[1];
			int q = tensorIndexes[2];
			
			final int inputOffset = n*params.C*params.H*params.W + c*params.H*params.W;
			int maxIndex = getMaxIndexSparse(p, q, inputOffset, n, c, params.input1, params);
			outputArray[maxIndex] += ijv.getV();
		}
		
	}
	
	private static void doPoolingBackwardDenseSparse(int n, double [] inputArray, 
			MatrixBlock dout, double [] outputArray, ConvolutionParameters params) throws DMLRuntimeException {
		// dout.isEmptyBlock() check is done by the caller
		Iterator<IJV> iter = dout.sparseBlock.getIterator(n, n+1);
		int [] tensorIndexes = new int[3];
		
		while(iter.hasNext()) {
			IJV ijv = iter.next();
			computeTensorIndexes(ijv.getJ(), tensorIndexes, params.P, params.Q);
			int c = tensorIndexes[0];
			int p = tensorIndexes[1];
			int q = tensorIndexes[2];
			
			final int inputOffset = n*params.C*params.H*params.W + c*params.H*params.W;
			int maxIndex = getMaxIndex(p, q, inputOffset, inputArray, params);
			outputArray[maxIndex] += ijv.getV();
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
					outputArray[maxIndex] += doutArray[outputOffset +  p * params.Q + q];
				}
			}
		}
	}
	
	private static int getMaxIndexSparse(int p, int q, int inputOffset, int n, int c, MatrixBlock input, ConvolutionParameters params) throws DMLRuntimeException {
		if(!input.isInSparseFormat())
			throw new DMLRuntimeException("Incorrect usage: Only sparse format supported");
		
		// input.isEmptyBlock() check is done by the caller
		Iterator<IJV> iter = input.sparseBlock.getIterator(n, n+1);
		int [] tensorIndexes = new int[3];
		
		int start_index_h = params.start_indexes_h[p];
		int end_index_h = params.end_indexes_h[p];
		int start_index_w = params.start_indexes_w[q];
		int end_index_w = params.end_indexes_w[q];
		
		int maxIndex = inputOffset +  start_index_h*params.W + start_index_w; 
		double maxVal = -Double.MAX_VALUE;

		// Find maxIndex
		double currDoutVal = -1;
		while(iter.hasNext()) {
			IJV ijv = iter.next();
			computeTensorIndexes(ijv.getJ(), tensorIndexes, params.H, params.W);
			if(c != tensorIndexes[0])
				continue;
			int h = tensorIndexes[1];
			int w = tensorIndexes[2];
			if(h >= start_index_h && h < end_index_h && w >= start_index_w && w < end_index_w) {
				currDoutVal = ijv.getV();
				if(maxVal < currDoutVal) {
					maxIndex = inputOffset +  h*params.W + w;
					maxVal = currDoutVal;
				}
			}	
		}
		return maxIndex;
	}
	
	private static int getMaxIndex(int p, int q, int inputOffset, double [] inputArray, ConvolutionParameters params) {
		int start_index_h = params.start_indexes_h[p];
		int end_index_h = params.end_indexes_h[p];
		int start_index_w = params.start_indexes_w[q];
		int end_index_w = params.end_indexes_w[q];
		
		int maxIndex = inputOffset +  start_index_h*params.W + start_index_w; 
		double maxVal = -Double.MAX_VALUE;

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
	}
	
	private static void doReluBackward(int n, ConvolutionParameters params) throws DMLRuntimeException {
		// (X > 0) * dout
		double [] outputArray = params.output.getDenseBlock();
		int numOutCols = params.input1.getNumColumns();
		
		if(!params.input1.isInSparseFormat() && !params.input2.isInSparseFormat()) {
			double [] inputArr = params.input1.getDenseBlock();
			double [] doutArr = params.input2.getDenseBlock();
			for(int i = n*numOutCols; i < (n+1)*numOutCols; i++) {
				outputArray[i] = inputArr[i] > 0 ? doutArr[i] : 0;
			}
		}
		else {
			// Perform (X > 0)
			if(params.input1.isInSparseFormat()) {
				Iterator<IJV> iter = params.input1.sparseBlock.getIterator(n, n+1);
				while(iter.hasNext()) {
					IJV ijv = iter.next();
					int i = ijv.getI();
					int j = ijv.getJ();
					outputArray[i*numOutCols + j] = ijv.getV() > 0 ? 1 : 0;
				}
			}
			else {
				double [] inputArr = params.input1.getDenseBlock();
				for(int i = n*numOutCols; i < (n+1)*numOutCols; i++) {
					outputArray[i] = inputArr[i] > 0 ? 1 : 0;
				}
			}
			// Then perform (X > 0) * dout
			if(params.input2.isInSparseFormat()) {
				Iterator<IJV> iter = params.input2.sparseBlock.getIterator(n, n+1);
				while(iter.hasNext()) {
					IJV ijv = iter.next();
					int i = ijv.getI();
					int j = ijv.getJ();
					outputArray[i*numOutCols + j] *= ijv.getV();
				}
			}
			else {
				double [] doutArr = params.input2.getDenseBlock();
				for(int i = n*numOutCols; i < (n+1)*numOutCols; i++) {
					outputArray[i] *= doutArr[i];
				}
			}
		}
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
		
		ConvolutionParameters params = new ConvolutionParameters(N, PQ, -1, -1, K, -1, -1, -1, -1, -1, -1, numThreads);
		params.input1 = input;
		params.input2 = bias;
		params.output = outputBlock;
		
		if(!input.isInSparseFormat() && TEST_SPARSE_INPUT) {
			input.denseToSparse();
		}
		if(!bias.isInSparseFormat() && TEST_SPARSE_FILTER) {
			bias.denseToSparse();
		}
		
		if(bias.getNumColumns() != 1 || input.getNumColumns() % K != 0) {
			throw new DMLRuntimeException("Incorrect inputs for bias_add: input[" + N + " X " + input.getNumColumns()  + "] and bias[" + K + " X " + bias.getNumColumns() + "]");
		}
		
		if(input.isEmptyBlock()) {
			double [] outputArray = outputBlock.getDenseBlock();
			for(int n = 0;  n < N; n++) 
				fillBias(bias, outputArray, n, n+1, N, K, PQ);
		}
		else {
			runConvTask(TaskType.BiasAdd, params);
		}
	}
	
	private static void doBiasAdd(int n1, int n2, ConvolutionParameters params) throws DMLRuntimeException {
		double [] outputArray = params.output.getDenseBlock();
		int PQ = params.C;
		int numOutCols = params.input1.getNumColumns();
		
		if(!params.input1.isInSparseFormat() && !params.input2.isInSparseFormat()) {
			double [] inputArr = params.input1.getDenseBlock();
			double [] biasArr = params.input2.getDenseBlock();
			int K = params.K;
			int index = n1*K*PQ;
			for(int n = n1; n < n2; n++) {
				for(int k = 0; k < K; k++) {
					for(int pq = 0; pq < PQ; pq++, index++) {
						outputArray[index] = inputArr[index] + biasArr[k];
					}
				}
			}
		}
		else {
			fillBias(params.input2, outputArray, n1, n2, params.N, params.K, PQ);
			if(params.input1.isInSparseFormat()) {
				Iterator<IJV> iter = params.input1.sparseBlock.getIterator(n1, n2);
				while(iter.hasNext()) {
					IJV ijv = iter.next();
					int i = ijv.getI();
					int j = ijv.getJ();
					outputArray[i*numOutCols + j] += ijv.getV();
				}
			}
			else {
				double [] inputArr = params.input1.getDenseBlock();
				for(int i = n1*numOutCols; i < n2*numOutCols; i++) {
					outputArray[i] += inputArr[i];
				}
			}
		}
		
	}
	
	private static void fillBias(MatrixBlock bias, double [] outputArray, int n1, int n2, int N, int K, int PQ) {
		if(bias.isInSparseFormat()) {
			Iterator<IJV> iter = bias.sparseBlock.getIterator();
			while(iter.hasNext()) {
				IJV ijv = iter.next();
				int k = ijv.getI();
				double val = ijv.getV();
				for(int n = n1; n < n2; n++) {
					int fromIndex = n*K*PQ + k*PQ;
					Arrays.fill(outputArray, fromIndex, fromIndex + PQ, val);
				}
			}
		}
		else {
			double [] biasArr = bias.getDenseBlock();
			for(int n = n1; n < n2; n++) {
				for(int k = 0; k < K; k++) {
					int fromIndex = n*K*PQ + k*PQ;
					double val = biasArr[k];
					Arrays.fill(outputArray, fromIndex, fromIndex + PQ, val);
				}
			}
		}
	}

	public static void maxpooling(MatrixBlock input, MatrixBlock outputBlock, ConvolutionParameters params) throws DMLRuntimeException {
		params.input1 = input;
		params.output = outputBlock;
		
		if(input.getNumColumns() != params.C*params.H*params.W || input.getNumRows() != params.N) {
			throw new DMLRuntimeException("Incorrect input dimensions in maxpooling:" + input.getNumRows() + " " + input.getNumColumns() + " " + params.N + " " + params.K*params.P*params.Q);
		}
		
		fillIndexesArray(params);
		runConvTask(TaskType.MaxPooling_Forward, params);
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
								double inVal = params.input1.quickGetValue(n, c*HW +  h*params.W + w);
								outputArray[out_index] = Math.max(outputArray[out_index], inVal);
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
				Iterator<IJV> iter = input.sparseBlock.getIterator(inputN, inputN+1);
				int [] tensorIndexes = new int[3];
				while(iter.hasNext()) {
					IJV ijv = iter.next();
					computeTensorIndexes(ijv.getJ(), tensorIndexes, params.P, params.Q);
					int k = tensorIndexes[0];
					int p = tensorIndexes[1];
					int q = tensorIndexes[2];
					outputArray[outputOffset + p*params.Q*params.K + q*params.K + k] = ijv.getV();
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
				im2ColOutBlock.allocateDenseBlock(true);
				im2ColOutBlocks.add(im2ColOutBlock);
			}
			
			if(type == TaskType.LoopedIm2ColConv2dBwdFilter) {
				MatrixBlock partialRetBlock = new MatrixBlock(params.K, params.C*params.R*params.S, false);
				partialRetBlock.allocateDenseBlock(true);
				partialRetBlocks.add(partialRetBlock);
			}
			
			if(type == TaskType.LoopedIm2ColConv2dBwdData || type == TaskType.LoopedIm2ColConv2dBwdFilter) {
				MatrixBlock doutReshapedBlock = new MatrixBlock(params.P*params.Q, params.K, false);
				doutReshapedBlock.allocateDenseBlock(true);
				doutReshapedBlocks.add(doutReshapedBlock);
			}
		}
	}
	// Methods to execute convolution-related tasks using multiple threads.
	private static void runConvTask(TaskType type, ConvolutionParameters params) throws DMLRuntimeException {
		int constrainedNumThreads = OptimizerUtils.getConstrainedNumThreads(params.numThreads);
		ConcurrentLinkedQueue<MatrixBlock> im2ColOutBlocks = new ConcurrentLinkedQueue<MatrixBlock>();
		ConcurrentLinkedQueue<MatrixBlock> doutReshapedBlocks = new ConcurrentLinkedQueue<MatrixBlock>();
		ConcurrentLinkedQueue<MatrixBlock> partialRetBlocks = new ConcurrentLinkedQueue<MatrixBlock>();
		if (ALLOW_MULTI_THREADED_OPS && params.isOutputThreadSafe() && constrainedNumThreads > 1) {
			int poolSize = Math.min(constrainedNumThreads, params.N);
			addMatrixBlocks(poolSize, type, params, im2ColOutBlocks, doutReshapedBlocks, partialRetBlocks);
			ArrayList<ConvTask> tasks = new ArrayList<ConvTask>();
			int NSize = params.N - poolSize;
			if(NSize >= constrainedNumThreads) {
				for(int n = 0; n < params.N; n++) 
					tasks.add(new ConvTask(n, n+1, type, params, im2ColOutBlocks, doutReshapedBlocks, partialRetBlocks));
			}
			else {
				int numNTasks = (int) Math.ceil(((double) NSize) / constrainedNumThreads);
				for (int n = 0; n < NSize; n += numNTasks) {
					tasks.add(new ConvTask(n, Math.min(NSize, n+numNTasks), type, params, im2ColOutBlocks, doutReshapedBlocks, partialRetBlocks));
				}
				for (int n = NSize; n < params.N; n++)
					tasks.add(new ConvTask(n, n+1, type, params, im2ColOutBlocks, doutReshapedBlocks, partialRetBlocks));
			}
			
			ExecutorService pool = Executors.newFixedThreadPool( poolSize );
			List<Future<Object>> taskret;
			try {
				taskret = pool.invokeAll(tasks);
				pool.shutdown();
				for( Future<Object> task : taskret ) {
					task.get();
				}
				if(type == TaskType.LoopedIm2ColConv2dBwdFilter) {
					for(MatrixBlock partialRetBlock : partialRetBlocks) {
						elementWiseInPlaceAddition(params.output, partialRetBlock);
					}
				}
			} catch (InterruptedException e) {
				throw new DMLRuntimeException("Error while executing multi-threaded " + type.name(), e);
			} catch (ExecutionException e) {
				throw new DMLRuntimeException("Error while executing multi-threaded " + type.name(), e);
			}
		}
		else {
			addMatrixBlocks(1, type, params, im2ColOutBlocks, doutReshapedBlocks, partialRetBlocks);
			ConvTask task = new ConvTask(0, 0, type, params, im2ColOutBlocks, doutReshapedBlocks, partialRetBlocks);
			try {
				for(int n = 0; n < params.N; n++) {
					task.n1 = n;
					task.n2 = n+1;
					task.call();
				}
				if(type == TaskType.LoopedIm2ColConv2dBwdFilter) {
					for(MatrixBlock partialRetBlock : partialRetBlocks) {
						elementWiseInPlaceAddition(params.output, partialRetBlock);
					}
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
	private static class ConvTask implements Callable<Object> {
		public int n1; public int n2; 
		ConvolutionParameters params;
		TaskType type;
		ConcurrentLinkedQueue<MatrixBlock> im2ColOutBlocks;
		ConcurrentLinkedQueue<MatrixBlock> partialRetBlocks;
		ConcurrentLinkedQueue<MatrixBlock> doutReshapedBlocks;
		public ConvTask(int n1, int n2, TaskType type, ConvolutionParameters params, 
				ConcurrentLinkedQueue<MatrixBlock> im2ColOutBlocks,
				ConcurrentLinkedQueue<MatrixBlock> doutReshapedBlocks,
				ConcurrentLinkedQueue<MatrixBlock> partialRetBlocks) {
			this.n1 = n1;
			this.n2 = n2;
			this.type = type;
			this.params = params;
			this.im2ColOutBlocks = im2ColOutBlocks;
			this.partialRetBlocks = partialRetBlocks;
			this.doutReshapedBlocks = doutReshapedBlocks;
		}
		
		@Override
		public Object call() throws DMLRuntimeException {
			switch(type) {
				case MaxPooling_Forward:
				{
					for(int n = n1; n < n2; n++) {
						doPooling(n, params);
					}
					break;
				}
				case MaxPooling_Backward:
					for(int n = n1; n < n2; n++) 
						doPoolingBackward(n, params);
					break;
				case BiasAdd:
					doBiasAdd(n1, n2, params);
					break;
				case ReluBackward:
					for(int n = n1; n < n2; n++) 
						doReluBackward(n, params);
					break;
				case LoopedIm2ColConv2d:
				{	
					MatrixBlock im2ColOutBlock = im2ColOutBlocks.remove();
					for(int n = n1; n < n2; n++) 
						doLoopedIm2ColConv2d(n, im2ColOutBlock, params);
					im2ColOutBlocks.add(im2ColOutBlock);
					if(params.bias != null)
						addBias(n1, n2, params);
					break;
				}
				case LoopedIm2ColConv2dBwdFilter:
				{
					MatrixBlock im2ColOutBlock = im2ColOutBlocks.remove();
					MatrixBlock partialRetBlock = partialRetBlocks.remove();
					MatrixBlock doutReshapedBlock = doutReshapedBlocks.remove();
					for(int n = n1; n < n2; n++) 
						partialRetBlock = doLoopedIm2ColConv2dBwdFilter(n, im2ColOutBlock, doutReshapedBlock, partialRetBlock, params);
					im2ColOutBlocks.add(im2ColOutBlock);
					partialRetBlocks.add(partialRetBlock);
					doutReshapedBlocks.add(doutReshapedBlock);
					break;
				}
				case LoopedIm2ColConv2dBwdData:
				{
					MatrixBlock doutReshapedBlock = doutReshapedBlocks.remove();
					for(int n = n1; n < n2; n++) 
						doLoopedIm2ColConv2dBwdData(n, doutReshapedBlock, params);
					doutReshapedBlocks.add(doutReshapedBlock);
					break;
				}
				default:
					throw new DMLRuntimeException("Unsupported ConvTask:" + type.name());
			}
			return null;
		}
	}
	
	private static void addBias(int n1, int n2, ConvolutionParameters params) {
		int PQ = params.P*params.Q;
		int K = params.K;
		double [] outputArr = params.output.getDenseBlock();
		if(!params.bias.isInSparseFormat()) {
			double [] biasArr = params.bias.getDenseBlock();
			int index = n1*K*PQ;
			for(int n = n1; n < n2; n++) {
				for(int k = 0; k < K; k++) {
					for(int pq = 0; pq < PQ; pq++, index++) {
						outputArr[index] += biasArr[k];
					}
				}
			}
		}
		else {
			Iterator<IJV> iter = params.bias.getSparseBlockIterator();
			while(iter.hasNext()) {
				IJV ijv = iter.next();
				int k = ijv.getI();
				double val = ijv.getV();
				for(int n = n1; n < n2; n++) {
					int index = n*K*PQ + k*PQ;
					for(int pq = 0; pq < PQ; pq++, index++) {
						outputArr[index] += val;
					}
				}
			}
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
			if(!input.isEmptyBlock())
				doCol2IMSparseInput(0, outputN, input.getSparseBlockIterator(), outputArray, params);
		}
	}
	
	private static void doCol2IMSparseInput(int inputN, int outputN, Iterator<IJV> inputIter, double [] outputArray, ConvolutionParameters params) throws DMLRuntimeException {
		int [] tensorIndexes = new int[3];
		
		while(inputIter.hasNext()) {
			IJV ijv = inputIter.next();
			computeTensorIndexes(ijv.getJ(), tensorIndexes, params.R, params.S);
			int c = tensorIndexes[0];
			int r = tensorIndexes[1];
			int s = tensorIndexes[2];
			computeTensorIndexes(ijv.getI(), tensorIndexes, params.P, params.Q);
			int p = tensorIndexes[1];
			int q = tensorIndexes[2];
			if(inputN != tensorIndexes[0]) {
				throw new DMLRuntimeException("Incorrect tensor indexes: " + inputN + " != " + tensorIndexes[0] + " <" + p + " " + q + " " + ijv.getI() + params.P + " " + params.Q + ">");
			}
			int h = p*params.stride_h + r - params.pad_h;
			int w = q*params.stride_w + s - params.pad_w;
			if(h >= 0 && h < params.H && w >= 0 && w < params.W) {
				int outIndex = outputN*params.C*params.H*params.W + c*params.H*params.W + h*params.W + w;
				outputArray[outIndex] += ijv.getV();
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
	
	// Keeping this as a separate sparse method to allow for further dense optimizations
	private static void doIm2colSparse(int n, MatrixBlock input, double [] outputArray, ConvolutionParameters params) {
		int CRS = params.C * params.R * params.S;
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
						if (wPadded >= 0 && wPadded < params.W) {
							// NOTE: Potential performance bottleneck as we have to do binary search to getValue
							outputArray[outOffset + w] = input.getValue(n, tempOffset + wPadded);
						}
						else
							outputArray[outOffset + w] = 0;
					}
				}
			}
		}
	}
	
	private static void doIm2col(int n, MatrixBlock output, ConvolutionParameters params) throws DMLRuntimeException {
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
			doIm2colSparse(n, params.input1, outputArray, params);
	}
	
	// ------------------------------------------------------------------------------------------------
	// Used in integration tests. Please donot edit them
	public static boolean TEST_SPARSE_INPUT = false;
	public static boolean TEST_SPARSE_FILTER = false;
	// ------------------------------------------------------------------------------------------------
}
