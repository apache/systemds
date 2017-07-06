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
import java.util.concurrent.Callable;

import org.apache.sysml.hops.OptimizerUtils;
import org.apache.sysml.runtime.DMLRuntimeException;
import org.apache.sysml.runtime.instructions.InstructionUtils;
import org.apache.sysml.runtime.util.ConvolutionUtils;
import org.apache.sysml.utils.NativeHelper;


public class LibMatrixDNNHelper {
	
	// *********************************** low-level runtime operator selection ***********************************************
	// *********************************** based on runtime properties (sparsity, native, etc) ********************************
	// These methods help reduce branch miss predictions and instruction-cache misses.
	// Also, they simplify the design of LibMatrixDNN and help in code-maintenance.
	
	/**
	 * Factory method that returns list of callable tasks for performing maxpooling operation
	 * 
	 * @param params convolution parameters
	 * @return list of callable tasks for performing maxpooling operation
	 * @throws DMLRuntimeException if error occurs
	 */
	public static ArrayList<Callable<Long>> getMaxPoolingWorkers(ConvolutionParameters params) throws DMLRuntimeException {
		ArrayList<Callable<Long>> ret = new ArrayList<Callable<Long>>();
		int k = OptimizerUtils.getConstrainedNumThreads(params.numThreads);
		int taskSize = (int)(Math.ceil((double)params.N / k));
		for(int i = 0; i*taskSize < params.N; i++) {
			if(params.input1.isInSparseFormat())
				ret.add(new LibMatrixDNNPoolingHelper.SparseMaxPooling(i*taskSize, Math.min((i+1)*taskSize, params.N), params));
			else
				ret.add(new LibMatrixDNNPoolingHelper.DenseMaxPooling(i*taskSize, Math.min((i+1)*taskSize, params.N), params));
		}
		return ret;
	}
	
	/**
	 * Factory method that returns list of callable tasks for performing maxpooling backward operation
	 * 
	 * @param params convolution parameters
	 * @param performReluBackward whether to perform ReLU backward
	 * @return list of callable tasks for performing maxpooling backward operation
	 * @throws DMLRuntimeException if error occurs
	 */
	public static ArrayList<Callable<Long>> getMaxPoolingBackwardWorkers(ConvolutionParameters params, boolean performReluBackward) throws DMLRuntimeException {
		ArrayList<Callable<Long>> ret = new ArrayList<Callable<Long>>();
		int k = OptimizerUtils.getConstrainedNumThreads(params.numThreads);
		int taskSize = (int)(Math.ceil((double)params.N / k));
		for(int i = 0; i*taskSize < params.N; i++) {
			if(!params.input1.isInSparseFormat()) {
				if(!params.input2.isInSparseFormat()) 
					ret.add(new LibMatrixDNNPoolingBackwardHelper.PoolingBackwardDenseDense(i*taskSize, Math.min((i+1)*taskSize, params.N), params, performReluBackward));
				else
					ret.add(new LibMatrixDNNPoolingBackwardHelper.PoolingBackwardDenseSparse(i*taskSize, Math.min((i+1)*taskSize, params.N), params, performReluBackward));
			}
			else {
				if(!params.input2.isInSparseFormat()) 
					ret.add(new LibMatrixDNNPoolingBackwardHelper.PoolingBackwardSparseDense(i*taskSize, Math.min((i+1)*taskSize, params.N), params, performReluBackward));
				else
					ret.add(new LibMatrixDNNPoolingBackwardHelper.PoolingBackwardSparseSparse(i*taskSize, Math.min((i+1)*taskSize, params.N), params, performReluBackward));
			}
		}
		return ret;
	}
	
	/**
	 * Factory method that returns list of callable tasks for performing relu backward operation
	 * 
	 * @param params convolution parameters
	 * @return list of callable tasks for performing relu backward operation
	 * @throws DMLRuntimeException if error occurs
	 */
	public static ArrayList<Callable<Long>> getReluBackwardWorkers(ConvolutionParameters params) throws DMLRuntimeException {
		ArrayList<Callable<Long>> ret = new ArrayList<Callable<Long>>();
		int k = OptimizerUtils.getConstrainedNumThreads(params.numThreads);
		int taskSize = (int)(Math.ceil((double)params.N / k));
		for(int i = 0; i*taskSize < params.N; i++) {
			ret.add(new ReluBackward(i*taskSize, Math.min((i+1)*taskSize, params.N), params));
		}
		return ret;
	}
	
	/**
	 * Factory method that returns list of callable tasks for performing conv2d
	 * 
	 * @param params convolution parameters
	 * @return list of callable tasks for performing conv2d
	 * @throws DMLRuntimeException if error occurs
	 */
	public static ArrayList<Callable<Long>> getConv2dWorkers(ConvolutionParameters params) throws DMLRuntimeException {
		ArrayList<Callable<Long>> ret = new ArrayList<Callable<Long>>();
		
		// Try to create as many tasks as threads. 
		// Creating more tasks will help in tail, but would have additional overhead of maintaining the intermediate
		// data structures such as im2col blocks.
		int k = OptimizerUtils.getConstrainedNumThreads(params.numThreads);
		int taskSize = (int)(Math.ceil((double)params.N / k));
		
		// TODO: Decide here based on params whether to use LoopedIm2ColConv2dAllChannels or LoopedIm2ColConv2dOneChannel
		// For now, let's stick to the existing approach of converting [1, CHW] to [CRS, PQ] as it allows matrix multiplication large enough matrix.
		boolean allChannels = true; ArrayList<MatrixBlock> filters = null;
		if(!allChannels) {
			filters = splitFilter(params);
		}
		
		boolean isEmptyDenseInput = !params.input1.isInSparseFormat() && params.input1.denseBlock == null;
		
		for(int i = 0; i*taskSize < params.N; i++) {
			if(LibMatrixDNN.isEligibleForConv2dSparse(params)) 
				ret.add(new LibMatrixDNNConv2dHelper.SparseNativeConv2d(i*taskSize, Math.min((i+1)*taskSize, params.N), params));
			else if(!isEmptyDenseInput && allChannels)
				ret.add(new LibMatrixDNNConv2dHelper.LoopedIm2ColConv2dAllChannels(i*taskSize, Math.min((i+1)*taskSize, params.N), params));
			else if(!isEmptyDenseInput && !allChannels)
				ret.add(new LibMatrixDNNConv2dHelper.LoopedIm2ColConv2dOneChannel(i*taskSize, Math.min((i+1)*taskSize, params.N), params, filters));
			else
				throw new DMLRuntimeException("Unsupported operator");
		}
		return ret;
	}
	
	/**
	 * Factory method that returns list of callable tasks for performing conv2d backward filter
	 * 
	 * @param params convolution parameters
	 * @return list of callable tasks for performing conv2d backward filter
	 * @throws DMLRuntimeException if error occurs
	 */
	public static ArrayList<Callable<Long>> getConv2dBackwardFilterWorkers(ConvolutionParameters params) throws DMLRuntimeException {
		ArrayList<Callable<Long>> ret = new ArrayList<Callable<Long>>();
		// Try to create as many tasks as threads. 
		// Creating more tasks will help in tail, but would have additional overhead of maintaining the intermediate
		// data structures such as im2col blocks.
		int k = OptimizerUtils.getConstrainedNumThreads(params.numThreads);
		int taskSize = (int)(Math.ceil((double)params.N / k));
		
		boolean isEmptyDenseInput = (!params.input1.isInSparseFormat() && params.input1.denseBlock == null) || 
																(!params.input2.isInSparseFormat() && params.input2.denseBlock == null);
		
		for(int i = 0; i*taskSize < params.N; i++) {
			if(LibMatrixDNN.isEligibleForConv2dBackwardFilterSparseDense(params)) 
				ret.add(new LibMatrixDNNConv2dBackwardFilterHelper.SparseNativeConv2dBackwardFilterDense(i*taskSize, Math.min((i+1)*taskSize, params.N), params));
			else if(!isEmptyDenseInput)
				ret.add(new LibMatrixDNNConv2dBackwardFilterHelper.Conv2dBackwardFilter(i*taskSize, Math.min((i+1)*taskSize, params.N), params));
			else
				throw new DMLRuntimeException("Unsupported operator");
		}
		return ret;
	}
	
	/**
	 * Factory method that returns list of callable tasks for performing conv2d backward data
	 * 
	 * @param params convolution parameters
	 * @return list of callable tasks for performing conv2d backward data
	 * @throws DMLRuntimeException if error occurs
	 */
	public static ArrayList<Callable<Long>> getConv2dBackwardDataWorkers(ConvolutionParameters params) throws DMLRuntimeException {
		ArrayList<Callable<Long>> ret = new ArrayList<Callable<Long>>();
		
		// Try to create as many tasks as threads. 
		// Creating more tasks will help in tail, but would have additional overhead of maintaining the intermediate
		// data structures such as im2col blocks.
		int k = OptimizerUtils.getConstrainedNumThreads(params.numThreads);
		int taskSize = (int)(Math.ceil((double)params.N / k));
		
		boolean isEmptyDenseInput = (!params.input1.isInSparseFormat() && params.input1.denseBlock == null) || 
																(!params.input2.isInSparseFormat() && params.input2.denseBlock == null);
		
		for(int i = 0; i*taskSize < params.N; i++) {
			if(LibMatrixDNN.isEligibleForConv2dBackwardDataDense(params)) 
				ret.add(new LibMatrixDNNConv2dBackwardDataHelper.SparseNativeConv2dBackwardDataDense(i*taskSize, Math.min((i+1)*taskSize, params.N), params));
			else if(!isEmptyDenseInput)
				ret.add(new LibMatrixDNNConv2dBackwardDataHelper.Conv2dBackwardData(i*taskSize, Math.min((i+1)*taskSize, params.N), params));
			else
				throw new DMLRuntimeException("Unsupported operator");
		}
			
		return ret;
	}
	
	// *********************************** relu backward operator ******************************************************
	
	/**
	 * Performs the operation: (X gt 0) * dout
	 */
	public static class ReluBackward implements Callable<Long> 
	{
		public int _rl; public int _ru; 
		private final ConvolutionParameters _params; 
		double [] outputArray; int numOutCols;
		public ReluBackward(int rl, int ru, ConvolutionParameters params) {
			_rl = rl; _ru = ru;
			_params = params;
			outputArray= params.output.getDenseBlock();
			numOutCols = params.input1.getNumColumns();
		}
		
		@Override
		public Long call() throws Exception {
			if(!_params.input1.isInSparseFormat() && !_params.input2.isInSparseFormat()) {
				double [] inputArr = _params.input1.getDenseBlock();
				double [] doutArr = _params.input2.getDenseBlock();
				for(int i = _rl*numOutCols; i < _ru*numOutCols; i++) {
					outputArray[i] = inputArr[i] > 0 ? doutArr[i] : 0;
				}
			}
			else {
				// Perform (X > 0)
				ConvolutionUtils.scalarOperations(_params.input1, outputArray, _rl*numOutCols, numOutCols, _rl, _ru, 
						InstructionUtils.parseScalarBinaryOperator(">", false, 0));
				// Then perform (X > 0) * dout
				ConvolutionUtils.binaryOperationInPlace(_params.input2, outputArray, _rl*numOutCols, numOutCols, _rl, _ru, 
						LibMatrixDNN._binaryElementWiseMultiplication);
			}
			return 0L;
		}
	}
	
	// *********************************** utility methods ******************************************************
	
	/**
	 * Computes tensor indexes from column index such that column index  is equal to ret[0]*HW + ret[1]*W + ret[2]
	 * 
	 * @param j column index
	 * @param ret tensor indexes
	 * @param H second last dimension
	 * @param W last dimension
	 */
	static void computeTensorIndexes(int j, int [] ret, int H, int W) {
		ret[0] = j / (H*W);
		ret[1] = (j - ret[0]*(H*W))/W;
		ret[2] = j % W;
	}
	
	//Split a filter of size [K, CRS] into c filters of [K, RS]
	private static ArrayList<MatrixBlock> splitFilter(ConvolutionParameters _params) {
		ArrayList<MatrixBlock> ret = new ArrayList<MatrixBlock>();
		int RS = _params.R*_params.S; int CRS = _params.C*_params.R*_params.S;
		double [] filter = _params.input2.getDenseBlock(); int S = _params.S;
		for(int c = 0; c < _params.C; c++) {
			MatrixBlock mb = new MatrixBlock(_params.K, RS, false);
			mb.allocateDenseBlock(); long nnz = 0;
			double [] outputArr = mb.getDenseBlock();
			if(filter != null) {
				for(int k = 0; k < _params.K; k++) {
					for(int rs = 0; rs < RS; rs++) {
						outputArr[k*RS + rs] = filter[k*CRS + c*RS + rs];
						nnz += outputArr[k*RS + rs] != 0 ? 1 : 0;
					}
				}
			}
			else {
				for(int k = 0; k < _params.K; k++) {
					if( !_params.input2.sparseBlock.isEmpty(k) ) {
						int [] tensorIndexes = new int[3];
						// Find maxIndex
						int apos = _params.input2.sparseBlock.pos(k);
						int alen = _params.input2.sparseBlock.size(k);
						int[] aix = _params.input2.sparseBlock.indexes(k);
						double[] avals = _params.input2.sparseBlock.values(k);
						for(int j=apos; j<apos+alen; j++) {
							computeTensorIndexes(aix[j], tensorIndexes, _params.R, _params.S);
							if(c != tensorIndexes[0])
								continue;
							int r = tensorIndexes[1];
							int s = tensorIndexes[2];
							outputArr[k*RS + r*S + s] = avals[j];
							nnz += outputArr[k*RS + r*S + s] != 0 ? 1 : 0;
						}
					}
				}
			}
			mb.setNonZeros(nnz);
			ret.add(mb);
		}
		return ret;
	}
	
	// Single-threaded matrix multiplication
	static void singleThreadedMatMult(MatrixBlock m1, MatrixBlock m2, MatrixBlock ret, 
			boolean recomputeNNZM1, boolean recomputeNNZM2, ConvolutionParameters params) throws DMLRuntimeException {
		if(!params.enableNative || m1.isInSparseFormat() || m2.isInSparseFormat()) {
			prepNonZerosForMatrixMult(m1, recomputeNNZM1);
			prepNonZerosForMatrixMult(m2, recomputeNNZM2);
			LibMatrixMult.matrixMult(m1, m2, ret, false);
		}
		else {
			ret.sparse = false;
			if(ret.getDenseBlock() == null)
				ret.allocateDenseBlock();
			NativeHelper.matrixMultDenseDense(m1.denseBlock, m2.denseBlock, 
					ret.denseBlock, m1.getNumRows(), m1.getNumColumns(), m2.getNumColumns(), 1);
			ret.recomputeNonZeros();
		}
	}
	
	static void addBias(int _rl, int _ru, double [] outputArr, double [] biasArr, int K, int PQ) {
		// double [] biasArr = _params.bias.getDenseBlock();
		
		int index = _rl*K*PQ;
		for(int n = _rl; n < _ru; n++) {
			for(int k = 0; k < K; k++) {
				for(int pq = 0; pq < PQ; pq++, index++) {
					outputArr[index] += biasArr[k];
				}
			}
		}
	}
	
	/**
	 * Returns the index of cell with maximum value. This method is optimized for dense input
	 * 
	 * @param p output feature map height
	 * @param q output feature map width
	 * @param inputOffset offset to be used for input index
	 * @param inputArray input array
	 * @param params convolution parameters
	 * @param performReluBackward perform ReLU backward
	 * @return index of cell with maximum value
	 */
	static int getMaxIndex(int p, int q, int inputOffset, double [] inputArray, ConvolutionParameters params, boolean performReluBackward) {
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
				currDoutVal = performReluBackward && currDoutVal < 0 ? 0 : currDoutVal;
				if(maxVal < currDoutVal) {
					maxIndex = inputOffset +  h*params.W + w;
					maxVal = currDoutVal;
				}
			}
		}
		return maxIndex;
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
	 * @param performReluBackward perform ReLU on input
	 * @return index of the cell with maximum value
	 * @throws DMLRuntimeException if error occurs
	 */
	static int getMaxIndexSparse(int p, int q, int inputOffset, int n, int c, MatrixBlock input, ConvolutionParameters params, boolean performReluBackward) throws DMLRuntimeException {
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
					double val = performReluBackward && avals[j] < 0 ? 0 : avals[j]; 
					if(maxVal < val) {
						maxIndex = inputOffset +  h*params.W + w;
						maxVal = val;
					}
				}
			}
		}
		else {
			maxIndex = inputOffset;
		}
		return maxIndex;
	}
	
	// Returns the row of matrix in dense format
	static void getRowInDenseFormat(MatrixBlock input, int n, double []  ret) throws DMLRuntimeException {
		if(input.getNumColumns() != ret.length) {
			throw new DMLRuntimeException("Invalid parameters");
		}
		// Use temporary array to avoid binary search
		if(input.isInSparseFormat()) {
			Arrays.fill(ret, 0);
			if( !input.sparseBlock.isEmpty(n) ) {
				int apos = input.sparseBlock.pos(n);
				int alen = input.sparseBlock.size(n);
				int[] aix = input.sparseBlock.indexes(n);
				double[] avals = input.sparseBlock.values(n);
				for(int j=apos; j<apos+alen; j++)
					ret[ aix[j] ] = avals[j];
			}
		}
		else {
			System.arraycopy(input.getDenseBlock(), n*input.getNumColumns(), ret, 0, input.getNumColumns());
		}
	}
	
	// ------------------------------------------------------------------------------------------------------
	// Since col2im always operates on intermediate generated as part of matmult, it is not clear which operator to select apriori.
	// Therefore, it is provided as utility function rather than an operator (like im2col or rotate180)
	
	//Converts input: PQ X CRS matrix and writes to 1 X CHW
	static void doCol2imOverSingleImage(int outputN, MatrixBlock input, ConvolutionParameters params) throws DMLRuntimeException {
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
	
	private static void prepNonZerosForMatrixMult(MatrixBlock mb, boolean update) {
		if( !update )
			return;
		//non-zeros are not evaluated for dense matrix multiplies
		//so we simply need to ensure the block is not marked empty 
		if( !mb.isInSparseFormat() )
			mb.setNonZeros(mb.getNumRows() * mb.getNumColumns());
		else
			mb.recomputeNonZeros();	
	}
}
