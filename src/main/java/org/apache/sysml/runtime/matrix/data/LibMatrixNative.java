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

import org.apache.sysml.hops.OptimizerUtils;
import org.apache.sysml.runtime.DMLRuntimeException;
import org.apache.sysml.utils.NativeHelper;
import org.apache.sysml.utils.Statistics;

public class LibMatrixNative {
	
	// We could encapsulate heuristics in this function
	// For now, we only consider matrix-vector operation to be memory bound
	private static boolean isMatMultMemoryBound(int m1Rlen, int m1Clen, int m2Clen) {
		return m1Rlen == 1 || m1Clen == 1 || m2Clen == 1;
	}

	/**
	 * Performs matrix multiplication using native library if BLAS is available or else falls back to
	 * Java BLAS.
	 * 
	 * @param m1 lhs matrix block
	 * @param m2 rhs matrix block
	 * @param ret output matrix block
	 * @param k number of threads
	 * @throws DMLRuntimeException if error occurs
	 */
	public static void matrixMult(MatrixBlock m1, MatrixBlock m2, MatrixBlock ret, int k) throws DMLRuntimeException {
		// Sanity check:
		k = k > 1 ? k : 1;
		
		// check inputs / outputs
		if (m1.isEmptyBlock(false) || m2.isEmptyBlock(false)) {
			ret.examSparsity(); // turn empty dense into sparse
			return;
		}
		if (NativeHelper.isNativeLibraryLoaded() && !isMatMultMemoryBound(m1.rlen, m1.clen, m2.clen) && !m1.isInSparseFormat() && !m2.isInSparseFormat()) {
			ret.sparse = false;
			ret.allocateDenseBlock();
			if (NativeHelper.matrixMultDenseDense(m1.denseBlock, m2.denseBlock, ret.denseBlock, m1.getNumRows(), m1.getNumColumns(), m2.getNumColumns(), k > 0 ? k : NativeHelper.getMaxNumThreads())) {
				Statistics.numNativeLibMatrixMultCalls.increment();
				ret.recomputeNonZeros();
				// post-processing (nnz maintained in parallel)
				ret.examSparsity();
				return;
			} else {
				// Else fall back to Java
				Statistics.incrementNativeFailuresCounter();
			}
		}
		if (k == 1)
			LibMatrixMult.matrixMult(m1, m2, ret);
		else
			LibMatrixMult.matrixMult(m1, m2, ret, k);
	}
	
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
		if(NativeHelper.isNativeLibraryLoaded() && !input.isInSparseFormat() && !filter.isInSparseFormat()) {
			setNumThreads(params);
			if(params.bias == null) {
				if(NativeHelper.conv2dDense(input.denseBlock, filter.denseBlock, outputBlock.denseBlock, params.N, params.C, params.H, params.W, 
						params.K, params.R, params.S, params.stride_h, params.stride_w, params.pad_h, params.pad_w, 
						params.P, params.Q, params.numThreads)) {
					Statistics.numNativeLibMatrixDNNCalls.increment();
					// post-processing: maintain nnz
					outputBlock.recomputeNonZeros();
					return;
				}
				else {
					// Fall back to Java when failures
					Statistics.incrementNativeFailuresCounter();
				}
			}
			else {
				if(params.bias.isInSparseFormat())
					params.bias.sparseToDense(); // Bias matrix is usually extremely small
				if(NativeHelper.conv2dBiasAddDense(input.denseBlock, params.bias.denseBlock, filter.denseBlock, outputBlock.denseBlock, 
						params.N, params.C, params.H, params.W, 
						params.K, params.R, params.S, params.stride_h, params.stride_w, params.pad_h, params.pad_w, 
						params.P, params.Q, params.numThreads)) {
					Statistics.numNativeLibMatrixDNNCalls.increment();
					// post-processing: maintain nnz
					outputBlock.recomputeNonZeros();
					return;
				}
				else {
					// Fall back to Java when failures
					Statistics.incrementNativeFailuresCounter();
				}
			}
		}
		
		// Fall back to Java when failures or sparse
		LibMatrixDNN.conv2d(input, filter, outputBlock, params);
	}
	
	private static void setNumThreads(ConvolutionParameters params) {
		params.numThreads = OptimizerUtils.getConstrainedNumThreads(params.numThreads);
		if (!(params.isOutputThreadSafe() && params.numThreads > 1))
			params.numThreads = 1;
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
		LibMatrixDNN.checkInputsConv2dBackwardFilter(input, dout, outputBlock, params);
		if(NativeHelper.isNativeLibraryLoaded() && !dout.isInSparseFormat() && !input.isInSparseFormat()) {
			setNumThreads(params);
			if(NativeHelper.conv2dBackwardFilterDense(input.denseBlock, dout.denseBlock, outputBlock.denseBlock, params.N, params.C, params.H, params.W, 
						params.K, params.R, params.S, params.stride_h, params.stride_w, params.pad_h, params.pad_w, 
						params.P, params.Q, params.numThreads)) {
				Statistics.numNativeLibMatrixDNNCalls.increment();
				// post-processing: maintain nnz
				outputBlock.recomputeNonZeros();
				return;
			}
			else {
				// Fall back to Java when failures
				Statistics.incrementNativeFailuresCounter();
			}
		}
		// Fall back to Java when failures or sparse
		LibMatrixDNN.conv2dBackwardFilter(input, dout, outputBlock, params);
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
		LibMatrixDNN.checkInputsConv2dBackwardData(filter, dout, outputBlock, params);
		if(NativeHelper.isNativeLibraryLoaded() && !dout.isInSparseFormat() && !filter.isInSparseFormat()) {
			setNumThreads(params);
			if(NativeHelper.conv2dBackwardDataDense(filter.denseBlock, dout.denseBlock, outputBlock.denseBlock, params.N, params.C, params.H, params.W, 
						params.K, params.R, params.S, params.stride_h, params.stride_w, params.pad_h, params.pad_w, 
						params.P, params.Q, params.numThreads)) {
				Statistics.numNativeLibMatrixDNNCalls.increment();
				// post-processing: maintain nnz
				outputBlock.recomputeNonZeros();
				return;
			}
			else {
				// Fall back to Java when failures
				Statistics.incrementNativeFailuresCounter();
			}
		}
		// Fall back to Java when failures or sparse
		LibMatrixDNN.conv2dBackwardData(filter, dout, outputBlock, params);
	}
}
