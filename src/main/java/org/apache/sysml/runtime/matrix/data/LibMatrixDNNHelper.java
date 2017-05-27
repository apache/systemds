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

import org.apache.sysml.api.DMLScript;
import org.apache.sysml.hops.OptimizerUtils;
import org.apache.sysml.runtime.DMLRuntimeException;
import org.apache.sysml.utils.NativeHelper;


public class LibMatrixDNNHelper {
	
	public static boolean isEligibleForConv2dSparse(ConvolutionParameters params) {
		// NativeHelper.conv2dSparse only if filter is dense and input is sparse
		return params.enableNative && params.input1.isInSparseFormat() && !params.input2.isInSparseFormat();
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
		int k = OptimizerUtils.getConstrainedNumThreads(params.numThreads);
		int taskSize = (int)(Math.ceil((double)params.N / k));
		
		for(int i = 0; i*taskSize < params.N; i++) {
			if(isEligibleForConv2dSparse(params)) 
				ret.add(new SparseNativeConv2d(i*taskSize, Math.min((i+1)*taskSize, params.N), params));
			else if(!params.input1.isInSparseFormat() && params.input1.denseBlock != null)
				ret.add(new LoopedIm2ColConv2dDenseInput(i*taskSize, Math.min((i+1)*taskSize, params.N), params));
			else if(params.input1.isInSparseFormat())
				ret.add(new LoopedIm2ColConv2dSparseInput(i*taskSize, Math.min((i+1)*taskSize, params.N), params));
			else
				throw new DMLRuntimeException("Unsupported operator");
		}
		return ret;
	}
	
	/**
	 * Performs convolution via: partialCopy1(filter %*% im2col(input)) => output
	 * Applicable when input is in dense format.
	 */
	public static class LoopedIm2ColConv2dDenseInput implements Callable<Long> 
	{
		public int _rl; public int _ru; 
		private final ConvolutionParameters _params;
		public LoopedIm2ColConv2dDenseInput(int rl, int ru, ConvolutionParameters params) {
			_rl = rl; _ru = ru;
			_params = params;
		}

		@Override
		public Long call() throws Exception {
			int PQ = _params.P*_params.Q; int K = _params.K; int CRS = _params.C*_params.R*_params.S;
			MatrixBlock im2ColOutBlock = new MatrixBlock(CRS, PQ, false);
			im2ColOutBlock.allocateDenseBlock();
			double [] inputArray = _params.input1.getDenseBlock();
			
			for(int n = _rl; n < _ru; n++)  {
				// im2col(input) => _im2ColOutBlock
				long t1 = DMLScript.STATISTICS && LibMatrixDNN.DISPLAY_STATISTICS ? System.nanoTime() : 0;
				doIm2colDense(n, inputArray, im2ColOutBlock.getDenseBlock(), _params);
				long t2 = DMLScript.STATISTICS && LibMatrixDNN.DISPLAY_STATISTICS ? System.nanoTime() : 0;
				
				// filter %*% _im2ColOutBlock => matMultOutBlock
				MatrixBlock matMultOutBlock = new MatrixBlock(K, PQ, false);
				singleThreadedMatMult(_params.input2, im2ColOutBlock, matMultOutBlock, false, true, _params);
				long t3 = DMLScript.STATISTICS && LibMatrixDNN.DISPLAY_STATISTICS ? System.nanoTime() : 0;
				
				if(DMLScript.STATISTICS && LibMatrixDNN.DISPLAY_STATISTICS) {
					LibMatrixDNN.loopedConvIm2ColTime.addAndGet(t2 - t1);
					LibMatrixDNN.loopedConvMatMultTime.addAndGet(t3 - t2);
				}
				// Copy the matrix matMultOutBlock of shape [K X PQ] to params.output.denseBlock + destPos
				partialCopy1(matMultOutBlock, _params.output.getDenseBlock(), n*K*PQ, K, PQ);
			}
			if(_params.bias != null) {
				// bias is always converted to dense format
				addBias(_rl, _ru, _params.output.getDenseBlock(), _params.bias.getDenseBlock(), K, PQ);
			}
			return 0L;
		}
	}
	
	/**
	 * Performs convolution via: partialCopy1(filter %*% im2col(input)) => output
	 * Applicable when input is in sparse format.
	 */
	public static class LoopedIm2ColConv2dSparseInput implements Callable<Long> 
	{
		public int _rl; public int _ru; 
		private final ConvolutionParameters _params;
		public LoopedIm2ColConv2dSparseInput(int rl, int ru, ConvolutionParameters params) {
			_rl = rl; _ru = ru;
			_params = params;
		}

		@Override
		public Long call() throws Exception {
			int PQ = _params.P*_params.Q; int K = _params.K; int CRS = _params.C*_params.R*_params.S;
			MatrixBlock im2ColOutBlock = new MatrixBlock(CRS, PQ, false);
			im2ColOutBlock.allocateDenseBlock();
			double [] temp = new double[_params.input1.getNumColumns()];
			for(int n = _rl; n < _ru; n++)  {
				// im2col(input) => _im2ColOutBlock
				long t1 = DMLScript.STATISTICS && LibMatrixDNN.DISPLAY_STATISTICS ? System.nanoTime() : 0;
				doIm2colSparse(n, _params.input1, im2ColOutBlock.getDenseBlock(), _params, temp);
				long t2 = DMLScript.STATISTICS && LibMatrixDNN.DISPLAY_STATISTICS ? System.nanoTime() : 0;
				
				// filter %*% _im2ColOutBlock => matMultOutBlock
				MatrixBlock matMultOutBlock = new MatrixBlock(K, PQ, false);
				singleThreadedMatMult(_params.input2, im2ColOutBlock, matMultOutBlock, false, true, _params);
				long t3 = DMLScript.STATISTICS && LibMatrixDNN.DISPLAY_STATISTICS ? System.nanoTime() : 0;
				
				if(DMLScript.STATISTICS && LibMatrixDNN.DISPLAY_STATISTICS) {
					LibMatrixDNN.loopedConvIm2ColTime.addAndGet(t2 - t1);
					LibMatrixDNN.loopedConvMatMultTime.addAndGet(t3 - t2);
				}
				// Copy the matrix matMultOutBlock of shape [K X PQ] to params.output.denseBlock + destPos
				partialCopy1(matMultOutBlock, _params.output.getDenseBlock(), n*K*PQ, K, PQ);
			}
			if(_params.bias != null) {
				// bias is always converted to dense format
				addBias(_rl, _ru, _params.output.getDenseBlock(), _params.bias.getDenseBlock(), K, PQ);
			}
			return 0L;
		}
	}
	
	/**
	 * This operator is used only if native is enabled, filter is dense and input is sparse
	 */
	public static class SparseNativeConv2d implements Callable<Long> 
	{
		public int _rl; public int _ru; 
		private final ConvolutionParameters _params;
		public SparseNativeConv2d(int rl, int ru, ConvolutionParameters params) {
			_rl = rl; _ru = ru;
			_params = params;
		}

		@Override
		public Long call() throws Exception {
			int KPQ = _params.K*_params.P*_params.Q;
			double[] temp = new double[KPQ];
			for(int n = _rl; n < _ru; n++)  {
				if( !_params.input1.getSparseBlock().isEmpty(n) ) {
					int apos = _params.input1.getSparseBlock().pos(n);
					int alen = _params.input1.getSparseBlock().size(n);
					int[] aix = _params.input1.getSparseBlock().indexes(n);
					double[] avals = _params.input1.getSparseBlock().values(n);
					NativeHelper.conv2dSparse(apos, alen, aix, avals, _params.input2.getDenseBlock(), temp, 
							1, _params.C, _params.H, _params.W, _params.K, _params.R, _params.S, 
							_params.stride_h, _params.stride_w, _params.pad_h, _params.pad_w, _params.P, _params.Q, 1);
					System.arraycopy(temp, 0, _params.output.denseBlock, n*KPQ, KPQ);
				}
			}
			return 0L;
		}
	}
	
	// Copy the matrix src of shape [K X PQ] to params.output.denseBlock + destPos
	private static void partialCopy1(MatrixBlock src, double [] dest, int destPos, int K, int PQ) {
		// Copying is required as LibMatrixMult.matrixMult (and/or Java) is not pointer aware.
		// This is not required in Native implementation
		if(!src.isEmptyBlock()) {
			if(src.isInSparseFormat()) {
				// Copy the sparse matrix matMultOutBlock of shape [K X PQ] to 
				// params.output.denseBlock + destPos
				for(int k = 0; k < src.getNumRows(); k++) {
					if( !src.sparseBlock.isEmpty(k) ) {
						int apos = src.sparseBlock.pos(k);
						int alen = src.sparseBlock.size(k);
						int[] aix = src.sparseBlock.indexes(k);
						double[] avals = src.sparseBlock.values(k);
						for(int j = apos; j < apos+alen; j++) {
							int pqIndex = aix[j];
							dest[destPos + k*PQ + pqIndex ] = avals[j];
						}
					}
				}
			}
			else 
				System.arraycopy(src.denseBlock, 0, dest, destPos, K * PQ);
		}
	}
	
	// Single-threaded matrix multiplication
	private static void singleThreadedMatMult(MatrixBlock m1, MatrixBlock m2, MatrixBlock ret, 
			boolean recomputeNNZM1, boolean recomputeNNZM2, ConvolutionParameters params) throws DMLRuntimeException {
		if(!params.enableNative || m1.isInSparseFormat() || m2.isInSparseFormat()) {
			if(recomputeNNZM1)
				m1.recomputeNonZeros();
			if(recomputeNNZM2)
				m2.recomputeNonZeros();
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
	
	private static void addBias(int _rl, int _ru, double [] outputArr, double [] biasArr, int K, int PQ) {
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
	
//Returns the row of matrix in dense format
	private static double [] getRowInDenseFormat(MatrixBlock input, int n, double []  temp) throws DMLRuntimeException {
		if(input.getNumColumns() != temp.length) {
			throw new DMLRuntimeException("Invalid parameters");
		}
		// Use temporary array to avoid binary search
		if(input.isInSparseFormat()) {
			Arrays.fill(temp, 0);
			if( !input.sparseBlock.isEmpty(n) ) {
				int apos = input.sparseBlock.pos(n);
				int alen = input.sparseBlock.size(n);
				int[] aix = input.sparseBlock.indexes(n);
				double[] avals = input.sparseBlock.values(n);
				for(int j=apos; j<apos+alen; j++)
					temp[ aix[j] ] = avals[j];
			}
		}
		else {
			System.arraycopy(input.getDenseBlock(), n*input.getNumColumns(), temp, 0, input.getNumColumns());
		}
		return temp;
	}
	
	// ------------------------------------------------------------
	// TODO:
	
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
}
