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
		
		// Try to create as many tasks as threads. 
		// Creating more tasks will help in tail, but would have additional overhead of maintaining the intermediate
		// data structures such as im2col blocks.
		int k = OptimizerUtils.getConstrainedNumThreads(params.numThreads);
		int taskSize = (int)(Math.ceil((double)params.N / k));
		
		for(int i = 0; i*taskSize < params.N; i++) {
			if(isEligibleForConv2dSparse(params)) 
				ret.add(new SparseNativeConv2d(i*taskSize, Math.min((i+1)*taskSize, params.N), params));
			else if(params.input1.denseBlock != null)
				ret.add(new LoopedIm2ColConv2d(i*taskSize, Math.min((i+1)*taskSize, params.N), params));
			else
				throw new DMLRuntimeException("Unsupported operator");
		}
		return ret;
	}
	
	/**
	 * Performs convolution via: partialCopy1(filter %*% im2col(input)) => output
	 */
	public static class LoopedIm2ColConv2d implements Callable<Long> 
	{
		public int _rl; public int _ru; 
		private final ConvolutionParameters _params;
		public LoopedIm2ColConv2d(int rl, int ru, ConvolutionParameters params) {
			_rl = rl; _ru = ru;
			_params = params;
		}

		@Override
		public Long call() throws Exception {
			int PQ = _params.P*_params.Q; int K = _params.K; int CRS = _params.C*_params.R*_params.S;
			MatrixBlock im2ColOutBlock = new MatrixBlock(CRS, PQ, false);
			im2ColOutBlock.allocateDenseBlock();
			LibMatrixDNNIm2ColHelper.Im2colWorker im2ColWorker = LibMatrixDNNIm2ColHelper.Im2colWorker.getWorker( _params.input1, im2ColOutBlock, _params);
			for(int n = _rl; n < _ru; n++)  {
				// im2col(input) => _im2ColOutBlock
				long t1 = DMLScript.STATISTICS && LibMatrixDNN.DISPLAY_STATISTICS ? System.nanoTime() : 0;
				im2ColWorker.execute(n);
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
		
		// Copy the matrix src of shape [K X PQ] to params.output.denseBlock + destPos
		private void partialCopy1(MatrixBlock src, double [] dest, int destPos, int K, int PQ) {
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
	
}
