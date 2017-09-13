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

import java.util.Arrays;
import java.util.concurrent.Callable;

/**
 * This class contains the set of operators used for performing pooling
 */
public class LibMatrixDNNPoolingHelper {
	
	/**
	 * Performs the dense maxpooling
	 */
	public static class DenseMaxPooling implements Callable<Long> 
	{
		public int _rl; public int _ru; 
		private final ConvolutionParameters _params;
		double [] inputArray; double [] outputArray;
		int C; int P; int Q; int W;
		public DenseMaxPooling(int rl, int ru, ConvolutionParameters params) {
			_rl = rl; _ru = ru;
			_params = params;
			inputArray = params.input1.getDenseBlock();
			outputArray = params.output.getDenseBlock();
			C = params.C; P = params.P; Q = params.Q; W = params.W;
		}
		
		@Override
		public Long call() throws Exception {
			final int HW = _params.H*_params.W;
			final int CHW = _params.C*_params.H*_params.W;
			final int CPQ = C*P*Q;
			for(int n = _rl; n < _ru; n++)  {
				final int inOffset = n*CHW;
				int out_index = n*CPQ;
				for (int c = 0; c < C; c++) {
					final int inOffset1 = inOffset + c*HW;
					for (int p = 0; p < P; p++) {
						for (int q = 0; q < Q; q++, out_index++) {
							double tmp = outputArray[out_index];
							for (int h = _params.start_indexes_h[p]; h < _params.end_indexes_h[p]; h++) {
								int inputIndex = inOffset1 +  h*W + _params.start_indexes_w[q];
								for (int w = _params.start_indexes_w[q]; w < _params.end_indexes_w[q]; w++, inputIndex++) {
									tmp = Math.max(tmp, inputArray[inputIndex]);
								}
							}
							outputArray[out_index] = tmp;
						}
					}
				}
			}
			return 0L;
		}
	}
	
	/**
	 * Performs the sparse maxpooling
	 */
	public static class SparseMaxPooling implements Callable<Long> 
	{
		public int _rl; public int _ru; 
		private final ConvolutionParameters _params;
		final int HW;
		double [] outputArray;
		final int C; final int P; final int Q; final int W; final int H; final int CPQ; final int PQ;
		public SparseMaxPooling(int rl, int ru, ConvolutionParameters params) {
			_rl = rl; _ru = ru;
			_params = params;
			outputArray = params.output.getDenseBlock();
			C = params.C; P = params.P; Q = params.Q; H = params.H; 
			W = params.W;
			HW = _params.H*_params.W;
			CPQ = C*P*Q;
			PQ = P*Q;
		}
		
		@Override
		public Long call() throws Exception {
			for(int n = _rl; n < _ru; n++)  {
				if( !_params.input1.sparseBlock.isEmpty(n) ) {
					final int apos = _params.input1.sparseBlock.pos(n);
					final int alen = _params.input1.sparseBlock.size(n);
					final int [] aix = _params.input1.sparseBlock.indexes(n);
					final double [] avals = _params.input1.sparseBlock.values(n);
					int chw = 0; int index = apos;
					for (int c = 0; c < C; c++) {
						final int outOffset = n*CPQ + c*PQ;
						for(int h = 0; h < H; h++) {
							for(int w = 0; w < W; w++, chw++) {
								// Take into account zero values as well
								double nchwVal = 0;
								if(aix[index] == chw) {
									nchwVal = avals[index++];
									// Ensure that we satisfy the condition index < apos+alen
									if(index >= apos+alen) index--;
								}
								// Perform maxpooling without binary search :)
								// Tradeoff as compared to dense maxpooling: 
								// In dense maxpooling, iteration space CPQHW where H and W iterations are restricted by _params.start_indexes_h[p] 
								// and are eligible for JIT optimizations.
								// In sparse maxpooling, iteration space CHWPQ without HW restrictions.
								for (int p = 0; p < P; p++) {
									if(h >= _params.start_indexes_h[p] && h < _params.end_indexes_h[p]) {
										final int outOffsetWithp = outOffset + p*Q;
										for (int q = 0; q < Q; q++) {
											if(w >= _params.start_indexes_w[q] && w < _params.end_indexes_w[q]) {
												outputArray[outOffsetWithp + q] = Math.max(outputArray[outOffsetWithp + q], nchwVal);
											}
										}
									}
								}
							}
						}
					}
				}
				else {
					// Empty input image
					Arrays.fill(outputArray, n*CPQ, (n+1)*CPQ, 0);
				}
			}
			return 0L;
		}
	}
}
