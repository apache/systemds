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
		private final int _rl, _ru; 
		private final ConvolutionParameters _params;
		
		public DenseMaxPooling(int rl, int ru, ConvolutionParameters params) {
			_rl = rl; _ru = ru;
			_params = params;
		}
		
		@Override
		public Long call() throws Exception {
			final int C = _params.C, P = _params.P, Q = _params.Q;
			final int R = _params.R, S = _params.S, H = _params.H, W = _params.W;
			final int HW = _params.H*_params.W;
			final int CHW = _params.C*_params.H*_params.W;
			final int CPQ = C*P*Q;
			double[] in = _params.input1.getDenseBlock();
			double[] out = _params.output.getDenseBlock();
			
			double minValForMaxPoolOperations = _params.minValForMaxPoolOperations;
			
			//thread-local initialization of output block 
			if( !(_params.isStride1Pad0() && _params.isAllOnes(P, Q, W)) )
				Arrays.fill(out, _rl*CPQ, _ru*CPQ, minValForMaxPoolOperations);
			
			if( _params.isStride1Pad0() && _params.isAllOnes(P, Q, W) ) { 
				//quick-path w/o materialized index arrays and 
				//simplified inner loops for P = 1, Q = 1, W = 1
				int lenh = Math.min(R,H);
				for(int i = _rl, oix=_rl*C; i < _ru; i++, oix+=C)
					for (int c = 0, off=i*CHW; c < C; c++, off+=H)
						out[oix+c] = max(minValForMaxPoolOperations, in, off, lenh);
			}
			else if( _params.isStride1Pad0() ) {
				//quick-path w/o materialized index arrays 
				for(int i = _rl; i < _ru; i++)
					for (int c = 0, off=i*CHW, oix=i*CPQ; c < C; c++, off+=HW)
						for (int p = 0; p < P; p++, oix+=Q)
							for (int h = p; h < Math.min(p+R,H); h++)
								for (int q = 0, off2=off+h*W; q < Q; q++)
									out[oix+q] = max(out[oix+q], in, off2+q, Math.min(S,W-q));
			}
			else { //general case
				int[] hl = _params.start_indexes_h, hu = _params.end_indexes_h;
				int[] wl = _params.start_indexes_w, wu = _params.end_indexes_w;
				for(int i = _rl; i < _ru; i++)
					for (int c = 0, off=i*CHW, oix=i*CPQ; c < C; c++, off+=HW)
						for (int p = 0; p < P; p++, oix+=Q)
							for (int h = hl[p]; h < hu[p]; h++)
								for (int q = 0, off2=off+h*W; q < Q; q++)
									out[oix+q] = max(out[oix+q], in, off2+wl[q], wu[q]-wl[q]);
			}
			
			//thread-local recomputation of non-zeros
			return _params.output.recomputeNonZeros(_rl, _ru-1);
		}
	}
	
	/**
	 * Performs the sparse maxpooling
	 */
	public static class SparseMaxPooling implements Callable<Long> 
	{
		private final int _rl, _ru; 
		private final ConvolutionParameters _params;
		private double [] outputArray;
		private final int C, P, Q, W, H, CPQ, PQ;
		
		public SparseMaxPooling(int rl, int ru, ConvolutionParameters params) {
			_rl = rl; _ru = ru;
			_params = params;
			outputArray = params.output.getDenseBlock();
			C = params.C; P = params.P; Q = params.Q; H = params.H; 
			W = params.W;
			CPQ = C*P*Q;
			PQ = P*Q;
		}
		
		@Override
		public Long call() throws Exception {
			//thread-local initialization of output block 
			Arrays.fill(outputArray, _rl *CPQ, _ru*CPQ, _params.minValForMaxPoolOperations);
			
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
			
			//thread-local recomputation of non-zeros
			return _params.output.recomputeNonZeros(_rl, _ru-1);
		}
	}
	
	private static double max(final double aval, double[] b, final int bi, final int len) {
		double ret = aval;
		for( int i = bi; i < bi+len; i++ )
			ret = Math.max(ret, b[i]);
		return ret;
	}
}
