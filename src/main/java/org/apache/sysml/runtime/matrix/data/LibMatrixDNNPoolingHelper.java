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
							for (int h = _params.start_indexes_h[p]; h < _params.end_indexes_h[p]; h++) {
								for (int w = _params.start_indexes_w[q]; w < _params.end_indexes_w[q]; w++) {
									outputArray[out_index] = Math.max(outputArray[out_index], inputArray[inOffset1 +  h*W + w]);
								}
							}
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
		int HW;
		double [] outputArray;
		int C; int P; int Q; int W;
		public SparseMaxPooling(int rl, int ru, ConvolutionParameters params) {
			_rl = rl; _ru = ru;
			_params = params;
			outputArray = params.output.getDenseBlock();
			C = params.C; P = params.P; Q = params.Q; W = params.W;
			HW = _params.H*_params.W;
		}
		
		boolean isNthRowEmpty = false;
		int apos; int alen; int[] aix; double[] avals;
		private void getNthSparseRow(int n) {
			if( !_params.input1.sparseBlock.isEmpty(n) ) {
				apos = _params.input1.sparseBlock.pos(n);
				alen = _params.input1.sparseBlock.size(n);
				aix = _params.input1.sparseBlock.indexes(n);
				avals = _params.input1.sparseBlock.values(n);
				isNthRowEmpty = false;
			}
			else
				isNthRowEmpty = true;
		}
		int fromIndex = -1; // as per C
		int toIndex = -1; // as per C
		private int setSearchIndex(int from, int searchVal) {
			for(int j = from; j < apos+alen; j++) {
				if(aix[j] > searchVal)
					return Math.max(from, j-1);
			}
			return apos+alen;
		}
		private double getValue(int col) {
			if( !isNthRowEmpty ) {
				int index = Arrays.binarySearch(aix, fromIndex, toIndex, col);
				return index > 0 ? avals[index] : 0;
			}
			return 0;
		}
		
		@Override
		public Long call() throws Exception {
			final int CPQ = C*P*Q;
			for(int n = _rl; n < _ru; n++)  {
				getNthSparseRow(n);
				int out_index = n*CPQ;
				for (int c = 0; c < C; c++) {
					// This allows for binary search in getValue to be more efficient
					fromIndex = setSearchIndex(apos, c*HW);
					toIndex = Math.min(apos+alen, setSearchIndex(fromIndex, (c+1)*HW));
					for (int p = 0; p < P; p++) {
						for (int q = 0; q < Q; q++, out_index++) {
							for (int h = _params.start_indexes_h[p]; h < _params.end_indexes_h[p]; h++) {
								for (int w = _params.start_indexes_w[q]; w < _params.end_indexes_w[q]; w++) {
									outputArray[out_index] = Math.max(outputArray[out_index], getValue(c*HW +  h*W + w));
								}
							}
						}
					}
				}
			}
			return 0L;
		}
	}
}
