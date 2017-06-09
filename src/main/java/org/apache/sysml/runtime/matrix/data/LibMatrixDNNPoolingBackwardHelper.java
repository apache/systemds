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

import java.util.concurrent.Callable;

/**
 * This class contains the set of operators used for performing pooling backward
 */
public class LibMatrixDNNPoolingBackwardHelper {
	/**
	 * Performs the maxpooling backward operation for dense input and dense error (dout)
	 */
	public static class PoolingBackwardDenseDense implements Callable<Long> 
	{
		public int _rl; public int _ru; 
		private final ConvolutionParameters _params; 
		double [] outputArray; boolean performReluBackward;
		double [] inputArray; double [] doutArray;
		int C; int CHW; int P; int Q; int HW; int CPQ; int PQ;
		public PoolingBackwardDenseDense(int rl, int ru, ConvolutionParameters params, boolean performReluBackward) {
			_rl = rl; _ru = ru;
			_params = params;
			this.performReluBackward = performReluBackward;
			inputArray = params.input1.getDenseBlock();
			doutArray = params.input2.getDenseBlock();
			outputArray = params.output.getDenseBlock();
			C = params.C; CHW = params.C*params.H*params.W; HW = params.H*params.W;
			P = params.P; Q = params.Q; CPQ = params.C*params.P*params.Q;
			PQ = params.P*params.Q;
			if (inputArray == null || doutArray == null || outputArray == null )
				throw new RuntimeException("Incorrect usage: empty inputs");
		}
		
		@Override
		public Long call() throws Exception {
			for(int n = _rl; n < _ru; n++)  {
				for (int c = 0; c < C; c++) {
					final int inputOffset = n*CHW + c*HW;
					final int outputOffset = n*CPQ + c*PQ;
					for (int p = 0; p < P; p++) {
						for (int q = 0; q < Q; q++) {
							int maxIndex = LibMatrixDNNHelper.getMaxIndex(p, q, inputOffset, inputArray, _params, performReluBackward);
							if(maxIndex != -1)
								outputArray[maxIndex] += doutArray[outputOffset +  p * Q + q];
						}
					}
				}
			}
			return 0L;
		}
	}
	
	/**
	 * Performs the maxpooling backward operation for dense input and sparse error (dout)
	 */
	public static class PoolingBackwardDenseSparse implements Callable<Long> 
	{
		public int _rl; public int _ru; 
		private final ConvolutionParameters _params; 
		double [] outputArray; boolean performReluBackward;
		double [] inputArray;  MatrixBlock dout;
		int C; int CHW; int P; int Q; int HW;
		public PoolingBackwardDenseSparse(int rl, int ru, ConvolutionParameters params, boolean performReluBackward) {
			_rl = rl; _ru = ru;
			_params = params;
			this.performReluBackward = performReluBackward;
			inputArray = params.input1.getDenseBlock();
			dout = params.input2;
			outputArray = params.output.getDenseBlock();
			C = params.C; CHW = params.C*params.H*params.W; HW = params.H*params.W;
			P = params.P; Q = params.Q; 
			if (inputArray == null || outputArray == null )
				throw new RuntimeException("Incorrect usage: empty inputs");
			if (!params.input2.isInSparseFormat())
				throw new RuntimeException("Incorrect usage: Call optimized versions");
		}
		
		@Override
		public Long call() throws Exception {
			for(int n = _rl; n < _ru; n++)  {
				if( !dout.sparseBlock.isEmpty(n) ) {
					int [] tensorIndexes = new int[3];
					int apos = dout.sparseBlock.pos(n);
					int alen = dout.sparseBlock.size(n);
					int[] aix = dout.sparseBlock.indexes(n);
					double[] avals = dout.sparseBlock.values(n);
					for(int j = apos; j < apos+alen; j++) {
						LibMatrixDNNHelper.computeTensorIndexes(aix[j], tensorIndexes, P, Q);
						int c = tensorIndexes[0];
						int p = tensorIndexes[1];
						int q = tensorIndexes[2];
						final int inputOffset = n*CHW + c*HW;
						int maxIndex = LibMatrixDNNHelper.getMaxIndex(p, q, inputOffset, inputArray, _params, performReluBackward);
						if(maxIndex != -1)
							outputArray[maxIndex] += avals[j];
					}
				}
			}
			return 0L;
		}
	}
	
	/**
	 * Performs the maxpooling backward operation for sparse input and dense error (dout)
	 */
	public static class PoolingBackwardSparseDense implements Callable<Long> 
	{
		public int _rl; public int _ru; 
		private final ConvolutionParameters _params; 
		double [] outputArray; boolean performReluBackward;
		double [] doutArray;
		int C; int CHW; int P; int Q; int HW; int CPQ; int PQ;
		public PoolingBackwardSparseDense(int rl, int ru, ConvolutionParameters params, boolean performReluBackward) {
			_rl = rl; _ru = ru;
			_params = params;
			this.performReluBackward = performReluBackward;
			doutArray = params.input2.getDenseBlock();
			outputArray = params.output.getDenseBlock();
			C = params.C; CHW = params.C*params.H*params.W; HW = params.H*params.W;
			P = params.P; Q = params.Q; CPQ = params.C*params.P*params.Q;
			PQ = params.P*params.Q;
			if (doutArray == null || outputArray == null )
				throw new RuntimeException("Incorrect usage: empty inputs");
			if (!params.input1.isInSparseFormat())
				throw new RuntimeException("Incorrect usage: Call optimized versions");
		}
		
		@Override
		public Long call() throws Exception {
			for(int n = _rl; n < _ru; n++)  {
				for (int c = 0; c < C; c++) {
					for (int p = 0; p < P; p++) {
						for (int q = 0; q < Q; q++) {
							double inVal = doutArray[n*CPQ + c*PQ +  p * Q + q];
							if(inVal != 0) {
								final int inputOffset = n*CHW + c*HW;
								int maxIndex = LibMatrixDNNHelper.getMaxIndexSparse(p, q, inputOffset, n, c, _params.input1, _params, performReluBackward);
								if(maxIndex != -1)
									outputArray[maxIndex] += inVal;
							}
						}
					}
				}
			}
			return 0L;
		}
	}
	
	/**
	 * Performs the maxpooling backward operation for sparse input and sparse error (dout)
	 */
	public static class PoolingBackwardSparseSparse implements Callable<Long> 
	{
		public int _rl; public int _ru; 
		private final ConvolutionParameters _params; 
		double [] outputArray; boolean performReluBackward;
		int C; int CHW; int P; int Q; int HW; 
		public PoolingBackwardSparseSparse(int rl, int ru, ConvolutionParameters params, boolean performReluBackward) {
			_rl = rl; _ru = ru;
			_params = params;
			this.performReluBackward = performReluBackward;
			outputArray = params.output.getDenseBlock();
			C = params.C; CHW = params.C*params.H*params.W; HW = params.H*params.W;
			P = params.P; Q = params.Q;
			if (outputArray == null )
				throw new RuntimeException("Incorrect usage: empty outputs");
			if (!params.input1.isInSparseFormat() || !params.input2.isInSparseFormat())
				throw new RuntimeException("Incorrect usage: Call optimized versions");
		}
		
		@Override
		public Long call() throws Exception {
			for(int n = _rl; n < _ru; n++)  {
				if( !_params.input2.sparseBlock.isEmpty(n) ) {
					int [] tensorIndexes = new int[3];
					int apos = _params.input2.sparseBlock.pos(n);
					int alen = _params.input2.sparseBlock.size(n);
					int[] aix = _params.input2.sparseBlock.indexes(n);
					double[] avals = _params.input2.sparseBlock.values(n);
					for(int j = apos; j < apos+alen; j++) {
						LibMatrixDNNHelper.computeTensorIndexes(aix[j], tensorIndexes, P, Q);
						int c = tensorIndexes[0];
						int p = tensorIndexes[1];
						int q = tensorIndexes[2];
						final int inputOffset = n*CHW + c*HW;
						int maxIndex = LibMatrixDNNHelper.getMaxIndexSparse(p, q, inputOffset, n, c, _params.input1, _params, performReluBackward);
						if(maxIndex != -1)
							outputArray[maxIndex] += avals[j];
					}
				}
			}
			return 0L;
		}
	}
}
