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
		boolean performReluBackward;
		double [] inputArray, doutArray;
		MatrixBlock output;
		int C; int CHW; int P; int Q; int HW; int CPQ; int PQ;
		public PoolingBackwardDenseDense(int rl, int ru, ConvolutionParameters params, boolean performReluBackward) {
			_rl = rl; _ru = ru;
			_params = params;
			this.performReluBackward = performReluBackward;
			inputArray = params.input1.getDenseBlock();
			doutArray = params.input2.getDenseBlock();
			output = params.output;
			C = params.C; CHW = params.C*params.H*params.W; HW = params.H*params.W;
			P = params.P; Q = params.Q; CPQ = params.C*params.P*params.Q;
			PQ = params.P*params.Q;
			if (inputArray == null || doutArray == null || output.getDenseBlock() == null )
				throw new RuntimeException("Incorrect usage: empty inputs");
		}
		
		@Override
		public Long call() throws Exception {
			double[] out = output.getDenseBlock();
			for(int n = _rl; n < _ru; n++)  {
				for (int c = 0; c < C; c++) {
					final int inputOffset = n*CHW + c*HW;
					final int outputOffset = n*CPQ + c*PQ;
					for (int p = 0; p < P; p++) {
						for (int q = 0; q < Q; q++) {
							int maxIndex = LibMatrixDNNHelper.getMaxIndex(p, q, inputOffset, inputArray, _params, performReluBackward);
							if(maxIndex != -1)
								out[maxIndex] += doutArray[outputOffset +  p * Q + q];
						}
					}
				}
			}
			//thread-local nnz maintenance
			return output.recomputeNonZeros(_rl, _ru-1);
		}
	}
	
	/**
	 * Performs the maxpooling backward operation for dense input and sparse error (dout)
	 */
	public static class PoolingBackwardDenseSparse implements Callable<Long> 
	{
		public int _rl; public int _ru; 
		private final ConvolutionParameters _params; 
		MatrixBlock output; 
		boolean performReluBackward;
		double [] inputArray;  MatrixBlock dout;
		int C; int CHW; int P; int Q; int HW;
		public PoolingBackwardDenseSparse(int rl, int ru, ConvolutionParameters params, boolean performReluBackward) {
			_rl = rl; _ru = ru;
			_params = params;
			this.performReluBackward = performReluBackward;
			inputArray = params.input1.getDenseBlock();
			dout = params.input2;
			output = params.output;
			C = params.C; CHW = params.C*params.H*params.W; HW = params.H*params.W;
			P = params.P; Q = params.Q; 
			if (inputArray == null || output.getDenseBlock() == null )
				throw new RuntimeException("Incorrect usage: empty inputs");
			if (!params.input2.isInSparseFormat())
				throw new RuntimeException("Incorrect usage: Call optimized versions");
		}
		
		@Override
		public Long call() throws Exception {
			double[] out = output.getDenseBlock();
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
							out[maxIndex] += avals[j];
					}
				}
			}
			//thread-local nnz maintenance
			return output.recomputeNonZeros(_rl, _ru-1);
		}
	}
	
	/**
	 * Performs the maxpooling backward operation for sparse input and dense error (dout)
	 */
	public static class PoolingBackwardSparseDense implements Callable<Long> 
	{
		private final int _rl, _ru; 
		private final ConvolutionParameters _params; 
		private final boolean reluBack;
		protected final MatrixBlock doutput, output;
		
		protected PoolingBackwardSparseDense(int rl, int ru, ConvolutionParameters params, boolean relu, MatrixBlock dout, MatrixBlock out) {
			_rl = rl; _ru = ru; 
			_params = params;
			reluBack = relu;
			doutput = dout;
			output = out;
		}
		
		public PoolingBackwardSparseDense(int rl, int ru, ConvolutionParameters params, boolean relu) {
			this(rl, ru, params, relu, params.input2, params.output);
			if (doutput.getDenseBlock() == null || output.getDenseBlock() == null )
				throw new RuntimeException("Incorrect usage: empty inputs");
			if (!params.input1.isInSparseFormat())
				throw new RuntimeException("Incorrect usage: sparse input1 expected");
		}
		
		@Override
		public Long call() throws Exception 
		{
			final int P = _params.P, Q = _params.Q, W = _params.W;
			final int C = _params.C, R = _params.R, S = _params.S;
			final int padh = _params.pad_h, padw = _params.pad_w;
			final int strideh = _params.stride_h, stridew = _params.stride_w;
			final int PQ = _params.P * _params.Q;
			final int CPQ = _params.C * _params.P * _params.Q;
			final int HW = _params.H * _params.W;
			final int CHW = _params.C * _params.H * _params.W;
			
			//allocate auxiliary data structures
			double[] maxVal = new double[PQ];
			int[] maxIx = new int[PQ];
			
			for(int n = _rl; n < _ru; n++)  {
				for (int c = 0; c < C; c++) {
					//step 0: basic initializations
					final int outOffset = n*CHW + c*HW;
					
					//step 1: perform maxpooling w/ index maintenance in a 
					//single, sequential pass over the sparse input matrix
					maxpoolingForward(maxVal, maxIx, n, c,
						padh, padw, strideh, stridew, C, P, Q, R, S, HW, W);
					
					//step 2: perform maxpooling backward
					maxpoolingBackward(maxIx, outOffset, n, c, C, Q, PQ, CPQ);
				}
			}
			//thread-local nnz maintenance
			return output.recomputeNonZeros(_rl, _ru-1);
		}
		
		protected void maxpoolingForward(double[] maxVal, int[] maxIx, int n, int c, int padh, int padw, int strideh, int stridew, int C, int P, int Q, int R, int S, int HW, int W) {
			SparseBlock sblock = _params.input1.getSparseBlock();
			if( !sblock.isEmpty(n) ) {
				Arrays.fill(maxVal, -Double.MAX_VALUE);
				int apos = sblock.pos(n);
				int alen = sblock.size(n);
				int[] aix = sblock.indexes(n);
				double[] avals = sblock.values(n);
				//find channel start and end, w/ robustness for non-existing entries
				int cpos = (c==0) ? 0 : sblock.posFIndexGTE(n, c*HW);
				int cpos2 = (c+1==C) ? alen : sblock.posFIndexGTE(n, (c+1)*HW);
				cpos = (cpos>=0) ? cpos : alen;
				cpos2 = (cpos2>=0) ? cpos2 : alen;
				int lastix = c*HW-1;
				for(int j=apos+cpos; j<apos+cpos2; j++) {
					//handle skipped zero values
					update0(lastix+1, aix[j], maxVal, maxIx, padh, padw, strideh, stridew, P, Q, R, S, HW, W);
					//handle current non-zero value
					int h = (aix[j] % HW) / W;
					int w = aix[j] % W;
					double val = reluBack && avals[j] < 0 ? 0 : avals[j];
					update(val, maxVal, maxIx, h, w, padh, padw, strideh, stridew, P, Q, R, S, W);
					//memoize last seen index
					lastix = aix[j];
				}
				//handle skipped zero values at end of row
				update0(lastix+1, (c+1)*HW, maxVal, maxIx, padh, padw, strideh, stridew, P, Q, R, S, HW, W);
			}
			else {
				//handle empty row
				Arrays.fill(maxVal, 0);
				for(int p = 0, ix=0; p < P; p++) {
					int h = Math.max(-padh+p*strideh, 0);
					for(int q = 0; q < Q; q++, ix++) {
						int w = Math.max(-padw+q*stridew, 0);
						maxIx[ix] = h * W + w;
					}
				}
			}
		}
		
		protected void maxpoolingBackward(int[] maxIx, int outOffset, int n, int c, int C, int Q, int PQ, int CPQ) {
			double[] dout = doutput.getDenseBlock();
			double[] out = output.getDenseBlock();
			final int doutOffset = n*CPQ + c*PQ;
			for( int pq = 0; pq < PQ; pq++ )
				out[ outOffset + maxIx[pq] ] += dout[ doutOffset + pq ];
		}
		
		private static void update0(int lix, int uix, double[] maxVal, int[] maxIx, int padh, int padw, int strideh, int stridew, int P, int Q, int R, int S, int HW, int W) {
			//TODO exploit constant value and overlap for potential early abort
			for(int i = lix; i<uix; i++)
				update(0, maxVal, maxIx, (i%HW)/W, i%W, padh, padw, strideh, stridew, P, Q, R, S, W);
		}
		
		private static void update(double val, double[] maxVal, int[] maxIx, int h, int w, int padh, int padw, int strideh, int stridew, int P, int Q, int R, int S, int W) {
			//determine lower and upper bounds for p and q
			//(see fillIndexesArray, solved for p and q, reversed)
			int lp = Math.max((h+padh-R+strideh)/strideh, 0);
			int up = Math.min((h+padh+strideh)/strideh, P);
			int lq = Math.max((w+padw-S+stridew)/stridew, 0);
			int uq = Math.min((w+padw+stridew)/stridew, Q);
			
			//maintain max index for all relevant p and q
			int maxIndex = h * W + w;
			for(int p = lp; p < up; p++) 
				for(int q = lq; q < uq; q++) {
					int ix = p * Q + q;
					if( maxVal[ix] < val ) {
						maxVal[ix] = val;
						maxIx[ix] = maxIndex;
					}
				}
		}
	}
	
	/**
	 * Performs the maxpooling backward operation for sparse input and sparse error (dout)
	 */
	public static class PoolingBackwardSparseSparse extends PoolingBackwardSparseDense
	{
		public PoolingBackwardSparseSparse(int rl, int ru, ConvolutionParameters params, boolean relu) {
			super(rl, ru, params, relu, params.input2, params.output);
			if (output.getDenseBlock() == null )
				throw new RuntimeException("Incorrect usage: empty outputs");
			if (!params.input1.isInSparseFormat() || !params.input2.isInSparseFormat())
				throw new RuntimeException("Incorrect usage: Call optimized versions");
		}
		
		@Override
		protected void maxpoolingBackward(int[] maxIx, int outOffset, int n, int c, int C, int Q, int PQ, int CPQ) {
			SparseBlock sblock = doutput.getSparseBlock();
			double[] out = output.getDenseBlock();
			if( sblock.isEmpty(n) )
				return;
			int apos = sblock.pos(n);
			int alen = sblock.size(n);
			int[] aix = sblock.indexes(n);
			double[] avals = sblock.values(n);
			//find channel start and end, w/ robustness for non-existing entries
			int cpos = (c==0) ? 0 : sblock.posFIndexGTE(n, c*PQ);
			int cpos2 = (c+1==C) ? alen : sblock.posFIndexGTE(n, (c+1)*PQ);
			cpos = (cpos>=0) ? cpos : alen;
			cpos2 = (cpos2>=0) ? cpos2 : alen;
			for(int j = apos+cpos; j<apos+cpos2; j++) {
				int p = (aix[j] % PQ) / Q;
				int q = aix[j] % Q;
				int pq = p * Q + q;
				out[ outOffset + maxIx[pq] ] += avals[j];
			}
		}
	}
}
