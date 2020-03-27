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
package org.apache.sysds.runtime.matrix.data;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.concurrent.Callable;

import org.apache.sysds.hops.OptimizerUtils;
import org.apache.sysds.runtime.codegen.LibSpoofPrimitives;
import org.apache.sysds.runtime.data.SparseBlock;
import org.apache.sysds.runtime.matrix.data.LibMatrixDNN.PoolingType;
import org.apache.sysds.runtime.matrix.data.LibMatrixDNNHelper.CellIndex3;

/**
 * This class contains the set of operators used for performing pooling
 */
public class LibMatrixDNNPooling {
	
	// *********************************** low-level runtime operator selection ***********************************************
	// *********************************** based on runtime properties (sparsity, native, etc) ********************************
	// These methods help reduce branch miss predictions and instruction-cache misses.
	// Also, they simplify the design of LibMatrixDNN and help in code-maintenance.
	
	/**
	 * Factory method that returns list of callable tasks for performing pooling operation
	 * 
	 * @param params convolution parameters
	 * @param poolType type of pooling
	 * @return list of callable tasks for performing pooling operation
	 */
	public static ArrayList<Callable<Long>> getPoolingWorkers(DnnParameters params, PoolingType poolType) {
		ArrayList<Callable<Long>> ret = new ArrayList<>();
		// Try to create twice as many tasks as threads for improved load balance
		int k = OptimizerUtils.getConstrainedNumThreads(params.numThreads);
		int taskSize = (int)(Math.ceil((double)params.N / k / 2));
		for(int i = 0; i*taskSize < params.N; i++) {
			if(params.input1.isInSparseFormat())
				ret.add(new SparsePooling(i*taskSize, Math.min((i+1)*taskSize, params.N), params, poolType));
			else
				ret.add(new DensePooling(i*taskSize, Math.min((i+1)*taskSize, params.N), params, poolType));
		}
		return ret;
	}
	
	/**
	 * Factory method that returns list of callable tasks for performing maxpooling backward operation
	 * 
	 * @param params convolution parameters
	 * @param performReluBackward whether to perform ReLU backward
	 * @param poolType type of pooling operation to perform
	 * @return list of callable tasks for performing maxpooling backward operation
	 */
	public static ArrayList<Callable<Long>> getPoolingBackwardWorkers(DnnParameters params, boolean performReluBackward, PoolingType poolType) {
		ArrayList<Callable<Long>> ret = new ArrayList<>();
		// Try to create twice as many tasks as threads for improved load balance
		int k = OptimizerUtils.getConstrainedNumThreads(params.numThreads);
		int taskSize = (int)(Math.ceil((double)params.N / k / 2));
		if(poolType == PoolingType.MAX) {
			boolean sparse1 = params.input1.isInSparseFormat();
			boolean sparse2 = params.input2.isInSparseFormat();
			for(int i = 0; i*taskSize < params.N; i++) {
				if( !sparse1 && !sparse2 )
					ret.add(new PoolingBackwardDenseDense(i*taskSize, Math.min((i+1)*taskSize, params.N), params, performReluBackward));
				else if( !sparse1 && sparse2 )
					ret.add(new PoolingBackwardDenseSparse(i*taskSize, Math.min((i+1)*taskSize, params.N), params, performReluBackward));
				else if( sparse1 && !sparse2 ) 
					ret.add(new PoolingBackwardSparseDense(i*taskSize, Math.min((i+1)*taskSize, params.N), params, performReluBackward));
				else if( sparse1 && sparse2 )
					ret.add(new PoolingBackwardSparseSparse(i*taskSize, Math.min((i+1)*taskSize, params.N), params, performReluBackward));
			}
		}
		else {
			boolean sparse = params.input2.isInSparseFormat();
			for(int i = 0; i*taskSize < params.N; i++) {
				if( !sparse )
					ret.add(new AvgPoolingBackwardDense(i*taskSize, Math.min((i+1)*taskSize, params.N), params));
				else
					ret.add(new AvgPoolingBackwardSparse(i*taskSize, Math.min((i+1)*taskSize, params.N), params));
			}
		}
		return ret;
	}
	
	public static void poolingDenseStride1Pad0(PoolingType pType, double minVal, double pFact, double[] in,
			double[] out, int rl, int ru, int ii, int oi, int C, int P, int Q, int R, int S, int H, int W) {
		boolean max = (pType == PoolingType.MAX);
		int CHW = C * H * W;
		
		if( P == 1 && Q == 1 && W == 1 ) {
			//quick-path w/o materialized index arrays and 
			//simplified inner loops for P = 1, Q = 1, W = 1
			int lenh = Math.min(R,H);
			for(int i = rl; i < ru; i++, oi+=C)
				for (int c = 0, off=ii+(i-rl)*CHW; c < C; c++, off+=H) {
					out[oi+c] = max ? max(minVal, in, off, lenh) :
						avg(minVal, in, off, lenh, pFact);
				}
		}
		else {
			int CPQ = C * P * Q, HW = H * W;
			Arrays.fill(out, rl*CPQ, ru*CPQ, minVal);
			//quick-path w/o materialized index arrays 
			for(int i = rl; i < ru; i++)
				for (int c = 0, off=ii+(i-rl)*CHW, oix=oi+(i-rl)*CPQ; c < C; c++, off+=HW)
					for (int p = 0; p < P; p++, oix+=Q)
						for (int h = p; h < Math.min(p+R,H); h++)
							for (int q = 0, off2=off+h*W; q < Q; q++) {
								out[oix+q] = max ? max(out[oix+q], in, off2+q, Math.min(S,W-q)) :
									avg(out[oix+q], in, off2+q, Math.min(S,W-q), pFact);
							}
		}
	}
	
	private static class DensePooling implements Callable<Long> 
	{
		private final int _rl, _ru; 
		private final DnnParameters _params;

		private final PoolingType _poolingType;
		private final double _poolingMultiplier;
		
		public DensePooling(int rl, int ru, DnnParameters params, PoolingType poolingType) {
			_rl = rl; _ru = ru;
			_params = params;
			_poolingType = poolingType;
			_poolingMultiplier = 1d/(params.R*params.S);
		}
		
		@Override
		public Long call() throws Exception {
			final int C = _params.C, P = _params.P, Q = _params.Q;
			final int R = _params.R, S = _params.S, H = _params.H, W = _params.W;
			final int HW = _params.H*_params.W;
			final int CHW = _params.C*_params.H*_params.W;
			final int CPQ = C*P*Q;
			double[] in = _params.input1.getDenseBlockValues();
			double[] out = _params.output.getDenseBlockValues();
			
			double minValForMaxPoolOperations = _poolingType == PoolingType.AVG ? 0 : _params.minValForMaxPoolOperations;
			boolean max = (_poolingType == PoolingType.MAX);
			
			if( _params.isStride1Pad0() ) {
				poolingDenseStride1Pad0(_poolingType, minValForMaxPoolOperations,
					_poolingMultiplier, in, out, _rl, _ru, _rl*CHW, _rl*CPQ, C, P, Q, R, S, H, W);
			}
			else { //general case
				//thread-local initialization of output block 
				Arrays.fill(out, _rl*CPQ, _ru*CPQ, minValForMaxPoolOperations);
				
				int[] hl = _params.start_indexes_h, hu = _params.end_indexes_h;
				int[] wl = _params.start_indexes_w, wu = _params.end_indexes_w;
				for(int i = _rl; i < _ru; i++)
					for (int c = 0, off=i*CHW, oix=i*CPQ; c < C; c++, off+=HW)
						for (int p = 0; p < P; p++, oix+=Q)
							for (int h = hl[p]; h < hu[p]; h++)
								for (int q = 0, off2=off+h*W; q < Q; q++) {
									out[oix+q] = max ? max(out[oix+q], in, off2+wl[q], wu[q]-wl[q]) :
										avg(out[oix+q], in, off2+wl[q], wu[q]-wl[q], _poolingMultiplier);
								}
			}
			
			//thread-local recomputation of non-zeros
			return _params.output.recomputeNonZeros(_rl, _ru-1);
		}
	}
	
	private static class SparsePooling implements Callable<Long> 
	{
		private final int _rl, _ru; 
		private final DnnParameters _params;
		private double [] outputArray;
		private final int C, P, Q, W, H, CPQ, PQ;
		private final PoolingType _poolingType;
		private final double _poolingMultiplier;
		
		public SparsePooling(int rl, int ru, DnnParameters params, PoolingType poolingType) {
			_rl = rl; _ru = ru;
			_params = params;
			outputArray = params.output.getDenseBlockValues();
			C = params.C; P = params.P; Q = params.Q; H = params.H; 
			W = params.W;
			CPQ = C*P*Q;
			PQ = P*Q;
			_poolingType = poolingType;
			_poolingMultiplier = Math.pow(params.R*params.S, -1);
		}
		
		@Override
		public Long call() throws Exception {
			//thread-local initialization of output block 
			if(_poolingType == PoolingType.MAX)
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
								if(_poolingType == PoolingType.MAX) {
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
								else {
									for (int p = 0; p < P; p++) {
										if(h >= _params.start_indexes_h[p] && h < _params.end_indexes_h[p]) {
											final int outOffsetWithp = outOffset + p*Q;
											for (int q = 0; q < Q; q++) {
												if(w >= _params.start_indexes_w[q] && w < _params.end_indexes_w[q]) {
													outputArray[outOffsetWithp + q] += _poolingMultiplier*nchwVal;
												}
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
	
	//BACKWARD
	
	/**
	 * Performs the avgpooling backward operation for dense error (dout)
	 */
	private static class AvgPoolingBackwardDense implements Callable<Long> 
	{
		public int _rl; public int _ru; 
		private final DnnParameters _params; 
		double [] doutArray;
		MatrixBlock output;
		final int C; final int CHW; final int P; final int Q; final int HW; final int CPQ; final int PQ;
		final double _poolingMultiplier;
		public AvgPoolingBackwardDense(int rl, int ru, DnnParameters params) {
			_rl = rl; _ru = ru;
			_params = params;
			doutArray = params.input2.getDenseBlockValues();
			output = params.output;
			C = params.C; CHW = params.C*params.H*params.W; HW = params.H*params.W;
			P = params.P; Q = params.Q; CPQ = params.C*params.P*params.Q;
			PQ = params.P*params.Q;
			_poolingMultiplier = Math.pow(params.R*params.S, -1);
			if (doutArray == null || output.getDenseBlock() == null )
				throw new RuntimeException("Incorrect usage: empty inputs");
		}
		
		@Override
		public Long call() throws Exception {
			double[] out = output.getDenseBlockValues();
			for(int n = _rl; n < _ru; n++)  {
				for (int c = 0; c < C; c++) {
					final int inputOffset = n*CHW + c*HW;
					final int outputOffset = n*CPQ + c*PQ;
					for (int p = 0; p < P; p++) {
						for (int q = 0; q < Q; q++) {
							int start_index_h = _params.start_indexes_h[p];
							int end_index_h = _params.end_indexes_h[p];
							int start_index_w = _params.start_indexes_w[q];
							int end_index_w = _params.end_indexes_w[q];
							for (int h = start_index_h; h < end_index_h; h++) {
								for (int w = start_index_w; w < end_index_w; w++) {
									out[inputOffset +  h*_params.W + w] += _poolingMultiplier*doutArray[outputOffset +  p * Q + q];
								}
							}
						}
					}
				}
			}
			//thread-local nnz maintenance
			return output.recomputeNonZeros(_rl, _ru-1);
		}
	}
	
	/**
	 * Performs the maxpooling backward operation for dense input and dense error (dout)
	 */
	private static class PoolingBackwardDenseDense implements Callable<Long> 
	{
		public int _rl; public int _ru; 
		private final DnnParameters _params; 
		boolean performReluBackward;
		double [] inputArray, doutArray;
		MatrixBlock output;
		int C; int CHW; int P; int Q; int HW; int CPQ; int PQ;
		public PoolingBackwardDenseDense(int rl, int ru, DnnParameters params, boolean performReluBackward) {
			_rl = rl; _ru = ru;
			_params = params;
			this.performReluBackward = performReluBackward;
			inputArray = params.input1.getDenseBlockValues();
			doutArray = params.input2.getDenseBlockValues();
			output = params.output;
			C = params.C; CHW = params.C*params.H*params.W; HW = params.H*params.W;
			P = params.P; Q = params.Q; CPQ = params.C*params.P*params.Q;
			PQ = params.P*params.Q;
			if (inputArray == null || doutArray == null || output.getDenseBlock() == null )
				throw new RuntimeException("Incorrect usage: empty inputs");
		}
		
		@Override
		public Long call() throws Exception {
			double[] out = output.getDenseBlockValues();
			for(int n = _rl; n < _ru; n++)  {
				for (int c = 0; c < C; c++) {
					final int inputOffset = n*CHW + c*HW;
					final int outputOffset = n*CPQ + c*PQ;
					for (int p = 0; p < P; p++) {
						for (int q = 0; q < Q; q++) {
							int maxIndex = getMaxIndex(p, q, inputOffset, inputArray, _params, performReluBackward);
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
	private static class PoolingBackwardDenseSparse implements Callable<Long> 
	{
		public int _rl; public int _ru; 
		private final DnnParameters _params; 
		MatrixBlock output; 
		boolean performReluBackward;
		double [] inputArray;  MatrixBlock dout;
		int CHW; int P; int Q; int HW;
		public PoolingBackwardDenseSparse(int rl, int ru, DnnParameters params, boolean performReluBackward) {
			_rl = rl; _ru = ru;
			_params = params;
			this.performReluBackward = performReluBackward;
			inputArray = params.input1.getDenseBlockValues();
			dout = params.input2;
			output = params.output;
			CHW = params.C*params.H*params.W; HW = params.H*params.W;
			P = params.P; Q = params.Q; 
			if (inputArray == null || output.getDenseBlock() == null )
				throw new RuntimeException("Incorrect usage: empty inputs");
			if (!params.input2.isInSparseFormat())
				throw new RuntimeException("Incorrect usage: Call optimized versions");
		}
		
		@Override
		public Long call() throws Exception {
			CellIndex3 ix = new CellIndex3();
			double[] out = output.getDenseBlockValues();
			SparseBlock sblock = dout.sparseBlock;
			for(int n = _rl; n < _ru; n++)  {
				if( sblock.isEmpty(n) ) continue;
				int apos = sblock.pos(n);
				int alen = sblock.size(n);
				int[] aix = sblock.indexes(n);
				double[] avals = sblock.values(n);
				for(int j = apos; j < apos+alen; j++) {
					ix = LibMatrixDNNHelper.computeTensorIndexes(aix[j], P, Q, ix);
					final int inputOffset = n*CHW + ix.ix1*HW;
					int maxIndex = getMaxIndex(ix.ix2, ix.ix3,
						inputOffset, inputArray, _params, performReluBackward);
					if(maxIndex != -1)
						out[maxIndex] += avals[j];
				}
			}
			//thread-local nnz maintenance
			return output.recomputeNonZeros(_rl, _ru-1);
		}
	}
	
	/**
	 * Performs the avgpooling backward operation for sparse error (dout)
	 */
	private static class AvgPoolingBackwardSparse implements Callable<Long> 
	{
		public int _rl; public int _ru; 
		private final DnnParameters _params; 
		MatrixBlock output; 
		MatrixBlock dout;
		int CHW; int P; int Q; int HW; final double _poolingMultiplier;
		public AvgPoolingBackwardSparse(int rl, int ru, DnnParameters params) {
			_rl = rl; _ru = ru;
			_params = params;
			dout = params.input2;
			output = params.output;
			CHW = params.C*params.H*params.W; HW = params.H*params.W;
			P = params.P; Q = params.Q; 
			_poolingMultiplier = Math.pow(params.R*params.S, -1);
			if (output.getDenseBlock() == null )
				throw new RuntimeException("Incorrect usage: empty inputs");
		}
		
		@Override
		public Long call() throws Exception {
			CellIndex3 ix = new CellIndex3();
			double[] out = output.getDenseBlockValues();
			SparseBlock sblock = dout.sparseBlock;
			for(int n = _rl; n < _ru; n++)  {
				if( sblock.isEmpty(n) ) continue;
				int apos = sblock.pos(n);
				int alen = sblock.size(n);
				int[] aix = sblock.indexes(n);
				double[] avals = sblock.values(n);
				for(int j = apos; j < apos+alen; j++) {
					ix = LibMatrixDNNHelper.computeTensorIndexes(aix[j], P, Q, ix);
					int c = ix.ix1;
					int p = ix.ix2;
					int q = ix.ix3;
					final int inputOffset = n*CHW + c*HW;
					int start_index_h = _params.start_indexes_h[p];
					int end_index_h = _params.end_indexes_h[p];
					int start_index_w = _params.start_indexes_w[q];
					int end_index_w = _params.end_indexes_w[q];
					for (int h = start_index_h; h < end_index_h; h++) {
						for (int w = start_index_w; w < end_index_w; w++) {
							out[inputOffset +  h*_params.W + w] += _poolingMultiplier*avals[j];
						}
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
	private static class PoolingBackwardSparseDense implements Callable<Long> 
	{
		private final int _rl, _ru; 
		private final DnnParameters _params; 
		private final boolean reluBack;
		protected final MatrixBlock doutput, output;
		
		protected PoolingBackwardSparseDense(int rl, int ru, DnnParameters params, boolean relu, MatrixBlock dout, MatrixBlock out) {
			_rl = rl; _ru = ru; 
			_params = params;
			reluBack = relu;
			doutput = dout;
			output = out;
		}
		
		public PoolingBackwardSparseDense(int rl, int ru, DnnParameters params, boolean relu) {
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
			double[] dout = doutput.getDenseBlockValues();
			double[] out = output.getDenseBlockValues();
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
	private static class PoolingBackwardSparseSparse extends PoolingBackwardSparseDense
	{
		public PoolingBackwardSparseSparse(int rl, int ru, DnnParameters params, boolean relu) {
			super(rl, ru, params, relu, params.input2, params.output);
			if (output.getDenseBlock() == null )
				throw new RuntimeException("Incorrect usage: empty outputs");
			if (!params.input1.isInSparseFormat() || !params.input2.isInSparseFormat())
				throw new RuntimeException("Incorrect usage: Call optimized versions");
		}
		
		@Override
		protected void maxpoolingBackward(int[] maxIx, int outOffset, int n, int c, int C, int Q, int PQ, int CPQ) {
			SparseBlock sblock = doutput.getSparseBlock();
			double[] out = output.getDenseBlockValues();
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
	
	private static double avg(final double aval, double[] b, final int bi, final int len, final double poolingMultiplier) {
		return LibSpoofPrimitives.vectSum(b, bi, len) * poolingMultiplier + aval;
	}
	
	private static double max(final double aval, double[] b, final int bi, final int len) {
		double ret = aval;
		for( int i = bi; i < bi+len; i++ )
			ret = Math.max(ret, b[i]);
		return ret;
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
	private static int getMaxIndex(int p, int q, int inputOffset, double [] inputArray, DnnParameters params, boolean performReluBackward) {
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
}
