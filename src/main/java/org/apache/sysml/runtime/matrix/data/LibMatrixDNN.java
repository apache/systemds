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

import java.lang.ref.SoftReference;
import java.util.ArrayList;
import java.util.Iterator;
import java.util.List;
import java.util.concurrent.Callable;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.concurrent.atomic.AtomicLong;

import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;
import org.apache.sysml.api.DMLScript;
import org.apache.sysml.hops.OptimizerUtils;
import org.apache.sysml.runtime.DMLRuntimeException;
import org.apache.sysml.runtime.util.ConvolutionUtils;

public class LibMatrixDNN {
	
	protected static final Log LOG =  LogFactory.getLog(LibMatrixDNN.class.getName());

	public static final boolean ALLOW_MULTI_THREADED_OPS = true;
	// Using hashmap to avoid any performance impacts of multimap
	private static final ConcurrentHashMap<Integer, SoftReference<double[]>> non_zeroed_double_arr = new ConcurrentHashMap<Integer, SoftReference<double[]>>();
	private static final int NON_ZEROED_DOUBLE_ARR_THRESHOLD = 100;
	public static void cacheReuseableData(double[] arr) {
		if(arr != null && arr.length >= NON_ZEROED_DOUBLE_ARR_THRESHOLD) {
			// Put the last recently removed arrays into the NON_ZEROED_DOUBLE_ARR as 
			// it has lower probability of being garbage collected
			// new Integer(arr.length) can be avoided here as autoboxing will do the trick
			non_zeroed_double_arr.put(arr.length, new SoftReference<double[]>(arr));
		}
	}
	public static double[] getReuseableData(long length) {
		if(length >= NON_ZEROED_DOUBLE_ARR_THRESHOLD) {
			// Explicit "new Integer" required here for HashMap.remove
			SoftReference<double[]> arr = non_zeroed_double_arr.remove(new Integer((int) length));
			if(arr != null) {
				return arr.get();
			}
		}
		return null;
	}
	
	enum TaskType {
		ReshapeCol, Rotate180, Im2Col, Col2Im, MaxPooling_Forward, MaxPooling_Backward, LoopBasedConv2d
	}
	
	public static class TemporaryConvolutionData {
		public int [] minIndexArrR;
		public int [] minIndexArrS;
		public int [] maxIndexArrR;
		public int [] maxIndexArrS;
		int minCommonIndexS;
		int maxCommonIndexS;
	}
	
	private static AtomicLong conv2dSparseCount = new AtomicLong(0);
	private static AtomicLong conv2dDenseCount = new AtomicLong(0);
	private static AtomicLong conv2dBwdFilterSparseCount = new AtomicLong(0);
	private static AtomicLong conv2dBwdFilterDenseCount = new AtomicLong(0);
	private static AtomicLong conv2dBwdDataSparseCount = new AtomicLong(0);
	private static AtomicLong conv2dBwdDataDenseCount = new AtomicLong(0);
	private static AtomicLong im2colSparseCount = new AtomicLong(0);
	private static AtomicLong im2colDenseCount = new AtomicLong(0);
	private static AtomicLong maxPoolBwdSparseCount = new AtomicLong(0);
	private static AtomicLong maxPoolBwdDenseCount = new AtomicLong(0);
	public static void appendStatistics(StringBuilder sb) {
		sb.append("LibMatrixDNN dense count (conv/bwdF/bwdD/im2col/maxBwd):\t" 
				+ conv2dDenseCount.get() + "/"
				+ conv2dBwdFilterDenseCount.get() + "/"
				+ conv2dBwdDataDenseCount.get() + "/"
				+ im2colDenseCount.get() + "/"
				+ maxPoolBwdDenseCount.get() + ".\n");
		sb.append("LibMatrixDNN sparse count (conv/bwdF/bwdD/im2col/maxBwd):\t" 
				+ conv2dSparseCount.get() + "/"
				+ conv2dBwdFilterSparseCount.get() + "/"
				+ conv2dBwdDataSparseCount.get() + "/"
				+ im2colSparseCount.get() + "/"
				+ maxPoolBwdSparseCount.get() + ".\n");
	}
	public static void resetStatistics() {
		conv2dDenseCount.set(0);
		conv2dBwdFilterDenseCount.set(0);
		conv2dBwdDataDenseCount.set(0);
		im2colDenseCount.set(0);
		maxPoolBwdDenseCount.set(0);
		
		conv2dSparseCount.set(0);
		conv2dBwdFilterSparseCount.set(0);
		conv2dBwdDataSparseCount.set(0);
		im2colSparseCount.set(0);
		maxPoolBwdSparseCount.set(0);
	}
	
	public static class ConvolutionParameters {
		public int N; public int C; public int H; public int W;
		public int K; public int R; public int S; public int stride_h; public int stride_w; public int pad_h; public int pad_w;
		public int P; public int Q; public int numThreads;
		
		public AtomicLong outputNNZ = new AtomicLong(-1);
		
		MatrixBlock input1; MatrixBlock input2; MatrixBlock output;
		boolean reuseNonZeroedOutput = false;
		
		public TemporaryConvolutionData tmpData;
		
		private int convertToInt(long val) throws DMLRuntimeException {
			if( val > Integer.MAX_VALUE ) {
				throw new DMLRuntimeException("The value for ConvolutionParameters is too large:" + val);
			}
			return (int) val;
		}
		
		public boolean compare(ConvolutionParameters that) {
			if(this.N == that.N && this.C == that.C && this.H == that.H && this.W == that.W
					&& this.K == that.K && this.R == that.R && this.S == that.S && this.stride_h == that.stride_h
					 && this.stride_w == that.stride_w  && this.pad_h == that.pad_h
					  && this.pad_w == that.pad_w   && this.numThreads == that.numThreads) {
				return true;
			}
			return false;
		}
		
		public ConvolutionParameters(long N, long C, long H, long W,
				long K, long R, long S, long stride_h, long stride_w, long pad_h, long pad_w, int numThreads) throws DMLRuntimeException {
			this.N = convertToInt(N);
			this.C = convertToInt(C);
			this.H = convertToInt(H);
			this.W = convertToInt(W);
			this.K = convertToInt(K);
			this.R = convertToInt(R);
			this.S = convertToInt(S);
			this.stride_h = convertToInt(stride_h);
			this.stride_w = convertToInt(stride_w);
			this.pad_h = convertToInt(pad_h);
			this.pad_w = convertToInt(pad_w);
			if(H >= 0 && pad_h >= 0 && R >= 0 && stride_h >= 0)
				P = (int) ((H + 2 * pad_h - R) / stride_h + 1);
			else
				P = -1;
			// P = convertToInt(ConvolutionUtils.getP(H, R, stride_h, pad_h));
			
			if(W >= 0 && pad_w >= 0 && S >= 0 && stride_w >= 0)
				Q = (int) ((W + 2 * pad_w - S) / stride_w + 1);
			else
				Q = -1;
			// Q = convertToInt(ConvolutionUtils.getQ(W, S, stride_w, pad_w));
			
			this.numThreads = numThreads;
		}
		
		public ConvolutionParameters(int N, int C, int H, int W,
			int K, int R, int S, int stride_h, int stride_w, int pad_h, int pad_w, int numThreads) {
			this.N = N;
			this.C = C;
			this.H = H;
			this.W = W;
			this.K = K;
			this.R = R;
			this.S = S;
			this.stride_h = stride_h;
			this.stride_w = stride_w;
			this.pad_h = pad_h;
			this.pad_w = pad_w;
			P = (int) ConvolutionUtils.getP(H, R, stride_h, pad_h);
			Q = (int) ConvolutionUtils.getQ(W, S, stride_w, pad_w);
			this.numThreads = numThreads;
		}
		
		public void setReuseNonZeroedOutput(boolean reuseNonZeroedOutput) {
			this.reuseNonZeroedOutput = reuseNonZeroedOutput;
		}

		public boolean isOutputThreadSafe() {
			return output.isThreadSafe();
		}
	}
	
	public static void conv2d_backward_filter(MatrixBlock input, MatrixBlock dout, MatrixBlock outputBlock, ConvolutionParameters params) throws DMLRuntimeException {
		params.input1 = input;
		params.input2 = dout;
		params.output = outputBlock;
		if(input.getNumRows() != params.N || input.getNumColumns() != params.C*params.H*params.W || 
				dout.getNumRows() != params.N || dout.getNumColumns() != params.K*params.P*params.Q) {
			throw new DMLRuntimeException("Incorrect input to conv2d_backward_filter");
		}
		if(params.stride_h <= 0 || params.stride_w <= 0) {
			throw new DMLRuntimeException("Only positive strides supported");
		}
		
		if(DMLScript.STATISTICS) {
			if(input.isInSparseFormat() || dout.isInSparseFormat()) {
				conv2dBwdFilterSparseCount.addAndGet(1);
			}
			else {
				conv2dBwdFilterDenseCount.addAndGet(1);
			}
		}
		
		int constrainedNumThreads = OptimizerUtils.getConstrainedNumThreads(params.numThreads);
		if(!ALLOW_MULTI_THREADED_OPS || constrainedNumThreads <= 1) {
			for (int c = 0; c < params.C; c++) {
				for (int k = 0; k < params.K; k++) {
					for (int r = 0; r < params.R; r++) {
						for (int s = 0; s < params.S; s++) {
							doConv2d_Backward_Filter(k, c, r, s, params);
						}
					}
				}
			}
		}
		else {
			ArrayList<ConvBackwardFilterTask> tasks = new ArrayList<ConvBackwardFilterTask>();		
			for (int c = 0; c < params.C; c++) {
				for (int k = 0; k < params.K; k++) {
					for (int r = 0; r < params.R; r++) {
						for (int s = 0; s < params.S; s++) {
							tasks.add(new ConvBackwardFilterTask(k, c, r, s, params));
						}
					}
				}
			}
			ExecutorService pool = Executors.newFixedThreadPool( Math.min(constrainedNumThreads, tasks.size()) );
			List<Future<Object>> taskret;
			try {
				taskret = pool.invokeAll(tasks);
				pool.shutdown();
				for( Future<Object> task : taskret )
					task.get();
			} catch (InterruptedException e) {
				throw new DMLRuntimeException("Error while executing multi-threaded ConvBackwardFilterTask", e);
			} catch (ExecutionException e) {
				throw new DMLRuntimeException("Error while executing multi-threaded ConvBackwardFilterTask", e);
			}
		}
	}
	
	private static void doConv2d_Backward_Filter(int k, int c, int r, int s, ConvolutionParameters params) throws DMLRuntimeException {
		double [] inputArray = null;
		if (!params.input1.isInSparseFormat())
			inputArray = params.input1.getDenseBlock();
		double [] doutArray = null;
		if (!params.input2.isInSparseFormat())
			doutArray = params.input2.getDenseBlock();
		double [] outputArray = params.output.getDenseBlock();
		
		double outputVal = 0;
		if(inputArray == null && doutArray == null) {
			outputVal = doConv2d_Backward_Filter_SparseSparse(k, c, r, s, params);
		}
		else if(inputArray != null && doutArray == null) {
			outputVal = doConv2d_Backward_Filter_DenseSparse(k, c, r, s, params, inputArray);
		}
		else if(inputArray == null && doutArray != null) {
			outputVal = doConv2d_Backward_Filter_SparseDense(k, c, r, s, params, doutArray);
		}
		else {
			outputVal = doConv2d_Backward_Filter_DenseDense(k, c, r, s, params, inputArray, doutArray);
		}
		
		outputArray[k*params.C*params.R*params.S + c*params.R*params.S + r*params.S + s] = outputVal;
	}
	
	private static double doConv2d_Backward_Filter_SparseDense(int k, int c, int r, int s, ConvolutionParameters params, double [] doutArray) throws DMLRuntimeException {
		double outputVal = 0;
		// To ensure h >= 0 && h < params.H 
		int pMin = (int) Math.max(0, Math.ceil(((double)(params.pad_h-r))/params.stride_h));
		int qMin = (int) Math.max(0, Math.ceil(((double)(params.pad_w-s))/params.stride_w));
		// To ensure w >= 0 && w < params.W 
		int pMax = (int) Math.min(params.P, Math.ceil(((double)(params.H+params.pad_h-r))/params.stride_h));
		int qMax = (int) Math.min(params.Q, Math.ceil(((double)(params.W+params.pad_w-s))/params.stride_w));
		
		// TODO: Optimize this case
		for (int n = 0; n < params.N; n++) {
			int doutOffset = n*params.K*params.P*params.Q + k*params.P*params.Q;
			for (int p = pMin; p < pMax; p++) {
				for (int q = qMin; q < qMax; q++) {
					int h = p*params.stride_h + r - params.pad_h;
					int w = q*params.stride_w + s - params.pad_w;
					outputVal += doutArray[doutOffset + p*params.Q + q]*params.input1.quickGetValue(n, c*params.H*params.W + h*params.W + w);
				}
			}
		}
		
		return outputVal;
	}
	
	private static double doConv2d_Backward_Filter_DenseDense(int k, int c, int r, int s, ConvolutionParameters params, double [] inputArray, double [] doutArray) {
		double outputVal = 0;
		// To ensure h >= 0 && h < params.H 
		int pMin = (int) Math.max(0, Math.ceil(((double)(params.pad_h-r))/params.stride_h));
		int qMin = (int) Math.max(0, Math.ceil(((double)(params.pad_w-s))/params.stride_w));
		// To ensure w >= 0 && w < params.W 
		int pMax = (int) Math.min(params.P, Math.ceil(((double)(params.H+params.pad_h-r))/params.stride_h));
		int qMax = (int) Math.min(params.Q, Math.ceil(((double)(params.W+params.pad_w-s))/params.stride_w));
		
		for (int n = 0; n < params.N; n++) {
			int inputOffset =  n*params.C*params.H*params.W + c*params.H*params.W + s - params.pad_w;
			int doutOffset = n*params.K*params.P*params.Q + k*params.P*params.Q;
			for (int p = pMin; p < pMax; p++) {
				int h = p*params.stride_h + r - params.pad_h;
				for (int q = qMin; q < qMax; q++) {
					int w = q*params.stride_w;
					outputVal += doutArray[doutOffset + p*params.Q + q]*inputArray[inputOffset + h*params.W+w];
				}
			}
		}
				
		return outputVal;
	}
	
	private static void computeTensorIndexes(int i, int j, int [] ret, int N, int C, int H, int W) throws DMLRuntimeException {
		ret[0] = i;
		ret[1] = j / (H*W);
		ret[2] = (j - ret[1]*(H*W))/W;
		ret[3] = j % W;
	}
	
	private static double doConv2d_Backward_Filter_DenseSparse(int k, int c, int r, int s, ConvolutionParameters params, double [] inputArray) throws DMLRuntimeException {
		MatrixBlock dout = params.input2;
		double outputVal = 0;
		Iterator<IJV> iter = dout.sparseBlock.getIterator();
		int [] tensorIndexes = new int[4];
		while(iter.hasNext()) {
			IJV ijv = iter.next();
			computeTensorIndexes(ijv.getI(), ijv.getJ(), tensorIndexes, params.N, params.K, params.P, params.Q);
			if(k == tensorIndexes[1]) {
				int n = tensorIndexes[0];
				int p = tensorIndexes[2];
				int q = tensorIndexes[3];
				
				double doutVal = ijv.getV();
				int h = p*params.stride_h + r - params.pad_h;
				int w = q*params.stride_w + s - params.pad_w;
				if(h >= 0 && h < params.H && w >= 0 && w < params.W) {
					outputVal += doutVal*inputArray[n*params.C*params.H*params.W + c*params.H*params.W + h*params.W+w];
				}
			}
		}
		return outputVal;
	}
	
	private static double doConv2d_Backward_Filter_SparseSparse(int k, int c, int r, int s, ConvolutionParameters params) throws DMLRuntimeException {
		MatrixBlock dout = params.input2;
		double outputVal = 0;
		Iterator<IJV> iter = dout.sparseBlock.getIterator();
		int [] tensorIndexes = new int[4];
		
		while(iter.hasNext()) {
			IJV ijv = iter.next();
			computeTensorIndexes(ijv.getI(), ijv.getJ(), tensorIndexes, params.N, params.K, params.P, params.Q);
			if(k == tensorIndexes[1]) {
				int n = tensorIndexes[0];
				int p = tensorIndexes[2];
				int q = tensorIndexes[3];
				
				double doutVal = ijv.getV();
				int h = p*params.stride_h + r - params.pad_h;
				int w = q*params.stride_w + s - params.pad_w;
				if(h >= 0 && h < params.H && w >= 0 && w < params.W) {
					outputVal += doutVal*params.input1.quickGetValue(n, c*params.H*params.W + h*params.W + w);
				}
			}
		}
		return outputVal;
	}
	
	private static class ConvBackwardFilterTask implements Callable<Object> {
		int k; int c; int r; int s;
		ConvolutionParameters params;
		public ConvBackwardFilterTask(int k, int c, int r, int s, ConvolutionParameters params) {
			this.k = k;
			this.c = c;
			this.r = r;
			this.s = s;
			this.params = params;
		}

		@Override
		public Object call() throws Exception {
			doConv2d_Backward_Filter(k, c, r, s, params);
			return null;
		}
		
	}
	
	public static void conv2d(MatrixBlock input, MatrixBlock filter, MatrixBlock outputBlock, ConvolutionParameters params) throws DMLRuntimeException {
		params.input1 = input;
		params.input2 = filter;
		params.output = outputBlock;
		
		if(input.getNumRows() != params.N || input.getNumColumns() != params.C*params.H*params.W || 
				filter.getNumRows() != params.K || filter.getNumColumns() != params.C*params.R*params.S) {
			throw new DMLRuntimeException("Incorrect input to conv2d");
		}
		
		if(DMLScript.STATISTICS) {
			if(input.isInSparseFormat() || filter.isInSparseFormat()) {
				conv2dSparseCount.addAndGet(1);
			}
			else {
				conv2dDenseCount.addAndGet(1);
			}
		}
		
		params.tmpData = new TemporaryConvolutionData();
		if(input.isInSparseFormat()) {
			params.tmpData.minIndexArrR = new int[params.H];
			params.tmpData.minIndexArrS = new int[params.W];
			for(int h = 0; h < params.H; h++) {
				for (int r = 0; r < params.R; r++) {
					// int h = p*params.stride_h + r - params.pad_h;
					if((h + params.pad_h - r) % params.stride_h == 0) {
						params.tmpData.minIndexArrR[h] = r;
						break;
					}
				}
			}
			for(int w = 0; w < params.W; w++) {
				for (int s = 0; s < params.S; s++) {
					// int h = p*params.stride_h + r - params.pad_h;
					if((w + params.pad_w - s) % params.stride_w == 0) {
						params.tmpData.minIndexArrS[w] = s;
						break;
					}
				}
			}
		}
		else {
			params.tmpData.minIndexArrR = new int[params.R];
			params.tmpData.maxIndexArrR = new int[params.R];
			params.tmpData.minIndexArrS = new int[params.S];
			params.tmpData.maxIndexArrS = new int[params.S];
			for (int r = 0; r < params.R; r++) {
				params.tmpData.minIndexArrR[r] = getMinPQ(params.pad_h, r, params.stride_h);
				params.tmpData.maxIndexArrR[r] = getMaxPQ(params.pad_h, r, params.stride_h, params.P, params.H);
			}
			for (int s = 0; s < params.S; s++) {
				params.tmpData.minIndexArrS[s] = getMinPQ(params.pad_w, s, params.stride_w);
				params.tmpData.maxIndexArrS[s] = getMaxPQ(params.pad_w, s, params.stride_w, params.Q, params.W);
			}
			params.tmpData.minCommonIndexS = params.tmpData.minIndexArrS[0];
			params.tmpData.maxCommonIndexS = params.tmpData.maxIndexArrS[0];
			for (int s = 1; s < params.S; s++) {
				params.tmpData.minCommonIndexS = Math.max(params.tmpData.minCommonIndexS, params.tmpData.minIndexArrS[s]);
				params.tmpData.maxCommonIndexS = Math.min(params.tmpData.maxCommonIndexS, params.tmpData.maxIndexArrS[s]);
			}
		}
		
		int constrainedNumThreads = OptimizerUtils.getConstrainedNumThreads(params.numThreads);
		if(!ALLOW_MULTI_THREADED_OPS || constrainedNumThreads <= 1) {
			for (int n = 0; n < params.N; n++) {
				for (int k = 0; k < params.K; k++) {
					doLoopBasedConv2d(n, n+1, k, params);
				}
			}
		}
		else
			runConvTask(constrainedNumThreads, params.K, TaskType.LoopBasedConv2d, params);
	}
	
	public static void maxpooling_backward(MatrixBlock input, MatrixBlock dout, MatrixBlock outputBlock, ConvolutionParameters params) throws DMLRuntimeException {
		params.input1 = input;
		params.input2 = dout;
		params.output = outputBlock;
		if(input.getNumColumns() != params.C*params.H*params.W || input.getNumRows() != params.N) {
			throw new DMLRuntimeException("Incorrect input dimensions in maxpooling_backward:" + input.getNumRows() + " " + input.getNumColumns() + " " + params.N + " " + params.K*params.P*params.Q);
		}

		if(dout.getNumColumns() != params.C*params.P*params.Q || dout.getNumRows() != params.N) {
			throw new DMLRuntimeException("Incorrect dout dimensions in maxpooling_backward:" + input.getNumRows() + " " + input.getNumColumns() + " " + params.N + " " + params.K*params.P*params.Q);
		}
		
		if(DMLScript.STATISTICS) {
			if(input.isInSparseFormat() || dout.isInSparseFormat()) {
				maxPoolBwdSparseCount.addAndGet(1);
			}
			else {
				maxPoolBwdDenseCount.addAndGet(1);
			}
		}

		int constrainedNumThreads = OptimizerUtils.getConstrainedNumThreads(params.numThreads);
		if(!ALLOW_MULTI_THREADED_OPS || constrainedNumThreads <= 1) {
			for (int n = 0; n < params.N; n++) {
				for (int c = 0; c < params.C; c++) {
					doPoolingBackward(n, c, params);
				}
			}
		}
		else {
			runConvTask(constrainedNumThreads, params.C, TaskType.MaxPooling_Backward, params);
		}
	}
	
	private static void doLoopBasedConv2dDenseDense(int n1, int n2, int k, ConvolutionParameters params, 
			double [] inputArray, double [] filterArray) {
		double [] outputArray = params.output.getDenseBlock();
		int [] minIndexArrR = params.tmpData.minIndexArrR;
		int [] maxIndexArrR = params.tmpData.maxIndexArrR;
		int [] minIndexArrS = params.tmpData.minIndexArrS;
		int [] maxIndexArrS = params.tmpData.maxIndexArrS;
		
		final int minCommonIndexS = params.tmpData.minCommonIndexS;
		final int maxCommonIndexS = params.tmpData.maxCommonIndexS;
		
		final int minS = (params.S >= 4) ? (params.S - params.S % 4) : 0;
		
		for (int n = n1; n < n2; n++) {
			for (int c = 0; c < params.C; c++) {
				for (int r = 0; r < params.R; r++) {
					final int filterOffset = k*params.C*params.R*params.S + c*params.R*params.S + r*params.S;
					for (int p = minIndexArrR[r]; p < maxIndexArrR[r]; p++) {
						final int h = p*params.stride_h + r - params.pad_h;
						final int inputOffSet = n*params.C*params.H*params.W + c*params.H*params.W + h*params.W - params.pad_w;
						final int outputOffset = n*params.K*params.P*params.Q + k*params.P*params.Q + p*params.Q;
						
						for (int q = minCommonIndexS; q < maxCommonIndexS; q++) {
							final int wOffset = inputOffSet + q*params.stride_w;
							// ------------------------------------------------------------------------
							// Efficient striding with vectorization
							final int outOffsetWithQ = outputOffset + q;
							for (int s = 0; s < minS; s += 4) {
								final int inOffsetWithS = wOffset + s;
								final int filterOffsetWithS = filterOffset + s;
								outputArray[outOffsetWithQ] += inputArray[inOffsetWithS]*filterArray[filterOffsetWithS]
										+ inputArray[inOffsetWithS+1]*filterArray[filterOffsetWithS+1]
										+ inputArray[inOffsetWithS+2]*filterArray[filterOffsetWithS+2]
										+ inputArray[inOffsetWithS+3]*filterArray[filterOffsetWithS+3];
							}
							// ------------------------------------------------------------------------
							// Efficient striding without vectorization
							for (int s = minS; s < params.S; s++) {
								outputArray[outputOffset + q] += inputArray[wOffset + s]*filterArray[filterOffset + s];
							}
							// ------------------------------------------------------------------------
						}
						// ------------------------------------------------------------------------
						// Inefficient striding
						for (int s = 0; s < params.S; s++) {
							for (int q = minIndexArrS[s]; q < minCommonIndexS; q++) {
								final int w = q*params.stride_w + s;
								outputArray[outputOffset + q] += inputArray[inputOffSet + w]*filterArray[filterOffset + s];
							}
							for (int q = maxCommonIndexS; q < maxIndexArrS[s]; q++) {
								final int w = q*params.stride_w + s;
								outputArray[outputOffset + q] += inputArray[inputOffSet + w]*filterArray[filterOffset + s];
							}
						}
						// ------------------------------------------------------------------------
					}
				}
			}
		}
	}
	
	private static void doLoopBasedConv2dDenseSparse(int n, int k, ConvolutionParameters params, double [] inputArray) throws DMLRuntimeException {
		double [] outputArray = params.output.getDenseBlock();
		int [] minIndexArrR = params.tmpData.minIndexArrR;
		int [] maxIndexArrR = params.tmpData.maxIndexArrR;
		int [] minIndexArrS = params.tmpData.minIndexArrS;
		int [] maxIndexArrS = params.tmpData.maxIndexArrS;
		final int outputOffset = n*params.K*params.P*params.Q + k*params.P*params.Q;
		
		Iterator<IJV> iter = params.input2.sparseBlock.getIterator();
		int [] tensorIndexes = new int[4];
		
		while(iter.hasNext()) {
			IJV ijv = iter.next();
			computeTensorIndexes(ijv.getI(), ijv.getJ(), tensorIndexes, params.K, params.C, params.R, params.S);
			if(k == tensorIndexes[0]) {
				int c = tensorIndexes[1];
				int r = tensorIndexes[2];
				int s = tensorIndexes[3];
				double filterVal = ijv.getV();
				final int inputOffset = n*params.C*params.H*params.W + c*params.H*params.W + s - params.pad_w;
				for (int p = minIndexArrR[r]; p < maxIndexArrR[r]; p++) {
					final int hOffset = inputOffset + (p*params.stride_h + r - params.pad_h)*params.W;
					final int pOffset = outputOffset + p*params.Q;
					for (int q = minIndexArrS[s]; q < maxIndexArrS[s]; q++) {
						final int w = q*params.stride_w;
						outputArray[pOffset + q] += inputArray[hOffset + w]*filterVal;
					}
				}
			}
		}
	}
	
	private static void doLoopBasedConv2dSparseDense(int n, int k, ConvolutionParameters params, double [] filterArray) throws DMLRuntimeException {
		double [] outputArray = params.output.getDenseBlock();
		int outputOffset = n*params.K*params.P*params.Q + k*params.P*params.Q;
		
		Iterator<IJV> iter = params.input1.sparseBlock.getIterator();
		int [] tensorIndexes = new int[4];
		
		int [] minIndexArrR = params.tmpData.minIndexArrR;
		int [] minIndexArrS = params.tmpData.minIndexArrS;
		while(iter.hasNext()) {
			IJV ijv = iter.next();
			computeTensorIndexes(ijv.getI(), ijv.getJ(), tensorIndexes, params.N, params.C, params.H, params.W);
			if(n == tensorIndexes[0]) {
				int c = tensorIndexes[1];
				int h = tensorIndexes[2];
				int w = tensorIndexes[3];
				double imgVal = ijv.getV();
				for (int r = minIndexArrR[h]; r < params.R; r += params.stride_h) {
					int filterOffset = k*params.C*params.R*params.S + c*params.R*params.S + r*params.S;
					for (int s = minIndexArrS[w]; s < params.S; s += params.stride_w) {
						int p = (int)Math.ceil(((double)(h + params.pad_h - r)) / params.stride_h);
						int q = (int)Math.ceil(((double)(w + params.pad_w - s)) / params.stride_w);
						if(p >= 0 && p < params.P && q >= 0 && q < params.Q) {
							double filterVal = filterArray[filterOffset + s];
							outputArray[outputOffset + p*params.Q + q] += imgVal*filterVal;
						}
					}
				}	
			}
		}
	}
	
	private static void doLoopBasedConv2dSparseSparse(int n, int k, ConvolutionParameters params) throws DMLRuntimeException {
		double [] outputArray = params.output.getDenseBlock();
		int [] minIndexArrR = params.tmpData.minIndexArrR;
		int [] maxIndexArrR = params.tmpData.maxIndexArrR;
		int [] minIndexArrS = params.tmpData.minIndexArrS;
		int [] maxIndexArrS = params.tmpData.maxIndexArrS;
		int outputOffset = n*params.K*params.P*params.Q + k*params.P*params.Q;
		
		
		int [] tensorIndexesImage = new int[4];
		int [] tensorIndexesFilter = new int[4];

		Iterator<IJV> iter = params.input1.sparseBlock.getIterator();
		
		while(iter.hasNext()) {
			IJV ijv = iter.next();
			computeTensorIndexes(ijv.getI(), ijv.getJ(), tensorIndexesImage, params.N, params.C, params.H, params.W);
			if(n == tensorIndexesImage[0]) {
				int c = tensorIndexesImage[1];
				int h = tensorIndexesImage[2];
				int w = tensorIndexesImage[3];
				double imgVal = ijv.getV();
		
				Iterator<IJV> iter1 = params.input2.sparseBlock.getIterator();
				while(iter1.hasNext()) {
					IJV ijv1 = iter1.next();
					computeTensorIndexes(ijv1.getI(), ijv1.getJ(), tensorIndexesFilter, params.K, params.C, params.R, params.S);
					if(k == tensorIndexesFilter[0] && c == tensorIndexesFilter[1]) {
						int r =  tensorIndexesFilter[2];
						int s =  tensorIndexesFilter[3];
						if((r-minIndexArrR[h])%params.stride_h == 0 && (s-minIndexArrS[w])%params.stride_w == 0) {
							int p = (int)Math.ceil(((double)(h + params.pad_h - r)) / params.stride_h);
							int q = (int)Math.ceil(((double)(w + params.pad_w - s)) / params.stride_w);
							if(p >= 0 && p < params.P && q >= 0 && q < params.Q) {
								double filterVal =  ijv1.getV();
								outputArray[outputOffset + p*params.Q + q] += imgVal*filterVal;
							}
						}
					}
				}
			}
		}
		
		while(iter.hasNext()) {
			IJV ijv = iter.next();
			computeTensorIndexes(ijv.getI(), ijv.getJ(), tensorIndexesFilter, params.K, params.C, params.R, params.S);
			if(k == tensorIndexesFilter[0]) {
				int c = tensorIndexesFilter[1];
				int r = tensorIndexesFilter[2];
				int s = tensorIndexesFilter[3];
				double filterVal = ijv.getV();
				for (int p = minIndexArrR[r]; p < maxIndexArrR[r]; p++) {
					int h = p*params.stride_h + r - params.pad_h;
					for (int q = minIndexArrS[s]; q < maxIndexArrS[s]; q++) {
						int w = q*params.stride_w + s - params.pad_w;
						// TODO: Improve the performance of sparse sparse 
						outputArray[outputOffset + p*params.Q + q] += sparseConvMultiply(filterVal, params, n, c, h, w);
					}
				}
			}
		}
	}
	
	/**
	 * This is essentially memory-less operation and can be used when the memory pressure is extremely high.
	 * @param n
	 * @param k
	 * @param params
	 * @throws DMLRuntimeException 
	 */
	private static void doLoopBasedConv2d(int n1, int n2, int k, ConvolutionParameters params) throws DMLRuntimeException {
		double [] inputArray = null;
		if (!params.input1.isInSparseFormat())
			inputArray = params.input1.getDenseBlock();
		double [] filterArray = null;
		if (!params.input2.isInSparseFormat())
			filterArray = params.input2.getDenseBlock();
		
		if(inputArray != null && filterArray != null) {
			doLoopBasedConv2dDenseDense(n1, n2, k, params, inputArray, filterArray);
		}
		else if(inputArray != null && filterArray == null) {
			for (int n = n1; n < n2; n++) 
				doLoopBasedConv2dDenseSparse(n, k, params, inputArray);
		}
		else if(inputArray == null && filterArray != null) {
			for (int n = n1; n < n2; n++)
				doLoopBasedConv2dSparseDense(n, k, params, filterArray);
		}
		else if(inputArray == null && filterArray == null) {
			for (int n = n1; n < n2; n++)
				doLoopBasedConv2dSparseSparse(n, k, params);
		}
	}
	
	private static int getMinPQ(int pad, int filterSize, int stride) {
		return Math.max(0, (int)Math.ceil(((double)(pad - filterSize))/stride));
	}
	
	private static int getMaxPQ(int pad, int filterSize, int stride, int outputSize, int inputSize) {
		return Math.min(outputSize, (int)Math.ceil(((double)(inputSize + pad - filterSize)) / stride));
	}
	
	private static double sparseConvMultiply(double filterVal, ConvolutionParameters params,
			int n, int c, int h, int w) {
		return params.input1.quickGetValue(n, c*params.H*params.W + h*params.W + w)*filterVal;
	}
	
	private static void doPoolingBackward(int n, int c, ConvolutionParameters params) {
		double [] inputArray = null;
		if (!params.input1.isInSparseFormat())
			inputArray = params.input1.getDenseBlock();
		double [] doutArray = null;
		if (!params.input2.isInSparseFormat())
			doutArray = params.input2.getDenseBlock();
		double [] outputArray = null;
		if (!params.output.isInSparseFormat())
			outputArray = params.output.getDenseBlock();
		
		for (int p = 0; p < params.P; p++) {
			for (int q = 0; q < params.Q; q++) {
				int start_index_h = p * params.stride_h - params.pad_h;
				int start_index_w = q * params.stride_w - params.pad_w;
				int end_index_h = Math.min(start_index_h + params.R, params.H);
				int end_index_w = Math.min(start_index_w + params.S, params.W);
				start_index_h = Math.max(start_index_h, 0);
				start_index_w = Math.max(start_index_w, 0);
				int maxIndex = n*params.C*params.H*params.W + c*params.H*params.W +  start_index_h*params.W + start_index_w; 
				double maxVal = -Double.MAX_VALUE; 


				double currDoutVal = -1;
				for (int h = start_index_h; h < end_index_h; h++) {
					for (int w = start_index_w; w < end_index_w; w++) {
						if(inputArray != null)
							currDoutVal = inputArray[n*params.C*params.H*params.W + c*params.H*params.W +  h*params.W + w];
						else
							currDoutVal = params.input1.quickGetValue(n, c*params.H*params.W + h*params.W + w);

						if(maxVal < currDoutVal) {
							maxIndex = n*params.C*params.H*params.W + c*params.H*params.W +  h*params.W + w;
							maxVal = currDoutVal;
						}
					}
				}

				double inVal = -1;
				if(doutArray != null)
					inVal = doutArray[n*params.C*params.P*params.Q + c*params.P*params.Q +  p * params.Q + q];
				else
					inVal = params.input2.quickGetValue(n, c*params.P*params.Q +  p * params.Q + q);

				// synchronized(this) {
					outputArray[maxIndex] += inVal;
				// }
			}
		}
	}

	public static void maxpooling(MatrixBlock input, MatrixBlock outputBlock, ConvolutionParameters params) throws DMLRuntimeException {
		params.input1 = input;
		params.output = outputBlock;
		
		if(input.getNumColumns() != params.C*params.H*params.W || input.getNumRows() != params.N) {
			throw new DMLRuntimeException("Incorrect input dimensions in maxpooling:" + input.getNumRows() + " " + input.getNumColumns() + " " + params.N + " " + params.K*params.P*params.Q);
		}
		
		params.outputNNZ.set(0);
		
		int constrainedNumThreads = OptimizerUtils.getConstrainedNumThreads(params.numThreads);
		if(!ALLOW_MULTI_THREADED_OPS || constrainedNumThreads <= 1) {
			for (int n = 0; n < params.N; n++) {
				for (int c = 0; c < params.C; c++) {
					doPooling(n, c, params);
				}
			}
		}
		else {
			runConvTask(constrainedNumThreads, params.C, TaskType.MaxPooling_Forward, params);
		}
		outputBlock.setNonZeros(params.outputNNZ.get());
	}

	private static void doPooling(int n, int c, ConvolutionParameters params) {
		double [] inputArray = null;
		if (!params.input1.isInSparseFormat())
			inputArray = params.input1.getDenseBlock();
		double [] outputArray = null;
		if (!params.output.isInSparseFormat())
			outputArray = params.output.getDenseBlock();
		
		long tmpNNZ = 0;
		for (int p = 0; p < params.P; p++) {
			for (int q = 0; q < params.Q; q++) {
				int start_index_h = p * params.stride_h - params.pad_h;
				int start_index_w = q * params.stride_w - params.pad_w;
				int end_index_h = Math.min(start_index_h + params.R, params.H);
				int end_index_w = Math.min(start_index_w + params.S, params.W);
				start_index_h = Math.max(start_index_h, 0);
				start_index_w = Math.max(start_index_w, 0);
				int out_index = n*params.C*params.P*params.Q + c*params.P*params.Q +  p * params.Q + q;
				outputArray[out_index] = -Double.MAX_VALUE;
				for (int h = start_index_h; h < end_index_h; h++) {
					for (int w = start_index_w; w < end_index_w; w++) {
						double inVal = -1;
						if(inputArray != null)
							inVal = inputArray[n*params.C*params.H*params.W + c*params.H*params.W +  h*params.W + w];
						else
							inVal = params.input1.quickGetValue(n, c*params.H*params.W +  h*params.W + w);
						outputArray[out_index] = Math.max(outputArray[out_index], inVal);
						if(outputArray[out_index] != 0)
							tmpNNZ++;
					}
				}
			}
		}
		params.outputNNZ.addAndGet(tmpNNZ);
	}
		
	// Reshape a 4D tensor of dimension (N, K, P, Q) to matrix of dimension (K, NPQ)
	public static void rotate180(MatrixBlock input, MatrixBlock outputBlock, ConvolutionParameters params) throws DMLRuntimeException {
		params.input1 = input;
		params.output = outputBlock;
		
		if(input.getNumColumns() != params.K*params.P*params.Q || input.getNumRows() != params.N) {
			throw new DMLRuntimeException("Incorrect input dimensions in rotate180:" + input.getNumRows() + " " + input.getNumColumns() + " " + params.N + " " + params.K*params.P*params.Q);
		}
		
		int constrainedNumThreads = OptimizerUtils.getConstrainedNumThreads(params.numThreads);
		if(!ALLOW_MULTI_THREADED_OPS || constrainedNumThreads <= 1) {
			for (int n = 0; n < params.N; n++) {
				doRotate180(n, params);
			}
		}
		else {
			runConvTask(constrainedNumThreads, 1, TaskType.Rotate180, params);
		}
		outputBlock.setNonZeros(input.getNonZeros()); // As number of non-zeros doesnot change for rotate180
	}
	
	private static void doRotate180(int n, ConvolutionParameters params) {
		double [] inputArray = null;
		if (!params.input1.isInSparseFormat())
			inputArray = params.input1.getDenseBlock();
		double [] outputArray = null;
		if (!params.output.isInSparseFormat())
			outputArray = params.output.getDenseBlock();
		
		for (int k = 0; k < params.K; k++) {
			for (int p = 0; p < params.P; p++) {
				for (int q = 0; q < params.Q; q++) {
					if(inputArray != null)
						outputArray[n*params.K*params.P*params.Q + p*params.Q*params.K + q*params.K + k] = inputArray[n*params.K*params.P*params.Q + k*params.P*params.Q + p*params.Q + q];
					else
						outputArray[n*params.P*params.Q*params.K + p*params.Q*params.K + q*params.K + k] = params.input1.quickGetValue(n, k*params.P*params.Q + p*params.Q + q);
				}
			}
		}
	}
	
	
	// Reshape a matrix of dimension (K, NPQ) to 4D tensor of dimension (N, K, P, params.Q)
	public static void reshape_col(MatrixBlock input, MatrixBlock outputBlock, ConvolutionParameters params) throws DMLRuntimeException {
		params.input1 = input;
		params.output = outputBlock;
		
		if(input.getNumColumns() != params.N*params.P*params.Q || input.getNumRows() != params.K) {
			throw new DMLRuntimeException("Incorrect input dimensions in reshape_col:" + input.getNumRows() + " " + input.getNumColumns());
		}
		
		int constrainedNumThreads = OptimizerUtils.getConstrainedNumThreads(params.numThreads);
		if(!ALLOW_MULTI_THREADED_OPS || constrainedNumThreads <= 1) {
			for (int n = 0; n < params.N; n++) { 
				doReshapeCol(n, params);
			}
		}
		else {
			runConvTask(constrainedNumThreads, 1, TaskType.ReshapeCol, params);
		}
		outputBlock.setNonZeros(input.getNonZeros()); // As number of non-zeros doesnot change for reshape_col
	}
	
	private static int [] getTaskSize(int constrainedNumThreads, int maxNumTaskSize1, int maxNumTaskSize2) {
		int taskSize1 = 1; int taskSize2 = 1;
		// Why this heuristics ? To reduce the impact of the thread-creation overhead in case of small tasks
		int approxNumTasksToCreate = 3*constrainedNumThreads;
		while((maxNumTaskSize1*maxNumTaskSize2)/(taskSize1*taskSize2) > approxNumTasksToCreate) {
			// Possibility of creating too many tasks, increase taskSize2
			taskSize2 *= 2;
			if(taskSize2 >= maxNumTaskSize2) {
				taskSize2 = maxNumTaskSize2;
				break;
			}
		}
		while((maxNumTaskSize1*maxNumTaskSize2)/(taskSize1*taskSize2) > approxNumTasksToCreate) {
			// Possibility of creating too many tasks, increase taskSize1
			taskSize1 *= 2;
			if(taskSize1 >= maxNumTaskSize1) {
				taskSize1 = maxNumTaskSize1;
				break;
			}
		}
		int [] ret = new int[2];
		ret[0] = taskSize1;
		ret[1] = taskSize2;
		return ret;
	}
	
	private static void runConvTask(int constrainedNumThreads, int Z, TaskType type, ConvolutionParameters params) throws DMLRuntimeException {
		if (params.isOutputThreadSafe() && constrainedNumThreads > 1) {
			runParallelConvTask(constrainedNumThreads, Z, type, params);
		} else {
			runSequentialConvTask(Z, type, params);
		}
	}
	
	private static void runSequentialConvTask(int Z, TaskType type, ConvolutionParameters params) throws DMLRuntimeException {
		ConvTask task = new ConvTask(0, params.N, 0, Z, type, params);
		try {
			task.call();
		} catch (Exception e) {
			throw new DMLRuntimeException("Error while executing single-threaded " + type.name(), e);
		}
	}
	
	private static void runParallelConvTask(int constrainedNumThreads, int Z, TaskType type,
			ConvolutionParameters params) throws DMLRuntimeException {
		ArrayList<ConvTask> tasks = new ArrayList<ConvTask>();
		int [] taskSizes = getTaskSize(constrainedNumThreads, params.N, Z);
		for (int n = 0; n < params.N; n += taskSizes[0]) {
			for (int z = 0; z < Z; z += taskSizes[1]) {
				tasks.add(new ConvTask(n, Math.min(params.N, n+taskSizes[0]), z, Math.min(Z, z+taskSizes[1]), type, params));
			}
		}
		LOG.debug("Reduce number of tasks from " + (params.N*Z)  + "(" + params.N + "," + Z + ") to " + tasks.size());

		ExecutorService pool = Executors.newFixedThreadPool( Math.min(constrainedNumThreads, tasks.size()) );
		List<Future<Object>> taskret;
		try {
			taskret = pool.invokeAll(tasks);
			pool.shutdown();
			for( Future<Object> task : taskret )
				task.get();
		} catch (InterruptedException e) {
			throw new DMLRuntimeException("Error while executing multi-threaded " + type.name(), e);
		} catch (ExecutionException e) {
			throw new DMLRuntimeException("Error while executing multi-threaded " + type.name(), e);
		}
	}
	
	private static class ConvTask implements Callable<Object> {
		int n1; int n2; int z1; int z2; 
		ConvolutionParameters params;
		TaskType type;
		public ConvTask(int n1, int n2, int z1, int z2, TaskType type, ConvolutionParameters params) {
			this.n1 = n1;
			this.n2 = n2;
			this.z1 = z1;
			this.z2 = z2;
			this.type = type;
			this.params = params;
		}
		
		@Override
		public Object call() throws DMLRuntimeException {
			switch(type) {
				case ReshapeCol:
					for (int n = n1; n < n2; n++) {
						LibMatrixDNN.doReshapeCol(n, params);
					}
					break;
				case Rotate180:
					for (int n = n1; n < n2; n++) {
						LibMatrixDNN.doRotate180(n, params);
					}
					break;
				case Im2Col:
					for (int n = n1; n < n2; n++) {
						for (int z = z1; z < z2; z++) {
							LibMatrixDNN.doIm2colOverInputPath_NCHW(n, z, params);
						}
					}
					break;
				case Col2Im:
					for (int n = n1; n < n2; n++) {
						for (int z = z1; z < z2; z++) {
							LibMatrixDNN.doCol2imOverInputPath_NCHW(n, z, params);
						}
					}
					break;
				case MaxPooling_Forward:
					for (int n = n1; n < n2; n++) {
						for (int z = z1; z < z2; z++) {
							LibMatrixDNN.doPooling(n, z, params);
						}
					}
					break;
				case MaxPooling_Backward:
					for (int n = n1; n < n2; n++) {
						for (int z = z1; z < z2; z++) {
							LibMatrixDNN.doPoolingBackward(n, z, params);
						}
					}
					break;
				case LoopBasedConv2d:
					for (int z = z1; z < z2; z++) {
						LibMatrixDNN.doLoopBasedConv2d(n1, n2, z, params);
					}
					break;
				default:
					throw new DMLRuntimeException("Unsupported ConvTask:" + type.name());
			}
			return null;
		}
	}
		
	private static void doReshapeCol(int n, ConvolutionParameters params) {
		double [] inputArray = null;
		if (!params.input1.isInSparseFormat())
			inputArray = params.input1.getDenseBlock();
		double [] outputArray = null;
		if (!params.output.isInSparseFormat())
			outputArray = params.output.getDenseBlock();
		
		if(inputArray != null) {
			for (int k = 0; k < params.K; k++)  {
				System.arraycopy(inputArray, k*params.N*params.P*params.Q + n*params.P*params.Q, outputArray, n*params.K*params.P*params.Q + k*params.P*params.Q, params.P*params.Q);
			}
		}
		else {
			for (int k = 0; k < params.K; k++) {
				for (int p = 0; p < params.P; p++) { 
					for (int q = 0; q < params.Q; q++) {
						outputArray[n*params.K*params.P*params.Q + k*params.P*params.Q + p*params.Q + q] = params.input1.quickGetValue(k, n*params.P*params.Q + p*params.Q + q);
					}
				}
			}
		}
	}
	
	// Converts a 4D tensor (N, C, R, S) to a matrix of dimension (CRS, NPQ)
	public static void im2col(MatrixBlock input, MatrixBlock outputBlock, ConvolutionParameters params) throws DMLRuntimeException {
		params.input1 = input;
		params.output = outputBlock;
		
		params.outputNNZ.set(0);
		
		if(DMLScript.STATISTICS) {
			if(input.isInSparseFormat()) {
				im2colSparseCount.addAndGet(1);
			}
			else {
				im2colDenseCount.addAndGet(1);
			}
		}
		
		int constrainedNumThreads = OptimizerUtils.getConstrainedNumThreads(params.numThreads);
		if(!ALLOW_MULTI_THREADED_OPS || constrainedNumThreads <= 1) {
			for (int n = 0; n < params.N; n++) { // Do following for all images
				for (int c = 0; c < params.C; c++) { // Since format is NCHW
					doIm2colOverInputPath_NCHW(n, c, params);
				}
			}
		}
		else {
			runConvTask(constrainedNumThreads, params.C, TaskType.Im2Col, params);
		}
		outputBlock.setNonZeros(params.outputNNZ.get());
	}
	
	// Converts a matrix of dimension (CRS, NPQ) to a 4D tensor (N, C, H, W)
	public static void col2im(MatrixBlock input, MatrixBlock outputBlock, ConvolutionParameters params) throws DMLRuntimeException {
		params.input1 = input;
		params.output = outputBlock;
		
		int constrainedNumThreads = OptimizerUtils.getConstrainedNumThreads(params.numThreads);
		if(!ALLOW_MULTI_THREADED_OPS || constrainedNumThreads <= 1) {
			// Sequential col2im
			for (int n = 0; n < params.N; n++) { // Do following for all images
				for (int c = 0; c < params.C; c++) { // Since format is NCHW
					doCol2imOverInputPath_NCHW(n, c, params);
				}
			}
		}
		else {
			// Parallel col2im
			runConvTask(constrainedNumThreads, params.C, TaskType.Col2Im, params);
		}
	}
	
		
	private static void doCol2imOverInputPath_NCHW(int n, int c, ConvolutionParameters params) {
		double [] inputArray = null;
		if (!params.input1.isInSparseFormat())
			inputArray = params.input1.getDenseBlock();
		double [] outputArray = null;
		if (!params.output.isInSparseFormat())
			outputArray = params.output.getDenseBlock();
		
		for (int r = 0; r < params.R; r++) { // Get an input patch of size R X S
			for (int s = 0; s < params.S; s++) {
				int localIndex = ((c*params.R*params.S*params.N + r*params.S*params.N + s*params.N + n)*params.P*params.Q);
				
				int input_row = r - params.pad_h;
				// And copy it to outputArray[i] (taking care of padding & striding)
				for (int p = params.P; p > 0; p--) {
					if (input_row >= 0 && input_row < params.H) {
						int input_col = s - params.pad_w;
						for (int q = params.Q; q > 0; q--, localIndex++) {
							if (input_col >= 0 && input_col < params.W) {
								// Copy from [channel c, height input_row, width input_col]
								int index = n*params.C*params.H*params.W + c*params.H*params.W + input_row*params.W + input_col;
								if (inputArray != null) {
									outputArray[index] += inputArray[localIndex];
								}
								else {
									// TODO: Optimize for sparse input
									// Note: localIndex = row*N*P*Q + col
									int row = localIndex / (params.N*params.P*params.Q);
									int col = localIndex % (params.N*params.P*params.Q);
									outputArray[index] += params.input1.quickGetValue(row, col); 
								}
							}
							input_col += params.stride_w;
						}
					} else {
						localIndex += params.Q;
					}
					input_row += params.stride_h;
				}
			}
		}
		
	}
	
	
	private static void doIm2colOverInputPath_NCHW(int n, int c, ConvolutionParameters params) {
		double [] inputArray = null;
		if (!params.input1.isInSparseFormat())
			inputArray = params.input1.getDenseBlock();
		double [] outputArray = null;
		if (!params.output.isInSparseFormat())
			outputArray = params.output.getDenseBlock();
		
		final int inputOffset = n*params.C*params.H*params.W + c*params.H*params.W;
		final int outputOffset = (c*params.R*params.S*params.N + n)*params.P*params.Q;
		
		long tmpNNZ = 0;
		for (int r = 0; r < params.R; r++) { // Get an input patch of size R X S
			for (int s = 0; s < params.S; s++) {
				int localIndex = outputOffset + ((r*params.S*params.N + s*params.N)*params.P*params.Q);
				
				int input_row = r - params.pad_h;
				// And copy it to outputArray[i] (taking care of padding & striding)
				for (int p = params.P; p > 0; p--) {
					if (input_row >= 0 && input_row < params.H) {
						int input_col = s - params.pad_w;
						for (int q = params.Q; q > 0; q--, localIndex++) {
							if (input_col >= 0 && input_col < params.W) {
								// Copy from [channel c, height input_row, width input_col]
								if(inputArray != null)
									outputArray[localIndex] = inputArray[inputOffset + input_row*params.W + input_col];
								else
									outputArray[localIndex] = params.input1.quickGetValue(n, c*params.H*params.W + input_row*params.W + input_col);
								if(outputArray[localIndex] != 0)
									tmpNNZ++;
							}
							else if(params.reuseNonZeroedOutput) {
								outputArray[localIndex] = 0;
							}
							input_col += params.stride_w;
						}
					} else {
						if(params.reuseNonZeroedOutput) {
							for(int i = localIndex; i < localIndex + params.Q; i++) {
								outputArray[localIndex] = 0;
							}
						}
						localIndex += params.Q;
					}
					input_row += params.stride_h;
				}
			}
		}
		
		params.outputNNZ.addAndGet(tmpNNZ);
	}
}