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
import java.util.Arrays;
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
	private static boolean warnedSingleThread = false;
	private static void warnSingleThreaded() {
		if(!warnedSingleThread) {
			throw new RuntimeException("WARN: Single thread execution in LibMatrixDNN");
			// LOG.warn("WARN: Single thread execution in LibMatrixDNN");
			// warnedSingleThread = true;
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
		ReshapeCol, Rotate180, Im2Col, Col2Im, MaxPooling_Forward, MaxPooling_Backward, 
		LoopedIm2ColConv2d, LoopedIm2ColConv2dBwdFilter, LoopedIm2ColConv2dBwdData
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
	private static AtomicLong loopedConvMatMultTime = new AtomicLong(0);
	private static AtomicLong loopedConvIm2ColTime = new AtomicLong(0);
	private static AtomicLong loopedConvBwdFilterMatMultTime = new AtomicLong(0);
	private static AtomicLong loopedConvBwdFilterIm2ColTime = new AtomicLong(0);
	private static AtomicLong loopedConvBwdDataMatMultTime = new AtomicLong(0);
	private static AtomicLong loopedConvBwdDataCol2ImTime = new AtomicLong(0);
	
	public static void appendStatistics(StringBuilder sb) {
		if(DMLScript.STATISTICS && (conv2dDenseCount.get() != 0 || conv2dSparseCount.get() != 0)) {
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
			if(loopedConvMatMultTime.get() != 0 || loopedConvIm2ColTime.get() != 0) {
				sb.append("LibMatrixDNN conv(im2col/matmult), bwdF (im2col/matmult), bwdD (col2im/matmult) time:\t" +
						String.format("%.3f", loopedConvIm2ColTime.get()*1e-9) + "/" +
						String.format("%.3f", loopedConvMatMultTime.get()*1e-9) + "/" + 
						String.format("%.3f", loopedConvBwdFilterIm2ColTime.get()*1e-9) + "/" +
						String.format("%.3f", loopedConvBwdFilterMatMultTime.get()*1e-9) + "/" +
						String.format("%.3f", loopedConvBwdDataCol2ImTime.get()*1e-9) + "/" +
						String.format("%.3f", loopedConvBwdDataMatMultTime.get()*1e-9) + " sec.\n");
			}
		}
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
		
		loopedConvIm2ColTime.set(0);
		loopedConvMatMultTime.set(0);
		loopedConvBwdFilterMatMultTime.set(0);
		loopedConvBwdFilterIm2ColTime.set(0);
		loopedConvBwdDataMatMultTime.set(0);
		loopedConvBwdDataCol2ImTime.set(0);
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
		
		public String toString() {
			return "(" + N + " " + C + " " + H + " " + W + " " + K + " " + R + " " + S + ")";  
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
	
	public static void conv2d_backward_data(MatrixBlock filter, MatrixBlock dout, MatrixBlock outputBlock, ConvolutionParameters params) throws DMLRuntimeException {
		params.input1 = filter;
		params.input2 = dout;
		params.output = outputBlock;
		if(filter.getNumRows() != params.K || filter.getNumColumns() != params.C*params.R*params.S || 
				dout.getNumRows() != params.N || dout.getNumColumns() != params.K*params.P*params.Q) {
			throw new DMLRuntimeException("Incorrect input to conv2d_backward_filter");
		}
		if(params.stride_h <= 0 || params.stride_w <= 0) {
			throw new DMLRuntimeException("Only positive strides supported");
		}
		
		if(DMLScript.STATISTICS) {
			if(filter.isInSparseFormat() || dout.isInSparseFormat()) {
				conv2dBwdDataSparseCount.addAndGet(1);
			}
			else {
				conv2dBwdDataDenseCount.addAndGet(1);
			}
		}
		
		params.reuseNonZeroedOutput = true;
		
		int constrainedNumThreads = OptimizerUtils.getConstrainedNumThreads(params.numThreads);
		if(!ALLOW_MULTI_THREADED_OPS || constrainedNumThreads <= 1) {
			warnSingleThreaded();
			MatrixBlock dout_reshaped = new MatrixBlock(params.P*params.Q, params.K, false);
			dout_reshaped.allocateDenseBlock(true);
			for (int n = 0; n < params.N; n++) {
				doLoopedIm2ColConv2dBwdData(n, dout_reshaped, params);
			}
		}
		else {
			runConvTask(constrainedNumThreads, 1, TaskType.LoopedIm2ColConv2dBwdData, params);
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
		
		params.reuseNonZeroedOutput = true;
		
		int constrainedNumThreads = OptimizerUtils.getConstrainedNumThreads(params.numThreads);
		if(!ALLOW_MULTI_THREADED_OPS || constrainedNumThreads <= 1) {
			warnSingleThreaded();
			MatrixBlock im2ColOutBlock = new MatrixBlock(params.C*params.R*params.S, params.P*params.Q, false);
			im2ColOutBlock.allocateDenseBlock(true);
			MatrixBlock dout_reshaped = new MatrixBlock(params.P*params.Q, params.K, false);
			dout_reshaped.allocateDenseBlock(true);
			for (int n = 0; n < params.N; n++) {
				params.output = doLoopedIm2ColConv2dBwdFilter(n, im2ColOutBlock, dout_reshaped, params.output, params);
			}
		}
		else {
			runConvTask(constrainedNumThreads, 1, TaskType.LoopedIm2ColConv2dBwdFilter, params);
		}
		
	}
	
	// ret += elem 
	private static void elementWiseInPlaceAddition(MatrixBlock ret, MatrixBlock elem) throws DMLRuntimeException {
		if(ret.getNumRows() != elem.getNumRows() || ret.getNumColumns() != elem.getNumColumns()) {
			throw new DMLRuntimeException("Incorrect dimensions");
		}
		if(!ret.isInSparseFormat() && !elem.isInSparseFormat()) {
			for(int i = 0; i < ret.getNumRows()*ret.getNumColumns(); i++) {
				ret.denseBlock[i] += elem.denseBlock[i];
			}
		}
		else if(!ret.isInSparseFormat() && elem.isInSparseFormat()) {
			Iterator<IJV> iter = elem.sparseBlock.getIterator();
			int numCol = ret.getNumColumns();
			while(iter.hasNext()) {
				IJV ijv = iter.next();
				int index = ijv.getI()*numCol + ijv.getJ();
				ret.denseBlock[index] += ijv.getV(); 
			}
		}
		else {
			throw new DMLRuntimeException("Sparse return format not supported");
		}
	}
	
	// ret += t(elem) 
	private static void elementWiseInPlaceTransposedAddition(MatrixBlock ret, MatrixBlock elem) throws DMLRuntimeException {
		if(ret.getNumRows() != elem.getNumColumns() || ret.getNumColumns() != elem.getNumRows()) {
			throw new DMLRuntimeException("Incorrect dimensions");
		}
		int numRow = ret.getNumColumns();
		if(!ret.isInSparseFormat() && !elem.isInSparseFormat()) {
			int iter = 0;
			for(int i = 0; i < elem.getNumRows(); i++) {
				for(int j = 0; j < elem.getNumColumns(); j++, iter++) {
					int index = j*numRow+i;
					ret.denseBlock[index] += elem.denseBlock[iter];
				}
			}
		}
		else if(!ret.isInSparseFormat() && elem.isInSparseFormat()) {
			Iterator<IJV> iter = elem.sparseBlock.getIterator();
			while(iter.hasNext()) {
				IJV ijv = iter.next();
				int index = ijv.getJ()*numRow + ijv.getI();
				ret.denseBlock[index] += ijv.getV(); 
			}
		}
		else {
			throw new DMLRuntimeException("Sparse return format not supported");
		}
	}
	
	private static void doLoopedIm2ColConv2dBwdData(int n, MatrixBlock dout_reshaped, ConvolutionParameters params) throws DMLRuntimeException {
		MatrixBlock filter = params.input1;
		MatrixBlock dout = params.input2;
		doRotate180(n, 0, dout, dout_reshaped.denseBlock, params, true);
		dout_reshaped.recomputeNonZeros();
		
		MatrixBlock temp = new MatrixBlock(params.P*params.Q, params.C*params.R*params.S, false);
		long t1 = DMLScript.STATISTICS ? System.nanoTime() : 0;
		LibMatrixMult.matrixMult(dout_reshaped, filter, temp);
		long t2 = DMLScript.STATISTICS ? System.nanoTime() : 0 ;
		doCol2imOverSingleImage(n, temp, params);
		long t3 = DMLScript.STATISTICS ? System.nanoTime() : 0 ;
		if(DMLScript.STATISTICS) {
			loopedConvBwdDataMatMultTime.addAndGet(t2-t1);
			loopedConvBwdDataCol2ImTime.addAndGet(t3-t2);
		}
	}
	
	private static MatrixBlock doLoopedIm2ColConv2dBwdFilter(int n, 
			MatrixBlock im2ColOutBlock, MatrixBlock dout_reshaped, MatrixBlock partialRetBlock, ConvolutionParameters params) throws DMLRuntimeException {
		long nnz = 0;
		long t1 = DMLScript.STATISTICS ? System.nanoTime() : 0;
		for (int c = 0; c < params.C; c++) {
			nnz += doIm2colOverInputPath_NCHW(n, c, im2ColOutBlock, params);
		}
		long t2 = DMLScript.STATISTICS ? System.nanoTime() : 0 ;
		im2ColOutBlock.setNonZeros(nnz);
		
		doRotate180(n, 0, params.input2, dout_reshaped.denseBlock, params, true);
		dout_reshaped.recomputeNonZeros();
		
		MatrixBlock temp = new MatrixBlock(params.C*params.R*params.S, params.K, false);
		long t3 = DMLScript.STATISTICS ? System.nanoTime() : 0 ;
		LibMatrixMult.matrixMult(im2ColOutBlock, dout_reshaped, temp);
		long t4 = DMLScript.STATISTICS ? System.nanoTime() : 0 ;
		if(DMLScript.STATISTICS) {
			loopedConvBwdFilterMatMultTime.addAndGet(t4-t3);
			loopedConvBwdFilterIm2ColTime.addAndGet(t2-t1);
		}
		
		elementWiseInPlaceTransposedAddition(partialRetBlock, temp);
		return partialRetBlock;
	}
	
	private static void computeTensorIndexes(int j, int [] ret, int H, int W) throws DMLRuntimeException {
		ret[0] = j / (H*W);
		ret[1] = (j - ret[0]*(H*W))/W;
		ret[2] = j % W;
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
		
		params.reuseNonZeroedOutput = true;
		int constrainedNumThreads = OptimizerUtils.getConstrainedNumThreads(params.numThreads);
		if(!ALLOW_MULTI_THREADED_OPS || constrainedNumThreads <= 1) {
			warnSingleThreaded();
			MatrixBlock im2ColOutBlock = new MatrixBlock(params.C*params.R*params.S, params.P*params.Q, false);
			im2ColOutBlock.allocateDenseBlock(true);
			for (int n = 0; n < params.N; n++) {
				doLoopedIm2ColConv2d(n, im2ColOutBlock, params);
			}
		}
		else {
			runConvTask(constrainedNumThreads, 1, TaskType.LoopedIm2ColConv2d, params);
		}
	}
	
	private static void doLoopedIm2ColConv2d(int n, MatrixBlock im2ColOutBlock, ConvolutionParameters params) throws DMLRuntimeException {
		long nnz = 0;
		long t1 = DMLScript.STATISTICS ? System.nanoTime() : 0;
		for (int c = 0; c < params.C; c++) {
			nnz += doIm2colOverInputPath_NCHW(n, c, im2ColOutBlock, params);
		}
		long t2 = DMLScript.STATISTICS ? System.nanoTime() : 0;
		
		im2ColOutBlock.setNonZeros(nnz);
		MatrixBlock matMultOutBlock = new MatrixBlock(params.K, params.P*params.Q, false);
		LibMatrixMult.matrixMult(params.input2, im2ColOutBlock, matMultOutBlock);
		long t3 = DMLScript.STATISTICS ? System.nanoTime() : 0;
		
		if(DMLScript.STATISTICS) {
			loopedConvIm2ColTime.addAndGet(t2 - t1);
			loopedConvMatMultTime.addAndGet(t3 - t2);
		}
		
		if(matMultOutBlock.isInSparseFormat()) {
			Iterator<IJV> iter = matMultOutBlock.sparseBlock.getIterator();
			final int outOffset = n*params.K*params.P*params.Q;
			while(iter.hasNext()) {
				IJV ijv = iter.next();
				int k = ijv.getI();
				int p = ijv.getJ() / params.Q;
				int q = ijv.getJ() % params.Q;
				params.output.denseBlock[outOffset + k*params.P*params.Q + p*params.Q + q] = ijv.getV();
			}
		}
		else
			System.arraycopy(matMultOutBlock.denseBlock, 0, params.output.denseBlock, n*params.K*params.P*params.Q, params.K*params.P*params.Q);
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
		
		if (params.output.isInSparseFormat())
			throw new DMLRuntimeException("Sparse maxpooling_backward is not supported");

		int constrainedNumThreads = OptimizerUtils.getConstrainedNumThreads(params.numThreads);
		if(!ALLOW_MULTI_THREADED_OPS || constrainedNumThreads <= 1) {
			warnSingleThreaded();
			for (int n = 0; n < params.N; n++) {
				doPoolingBackward(n, params);
			}
		}
		else {
			runConvTask(constrainedNumThreads, 1, TaskType.MaxPooling_Backward, params);
		}
	}
	
	private static void doPoolingBackward(int n, ConvolutionParameters params) throws DMLRuntimeException {
		double [] inputArray = null;
		if (!params.input1.isInSparseFormat())
			inputArray = params.input1.getDenseBlock();
		double [] doutArray = null;
		if (!params.input2.isInSparseFormat())
			doutArray = params.input2.getDenseBlock();
		double [] outputArray = null;
		if (!params.output.isInSparseFormat())
			outputArray = params.output.getDenseBlock();
		else
			throw new DMLRuntimeException("Only dense output supported for pooling_backward");
			
		if(inputArray != null) {
			if(doutArray != null)
				doPoolingBackwardDenseDense(n, inputArray, doutArray, outputArray, params);
			else
				doPoolingBackwardDenseSparse(n, inputArray, params.input2, outputArray, params);
		}
		else {
			if(doutArray != null)
				doPoolingBackwardSparseDense(n, doutArray, outputArray, params);
			else
				doPoolingBackwardSparseSparse(n, outputArray, params);
		}
	}
	
	private static void doPoolingBackwardSparseDense(int n, double [] doutArray,  double [] outputArray, ConvolutionParameters params) throws DMLRuntimeException {
		if (!params.input1.isInSparseFormat())
			throw new DMLRuntimeException("Incorrect usage: Call optimized versions");
		
		for (int c = 0; c < params.C; c++) {
			for (int p = 0; p < params.P; p++) {
				for (int q = 0; q < params.Q; q++) {
					double inVal = doutArray[n*params.C*params.P*params.Q + c*params.P*params.Q +  p * params.Q + q];
					if(inVal != 0) {
						final int inputOffset = n*params.C*params.H*params.W + c*params.H*params.W;
						int start_index_h = p * params.stride_h - params.pad_h;
						final int end_index_h = Math.min(start_index_h + params.R, params.H);
						start_index_h = Math.max(start_index_h, 0);
						int maxIndex = getMaxIndexSparse(start_index_h, end_index_h, q, inputOffset, n, c, params.input1, params);
						outputArray[maxIndex] += inVal;
					}
				}
			}
		}
	}
	
	private static void doPoolingBackwardSparseSparse(int n, double [] outputArray, ConvolutionParameters params) throws DMLRuntimeException {
		if (!params.input1.isInSparseFormat())
			throw new DMLRuntimeException("Incorrect usage: Call optimized versions");
		
		Iterator<IJV> iter = params.input2.sparseBlock.getIterator(n, n+1);
		int [] tensorIndexes = new int[3];
		
		while(iter.hasNext()) {
			IJV ijv = iter.next();
			computeTensorIndexes(ijv.getJ(), tensorIndexes, params.P, params.Q);
			int c = tensorIndexes[0];
			int p = tensorIndexes[1];
			int q = tensorIndexes[2];
			
			final int inputOffset = n*params.C*params.H*params.W + c*params.H*params.W;
			int start_index_h = p * params.stride_h - params.pad_h;
			final int end_index_h = Math.min(start_index_h + params.R, params.H);
			start_index_h = Math.max(start_index_h, 0);
			int maxIndex = getMaxIndexSparse(start_index_h, end_index_h, q, inputOffset, n, c, params.input1, params);
			outputArray[maxIndex] += ijv.getV();
		}	
		
	}
	
	private static void doPoolingBackwardDenseSparse(int n, double [] inputArray, 
			MatrixBlock dout, double [] outputArray, ConvolutionParameters params) throws DMLRuntimeException {
		Iterator<IJV> iter = dout.sparseBlock.getIterator(n, n+1);
		int [] tensorIndexes = new int[3];
		
		while(iter.hasNext()) {
			IJV ijv = iter.next();
			computeTensorIndexes(ijv.getJ(), tensorIndexes, params.P, params.Q);
			int c = tensorIndexes[0];
			int p = tensorIndexes[1];
			int q = tensorIndexes[2];
			
			final int inputOffset = n*params.C*params.H*params.W + c*params.H*params.W;
			int start_index_h = p * params.stride_h - params.pad_h;
			final int end_index_h = Math.min(start_index_h + params.R, params.H);
			start_index_h = Math.max(start_index_h, 0);
			int maxIndex = getMaxIndex(start_index_h, end_index_h, q, inputOffset, inputArray, params);
			outputArray[maxIndex] += ijv.getV();
		}
	}
	
	private static void doPoolingBackwardDenseDense(int n, double [] inputArray, double [] doutArray, 
			double [] outputArray, ConvolutionParameters params) {
		for (int c = 0; c < params.C; c++) {
			final int inputOffset = n*params.C*params.H*params.W + c*params.H*params.W;
			final int outputOffset = n*params.C*params.P*params.Q + c*params.P*params.Q;
			
			for (int p = 0; p < params.P; p++) {
				int start_index_h = p * params.stride_h - params.pad_h;
				final int end_index_h = Math.min(start_index_h + params.R, params.H);
				start_index_h = Math.max(start_index_h, 0);
				
				for (int q = 0; q < params.Q; q++) {
					int maxIndex = getMaxIndex(start_index_h, end_index_h, q, inputOffset, inputArray, params);
					outputArray[maxIndex] += doutArray[outputOffset +  p * params.Q + q];
				}
			}
		}
	}
	
	private static int getMaxIndexSparse(int start_index_h, int end_index_h, 
			int q, int inputOffset, int n, int c, MatrixBlock input, ConvolutionParameters params) throws DMLRuntimeException {
		if(!input.isInSparseFormat())
			throw new DMLRuntimeException("Incorrect usage: Only sparse format supported");
		
		Iterator<IJV> iter = input.sparseBlock.getIterator(n, n+1);
		int [] tensorIndexes = new int[3];
		
		int start_index_w = Math.max(q * params.stride_w - params.pad_w, 0);
		int end_index_w = Math.min(start_index_w + params.S, params.W);
		start_index_w = Math.max(start_index_w, 0);
		
		int maxIndex = inputOffset +  start_index_h*params.W + start_index_w; 
		double maxVal = -Double.MAX_VALUE;

		// Find maxIndex
		double currDoutVal = -1;
		while(iter.hasNext()) {
			IJV ijv = iter.next();
			computeTensorIndexes(ijv.getJ(), tensorIndexes, params.H, params.W);
			if(c != tensorIndexes[0])
				continue;
			int h = tensorIndexes[1];
			int w = tensorIndexes[2];
			if(h >= start_index_h && h < end_index_h && w >= start_index_w && w < end_index_w) {
				currDoutVal = ijv.getV();
				if(maxVal < currDoutVal) {
					maxIndex = inputOffset +  h*params.W + w;
					maxVal = currDoutVal;
				}
			}	
		}
		return maxIndex;
	}
	
	private static int getMaxIndex(int start_index_h, int end_index_h, 
			int q, int inputOffset, double [] inputArray, ConvolutionParameters params) {
		int start_index_w = q * params.stride_w - params.pad_w;
		int end_index_w = Math.min(start_index_w + params.S, params.W);
		start_index_w = Math.max(start_index_w, 0);
		
		int maxIndex = inputOffset +  start_index_h*params.W + start_index_w; 
		double maxVal = -Double.MAX_VALUE;

		// Find maxIndex
		double currDoutVal = -1;
		for (int h = start_index_h; h < end_index_h; h++) {
			for (int w = start_index_w; w < end_index_w; w++) {
				currDoutVal = inputArray[inputOffset +  h*params.W + w];
				if(maxVal < currDoutVal) {
					maxIndex = inputOffset +  h*params.W + w;
					maxVal = currDoutVal;
				}
			}
		}
		return maxIndex;
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
			warnSingleThreaded();
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
		
	// Reshape a 4D tensor of dimension (N, K, P, Q) to matrix of dimension (NPQ, K)
	public static void rotate180(MatrixBlock input, MatrixBlock outputBlock, ConvolutionParameters params) throws DMLRuntimeException {
		params.input1 = input;
		params.output = outputBlock;
		
		if(input.getNumColumns() != params.K*params.P*params.Q || input.getNumRows() != params.N) {
			throw new DMLRuntimeException("Incorrect input dimensions in rotate180:" + input.getNumRows() + " " + input.getNumColumns() + " " + params.N + " " + params.K*params.P*params.Q);
		}
		
		int constrainedNumThreads = OptimizerUtils.getConstrainedNumThreads(params.numThreads);
		if(!ALLOW_MULTI_THREADED_OPS || constrainedNumThreads <= 1) {
			warnSingleThreaded();
			for (int n = 0; n < params.N; n++) {
				doRotate180(n, params);
			}
		}
		else {
			runConvTask(constrainedNumThreads, 1, TaskType.Rotate180, params);
		}
		outputBlock.setNonZeros(input.getNonZeros()); // As number of non-zeros doesnot change for rotate180
	}
	
	private static void doRotate180(int n, ConvolutionParameters params) throws DMLRuntimeException {
		double [] outputArray = null;
		if (!params.output.isInSparseFormat())
			outputArray = params.output.getDenseBlock();
		else
			throw new DMLRuntimeException("Sparse output is not supported for rotate180");
		doRotate180(n, n, params.input1, outputArray, params, false);
	}
	
	private static void doRotate180(int inputN, int outputN, MatrixBlock input, 
			double [] outputArray,  ConvolutionParameters params, boolean zeroOutSparseOutput) throws DMLRuntimeException {
		double [] inputArray = null;
		if (!input.isInSparseFormat())
			inputArray = input.getDenseBlock();
		if(outputArray == null)
			throw new DMLRuntimeException("Sparse output is not supported for rotate180");
		
		int outputOffset = outputN*params.K*params.P*params.Q;
		if(inputArray != null) {
			for (int k = 0; k < params.K; k++) {
				for (int p = 0; p < params.P; p++) {
					for (int q = 0; q < params.Q; q++) {
						outputArray[outputOffset + p*params.Q*params.K + q*params.K + k] = inputArray[inputN*params.K*params.P*params.Q + k*params.P*params.Q + p*params.Q + q];
					}
				}
			}
		}
		else {
			if(zeroOutSparseOutput)
				Arrays.fill(outputArray, 0);
			
			Iterator<IJV> iter = input.sparseBlock.getIterator(inputN, inputN+1);
			int [] tensorIndexes = new int[3];
			while(iter.hasNext()) {
				IJV ijv = iter.next();
				computeTensorIndexes(ijv.getJ(), tensorIndexes, params.P, params.Q);
				int k = tensorIndexes[0];
				int p = tensorIndexes[1];
				int q = tensorIndexes[2];
				outputArray[outputOffset + p*params.Q*params.K + q*params.K + k] = ijv.getV();
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
			warnSingleThreaded();
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
	
	private static void runSequentialConvTask(int NSize, int Z, TaskType type, ConvolutionParameters params) throws DMLRuntimeException {
		ConvTask task = new ConvTask(0, NSize, 0, Z, type, params);
		warnSingleThreaded();
		try {
			task.call();
		} catch (Exception e) {
			throw new DMLRuntimeException("Error while executing single-threaded " + type.name(), e);
		}
	}
	
	private static void runConvTask(int constrainedNumThreads, int Z, TaskType type, ConvolutionParameters params) throws DMLRuntimeException {
		if (params.isOutputThreadSafe() && constrainedNumThreads > 1)
			runParallelConvTask(constrainedNumThreads, params.N, Z, type, params);
		else
			runSequentialConvTask(params.N, Z, type, params);
	}
	
	private static void runParallelConvTask(int constrainedNumThreads, int NSize, int Z, TaskType type, ConvolutionParameters params) throws DMLRuntimeException {
		ArrayList<ConvTask> tasks = new ArrayList<ConvTask>();
		if(NSize >= constrainedNumThreads || Z == 1) {
			int numNTasks = (int) Math.ceil(((double) NSize) / constrainedNumThreads);
			for (int n = 0; n < NSize; n += numNTasks) {
				tasks.add(new ConvTask(n, Math.min(NSize, n+numNTasks), 0, Z, type, params));
			}
		}
		else {
			int [] taskSizes = getTaskSize(constrainedNumThreads, NSize, Z);
			for (int n = 0; n < NSize; n += taskSizes[0]) {
				for (int z = 0; z < Z; z += taskSizes[1]) {
					tasks.add(new ConvTask(n, Math.min(NSize, n+taskSizes[0]), z, Math.min(Z, z+taskSizes[1]), type, params));
				}
			}
			LOG.debug("Reduce number of tasks from " + (NSize*Z)  + "(" + NSize + "," + Z + ") to " + tasks.size());
		}

		ExecutorService pool = Executors.newFixedThreadPool( Math.min(constrainedNumThreads, tasks.size()) );
		List<Future<Object>> taskret;
		try {
			taskret = pool.invokeAll(tasks);
			pool.shutdown();
			for( Future<Object> task : taskret ) {
				switch(type) {
					case LoopedIm2ColConv2dBwdFilter:
						elementWiseInPlaceAddition(params.output, (MatrixBlock) task.get());
						break;
					default:
						task.get();
				}
			}
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
						doReshapeCol(n, params);
					}
					break;
				case Rotate180:
					for (int n = n1; n < n2; n++) {
						doRotate180(n, params);
					}
					break;
				case Im2Col:
					long nnz = 0;
					for (int n = n1; n < n2; n++) {
						for (int z = z1; z < z2; z++) {
							nnz += doIm2colOverInputPath_NCHW(n, z, params);
						}
					}
					params.outputNNZ.addAndGet(nnz);
					break;
				case Col2Im:
					for (int n = n1; n < n2; n++) {
						doCol2imOverMultipleImages(n, params);
					}
					break;
				case MaxPooling_Forward:
					for (int n = n1; n < n2; n++) {
						for (int z = z1; z < z2; z++) {
							doPooling(n, z, params);
						}
					}
					break;
				case MaxPooling_Backward:
					for (int n = n1; n < n2; n++) {
						doPoolingBackward(n, params);
					}
					break;
				case LoopedIm2ColConv2d:
					MatrixBlock im2ColOutBlock = new MatrixBlock(params.C*params.R*params.S, params.P*params.Q, false);
					im2ColOutBlock.allocateDenseBlock(true);
					for (int n = n1; n < n2; n++) {
						doLoopedIm2ColConv2d(n, im2ColOutBlock, params);
					}
					break;
				case LoopedIm2ColConv2dBwdFilter:
				{
					MatrixBlock im2ColOutBlock1 = new MatrixBlock(params.C*params.R*params.S, params.P*params.Q, false);
					im2ColOutBlock1.allocateDenseBlock(true);
					MatrixBlock partialRetBlock = new MatrixBlock(params.K, params.C*params.R*params.S, false);
					partialRetBlock.allocateDenseBlock(true);
					MatrixBlock dout_reshaped = new MatrixBlock(params.P*params.Q, params.K, false);
					dout_reshaped.allocateDenseBlock(true);
					for (int n = n1; n < n2; n++) {
						partialRetBlock = doLoopedIm2ColConv2dBwdFilter(n, im2ColOutBlock1, dout_reshaped, partialRetBlock, params);
					}
					return partialRetBlock;
				}
				case LoopedIm2ColConv2dBwdData:
				{
					MatrixBlock dout_reshaped = new MatrixBlock(params.P*params.Q, params.K, false);
					dout_reshaped.allocateDenseBlock(true);
					for (int n = n1; n < n2; n++) {
						doLoopedIm2ColConv2dBwdData(n, dout_reshaped, params);
					}
					break;
				}
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
			warnSingleThreaded();
			long nnz = 0;
			for (int n = 0; n < params.N; n++) { // Do following for all images
				for (int c = 0; c < params.C; c++) { // Since format is NCHW
					nnz += doIm2colOverInputPath_NCHW(n, c, params);
				}
			}
			outputBlock.setNonZeros(nnz);
		}
		else {
			runConvTask(constrainedNumThreads, params.C, TaskType.Im2Col, params);
			outputBlock.setNonZeros(params.outputNNZ.get());
		}
		
	}
	
	// Converts a matrix of dimension (CRS, NPQ) to a 4D tensor (N, C, H, W)
	public static void col2im(MatrixBlock input, MatrixBlock outputBlock, ConvolutionParameters params) throws DMLRuntimeException {
		params.input1 = input;
		params.output = outputBlock;
		
		int constrainedNumThreads = OptimizerUtils.getConstrainedNumThreads(params.numThreads);
		if(!ALLOW_MULTI_THREADED_OPS || constrainedNumThreads <= 1) {
			warnSingleThreaded();
			// Sequential col2im
			for (int n = 0; n < params.N; n++) { // Do following for all images
				doCol2imOverMultipleImages(n, params);
			}
		}
		else {
			// Parallel col2im
			runConvTask(constrainedNumThreads, 1, TaskType.Col2Im, params);
		}
	}
	
	
	// Converts input: PQ X CRS matrix and writes to 1 X CHW
	private static void doCol2imOverSingleImage(int outputN, MatrixBlock input, ConvolutionParameters params) throws DMLRuntimeException {
		if(input.rlen != params.P*params.Q || input.clen != params.C*params.R*params.S) {
			throw new DMLRuntimeException("Incorrect input dimensions");
		}
		
		double [] outputArray = null;
		if (!params.output.isInSparseFormat())
			outputArray = params.output.getDenseBlock();
		else {
			throw new DMLRuntimeException("Only dense output is implemented");
		}
		
		if(!input.isInSparseFormat()) {
			double [] inputArray = input.getDenseBlock();
			doCol2IMDenseInput(0, outputN, inputArray, outputArray, params);
		}
		else {
			doCol2IMSparseInput(0, outputN, input.getSparseBlockIterator(), outputArray, params);
		}
	}
	
	private static void doCol2IMSparseInput(int inputN, int outputN, Iterator<IJV> inputIter, double [] outputArray, ConvolutionParameters params) throws DMLRuntimeException {
		int [] tensorIndexes = new int[3];
		
		while(inputIter.hasNext()) {
			IJV ijv = inputIter.next();
			computeTensorIndexes(ijv.getJ(), tensorIndexes, params.R, params.S);
			int c = tensorIndexes[0];
			int r = tensorIndexes[1];
			int s = tensorIndexes[2];
			computeTensorIndexes(ijv.getI(), tensorIndexes, params.P, params.Q);
			int p = tensorIndexes[1];
			int q = tensorIndexes[2];
			if(inputN != tensorIndexes[0]) {
				throw new DMLRuntimeException("Incorrect tensor indexes: " + inputN + " != " + tensorIndexes[0] + " <" + p + " " + q + " " + ijv.getI() + params.P + " " + params.Q + ">");
			}
			int h = p*params.stride_h + r - params.pad_h;
			int w = q*params.stride_w + s - params.pad_w;
			if(h >= 0 && h < params.H && w >= 0 && w < params.W) {
				int outIndex = outputN*params.C*params.H*params.W + c*params.H*params.W + h*params.W + w;
				outputArray[outIndex] += ijv.getV();
			}
		}
	}
	
	// Converts input: PQ X CRS matrix and writes to 1 X CHW if inputN == 0
	// Or converts input: NPQ X CRS matrix and writes to N X CHW 
	private static void doCol2IMDenseInput(int inputN, int outputN, double [] inputArray, double [] outputArray, ConvolutionParameters params) throws DMLRuntimeException {
		final int outputNOffset = outputN*params.C*params.H*params.W;
		for (int p = 0; p < params.P; p++) {
			// h = p*params.stride_h + r - params.pad_h
			//   = r + hOffset
			// Based on restrictions: h >= 0 and r >= 0 and h < params.H and r < params.R, we get
			// max(0, - hOffset) <= r < min(params.R, params.H - hOffset)
			final int hOffset = p*params.stride_h - params.pad_h;
			final int rStart = Math.max(0, - hOffset);
			final int rEnd = Math.min(params.R, params.H - hOffset);
			for (int q = 0; q < params.Q; q++) {
				// Using the same logic as above on following:
				// w = q*params.stride_w + s - params.pad_w
				final int wOffset = q*params.stride_w - params.pad_w;
				final int sStart = Math.max(0, - wOffset);
				final int sEnd = Math.min(params.S, params.W - wOffset);
				final int tempOffset = (inputN*params.P*params.Q + p*params.Q + q)*params.C*params.R*params.S;
				for (int c = 0; c < params.C; c++) {
					final int outOffset = outputNOffset + c*params.H*params.W;
					final int inputOffset = tempOffset + c*params.R*params.S;
					for (int r = rStart; r < rEnd; r++) {
						for (int s = sStart; s < sEnd; s++) {
							int inputIndex = inputOffset + r*params.S + s;
							int outIndex = outOffset + (hOffset + r)*params.W + wOffset + s;
							outputArray[outIndex] += inputArray[inputIndex];
						}
					}
				}
			}
		}
	}
		
	// NPQ X CRS
	private static void doCol2imOverMultipleImages(int n, ConvolutionParameters params) throws DMLRuntimeException {
		MatrixBlock input = params.input1;
		
		if(input.rlen != params.N*params.P*params.Q || input.clen != params.C*params.R*params.S) {
			throw new DMLRuntimeException("Incorrect input dimensions");
		}
		
		double [] outputArray = null;
		if (!params.output.isInSparseFormat())
			outputArray = params.output.getDenseBlock();
		else {
			throw new DMLRuntimeException("Only dense output is implemented");
		}
		
		if(!input.isInSparseFormat()) {
			double [] inputArray = input.getDenseBlock();
			doCol2IMDenseInput(n, n, inputArray, outputArray, params);
		}
		else {
			doCol2IMSparseInput(n, n, input.getSparseBlockIterator(n*params.P*params.Q, (n+1)*params.P*params.Q), outputArray, params);
		}
	}
	
	private static long doIm2colOverInputPath_NCHW(int n, int c, ConvolutionParameters params) throws DMLRuntimeException {
		return doIm2colOverInputPath_NCHW(n, c, null, params);
	}
	
	private static long doIm2colOverInputPath_NCHW(int n, int c, MatrixBlock output, ConvolutionParameters params) throws DMLRuntimeException {
		double [] inputArray = null;
		if (!params.input1.isInSparseFormat())
			inputArray = params.input1.getDenseBlock();
		double [] outputArray = null;
		if(output == null && !params.output.isInSparseFormat())
			outputArray = params.output.getDenseBlock();
		else if(output != null && !output.isInSparseFormat())
			outputArray = output.getDenseBlock();
		else {
			throw new DMLRuntimeException("Sparse output is not supported for im2col");
		}
		
		final int inputOffset = n*params.C*params.H*params.W + c*params.H*params.W;
		int outputOffset;
		if(output == null)
			outputOffset = (c*params.R*params.S*params.N + n)*params.P*params.Q;
		else
			outputOffset = (c*params.R*params.S)*params.P*params.Q;
		
		long tmpNNZ = 0;
		for (int r = 0; r < params.R; r++) { // Get an input patch of size R X S
			for (int s = 0; s < params.S; s++) {
				int localIndex;
				if(output == null)
					localIndex = outputOffset + ((r*params.S*params.N + s*params.N)*params.P*params.Q);
				else
					localIndex = outputOffset + ((r*params.S + s)*params.P*params.Q);
				
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
		
		return tmpNNZ;
	}
}
