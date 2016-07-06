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
import java.util.List;
import java.util.concurrent.Callable;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.concurrent.atomic.AtomicLong;

import org.apache.sysml.hops.OptimizerUtils;
import org.apache.sysml.runtime.DMLRuntimeException;
import org.apache.sysml.runtime.util.ConvolutionUtils;


public class LibMatrixDNN {

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
	public static final int TASK_SIZE = 64; // to take care of extremely small tasks
	
	public static class TemporaryConvolutionData {
		public int [] minIndexArrR;
		public int [] minIndexArrS;
		public int [] maxIndexArrR;
		public int [] maxIndexArrS;
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
	}
	
	public static void conv2d_backward_filter(MatrixBlock input, MatrixBlock dout, MatrixBlock outputBlock, ConvolutionParameters params) throws DMLRuntimeException {
		params.input1 = input;
		params.input2 = dout;
		params.output = outputBlock;
		if(input.getNumRows() != params.N || input.getNumColumns() != params.C*params.H*params.W || 
				dout.getNumRows() != params.N || dout.getNumColumns() != params.K*params.P*params.Q) {
			throw new DMLRuntimeException("Incorrect input to conv2d_backward_filter");
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
	
	public static void doConv2d_Backward_Filter(int k, int c, int r, int s, ConvolutionParameters params) {
		double [] inputArray = null;
		if (!params.input1.isInSparseFormat())
			inputArray = params.input1.getDenseBlock();
		double [] doutArray = null;
		if (!params.input2.isInSparseFormat())
			doutArray = params.input2.getDenseBlock();
		double [] outputArray = params.output.getDenseBlock();
		
		long outputVal = 0;
		for (int n = 0; n < params.N; n++) {
			for (int p = 0; p < params.P; p++) {
				for (int q = 0; q < params.Q; q++) {
					double doutVal = 0;
					if(doutArray != null)
						doutVal = doutArray[n*params.K*params.P*params.Q + k*params.P*params.Q + p*params.Q + q];
					else
						doutVal = params.input2.quickGetValue(n, k*params.P*params.Q + p*params.Q + q);
					if(doutVal != 0) {
						// TODO: Improve the performance by striding
						for (int h = 0; h < params.H; h++) {
							for (int w = 0; w < params.W; w++) { 
								if(h == p*params.stride_h + r - params.pad_h &&
										w == q*params.stride_w + s - params.pad_w) {
									if(inputArray != null)
										outputVal += doutVal*inputArray[n*params.C*params.H*params.W + c*params.H*params.W + h*params.W+w];
									else 
										outputVal += doutVal*params.input1.quickGetValue(n, c*params.H*params.W + h*params.W + w);
								}
							}
						}
					}
				}
			}
		}
		outputArray[k*params.C*params.R*params.S + c*params.R*params.S + r*params.S + s] = outputVal;
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
		
		params.tmpData = new TemporaryConvolutionData();		
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
		
		int constrainedNumThreads = OptimizerUtils.getConstrainedNumThreads(params.numThreads);
		if(!ALLOW_MULTI_THREADED_OPS || constrainedNumThreads <= 1) {
			for (int n = 0; n < params.N; n++) {
				for (int k = 0; k < params.K; k++) {
					doLoopBasedConv2d(n, k, params);
				}
			}
		}
		else
			runParallelConvTask(constrainedNumThreads, params.K, TaskType.LoopBasedConv2d, params);
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

		int constrainedNumThreads = OptimizerUtils.getConstrainedNumThreads(params.numThreads);
		if(!ALLOW_MULTI_THREADED_OPS || constrainedNumThreads <= 1) {
			for (int n = 0; n < params.N; n++) {
				for (int c = 0; c < params.C; c++) {
					doPoolingBackward(n, c, params);
				}
			}
		}
		else {
			runParallelConvTask(constrainedNumThreads, params.C, TaskType.MaxPooling_Backward, params);
		}
	}
	
	/**
	 * This is essentially memory-less operation and can be used when the memory pressure is extremely high.
	 * @param n
	 * @param k
	 * @param params
	 */
	private static void doLoopBasedConv2d(int n, int k, ConvolutionParameters params) {
		double [] inputArray = null;
		if (!params.input1.isInSparseFormat())
			inputArray = params.input1.getDenseBlock();
		double [] filterArray = null;
		if (!params.input2.isInSparseFormat())
			filterArray = params.input2.getDenseBlock();
		double [] outputArray = params.output.getDenseBlock();
		
		int outputOffset = n*params.K*params.P*params.Q + k*params.P*params.Q;
		
		int [] minIndexArrR = params.tmpData.minIndexArrR;
		int [] maxIndexArrR = params.tmpData.maxIndexArrR;
		int [] minIndexArrS = params.tmpData.minIndexArrS;
		int [] maxIndexArrS = params.tmpData.maxIndexArrS;
		
		if(inputArray != null && filterArray != null) {
			for (int c = 0; c < params.C; c++) {
				for (int r = 0; r < params.R; r++) {
					int filterOffset = k*params.C*params.R*params.S + c*params.R*params.S + r*params.S;
					for (int p = minIndexArrR[r]; p < maxIndexArrR[r]; p++) {
						for (int s = 0; s < params.S; s++) {
							double filterVal = filterArray[filterOffset + s];
							if(filterVal != 0) {
								int h = p*params.stride_h + r - params.pad_h;
								for (int q = minIndexArrS[s]; q < maxIndexArrS[s]; q++) {
									int w = q*params.stride_w + s - params.pad_w;
									outputArray[outputOffset + p*params.Q + q] += denseConvMultiply(inputArray, filterVal, params, n, c, h, w);
								}
							}
						}
					}
				}
			}
		}
		else if(inputArray != null && filterArray == null) {
			for (int c = 0; c < params.C; c++) {
				for (int r = 0; r < params.R; r++) {
					for (int p = minIndexArrR[r]; p < maxIndexArrR[r]; p++) {
						for (int s = 0; s < params.S; s++) {
							double filterVal = params.input2.quickGetValue(k, c*params.R*params.S + r*params.S + s);
							if(filterVal != 0) {
								int h = p*params.stride_h + r - params.pad_h;
								for (int q = minIndexArrS[s]; q < maxIndexArrS[s]; q++) {
									int w = q*params.stride_w + s - params.pad_w;
									outputArray[outputOffset + p*params.Q + q] += denseConvMultiply(inputArray, filterVal, params, n, c, h, w);
								}
							}
						}
					}
				}
			}
		}
		else if(inputArray == null && filterArray != null) {
			for (int c = 0; c < params.C; c++) {
				for (int r = 0; r < params.R; r++) {
					int filterOffset = k*params.C*params.R*params.S + c*params.R*params.S + r*params.S;
					for (int p = minIndexArrR[r]; p < maxIndexArrR[r]; p++) {
						for (int s = 0; s < params.S; s++) {
							double filterVal = filterArray[filterOffset + s];
							if(filterVal != 0) {
								int h = p*params.stride_h + r - params.pad_h;
								for (int q = minIndexArrS[s]; q < maxIndexArrS[s]; q++) {
									int w = q*params.stride_w + s - params.pad_w;
									outputArray[outputOffset + p*params.Q + q] += sparseConvMultiply(inputArray, filterVal, params, n, c, h, w);
								}
							}
						}
					}
				}
			}
		}
		else if(inputArray == null && filterArray == null) {
			for (int c = 0; c < params.C; c++) {
				for (int r = 0; r < params.R; r++) {
					for (int p = minIndexArrR[r]; p < maxIndexArrR[r]; p++) {
						for (int s = 0; s < params.S; s++) {
							double filterVal = params.input2.quickGetValue(k, c*params.R*params.S + r*params.S + s);
							if(filterVal != 0) {
								int h = p*params.stride_h + r - params.pad_h;
								for (int q = minIndexArrS[s]; q < maxIndexArrS[s]; q++) {
									int w = q*params.stride_w + s - params.pad_w;
									outputArray[outputOffset + p*params.Q + q] += sparseConvMultiply(inputArray, filterVal, params, n, c, h, w);
								}
							}
						}
					}
				}
			}
		}
	}
	
	private static int getMinPQ(int pad, int filterSize, int stride) {
		return Math.max(0, (int)Math.ceil(((double)(pad - filterSize))/stride));
	}
	
	private static int getMaxPQ(int pad, int filterSize, int stride, int outputSize, int inputSize) {
		return Math.min(outputSize, (int)Math.ceil(((double)(inputSize + pad - filterSize)) / stride));
	}
	
	private static double denseConvMultiply(double [] inputArray, double filterVal, ConvolutionParameters params,
			int n, int c, int h, int w) {
		return inputArray[n*params.C*params.H*params.W + c*params.H*params.W + h*params.W+w]*filterVal;
	}
	
	private static double sparseConvMultiply(double [] inputArray, double filterVal, ConvolutionParameters params,
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
			runParallelConvTask(constrainedNumThreads, params.C, TaskType.MaxPooling_Forward, params);
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
			runParallelConvTask(constrainedNumThreads, 1, TaskType.Rotate180, params);
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
			runParallelConvTask(constrainedNumThreads, 1, TaskType.ReshapeCol, params);
		}
		outputBlock.setNonZeros(input.getNonZeros()); // As number of non-zeros doesnot change for reshape_col
	}
	
	private static void runParallelConvTask(int constrainedNumThreads, int Z, TaskType type, ConvolutionParameters params) throws DMLRuntimeException {
		// Total number of compute units available: constrainedNumThreads
		// Static task allocation. TODO: Do this in dynamic way
		int taskSize = TASK_SIZE;
		while(true) {
			if(params.N * Math.ceil(Z/taskSize) > constrainedNumThreads || taskSize == 1) {
				doRunParallelConvTask(constrainedNumThreads, Z, type, params, taskSize);
				return;
			}
			taskSize = Math.max(taskSize/2, 1);
		}
	}
	
	private static void doRunParallelConvTask(int constrainedNumThreads, int Z, TaskType type, ConvolutionParameters params, int taskSize) throws DMLRuntimeException {
		ArrayList<ConvTask> tasks = new ArrayList<ConvTask>();		
		
		for (int n = 0; n < params.N; n++) {
			for (int z = 0; z < Z; z += taskSize) {
				tasks.add(new ConvTask(n, n+1, z, Math.min(Z, z+taskSize), type, params));
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
					for (int n = n1; n < n2; n++) {
						for (int z = z1; z < z2; z++) {
							LibMatrixDNN.doLoopBasedConv2d(n, z, params);
						}
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
		int constrainedNumThreads = OptimizerUtils.getConstrainedNumThreads(params.numThreads);
		if(!ALLOW_MULTI_THREADED_OPS || constrainedNumThreads <= 1) {
			for (int n = 0; n < params.N; n++) { // Do following for all images
				for (int c = 0; c < params.C; c++) { // Since format is NCHW
					doIm2colOverInputPath_NCHW(n, c, params);
				}
			}
		}
		else {
			runParallelConvTask(constrainedNumThreads, params.C, TaskType.Im2Col, params);
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
			runParallelConvTask(constrainedNumThreads, params.C, TaskType.Col2Im, params);
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
