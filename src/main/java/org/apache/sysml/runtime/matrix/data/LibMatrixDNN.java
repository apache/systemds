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
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

import org.apache.sysml.hops.OptimizerUtils;
import org.apache.sysml.runtime.DMLRuntimeException;
import org.apache.sysml.runtime.util.ConvolutionUtils;

public class LibMatrixDNN {

	public static boolean ALLOW_MULTI_THREADED_OPS = true;
	enum TaskType {
		ReshapeCol, Rotate180, Im2Col, Col2Im, MaxPooling_Forward, MaxPooling_Backward
	}
	public static final int TASK_SIZE = 64; // to take care of extremely small tasks
	
	public static class ConvolutionParameters {
		public int N; public int C; public int H; public int W;
		public int K; public int R; public int S; public int stride_h; public int stride_w; public int pad_h; public int pad_w;
		public int P; public int Q;
		
		MatrixBlock input1; MatrixBlock input2; MatrixBlock output;
		boolean reuseNonZeroedOutput = false;
		
		public ConvolutionParameters(int N, int C, int H, int W,
			int K, int R, int S, int stride_h, int stride_w, int pad_h, int pad_w) {
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
		}
		
		public void setReuseNonZeroedOutput(boolean reuseNonZeroedOutput) {
			this.reuseNonZeroedOutput = reuseNonZeroedOutput;
		}
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

		int constrainedNumThreads = OptimizerUtils.getConstrainedNumThreads(-1);
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
				double maxVal = Double.MIN_VALUE; 


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
		
		int constrainedNumThreads = OptimizerUtils.getConstrainedNumThreads(-1);
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
	}

	private static void doPooling(int n, int c, ConvolutionParameters params) {
		double [] inputArray = null;
		if (!params.input1.isInSparseFormat())
			inputArray = params.input1.getDenseBlock();
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
				int out_index = n*params.C*params.P*params.Q + c*params.P*params.Q +  p * params.Q + q;
				outputArray[out_index] = 0;
				for (int h = start_index_h; h < end_index_h; h++) {
					for (int w = start_index_w; w < end_index_w; w++) {
						double inVal = -1;
						if(inputArray != null)
							inVal = inputArray[n*params.C*params.H*params.W + c*params.H*params.W +  h*params.W + w];
						else
							inVal = params.input1.quickGetValue(n, c*params.H*params.W +  h*params.W + w);
						outputArray[out_index] = Math.max(outputArray[out_index], inVal);
					}
				}
			}
		}
	}
		
	// Reshape a 4D tensor of dimension (N, K, P, Q) to matrix of dimension (K, NPQ)
	public static void rotate180(MatrixBlock input, MatrixBlock outputBlock, ConvolutionParameters params) throws DMLRuntimeException {
		params.input1 = input;
		params.output = outputBlock;
		
		if(input.getNumColumns() != params.K*params.P*params.Q || input.getNumRows() != params.N) {
			throw new DMLRuntimeException("Incorrect input dimensions in rotate180:" + input.getNumRows() + " " + input.getNumColumns() + " " + params.N + " " + params.K*params.P*params.Q);
		}
		
		int constrainedNumThreads = OptimizerUtils.getConstrainedNumThreads(-1);
		if(!ALLOW_MULTI_THREADED_OPS || constrainedNumThreads <= 1) {
			for (int n = 0; n < params.N; n++) {
				doRotate180(n, params);
			}
		}
		else {
			runParallelConvTask(constrainedNumThreads, 1, TaskType.Rotate180, params);
		}
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
		
		int constrainedNumThreads = OptimizerUtils.getConstrainedNumThreads(-1);
		if(!ALLOW_MULTI_THREADED_OPS || constrainedNumThreads <= 1) {
			for (int n = 0; n < params.N; n++) { 
				doReshapeCol(n, params);
			}
		}
		else {
			runParallelConvTask(constrainedNumThreads, 1, TaskType.ReshapeCol, params);
		}
		
	}
	
	private static void runParallelConvTask(int constrainedNumThreads, int Z, TaskType type, ConvolutionParameters params) throws DMLRuntimeException {
		ArrayList<ConvTask> tasks = new ArrayList<ConvTask>();		
		
		// Total number of compute units available: constrainedNumThreads
		// Static task allocation. TODO: Do this in dynamic way
		for (int n = 0; n < params.N; n++) {
			for (int z = 0; z < Z; z += TASK_SIZE) {
				tasks.add(new ConvTask(n, n+1, z, Math.min(Z, z+TASK_SIZE), type, params));
			}
		}

		ExecutorService pool = Executors.newFixedThreadPool( Math.min(constrainedNumThreads, tasks.size()) );
		try {
			pool.invokeAll(tasks);
		} catch (InterruptedException e) {
			throw new DMLRuntimeException("Error while executing multi-threaded " + type.name(), e);
		}	
		pool.shutdown();
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
		public Object call() throws Exception {
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
				default:
					throw new RuntimeException("Unsupported ConvTask:" + type.name());
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
		
		int constrainedNumThreads = OptimizerUtils.getConstrainedNumThreads(-1);
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
	}
	
	// Converts a matrix of dimension (CRS, NPQ) to a 4D tensor (N, C, H, W)
	public static void col2im(MatrixBlock input, MatrixBlock outputBlock, ConvolutionParameters params) throws DMLRuntimeException {
		params.input1 = input;
		params.output = outputBlock;
		
		int constrainedNumThreads = OptimizerUtils.getConstrainedNumThreads(-1);
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
		
	}
}
