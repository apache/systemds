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

package org.apache.sysml.runtime.instructions.cp;

import java.util.ArrayList;
import java.util.concurrent.Callable;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

import org.apache.sysml.api.DMLScript;
import org.apache.sysml.hops.OptimizerUtils;
import org.apache.sysml.parser.Expression.DataType;
import org.apache.sysml.parser.Expression.ValueType;
import org.apache.sysml.runtime.DMLRuntimeException;
import org.apache.sysml.runtime.controlprogram.context.ExecutionContext;
import org.apache.sysml.runtime.functionobjects.SwapIndex;
import org.apache.sysml.runtime.instructions.InstructionUtils;
import org.apache.sysml.runtime.matrix.data.MatrixBlock;
import org.apache.sysml.runtime.matrix.operators.ReorgOperator;
import org.apache.sysml.runtime.util.ConvolutionUtils;
import org.apache.sysml.utils.Statistics;

public class ConvolutionCPInstruction extends UnaryCPInstruction {
	
	public static boolean ALLOW_MULTI_THREADED_OPS = true;
	
	private CPOperand _in2; // used for pooling backward
	private ArrayList<CPOperand> _input_shape;
	private ArrayList<CPOperand> _filter_shape;
	private ArrayList<CPOperand> _stride = new ArrayList<CPOperand>();
	private ArrayList<CPOperand> _padding = new ArrayList<CPOperand>();
	
	int N; int C; int H; int W;
	int K; int R; int S; int stride_h; int stride_w; int pad_h; int pad_w;
	int P; int Q;
	double[] outputArray; double[] inputArray; MatrixBlock input;
	double [] doutArray; MatrixBlock dout;
	long outNNZ = 0;
	
	boolean reuseNonZeroedOutput = false;
	
	enum TaskType {
		ReshapeCol, Rotate180, Im2Col, Col2Im, MaxPooling_Forward, MaxPooling_Backward
	}
	public static final int TASK_SIZE = 64; // to take care of extremely small tasks
	
	public ConvolutionCPInstruction(CPOperand in, CPOperand out, String opcode,
			String istr, ArrayList<CPOperand> stride,
			ArrayList<CPOperand> padding, ArrayList<CPOperand> input_shape,
			ArrayList<CPOperand> filter_shape) {
		super(new ReorgOperator(SwapIndex.getSwapIndexFnObject()), in, out,
				opcode, istr);
		_cptype = CPINSTRUCTION_TYPE.Convolution;
		_stride = stride;
		_padding = padding;
		_input_shape = input_shape;
		_filter_shape = filter_shape;
	}
	
	public ConvolutionCPInstruction(CPOperand in, CPOperand in2, CPOperand out, String opcode,
			String istr, ArrayList<CPOperand> stride,
			ArrayList<CPOperand> padding, ArrayList<CPOperand> input_shape,
			ArrayList<CPOperand> filter_shape) {
		super(new ReorgOperator(SwapIndex.getSwapIndexFnObject()), in, out,
				opcode, istr);
		_in2 = in2;
		_cptype = CPINSTRUCTION_TYPE.Convolution;
		_stride = stride;
		_padding = padding;
		_input_shape = input_shape;
		_filter_shape = filter_shape;
	}

	public static ConvolutionCPInstruction parseInstruction(String str)
			throws DMLRuntimeException {
		CPOperand in = new CPOperand("", ValueType.UNKNOWN, DataType.UNKNOWN);
		CPOperand out = new CPOperand("", ValueType.UNKNOWN, DataType.UNKNOWN);

		String[] parts = InstructionUtils.getInstructionPartsWithValueType(str);
		String opcode = parts[0];
		if (opcode.equalsIgnoreCase("reshape_col")
				|| opcode.equalsIgnoreCase("rotate180")
				|| opcode.equalsIgnoreCase("im2col")
				|| opcode.equalsIgnoreCase("col2im")
				|| opcode.equalsIgnoreCase("pooling_pre_reshape")
				|| opcode.equalsIgnoreCase("pooling_post_reshape")
				|| opcode.equalsIgnoreCase("maxpooling")) {
			InstructionUtils.checkNumFields(parts, 14);
			// stride1, stride2, padding1, padding2
			// input_shape1, input_shape2, input_shape3, input_shape4,
			// filter_shape1, filter_shape2, filter_shape3, filter_shape4,
			in.split(parts[1]);
			out.split(parts[14]);

			ArrayList<CPOperand> stride = new ArrayList<CPOperand>();
			ArrayList<CPOperand> padding = new ArrayList<CPOperand>();
			ArrayList<CPOperand> input_shape = new ArrayList<CPOperand>();
			ArrayList<CPOperand> filter_shape = new ArrayList<CPOperand>();
			stride.add(new CPOperand(parts[2]));
			stride.add(new CPOperand(parts[3]));
			padding.add(new CPOperand(parts[4]));
			padding.add(new CPOperand(parts[5]));
			input_shape.add(new CPOperand(parts[6]));
			input_shape.add(new CPOperand(parts[7]));
			input_shape.add(new CPOperand(parts[8]));
			input_shape.add(new CPOperand(parts[9]));
			filter_shape.add(new CPOperand(parts[10]));
			filter_shape.add(new CPOperand(parts[11]));
			filter_shape.add(new CPOperand(parts[12]));
			filter_shape.add(new CPOperand(parts[13]));

			return new ConvolutionCPInstruction(in, out, opcode, str, stride,
					padding, input_shape, filter_shape);
		} 
		else if (opcode.equalsIgnoreCase("pooling_backward_reshape")
				|| opcode.equalsIgnoreCase("maxpooling_backward")) {
			InstructionUtils.checkNumFields(parts, 15);
			// dout, stride1, stride2, padding1, padding2
			// input_shape1, input_shape2, input_shape3, input_shape4,
			// filter_shape1, filter_shape2, filter_shape3, filter_shape4,
			in.split(parts[1]);
			CPOperand in2 = new CPOperand("", ValueType.UNKNOWN, DataType.UNKNOWN);
			in2.split(parts[2]);
			out.split(parts[15]);

			ArrayList<CPOperand> stride = new ArrayList<CPOperand>();
			ArrayList<CPOperand> padding = new ArrayList<CPOperand>();
			ArrayList<CPOperand> input_shape = new ArrayList<CPOperand>();
			ArrayList<CPOperand> filter_shape = new ArrayList<CPOperand>();
			stride.add(new CPOperand(parts[3]));
			stride.add(new CPOperand(parts[4]));
			padding.add(new CPOperand(parts[5]));
			padding.add(new CPOperand(parts[6]));
			input_shape.add(new CPOperand(parts[7]));
			input_shape.add(new CPOperand(parts[8]));
			input_shape.add(new CPOperand(parts[9]));
			input_shape.add(new CPOperand(parts[10]));
			filter_shape.add(new CPOperand(parts[11]));
			filter_shape.add(new CPOperand(parts[12]));
			filter_shape.add(new CPOperand(parts[13]));
			filter_shape.add(new CPOperand(parts[14]));

			return new ConvolutionCPInstruction(in, in2, out, opcode, str, stride,
					padding, input_shape, filter_shape);
		} 
		else {
			throw new DMLRuntimeException("Unknown opcode while parsing a ConvolutionCPInstruction: " + str);
		}
	}

	private int getScalarInput(ExecutionContext ec, ArrayList<CPOperand> aL,
			int index) throws DMLRuntimeException {
		return (int) ec.getScalarInput(aL.get(index).getName(),
				aL.get(index).getValueType(), aL.get(index).isLiteral())
				.getLongValue();
	}
	
	// TODO: optimize "Sparse operations" once we are happy with the performance of single node Lenet script on dense MNIST dataset
	@Override
	public void processInstruction(ExecutionContext ec)
			throws DMLRuntimeException {
		// acquire inputs
		MatrixBlock outputBlock = null;
		
		MatrixBlock matBlock = ec.getMatrixInput(input1.getName());
		pad_h = getScalarInput(ec, _padding, 0);
		pad_w = getScalarInput(ec, _padding, 1);
		stride_h = getScalarInput(ec, _stride, 0);
		stride_w = getScalarInput(ec, _stride, 1);

		N = getScalarInput(ec, _input_shape, 0);
		C = getScalarInput(ec, _input_shape, 1);
		H = getScalarInput(ec, _input_shape, 2);
		W = getScalarInput(ec, _input_shape, 3);

		K = getScalarInput(ec, _filter_shape, 0);
		
		R = getScalarInput(ec, _filter_shape, 2);
		S = getScalarInput(ec, _filter_shape, 3);
		
		P = (int) ConvolutionUtils.getP(H, R, stride_h, pad_h);
		Q = (int) ConvolutionUtils.getQ(W, S, stride_w, pad_w);
		
		
		if (instOpcode.equalsIgnoreCase("im2col")) {
			checkHeightWidth(ec);
			checkInputDimensionForIm2col(matBlock);
			this.input = matBlock;
			outputBlock = getDenseOutputBlock(ec, C * R * S, N * P * Q, true);
			im2col(matBlock, outputBlock);
		}
		else if (instOpcode.equalsIgnoreCase("reshape_col")) {
			checkHeightWidth(ec);
			this.input = matBlock;
			// Is eligible for REUSE_NONZEROED_OUTPUT but cannot guarantee that previous output has been rmvar-ed
			// without somewhat expensive HashMap checks
			outputBlock = getDenseOutputBlock(ec, N, K * P * Q, true);
			reshape_col(matBlock, outputBlock);
		}
		else if (instOpcode.equalsIgnoreCase("rotate180")) {
			checkHeightWidth(ec);
			this.input = matBlock;
			// Is eligible for REUSE_NONZEROED_OUTPUT and always an intermediate instruction
			outputBlock = getDenseOutputBlock(ec, N * P * Q, K, true);
			rotate180(matBlock, outputBlock);
		}
		else if (instOpcode.equalsIgnoreCase("col2im")) {
			checkHeightWidth(ec);
			checkInputDimensionForCol2im(matBlock);
			this.input = matBlock;
			// needs to be zeroed-out
			outputBlock = getDenseOutputBlock(ec, N, C * H * W, false);
			col2im(matBlock, outputBlock);
		}
		else if (instOpcode.equalsIgnoreCase("maxpooling")) {
			this.input = matBlock;
			// Is eligible for REUSE_NONZEROED_OUTPUT but cannot guarantee that previous output has been rmvar-ed
			// without somewhat expensive HashMap checks
			outputBlock = getDenseOutputBlock(ec, N, C*P*Q, true);
			maxpooling(matBlock, outputBlock);
		}
		else if (instOpcode.equalsIgnoreCase("maxpooling_backward")) {
			MatrixBlock dout = ec.getMatrixInput(_in2.getName());
			this.input = matBlock;
			// Is eligible for REUSE_NONZEROED_OUTPUT but cannot guarantee that previous output has been rmvar-ed
			// without somewhat expensive HashMap checks
			outputBlock = getDenseOutputBlock(ec, N, C*H*W, false); 
			maxpooling_backward(matBlock, dout, outputBlock);
			ec.releaseMatrixInput(_in2.getName());
		}
		else {
			throw new DMLRuntimeException("Unsupported op code " + instOpcode);
		}
		outputArray = null;
		inputArray = null;
		
		// release inputs/outputs
		ec.releaseMatrixInput(input1.getName());
		ec.setMatrixOutput(getOutputVariableName(), outputBlock);
	}
	
	private MatrixBlock getDenseOutputBlock(ExecutionContext ec, int numRows, int numCols, boolean reuseNonZeroedOutput1) throws DMLRuntimeException {
		long start = -1;
		if(DMLScript.STATISTICS)
			start = System.nanoTime();
		
		MatrixBlock outputBlock = new MatrixBlock(numRows, numCols, numRows * numCols);
		reuseNonZeroedOutput = false;
		if(reuseNonZeroedOutput1 && MatrixBlock.REUSE_NONZEROED_OUTPUT) {
			reuseNonZeroedOutput = true;
			outputBlock.allocateDenseBlock(true, !reuseNonZeroedOutput);  
		}
		else  {
			outputBlock.allocateDenseBlock();
		}
		outputBlock.setNonZeros(numRows * numCols);
		outputArray = outputBlock.getDenseBlock();
		if(DMLScript.STATISTICS)
			Statistics.incrementAllocationTime(System.nanoTime()-start, false);
		return outputBlock;
	}
	
	private void checkHeightWidth(ExecutionContext ec) throws DMLRuntimeException {
		int numChannelsInFilter = getScalarInput(ec, _filter_shape, 1);
		
		if (numChannelsInFilter != C) { 
			throw new DMLRuntimeException("The number of channels of input and filter should match");
		}
		if((W + 2 * pad_w - S) % stride_w != 0) {
			throw new DMLRuntimeException("The width does not work (Hint: (W + 2 * pad_w - S) % stride_w should be 0 [ ==> (" + W + "+" + " 2*" + pad_w + "-" +  S + ") % " + stride_w + "!= 0] ");
		}
		if((H + 2 * pad_h - R) % stride_h != 0) {
			throw new DMLRuntimeException("The height does not work (Hint: (H + 2 * pad_h - R) % stride_h should be 0 [ ==> (" + H + "+" + " 2*" + pad_h + "-" +  R + ") % " + stride_h + "!= 0] ");
		}
		if(H <= 0) {
			throw new DMLRuntimeException("Height of output patch should be zero");
		}
		if(Q <= 0) {
			throw new DMLRuntimeException("Width of output patch should be zero");
		}
	}

	private void maxpooling_backward(MatrixBlock input, MatrixBlock dout, MatrixBlock outputBlock) throws DMLRuntimeException {
		inputArray = null;
		if (!input.isInSparseFormat())
			inputArray = input.getDenseBlock();
		if(input.getNumColumns() != C*H*W || input.getNumRows() != N) {
			throw new DMLRuntimeException("Incorrect input dimensions in maxpooling_backward:" + input.getNumRows() + " " + input.getNumColumns() + " " + N + " " + K*P*Q);
		}
		this.dout = dout;

		doutArray = null;
		if(!dout.isInSparseFormat()) 
			doutArray = dout.getDenseBlock();
		if(dout.getNumColumns() != C*P*Q || dout.getNumRows() != N) {
			throw new DMLRuntimeException("Incorrect dout dimensions in maxpooling_backward:" + input.getNumRows() + " " + input.getNumColumns() + " " + N + " " + K*P*Q);
		}

		int constrainedNumThreads = OptimizerUtils.getConstrainedNumThreads(-1);
		if(!ALLOW_MULTI_THREADED_OPS || constrainedNumThreads <= 1) {
			for (int n = 0; n < N; n++) {
				for (int c = 0; c < C; c++) {
					doPoolingBackward(n, c);
				}
			}
		}
		else {
			runParallelConvTask(constrainedNumThreads, C, TaskType.MaxPooling_Backward);
		}
	}
	
	public void doPoolingBackward(int n, int c) {
		for (int p = 0; p < P; p++) {
			for (int q = 0; q < Q; q++) {
				int start_index_h = p * stride_h - pad_h;
				int start_index_w = q * stride_w - pad_w;
				int end_index_h = Math.min(start_index_h + R, H);
				int end_index_w = Math.min(start_index_w + S, W);
				start_index_h = Math.max(start_index_h, 0);
				start_index_w = Math.max(start_index_w, 0);
				int maxIndex = n*C*H*W + c*H*W +  start_index_h*W + start_index_w; 
				double maxVal = Double.MIN_VALUE; 


				double currDoutVal = -1;
				for (int h = start_index_h; h < end_index_h; h++) {
					for (int w = start_index_w; w < end_index_w; w++) {
						if(inputArray != null)
							currDoutVal = inputArray[n*C*H*W + c*H*W +  h*W + w];
						else
							currDoutVal = input.quickGetValue(n, c*H*W + h*W + w);

						if(maxVal < currDoutVal) {
							maxIndex = n*C*H*W + c*H*W +  h*W + w;
							maxVal = currDoutVal;
						}
					}
				}

				double inVal = -1;
				if(doutArray != null)
					inVal = doutArray[n*C*P*Q + c*P*Q +  p * Q + q];
				else
					inVal = dout.quickGetValue(n, c*P*Q +  p * Q + q);

				// synchronized(this) {
					outputArray[maxIndex] += inVal;
				// }
			}
		}
	}

	private void maxpooling(MatrixBlock input, MatrixBlock outputBlock) throws DMLRuntimeException {
		inputArray = null;
		if (!input.isInSparseFormat())
			inputArray = input.getDenseBlock();
		
		if(input.getNumColumns() != C*H*W || input.getNumRows() != N) {
			throw new DMLRuntimeException("Incorrect input dimensions in maxpooling:" + input.getNumRows() + " " + input.getNumColumns() + " " + N + " " + K*P*Q);
		}
		
		int constrainedNumThreads = OptimizerUtils.getConstrainedNumThreads(-1);
		if(!ALLOW_MULTI_THREADED_OPS || constrainedNumThreads <= 1) {
			for (int n = 0; n < N; n++) {
				for (int c = 0; c < C; c++) {
					doPooling(n, c);
				}
			}
		}
		else {
			runParallelConvTask(constrainedNumThreads, C, TaskType.MaxPooling_Forward);
		}	
	}

	public void doPooling(int n, int c) {
		for (int p = 0; p < P; p++) {
			for (int q = 0; q < Q; q++) {
				int start_index_h = p * stride_h - pad_h;
				int start_index_w = q * stride_w - pad_w;
				int end_index_h = Math.min(start_index_h + R, H);
				int end_index_w = Math.min(start_index_w + S, W);
				start_index_h = Math.max(start_index_h, 0);
				start_index_w = Math.max(start_index_w, 0);
				int out_index = n*C*P*Q + c*P*Q +  p * Q + q;
				outputArray[out_index] = 0;
				for (int h = start_index_h; h < end_index_h; h++) {
					for (int w = start_index_w; w < end_index_w; w++) {
						double inVal = -1;
						if(inputArray != null)
							inVal = inputArray[n*C*H*W + c*H*W +  h*W + w];
						else
							inVal = input.quickGetValue(n, c*H*W +  h*W + w);
						outputArray[out_index] = Math.max(outputArray[out_index], inVal);
					}
				}
			}
		}
	}
		
	// Reshape a 4D tensor of dimension (N, K, P, Q) to matrix of dimension (K, NPQ)
	private void rotate180(MatrixBlock input, MatrixBlock outputBlock) throws DMLRuntimeException {
		inputArray = null;

		if (!input.isInSparseFormat())
			inputArray = input.getDenseBlock();
		
		if(input.getNumColumns() != K*P*Q || input.getNumRows() != N) {
			throw new DMLRuntimeException("Incorrect input dimensions in rotate180:" + input.getNumRows() + " " + input.getNumColumns() + " " + N + " " + K*P*Q);
		}
		
		int constrainedNumThreads = OptimizerUtils.getConstrainedNumThreads(-1);
		if(!ALLOW_MULTI_THREADED_OPS || constrainedNumThreads <= 1) {
			for (int n = 0; n < N; n++) {
				doRotate180(n);
			}
		}
		else {
			runParallelConvTask(constrainedNumThreads, 1, TaskType.Rotate180);
		}
	}
	
	public void doRotate180(int n) {
		for (int k = 0; k < K; k++) {
			for (int p = 0; p < P; p++) {
				for (int q = 0; q < Q; q++) {
					if(inputArray != null)
						outputArray[n*K*P*Q + p*Q*K + q*K + k] = inputArray[n*K*P*Q + k*P*Q + p*Q + q];
					else
						outputArray[n*P*Q*K + p*Q*K + q*K + k] = input.quickGetValue(n, k*P*Q + p*Q + q);
				}
			}
		}
	}
	
	
	// Reshape a matrix of dimension (K, NPQ) to 4D tensor of dimension (N, K, P, Q)
	private void reshape_col(MatrixBlock input, MatrixBlock outputBlock) throws DMLRuntimeException {
		inputArray = null;

		if (!input.isInSparseFormat())
			inputArray = input.getDenseBlock();
		
		if(input.getNumColumns() != N*P*Q || input.getNumRows() != K) {
			throw new DMLRuntimeException("Incorrect input dimensions in reshape_col:" + input.getNumRows() + " " + input.getNumColumns());
		}
		
		int constrainedNumThreads = OptimizerUtils.getConstrainedNumThreads(-1);
		if(!ALLOW_MULTI_THREADED_OPS || constrainedNumThreads <= 1) {
			for (int n = 0; n < N; n++) { 
				doReshapeCol(n);
			}
		}
		else {
			runParallelConvTask(constrainedNumThreads, 1, TaskType.ReshapeCol);
		}
		
	}
	
	private void runParallelConvTask(int constrainedNumThreads, int Z, TaskType type) throws DMLRuntimeException {
		ArrayList<ConvTask> tasks = new ArrayList<ConvTask>();		
		
		// Total number of compute units available: constrainedNumThreads
		// Static task allocation. TODO: Do this in dynamic way
		for (int n = 0; n < N; n++) {
			for (int z = 0; z < Z; z += TASK_SIZE) {
				tasks.add(new ConvTask(this, n, n+1, z, Math.min(Z, z+TASK_SIZE), type));
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
		ConvolutionCPInstruction curr; int n1; int n2; int z1; int z2; 
		TaskType type;
		public ConvTask(ConvolutionCPInstruction curr, int n1, int n2, int z1, int z2, TaskType type) {
			this.curr = curr; 
			this.n1 = n1;
			this.n2 = n2;
			this.z1 = z1;
			this.z2 = z2;
			this.type = type;
		}
		
		@Override
		public Object call() throws Exception {
			switch(type) {
				case ReshapeCol:
					for (int n = n1; n < n2; n++) {
						curr.doReshapeCol(n);
					}
					break;
				case Rotate180:
					for (int n = n1; n < n2; n++) {
						curr.doRotate180(n);
					}
					break;
				case Im2Col:
					for (int n = n1; n < n2; n++) {
						for (int z = z1; z < z2; z++) {
							curr.doIm2colOverInputPath_NCHW(n, z);
						}
					}
					break;
				case Col2Im:
					for (int n = n1; n < n2; n++) {
						for (int z = z1; z < z2; z++) {
							curr.doCol2imOverInputPath_NCHW(n, z);
						}
					}
					break;
				case MaxPooling_Forward:
					for (int n = n1; n < n2; n++) {
						for (int z = z1; z < z2; z++) {
							curr.doPooling(n, z);
						}
					}
					break;
				case MaxPooling_Backward:
					for (int n = n1; n < n2; n++) {
						for (int z = z1; z < z2; z++) {
							curr.doPoolingBackward(n, z);
						}
					}
					break;
				default:
					throw new RuntimeException("Unsupported ConvTask:" + type.name());
			}
			return null;
		}
	}
		
	private void doReshapeCol(int n) {
		if(inputArray != null) {
			for (int k = 0; k < K; k++)  {
				System.arraycopy(inputArray, k*N*P*Q + n*P*Q, outputArray, n*K*P*Q + k*P*Q, P*Q);
			}
		}
		else {
			for (int k = 0; k < K; k++) {
				for (int p = 0; p < P; p++) { 
					for (int q = 0; q < Q; q++) {
						outputArray[n*K*P*Q + k*P*Q + p*Q + q] = input.quickGetValue(k, n*P*Q + p*Q + q);
					}
				}
			}
		}
	}
	
	// Converts a 4D tensor (N, C, R, S) to a matrix of dimension (CRS, NPQ)
	private void im2col(MatrixBlock input, MatrixBlock outputBlock) throws DMLRuntimeException {
		inputArray = null;
		if (!input.isInSparseFormat())
			inputArray = input.getDenseBlock();
		
		int constrainedNumThreads = OptimizerUtils.getConstrainedNumThreads(-1);
		if(!ALLOW_MULTI_THREADED_OPS || constrainedNumThreads <= 1) {
			for (int n = 0; n < N; n++) { // Do following for all images
				for (int c = 0; c < C; c++) { // Since format is NCHW
					doIm2colOverInputPath_NCHW(n, c);
				}
			}
		}
		else {
			runParallelConvTask(constrainedNumThreads, C, TaskType.Im2Col);
		}
	}
	
	// Converts a matrix of dimension (CRS, NPQ) to a 4D tensor (N, C, H, W)
	private void col2im(MatrixBlock input, MatrixBlock outputBlock) throws DMLRuntimeException {
		inputArray = null;
		if (!input.isInSparseFormat())
			inputArray = input.getDenseBlock();
		
		int constrainedNumThreads = OptimizerUtils.getConstrainedNumThreads(-1);
		if(!ALLOW_MULTI_THREADED_OPS || constrainedNumThreads <= 1) {
			// Sequential col2im
			for (int n = 0; n < N; n++) { // Do following for all images
				for (int c = 0; c < C; c++) { // Since format is NCHW
					doCol2imOverInputPath_NCHW(n, c);
				}
			}
		}
		else {
			// Parallel col2im
			runParallelConvTask(constrainedNumThreads, C, TaskType.Col2Im);
		}
	}
	
		
	private void doCol2imOverInputPath_NCHW(int n, int c) {
		for (int r = 0; r < R; r++) { // Get an input patch of size R X S
			for (int s = 0; s < S; s++) {
				int localIndex = ((c*R*S*N + r*S*N + s*N + n)*P*Q);
				
				int input_row = r - pad_h;
				// And copy it to outputArray[i] (taking care of padding & striding)
				for (int p = P; p > 0; p--) {
					if (input_row >= 0 && input_row < H) {
						int input_col = s - pad_w;
						for (int q = Q; q > 0; q--, localIndex++) {
							if (input_col >= 0 && input_col < W) {
								// Copy from [channel c, height input_row, width input_col]
								int index = n*C*H*W + c*H*W + input_row*W + input_col;
								if (inputArray != null) {
									outputArray[index] += inputArray[localIndex];
								}
								else {
									// TODO: Optimize for sparse input
									// Note: localIndex = row*N*P*Q + col
									int row = localIndex / (N*P*Q);
									int col = localIndex % (N*P*Q);
									outputArray[index] += input.quickGetValue(row, col); 
								}
							}
							input_col += stride_w;
						}
					} else {
						localIndex += Q;
					}
					input_row += stride_h;
				}
			}
		}
		
	}
	
	private void doIm2colOverInputPath_NCHW(int n, int c) {
		final int inputOffset = n*C*H*W + c*H*W;
		final int outputOffset = (c*R*S*N + n)*P*Q;
		
		for (int r = 0; r < R; r++) { // Get an input patch of size R X S
			for (int s = 0; s < S; s++) {
				int localIndex = outputOffset + ((r*S*N + s*N)*P*Q);
				
				int input_row = r - pad_h;
				// And copy it to outputArray[i] (taking care of padding & striding)
				for (int p = P; p > 0; p--) {
					if (input_row >= 0 && input_row < H) {
						int input_col = s - pad_w;
						for (int q = Q; q > 0; q--, localIndex++) {
							if (input_col >= 0 && input_col < W) {
								// Copy from [channel c, height input_row, width input_col]
								if(inputArray != null)
									outputArray[localIndex] = inputArray[inputOffset + input_row*W + input_col];
								else
									outputArray[localIndex] = input.quickGetValue(n, c*H*W + input_row*W + input_col);
							}
							else if(reuseNonZeroedOutput) {
								outputArray[localIndex] = 0;
							}
							input_col += stride_w;
						}
					} else {
						if(reuseNonZeroedOutput) {
							for(int i = localIndex; i < localIndex + Q; i++) {
								outputArray[localIndex] = 0;
							}
						}
						localIndex += Q;
					}
					input_row += stride_h;
				}
			}
		}
		
	}

	private void checkInputDimensionForIm2col(MatrixBlock matBlock) throws DMLRuntimeException {
		if((N != matBlock.getNumRows() || C*H*W != matBlock.getNumColumns())) {
			throw new DMLRuntimeException("Incorrect input shape in conv2d");
		}
	}
	
	private void checkInputDimensionForCol2im(MatrixBlock matBlock) throws DMLRuntimeException {
		if((C*R*S != matBlock.getNumRows() || N*P*Q != matBlock.getNumColumns())) {
			throw new DMLRuntimeException("Incorrect input shape in conv2d_backward_data");
		}
	}
}
