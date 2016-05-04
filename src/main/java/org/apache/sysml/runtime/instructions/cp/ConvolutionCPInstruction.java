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
import org.apache.sysml.api.DMLScript;
import org.apache.sysml.parser.Expression.DataType;
import org.apache.sysml.parser.Expression.ValueType;
import org.apache.sysml.runtime.DMLRuntimeException;
import org.apache.sysml.runtime.controlprogram.context.ExecutionContext;
import org.apache.sysml.runtime.functionobjects.SwapIndex;
import org.apache.sysml.runtime.instructions.InstructionUtils;
import org.apache.sysml.runtime.matrix.data.LibMatrixDNN;
import org.apache.sysml.runtime.matrix.data.LibMatrixDNN.ConvolutionParameters;
import org.apache.sysml.runtime.matrix.data.MatrixBlock;
import org.apache.sysml.runtime.matrix.operators.ReorgOperator;
import org.apache.sysml.runtime.util.ConvolutionUtils;
import org.apache.sysml.utils.Statistics;

public class ConvolutionCPInstruction extends UnaryCPInstruction {
	
	private CPOperand _in2; // used for pooling backward
	private ArrayList<CPOperand> _input_shape;
	private ArrayList<CPOperand> _filter_shape;
	private ArrayList<CPOperand> _stride = new ArrayList<CPOperand>();
	private ArrayList<CPOperand> _padding = new ArrayList<CPOperand>();
	private boolean reuseNonZeroedOutput = false;
	
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
		int N; int C; int H; int W;
		int K; int R; int S; int stride_h; int stride_w; int pad_h; int pad_w;
		int P; int Q;
		
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
		
		ConvolutionParameters params = new ConvolutionParameters(N, C, H, W, K, R, S, stride_h, stride_w, pad_h, pad_w);
		
		if (instOpcode.equalsIgnoreCase("im2col")) {
			checkHeightWidth(ec, params);
			checkInputDimensionForIm2col(matBlock, params);
			outputBlock = getDenseOutputBlock(ec, C * R * S, N * P * Q, true);
			params.setReuseNonZeroedOutput(reuseNonZeroedOutput);
			LibMatrixDNN.im2col(matBlock, outputBlock, params);
		}
		else if (instOpcode.equalsIgnoreCase("reshape_col")) {
			checkHeightWidth(ec, params);
			// Is eligible for REUSE_NONZEROED_OUTPUT but cannot guarantee that previous output has been rmvar-ed
			// without somewhat expensive HashMap checks
			outputBlock = getDenseOutputBlock(ec, N, K * P * Q, true);
			params.setReuseNonZeroedOutput(reuseNonZeroedOutput);
			LibMatrixDNN.reshape_col(matBlock, outputBlock, params);
		}
		else if (instOpcode.equalsIgnoreCase("rotate180")) {
			checkHeightWidth(ec, params);
			// Is eligible for REUSE_NONZEROED_OUTPUT and always an intermediate instruction
			outputBlock = getDenseOutputBlock(ec, N * P * Q, K, true);
			params.setReuseNonZeroedOutput(reuseNonZeroedOutput);
			LibMatrixDNN.rotate180(matBlock, outputBlock, params);
		}
		else if (instOpcode.equalsIgnoreCase("col2im")) {
			checkHeightWidth(ec, params);
			checkInputDimensionForCol2im(matBlock, params);
			// needs to be zeroed-out
			outputBlock = getDenseOutputBlock(ec, N, C * H * W, false);
			params.setReuseNonZeroedOutput(reuseNonZeroedOutput);
			LibMatrixDNN.col2im(matBlock, outputBlock, params);
		}
		else if (instOpcode.equalsIgnoreCase("maxpooling")) {
			// Is eligible for REUSE_NONZEROED_OUTPUT but cannot guarantee that previous output has been rmvar-ed
			// without somewhat expensive HashMap checks
			outputBlock = getDenseOutputBlock(ec, N, C*P*Q, true);
			params.setReuseNonZeroedOutput(reuseNonZeroedOutput);
			LibMatrixDNN.maxpooling(matBlock, outputBlock, params);
		}
		else if (instOpcode.equalsIgnoreCase("maxpooling_backward")) {
			MatrixBlock dout = ec.getMatrixInput(_in2.getName());
			// Is eligible for REUSE_NONZEROED_OUTPUT but cannot guarantee that previous output has been rmvar-ed
			// without somewhat expensive HashMap checks
			outputBlock = getDenseOutputBlock(ec, N, C*H*W, false);
			params.setReuseNonZeroedOutput(reuseNonZeroedOutput);
			LibMatrixDNN.maxpooling_backward(matBlock, dout, outputBlock, params);
			ec.releaseMatrixInput(_in2.getName());
		}
		else {
			throw new DMLRuntimeException("Unsupported op code " + instOpcode);
		}
		
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
		if(DMLScript.STATISTICS)
			Statistics.incrementAllocationTime(System.nanoTime()-start, false);
		return outputBlock;
	}
	
	private void checkHeightWidth(ExecutionContext ec, ConvolutionParameters params) throws DMLRuntimeException {
		int numChannelsInFilter = getScalarInput(ec, _filter_shape, 1);
		
		if (numChannelsInFilter != params.C) { 
			throw new DMLRuntimeException("The number of channels of input and filter should match");
		}
		if((params.W + 2 * params.pad_w - params.S) % params.stride_w != 0) {
			throw new DMLRuntimeException("The width does not work (Hint: (W + 2 * pad_w - S) % stride_w should be 0 [ ==> (" + params.W + "+" + " 2*" + params.pad_w + "-" +  params.S + ") % " + params.stride_w + "!= 0] ");
		}
		if((params.H + 2 * params.pad_h - params.R) % params.stride_h != 0) {
			throw new DMLRuntimeException("The height does not work (Hint: (H + 2 * pad_h - R) % stride_h should be 0 [ ==> (" + params.H + "+" + " 2*" + params.pad_h + "-" +  params.R + ") % " + params.stride_h + "!= 0] ");
		}
		if(params.H <= 0) {
			throw new DMLRuntimeException("Height of output patch should be zero");
		}
		if(params.Q <= 0) {
			throw new DMLRuntimeException("Width of output patch should be zero");
		}
	}


	private void checkInputDimensionForIm2col(MatrixBlock matBlock, ConvolutionParameters params) throws DMLRuntimeException {
		if((params.N != matBlock.getNumRows() || params.C*params.H*params.W != matBlock.getNumColumns())) {
			throw new DMLRuntimeException("Incorrect input shape in conv2d");
		}
	}
	
	private void checkInputDimensionForCol2im(MatrixBlock matBlock, ConvolutionParameters params) throws DMLRuntimeException {
		if((params.C*params.R*params.S != matBlock.getNumRows() || params.N*params.P*params.Q != matBlock.getNumColumns())) {
			throw new DMLRuntimeException("Incorrect input shape in conv2d_backward_data");
		}
	}
}
