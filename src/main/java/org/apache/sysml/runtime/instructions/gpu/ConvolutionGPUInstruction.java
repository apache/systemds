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
package org.apache.sysml.runtime.instructions.gpu;

import java.util.ArrayList;

import org.apache.sysml.runtime.DMLRuntimeException;
import org.apache.sysml.runtime.controlprogram.caching.MatrixObject;
import org.apache.sysml.runtime.controlprogram.context.ExecutionContext;
import org.apache.sysml.runtime.functionobjects.SwapIndex;
import org.apache.sysml.runtime.instructions.InstructionUtils;
import org.apache.sysml.runtime.instructions.cp.CPOperand;
import org.apache.sysml.runtime.matrix.data.LibMatrixCUDA;
import org.apache.sysml.runtime.matrix.operators.ReorgOperator;
import org.apache.sysml.runtime.util.ConvolutionUtils;
import org.apache.sysml.utils.GPUStatistics;

public class ConvolutionGPUInstruction extends GPUInstruction 
{
	private CPOperand _input1; 
	private CPOperand _input2; 
	private CPOperand _input3; 
	private CPOperand _output; 
	private ArrayList<CPOperand> _input_shape;
	private ArrayList<CPOperand> _filter_shape;
	private ArrayList<CPOperand> _stride = new ArrayList<CPOperand>();
	private ArrayList<CPOperand> _padding = new ArrayList<CPOperand>();
	
	public ConvolutionGPUInstruction(CPOperand in1, CPOperand in2, CPOperand out, String opcode, String istr) throws DMLRuntimeException {
		super(new ReorgOperator(SwapIndex.getSwapIndexFnObject()), opcode, istr);
		if(!(opcode.equals("bias_add") || opcode.equals("bias_multiply") || opcode.equals("relu_backward"))) {
			throw new DMLRuntimeException("Incorrect usage. Expected the opcode to be bias_add or bias_multiply or relu_backward, but found " + opcode);
		}
		_input1 = in1;
		_input2 = in2;
		_gputype = GPUINSTRUCTION_TYPE.Convolution;
		_output = out;
	}
	
	public ConvolutionGPUInstruction(CPOperand in1, CPOperand in2, CPOperand in3, CPOperand out, String opcode,
			String istr, ArrayList<CPOperand> stride,
			ArrayList<CPOperand> padding, ArrayList<CPOperand> input_shape,
			ArrayList<CPOperand> filter_shape) 
	{
		this(in1, in2, out, opcode, istr, stride, padding,  input_shape, filter_shape);
		_input3 = in3;
	}
	
	public ConvolutionGPUInstruction(CPOperand in1, CPOperand in2, CPOperand out, String opcode,
			String istr, ArrayList<CPOperand> stride,
			ArrayList<CPOperand> padding, ArrayList<CPOperand> input_shape,
			ArrayList<CPOperand> filter_shape) 
	{
		super(new ReorgOperator(SwapIndex.getSwapIndexFnObject()), opcode, istr);
		_gputype = GPUINSTRUCTION_TYPE.Convolution;
		
		_input1 = in1;
		_input2 = in2;
		_output = out;
		_stride = stride;
		_padding = padding;
		_input_shape = input_shape;
		_filter_shape = filter_shape;
	}
	
	public static ConvolutionGPUInstruction parseInstruction(String str)
		throws DMLRuntimeException 
	{
		String[] parts = InstructionUtils.getInstructionPartsWithValueType(str);
		String opcode = parts[0];
		
		if( ( opcode.equalsIgnoreCase("conv2d")
			 || opcode.equalsIgnoreCase("conv2d_backward_filter")
			 || opcode.equalsIgnoreCase("conv2d_backward_data")
			 || opcode.equalsIgnoreCase("maxpooling_backward")) ) {
			InstructionUtils.checkNumFields(parts, 15);
			CPOperand in1 = new CPOperand(parts[1]);
			CPOperand in2 = new CPOperand(parts[2]);
			CPOperand out = new CPOperand(parts[15]);
		
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

			return new ConvolutionGPUInstruction(in1, in2, out, opcode, str, stride,
					padding, input_shape, filter_shape);
		}
		else if (opcode.equalsIgnoreCase("conv2d_bias_add")) {
			InstructionUtils.checkNumFields(parts, 16);
			CPOperand in1 = new CPOperand(parts[1]);
			CPOperand in2 = new CPOperand(parts[2]);
			CPOperand in3 = new CPOperand(parts[3]);
			CPOperand out = new CPOperand(parts[16]);
		
			ArrayList<CPOperand> stride = new ArrayList<CPOperand>();
			ArrayList<CPOperand> padding = new ArrayList<CPOperand>();
			ArrayList<CPOperand> input_shape = new ArrayList<CPOperand>();
			ArrayList<CPOperand> filter_shape = new ArrayList<CPOperand>();
			stride.add(new CPOperand(parts[4]));
			stride.add(new CPOperand(parts[5]));
			padding.add(new CPOperand(parts[6]));
			padding.add(new CPOperand(parts[7]));
			input_shape.add(new CPOperand(parts[8]));
			input_shape.add(new CPOperand(parts[9]));
			input_shape.add(new CPOperand(parts[10]));
			input_shape.add(new CPOperand(parts[11]));
			filter_shape.add(new CPOperand(parts[12]));
			filter_shape.add(new CPOperand(parts[13]));
			filter_shape.add(new CPOperand(parts[14]));
			filter_shape.add(new CPOperand(parts[15]));

			return new ConvolutionGPUInstruction(in1, in2, in3, out, opcode, str, stride,
					padding, input_shape, filter_shape);
		}
		else if (opcode.equalsIgnoreCase("maxpooling")) {
			InstructionUtils.checkNumFields(parts, 14);
			CPOperand in1 = new CPOperand(parts[1]);
			CPOperand out = new CPOperand(parts[14]);
		
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

			return new ConvolutionGPUInstruction(in1, null, out, opcode, str, stride,
					padding, input_shape, filter_shape);
		}
		else if( opcode.equalsIgnoreCase("bias_add") || opcode.equalsIgnoreCase("relu_backward") || opcode.equalsIgnoreCase("bias_multiply")  ) {
			InstructionUtils.checkNumFields(parts, 3);
			CPOperand in1 = new CPOperand(parts[1]);
			CPOperand in2 = new CPOperand(parts[2]);
			CPOperand out = new CPOperand(parts[3]);
			return new ConvolutionGPUInstruction(in1, in2, out, opcode, str);
		}
		else {
			throw new DMLRuntimeException("Unknown opcode while parsing a ConvolutionGPUInstruction: " + str);	
		}
	}

	public void processBiasInstruction(String instOpcode, ExecutionContext ec) throws DMLRuntimeException {
		GPUStatistics.incrementNoOfExecutedGPUInst();
		MatrixObject input = getMatrixInputForGPUInstruction(ec, _input1.getName());
		MatrixObject bias = getMatrixInputForGPUInstruction(ec, _input2.getName());
		
		ec.setMetaData(_output.getName(), input.getNumRows(), input.getNumColumns());
		MatrixObject out = getDenseMatrixOutputForGPUInstruction(ec, _output.getName());
		if(instOpcode.equalsIgnoreCase("bias_add"))
			LibMatrixCUDA.biasAdd(ec.getGPUContext(0), getExtendedOpcode(), input, bias, out);
		else if(instOpcode.equalsIgnoreCase("bias_multiply"))
			LibMatrixCUDA.biasMultiply(ec.getGPUContext(0), getExtendedOpcode(), input, bias, out);
		// release inputs/outputs
		ec.releaseMatrixInputForGPUInstruction(_input1.getName());
		ec.releaseMatrixInputForGPUInstruction(_input2.getName());
		ec.releaseMatrixOutputForGPUInstruction(_output.getName());
	}

	public void processReLUBackwardInstruction(ExecutionContext ec) throws DMLRuntimeException {
		GPUStatistics.incrementNoOfExecutedGPUInst();
		MatrixObject input = getMatrixInputForGPUInstruction(ec, _input1.getName());
		MatrixObject dout = getMatrixInputForGPUInstruction(ec, _input2.getName());
		
		MatrixObject out = getDenseMatrixOutputForGPUInstruction(ec, _output.getName());
		ec.setMetaData(_output.getName(), input.getNumRows(), input.getNumColumns());
		LibMatrixCUDA.reluBackward(ec.getGPUContext(0), getExtendedOpcode(), input, dout, out);
		// release inputs/outputs
		ec.releaseMatrixInputForGPUInstruction(_input1.getName());
		ec.releaseMatrixInputForGPUInstruction(_input2.getName());
		ec.releaseMatrixOutputForGPUInstruction(_output.getName());
	}
	
	@Override
	public void processInstruction(ExecutionContext ec) 
			throws DMLRuntimeException 
	{
		if (instOpcode.equalsIgnoreCase("bias_add") || instOpcode.equalsIgnoreCase("bias_multiply")) {
			processBiasInstruction(instOpcode, ec);
			return;
		}
		else if (instOpcode.equalsIgnoreCase("relu_backward")) {
			processReLUBackwardInstruction(ec);
			return;
		}
		
		GPUStatistics.incrementNoOfExecutedGPUInst();
					
		int pad_h = getScalarInput(ec, _padding, 0);
		int pad_w = getScalarInput(ec, _padding, 1);
		int stride_h = getScalarInput(ec, _stride, 0);
		int stride_w = getScalarInput(ec, _stride, 1);

		int N = getScalarInput(ec, _input_shape, 0);
		int C = getScalarInput(ec, _input_shape, 1);
		int H = getScalarInput(ec, _input_shape, 2);
		int W = getScalarInput(ec, _input_shape, 3);

		int K = getScalarInput(ec, _filter_shape, 0);
		
		int R = getScalarInput(ec, _filter_shape, 2);
		int S = getScalarInput(ec, _filter_shape, 3);
		
		int P = (int) ConvolutionUtils.getP(H, R, stride_h, pad_h);
		int Q = (int) ConvolutionUtils.getQ(W, S, stride_w, pad_w);
		
		if (instOpcode.equalsIgnoreCase("conv2d")) {
			MatrixObject image = getMatrixInputForGPUInstruction(ec, _input1.getName());
			MatrixObject filter = getMatrixInputForGPUInstruction(ec, _input2.getName());

			if(image.getNumRows() != N || image.getNumColumns() != C*H*W) 
				throw new DMLRuntimeException("Incorrect dimensions for image in conv2d");
			if(filter.getNumRows() != K || filter.getNumColumns() != C*R*S) 
				throw new DMLRuntimeException("Incorrect dimensions for filter in conv2d");
			
			ec.setMetaData(_output.getName(), N, K * P * Q);
			MatrixObject out = getDenseMatrixOutputForGPUInstruction(ec, _output.getName());
			LibMatrixCUDA.conv2d(ec.getGPUContext(0), getExtendedOpcode(), image, filter, out, N, C, H, W,
					K, R, S, pad_h, pad_w, stride_h, stride_w, P, Q);
		}
		else if (instOpcode.equalsIgnoreCase("conv2d_bias_add")) {
			MatrixObject image = getMatrixInputForGPUInstruction(ec, _input1.getName());
			MatrixObject bias = getMatrixInputForGPUInstruction(ec, _input2.getName());
			MatrixObject filter = getMatrixInputForGPUInstruction(ec, _input3.getName());

			if(image.getNumRows() != N || image.getNumColumns() != C*H*W) 
				throw new DMLRuntimeException("Incorrect dimensions for image in conv2d");
			if(filter.getNumRows() != K || filter.getNumColumns() != C*R*S) 
				throw new DMLRuntimeException("Incorrect dimensions for filter in conv2d");
			
			ec.setMetaData(_output.getName(), N, K * P * Q);
			MatrixObject out = getDenseMatrixOutputForGPUInstruction(ec, _output.getName());
			LibMatrixCUDA.conv2dBiasAdd(ec.getGPUContext(0), getExtendedOpcode(), image, bias, filter, out, N, C, H, W,
						K, R, S, pad_h, pad_w, stride_h, stride_w, P, Q);
		}
		else if (instOpcode.equalsIgnoreCase("conv2d_backward_filter")) {
			MatrixObject image = getMatrixInputForGPUInstruction(ec, _input1.getName());
			MatrixObject dout = getMatrixInputForGPUInstruction(ec, _input2.getName());

			if(image.getNumRows() != N || image.getNumColumns() != C*H*W) 
				throw new DMLRuntimeException("Incorrect dimensions for image in conv2d_backward_filter");
			if(dout.getNumRows() != N || dout.getNumColumns() != K*P*Q) 
				throw new DMLRuntimeException("Incorrect dimensions for dout in conv2d_backward_filter: " + 
						dout.getNumRows() + " != " +  N + " || " + dout.getNumColumns() + " != " + K*P*Q);
			
			ec.setMetaData(_output.getName(), K, C * R * S);
			MatrixObject out = getDenseMatrixOutputForGPUInstruction(ec, _output.getName());
			LibMatrixCUDA.conv2dBackwardFilter(ec.getGPUContext(0), getExtendedOpcode(), image, dout, out, N, C, H, W,
					K, R, S, pad_h, pad_w, stride_h, stride_w, P, Q);
			// TODO: For now always copy the device data to host
			// ec.gpuCtx.copyDeviceToHost(outputBlock);
		}
		else if (instOpcode.equalsIgnoreCase("conv2d_backward_data")) {
			MatrixObject filter = getMatrixInputForGPUInstruction(ec, _input1.getName());
			MatrixObject dout = getMatrixInputForGPUInstruction(ec, _input2.getName());

			if(filter.getNumRows() != K || filter.getNumColumns() != C*R*S) 
				throw new DMLRuntimeException("Incorrect dimensions for filter in convolution_backward_data");
			if(dout.getNumRows() != N || dout.getNumColumns() != K*P*Q) 
				throw new DMLRuntimeException("Incorrect dimensions for dout in conv2d_backward_data: " + 
						dout.getNumRows() + " != " +  N + " || " + dout.getNumColumns() + " != " + K*P*Q);
			
			ec.setMetaData(_output.getName(), N, C * H * W);
			MatrixObject out = getDenseMatrixOutputForGPUInstruction(ec, _output.getName());
			LibMatrixCUDA.conv2dBackwardData(ec.getGPUContext(0), getExtendedOpcode(), filter, dout, out, N, C, H, W,
					K, R, S, pad_h, pad_w, stride_h, stride_w, P, Q);
		}
		else if (instOpcode.equalsIgnoreCase("maxpooling")) {
			MatrixObject image = getMatrixInputForGPUInstruction(ec, _input1.getName());

			if(image.getNumRows() != N || image.getNumColumns() != C*H*W) 
				throw new DMLRuntimeException("Incorrect dimensions for image in maxpooling: " + 
						image.getNumRows() + " != " +  N + " || " + image.getNumColumns() + " != " + C*H*W);
			
			ec.setMetaData(_output.getName(), N, C * P * Q);
			MatrixObject out = getDenseMatrixOutputForGPUInstruction(ec, _output.getName());
			if(instOpcode.equalsIgnoreCase("maxpooling"))
				LibMatrixCUDA.maxpooling(ec.getGPUContext(0), getExtendedOpcode(), image, out, N, C, H, W,
					K, R, S, pad_h, pad_w, stride_h, stride_w, P, Q);
		}
		else if (instOpcode.equalsIgnoreCase("maxpooling_backward")) {
			MatrixObject image = getMatrixInputForGPUInstruction(ec, _input1.getName());
			MatrixObject dout = getMatrixInputForGPUInstruction(ec, _input2.getName());
			
			if(dout.getNumRows() != N || dout.getNumColumns() != C*P*Q) 
				throw new DMLRuntimeException("Incorrect dimensions for dout in maxpooling_backward");
			if(image.getNumRows() != N || image.getNumColumns() != C*H*W) 
				throw new DMLRuntimeException("Incorrect dimensions for image in maxpooling_backward: " + 
						image.getNumRows() + " != " +  N + " || " + image.getNumColumns() + " != " + K*P*Q);
			
			ec.setMetaData(_output.getName(), N, C * H * W);
			MatrixObject out = getDenseMatrixOutputForGPUInstruction(ec, _output.getName());
			LibMatrixCUDA.maxpoolingBackward(ec.getGPUContext(0), getExtendedOpcode(), image, dout, out, N, C, H, W,
					K, R, S, pad_h, pad_w, stride_h, stride_w, P, Q);
		}
		else {
			throw new DMLRuntimeException("Unsupported GPU context for " + instOpcode);
		}
		
		// release inputs/outputs
		ec.releaseMatrixInputForGPUInstruction(_input1.getName());

		if ( !instOpcode.equalsIgnoreCase("maxpooling") )
			ec.releaseMatrixInputForGPUInstruction(_input2.getName());

		if (instOpcode.equalsIgnoreCase("conv2d_bias_add"))
			ec.releaseMatrixInputForGPUInstruction(_input3.getName());

		ec.releaseMatrixOutputForGPUInstruction(_output.getName());
	}

	private int getScalarInput(ExecutionContext ec, ArrayList<CPOperand> aL, int index) 
		throws DMLRuntimeException 
	{
		return (int) ec.getScalarInput(aL.get(index).getName(),
				aL.get(index).getValueType(), aL.get(index).isLiteral())
				.getLongValue();
	}
}
