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
import jcuda.Pointer;
import jcuda.jcudnn.JCudnn;

import org.apache.sysml.runtime.DMLRuntimeException;
import org.apache.sysml.runtime.controlprogram.caching.MatrixObject;
import org.apache.sysml.runtime.controlprogram.context.ExecutionContext;
import org.apache.sysml.runtime.functionobjects.SwapIndex;
import org.apache.sysml.runtime.instructions.InstructionUtils;
import org.apache.sysml.runtime.instructions.cp.CPOperand;
import org.apache.sysml.runtime.instructions.gpu.context.ExecutionConfig;
import org.apache.sysml.runtime.instructions.gpu.context.GPUContext;
import org.apache.sysml.runtime.matrix.data.LibMatrixCUDA;
import org.apache.sysml.runtime.matrix.data.LibMatrixCuDNN;
import org.apache.sysml.runtime.matrix.data.LibMatrixDNN.PoolingType;
import org.apache.sysml.runtime.matrix.operators.ReorgOperator;
import org.apache.sysml.runtime.util.DnnUtils;
import org.apache.sysml.utils.GPUStatistics;

public class DnnGPUInstruction extends GPUInstruction {
	private CPOperand _input1;
	private CPOperand _input2;
	private CPOperand _input3;
	private CPOperand _input4;
	private CPOperand _input5;
	private CPOperand _input6;
	private CPOperand _input7;
	private CPOperand _input8;
	private CPOperand _output;
	private CPOperand _output2;
	private CPOperand _output3;
	private CPOperand _output4;
	private CPOperand _output5;
	private ArrayList<CPOperand> _input_shape;
	private ArrayList<CPOperand> _filter_shape;
	private ArrayList<CPOperand> _stride = new ArrayList<>();
	private ArrayList<CPOperand> _padding = new ArrayList<>();
	private double _intermediateMemoryBudget = 0;
	private GPUContext gCtx;
	private String instName;
	
	public DnnGPUInstruction(CPOperand in1, CPOperand in2, CPOperand out, String opcode, String istr, double intermediateMemoryBudget) {
		super(new ReorgOperator(SwapIndex.getSwapIndexFnObject()), opcode, istr);
		if (!(opcode.equals("bias_add") || opcode.equals("bias_multiply") || opcode.equals("relu_backward") || opcode.equals("inv_var") )) {
			throw new DMLRuntimeException(
					"Incorrect usage. Expected the opcode to be bias_add or bias_multiply or relu_backward or inv_var, but found "
							+ opcode);
		}
		_input1 = in1;
		_input2 = in2;
		_gputype = GPUINSTRUCTION_TYPE.Dnn;
		_output = out;
		_intermediateMemoryBudget = intermediateMemoryBudget;
	}
	public DnnGPUInstruction(CPOperand in1, CPOperand in2, CPOperand in3, CPOperand in4, CPOperand in5, CPOperand in6, 
			CPOperand out, CPOperand out2, String opcode, String istr, 
			double intermediateMemoryBudget) throws DMLRuntimeException {
		super(new ReorgOperator(SwapIndex.getSwapIndexFnObject()), opcode, istr);
		_input1 = in1;
		_input2 = in2;
		_input3 = in3;
		_input4 = in4;
		_input5 = in5;
		_input6 = in6;
		_gputype = GPUINSTRUCTION_TYPE.Dnn;
		_output = out;
		_output2 = out2;
		_intermediateMemoryBudget = intermediateMemoryBudget;
	}
	
	public DnnGPUInstruction(CPOperand in1, CPOperand in2, CPOperand in3, CPOperand in4, CPOperand in5,
			CPOperand in6, CPOperand in7, CPOperand in8,
			CPOperand out, CPOperand out2, CPOperand out3, CPOperand out4, CPOperand out5, String opcode, String istr, 
			double intermediateMemoryBudget) throws DMLRuntimeException {
		super(new ReorgOperator(SwapIndex.getSwapIndexFnObject()), opcode, istr);
		_input1 = in1;
		_input2 = in2;
		_input3 = in3;
		_input4 = in4;
		_input5 = in5;
		_input6 = in6;
		_input7 = in7;
		_input8 = in8;
		_gputype = GPUINSTRUCTION_TYPE.Dnn;
		_output = out;
		_output2 = out2;
		_output3 = out3;
		_output4 = out4;
		_output5 = out5;
		_intermediateMemoryBudget = intermediateMemoryBudget;
	}
	
	public DnnGPUInstruction(CPOperand in1, CPOperand in2, CPOperand in3, CPOperand out, String opcode, String istr, 
			double intermediateMemoryBudget) throws DMLRuntimeException {
		super(new ReorgOperator(SwapIndex.getSwapIndexFnObject()), opcode, istr);
		if( !(opcode.equals("channel_sums") || opcode.equals("reshape_colmeans") || opcode.equals("update_ema") ) ) {
			throw new DMLRuntimeException("Incorrect usage. Expected the opcode to be channel_sums or reshape_colmeans or update_ema, but found " + opcode);
		}
		_input1 = in1;
		_input2 = in2;
		_input3 = in3;
		_gputype = GPUINSTRUCTION_TYPE.Dnn;
		_output = out;
		_intermediateMemoryBudget = intermediateMemoryBudget;
	}
	
	public DnnGPUInstruction(CPOperand in1, CPOperand in2, CPOperand in3, CPOperand in4, CPOperand out, String opcode, String istr, 
			double intermediateMemoryBudget) throws DMLRuntimeException {
		super(new ReorgOperator(SwapIndex.getSwapIndexFnObject()), opcode, istr);
		if( !( opcode.equals("update_nesterov_x")) ) {
			throw new DMLRuntimeException("Incorrect opcode: " + opcode);
		}
		_input1 = in1;
		_input2 = in2;
		_input3 = in3;
		_input4 = in4;
		_gputype = GPUINSTRUCTION_TYPE.Dnn;
		_output = out;
		_intermediateMemoryBudget = intermediateMemoryBudget;
	}
	
	public DnnGPUInstruction(CPOperand in1, CPOperand in2, CPOperand in3, CPOperand out, String opcode,
			String istr, ArrayList<CPOperand> stride,
			ArrayList<CPOperand> padding, ArrayList<CPOperand> input_shape,
			ArrayList<CPOperand> filter_shape, double intermediateMemoryBudget) 
	{
		this(in1, in2, out, opcode, istr, stride, padding,  input_shape, filter_shape, intermediateMemoryBudget);
		_input3 = in3;
	}
	
	public DnnGPUInstruction(CPOperand in1, CPOperand in2, CPOperand out, String opcode,
			String istr, ArrayList<CPOperand> stride,
			ArrayList<CPOperand> padding, ArrayList<CPOperand> input_shape,
			ArrayList<CPOperand> filter_shape, double intermediateMemoryBudget) 
	{
		super(new ReorgOperator(SwapIndex.getSwapIndexFnObject()), opcode, istr);
		_gputype = GPUINSTRUCTION_TYPE.Dnn;

		_input1 = in1;
		_input2 = in2;
		_output = out;
		_stride = stride;
		_padding = padding;
		_input_shape = input_shape;
		_filter_shape = filter_shape;
		_intermediateMemoryBudget = intermediateMemoryBudget;
	}

	public DnnGPUInstruction(CPOperand in, CPOperand in2, CPOperand in3, CPOperand in4, CPOperand in5, CPOperand in6,
			CPOperand out, String opcode, String istr, double intermediateMemoryBudget) {
		super(new ReorgOperator(SwapIndex.getSwapIndexFnObject()), opcode, istr);
		if( !opcode.equals("batch_norm2d_test") ) {
			throw new DMLRuntimeException("Incorrect usage. Expected the opcode to be batch_norm2d_test, but found " + opcode);
		}
		_input1 = in;
		_input2 = in2;
		_input3 = in3;
		_input4 = in4;
		_input5 = in5;
		_input6 = in6;
		_gputype = GPUINSTRUCTION_TYPE.Dnn;
		_output = out;
		_intermediateMemoryBudget = intermediateMemoryBudget;
	}
	
	public DnnGPUInstruction(CPOperand in, CPOperand in2, CPOperand in3, CPOperand in4, CPOperand in5, 
			CPOperand out, String opcode, String istr, double intermediateMemoryBudget) {
		super(new ReorgOperator(SwapIndex.getSwapIndexFnObject()), opcode, istr);
		if( !(opcode.equals("update_ema_var") || opcode.equals("batch_norm2d_bwd_dx")) ) {
			throw new DMLRuntimeException("Incorrect usage. Expected the opcode to be update_ema_var or batch_norm2d_bwd_dx, but found " + opcode);
		}
		_input1 = in;
		_input2 = in2;
		_input3 = in3;
		_input4 = in4;
		_input5 = in5;
		_gputype = GPUINSTRUCTION_TYPE.Dnn;
		_output = out;
		_intermediateMemoryBudget = intermediateMemoryBudget;
	}
	
	public static DnnGPUInstruction parseInstruction(String str) {
		String[] parts = InstructionUtils.getInstructionPartsWithValueType(str);
		String opcode = parts[0];
		if( ( opcode.equalsIgnoreCase("conv2d")
			 || opcode.equalsIgnoreCase("conv2d_backward_filter")
			 || opcode.equalsIgnoreCase("conv2d_backward_data")) ) {
			InstructionUtils.checkNumFields(parts, 16);
			CPOperand in1 = new CPOperand(parts[1]);
			CPOperand in2 = new CPOperand(parts[2]);
			CPOperand out = new CPOperand(parts[15]);
			ArrayList<CPOperand> stride = new ArrayList<>();
			ArrayList<CPOperand> padding = new ArrayList<>();
			ArrayList<CPOperand> input_shape = new ArrayList<>();
			ArrayList<CPOperand> filter_shape = new ArrayList<>();
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

			return new DnnGPUInstruction(in1, in2, out, opcode, str, stride,
					padding, input_shape, filter_shape, Double.parseDouble(parts[16]));
		}
		else if( opcode.equalsIgnoreCase("maxpooling_backward") || opcode.equalsIgnoreCase("avgpooling_backward") ) {
			boolean withMaxPoolOut = false;
			if(parts.length == 18) {
				withMaxPoolOut = true;
			}
			else
				InstructionUtils.checkNumFields(parts, 16);
			CPOperand in1 = new CPOperand(parts[1]);
			CPOperand in2 = new CPOperand(parts[2]);
			CPOperand in3 = withMaxPoolOut ? new CPOperand(parts[15]) : null;
			CPOperand out = withMaxPoolOut ? new CPOperand(parts[16]) : new CPOperand(parts[15]);
			double memBudget = withMaxPoolOut ? Double.parseDouble(parts[17]) : Double.parseDouble(parts[16]);
		
			ArrayList<CPOperand> stride = new ArrayList<>();
			ArrayList<CPOperand> padding = new ArrayList<>();
			ArrayList<CPOperand> input_shape = new ArrayList<>();
			ArrayList<CPOperand> filter_shape = new ArrayList<>();
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

			return new DnnGPUInstruction(in1, in2, in3, out, opcode, str, stride,
					padding, input_shape, filter_shape, memBudget);
		}
		else if (opcode.equalsIgnoreCase("conv2d_bias_add")) {
			InstructionUtils.checkNumFields(parts, 17);
			CPOperand in1 = new CPOperand(parts[1]);
			CPOperand in2 = new CPOperand(parts[2]);
			CPOperand in3 = new CPOperand(parts[3]);
			CPOperand out = new CPOperand(parts[16]);
		
			ArrayList<CPOperand> stride = new ArrayList<>();
			ArrayList<CPOperand> padding = new ArrayList<>();
			ArrayList<CPOperand> input_shape = new ArrayList<>();
			ArrayList<CPOperand> filter_shape = new ArrayList<>();
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

			return new DnnGPUInstruction(in1, in2, in3, out, opcode, str, stride,
					padding, input_shape, filter_shape, Double.parseDouble(parts[17]));
		}
		else if (opcode.equalsIgnoreCase("maxpooling") || opcode.equalsIgnoreCase("avgpooling")) {
			InstructionUtils.checkNumFields(parts, 15);
			CPOperand in1 = new CPOperand(parts[1]);
			CPOperand out = new CPOperand(parts[14]);
		
			ArrayList<CPOperand> stride = new ArrayList<>();
			ArrayList<CPOperand> padding = new ArrayList<>();
			ArrayList<CPOperand> input_shape = new ArrayList<>();
			ArrayList<CPOperand> filter_shape = new ArrayList<>();
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

			return new DnnGPUInstruction(in1, null, out, opcode, str, stride,
					padding, input_shape, filter_shape, Double.parseDouble(parts[15]));
		}
		else if( opcode.equalsIgnoreCase("bias_add") || opcode.equalsIgnoreCase("relu_backward") || opcode.equalsIgnoreCase("bias_multiply") 
				|| opcode.equalsIgnoreCase("inv_var") ) {
			InstructionUtils.checkNumFields(parts, 4);
			CPOperand in1 = new CPOperand(parts[1]);
			CPOperand in2 = new CPOperand(parts[2]);
			CPOperand out = new CPOperand(parts[3]);
			return new DnnGPUInstruction(in1, in2, out, opcode, str, Double.parseDouble(parts[4]));
		}
		else if (opcode.equalsIgnoreCase("channel_sums") || opcode.equals("reshape_colmeans") || opcode.equals("update_ema")) {
			InstructionUtils.checkNumFields(parts, 4);
			CPOperand in = new CPOperand(parts[1]);
			CPOperand in2 = new CPOperand(parts[2]);
			CPOperand in3 = new CPOperand(parts[3]);
			CPOperand out = new CPOperand(parts[4]);
			return new DnnGPUInstruction(in, in2, in3, out, opcode, str, 0);
		}
		else if (opcode.equalsIgnoreCase("update_nesterov_x")) {
			InstructionUtils.checkNumFields(parts, 5);
			CPOperand in = new CPOperand(parts[1]);
			CPOperand in2 = new CPOperand(parts[2]);
			CPOperand in3 = new CPOperand(parts[3]);
			CPOperand in4 = new CPOperand(parts[4]);
			CPOperand out = new CPOperand(parts[5]);
			return new DnnGPUInstruction(in, in2, in3, in4, out, opcode, str, 0);
		}
		else if (opcode.equalsIgnoreCase("lstm")) {
			InstructionUtils.checkNumFields(parts, 8);
			CPOperand in1 = new CPOperand(parts[1]);
			CPOperand in2 = new CPOperand(parts[2]);
			CPOperand in3 = new CPOperand(parts[3]);
			CPOperand in4 = new CPOperand(parts[4]);
			CPOperand in5 = new CPOperand(parts[5]);
			CPOperand in6 = new CPOperand(parts[6]);
			CPOperand out = new CPOperand(parts[7]);
			CPOperand out2 = new CPOperand(parts[8]);
			return new DnnGPUInstruction(in1, in2, in3, in4, in5, in6, out, out2, opcode, str, 0);
		}
		else if (opcode.equalsIgnoreCase("lstm_backward")) {
			InstructionUtils.checkNumFields(parts, 13);
			CPOperand in1 = new CPOperand(parts[1]); // image
			CPOperand in2 = new CPOperand(parts[2]); // scale
			CPOperand in3 = new CPOperand(parts[3]); // bias
			CPOperand in4 = new CPOperand(parts[4]); // runningMean
			CPOperand in5 = new CPOperand(parts[5]); // runningVar
			CPOperand in6 = new CPOperand(parts[6]); // mode
			CPOperand in7 = new CPOperand(parts[7]); // epsilon
			CPOperand in8 = new CPOperand(parts[8]); // exponentialAverageFactor
			CPOperand out = new CPOperand(parts[9]);  // ret
			CPOperand out2 = new CPOperand(parts[10]); // retRunningMean
			CPOperand out3 = new CPOperand(parts[11]); // retRunningVar
			CPOperand out4 = new CPOperand(parts[12]); // resultSaveMean
			CPOperand out5 = new CPOperand(parts[13]); // resultSaveInvVariance
			return new DnnGPUInstruction(in1, in2, in3, in4, in5, in6, in7, in8, out, out2, out3, out4, out5, opcode, str, 0);
		}
		else if (opcode.equalsIgnoreCase("batch_norm2d_test")) {
			InstructionUtils.checkNumFields(parts, 7);
			CPOperand in = new CPOperand(parts[1]);
			CPOperand in2 = new CPOperand(parts[2]);
			CPOperand in3 = new CPOperand(parts[3]);
			CPOperand in4 = new CPOperand(parts[4]);
			CPOperand in5 = new CPOperand(parts[5]);
			CPOperand in6 = new CPOperand(parts[6]);
			CPOperand out = new CPOperand(parts[7]);
			return new DnnGPUInstruction(in, in2, in3, in4, in5, in6, out, opcode, str, 0);
		}
		else if (opcode.equalsIgnoreCase("batch_norm2d_bwd_dx")) {
			InstructionUtils.checkNumFields(parts, 6);
			CPOperand in = new CPOperand(parts[1]);
			CPOperand in2 = new CPOperand(parts[2]);
			CPOperand in3 = new CPOperand(parts[3]);
			CPOperand in4 = new CPOperand(parts[4]);
			CPOperand in5 = new CPOperand(parts[5]);
			CPOperand out = new CPOperand(parts[6]);
			return new DnnGPUInstruction(in, in2, in3, in4, in5, out, opcode, str, 0);
		}
		else if (opcode.equalsIgnoreCase("update_ema_var")) {
			InstructionUtils.checkNumFields(parts, 6);
			CPOperand in = new CPOperand(parts[1]);
			CPOperand in2 = new CPOperand(parts[2]);
			CPOperand in3 = new CPOperand(parts[3]);
			CPOperand in4 = new CPOperand(parts[4]);
			CPOperand in5 = new CPOperand(parts[5]);
			CPOperand out = new CPOperand(parts[6]);
			return new DnnGPUInstruction(in, in2, in3, in4, in5, out, opcode, str, 0);
		}
		else {
			throw new DMLRuntimeException("Unknown opcode while parsing a DnnGPUInstruction: " + str);	
		}
	}

	private void processBiasInstruction(String instOpcode, ExecutionContext ec) {
		try(GPUDenseInputPointerFetcher fetcher = new GPUDenseInputPointerFetcher(ec, gCtx, instName, _output)) {
			fetcher.add("input", _input1).add("bias", _input2);
			
			MatrixObject input = fetcher.getInputMatrixObject("input");
			MatrixObject bias = fetcher.getInputMatrixObject("bias");
			MatrixObject out = fetcher.getOutputMatrixObject(input.getNumRows(), input.getNumColumns());
			
			if(instOpcode.equalsIgnoreCase("bias_add"))
				LibMatrixCUDA.biasAdd(gCtx, instName, input, bias, out);
			else if(instOpcode.equalsIgnoreCase("bias_multiply"))
				LibMatrixCUDA.biasMultiply(gCtx, instName, input, bias, out);
		}
	}
	
	private void processInverseVarianceInstruction(String instOpcode, ExecutionContext ec) {
		try(GPUDenseInputPointerFetcher fetcher = new GPUDenseInputPointerFetcher(ec, gCtx, instName, _output)) {
			fetcher.add("X", _input1).addScalar("eps", _input2);
			
			int rows = LibMatrixCUDA.toInt(fetcher.getInputNumRows("X"));
			int cols = LibMatrixCUDA.toInt(fetcher.getInputNumColumns("X"));
			
			// invVar(X, C, eps, size);
			LibMatrixCUDA.getCudaKernels(gCtx).launchKernel("invVar", 
					ExecutionConfig.getConfigForSimpleVectorOperations(rows*cols),
					fetcher.getInputPointer("X"), fetcher.getOutputPointer(rows, cols),
					fetcher.getDouble("eps"), rows*cols);
		}
	}
	
	private void processBatchNorm2dTestInstruction(ExecutionContext ec) throws DMLRuntimeException {
		try(GPUDenseInputPointerFetcher fetcher = new GPUDenseInputPointerFetcher(ec, gCtx, instName, _output)) {
			fetcher.add("image", _input1).add("scale", _input2).add("bias", _input3)
			.add("runningMean", _input4).add("runningVar", _input5).addScalar("epsilon", _input6);
			
			double epsilon = fetcher.getDouble("epsilon");
			if(epsilon < JCudnn.CUDNN_BN_MIN_EPSILON) {
				throw new DMLRuntimeException("The epsilon (" + epsilon + ") cannot be less than CUDNN_BN_MIN_EPSILON=(" + JCudnn.CUDNN_BN_MIN_EPSILON + ")");
			}
			
			MatrixObject image = fetcher.getInputMatrixObject("image");
			LibMatrixCuDNN.batchNormalizationForwardInference(gCtx, instName, 
				image, fetcher.getInputMatrixObject("scale"), fetcher.getInputMatrixObject("bias"), 
				fetcher.getInputMatrixObject("runningMean"), fetcher.getInputMatrixObject("runningVar"), 
				fetcher.getOutputMatrixObject(image.getNumRows(), image.getNumColumns()), epsilon);
		}
	}
	
	private void processBatchNorm2dBackwardDxInstruction(ExecutionContext ec) throws DMLRuntimeException {
		try(GPUDenseInputPointerFetcher fetcher = new GPUDenseInputPointerFetcher(ec, gCtx, instName, _output)) {
			fetcher.add("X", _input1).add("dout", _input2).add("gamma", _input3)
			.add("resultSaveMean", _input4).add("resultSaveInvVariance", _input5);
			
			// #define CUDNN_BN_MIN_EPSILON 1e-5 // Minimum epsilon allowed to be used in the Batch Normalization formula
			double epsilon = 1e-4; 
			MatrixObject image = fetcher.getInputMatrixObject("X");
			LibMatrixCuDNN.batchNormalizationBackwardDX(gCtx, instName, image, 
					fetcher.getInputMatrixObject("dout"), fetcher.getInputMatrixObject("gamma"), 
					fetcher.getOutputMatrixObject(image.getNumRows(), image.getNumColumns()), epsilon, fetcher.getInputMatrixObject("resultSaveMean"), 
					fetcher.getInputMatrixObject("resultSaveInvVariance")); 
		}
	}
	
	
	// (X > 0) * dout
	public void processReLUBackwardInstruction(ExecutionContext ec) {
		try(GPUDenseInputPointerFetcher fetcher = new GPUDenseInputPointerFetcher(ec, gCtx, instName, _output)) {
			fetcher.add("X", _input1).add("dout", _input2);
			MatrixObject X = fetcher.getInputMatrixObject("X");
			LibMatrixCUDA.reluBackward(gCtx, instName, X, 
					fetcher.getInputMatrixObject("dout"), fetcher.getOutputMatrixObject(X.getNumRows(), X.getNumColumns()));
		}
	}
	
	private void processChannelSumsInstruction(ExecutionContext ec) {
		try(GPUDenseInputPointerFetcher fetcher = new GPUDenseInputPointerFetcher(ec, gCtx, instName, _output)) {
			fetcher.add("X", _input1).addScalar("C", _input2).addScalar("HW", _input3);
			int C = fetcher.getInteger("C");
			int HW = fetcher.getInteger("HW");
			fetcher.validateDimensions("X", -1, C*HW);
			LibMatrixCUDA.channelSums(gCtx, instName, 
					fetcher.getInputMatrixObject("X"), 
					fetcher.getOutputMatrixObject(C, 1), C, HW);
		}
	}
	
	private void processEMAInstruction(ExecutionContext ec) {
		// "ema_mean", "mean", "mu"
		try(GPUDenseInputPointerFetcher fetcher = new GPUDenseInputPointerFetcher(ec, gCtx, instName, _output)) {
			fetcher.add("ema_mean", _input1).add("mean", _input2).addScalar("mu", _input3);
			double mu = fetcher.getDouble("mu");
			
			int rows = LibMatrixCUDA.toInt(fetcher.getInputNumRows("ema_mean"));
			int cols = LibMatrixCUDA.toInt(fetcher.getInputNumColumns("ema_mean"));
			
			fetcher.validateDimensions("mean", rows, cols);
			
			// aXplusbY(X, Y, C, a, b, size);
			LibMatrixCUDA.getCudaKernels(gCtx).launchKernel("aXplusbY", 
					ExecutionConfig.getConfigForSimpleVectorOperations(rows*cols),
					fetcher.getInputPointer("ema_mean"), fetcher.getInputPointer("mean"), 
					fetcher.getOutputPointer(rows, cols),
					mu, (1-mu), rows*cols);
		}
	}
	
	private void processReshapeColMeansInstruction(ExecutionContext ec) {
		try(GPUDenseInputPointerFetcher fetcher = new GPUDenseInputPointerFetcher(ec, gCtx, instName, _output)) {
			fetcher.add("X", _input1).addScalar("C", _input2).addScalar("HW", _input3);
			int C = fetcher.getInteger("C");
			int HW = fetcher.getInteger("HW");
			fetcher.validateDimensions("X", -1, C*HW);
			int rows = LibMatrixCUDA.toInt(fetcher.getInputNumRows("X"));
			int cols = LibMatrixCUDA.toInt(fetcher.getInputNumColumns("X"));
			// output = matrix(colMeans(X), rows=C, cols=Hin*Win)
			LibMatrixCUDA.colMeans(gCtx, instName,  
					fetcher.getInputPointer("X"), 
					fetcher.getOutputPointer(C, HW), rows, cols);
		}
	}
	
	private void processUpdateEMAVarInstruction(ExecutionContext ec) {
		try(GPUDenseInputPointerFetcher fetcher = new GPUDenseInputPointerFetcher(ec, gCtx, instName, _output)) {
			// "subgrp_means", "X", "C", "HW", "varConst1"
			fetcher.add("subgrp_means", _input1).add("X", _input2).addScalar("C", _input3)
				.addScalar("HW", _input4).addScalar("varConst1", _input5);
			
			// subgrp_vars = matrix(colVars(X) * varConst1, rows=C, cols=Hin*Win)
			// var = rowMeans(subgrp_vars) + rowVars(subgrp_means)*(((Hin*Win)-1)/(Hin*Win))
			// --->
			// subgrp_vars = matrix(colVars(X), rows=C, cols=HW)
			// var = rowMeans(subgrp_vars)*varConst1 + rowVars(subgrp_means)*((HW-1)/HW)  
			int C = fetcher.getInteger("C");
			int HW = fetcher.getInteger("HW");
			double varConst1 = fetcher.getDouble("varConst1");
			fetcher.validateDimensions("subgrp_means", C, HW);
			fetcher.validateDimensions("X", -1, C*HW);
			
			Pointer subgrp_vars = gCtx.allocate(instName, C*HW*LibMatrixCUDA.sizeOfDataType);
			// subgrp_vars <- colVars(X)
			LibMatrixCUDA.colVars(gCtx, instName, fetcher.getInputPointer("X"), subgrp_vars, 
					LibMatrixCUDA.toInt(fetcher.getInputNumRows("X")), C*HW);
			
			// tmp1 <- rowMeans(subgrp_vars)
			Pointer tmp1 = gCtx.allocate(instName, C*LibMatrixCUDA.sizeOfDataType);
			LibMatrixCUDA.rowMeans(gCtx, instName, subgrp_vars, tmp1, C, HW);
			gCtx.cudaFreeHelper(instName, subgrp_vars, gCtx.EAGER_CUDA_FREE);
			
			// out <- rowVars(subgrp_means)
			Pointer out = fetcher.getOutputPointer(C, 1);
			LibMatrixCUDA.rowVars(gCtx, instName, fetcher.getInputPointer("subgrp_means"), out, C, HW);
			
			// var = tmp1*varConst1 + out*((HW-1)/HW)
			LibMatrixCUDA.getCudaKernels(gCtx).launchKernel("aXplusbC", 
					ExecutionConfig.getConfigForSimpleVectorOperations(C),
					tmp1, out,
					varConst1, (((double)HW-1)/HW), C);
			gCtx.cudaFreeHelper(instName, tmp1, gCtx.EAGER_CUDA_FREE);
		}
	}
	
	
	
	private void processNesterovUpdateInstruction(ExecutionContext ec) {
		try(GPUDenseInputPointerFetcher fetcher = new GPUDenseInputPointerFetcher(ec, gCtx, instName, _output)) {
			fetcher.add("input", _input1).add("v", _input2).add("v_prev", _input3)
			.addScalar("mu", _input4);
			MatrixObject input = fetcher.getInputMatrixObject("input");
			int rows = LibMatrixCUDA.toInt(input.getNumRows());
			int cols = LibMatrixCUDA.toInt(input.getNumColumns());
			
			LibMatrixCUDA.getCudaKernels(gCtx).launchKernel("update_nesterov_x", 
					ExecutionConfig.getConfigForSimpleVectorOperations(LibMatrixCUDA.toInt(rows*cols)),
					fetcher.getInputPointer("input"), 
					fetcher.getInputPointer("v"),
					fetcher.getInputPointer("v_prev"),
					fetcher.getDouble("mu"), 
					fetcher.getOutputPointer(rows, cols),
					rows*cols);
		}
	}
	
	private static int toInt(long num) throws DMLRuntimeException {
		if(num >= Integer.MAX_VALUE || num <= Integer.MIN_VALUE) {
			throw new DMLRuntimeException("GPU : Exceeded supported size " + num);
		}
		return (int)num;
	}
	
	private void processLstmBackwardInstruction(ExecutionContext ec) throws DMLRuntimeException {
		MatrixObject out0 = getMatrixInputForGPUInstruction(ec, _input4.getName());
		int M = toInt(out0.getNumColumns()); // hiddenSize .. since out0: (N, M)
		Pointer out0Pointer =  LibMatrixCUDA.getDensePointer(gCtx, out0, instName);
		
		MatrixObject W = getMatrixInputForGPUInstruction(ec, _input2.getName());
		MatrixObject bias = getMatrixInputForGPUInstruction(ec, _input3.getName());
		long numRowsW = W.getNumRows();
		int D = toInt(numRowsW) - M; // since W:(D+M, 4M) ... numFeatures 
		Pointer sysmlWPointer = LibMatrixCuDNN.getDensePointerForCuDNN(gCtx, W, instName, D+M, 4*M);
		Pointer sysmlBiasPointer = LibMatrixCuDNN.getDensePointerForCuDNN(gCtx, bias, instName, 1, 4*M);
		Pointer cudnnWPointer = gCtx.allocate(instName, (D+M+2)*(4*M)*LibMatrixCUDA.sizeOfDataType);
		LibMatrixCUDA.getCudaKernels(gCtx).launchKernel("prepare_lstm_weight",
				ExecutionConfig.getConfigForSimpleVectorOperations((D+M+2)*(4*M)),
				sysmlWPointer, sysmlBiasPointer, cudnnWPointer, D, M);
		ec.releaseMatrixInputForGPUInstruction(_input2.getName());
		ec.releaseMatrixInputForGPUInstruction(_input3.getName());
		
		
		MatrixObject X = getMatrixInputForGPUInstruction(ec, _input1.getName());
		Pointer xPointer = LibMatrixCUDA.getDensePointer(gCtx, X, instName); 
		int N = toInt(X.getNumRows()); // batchSize .. since X:(N, T*D)
		long numColsX = X.getNumColumns();
		int T = toInt(numColsX/ D); // since X:(N, T*D) ... seqLength
		Pointer cudnnInput = gCtx.allocate(instName, (N*T*D)*LibMatrixCUDA.sizeOfDataType);
		LibMatrixCUDA.getCudaKernels(gCtx).launchKernel("prepare_lstm_input",
				ExecutionConfig.getConfigForSimpleVectorOperations(N*T*D),
				xPointer, cudnnInput, N, D, T*D, N*T*D);
		ec.releaseMatrixInputForGPUInstruction(_input1.getName());
		
		Pointer c0Pointer = LibMatrixCUDA.getDensePointer(gCtx, getMatrixInputForGPUInstruction(ec, _input5.getName()), instName);
		boolean return_sequences = ec.getScalarInput(_input6.getName(), _input6.getValueType(), _input6.isLiteral()).getBooleanValue();
		
		// LibMatrixCuDNN.lstm(ec, gCtx, instName, 
				// cudnnInput, cudnnWPointer, out0Pointer, c0Pointer, return_sequences, _output.getName(), _output2.getName(), N, M, D, T);
				// String xName, Pointer hx, Pointer cx, Pointer wPointer, String doutName, String dcyName,  // input
				// String dxName, String dwName, String dbName, String dhxName, String dcxName,  	// output
		String dxName = _output.getName();
		String dwName = _output2.getName();
		String dbName = _output3.getName();
		String dhxName = _output4.getName();
		String dcxName = _output5.getName();
		String doutName = _input7.getName();
		String dcyName = _input8.getName();
		LibMatrixCuDNN.lstmBackward(ec, gCtx, instName, 
				cudnnInput, out0Pointer, c0Pointer, cudnnWPointer, doutName, dcyName,  // input
				dxName, dwName, dbName, dhxName, dcxName, // output 
				return_sequences, N, M, D, T);
		gCtx.cudaFreeHelper(instName, cudnnWPointer, gCtx.EAGER_CUDA_FREE);
		gCtx.cudaFreeHelper(instName, cudnnInput, gCtx.EAGER_CUDA_FREE);
		
		// release inputs/outputs
		ec.releaseMatrixInputForGPUInstruction(_input4.getName());
		ec.releaseMatrixInputForGPUInstruction(_input5.getName());
	}
	
	private void processLstmInstruction(ExecutionContext ec) throws DMLRuntimeException {
		// batchSize=N, seqLength=T, numFeatures=D and hiddenSize=M
		// input  X:(N, T*D), 	==> (T, D, N)
		// weight W:(D+M+2, 4M) 
		// previous output out0 (also represented by hx) and cell state c0 (also represented by cx): (N, M) ==> (1, M, N)
		// out: (N, T*M) or (N, M) ==> (T, M, N)
		MatrixObject out0 = getMatrixInputForGPUInstruction(ec, _input4.getName());
		int M = toInt(out0.getNumColumns()); // hiddenSize .. since out0: (N, M)
		Pointer out0Pointer =  LibMatrixCUDA.getDensePointer(gCtx, out0, instName);
		
		MatrixObject W = getMatrixInputForGPUInstruction(ec, _input2.getName());
		MatrixObject bias = getMatrixInputForGPUInstruction(ec, _input3.getName());
		long numRowsW = W.getNumRows();
		int D = toInt(numRowsW) - M; // since W:(D+M, 4M) ... numFeatures 
		Pointer sysmlWPointer = LibMatrixCuDNN.getDensePointerForCuDNN(gCtx, W, instName, D+M, 4*M);
		Pointer sysmlBiasPointer = LibMatrixCuDNN.getDensePointerForCuDNN(gCtx, bias, instName, 1, 4*M);
		Pointer cudnnWPointer = gCtx.allocate(instName, (D+M+2)*(4*M)*LibMatrixCUDA.sizeOfDataType);
		LibMatrixCUDA.getCudaKernels(gCtx).launchKernel("prepare_lstm_weight",
				ExecutionConfig.getConfigForSimpleVectorOperations((D+M+2)*(4*M)),
				sysmlWPointer, sysmlBiasPointer, cudnnWPointer, D, M);
		ec.releaseMatrixInputForGPUInstruction(_input2.getName());
		ec.releaseMatrixInputForGPUInstruction(_input3.getName());
		
		boolean return_sequences = ec.getScalarInput(_input6.getName(), _input6.getValueType(), _input6.isLiteral()).getBooleanValue();
		
		// Beause the matrices are released immediately, the output for transpose need not be taken into account
		MatrixObject X = getMatrixInputForGPUInstruction(ec, _input1.getName());
		Pointer xPointer = LibMatrixCUDA.getDensePointer(gCtx, X, instName); 
		int N = toInt(X.getNumRows()); // batchSize .. since X:(N, T*D)
		long numColsX = X.getNumColumns();
		int T = toInt(numColsX/ D); // since X:(N, T*D) ... seqLength
		Pointer cudnnInput = gCtx.allocate(instName, (N*T*D)*LibMatrixCUDA.sizeOfDataType);
		LibMatrixCUDA.getCudaKernels(gCtx).launchKernel("prepare_lstm_input",
				ExecutionConfig.getConfigForSimpleVectorOperations(N*T*D),
				xPointer, cudnnInput, N, D, T*D, N*T*D);
		ec.releaseMatrixInputForGPUInstruction(_input1.getName());
		
		Pointer c0Pointer = LibMatrixCUDA.getDensePointer(gCtx, getMatrixInputForGPUInstruction(ec, _input5.getName()), instName); 
		
		LibMatrixCuDNN.lstm(ec, gCtx, instName, cudnnInput, cudnnWPointer, out0Pointer, c0Pointer, return_sequences, _output.getName(), _output2.getName(), N, M, D, T);
		gCtx.cudaFreeHelper(instName, cudnnWPointer, gCtx.EAGER_CUDA_FREE);
		gCtx.cudaFreeHelper(instName, cudnnInput, gCtx.EAGER_CUDA_FREE);
		
		// release inputs/outputs
		ec.releaseMatrixInputForGPUInstruction(_input4.getName());
		ec.releaseMatrixInputForGPUInstruction(_input5.getName());
		ec.releaseMatrixOutputForGPUInstruction(_output2.getName());
		ec.releaseMatrixOutputForGPUInstruction(_output.getName());
	}
	
	@Override
	public void processInstruction(ExecutionContext ec) {
		GPUStatistics.incrementNoOfExecutedGPUInst();
		gCtx = ec.getGPUContext(0);
		instName = getExtendedOpcode();
		if (instOpcode.equalsIgnoreCase("bias_add") || instOpcode.equalsIgnoreCase("bias_multiply")) {
			processBiasInstruction(instOpcode, ec);
			return;
		}
		else if (instOpcode.equalsIgnoreCase("inv_var")) {
			processInverseVarianceInstruction(instOpcode, ec);
			return;
		}
		else if (instOpcode.equalsIgnoreCase("relu_backward")) {
			processReLUBackwardInstruction(ec);
			return;
		}
		else if (instOpcode.equalsIgnoreCase("channel_sums")) {
			processChannelSumsInstruction(ec);
			return;
		}
		else if (instOpcode.equalsIgnoreCase("update_ema")) {
			processEMAInstruction(ec);
			return;
		}
		else if (instOpcode.equalsIgnoreCase("reshape_colmeans")) {
			processReshapeColMeansInstruction(ec);
			return;
		}
		else if (instOpcode.equalsIgnoreCase("update_nesterov_x")) {
			processNesterovUpdateInstruction(ec);
			return;
		}
		else if (instOpcode.equalsIgnoreCase("update_ema_var")) {
			processUpdateEMAVarInstruction(ec);
			return;
		}
		else if (instOpcode.equalsIgnoreCase("lstm")) {
			processLstmInstruction(ec);
			return;
		}
		else if (instOpcode.equalsIgnoreCase("lstm_backward")) {
			processLstmBackwardInstruction(ec);
			return;
		}
		else if (instOpcode.equalsIgnoreCase("batch_norm2d_test")) {
			processBatchNorm2dTestInstruction(ec);
			return;
		}
		else if (instOpcode.equalsIgnoreCase("batch_norm2d_bwd_dx")) {
			processBatchNorm2dBackwardDxInstruction(ec);
			return;
		}
					
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
		
		int P = (int) DnnUtils.getP(H, R, stride_h, pad_h);
		int Q = (int) DnnUtils.getQ(W, S, stride_w, pad_w);
		
		if (instOpcode.equalsIgnoreCase("conv2d")) {
			MatrixObject image = getMatrixInputForGPUInstruction(ec, _input1.getName());
			MatrixObject filter = getMatrixInputForGPUInstruction(ec, _input2.getName());

			if(image.getNumRows() != N || image.getNumColumns() != C*H*W) 
				throw new DMLRuntimeException("Incorrect dimensions for image in conv2d");
			if(filter.getNumRows() != K || filter.getNumColumns() != C*R*S) 
				throw new DMLRuntimeException("Incorrect dimensions for filter in conv2d");
			
			MatrixObject out = getDenseMatrixOutputForGPUInstruction(ec, _output.getName(), N, K * P * Q);
			
			LibMatrixCuDNN.conv2d(ec.getGPUContext(0), getExtendedOpcode(), image, filter, out, N, C, H, W,
					K, R, S, pad_h, pad_w, stride_h, stride_w, P, Q, _intermediateMemoryBudget);
		}
		else if (instOpcode.equalsIgnoreCase("conv2d_bias_add")) {
			MatrixObject image = getMatrixInputForGPUInstruction(ec, _input1.getName());
			MatrixObject bias = getMatrixInputForGPUInstruction(ec, _input2.getName());
			MatrixObject filter = getMatrixInputForGPUInstruction(ec, _input3.getName());

			if(image.getNumRows() != N || image.getNumColumns() != C*H*W) 
				throw new DMLRuntimeException("Incorrect dimensions for image in conv2d");
			if(filter.getNumRows() != K || filter.getNumColumns() != C*R*S) 
				throw new DMLRuntimeException("Incorrect dimensions for filter in conv2d");
			
			MatrixObject out = getDenseMatrixOutputForGPUInstruction(ec, _output.getName(), N, K * P * Q);
			
			LibMatrixCuDNN.conv2dBiasAdd(ec.getGPUContext(0), getExtendedOpcode(), image, bias, filter, out, N, C, H, W,
						K, R, S, pad_h, pad_w, stride_h, stride_w, P, Q, _intermediateMemoryBudget);
		}
		else if (instOpcode.equalsIgnoreCase("conv2d_backward_filter")) {
			MatrixObject image = getMatrixInputForGPUInstruction(ec, _input1.getName());
			MatrixObject dout = getMatrixInputForGPUInstruction(ec, _input2.getName());

			if(image.getNumRows() != N || image.getNumColumns() != C*H*W) 
				throw new DMLRuntimeException("Incorrect dimensions for image in conv2d_backward_filter");
			if(dout.getNumRows() != N || dout.getNumColumns() != K*P*Q) 
				throw new DMLRuntimeException("Incorrect dimensions for dout in conv2d_backward_filter: " + 
						dout.getNumRows() + " != " +  N + " || " + dout.getNumColumns() + " != " + K*P*Q);
			
			MatrixObject out = getDenseMatrixOutputForGPUInstruction(ec, _output.getName(), K, C * R * S);
			
			LibMatrixCuDNN.conv2dBackwardFilter(ec.getGPUContext(0), getExtendedOpcode(), image, dout, out, N, C, H, W,
					K, R, S, pad_h, pad_w, stride_h, stride_w, P, Q, _intermediateMemoryBudget);
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
			
			MatrixObject out = getDenseMatrixOutputForGPUInstruction(ec, _output.getName(), N, C * H * W);
			
			LibMatrixCuDNN.conv2dBackwardData(ec.getGPUContext(0), getExtendedOpcode(), filter, dout, out, N, C, H, W,
					K, R, S, pad_h, pad_w, stride_h, stride_w, P, Q, _intermediateMemoryBudget);
		}
		else if (instOpcode.equalsIgnoreCase("maxpooling") || instOpcode.equalsIgnoreCase("avgpooling")) {
			MatrixObject image = getMatrixInputForGPUInstruction(ec, _input1.getName());

			if(image.getNumRows() != N || image.getNumColumns() != C*H*W) 
				throw new DMLRuntimeException("Incorrect dimensions for image in maxpooling: " + 
						image.getNumRows() + " != " +  N + " || " + image.getNumColumns() + " != " + C*H*W);
			
			MatrixObject out = getDenseMatrixOutputForGPUInstruction(ec, _output.getName(), N, C * P * Q);
			PoolingType poolType = instOpcode.equalsIgnoreCase("maxpooling") ? PoolingType.MAX : PoolingType.AVG;
			LibMatrixCuDNN.pooling(ec.getGPUContext(0), getExtendedOpcode(), image, out, N, C, H, W,
					K, R, S, pad_h, pad_w, stride_h, stride_w, P, Q, poolType, _intermediateMemoryBudget);
		}
		else if (instOpcode.equalsIgnoreCase("maxpooling_backward") || instOpcode.equalsIgnoreCase("avgpooling_backward")) {
			MatrixObject image = getMatrixInputForGPUInstruction(ec, _input1.getName());
			MatrixObject dout = getMatrixInputForGPUInstruction(ec, _input2.getName());
			MatrixObject maxPoolOutput = _input3 != null ? getMatrixInputForGPUInstruction(ec, _input3.getName()) : null;
			if(dout.getNumRows() != N || dout.getNumColumns() != C*P*Q) 
				throw new DMLRuntimeException("Incorrect dimensions for dout in maxpooling_backward");
			if(image.getNumRows() != N || image.getNumColumns() != C*H*W) 
				throw new DMLRuntimeException("Incorrect dimensions for image in maxpooling_backward: " + 
						image.getNumRows() + " != " +  N + " || " + image.getNumColumns() + " != " + K*P*Q);
			
			MatrixObject out = getDenseMatrixOutputForGPUInstruction(ec, _output.getName(), N, C * H * W);
			PoolingType poolType = instOpcode.equalsIgnoreCase("maxpooling_backward") ? PoolingType.MAX : PoolingType.AVG;
			LibMatrixCuDNN.poolingBackward(ec.getGPUContext(0), getExtendedOpcode(), image, dout, maxPoolOutput, out, N, C, H, W,
					K, R, S, pad_h, pad_w, stride_h, stride_w, P, Q, poolType, _intermediateMemoryBudget);
		}
		else {
			throw new DMLRuntimeException("Unsupported GPU context for " + instOpcode);
		}
		
		// release inputs/outputs
		ec.releaseMatrixInputForGPUInstruction(_input1.getName());
		
		boolean isPool = instOpcode.equalsIgnoreCase("maxpooling") || instOpcode.equalsIgnoreCase("avgpooling");
		boolean isPoolBackward = instOpcode.equalsIgnoreCase("maxpooling_backward") || instOpcode.equalsIgnoreCase("avgpooling_backward");

		if ( !isPool )
			ec.releaseMatrixInputForGPUInstruction(_input2.getName());

		if (instOpcode.equalsIgnoreCase("conv2d_bias_add") || 
			(isPoolBackward && _input3 != null))
			ec.releaseMatrixInputForGPUInstruction(_input3.getName());

		ec.releaseMatrixOutputForGPUInstruction(_output.getName());
	}


	private static int getScalarInput(ExecutionContext ec, ArrayList<CPOperand> aL, int index) {
		return (int) ec.getScalarInput(aL.get(index).getName(),
			aL.get(index).getValueType(), aL.get(index).isLiteral()).getLongValue();
	}
}
