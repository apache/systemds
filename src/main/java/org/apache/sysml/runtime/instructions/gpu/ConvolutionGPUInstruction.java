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

import org.apache.sysml.parser.Expression.DataType;
import org.apache.sysml.parser.Expression.ValueType;
import org.apache.sysml.runtime.DMLRuntimeException;
import org.apache.sysml.runtime.controlprogram.caching.MatrixObject;
import org.apache.sysml.runtime.controlprogram.context.ExecutionContext;
import org.apache.sysml.runtime.functionobjects.SwapIndex;
import org.apache.sysml.runtime.instructions.InstructionUtils;
import org.apache.sysml.runtime.instructions.cp.CPOperand;
import org.apache.sysml.runtime.instructions.cp.UnaryCPInstruction;
import org.apache.sysml.runtime.matrix.data.LibMatrixCUDA;
import org.apache.sysml.runtime.matrix.operators.ReorgOperator;
import org.apache.sysml.runtime.util.ConvolutionUtils;
import org.apache.sysml.utils.Statistics;

public class ConvolutionGPUInstruction extends UnaryCPInstruction {
	
	private CPOperand _in2; 
	private ArrayList<CPOperand> _input_shape;
	private ArrayList<CPOperand> _filter_shape;
	private ArrayList<CPOperand> _stride = new ArrayList<CPOperand>();
	private ArrayList<CPOperand> _padding = new ArrayList<CPOperand>();
	
	int N; int C; int H; int W;
	int K; int R; int S; int stride_h; int stride_w; int pad_h; int pad_w;
	int P; int Q;
	
	public ConvolutionGPUInstruction(CPOperand in, CPOperand in2, CPOperand out, String opcode,
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
	
	public static ConvolutionGPUInstruction parseInstruction(String str)
			throws DMLRuntimeException {
		CPOperand in = new CPOperand("", ValueType.UNKNOWN, DataType.UNKNOWN);
		CPOperand out = new CPOperand("", ValueType.UNKNOWN, DataType.UNKNOWN);

		String[] parts = InstructionUtils.getInstructionPartsWithValueType(str);
		String opcode = parts[0];
		if (opcode.equalsIgnoreCase("conv2d") ||
				opcode.equalsIgnoreCase("conv2d_backward_filter") ||
				opcode.equalsIgnoreCase("conv2d_backward_data")) {
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
	
			return new ConvolutionGPUInstruction(in, in2, out, opcode, str, stride,
					padding, input_shape, filter_shape);
		} 
		else {
			throw new DMLRuntimeException("Unknown opcode while parsing a ConvolutionGPUInstruction: " + str);
		}
	}
	
	private int getScalarInput(ExecutionContext ec, ArrayList<CPOperand> aL,
			int index) throws DMLRuntimeException {
		return (int) ec.getScalarInput(aL.get(index).getName(),
				aL.get(index).getValueType(), aL.get(index).isLiteral())
				.getLongValue();
//		try {
//			// TODO: Temporary fix cases where we use ¶n¶·SCALAR·INT·false
//			// Need to investigate why n is not marked as literal !!!
//			return (int) Double.parseDouble(aL.get(index).getName());
//		} catch(Exception e) {
//			return (int) ec.getScalarInput(aL.get(index).getName(),
//					aL.get(index).getValueType(), aL.get(index).isLiteral())
//					.getLongValue();
//		}
	}
	
	@Override
	public void processInstruction(ExecutionContext ec) 
			throws DMLRuntimeException {
		
		Statistics.incrementNoOfExecutedGPUInst();
		
		MatrixObject out = null;
		if (instOpcode.equalsIgnoreCase("conv2d") || 
				instOpcode.equalsIgnoreCase("conv2d_backward_filter") ||
				instOpcode.equalsIgnoreCase("conv2d_backward_data")) {
			
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
			
			if (instOpcode.equalsIgnoreCase("conv2d")) {
				MatrixObject image = ec.getMatrixInputForGPUInstruction(input1.getName());
				MatrixObject filter = ec.getMatrixInputForGPUInstruction(_in2.getName());
				if(image.getMatrixBlock().isInSparseFormat() || filter.getMatrixBlock().isInSparseFormat()) {
					throw new DMLRuntimeException("Sparse convolution not implemented");
				}
				if(image.getNumRows() != N || image.getNumColumns() != C*H*W) 
					throw new DMLRuntimeException("Incorrect dimensions for image in conv2d");
				if(filter.getNumRows() != K || filter.getNumColumns() != C*R*S) 
					throw new DMLRuntimeException("Incorrect dimensions for filter in conv2d");
				
				out = ec.getDenseMatrixOutputForGPUInstruction(output.getName(), N, K * P * Q);
				LibMatrixCUDA.conv2d(image, filter, out, N, C, H, W,
						K, R, S, pad_h, pad_w, stride_h, stride_w, P, Q);
				
			}
			else if (instOpcode.equalsIgnoreCase("conv2d_backward_filter")) {
				MatrixObject image = ec.getMatrixInputForGPUInstruction(input1.getName());
				MatrixObject dout = ec.getMatrixInputForGPUInstruction(_in2.getName());
				if(image.getMatrixBlock().isInSparseFormat() || dout.getMatrixBlock().isInSparseFormat())
					throw new DMLRuntimeException("Sparse convolution_backward_filter not implemented");
				if(image.getNumRows() != N || image.getNumColumns() != C*H*W) 
					throw new DMLRuntimeException("Incorrect dimensions for image in conv2d_backward_filter");
				if(dout.getNumRows() != N || dout.getNumColumns() != K*P*Q) 
					throw new DMLRuntimeException("Incorrect dimensions for dout in conv2d_backward_filter: " + 
							dout.getNumRows() + " != " +  N + " || " + dout.getNumColumns() + " != " + K*P*Q);
				
				out = ec.getDenseMatrixOutputForGPUInstruction(output.getName(), K, C * R * S);
				LibMatrixCUDA.conv2d_backward_filter(image, dout, out, N, C, H, W,
						K, R, S, pad_h, pad_w, stride_h, stride_w, P, Q);
				// TODO: For now always copy the device data to host
				// ec.gpuCtx.copyDeviceToHost(outputBlock);
			}
			else if (instOpcode.equalsIgnoreCase("conv2d_backward_data")) {
				MatrixObject filter = ec.getMatrixInputForGPUInstruction(input1.getName());
				MatrixObject dout = ec.getMatrixInputForGPUInstruction(_in2.getName());
				if(filter.getMatrixBlock().isInSparseFormat() || dout.getMatrixBlock().isInSparseFormat())
					throw new DMLRuntimeException("Sparse convolution_backward_data not implemented");
				if(filter.getNumRows() != K || filter.getNumColumns() != C*R*S) 
					throw new DMLRuntimeException("Incorrect dimensions for filter in convolution_backward_data");
				if(dout.getNumRows() != N || dout.getNumColumns() != K*P*Q) 
					throw new DMLRuntimeException("Incorrect dimensions for dout in conv2d_backward_data: " + 
							dout.getNumRows() + " != " +  N + " || " + dout.getNumColumns() + " != " + K*P*Q);
				
				out = ec.getDenseMatrixOutputForGPUInstruction(output.getName(), N, C * H * W);
				LibMatrixCUDA.conv2d_backward_data(filter, dout, out, N, C, H, W,
						K, R, S, pad_h, pad_w, stride_h, stride_w, P, Q);
			}
			else {
				throw new DMLRuntimeException("Unsupported GPU context for " + instOpcode);
			}
		}
		else {
			throw new DMLRuntimeException("Unsupported op code " + instOpcode);
		}
		// release inputs/outputs
		ec.releaseMatrixInputForGPUInstruction(input1.getName());
		ec.releaseMatrixInputForGPUInstruction(_in2.getName());
		ec.releaseMatrixOutputForGPUInstruction(output.getName());
	}
	
}