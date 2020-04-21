/*
 * Modifications Copyright 2019 Graz University of Technology
 *
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

package org.tugraz.sysds.runtime.instructions.cp;

import org.tugraz.sysds.common.Types;
import org.tugraz.sysds.common.Types.ValueType;
import org.tugraz.sysds.runtime.DMLRuntimeException;
import org.tugraz.sysds.runtime.controlprogram.context.ExecutionContext;
import org.tugraz.sysds.runtime.data.LibTensorReorg;
import org.tugraz.sysds.runtime.data.TensorBlock;
import org.tugraz.sysds.runtime.instructions.InstructionUtils;
import org.tugraz.sysds.runtime.matrix.data.LibMatrixReorg;
import org.tugraz.sysds.runtime.matrix.data.MatrixBlock;
import org.tugraz.sysds.runtime.matrix.operators.Operator;
import org.tugraz.sysds.runtime.util.DataConverter;

public class ReshapeCPInstruction extends UnaryCPInstruction {
	private final CPOperand _opRows;
	private final CPOperand _opCols;
	private final CPOperand _opDims;
	private final CPOperand _opByRow;

	private ReshapeCPInstruction(Operator op, CPOperand in1, CPOperand in2, CPOperand in3,
		CPOperand in4, CPOperand in5, CPOperand out, String opcode, String istr) {
		super(CPType.Reshape, op, in1, out, opcode, istr);
		_opRows = in2;
		_opCols = in3;
		_opDims = in4;
		_opByRow = in5;
	}

	public static ReshapeCPInstruction parseInstruction (String str ) {
		String[] parts = InstructionUtils.getInstructionPartsWithValueType(str);
		InstructionUtils.checkNumFields( parts, 6 );
		String opcode = parts[0];
		CPOperand in1 = new CPOperand(parts[1]);
		CPOperand in2 = new CPOperand(parts[2]);
		CPOperand in3 = new CPOperand(parts[3]);
		CPOperand in4 = new CPOperand(parts[4]);
		CPOperand in5 = new CPOperand(parts[5]);
		CPOperand out = new CPOperand(parts[6]);
		if(!opcode.equalsIgnoreCase("rshape"))
			throw new DMLRuntimeException("Unknown opcode while parsing an ReshapeInstruction: " + str);
		else
			return new ReshapeCPInstruction(new Operator(true), in1, in2, in3, in4, in5, out, opcode, str);
	}
	
	@Override
	public void processInstruction(ExecutionContext ec) {
		if (output.getDataType() == Types.DataType.TENSOR) {
			int[] dims = DataConverter.getTensorDimensions(ec, _opDims);
			TensorBlock out = new TensorBlock(output.getValueType(), dims);
			if (input1.getDataType() == Types.DataType.TENSOR) {
				//get Tensor-data from tensor (reshape)
				// TODO support DataTensor
				TensorBlock data = ec.getTensorInput(input1.getName());
				LibTensorReorg.reshape(data.getBasicTensor(), out.getBasicTensor(), dims);
				ec.releaseTensorInput(input1.getName());
			}
			else if (input1.getDataType() == Types.DataType.MATRIX) {
				out.allocateBlock();
				//get Tensor-data from matrix
				MatrixBlock data = ec.getMatrixInput(input1.getName());
				// TODO metadata operation
				out.getBasicTensor().set(data);
				ec.releaseMatrixInput(input1.getName());
			}
			else {
				// TODO support frame and list. Before we implement list it might be good to implement heterogeneous tensors
				throw new DMLRuntimeException("ReshapeInstruction only supports tensor and matrix as data parameter.");
			}
			ec.setTensorOutput(output.getName(), out);
		}
		else {
			//get inputs
			MatrixBlock in = ec.getMatrixInput(input1.getName());
			int rows = (int) ec.getScalarInput(_opRows).getLongValue(); //save cast
			int cols = (int) ec.getScalarInput(_opCols).getLongValue(); //save cast
			BooleanObject byRow = (BooleanObject) ec.getScalarInput(_opByRow.getName(), ValueType.BOOLEAN, _opByRow.isLiteral());

			//execute operations
			MatrixBlock out = new MatrixBlock();
			LibMatrixReorg.reshape(in, out, rows, cols, byRow.getBooleanValue());

			//set output and release inputs
			ec.setMatrixOutput(output.getName(), out);
			ec.releaseMatrixInput(input1.getName());
		}
	}
}
