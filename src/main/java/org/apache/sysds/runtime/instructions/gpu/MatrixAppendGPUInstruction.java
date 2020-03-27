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

package org.apache.sysds.runtime.instructions.gpu;

import org.apache.sysds.common.Types.DataType;
import org.apache.sysds.runtime.DMLRuntimeException;
import org.apache.sysds.runtime.controlprogram.caching.MatrixObject;
import org.apache.sysds.runtime.controlprogram.context.ExecutionContext;
import org.apache.sysds.runtime.functionobjects.OffsetColumnIndex;
import org.apache.sysds.runtime.instructions.InstructionUtils;
import org.apache.sysds.runtime.instructions.cp.AppendCPInstruction;
import org.apache.sysds.runtime.instructions.cp.CPOperand;
import org.apache.sysds.runtime.matrix.data.LibMatrixCUDA;
import org.apache.sysds.runtime.matrix.operators.Operator;
import org.apache.sysds.runtime.matrix.operators.ReorgOperator;
import org.apache.sysds.utils.GPUStatistics;

/**
 * Implements the cbind and rbind functions for matrices
 */
public class MatrixAppendGPUInstruction extends GPUInstruction {

	CPOperand output;
	CPOperand input1, input2;
	AppendCPInstruction.AppendType atype;

	private MatrixAppendGPUInstruction(Operator op, CPOperand in1, CPOperand in2, CPOperand out,
			AppendCPInstruction.AppendType type, String opcode, String istr) {
		super(op, opcode, istr);
		this.output = out;
		this.input1 = in1;
		this.input2 = in2;
		this.atype = type;
	}

	public static MatrixAppendGPUInstruction parseInstruction ( String str ) {
		String[] parts = InstructionUtils.getInstructionPartsWithValueType(str);
		InstructionUtils.checkNumFields (parts, 5);
		String opcode = parts[0];
		CPOperand in1 = new CPOperand(parts[1]);
		CPOperand in2 = new CPOperand(parts[2]);
		@SuppressWarnings("unused")
		CPOperand in3 = new CPOperand(parts[3]);
		CPOperand out = new CPOperand(parts[4]);
		boolean cbind = Boolean.parseBoolean(parts[5]);
		AppendCPInstruction.AppendType type = (in1.getDataType()!= DataType.MATRIX && in1.getDataType()!= DataType.FRAME) ?
				AppendCPInstruction.AppendType.STRING : cbind ? AppendCPInstruction.AppendType.CBIND : AppendCPInstruction.AppendType.RBIND;
		if (in1.getDataType()!= DataType.MATRIX || in2.getDataType()!= DataType.MATRIX){
			throw new DMLRuntimeException("GPU : Error in internal state - Append was called on data other than matrices");
		}
		if(!opcode.equalsIgnoreCase("append"))
			throw new DMLRuntimeException("Unknown opcode while parsing a AppendCPInstruction: " + str);
		Operator op = new ReorgOperator(OffsetColumnIndex.getOffsetColumnIndexFnObject(-1));
		return new MatrixAppendGPUInstruction(op, in1, in2, out, type, opcode, str);
	}

	@Override
	public void processInstruction(ExecutionContext ec) {
		GPUStatistics.incrementNoOfExecutedGPUInst();
		String opcode = getOpcode();
		MatrixObject mat1 = getMatrixInputForGPUInstruction(ec, input1.getName());
		MatrixObject mat2 = getMatrixInputForGPUInstruction(ec, input2.getName());
		if(atype == AppendCPInstruction.AppendType.CBIND)
			LibMatrixCUDA.cbind(ec, ec.getGPUContext(0), getExtendedOpcode(), mat1, mat2, output.getName());
		else if (atype == AppendCPInstruction.AppendType.RBIND )
			LibMatrixCUDA.rbind(ec, ec.getGPUContext(0), getExtendedOpcode(), mat1, mat2, output.getName());
		else
			throw new DMLRuntimeException("Unsupported GPU operator:" + opcode);
		ec.releaseMatrixInputForGPUInstruction(input1.getName());
		ec.releaseMatrixInputForGPUInstruction(input2.getName());
		ec.releaseMatrixOutputForGPUInstruction(output.getName());
	}
}
