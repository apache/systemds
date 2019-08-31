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

package org.tugraz.sysds.runtime.instructions.cp;

import org.tugraz.sysds.common.Types.DataType;
import org.tugraz.sysds.common.Types.ValueType;
import org.tugraz.sysds.runtime.DMLRuntimeException;
import org.tugraz.sysds.runtime.instructions.InstructionUtils;
import org.tugraz.sysds.runtime.matrix.operators.Operator;

public abstract class BinaryCPInstruction extends ComputationCPInstruction {

	protected BinaryCPInstruction(CPType type, Operator op, CPOperand in1, CPOperand in2, CPOperand out, String opcode, String istr) {
		super(type, op, in1, in2, out, opcode, istr);
	}

	protected BinaryCPInstruction(CPType type, Operator op, CPOperand in1, CPOperand in2, CPOperand in3, CPOperand out,
			String opcode, String istr) {
		super(type, op, in1, in2, in3, out, opcode, istr);
	}

	public static BinaryCPInstruction parseInstruction( String str ) {
		CPOperand in1 = new CPOperand("", ValueType.UNKNOWN, DataType.UNKNOWN);
		CPOperand in2 = new CPOperand("", ValueType.UNKNOWN, DataType.UNKNOWN);
		CPOperand out = new CPOperand("", ValueType.UNKNOWN, DataType.UNKNOWN);
		String opcode = parseBinaryInstruction(str, in1, in2, out);
		
		checkOutputDataType(in1, in2, out);
		
		Operator operator = InstructionUtils.parseBinaryOrBuiltinOperator(opcode, in1, in2);

		if (in1.getDataType() == DataType.SCALAR && in2.getDataType() == DataType.SCALAR)
			return new BinaryScalarScalarCPInstruction(operator, in1, in2, out, opcode, str);
		else if (in1.getDataType() == DataType.MATRIX && in2.getDataType() == DataType.MATRIX)
			return new BinaryMatrixMatrixCPInstruction(operator, in1, in2, out, opcode, str);
		else if (in1.getDataType() == DataType.TENSOR && in2.getDataType() == DataType.TENSOR)
			return new BinaryTensorTensorCPInstruction(operator, in1, in2, out, opcode, str);
		else
			return new BinaryMatrixScalarCPInstruction(operator, in1, in2, out, opcode, str);
	}
	
	protected static String parseBinaryInstruction(String instr, CPOperand in1, CPOperand in2, CPOperand out) {
		String[] parts = InstructionUtils.getInstructionPartsWithValueType(instr);
		InstructionUtils.checkNumFields ( parts, 3 );
		String opcode = parts[0];
		in1.split(parts[1]);
		in2.split(parts[2]);
		out.split(parts[3]);
		return opcode;
	}
	
	protected static String parseBinaryInstruction(String instr, CPOperand in1, CPOperand in2, CPOperand in3, CPOperand out) {
		String[] parts = InstructionUtils.getInstructionPartsWithValueType(instr);
		InstructionUtils.checkNumFields ( parts, 4 );
		
		String opcode = parts[0];
		in1.split(parts[1]);
		in2.split(parts[2]);
		in3.split(parts[3]);
		out.split(parts[4]);
		
		return opcode;
	}
	
	protected static void checkOutputDataType(CPOperand in1, CPOperand in2, CPOperand out) {
		// check for valid data type of output
		if((in1.getDataType() == DataType.MATRIX || in2.getDataType() == DataType.MATRIX) && out.getDataType() != DataType.MATRIX)
			throw new DMLRuntimeException("Element-wise matrix operations between variables " + in1.getName() + 
					" and " + in2.getName() + " must produce a matrix, which " + out.getName() + " is not");
	}
}
