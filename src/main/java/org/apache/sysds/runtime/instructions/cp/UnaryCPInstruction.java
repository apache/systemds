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

package org.apache.sysds.runtime.instructions.cp;

import java.util.Arrays;

import org.apache.sysds.common.Types.DataType;
import org.apache.sysds.common.Types.ValueType;
import org.apache.sysds.runtime.DMLRuntimeException;
import org.apache.sysds.runtime.functionobjects.Builtin;
import org.apache.sysds.runtime.functionobjects.ValueFunction;
import org.apache.sysds.runtime.instructions.InstructionUtils;
import org.apache.sysds.runtime.matrix.data.LibCommonsMath;
import org.apache.sysds.runtime.matrix.operators.Operator;
import org.apache.sysds.runtime.matrix.operators.UnaryOperator;

public abstract class UnaryCPInstruction extends ComputationCPInstruction {

	protected UnaryCPInstruction(CPType type, Operator op, CPOperand in, CPOperand out, String opcode, String instr) {
		this(type, op, in, null, null, out, opcode, instr);
	}

	protected UnaryCPInstruction(CPType type, Operator op, CPOperand in1, CPOperand in2, CPOperand out, String opcode,
			String instr) {
		this(type, op, in1, in2, null, out, opcode, instr);
	}

	protected UnaryCPInstruction(CPType type, Operator op, CPOperand in1, CPOperand in2, CPOperand in3, CPOperand out, String opcode,
			String instr) {
		super(type, op, in1, in2, in3, out, opcode, instr);
	}
	
	public static UnaryCPInstruction parseInstruction ( String str ) {
		CPOperand in = new CPOperand("", ValueType.UNKNOWN, DataType.UNKNOWN);
		CPOperand out = new CPOperand("", ValueType.UNKNOWN, DataType.UNKNOWN);
		
		String[] parts = InstructionUtils.getInstructionPartsWithValueType(str);
		String opcode = null;
		ValueFunction func = null;
		
		//print or stop or cumulative aggregates
		if( parts.length==5 ) {
			opcode = parts[0];
			in.split(parts[1]);
			out.split(parts[2]);
			func = Builtin.getBuiltinFnObject(opcode);
			
			if( Arrays.asList(new String[]{"ucumk+","ucum*","ucumk+*","ucummin","ucummax","exp","log","sigmoid"}).contains(opcode) )
				return new UnaryMatrixCPInstruction(new UnaryOperator(func,
					Integer.parseInt(parts[3]),Boolean.parseBoolean(parts[4])), in, out, opcode, str);
			else
				return new UnaryScalarCPInstruction(null, in, out, opcode, str);
		}
		else { //2+1, general case
			opcode = parseUnaryInstruction(str, in, out);
			
			if(in.getDataType() == DataType.SCALAR)
				return new UnaryScalarCPInstruction(InstructionUtils.parseUnaryOperator(opcode), in, out, opcode, str);
			else if(in.getDataType() == DataType.MATRIX)
				return new UnaryMatrixCPInstruction(LibCommonsMath.isSupportedUnaryOperation(opcode) ?
					null : InstructionUtils.parseUnaryOperator(opcode), in, out, opcode, str);
			else if(in.getDataType() == DataType.FRAME)
				return new UnaryFrameCPInstruction(InstructionUtils.parseUnaryOperator(opcode), in, out, opcode, str);
		}
		return null;
	}

	static String parseUnaryInstruction(String instr, CPOperand in, CPOperand out) {
		InstructionUtils.checkNumFields(instr, 2);
		return parse(instr, in, null, null, out);
	}

	static String parseUnaryInstruction(String instr, CPOperand in1, CPOperand in2, CPOperand out) {
		InstructionUtils.checkNumFields(instr, 3);
		return parse(instr, in1, in2, null, out);
	}

	static String parseUnaryInstruction(String instr, CPOperand in1, CPOperand in2, CPOperand in3, CPOperand out) {
		InstructionUtils.checkNumFields(instr, 4);
		return parse(instr, in1, in2, in3, out);
	}

	private static String parse(String instr, CPOperand in1, CPOperand in2, CPOperand in3, CPOperand out) {
		String[] parts = InstructionUtils.getInstructionPartsWithValueType(instr);
		
		// first part is the opcode, last part is the output, middle parts are input operands
		String opcode = parts[0];
		out.split(parts[parts.length-1]);
		
		switch(parts.length) {
		case 3:
			in1.split(parts[1]);
			in2 = null;
			in3 = null;
			break;
		case 4:
			in1.split(parts[1]);
			in2.split(parts[2]);
			in3 = null;
			break;
		case 5:
			in1.split(parts[1]);
			in2.split(parts[2]);
			in3.split(parts[3]);
			break;
		default:
			throw new DMLRuntimeException("Unexpected number of operands in the instruction: " + instr);
		}
		return opcode;
	}
}
