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

package org.apache.sysds.runtime.instructions.fed;

import org.apache.sysds.runtime.DMLRuntimeException;
import org.apache.sysds.runtime.instructions.InstructionUtils;
import org.apache.sysds.runtime.instructions.cp.CPOperand;
import org.apache.sysds.runtime.matrix.operators.Operator;

public abstract class UnaryFEDInstruction extends ComputationFEDInstruction {
	
	protected UnaryFEDInstruction(FEDType type, Operator op, CPOperand in, CPOperand out, String opcode, String instr) {
		this(type, op, in, null, null, out, opcode, instr);
	}

	protected UnaryFEDInstruction(FEDType type, Operator op, CPOperand in, CPOperand out, String opcode, String instr,
		FederatedOutput fedOut) {
		this(type, op, in, null, null, out, opcode, instr, fedOut);
	}
	
	protected UnaryFEDInstruction(FEDType type, Operator op, CPOperand in1, CPOperand in2, CPOperand out, String opcode,
			String instr) {
		this(type, op, in1, in2, null, out, opcode, instr);
	}

	protected UnaryFEDInstruction(FEDType type, Operator op, CPOperand in1, CPOperand in2, CPOperand out, String opcode,
		String instr, FederatedOutput fedOut) {
		this(type, op, in1, in2, null, out, opcode, instr, fedOut);
	}
	
	protected UnaryFEDInstruction(FEDType type, Operator op, CPOperand in1, CPOperand in2, CPOperand in3, CPOperand out,
			String opcode, String instr) {
		this(type, op, in1, in2, in3, out, opcode, instr, FederatedOutput.NONE);
	}

	protected UnaryFEDInstruction(FEDType type, Operator op, CPOperand in1, CPOperand in2, CPOperand in3, CPOperand out,
		String opcode, String instr, FederatedOutput fedOut) {
		super(type, op, in1, in2, in3, out, opcode, instr, fedOut);
	}
	
	static String parseUnaryInstruction(String instr, CPOperand in, CPOperand out) {
		//TODO: simplify once all fed instructions have consistent flags
		int num = InstructionUtils.checkNumFields(instr, 2, 3, 4);
		if(num == 2)
			return parse(instr, in, null, null, out); 
		else {
			String[] parts = InstructionUtils.getInstructionPartsWithValueType(instr);
			String opcode = parts[0];
			in.split(parts[1]);
			out.split(parts[2]);
			return opcode;
		}
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
		out.split(parts[parts.length - 1]);
		
		switch (parts.length) {
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
	
	/**
	 * Parse and return federated output flag from given instr string at given position.
	 * If the position given is greater than the length of the instruction, FederatedOutput.NONE is returned.
	 * @param instr instruction string to be parsed
	 * @param position of federated output flag
	 * @return parsed federated output flag or FederatedOutput.NONE
	 */
	static FederatedOutput parseFedOutFlag(String instr, int position){
		String[] parts = InstructionUtils.getInstructionPartsWithValueType(instr);
		if ( parts.length > position )
			return FederatedOutput.valueOf(parts[position]);
		else return FederatedOutput.NONE;
	}
}
