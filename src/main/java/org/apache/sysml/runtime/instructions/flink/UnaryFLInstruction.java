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

package org.apache.sysml.runtime.instructions.flink;

import org.apache.sysml.runtime.DMLRuntimeException;
import org.apache.sysml.runtime.instructions.InstructionUtils;
import org.apache.sysml.runtime.instructions.cp.CPOperand;
import org.apache.sysml.runtime.matrix.operators.Operator;

public abstract class UnaryFLInstruction extends ComputationFLInstruction {

	public UnaryFLInstruction(Operator op, CPOperand in, CPOperand out,
							  String opcode, String instr) {
		this(op, in, null, null, out, opcode, instr);
	}

	public UnaryFLInstruction(Operator op, CPOperand in1, CPOperand in2, CPOperand out,
							  String opcode, String instr) {
		this(op, in1, in2, null, out, opcode, instr);
	}

	public UnaryFLInstruction(Operator op, CPOperand in1, CPOperand in2, CPOperand in3, CPOperand out,
							  String opcode, String instr) {
		super(op, in1, in2, in3, out, opcode, instr);
	}

	static String parseUnaryInstruction(String instr, CPOperand in,
										CPOperand out) throws DMLRuntimeException {
		InstructionUtils.checkNumFields(instr, 2);
		return parse(instr, in, null, null, out);
	}

	private static String parse(String instr, CPOperand in1, CPOperand in2, CPOperand in3,
								CPOperand out) throws DMLRuntimeException {
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
}
