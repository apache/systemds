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

import java.util.Arrays;

import org.apache.sysml.lops.MultipleCP;
import org.apache.sysml.runtime.DMLRuntimeException;
import org.apache.sysml.runtime.functionobjects.Builtin;
import org.apache.sysml.runtime.functionobjects.ValueFunction;
import org.apache.sysml.runtime.instructions.InstructionUtils;
import org.apache.sysml.runtime.matrix.operators.Operator;
import org.apache.sysml.runtime.matrix.operators.SimpleOperator;

/**
 * Instruction to handle a variable number of input operands. It parses an
 * instruction string to generate an object that is a subclass of
 * BuiltinMultipleCPInstruction. Currently the only subclass of
 * BuiltinMultipleCPInstruction is ScalarBuiltinMultipleCPInstruction. The
 * ScalarBuiltinMultipleCPInstruction class is responsible for printf-style
 * Java-based string formatting.
 *
 */
public abstract class BuiltinMultipleCPInstruction extends CPInstruction {

	public CPOperand output;
	public CPOperand[] inputs;

	public BuiltinMultipleCPInstruction(Operator op, String opcode, String istr, CPOperand output,
			CPOperand... inputs) {
		super(op, opcode, istr);
		_cptype = CPINSTRUCTION_TYPE.BuiltinMultiple;
		this.output = output;
		this.inputs = inputs;
	}

	public static BuiltinMultipleCPInstruction parseInstruction(String str) throws DMLRuntimeException {
		String[] parts = InstructionUtils.getInstructionPartsWithValueType(str);

		String opcode = parts[0];

		String outputString = parts[parts.length - 1];
		CPOperand outputOperand = new CPOperand(outputString);

		String[] inputStrings = null;
		CPOperand[] inputOperands = null;
		if (parts.length > 2) {
			inputStrings = Arrays.copyOfRange(parts, 1, parts.length - 1);
			inputOperands = new CPOperand[parts.length - 2];
			for (int i = 0; i < inputStrings.length; i++) {
				inputOperands[i] = new CPOperand(inputStrings[i]);
			}
		}

		if (MultipleCP.OperationType.PRINTF.toString().equalsIgnoreCase(opcode)) {
			ValueFunction func = Builtin.getBuiltinFnObject(opcode);
			return new ScalarBuiltinMultipleCPInstruction(new SimpleOperator(func), opcode, str, outputOperand,
					inputOperands);
		}
		throw new DMLRuntimeException("Opcode (" + opcode + ") not recognized in BuiltinMultipleCPInstruction");
	}
}
