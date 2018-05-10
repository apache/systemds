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

import org.apache.sysml.lops.Nary;
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
public abstract class BuiltinNaryCPInstruction extends CPInstruction 
{
	protected final CPOperand output;
	protected final CPOperand[] inputs;

	public BuiltinNaryCPInstruction(Operator op, String opcode, String istr, CPOperand output, CPOperand... inputs) {
		super(CPType.BuiltinNary, op, opcode, istr);
		this.output = output;
		this.inputs = inputs;
	}

	public static BuiltinNaryCPInstruction parseInstruction(String str) {
		String[] parts = InstructionUtils.getInstructionPartsWithValueType(str);
		String opcode = parts[0];
		CPOperand outputOperand = new CPOperand(parts[parts.length - 1]);
		CPOperand[] inputOperands = null;
		if (parts.length > 2) {
			inputOperands = new CPOperand[parts.length - 2];
			for (int i = 1; i < parts.length-1; i++)
				inputOperands[i-1] = new CPOperand(parts[i]);
		}
		
		if( "printf".equals(opcode) || "list".equals(opcode)) {
			ValueFunction func = Builtin.getBuiltinFnObject(opcode);
			return new ScalarBuiltinNaryCPInstruction(new SimpleOperator(func), 
				opcode, str, outputOperand, inputOperands);
		}
		else if( opcode.equals("cbind") || opcode.equals("rbind") ) {
			return new MatrixBuiltinNaryCPInstruction(null, 
					opcode, str, outputOperand, inputOperands);
		} 
		else if (Nary.OperationType.EVAL.name().equalsIgnoreCase(opcode)) {
			return new EvalNaryCPInstruction(null, opcode, str, outputOperand, inputOperands);
		}
		
		throw new DMLRuntimeException("Opcode (" + opcode + ") not recognized in BuiltinMultipleCPInstruction");
	}
}
