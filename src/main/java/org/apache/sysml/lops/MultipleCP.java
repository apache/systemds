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

package org.apache.sysml.lops;

import org.apache.sysml.lops.LopProperties.ExecLocation;
import org.apache.sysml.lops.LopProperties.ExecType;
import org.apache.sysml.lops.compile.JobType;
import org.apache.sysml.parser.Expression.DataType;
import org.apache.sysml.parser.Expression.ValueType;

/**
 * Lop to perform an operation on a variable number of operands.
 * 
 */
public class MultipleCP extends Lop {

	public enum OperationType {
		PRINTF
	};

	OperationType operationType;

	public MultipleCP(OperationType operationType, DataType dt, ValueType vt, Lop... inputLops) {
		super(Lop.Type.MULTIPLE_CP, dt, vt);
		this.operationType = operationType;
		for (Lop inputLop : inputLops) {
			addInput(inputLop);
			inputLop.addOutput(this);
		}

		boolean breaksAlignment = false; // ?
		boolean aligner = false; // ?
		boolean definesMRJob = false; // ?
		lps.addCompatibility(JobType.INVALID); // ?
		this.lps.setProperties(inputs, ExecType.CP, ExecLocation.ControlProgram, breaksAlignment, aligner,
				definesMRJob); // ?
	}

	@Override
	public String toString() {
		return "Operation Type: " + operationType;
	}

	public OperationType getOperationType() {
		return operationType;
	}

	/**
	 * Generate the complete instruction string for this Lop. This instruction
	 * string can have a variable number of input operands. It displays the
	 * following:
	 * 
	 * <ul>
	 * <li>Execution type (CP, SPARK, etc.)
	 * <li>Operand delimiter (&deg;)</li>
	 * <li>Opcode (printf, etc.)</li>
	 * <li>Operand delimiter (&deg;)</li>
	 * <li>Variable number of inputs, each followed by an operand delimiter
	 * (&deg;)</li>
	 * <ul>
	 * <li>Input consists of (label &middot; data type &middot; value type
	 * &middot; is literal)</li>
	 * </ul>
	 * <li>Output consisting of (label &middot; data type &middot; value
	 * type)</li>
	 * </ul>
	 *
	 * Example: <br>
	 * The following DML<br>
	 * <code>print('hello %s', 'world')</code><br>
	 * generates the instruction string:<br>
	 * <code>CP&deg;printf&deg;hello %s&middot;SCALAR&middot;STRING&middot;true&deg;world&middot;SCALAR&middot;STRING&middot;true&deg;_Var1&middot;SCALAR&middot;STRING</code><br>
	 * 
	 * Note: This generated instruction string is parsed in the
	 * parseInstruction() method of BuiltinMultipleCPInstruction, which parses
	 * the instruction string to generate an instruction object that is a
	 * subclass of BuiltinMultipleCPInstruction.
	 */
	@Override
	public String getInstructions(String output) throws LopsException {
		String opString = getOpcode();

		StringBuilder sb = new StringBuilder();

		sb.append(getExecType());
		sb.append(Lop.OPERAND_DELIMITOR);

		sb.append(opString);
		sb.append(OPERAND_DELIMITOR);

		for (Lop input : inputs) {
			sb.append(input.prepScalarInputOperand(getExecType()));
			sb.append(OPERAND_DELIMITOR);
		}

		sb.append(prepOutputOperand(output));

		return sb.toString();
	}

	private String getOpcode() throws LopsException {
		switch (operationType) {
		case PRINTF:
			return OperationType.PRINTF.toString().toLowerCase();
		default:
			throw new UnsupportedOperationException(
					"MultipleCP operation type (" + operationType + ") is not defined.");
		}
	}

}
