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

package org.apache.sysds.lops;

 
import org.apache.sysds.common.Opcodes;
import org.apache.sysds.common.Types.ExecType;

import org.apache.sysds.common.Types.DataType;
import org.apache.sysds.common.Types.OpOpN;
import org.apache.sysds.common.Types.ValueType;
import org.apache.sysds.runtime.instructions.InstructionUtils;

/**
 * Lop to perform an operation on a variable number of operands.
 * 
 */
public class Nary extends Lop {
	private OpOpN operationType;

	public Nary(OpOpN operationType, DataType dt, ValueType vt, Lop[] inputLops, ExecType et)
	{
		super(Lop.Type.Nary, dt, vt);
		this.operationType = operationType;
		for (Lop inputLop : inputLops) {
			addInput(inputLop);
			inputLop.addOutput(this);
		}
		
		if( et == ExecType.CP || et == ExecType.SPARK ) {
			lps.setProperties(inputs, et);
		}
		else {
			throw new LopsException("Unsupported exec type for nary lop:" + et.name());
		}
	}

	@Override
	public String toString() {
		return "Operation Type: " + operationType;
	}

	public OpOpN getOp() {
		return operationType;
	}

	/**
	 * Generate the complete instruction string for this Lop. This instruction
	 * string can have a variable number of input operands. It displays the
	 * following:
	 * 
	 * <ul>
	 * <li>Execution type (CP, SPARK, etc.)
	 * <li>Operand delimiter (&deg;)
	 * <li>Opcode (printf, etc.)
	 * <li>Operand delimiter (&deg;)
	 * <li>Variable number of inputs, each followed by an operand delimiter
	 * (&deg;)
	 * <ul>
	 * <li>Input consists of (label &middot; data type &middot; value type
	 * &middot; is literal)
	 * </ul>
	 * <li>Output consisting of (label &middot; data type &middot; value
	 * type)
	 * </ul>
	 *
	 * Example: <br>
	 * The following DML<br>
	 * <code>print('hello %s', 'world')</code><br>
	 * generates the instruction string:<br>
	 * <code>CP&deg;printf&deg;hello %s&middot;SCALAR&middot;STRING&middot;true&deg;world&middot;SCALAR&middot;STRING&middot;true&deg;_Var1&middot;SCALAR&middot;STRING</code><br>
	 * 
	 */
	@Override
	public String getInstructions(String[] inputs, String output) {
		StringBuilder sb = InstructionUtils.getStringBuilder();
		sb.append(getExecType());
		sb.append(Lop.OPERAND_DELIMITOR);
		sb.append(getOpcode());
		sb.append(OPERAND_DELIMITOR);
		for( int i=0; i<inputs.length; i++ ) {
			sb.append(getInputs().get(i).prepInputOperand(inputs[i]));
			sb.append(OPERAND_DELIMITOR);
		}
		sb.append(prepOutputOperand(output));

		return sb.toString();
	}
	
	private String getOpcode() {
		switch (operationType) {
			case PRINTF:
			case CBIND:
			case RBIND:
			case EVAL:
			case LIST:
				return operationType.name().toLowerCase();
			case MIN:
			case MAX:
				//need to differentiate from binary min/max operations
				return "n"+operationType.name().toLowerCase();
			case PLUS:
				return Opcodes.NP.toString();
			case MULT:
				return Opcodes.NM.toString();
			default:
				throw new UnsupportedOperationException(
					"Nary operation type (" + operationType + ") is not defined.");
		}
	}
}
