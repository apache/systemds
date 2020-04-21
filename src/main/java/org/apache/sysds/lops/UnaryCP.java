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

 
import org.apache.sysds.lops.LopProperties.ExecType;
import org.apache.sysds.runtime.instructions.InstructionUtils;
import org.apache.sysds.common.Types.DataType;
import org.apache.sysds.common.Types.OpOp1;
import org.apache.sysds.common.Types.ValueType;

public class UnaryCP extends Lop 
{
	public static final String CAST_AS_SCALAR_OPCODE = "castdts";
	public static final String CAST_AS_MATRIX_OPCODE = "castdtm";
	public static final String CAST_AS_FRAME_OPCODE = "castdtf";
	public static final String CAST_AS_DOUBLE_OPCODE = "castvtd";
	public static final String CAST_AS_INT_OPCODE    = "castvti";
	public static final String CAST_AS_BOOLEAN_OPCODE = "castvtb";

	private OpOp1 operation;

	/**
	 * Constructor to perform a scalar operation
	 * 
	 * @param input low-level operator 1
	 * @param op operation type
	 * @param dt data type
	 * @param vt value type
	 * @param et exec type
	 */
	public UnaryCP(Lop input, OpOp1 op, DataType dt, ValueType vt, ExecType et) {
		super(Lop.Type.UnaryCP, dt, vt);
		operation = op;
		addInput(input);
		input.addOutput(this);
		lps.setProperties(inputs, et);
	}
	
	public UnaryCP(Lop input, OpOp1 op, DataType dt, ValueType vt) {
		this(input, op, dt, vt, ExecType.CP);
	}

	@Override
	public String toString() {
		return "Operation: " + operation;
	}
	
	private String getOpCode() {
		return operation.toString();
	}

	@Override
	public String getInstructions(String input, String output) {
		return InstructionUtils.concatOperands(
			getExecType().name(), getOpCode(),
			getInputs().get(0).prepScalarInputOperand(getExecType()),
			prepOutputOperand(output));
	}
}
