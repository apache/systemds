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

package org.apache.sysds.runtime.instructions.ooc;

import org.apache.sysds.common.Opcodes;
import org.apache.sysds.runtime.DMLRuntimeException;
import org.apache.sysds.runtime.controlprogram.caching.MatrixObject;
import org.apache.sysds.runtime.controlprogram.context.ExecutionContext;
import org.apache.sysds.runtime.instructions.InstructionUtils;
import org.apache.sysds.runtime.instructions.cp.CPOperand;
import org.apache.sysds.runtime.instructions.spark.data.IndexedMatrixValue;
import org.apache.sysds.runtime.matrix.operators.Operator;
import org.apache.sysds.runtime.ooc.util.OOCInstructionUtils;


public class ReshapeOOCInstruction extends ComputationOOCInstruction {
	private final CPOperand _opRows;
	private final CPOperand _opCols;
	// private final CPOperand _opDims;
	private final CPOperand _opByRow;

	private ReshapeOOCInstruction(Operator op, CPOperand in, CPOperand out, CPOperand rows, CPOperand cols,
		CPOperand dims, CPOperand byRow, String opcode, String istr) {
		super(OOCType.Reshape, op, in, out, opcode, istr);
		_opRows = rows;
		_opCols = cols;
		// _opDims = dims;
		_opByRow = byRow;
	}

	public static ReshapeOOCInstruction parseInstruction(String str) {
		String[] parts = InstructionUtils.getInstructionPartsWithValueType(str);
		InstructionUtils.checkNumFields(parts, 6);
		String opcode = parts[0];

		if(!opcode.equalsIgnoreCase(Opcodes.RESHAPE.toString()))
			throw new DMLRuntimeException("Unknown opcode while parsing ReshapeInstruction: " + str);

		CPOperand in = new CPOperand(parts[1]);
		CPOperand rows = new CPOperand(parts[2]);
		CPOperand cols = new CPOperand(parts[3]);
		CPOperand dims = new CPOperand(parts[4]);
		CPOperand byRow = new CPOperand(parts[5]);
		CPOperand out = new CPOperand(parts[6]);

		return new ReshapeOOCInstruction(new Operator(true), in, out, rows, cols, dims, byRow, opcode, str);
	}

	public void processInstruction(ExecutionContext ec) {
		long rows = ec.getScalarInput(_opRows).getLongValue();
		long cols = ec.getScalarInput(_opCols).getLongValue();
		boolean byRow = ec.getScalarInput(_opByRow).getBooleanValue();

		OOCStream<IndexedMatrixValue> qOut = createWritableStream();
		ec.getMatrixObject(output).setStreamHandle(qOut);

		MatrixObject in = ec.getMatrixObject(input1);
		OOCStream<IndexedMatrixValue> qIn = in.getStreamHandle();

		OOCInstructionUtils.reshape(qIn, qOut, rows, cols, byRow, getContext());
	}
}
