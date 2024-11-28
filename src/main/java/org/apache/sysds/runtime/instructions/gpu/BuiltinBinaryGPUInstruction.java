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

package org.apache.sysds.runtime.instructions.gpu;

import org.apache.sysds.common.Types.DataType;
import org.apache.sysds.common.Types.ValueType;
import org.apache.sysds.runtime.DMLRuntimeException;
import org.apache.sysds.runtime.functionobjects.Builtin;
import org.apache.sysds.runtime.functionobjects.ValueFunction;
import org.apache.sysds.runtime.instructions.InstructionUtils;
import org.apache.sysds.runtime.instructions.cp.CPOperand;
import org.apache.sysds.runtime.matrix.operators.BinaryOperator;
import org.apache.sysds.runtime.matrix.operators.Operator;

public abstract class BuiltinBinaryGPUInstruction extends GPUInstruction {
	@SuppressWarnings("unused")
	private int _arity;

	protected BuiltinBinaryGPUInstruction(Operator op, CPOperand input1, CPOperand input2, CPOperand output,
			String opcode, String istr, int _arity) {
		super(op, input1, input2, output, opcode, istr);
		this._arity = _arity;
	}

	public static BuiltinBinaryGPUInstruction parseInstruction(String str) {
		CPOperand in1 = new CPOperand("", ValueType.UNKNOWN, DataType.UNKNOWN);
		CPOperand in2 = new CPOperand("", ValueType.UNKNOWN, DataType.UNKNOWN);
		CPOperand out = new CPOperand("", ValueType.UNKNOWN, DataType.UNKNOWN);

		String[] parts = InstructionUtils.getInstructionPartsWithValueType(str);
		InstructionUtils.checkNumFields(parts, 3);

		String opcode = parts[0];
		in1.split(parts[1]);
		in2.split(parts[2]);
		out.split(parts[3]);

		// check for valid data type of output
		if ((in1.getDataType() == DataType.MATRIX || in2.getDataType() == DataType.MATRIX) &&
				out.getDataType() != DataType.MATRIX)
			throw new DMLRuntimeException("Element-wise matrix operations between variables " + in1.getName() + " and "
				+ in2.getName() + " must produce a matrix, which " + out.getName() + " is not");

		// Determine appropriate Function Object based on opcode
		ValueFunction func = Builtin.getBuiltinFnObject(opcode);

		boolean isMatrixMatrix = in1.getDataType() == DataType.MATRIX && in2.getDataType() == DataType.MATRIX;
		boolean isMatrixScalar = (in1.getDataType() == DataType.MATRIX && in2.getDataType() == DataType.SCALAR) ||
				(in1.getDataType() == DataType.SCALAR && in2.getDataType() == DataType.MATRIX);

		if (in1.getDataType() == DataType.SCALAR && in2.getDataType() == DataType.SCALAR)
			throw new DMLRuntimeException("GPU : Unsupported GPU builtin operations on 2 scalars");
		else if (isMatrixMatrix && opcode.equals("solve"))
			return new MatrixMatrixBuiltinGPUInstruction(new BinaryOperator(func), in1, in2, out, opcode, str, 2);
		else if (isMatrixScalar && (opcode.equals("min") || opcode.equals("max")))
			return new ScalarMatrixBuiltinGPUInstruction(new BinaryOperator(func), in1, in2, out, opcode, str, 2);

		else
			throw new DMLRuntimeException(
				"GPU : Unsupported GPU builtin operations on a matrix and a scalar:" + opcode);
	}
}
