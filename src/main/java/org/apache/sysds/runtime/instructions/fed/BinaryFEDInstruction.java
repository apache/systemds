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

import org.apache.sysds.common.Types.DataType;
import org.apache.sysds.runtime.DMLRuntimeException;
import org.apache.sysds.runtime.instructions.InstructionUtils;
import org.apache.sysds.runtime.instructions.cp.CPOperand;
import org.apache.sysds.runtime.matrix.operators.Operator;

public abstract class BinaryFEDInstruction extends ComputationFEDInstruction {
	
	protected BinaryFEDInstruction(FEDInstruction.FEDType type, Operator op,
		CPOperand in1, CPOperand in2, CPOperand out, String opcode, String istr) {
		super(type, op, in1, in2, out, opcode, istr);
	}

	public static BinaryFEDInstruction parseInstruction(String str) {
		String[] parts = InstructionUtils.getInstructionPartsWithValueType(str);
		InstructionUtils.checkNumFields(parts, 3);
		String opcode = parts[0];
		CPOperand in1 = new CPOperand(parts[1]);
		CPOperand in2 = new CPOperand(parts[2]);
		CPOperand out = new CPOperand(parts[3]);

		checkOutputDataType(in1, in2, out);
		Operator operator = InstructionUtils.parseBinaryOrBuiltinOperator(opcode, in1, in2);

		//Operator operator = InstructionUtils.parseBinaryOrBuiltinOperator(opcode, in1, in2);
		// TODO different binary instructions
		if( in1.getDataType() == DataType.SCALAR && in2.getDataType() == DataType.SCALAR )
			throw new DMLRuntimeException("Federated binary scalar scalar operations not yet supported");
		else if( in1.getDataType() == DataType.MATRIX && in2.getDataType() == DataType.MATRIX )
			return new BinaryMatrixMatrixFEDInstruction(operator, in1, in2, out, opcode, str);
		else if( in1.getDataType() == DataType.TENSOR && in2.getDataType() == DataType.TENSOR )
			throw new DMLRuntimeException("Federated binary tensor tensor operations not yet supported");
		else if( in1.isMatrix() && in2.isScalar() )
			return new BinaryMatrixScalarFEDInstruction(operator, in1, in2, out, opcode, str);
		else if( in2.isMatrix() && in1.isScalar() )
			return new BinaryMatrixScalarFEDInstruction(operator, in1, in2, out, opcode, str);
		else
			throw new DMLRuntimeException("Federated binary operations not yet supported:" + opcode);
	}
	
	protected static void checkOutputDataType(CPOperand in1, CPOperand in2, CPOperand out) {
		// check for valid data type of output
		if( (in1.getDataType() == DataType.MATRIX || in2.getDataType() == DataType.MATRIX) && out.getDataType() != DataType.MATRIX )
			throw new DMLRuntimeException("Element-wise matrix operations between variables " + in1.getName() +
				" and " + in2.getName() + " must produce a matrix, which " + out.getName() + " is not");
	}
}
