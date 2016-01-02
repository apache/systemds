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

package org.apache.sysml.runtime.instructions.spark;

import org.apache.sysml.lops.BinaryM.VectorType;
import org.apache.sysml.lops.Lop;
import org.apache.sysml.parser.Expression.DataType;
import org.apache.sysml.parser.Expression.ValueType;
import org.apache.sysml.runtime.DMLRuntimeException;
import org.apache.sysml.runtime.DMLUnsupportedOperationException;
import org.apache.sysml.runtime.instructions.InstructionUtils;
import org.apache.sysml.runtime.instructions.cp.CPOperand;
import org.apache.sysml.runtime.matrix.operators.Operator;

public abstract class ArithmeticBinarySPInstruction extends BinarySPInstruction 
{
		
	public ArithmeticBinarySPInstruction(Operator op, CPOperand in1, CPOperand in2, CPOperand out, String opcode, String istr)
	{
		super(op, in1, in2, out, opcode, istr);
		_sptype = SPINSTRUCTION_TYPE.ArithmeticBinary;
	}
	
	/**
	 * 
	 * @param str
	 * @return
	 * @throws DMLRuntimeException
	 * @throws DMLUnsupportedOperationException
	 */
	public static ArithmeticBinarySPInstruction parseInstruction ( String str ) 
		throws DMLRuntimeException, DMLUnsupportedOperationException 
	{
		CPOperand in1 = new CPOperand("", ValueType.UNKNOWN, DataType.UNKNOWN);
		CPOperand in2 = new CPOperand("", ValueType.UNKNOWN, DataType.UNKNOWN);
		CPOperand out = new CPOperand("", ValueType.UNKNOWN, DataType.UNKNOWN);
		String opcode = null;
		boolean isBroadcast = false;
		VectorType vtype = null;
		
		if(str.startsWith("SPARK"+Lop.OPERAND_DELIMITOR+"map")) {
			String[] parts = InstructionUtils.getInstructionPartsWithValueType(str);
			InstructionUtils.checkNumFields ( parts, 5 );
			
			opcode = parts[0];
			in1.split(parts[1]);
			in2.split(parts[2]);
			out.split(parts[3]);
			vtype = VectorType.valueOf(parts[5]);
			isBroadcast = true;
		}
		else {
			opcode = parseBinaryInstruction(str, in1, in2, out);
		}
		
		// Arithmetic operations must be performed on DOUBLE or INT
		DataType dt1 = in1.getDataType();
		DataType dt2 = in2.getDataType();
		
		Operator operator = (dt1 != dt2) ?
					InstructionUtils.parseScalarBinaryOperator(opcode, (dt1 == DataType.SCALAR))
					: InstructionUtils.parseExtendedBinaryOperator(opcode);
		
		if (dt1 == DataType.MATRIX || dt2 == DataType.MATRIX)
		{				
			if(dt1 == DataType.MATRIX && dt2 == DataType.MATRIX) {
				if(isBroadcast)
					return new MatrixBVectorArithmeticSPInstruction(operator, in1, in2, out, vtype, opcode, str);
				else
					return new MatrixMatrixArithmeticSPInstruction(operator, in1, in2, out, opcode, str);
			}
			else
				return new MatrixScalarArithmeticSPInstruction(operator, in1, in2, out, opcode, str);	
		}
		
		return null;
	}
}
