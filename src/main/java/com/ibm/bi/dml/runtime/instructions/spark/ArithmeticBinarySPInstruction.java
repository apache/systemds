/**
 * (C) Copyright IBM Corp. 2010, 2015
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * 
 */

package com.ibm.bi.dml.runtime.instructions.spark;

import com.ibm.bi.dml.lops.BinaryM.VectorType;
import com.ibm.bi.dml.lops.Lop;
import com.ibm.bi.dml.parser.Expression.DataType;
import com.ibm.bi.dml.parser.Expression.ValueType;
import com.ibm.bi.dml.runtime.DMLRuntimeException;
import com.ibm.bi.dml.runtime.DMLUnsupportedOperationException;
import com.ibm.bi.dml.runtime.instructions.Instruction;
import com.ibm.bi.dml.runtime.instructions.InstructionUtils;
import com.ibm.bi.dml.runtime.instructions.cp.CPOperand;
import com.ibm.bi.dml.runtime.matrix.operators.Operator;

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
	public static Instruction parseInstruction ( String str ) 
		throws DMLRuntimeException, DMLUnsupportedOperationException 
	{
		CPOperand in1 = new CPOperand("", ValueType.UNKNOWN, DataType.UNKNOWN);
		CPOperand in2 = new CPOperand("", ValueType.UNKNOWN, DataType.UNKNOWN);
		CPOperand out = new CPOperand("", ValueType.UNKNOWN, DataType.UNKNOWN);
		String opcode = null;
		boolean isBroadcast = false;
		VectorType vtype = null;
		
		if(str.startsWith("SPARK"+Lop.OPERAND_DELIMITOR+"map")) {
			InstructionUtils.checkNumFields ( str, 5 );
			
			String[] parts = InstructionUtils.getInstructionPartsWithValueType(str);
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
