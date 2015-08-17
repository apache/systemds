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

package com.ibm.bi.dml.runtime.instructions.cp;

import com.ibm.bi.dml.parser.Expression.DataType;
import com.ibm.bi.dml.parser.Expression.ValueType;
import com.ibm.bi.dml.runtime.DMLRuntimeException;
import com.ibm.bi.dml.runtime.instructions.Instruction;
import com.ibm.bi.dml.runtime.instructions.InstructionUtils;
import com.ibm.bi.dml.runtime.matrix.operators.Operator;


public abstract class RelationalBinaryCPInstruction extends BinaryCPInstruction 
{
	
	public RelationalBinaryCPInstruction(Operator op, CPOperand in1, CPOperand in2, CPOperand out, String opcode, String istr )
	{
		super(op, in1, in2, out, opcode, istr);
		_cptype = CPINSTRUCTION_TYPE.RelationalBinary;
	}
	
	public static Instruction parseInstruction ( String str ) throws DMLRuntimeException {
		InstructionUtils.checkNumFields (str, 3);
		CPOperand in1 = new CPOperand("", ValueType.UNKNOWN, DataType.UNKNOWN);
		CPOperand in2 = new CPOperand("", ValueType.UNKNOWN, DataType.UNKNOWN);
		CPOperand out = new CPOperand("", ValueType.UNKNOWN, DataType.UNKNOWN);
		String opcode = parseBinaryInstruction(str, in1, in2, out);
		
		// TODO: Relational operations need not have value type checking
		ValueType vt1 = in1.getValueType();
		DataType dt1 = in1.getDataType();
		ValueType vt2 = in2.getValueType();
		DataType dt2 = in2.getDataType();
		DataType dt3 = out.getDataType();
		
		//if ( vt3 != ValueType.BOOLEAN )
		//	throw new DMLRuntimeException("Unexpected ValueType in RelationalCPInstruction: " + str);
		
		if ( vt1 == ValueType.BOOLEAN && !opcode.equalsIgnoreCase("==") && !opcode.equalsIgnoreCase("!=") ) 
			throw new DMLRuntimeException("Operation " + opcode + " can not be applied on boolean values "
					 					  + "(Instruction = " + str + ").");
		
		//prithvi TODO
		//make sure these checks belong here
		//if either input is a matrix, then output
		//has to be a matrix
		if((dt1 == DataType.MATRIX 
			|| dt2 == DataType.MATRIX) 
		   && dt3 != DataType.MATRIX)
			throw new DMLRuntimeException("Element-wise matrix operations between variables "
										  + in1.getName()
										  + " and "
										  + in2.getName()
										  + " must produce a matrix, which "
										  + out.getName()
										  + " is not");
		
		Operator operator = (dt1 != dt2) ?
					InstructionUtils.parseScalarBinaryOperator(opcode, (dt1 == DataType.SCALAR)) : 
					InstructionUtils.parseBinaryOperator(opcode);
		
		//for scalar relational operations we only allow boolean operands
		//or when both operands are numeric (int or double)
		if(dt1 == DataType.SCALAR && dt2 == DataType.SCALAR){
			if (!(  (vt1 == ValueType.BOOLEAN && vt2 == ValueType.BOOLEAN)
				  ||(vt1 == ValueType.STRING && vt2 == ValueType.STRING)
				  ||( (vt1 == ValueType.DOUBLE || vt1 == ValueType.INT) && (vt2 == ValueType.DOUBLE || vt2 == ValueType.INT))))
			{
				throw new DMLRuntimeException("unexpected value-type in "
											  + "Relational Binary Instruction "
											  + "involving scalar operands.");
			}
			return new ScalarScalarRelationalCPInstruction(operator, in1, in2, out, opcode, str);
		
		}else if (dt1 == DataType.MATRIX || dt2 == DataType.MATRIX){
			if(dt1 == DataType.MATRIX && dt2 == DataType.MATRIX)
				return new MatrixMatrixRelationalCPInstruction(operator, in1, in2, out, opcode, str);
			else
				return new ScalarMatrixRelationalCPInstruction(operator, in1, in2, out, opcode, str);
		}
		
		return null;
	}
}
