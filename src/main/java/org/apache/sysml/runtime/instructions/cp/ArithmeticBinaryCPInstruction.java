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

import org.apache.sysml.parser.Expression.DataType;
import org.apache.sysml.parser.Expression.ValueType;
import org.apache.sysml.runtime.DMLRuntimeException;
import org.apache.sysml.runtime.instructions.InstructionUtils;
import org.apache.sysml.runtime.matrix.operators.Operator;


public abstract class ArithmeticBinaryCPInstruction extends BinaryCPInstruction 
{
		
	public ArithmeticBinaryCPInstruction(Operator op, 
								   CPOperand in1, 
								   CPOperand in2, 
								   CPOperand out, 
								   String opcode,
								   String istr )
	{
		super(op, in1, in2, out, opcode, istr);
		_cptype = CPINSTRUCTION_TYPE.ArithmeticBinary;
	}
	
	public static ArithmeticBinaryCPInstruction parseInstruction ( String str ) 
		throws DMLRuntimeException 
	{
		CPOperand in1 = new CPOperand("", ValueType.UNKNOWN, DataType.UNKNOWN);
		CPOperand in2 = new CPOperand("", ValueType.UNKNOWN, DataType.UNKNOWN);
		CPOperand out = new CPOperand("", ValueType.UNKNOWN, DataType.UNKNOWN);
		String opcode = parseBinaryInstruction(str, in1, in2, out);
		
		// Arithmetic operations must be performed on DOUBLE or INT
		ValueType vt1 = in1.getValueType();
		DataType dt1 = in1.getDataType();
		ValueType vt2 = in2.getValueType();
		DataType dt2 = in2.getDataType();
		ValueType vt3 = out.getValueType();
		DataType dt3 = out.getDataType();
		
		//prithvi TODO
		//make sure these checks belong here
		//if either input is a matrix, then output
		//has to be a matrix
		if((dt1 == DataType.MATRIX  || dt2 == DataType.MATRIX) && dt3 != DataType.MATRIX) {
			throw new DMLRuntimeException("Element-wise matrix operations between variables " + in1.getName() + 
					" and " + in2.getName() + " must produce a matrix, which " + out.getName() + "is not");
		}
		
		Operator operator = (dt1 != dt2) ?
					InstructionUtils.parseScalarBinaryOperator(opcode, (dt1 == DataType.SCALAR)) : 
					InstructionUtils.parseBinaryOperator(opcode);
		
		if ( opcode.equalsIgnoreCase("+") && dt1 == DataType.SCALAR && dt2 == DataType.SCALAR) 
		{
			return new ScalarScalarArithmeticCPInstruction(operator, in1, in2, out, opcode, str);
		} 
		else if(dt1 == DataType.SCALAR && dt2 == DataType.SCALAR){
			if ( (vt1 != ValueType.DOUBLE && vt1 != ValueType.INT)
					|| (vt2 != ValueType.DOUBLE && vt2 != ValueType.INT)
					|| (vt3 != ValueType.DOUBLE && vt3 != ValueType.INT) )
				throw new DMLRuntimeException("Unexpected ValueType in ArithmeticInstruction.");
			
			//haven't we already checked for this above -- prithvi
			if ( vt1 != ValueType.DOUBLE && vt1 != ValueType.INT ) {
				throw new DMLRuntimeException("Unexpected ValueType (" + vt1 + ") in ArithmeticInstruction: " + str);
			}
			
			return new ScalarScalarArithmeticCPInstruction(operator, in1, in2, out, opcode, str);
		
		} else if (dt1 == DataType.MATRIX || dt2 == DataType.MATRIX){
			if(vt1 == ValueType.STRING 
			   || vt2 == ValueType.STRING 
			   || vt3 == ValueType.STRING)
				throw new DMLRuntimeException("We do not support element-wise string operations on matrices "
												  + in1.getName()
												  + ", "
												  + in2.getName()
												  + " and "
												  + out.getName());
				
			if(dt1 == DataType.MATRIX && dt2 == DataType.MATRIX)
				return new MatrixMatrixArithmeticCPInstruction(operator, in1, in2, out, opcode, str);
			else
				return new ScalarMatrixArithmeticCPInstruction(operator, in1, in2, out, opcode, str);	
		}
		
		return null;
	}
}
