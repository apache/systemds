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


public abstract class RelationalBinaryCPInstruction extends BinaryCPInstruction 
{
	public RelationalBinaryCPInstruction(Operator op, CPOperand in1, CPOperand in2, CPOperand out, String opcode, String istr) {
		super(op, in1, in2, out, opcode, istr);
		_cptype = CPINSTRUCTION_TYPE.RelationalBinary;
	}
	
	public static RelationalBinaryCPInstruction parseInstruction ( String str ) throws DMLRuntimeException {
		InstructionUtils.checkNumFields (str, 3);
		CPOperand in1 = new CPOperand("", ValueType.UNKNOWN, DataType.UNKNOWN);
		CPOperand in2 = new CPOperand("", ValueType.UNKNOWN, DataType.UNKNOWN);
		CPOperand out = new CPOperand("", ValueType.UNKNOWN, DataType.UNKNOWN);
		String opcode = parseBinaryInstruction(str, in1, in2, out);
		
		checkOutputDataType(in1, in2, out);
		
		Operator operator = (in1.getDataType() != in2.getDataType()) ?
					InstructionUtils.parseScalarBinaryOperator(opcode, (in1.getDataType() == DataType.SCALAR)) : 
					InstructionUtils.parseBinaryOperator(opcode);
		
		if ( in1.getDataType() == DataType.SCALAR && in2.getDataType() == DataType.SCALAR )
			return new ScalarScalarRelationalCPInstruction(operator, in1, in2, out, opcode, str);
		else if ( in1.getDataType() == DataType.MATRIX && in2.getDataType() == DataType.MATRIX )
			return new MatrixMatrixRelationalCPInstruction(operator, in1, in2, out, opcode, str);
		else
			return new ScalarMatrixRelationalCPInstruction(operator, in1, in2, out, opcode, str);
	}
}
