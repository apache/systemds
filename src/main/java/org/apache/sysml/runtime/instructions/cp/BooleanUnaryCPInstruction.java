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
import org.apache.sysml.runtime.controlprogram.context.ExecutionContext;
import org.apache.sysml.runtime.matrix.data.MatrixBlock;
import org.apache.sysml.runtime.matrix.operators.Operator;
import org.apache.sysml.runtime.matrix.operators.UnaryOperator;

public class BooleanUnaryCPInstruction extends UnaryCPInstruction {

	private BooleanUnaryCPInstruction(Operator op, CPOperand in, CPOperand out, String opcode, String instr) {
		super(CPType.BooleanUnary, op, in, out, opcode, instr);
	}

	public static BooleanUnaryCPInstruction parseInstruction (String str) 
		throws DMLRuntimeException 
	{
		CPOperand in = new CPOperand("", ValueType.UNKNOWN, DataType.UNKNOWN);
		CPOperand out = new CPOperand("", ValueType.UNKNOWN, DataType.UNKNOWN);
		String opcode = parseUnaryInstruction(str, in, out);
		
		// Boolean operations must be performed on BOOLEAN
	//	ValueType vt1 = in.getValueType();
	//	ValueType vt2 = out.getValueType();
	//	if ( (vt1 != ValueType.BOOLEAN || vt2 != ValueType.BOOLEAN) )
	//		throw new DMLRuntimeException("Unexpected ValueType in ArithmeticInstruction.");
		
		// Determine appropriate Function Object based on opcode	
		return new BooleanUnaryCPInstruction(getUnaryOperator(opcode), in, out, opcode, str);
	}
	
	@Override
	public void processInstruction(ExecutionContext ec) 
		throws DMLRuntimeException 
	{
		if(input1.getDataType() == DataType.SCALAR ) {
			// 1) Obtain data objects associated with inputs
			ScalarObject so = ec.getScalarInput(input1.getName(), input1.getValueType(), input1.isLiteral());

			// 2) Compute the result value & make an appropriate data object
			UnaryOperator dop = (UnaryOperator) _optr;
			boolean rval = dop.fn.execute(so.getBooleanValue());

			ScalarObject sores = (ScalarObject) new BooleanObject(rval);

			// 3) Put the result value into ProgramBlock
			ec.setScalarOutput(output.getName(), sores);
		}
		else // for the matrix datatype
		{
			// 1) Obtain data objects associated with inputs
			String opcode = getOpcode();

			MatrixBlock inBlock1 = ec.getMatrixInput(input1.getName(), getExtendedOpcode());

			// 2) Compute the result value & make an appropriate data object
			UnaryOperator dop = (UnaryOperator) _optr;
			String ouput_name = output.getName();

			MatrixBlock retBlock = (MatrixBlock) inBlock1.unaryOperations(dop, new MatrixBlock());

			ec.releaseMatrixInput(input1.getName(), getExtendedOpcode());

			// 3) Put the result value into ProgramBlock
			ec.setMatrixOutput(ouput_name, retBlock, getExtendedOpcode());
		}

	}
}
