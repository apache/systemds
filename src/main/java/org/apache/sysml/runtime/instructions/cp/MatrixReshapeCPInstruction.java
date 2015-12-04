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

import org.apache.sysml.parser.Expression.ValueType;
import org.apache.sysml.runtime.DMLRuntimeException;
import org.apache.sysml.runtime.DMLUnsupportedOperationException;
import org.apache.sysml.runtime.controlprogram.context.ExecutionContext;
import org.apache.sysml.runtime.instructions.Instruction;
import org.apache.sysml.runtime.instructions.InstructionUtils;
import org.apache.sysml.runtime.matrix.data.LibMatrixReorg;
import org.apache.sysml.runtime.matrix.data.MatrixBlock;
import org.apache.sysml.runtime.matrix.operators.Operator;

/**
 * 
 * 
 */
public class MatrixReshapeCPInstruction extends UnaryCPInstruction
{	
	
	private CPOperand _opRows = null;
	private CPOperand _opCols = null;
	private CPOperand _opByRow = null;
	
	public MatrixReshapeCPInstruction(Operator op, CPOperand in1, CPOperand in2, CPOperand in3, CPOperand in4, CPOperand out, String opcode, String istr)
	{
		super(op, in1, out, opcode, istr);
		_cptype = CPINSTRUCTION_TYPE.MatrixReshape;
		
		_opRows = in2;
		_opCols = in3;
		_opByRow = in4;
	}
	
	/**
	 * 
	 * @param str
	 * @return
	 * @throws DMLRuntimeException
	 */
	public static Instruction parseInstruction ( String str ) 
		throws DMLRuntimeException 
	{
		String[] parts = InstructionUtils.getInstructionPartsWithValueType(str);
		InstructionUtils.checkNumFields( parts, 5 );
		
		String opcode = parts[0];
		CPOperand in1 = new CPOperand(parts[1]);
		CPOperand in2 = new CPOperand(parts[2]);
		CPOperand in3 = new CPOperand(parts[3]);
		CPOperand in4 = new CPOperand(parts[4]);
		CPOperand out = new CPOperand(parts[5]);
			 
		if(!opcode.equalsIgnoreCase("rshape"))
			throw new DMLRuntimeException("Unknown opcode while parsing an MatrixReshapeInstruction: " + str);
		else
			return new MatrixReshapeCPInstruction(new Operator(true), in1, in2, in3, in4, out, opcode, str);
	}
	
	@Override
	public void processInstruction(ExecutionContext ec)
		throws DMLUnsupportedOperationException, DMLRuntimeException 
	{
		//get inputs
		MatrixBlock in = ec.getMatrixInput(input1.getName());
		int rows = (int)ec.getScalarInput(_opRows.getName(), _opRows.getValueType(), _opRows.isLiteral()).getLongValue(); //save cast
		int cols = (int)ec.getScalarInput(_opCols.getName(), _opCols.getValueType(), _opCols.isLiteral()).getLongValue(); //save cast
		BooleanObject byRow = (BooleanObject) ec.getScalarInput(_opByRow.getName(), ValueType.BOOLEAN, _opByRow.isLiteral());

		//execute operations 
		MatrixBlock out = new MatrixBlock();
		out = LibMatrixReorg.reshape(in, out, rows, cols, byRow.getBooleanValue());
		
		//set output and release inputs
		ec.setMatrixOutput(output.getName(), out);
		ec.releaseMatrixInput(input1.getName());
	}
	
}
