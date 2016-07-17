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

import org.apache.sysml.runtime.DMLRuntimeException;
import org.apache.sysml.runtime.controlprogram.context.ExecutionContext;
import org.apache.sysml.runtime.controlprogram.context.SparkExecutionContext;
import org.apache.sysml.runtime.functionobjects.MinusMultiply;
import org.apache.sysml.runtime.functionobjects.PlusMultiply;
import org.apache.sysml.runtime.functionobjects.ValueFunctionWithConstant;
import org.apache.sysml.runtime.instructions.InstructionUtils;
import org.apache.sysml.runtime.instructions.cp.CPOperand;
import org.apache.sysml.runtime.instructions.cp.ScalarObject;
import org.apache.sysml.runtime.matrix.operators.BinaryOperator;

public class PlusMultSPInstruction extends  ArithmeticBinarySPInstruction
{
	public PlusMultSPInstruction(BinaryOperator op, CPOperand in1, CPOperand in2, 
			CPOperand in3, CPOperand out, String opcode, String str) throws DMLRuntimeException 
	{
		super(op, in1, in2, out, opcode, str);
		input3= in3;
		
		//sanity check opcodes
		if ( !(  opcode.equalsIgnoreCase("+*") || opcode.equalsIgnoreCase("-*")  ) ) 
		{
			throw new DMLRuntimeException("Unknown opcode in PlusMultSPInstruction: " + toString());
		}		
	}
	public static PlusMultSPInstruction parseInstruction(String str) throws DMLRuntimeException
	{
		String[] parts = InstructionUtils.getInstructionPartsWithValueType(str);
		String opcode=parts[0];
		CPOperand operand1 = new CPOperand(parts[1]);
		CPOperand operand2 = new CPOperand(parts[3]);	//put the second matrix (parts[3]) in Operand2 to make using Binary matrix operations easier
		CPOperand operand3 = new CPOperand(parts[2]);
		CPOperand outOperand = new CPOperand(parts[4]);
		BinaryOperator bOperator = null;
		if(opcode.equals("+*"))
			bOperator = new BinaryOperator(new PlusMultiply());
		else if (opcode.equals("-*"))
			bOperator = new BinaryOperator(new MinusMultiply());
		return new PlusMultSPInstruction(bOperator,operand1, operand2, operand3, outOperand, opcode,str);	
	}
	
	
	@Override
	public void processInstruction(ExecutionContext ec) 
		throws DMLRuntimeException
	{
		SparkExecutionContext sec = (SparkExecutionContext)ec;
		
		//pass the scalar
		ScalarObject constant = (ScalarObject) ec.getScalarInput(input3.getName(), input3.getValueType(), input3.isLiteral());
		((ValueFunctionWithConstant) ((BinaryOperator)_optr).fn).setConstant(constant.getDoubleValue());

		super.processMatrixMatrixBinaryInstruction(sec);
	
	}

}