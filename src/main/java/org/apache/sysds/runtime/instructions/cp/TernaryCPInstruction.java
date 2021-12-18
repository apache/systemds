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

package org.apache.sysds.runtime.instructions.cp;

import org.apache.sysds.runtime.controlprogram.context.ExecutionContext;
import org.apache.sysds.runtime.instructions.InstructionUtils;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;
import org.apache.sysds.runtime.matrix.operators.TernaryOperator;

public class TernaryCPInstruction extends ComputationCPInstruction {
	
	protected TernaryCPInstruction(TernaryOperator op, CPOperand in1, CPOperand in2, CPOperand in3, CPOperand out, String opcode, String str) {
		super(CPType.Ternary, op, in1, in2, in3, out, opcode, str);
	}

	public static TernaryCPInstruction parseInstruction(String str)
	{
		String[] parts = InstructionUtils.getInstructionPartsWithValueType(str);
		String opcode=parts[0];
		CPOperand operand1 = new CPOperand(parts[1]);
		CPOperand operand2 = new CPOperand(parts[2]);
		CPOperand operand3 = new CPOperand(parts[3]);
		CPOperand outOperand = new CPOperand(parts[4]);
		int numThreads = parts.length>5 ? Integer.parseInt(parts[5]) : 1;
		TernaryOperator op = InstructionUtils.parseTernaryOperator(opcode, numThreads);
		if(operand1.isFrame() && operand2.isScalar() && opcode.contains("map"))
			return  new TernaryFrameScalarCPInstruction(op, operand1, operand2, operand3, outOperand, opcode, str);
		else
			return new TernaryCPInstruction(op, operand1, operand2, operand3, outOperand, opcode,str);
	}
	
	@Override
	public void processInstruction( ExecutionContext ec ) {
		if( input1.isMatrix() || input2.isMatrix() || input3.isMatrix() )
		{
			//get all inputs as matrix blocks
			MatrixBlock m1 = input1.isMatrix() ? ec.getMatrixInput(input1.getName()) :
				new MatrixBlock(ec.getScalarInput(input1).getDoubleValue());
			MatrixBlock m2 = input2.isMatrix() ? ec.getMatrixInput(input2.getName()) :
				new MatrixBlock(ec.getScalarInput(input2).getDoubleValue());
			MatrixBlock m3 = input3.isMatrix() ? ec.getMatrixInput(input3.getName()) :
				new MatrixBlock(ec.getScalarInput(input3).getDoubleValue());
			
			//execution
			MatrixBlock out = m1.ternaryOperations((TernaryOperator)_optr, m2, m3, new MatrixBlock());
			
			//release the inputs and output
			if( input1.isMatrix() )
				ec.releaseMatrixInput(input1.getName());
			if( input2.isMatrix() )
				ec.releaseMatrixInput(input2.getName());
			if( input3.isMatrix() )
				ec.releaseMatrixInput(input3.getName());
			ec.setMatrixOutput(output.getName(), out);
		}
		else { //SCALARS
			double value = ((TernaryOperator)_optr).fn.execute(
				ec.getScalarInput(input1).getDoubleValue(),
				ec.getScalarInput(input2).getDoubleValue(),
				ec.getScalarInput(input3).getDoubleValue());
			ec.setScalarOutput(output.getName(), ScalarObjectFactory
				.createScalarObject(output.getValueType(), value));
		}
	}
}
