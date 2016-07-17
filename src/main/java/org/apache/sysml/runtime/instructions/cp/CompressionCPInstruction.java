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

import org.apache.sysml.runtime.DMLRuntimeException;
import org.apache.sysml.runtime.compress.CompressedMatrixBlock;
import org.apache.sysml.runtime.controlprogram.context.ExecutionContext;
import org.apache.sysml.runtime.instructions.Instruction;
import org.apache.sysml.runtime.instructions.InstructionUtils;
import org.apache.sysml.runtime.matrix.data.MatrixBlock;
import org.apache.sysml.runtime.matrix.operators.Operator;

public class CompressionCPInstruction extends UnaryCPInstruction
{
	
	public CompressionCPInstruction(Operator op, CPOperand in, CPOperand out, String opcode, String istr){
		super(op, in, null, null, out, opcode, istr);
	}
	
	public static Instruction parseInstruction(String str)
		throws DMLRuntimeException 
	{
		String[] parts = InstructionUtils.getInstructionPartsWithValueType(str);
		String opcode = parts[0];		
		CPOperand in1 = new CPOperand(parts[1]);
		CPOperand out = new CPOperand(parts[2]);
		
		return new CompressionCPInstruction(null, in1, out, opcode, str);				
	}
	
	@Override
	public void processInstruction( ExecutionContext ec )
		throws DMLRuntimeException
	{
		//get matrix block input
		MatrixBlock in = ec.getMatrixInput(input1.getName());
		
		//compress the matrix block
		CompressedMatrixBlock cmb = new CompressedMatrixBlock(in);
		cmb.compress();
		
		//set output and release input
		ec.releaseMatrixInput(input1.getName());
		ec.setMatrixOutput(output.getName(), cmb);
	}
}
