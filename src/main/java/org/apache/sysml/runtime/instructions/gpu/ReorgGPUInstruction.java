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

package org.apache.sysml.runtime.instructions.gpu;

import org.apache.sysml.runtime.DMLRuntimeException;
import org.apache.sysml.runtime.controlprogram.caching.MatrixObject;
import org.apache.sysml.runtime.controlprogram.context.ExecutionContext;
import org.apache.sysml.runtime.functionobjects.SwapIndex;
import org.apache.sysml.runtime.instructions.InstructionUtils;
import org.apache.sysml.runtime.instructions.cp.CPOperand;
import org.apache.sysml.runtime.matrix.data.LibMatrixCUDA;
import org.apache.sysml.runtime.matrix.operators.Operator;
import org.apache.sysml.runtime.matrix.operators.ReorgOperator;
import org.apache.sysml.utils.GPUStatistics;


public class ReorgGPUInstruction extends GPUInstruction
{
 	private CPOperand _input;
    private CPOperand _output;
 	/**
 	 * for opcodes r'
 	 * 
 	 * @param op operator
 	 * @param in input operand
 	 * @param out output operand
 	 * @param opcode the opcode
 	 * @param istr instruction string
 	 */
	public ReorgGPUInstruction(Operator op, CPOperand in, CPOperand out, String opcode, String istr) {
		super(op, opcode, istr);
		_gputype = GPUINSTRUCTION_TYPE.Reorg;
		_input = in;
        _output = out;
	}
	
	public static ReorgGPUInstruction parseInstruction ( String str ) 
		throws DMLRuntimeException 
	{
		String[] parts = InstructionUtils.getInstructionPartsWithValueType(str);
        InstructionUtils.checkNumFields ( parts, 2 );
        
		String opcode = parts[0];
		CPOperand in = new CPOperand(parts[1]);
		CPOperand out = new CPOperand(parts[2]);	
					
		if ( !(opcode.equalsIgnoreCase("r'")) ) {
			throw new DMLRuntimeException("Unknown opcode while parsing a ReorgInstruction: " + str);
		}
		else
			return new ReorgGPUInstruction(new ReorgOperator(SwapIndex.getSwapIndexFnObject()), in, out, opcode, str);
	}
	
	@Override
	public void processInstruction(ExecutionContext ec)
			throws DMLRuntimeException 
	{
		GPUStatistics.incrementNoOfExecutedGPUInst();
		//acquire input
		MatrixObject mat = getMatrixInputForGPUInstruction(ec, _input.getName());

		int rlen = (int) mat.getNumColumns();
		int clen = (int) mat.getNumRows();
		
		//execute operation
		ec.setMetaData(_output.getName(), rlen, clen);
		LibMatrixCUDA.transpose(ec, ec.getGPUContext(0), getExtendedOpcode(), mat, _output.getName());
		
		//release inputs/outputs
		ec.releaseMatrixInputForGPUInstruction(_input.getName());
		ec.releaseMatrixOutputForGPUInstruction(_output.getName());
	}
}