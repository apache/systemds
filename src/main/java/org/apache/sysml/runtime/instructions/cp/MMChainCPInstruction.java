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

import org.apache.sysml.lops.MapMultChain.ChainType;
import org.apache.sysml.runtime.DMLRuntimeException;
import org.apache.sysml.runtime.DMLUnsupportedOperationException;
import org.apache.sysml.runtime.controlprogram.context.ExecutionContext;
import org.apache.sysml.runtime.instructions.Instruction;
import org.apache.sysml.runtime.instructions.InstructionUtils;
import org.apache.sysml.runtime.matrix.data.MatrixBlock;
import org.apache.sysml.runtime.matrix.operators.Operator;

/**
 * 
 * 
 */
public class MMChainCPInstruction extends UnaryCPInstruction
{	
	
	private ChainType _type = null;
	private int _numThreads = -1;
	
	public MMChainCPInstruction(Operator op, CPOperand in1, CPOperand in2, CPOperand in3, CPOperand out, ChainType type, int k, String opcode, String istr)
	{
		super(op, in1, in2, in3, out, opcode, istr);
		_cptype = CPINSTRUCTION_TYPE.MMChain;
		_type = type;
		_numThreads = k;
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
		//parse instruction parts (without exec type)
		String[] parts = InstructionUtils.getInstructionPartsWithValueType( str );		
		InstructionUtils.checkNumFields( parts, 5, 6 );
	
		String opcode = parts[0];
		CPOperand in1 = new CPOperand(parts[1]);
		CPOperand in2 = new CPOperand(parts[2]);
		
		if( parts.length==6 )
		{
			CPOperand out= new CPOperand(parts[3]);
			ChainType type = ChainType.valueOf(parts[4]);
			int k = Integer.parseInt(parts[5]);
			
			return new MMChainCPInstruction(null, in1, in2, null, out, type, k, opcode, str);
		}
		else //parts.length==7
		{
			CPOperand in3 = new CPOperand(parts[3]);
			CPOperand out = new CPOperand(parts[4]);
			ChainType type = ChainType.valueOf(parts[5]);
			int k = Integer.parseInt(parts[6]);
			
			return new MMChainCPInstruction(null, in1, in2, in3, out, type, k, opcode, str);
		}
	}
	
	@Override
	public void processInstruction(ExecutionContext ec)
		throws DMLUnsupportedOperationException, DMLRuntimeException 
	{
		//get inputs
		MatrixBlock X = ec.getMatrixInput(input1.getName());
		MatrixBlock v = ec.getMatrixInput(input2.getName());
		MatrixBlock w = (_type==ChainType.XtwXv) ? ec.getMatrixInput(input3.getName()) : null;

		//execute mmchain operation 
		 MatrixBlock out = (MatrixBlock) X.chainMatrixMultOperations(v, w, new MatrixBlock(), _type, _numThreads);
				
		//set output and release inputs
		ec.setMatrixOutput(output.getName(), out);
		ec.releaseMatrixInput(input1.getName());
		ec.releaseMatrixInput(input2.getName());
		if( w !=null )
			ec.releaseMatrixInput(input3.getName());
	}
	
	public ChainType getMMChainType()
	{
		return _type;
	}
}
