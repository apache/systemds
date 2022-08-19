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

import org.apache.sysds.lops.SortKeys;
import org.apache.sysds.common.Types.DataType;
import org.apache.sysds.common.Types.ValueType;
import org.apache.sysds.runtime.DMLRuntimeException;
import org.apache.sysds.runtime.controlprogram.context.ExecutionContext;
import org.apache.sysds.runtime.instructions.InstructionUtils;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;

/**
 * This class supports two variants of sort operation on a 1-dimensional input matrix. 
 * The two variants are <code> weighted </code> and <code> unweighted </code>.
 * Example instructions: 
 *     sort:mVar1:mVar2 (input=mVar1, output=mVar2)
 *     sort:mVar1:mVar2:mVar3 (input=mVar1, weights=mVar2, output=mVar3)
 *  
 */
public class QuantileSortCPInstruction extends UnaryCPInstruction {
	int _numThreads;

	private QuantileSortCPInstruction(CPOperand in, CPOperand out, String opcode, String istr, int k) {
		this(in, null, out, opcode, istr, k);
	}

	private QuantileSortCPInstruction(CPOperand in1, CPOperand in2, CPOperand out, String opcode,
			String istr, int k) {
		super(CPType.QSort, null, in1, in2, out, opcode, istr);
		_numThreads = k;
	}

	private static void parseInstruction(String instr, CPOperand in1, CPOperand in2, CPOperand out) {
		String[] parts = InstructionUtils.getInstructionPartsWithValueType(instr);

		out.split(parts[parts.length-2]);

		switch(parts.length) {
			case 4:
				in1.split(parts[1]);
				in2 = null;
				break;
			case 5:
				in1.split(parts[1]);
				in2.split(parts[2]);
				break;
			default:
				throw new DMLRuntimeException("Unexpected number of operands in the instruction: " + instr);
		}
	}

	public static QuantileSortCPInstruction parseInstruction ( String str ) {
		CPOperand in1 = new CPOperand("", ValueType.UNKNOWN, DataType.UNKNOWN);
		CPOperand in2 = null;
		CPOperand out = new CPOperand("", ValueType.UNKNOWN, DataType.UNKNOWN);
		
		String[] parts = InstructionUtils.getInstructionPartsWithValueType(str);
		String opcode = parts[0];
		
		if ( opcode.equalsIgnoreCase(SortKeys.OPCODE) ) {
			int k = Integer.parseInt(parts[parts.length-1]); //#threads
			if ( parts.length == 4 ) {
				// Example: sort:mVar1:mVar2 (input=mVar1, output=mVar2)
				InstructionUtils.checkNumFields(str, 3);
				parseInstruction(str, in1, null, out);
				return new QuantileSortCPInstruction(in1, out, opcode, str, k);
			}
			else if ( parts.length == 5 ) {
				// Example: sort:mVar1:mVar2:mVar3 (input=mVar1, weights=mVar2, output=mVar3)
				InstructionUtils.checkNumFields(str, 4);
				in2 = new CPOperand("", ValueType.UNKNOWN, DataType.UNKNOWN);
				parseInstruction(str, in1, in2, out);
				return new QuantileSortCPInstruction(in1, in2, out, opcode, str, k);
			}
			else {
				throw new DMLRuntimeException("Invalid number of operands in instruction: " + str);
			}
		} 
		else {
			throw new DMLRuntimeException("Unknown opcode while parsing a QuantileSortCPInstruction: " + str);
		}
	}
	
	@Override
	public void processInstruction(ExecutionContext ec) {
		//acquire inputs matrices
		MatrixBlock matBlock = ec.getMatrixInput(input1.getName());
		MatrixBlock wtBlock = null;
 		if (input2 != null) {
			wtBlock = ec.getMatrixInput(input2.getName());
		}
		
 		//process core instruction
		MatrixBlock resultBlock = matBlock.sortOperations(wtBlock, new MatrixBlock(), _numThreads);
		
		//release inputs
		ec.releaseMatrixInput(input1.getName());
		if (input2 != null)
			ec.releaseMatrixInput(input2.getName());
		
		//set and release output
		ec.setMatrixOutput(output.getName(), resultBlock);
	}

	public int getNumThreads() {
		return _numThreads;
	}
}
