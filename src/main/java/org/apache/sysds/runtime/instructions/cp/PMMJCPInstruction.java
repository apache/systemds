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

import org.apache.sysds.common.Opcodes;
import org.apache.sysds.runtime.DMLRuntimeException;
import org.apache.sysds.runtime.controlprogram.context.ExecutionContext;
import org.apache.sysds.runtime.instructions.InstructionUtils;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;
import org.apache.sysds.runtime.matrix.operators.Operator;

public class PMMJCPInstruction extends ComputationCPInstruction {

	private final int _numThreads;

	private PMMJCPInstruction(Operator op, CPOperand in1, CPOperand in2, CPOperand in3, CPOperand out, int k,
			String opcode, String istr) {
		super(CPType.AggregateBinary, op, in1, in2, in3, out, opcode, istr);
		_numThreads = k;
	}

	public static PMMJCPInstruction parseInstruction ( String str ) {
		String[] parts = InstructionUtils.getInstructionPartsWithValueType(str);
		InstructionUtils.checkNumFields ( parts, 5 );
		String opcode = parts[0];
		CPOperand in1 = new CPOperand(parts[1]);
		CPOperand in2 = new CPOperand(parts[2]);
		CPOperand in3 = new CPOperand(parts[3]);
		CPOperand out = new CPOperand(parts[4]);
		int k = Integer.parseInt(parts[5]);
		if(!opcode.equalsIgnoreCase(Opcodes.PMM.toString()))
			throw new DMLRuntimeException("Unknown opcode while parsing an PMMJCPInstruction: " + str);
		else
			return new PMMJCPInstruction(new Operator(true), in1, in2, in3, out, k, opcode, str);
	}
	
	@Override
	public void processInstruction(ExecutionContext ec) {
		//get inputs
		MatrixBlock matBlock1 = ec.getMatrixInput(input1.getName());
		MatrixBlock matBlock2 = ec.getMatrixInput(input2.getName());
		int rlen = (int)ec.getScalarInput(input3).getLongValue();
		//execute operations
		MatrixBlock ret = new MatrixBlock(rlen, matBlock2.getNumColumns(), matBlock2.isInSparseFormat());
		matBlock1.permutationMatrixMultOperations(matBlock2, ret, null, _numThreads);
		//set output and release inputs
		ec.setMatrixOutput(output.getName(), ret);
		ec.releaseMatrixInput(input1.getName(), input2.getName());
	}
}
