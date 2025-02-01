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
import org.apache.sysds.lops.MMTSJ.MMTSJType;
import org.apache.sysds.runtime.DMLRuntimeException;
import org.apache.sysds.runtime.controlprogram.context.ExecutionContext;
import org.apache.sysds.runtime.instructions.InstructionUtils;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;
import org.apache.sysds.runtime.matrix.operators.Operator;

public class MMTSJCPInstruction extends UnaryCPInstruction {
	private final MMTSJType _type;
	private final int _numThreads;

	private MMTSJCPInstruction(Operator op, CPOperand in1, MMTSJType type, CPOperand out, int k, String opcode,
			String istr) {
		super(CPType.MMTSJ, op, in1, out, opcode, istr);
		_type = type;
		_numThreads = k;
	}

	public static MMTSJCPInstruction parseInstruction ( String str ) {
		String[] parts = InstructionUtils.getInstructionPartsWithValueType(str);
		InstructionUtils.checkNumFields ( parts, 4 );
		
		String opcode = parts[0];
		CPOperand in1 = new CPOperand(parts[1]);
		CPOperand out = new CPOperand(parts[2]);
		MMTSJType titype = MMTSJType.valueOf(parts[3]);
		int k = Integer.parseInt(parts[4]);
		
		if(!opcode.equalsIgnoreCase(Opcodes.TSMM.toString()))
			throw new DMLRuntimeException("Unknown opcode while parsing an MMTSJCPInstruction: " + str);
		else
			return new MMTSJCPInstruction(new Operator(true), in1, titype, out, k, opcode, str);
	}
	
	@Override
	public void processInstruction(ExecutionContext ec) {
		//get inputs
		MatrixBlock matBlock1 = ec.getMatrixInput(input1.getName());

		//execute operations 
		MatrixBlock ret = matBlock1.transposeSelfMatrixMultOperations(new MatrixBlock(), _type, _numThreads );
		
		//set output and release inputs
		ec.setMatrixOutput(output.getName(), ret);
		ec.releaseMatrixInput(input1.getName());
	}
	
	public MMTSJType getMMTSJType()
	{
		return _type;
	}

	public int getNumThreads() {
		return _numThreads;
	}
}
