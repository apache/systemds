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

package org.apache.sysds.runtime.instructions.spark;


import org.apache.sysds.common.Opcodes;
import org.apache.sysds.runtime.DMLRuntimeException;
import org.apache.sysds.runtime.functionobjects.OffsetColumnIndex;
import org.apache.sysds.runtime.instructions.InstructionUtils;
import org.apache.sysds.runtime.instructions.cp.CPOperand;
import org.apache.sysds.runtime.matrix.operators.Operator;
import org.apache.sysds.runtime.matrix.operators.ReorgOperator;

public abstract class AppendMSPInstruction extends AppendSPInstruction {
	protected CPOperand _offset = null;

	protected AppendMSPInstruction(Operator op, CPOperand in1, CPOperand in2, CPOperand offset, CPOperand out,
		boolean cbind, String opcode, String istr) {
		super(SPType.MAppend, op, in1, in2, out, cbind, opcode, istr);
		_offset = offset;
	}

	public static AppendMSPInstruction parseInstruction( String str ) {
		String[] parts = InstructionUtils.getInstructionPartsWithValueType(str);
		InstructionUtils.checkNumFields (parts, 5);
		
		String opcode = parts[0];
		CPOperand in1 = new CPOperand(parts[1]);
		CPOperand in2 = new CPOperand(parts[2]);
		CPOperand offset = new CPOperand(parts[3]);
		CPOperand out = new CPOperand(parts[4]);
		boolean cbind = Boolean.parseBoolean(parts[5]);
		
		if(!opcode.equalsIgnoreCase(Opcodes.MAPPEND.toString()))
			throw new DMLRuntimeException("Unknown opcode while parsing a AppendMSPInstruction: " + str);
		
		//construct matrix/frame appendm instruction
		if( in1.getDataType().isMatrix() ) {
			return new MatrixAppendMSPInstruction(new ReorgOperator(OffsetColumnIndex
					.getOffsetColumnIndexFnObject(-1)), in1, in2, offset, out, cbind, opcode, str);
		}
		else { //frame
			return new FrameAppendMSPInstruction(new ReorgOperator(OffsetColumnIndex
					.getOffsetColumnIndexFnObject(-1)), in1, in2, offset, out, cbind, opcode, str);
		}
	}
}
