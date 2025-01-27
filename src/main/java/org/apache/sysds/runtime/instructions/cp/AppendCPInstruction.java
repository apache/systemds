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
import org.apache.sysds.common.Types.DataType;
import org.apache.sysds.runtime.DMLRuntimeException;
import org.apache.sysds.runtime.functionobjects.OffsetColumnIndex;
import org.apache.sysds.runtime.instructions.InstructionUtils;
import org.apache.sysds.runtime.matrix.operators.Operator;
import org.apache.sysds.runtime.matrix.operators.ReorgOperator;


public abstract class AppendCPInstruction extends BinaryCPInstruction
{	
	public enum AppendType{
		CBIND,
		RBIND,
		STRING,
		LIST,
	}

	//type (matrix cbind / scalar string concatenation)
	protected final AppendType _type;

	protected AppendCPInstruction(Operator op, CPOperand in1, CPOperand in2, CPOperand out,
			AppendType type, String opcode, String istr) {
		super(CPType.Append, op, in1, in2, out, opcode, istr);
		_type = type;
	}

	public static AppendCPInstruction parseInstruction ( String str ) {
		String[] parts = InstructionUtils.getInstructionPartsWithValueType(str);
		InstructionUtils.checkNumFields (parts, 5, 4);
		
		String opcode = parts[0];
		CPOperand in1 = new CPOperand(parts[1]);
		CPOperand in2 = new CPOperand(parts[2]);
		CPOperand out = new CPOperand(parts[parts.length-2]);
		boolean cbind = Boolean.parseBoolean(parts[parts.length-1]);
		
		AppendType type = (in1.getDataType()!=DataType.MATRIX && in1.getDataType()!=DataType.FRAME) ? 
			in1.getDataType()==DataType.LIST ? AppendType.LIST : AppendType.STRING : 
			cbind ? AppendType.CBIND : AppendType.RBIND;
		
		if(!opcode.equalsIgnoreCase(Opcodes.APPEND.toString()) && !opcode.equalsIgnoreCase(Opcodes.REMOVE.toString()))
			throw new DMLRuntimeException("Unknown opcode while parsing a AppendCPInstruction: " + str);

		Operator op = new ReorgOperator(OffsetColumnIndex.getOffsetColumnIndexFnObject(-1));
		if( type == AppendType.STRING )
			return new ScalarAppendCPInstruction(op, in1, in2, out, type, opcode, str);
		else if( type == AppendType.LIST )
			return new ListAppendRemoveCPInstruction(op, in1, in2, out, type, opcode, str);
		else if( in1.getDataType()==DataType.MATRIX )
			return new MatrixAppendCPInstruction(op, in1, in2, out, type, opcode, str);
		else //DataType.FRAME
			return new FrameAppendCPInstruction(op, in1, in2, out, type, opcode, str);
	}

	public AppendType getAppendType() {
		return _type;
	}
}
