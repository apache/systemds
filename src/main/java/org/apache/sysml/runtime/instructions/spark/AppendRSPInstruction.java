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
import org.apache.sysml.runtime.functionobjects.OffsetColumnIndex;
import org.apache.sysml.runtime.instructions.InstructionUtils;
import org.apache.sysml.runtime.instructions.cp.CPOperand;
import org.apache.sysml.runtime.matrix.operators.Operator;
import org.apache.sysml.runtime.matrix.operators.ReorgOperator;


public abstract class AppendRSPInstruction extends BinarySPInstruction
{
	protected boolean _cbind = true;
	
	public AppendRSPInstruction(Operator op, CPOperand in1, CPOperand in2, CPOperand out, boolean cbind, String opcode, String istr)
	{
		super(op, in1, in2, out, opcode, istr);
		_sptype = SPINSTRUCTION_TYPE.RAppend;
		_cbind = cbind;
	}

	public static AppendRSPInstruction parseInstruction ( String str ) 
		throws DMLRuntimeException 
	{	
		String[] parts = InstructionUtils.getInstructionPartsWithValueType(str);
		InstructionUtils.checkNumFields (parts, 4);
		
		String opcode = parts[0];
		CPOperand in1 = new CPOperand(parts[1]);
		CPOperand in2 = new CPOperand(parts[2]);
		CPOperand out = new CPOperand(parts[3]);
		boolean cbind = Boolean.parseBoolean(parts[4]);
		
		if(!opcode.equalsIgnoreCase("rappend"))
			throw new DMLRuntimeException("Unknown opcode while parsing a MatrixAppendRSPInstruction: " + str);
		
		if( in1.getDataType().isMatrix() ) {
			return new MatrixAppendRSPInstruction(new ReorgOperator(OffsetColumnIndex
					.getOffsetColumnIndexFnObject(-1)), in1, in2, out, cbind, opcode, str);
		}
		else { //frame

			return new FrameAppendRSPInstruction(new ReorgOperator(OffsetColumnIndex
					.getOffsetColumnIndexFnObject(-1)), in1, in2, out, cbind, opcode, str);
		}
	}
}

