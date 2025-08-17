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

package org.apache.sysds.runtime.instructions;

import org.apache.sysds.common.InstructionType;
import org.apache.sysds.common.Types.ExecType;
import org.apache.sysds.runtime.DMLRuntimeException;
import org.apache.sysds.runtime.instructions.gpu.GPUInstruction.GPUINSTRUCTION_TYPE;

public class InstructionParser 
{
	public static Instruction parseSingleInstruction ( String str ) {
		if ( str == null || str.isEmpty() )
			return null;
		
		ExecType et = InstructionUtils.getExecType(str);
		switch( et ) {
			case CP:
			case CP_FILE: 
				InstructionType cptype = InstructionUtils.getCPType(str);
				if( cptype == null )
					throw new DMLRuntimeException("Unknown CP instruction: " + str);
				return CPInstructionParser.parseSingleInstruction (cptype, str);
			case SPARK: 
				InstructionType sptype = InstructionUtils.getSPType(str);
				if( sptype == null )
					throw new DMLRuntimeException("Unknown SPARK instruction: " + str);
				return SPInstructionParser.parseSingleInstruction (sptype, str);
			case GPU: 
				GPUINSTRUCTION_TYPE gputype = InstructionUtils.getGPUType(str);
				if( gputype == null )
					throw new DMLRuntimeException("Unknown GPU instruction: " + str);
				return GPUInstructionParser.parseSingleInstruction (gputype, str);
			case FED: 
				InstructionType fedtype = InstructionUtils.getFEDType(str);
				if( fedtype == null )
					throw new DMLRuntimeException("Unknown FEDERATED instruction: " + str);
				return FEDInstructionParser.parseSingleInstruction (fedtype, str);
			case OOC:
				// --- THIS IS THE WORKAROUND ---
				// Manually check for our new 'tee' opcode before the general lookup.
				if ( InstructionUtils.getOpCode(str).equals("tee") ) {
					return OOCInstructionParser.parseSingleInstruction(
							InstructionType.Tee, str);
				}
				// --- END OF WORKAROUND ---
				InstructionType ooctype = InstructionUtils.getOOCType(str);
				if( ooctype == null )
					throw new DMLRuntimeException("Unknown OOC instruction: " + str);
				return OOCInstructionParser.parseSingleInstruction (ooctype, str);
			default:
				throw new DMLRuntimeException("Unknown execution type in instruction: " + str);
		}
	}
	
	public static Instruction[] parseMixedInstructions ( String str ) {
		if ( str == null || str.isEmpty() )
			return null;
		String[] strlist = str.split(Instruction.INSTRUCTION_DELIM);
		Instruction[] inst = new Instruction[strlist.length];
		for ( int i=0; i < inst.length; i++ )
			inst[i] = parseSingleInstruction ( strlist[i] );
		return inst;
	}
}
