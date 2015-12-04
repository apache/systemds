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

package org.apache.sysml.runtime.instructions;

import org.apache.sysml.lops.LopProperties.ExecType;
import org.apache.sysml.runtime.DMLRuntimeException;
import org.apache.sysml.runtime.DMLUnsupportedOperationException;
import org.apache.sysml.runtime.instructions.cp.CPInstruction.CPINSTRUCTION_TYPE;
import org.apache.sysml.runtime.instructions.mr.MRInstruction.MRINSTRUCTION_TYPE;
import org.apache.sysml.runtime.instructions.spark.SPInstruction.SPINSTRUCTION_TYPE;


public class InstructionParser 
{
		
	public static Instruction parseSingleInstruction ( String str ) throws DMLUnsupportedOperationException, DMLRuntimeException {
		
		if ( str == null || str.isEmpty() )
			return null;
		
		String execType = str.split(Instruction.OPERAND_DELIM)[0]; 
		if (   execType.equalsIgnoreCase(ExecType.CP.toString()) 
			|| execType.equalsIgnoreCase(ExecType.CP_FILE.toString()) ) 
		{
			CPINSTRUCTION_TYPE cptype = InstructionUtils.getCPType(str); 
			return CPInstructionParser.parseSingleInstruction (cptype, str);
		}
		else if (   execType.equalsIgnoreCase(ExecType.SPARK.toString()) ) 
		{
			SPINSTRUCTION_TYPE sptype = InstructionUtils.getSPType(str); 
			return SPInstructionParser.parseSingleInstruction (sptype, str);
		}
		else if ( execType.equalsIgnoreCase("MR") ) {
			MRINSTRUCTION_TYPE mrtype = InstructionUtils.getMRType(str); 
			if ( mrtype == null )
				throw new DMLRuntimeException("Can not determine MRType for instruction: " + str);
			return MRInstructionParser.parseSingleInstruction (mrtype, str);
		}
		else {
			throw new DMLRuntimeException("Unknown execution type in instruction: " + str);
		}
	}
	
	public static Instruction[] parseMixedInstructions ( String str ) throws DMLUnsupportedOperationException, DMLRuntimeException {
		if ( str == null || str.isEmpty() )
			return null;
		
		String[] strlist = str.split(Instruction.INSTRUCTION_DELIM);
		Instruction[] inst = new Instruction[strlist.length];
		
		for ( int i=0; i < inst.length; i++ ) {
			inst[i] = parseSingleInstruction ( strlist[i] );
		}
		
		return inst;
	}
	
	public static void printInstructions(Instruction[] instructions)
	{
		for(Instruction ins: instructions)
			System.out.println(ins.toString());
	}

}
