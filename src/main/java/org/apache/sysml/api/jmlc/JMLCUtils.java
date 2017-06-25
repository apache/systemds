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

package org.apache.sysml.api.jmlc;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashSet;
import java.util.Map;
import java.util.Map.Entry;

import org.apache.sysml.runtime.controlprogram.ForProgramBlock;
import org.apache.sysml.runtime.controlprogram.FunctionProgramBlock;
import org.apache.sysml.runtime.controlprogram.IfProgramBlock;
import org.apache.sysml.runtime.controlprogram.Program;
import org.apache.sysml.runtime.controlprogram.ProgramBlock;
import org.apache.sysml.runtime.controlprogram.WhileProgramBlock;
import org.apache.sysml.runtime.instructions.Instruction;
import org.apache.sysml.runtime.instructions.cp.CPOperand;
import org.apache.sysml.runtime.instructions.cp.VariableCPInstruction;

/**
 * Utility class containing static methods for working with JMLC.
 *
 */
public class JMLCUtils 
{
	/**
	 * Removes rmvar instructions that would remove any of the given outputs.
	 * This is important for keeping registered outputs after the program terminates.
	 * 
	 * @param prog the DML/PyDML program
	 * @param outputs registered output variables
	 */
	public static void cleanupRuntimeProgram( Program prog, String[] outputs)
	{
		Map<String, FunctionProgramBlock> funcMap = prog.getFunctionProgramBlocks();
		HashSet<String> blacklist = new HashSet<String>(Arrays.asList(outputs));
		
		if( funcMap != null && !funcMap.isEmpty() )
		{
			for( Entry<String, FunctionProgramBlock> e : funcMap.entrySet() )
			{
				FunctionProgramBlock fpb = e.getValue();
				for( ProgramBlock pb : fpb.getChildBlocks() )
					rCleanupRuntimeProgram(pb, blacklist);
			}
		}
		
		for( ProgramBlock pb : prog.getProgramBlocks() )
			rCleanupRuntimeProgram(pb, blacklist);
	}
	
	/**
	 * Cleanup program blocks (called recursively).
	 * 
	 * @param pb program block
	 * @param outputs registered output variables
	 */
	public static void rCleanupRuntimeProgram( ProgramBlock pb, HashSet<String> outputs )
	{
		if( pb instanceof WhileProgramBlock )
		{
			WhileProgramBlock wpb = (WhileProgramBlock)pb;
			for( ProgramBlock pbc : wpb.getChildBlocks() )
				rCleanupRuntimeProgram(pbc,outputs);
		}
		else if( pb instanceof IfProgramBlock )
		{
			IfProgramBlock ipb = (IfProgramBlock)pb;
			for( ProgramBlock pbc : ipb.getChildBlocksIfBody() )
				rCleanupRuntimeProgram(pbc,outputs);
			for( ProgramBlock pbc : ipb.getChildBlocksElseBody() )
				rCleanupRuntimeProgram(pbc,outputs);
		}
		else if( pb instanceof ForProgramBlock )
		{
			ForProgramBlock fpb = (ForProgramBlock)pb;
			for( ProgramBlock pbc : fpb.getChildBlocks() )
				rCleanupRuntimeProgram(pbc,outputs);
		}
		else {
			pb.setInstructions(cleanupRuntimeInstructions(
				pb.getInstructions(), outputs));
		}
	}
	
	/**
	 * Cleanup runtime instructions, removing rmvar instructions for
	 * any of the given output variable names.
	 * 
	 * @param insts list of instructions
	 * @param outputs registered output variables
	 * @return list of instructions
	 */
	public static ArrayList<Instruction> cleanupRuntimeInstructions( ArrayList<Instruction> insts, String[] outputs ) {
		return cleanupRuntimeInstructions(insts, new HashSet<String>(Arrays.asList(outputs)));
	}
	
	/**
	 * Cleanup runtime instructions, removing rmvar instructions for
	 * any of the given output variable names.
	 * 
	 * @param insts list of instructions
	 * @param outputs registered output variables
	 * @return list of instructions
	 */
	public static ArrayList<Instruction> cleanupRuntimeInstructions( ArrayList<Instruction> insts, HashSet<String> outputs )
	{
		ArrayList<Instruction> ret = new ArrayList<Instruction>();
		
		for( Instruction inst : insts ) {
			if( inst instanceof VariableCPInstruction && ((VariableCPInstruction)inst).isRemoveVariable() )
			{
				ArrayList<String> currRmVar = new ArrayList<String>();
				for( CPOperand input : ((VariableCPInstruction)inst).getInputs() )
					if( !outputs.contains(input.getName()) )
						currRmVar.add(input.getName());
				if( !currRmVar.isEmpty() ) {
					ret.add(VariableCPInstruction.prepareRemoveInstruction(
						currRmVar.toArray(new String[0])));
				}
			}
			else
				ret.add(inst);
		}
		return ret;
	}
}
