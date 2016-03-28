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

package org.apache.sysml.debug;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.TreeMap;
import java.util.Map.Entry;

import org.apache.sysml.runtime.controlprogram.ExternalFunctionProgramBlock;
import org.apache.sysml.runtime.controlprogram.ForProgramBlock;
import org.apache.sysml.runtime.controlprogram.FunctionProgramBlock;
import org.apache.sysml.runtime.controlprogram.IfProgramBlock;
import org.apache.sysml.runtime.controlprogram.Program;
import org.apache.sysml.runtime.controlprogram.ProgramBlock;
import org.apache.sysml.runtime.controlprogram.WhileProgramBlock;
import org.apache.sysml.runtime.instructions.Instruction;
import org.apache.sysml.runtime.instructions.MRJobInstruction;
import org.apache.sysml.runtime.instructions.cp.BreakPointInstruction;
import org.apache.sysml.runtime.instructions.cp.CPInstruction;


/**
 *  Class to produce mapping between source file lines and runtime instructions
 */
public class DMLDisassemble 
{
	private Program _prog;
	//Map between DML program line numbers and corresponding runtime instruction(s)
	private TreeMap<Integer, ArrayList<Instruction>> DMLInstMap; 
	
	/**
	 * Constructor for DML disassembler class 
	 * @param prog Program reference (e.g., function repository)
	 */
	public DMLDisassemble(Program prog) 
	{
		_prog = prog;
		DMLInstMap = new TreeMap<Integer, ArrayList<Instruction>>();
	}

	////////////////////////////////////////////////
	// getter and setter functionality
	////////////////////////////////////////////////
	
	/** 
	 * Getter for program reference field
	 * @return _prog Program reference
	 */
	public Program getProgram() 
	{
		return _prog;
	}

	/** 
	 * Getter for DML script to instruction map field
	 * @return DMLInstMap Map between DML script line numbers and runtime instruction(s)
	 */
	public TreeMap<Integer, ArrayList<Instruction>> getDMLInstMap() 
	{
		return DMLInstMap;
	}
	
	/**
	 * Setter for program reference field
	 * @param prog Program reference
	 */
	public void setProgram(Program prog) 
	{
		this._prog = prog;		
	}

	/**
	 * Setter for DML instruction map field (i.e. disassembler functionality) 
	 * For each DML script line, get runtime instructions (if any)
	 */
	public void setDMLInstMap() 
	{
		DMLInstMap = new TreeMap<Integer, ArrayList<Instruction>>();
		if (_prog != null)  
		{
			//Functions: For each function program block (if any), get instructions corresponding to each line number
			HashMap<String,FunctionProgramBlock> funcMap = this._prog.getFunctionProgramBlocks();
			if (funcMap != null && !funcMap.isEmpty() )
			{
				for (Entry<String, FunctionProgramBlock> e : funcMap.entrySet())
				{
					String fkey = e.getKey();
					FunctionProgramBlock fpb = e.getValue();
					if(fpb instanceof ExternalFunctionProgramBlock)
						System.err.println("----EXTERNAL FUNCTION "+fkey+"\n");
					else
					{
						System.err.println("----FUNCTION "+fkey+"\n");
						for (ProgramBlock pb : fpb.getChildBlocks())
							setProgramBlockInstMap(pb);
					}
				}
			}
			//Main program: for each program block, get instructions corresponding to current line number (if any)
			for (ProgramBlock pb : this._prog.getProgramBlocks())  
			{
				if (pb != null)
					setProgramBlockInstMap(pb);
			}
		}
	}
	
	/**
	 * For each program block, get runtime instructions (if any)
	 * @param pb Current program block 
	 */
	private void setProgramBlockInstMap(ProgramBlock pb) {		
		if (pb instanceof FunctionProgramBlock)
		{
			FunctionProgramBlock fpb = (FunctionProgramBlock)pb;
			for( ProgramBlock pbc : fpb.getChildBlocks() )
				setProgramBlockInstMap(pbc);
		}
		else if (pb instanceof WhileProgramBlock)
		{
			WhileProgramBlock wpb = (WhileProgramBlock) pb;
			this.setInstMap(wpb.getPredicate());			
			for( ProgramBlock pbc : wpb.getChildBlocks() )
				setProgramBlockInstMap(pbc);				
		}	
		else if (pb instanceof IfProgramBlock)
		{
			IfProgramBlock ipb = (IfProgramBlock) pb;
			this.setInstMap(ipb.getPredicate());
			for( ProgramBlock pbc : ipb.getChildBlocksIfBody() )
				setProgramBlockInstMap(pbc);
			if( !ipb.getChildBlocksElseBody().isEmpty() )
			{	
				for( ProgramBlock pbc : ipb.getChildBlocksElseBody() ) 
					setProgramBlockInstMap(pbc);
			}
		}
		else if (pb instanceof ForProgramBlock) //incl parfor
		{
			ForProgramBlock fpb = (ForProgramBlock) pb;
			this.setInstMap(fpb.getFromInstructions());
			this.setInstMap(fpb.getToInstructions());
			this.setInstMap(fpb.getIncrementInstructions());
			for( ProgramBlock pbc : fpb.getChildBlocks() ) 
				setProgramBlockInstMap(pbc);			
		}
		else
		{
			this.setInstMap(pb.getInstructions());
		}
	}

	/**
	 * For each instruction, generate map with corresponding DML 
	 * script line number 
	 * @param instructions Instructions for current program block 
	 */
	private void setInstMap(ArrayList<Instruction> instructions ) {
		for (int i = 0; i < instructions.size() ; i++) {
			Instruction currInst = instructions.get(i);
			if (currInst instanceof MRJobInstruction)  {
				MRJobInstruction currMRInst = (MRJobInstruction) currInst;
				for (Integer lineNumber : currMRInst.getMRJobInstructionsLineNumbers())  {
					if (!DMLInstMap.containsKey(lineNumber))
						DMLInstMap.put(lineNumber, new ArrayList<Instruction>());
					DMLInstMap.get(lineNumber).add(currMRInst);
				}
			}
			else if (currInst instanceof CPInstruction) {
				CPInstruction currCPInst = (CPInstruction) currInst;
				// Check if current instruction line number correspond to source line number
				if (!DMLInstMap.containsKey(currInst.getLineNum()))
					DMLInstMap.put(currInst.getLineNum(), new ArrayList<Instruction>());
				DMLInstMap.get(currInst.getLineNum()).add(currCPInst);
			}
			else if (currInst instanceof BreakPointInstruction) {
				BreakPointInstruction currBPInst = (BreakPointInstruction) currInst;
				// Check if current instruction line number correspond to source line number
				if (!DMLInstMap.containsKey(currInst.getLineNum()))
					DMLInstMap.put(currInst.getLineNum(), new ArrayList<Instruction>());
				DMLInstMap.get(currInst.getLineNum()).add(currBPInst);
			}			
		}
	}
}
