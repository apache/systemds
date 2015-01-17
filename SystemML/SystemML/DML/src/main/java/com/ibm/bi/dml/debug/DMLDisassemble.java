/**
 * IBM Confidential
 * OCO Source Materials
 * (C) Copyright IBM Corp. 2010, 2015
 * The source code for this program is not published or otherwise divested of its trade secrets, irrespective of what has been deposited with the U.S. Copyright Office.
 */

package com.ibm.bi.dml.debug;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.TreeMap;
import java.util.Map.Entry;

import com.ibm.bi.dml.runtime.controlprogram.ExternalFunctionProgramBlock;
import com.ibm.bi.dml.runtime.controlprogram.ForProgramBlock;
import com.ibm.bi.dml.runtime.controlprogram.FunctionProgramBlock;
import com.ibm.bi.dml.runtime.controlprogram.IfProgramBlock;
import com.ibm.bi.dml.runtime.controlprogram.Program;
import com.ibm.bi.dml.runtime.controlprogram.ProgramBlock;
import com.ibm.bi.dml.runtime.controlprogram.WhileProgramBlock;
import com.ibm.bi.dml.runtime.instructions.Instruction;
import com.ibm.bi.dml.runtime.instructions.MRJobInstruction;
import com.ibm.bi.dml.runtime.instructions.cp.BreakPointInstruction;
import com.ibm.bi.dml.runtime.instructions.cp.CPInstruction;


/**
 *  Class to produce mapping between source file lines and runtime instructions
 */
public class DMLDisassemble 
{
	@SuppressWarnings("unused")
	private static final String _COPYRIGHT = "Licensed Materials - Property of IBM\n(C) Copyright IBM Corp. 2010, 2015\n" +
                                             "US Government Users Restricted Rights - Use, duplication  disclosure restricted by GSA ADP Schedule Contract with IBM Corp.";
	Program _prog;
	//Map between DML program line numbers and corresponding runtime instruction(s)
	TreeMap<Integer, ArrayList<Instruction>> DMLInstMap; 
	
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
