/**
 * IBM Confidential
 * OCO Source Materials
 * (C) Copyright IBM Corp. 2010, 2014
 * The source code for this program is not published or otherwise divested of its trade secrets, irrespective of what has been deposited with the U.S. Copyright Office.
 */

package com.ibm.bi.dml.runtime.instructions.CPInstructions;

import com.ibm.bi.dml.debugger.DebugState;
import com.ibm.bi.dml.runtime.DMLRuntimeException;
import com.ibm.bi.dml.runtime.controlprogram.ExecutionContext;
import com.ibm.bi.dml.runtime.instructions.Instruction;


/**
 * Class for breakpoint instructions 
 * Note: ONLY USED FOR DEBUGGING PURPOSES  
 */
public class BreakPointInstruction extends Instruction
{
	@SuppressWarnings("unused")
	private static final String _COPYRIGHT = "Licensed Materials - Property of IBM\n(C) Copyright IBM Corp. 2010, 2014\n" +
                                             "US Government Users Restricted Rights - Use, duplication  disclosure restricted by GSA ADP Schedule Contract with IBM Corp.";

	public enum BPINSTRUCTION_STATUS { INVISIBLE, ENABLED, DISABLED };
	
	private BPINSTRUCTION_STATUS bpStatus; //indicates breakpoint status	
	private String location=null; //namespace and name of function containing breakpoint


	/**
	 * Constructor for a breakpoint instruction
	 */
	public BreakPointInstruction() {
		type = INSTRUCTION_TYPE.BREAKPOINT;
		bpStatus = BPINSTRUCTION_STATUS.ENABLED;
	}

	/**
	 * Parameterized constructor for a breakpoint instruction
	 * @param tp Breakpoint instruction status
	 */
	public BreakPointInstruction(BPINSTRUCTION_STATUS status) {
		type = INSTRUCTION_TYPE.BREAKPOINT;
		bpStatus = status;
	}

	/**
	 * Setter for breakpoint instruction status
     * @param st Breakpoint current status
	 */
	public void setBPInstructionStatus(BPINSTRUCTION_STATUS status) {
		bpStatus = status;
	}
		
	/**
	 * Setter for a breakpoint instruction location
	 * @param loc Namespace and name of function where breakpoint instruction was inserted 
	 */
	public void setBPInstructionLocation(String loc) {
		location = loc;
	}
		
	/**
	 * Getter for breakpoint instruction status
	 * @return Breakpoint instruction current status. True if enabled, false otherwise)
	 */
	public BPINSTRUCTION_STATUS getBPInstructionStatus() {
		return bpStatus;
	}
	
	/**
	 * Getter for a breakpoint instruction namespace
	 * @return Namespace and name of function where breakpoint instruction was inserted  
	 */
	public String getBPInstructionLocation() {
		return location;
	}
		
	/**
	 * Check if breakpoint instruction is enabled 
	 * @return If true, BPInstruction is currently enabled, otherwise it is disabled.
	 */
	public boolean isBPInstructionEnabled() {
		return (bpStatus == BPINSTRUCTION_STATUS.ENABLED);
	}
	
	@Override
	public byte[] getInputIndexes() throws DMLRuntimeException {
		// TODO Auto-generated method stub
		return null;
	}

	@Override
	public byte[] getAllIndexes() throws DMLRuntimeException {
		// TODO Auto-generated method stub
		return null;
	}

	//@Override
	public void processInstruction(ExecutionContext ec)
	{
		if( isBPInstructionEnabled()) {
			DebugState dbState = ec.getDebugState();
			
			System.out.format("Breakpoint reached at %s.\n", dbState.getPC().toString());					
			dbState.suspend = true;
		}
	}
	
	public String toString()
	{
		StringBuffer sb = new StringBuffer();
		sb.append("BP");
		sb.append(" ");
		if (bpStatus == BPINSTRUCTION_STATUS.ENABLED)
			sb.append("enabled");
		else if (bpStatus == BPINSTRUCTION_STATUS.DISABLED)  
			sb.append("disabled");
		else
			sb.append("invisible");
		return sb.toString();
	}
}
