/**
 * (C) Copyright IBM Corp. 2010, 2015
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * 
 */

package org.apache.sysml.runtime.instructions.cp;

import org.apache.sysml.api.DMLScript;
import org.apache.sysml.debug.DebugState;
import org.apache.sysml.runtime.DMLRuntimeException;
import org.apache.sysml.runtime.controlprogram.context.ExecutionContext;
import org.apache.sysml.runtime.instructions.Instruction;


/**
 * Class for breakpoint instructions 
 * Note: ONLY USED FOR DEBUGGING PURPOSES  
 */
public class BreakPointInstruction extends Instruction
{

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
	public void processInstruction(ExecutionContext ec) 
		throws DMLRuntimeException
	{
		if( DMLScript.ENABLE_DEBUG_MODE && isBPInstructionEnabled()) {
			DebugState dbState = ec.getDebugState();
			
			System.out.format("Breakpoint reached at %s.\n", dbState.getPC().toString());					
			dbState.suspend = true;
		}
	}
	
	public String toString()
	{
		StringBuilder sb = new StringBuilder();
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
