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

import java.util.TreeMap;

import org.apache.sysml.runtime.instructions.cp.BreakPointInstruction;
import org.apache.sysml.runtime.instructions.cp.BreakPointInstruction.BPINSTRUCTION_STATUS;


/**
 *  Class for managing breakpoints within DML compiler and debugger
 */
public class DMLBreakpointManager {
	
	/** Map between DML script line numbers and breakpoint instructions */
	private static TreeMap<Integer, BreakPointInstruction> breakpoints = new TreeMap<Integer, BreakPointInstruction>();
	
	
	/**
	 * Getter for DML breakpoints
	 * @return List of breakpoints indexed by DML script line number
	 */
	public static TreeMap<Integer, BreakPointInstruction> getBreakpoints() 
	{		
		if (breakpoints.size() > 0)
			return breakpoints;
		return null;
	}

	/**
	 * Returns breakpoint instruction at a particular line number (if any)
	 * @param lineNumber Location of breakpoint
	 * @return Breakpoint instruction at indicated line number (if any)
	 */
	public static BreakPointInstruction getBreakpoint(int lineNumber) {
		if (!breakpoints.containsKey(lineNumber))
			return null;
		return breakpoints.get(lineNumber);
	}

	/**
	 * Insert a breakpoint instruction into list of existing breakpoints.
	 * 
	 * @param breakpoint the breakpoint instruction
	 * @param lineNumber Location for inserting breakpoint
	 */
	public static void insertBreakpoint (BreakPointInstruction breakpoint, int lineNumber) {	
		if (breakpoints.containsKey(lineNumber)) {
			if (breakpoints.get(lineNumber).getBPInstructionStatus() != BPINSTRUCTION_STATUS.INVISIBLE)
				System.out.format("Breakpoint updated at %s, line, %d.\n", breakpoint.getBPInstructionLocation(), lineNumber);
			else 
				System.out.format("Breakpoint added at %s, line %d.\n", breakpoint.getBPInstructionLocation(), lineNumber);
			breakpoints.put(lineNumber, breakpoint);
		}
	}
	
	/**
	 * Insert a breakpoint instruction into list of breakpoints  
	 * @param lineNumber Location for inserting breakpoint
	 */
	public static void insertBreakpoint (int lineNumber) {	
		if (breakpoints.containsKey(lineNumber)) {
			breakpoints.get(lineNumber).setBPInstructionStatus(BPINSTRUCTION_STATUS.INVISIBLE);			
		}
		else {
			breakpoints.put(lineNumber, new BreakPointInstruction(BPINSTRUCTION_STATUS.INVISIBLE));
		}
	}
	
	/**
	 * Updates breakpoint status for a given breakpoint id 
	 * @param lineNumber line number of breakpoint
	 * @param status Current breakpoint status  
	 */
	public static void updateBreakpoint(int lineNumber, BPINSTRUCTION_STATUS status) {
		if (breakpoints.containsKey(lineNumber)) {
			breakpoints.get(lineNumber).setBPInstructionStatus(status);
			System.out.format("Breakpoint updated at %s, line %d.\n", breakpoints.get(lineNumber).getBPInstructionLocation(), lineNumber);
		}
	}

	/**
	 * Removes breakpoint instruction at given line number 
	 * @param lineNumber Location for inserting breakpoint
	 * @param status Current breakpoint status
	 */	
	public static void removeBreakpoint(int lineNumber, BPINSTRUCTION_STATUS status) {
		if (breakpoints.containsKey(lineNumber)) {			
			breakpoints.get(lineNumber).setBPInstructionStatus(status);
			System.out.format("Breakpoint deleted at %s, line %d.\n", breakpoints.get(lineNumber).getBPInstructionLocation(), lineNumber);
		}
	}
}
