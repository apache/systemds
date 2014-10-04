/**
 * IBM Confidential
 * OCO Source Materials
 * (C) Copyright IBM Corp. 2010, 2014
 * The source code for this program is not published or otherwise divested of its trade secrets, irrespective of what has been deposited with the U.S. Copyright Office.
 */

package com.ibm.bi.dml.debugger;

import java.util.TreeMap;

import com.ibm.bi.dml.runtime.instructions.CPInstructions.BreakPointInstruction;
import com.ibm.bi.dml.runtime.instructions.CPInstructions.BreakPointInstruction.BPINSTRUCTION_STATUS;


/**
 *  Class for managing breakpoints within DML compiler and debugger
 */
public class DMLBreakpointManager {
	@SuppressWarnings("unused")
	private static final String _COPYRIGHT = "Licensed Materials - Property of IBM\n(C) Copyright IBM Corp. 2010, 2014\n" +
                                             "US Government Users Restricted Rights - Use, duplication  disclosure restricted by GSA ADP Schedule Contract with IBM Corp.";
	
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
	 * Returns size of active DML breakpoints
	 * @return size Size of active breakpoints
	 */
	public static int getBreakpointsSize() 
	{	
		int size = 0;
		for (Integer lineNumber : breakpoints.keySet()) {
			if (breakpoints.get(lineNumber).getBPInstructionStatus() != BPINSTRUCTION_STATUS.INVISIBLE)
				size++;
		}
		return size;
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
	 * Returns breakpoint instruction with given breakpoint id
	 * @param location Breakpoint id
	 * @return Breakpoint instruction at indicated id
	 */
	public static BreakPointInstruction getBreakpointAtIndex(int location) {
		int index = 1;
		for (Integer lineNumber : breakpoints.keySet()) {
			if (index++ == location) {
				return breakpoints.get(lineNumber);
			}
		}
		return null;
	}
	
	/**
	 * Returns breakpoint line number with given breakpoint id 
	 * @param location Breakpoint id
	 * @return Breakpoint instruction line number (-1 if not found)   
	 */
	public static int getBreakpointLineNumber(int location) {
		int index = 1;
		for (Integer lineNumber : breakpoints.keySet()) {
			if (index++ == location) {
				return lineNumber;
			}
		}
		return -1;
	}
	
	/**
	 * Returns breakpoint identifier with given line number 
	 * @param Line number Location of breakpoint in DML script
	 * @return bpID Breakpoint id within all breakpoints (-1 if not found)
	 */
	public static int getBreakpointID(int lineNum) {
		int bpID=1;
		for (Integer lineNumber : breakpoints.keySet()) {
			if (lineNum == lineNumber) {
				return bpID;
			}
			bpID++;
		}
		return -1;
	}
	
	/**
	 * Insert a breakpoint instruction into list of existing breakpoints  
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
	 * @param location Breakpoint identifier
	 * @param status Current breakpoint status  
	 */
	public static void updateBreakpoint(int lineNumber, BPINSTRUCTION_STATUS status) {
		if (breakpoints.containsKey(lineNumber)) {
			breakpoints.get(lineNumber).setBPInstructionStatus(status);
			System.out.format("Breakpoint updated at %s, line %d.\n", breakpoints.get(lineNumber).getBPInstructionLocation(), lineNumber);
		}
	}
	
	/**
	 * Updates breakpoint status for a given breakpoint id 
	 * @param location Breakpoint identifier
	 * @param status Current breakpoint status  
	 */
	public static void updateBreakpointID(int location, BPINSTRUCTION_STATUS status) {
		int lineNumber = getBreakpointLineNumber(location);
		if (lineNumber != -1) {			
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

	/**
	 * Removes breakpoint instruction at given location  
	 * @param location Breakpoint instruction id
	 * @param status Current breakpoint status
	 */	
	public static void removeBreakpointIndex(int location, BPINSTRUCTION_STATUS status) {
		int lineNumber = getBreakpointLineNumber(location);
		if (lineNumber != -1)
			breakpoints.get(lineNumber).setBPInstructionStatus(status);
	}
}
