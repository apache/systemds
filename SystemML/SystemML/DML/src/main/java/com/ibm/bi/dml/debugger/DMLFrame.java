/**
 * IBM Confidential
 * OCO Source Materials
 * (C) Copyright IBM Corp. 2010, 2014
 * The source code for this program is not published or otherwise divested of its trade secrets, irrespective of what has been deposited with the U.S. Copyright Office.
 */

package com.ibm.bi.dml.debugger;

import com.ibm.bi.dml.runtime.controlprogram.LocalVariableMap;

public class DMLFrame {
	@SuppressWarnings("unused")
	private static final String _COPYRIGHT = "Licensed Materials - Property of IBM\n(C) Copyright IBM Corp. 2010, 2013\n" +
                                             "US Government Users Restricted Rights - Use, duplication  disclosure restricted by GSA ADP Schedule Contract with IBM Corp.";
	
	private LocalVariableMap variables; //Symbol table of frame variables
	private DMLProgramCounter pc; //Program counter of frame upon return
	
	/**
	 * Constructor for a DML frame
	 * @param vars Current frame variables (local symbol table)
	 * @param pc  Current frame program counter 
	 */
	public DMLFrame(LocalVariableMap vars, DMLProgramCounter pc) {
		variables = vars;
		this.pc = pc;
	}

	/**
	 * Getter for variables field
	 * @return Frame's symbol table
	 */
	public LocalVariableMap getVariables() {
		return variables;
	}
	
	/**
	 * Getter program counter field
	 * @return Frame's program counter
	 */
	public DMLProgramCounter getPC() {
		return pc;
	}
	
	/**
	 * Setter for variables field
	 * @param vars Frame's local variables
	 */
	public void setVariables(LocalVariableMap vars) {
		this.variables = vars;
	}
	
	/**
	 * Setter for program counter field
	 * @param currPC Frame's program counter
	 */
	public void setPC(DMLProgramCounter currPC) {
		this.pc = currPC;
	}
}
