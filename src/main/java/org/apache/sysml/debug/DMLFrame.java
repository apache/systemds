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

package org.apache.sysml.debug;

import org.apache.sysml.runtime.controlprogram.LocalVariableMap;

public class DMLFrame {
	
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
