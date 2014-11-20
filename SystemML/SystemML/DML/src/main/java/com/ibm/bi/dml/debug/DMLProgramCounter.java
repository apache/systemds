
/**
 * IBM Confidential
 * OCO Source Materials
 * (C) Copyright IBM Corp. 2010, 2014
 * The source code for this program is not published or otherwise divested of its trade secrets, irrespective of what has been deposited with the U.S. Copyright Office.
 */

package com.ibm.bi.dml.debug;

import com.ibm.bi.dml.parser.DMLProgram;


public class DMLProgramCounter {
	@SuppressWarnings("unused")
	private static final String _COPYRIGHT = "Licensed Materials - Property of IBM\n(C) Copyright IBM Corp. 2010, 2014\n" +
                                             "US Government Users Restricted Rights - Use, duplication  disclosure restricted by GSA ADP Schedule Contract with IBM Corp.";

	private String namespace; //currently executing namespace 
	private String fname; //currently executing  function name
	private int programBlockNumber; //currently executing program block number within current function
	private int instNumber; //currently executing instruction number within current program block
	
	private long instID; //currently executing instruction  
	private int lineNumber; //currently executing line number (not always correctly set)

	/** 
	 * Constructor for DML pc
	 */
	public DMLProgramCounter() {
		instID = 0;
		lineNumber = 0;
	}

	/**
	 * Basic parameterized constructor for DML pc
	 * @param name Current namespace 
	 * @param fn  Current function name in namespace
	 * @param blockNum Current program block within function
	 * @param instNum Current instruction within program block
	 */
	public DMLProgramCounter(String name, String fn, int blockNum, int instNum) {
		this();
		namespace = name;
		fname = fn;		
		programBlockNumber = blockNum;
		instNumber = instNum;
	}

	/**
	 * Parameterized constructor for DML pc without line number information
	 * @param name Current namespace 
	 * @param fn  Current function name in namespace
	 * @param blockNum Current program block within function
	 * @param instNum Current instruction within program block
	 * @param instID Current instruction unique identifier
	 */
	public DMLProgramCounter(String name, String fn, int blockNum, int instNum, long instID) {
		this(name, fn, blockNum, instNum);
		this.instID = instID; 
	}

	/**
	 * Fully parameterized constructor for DML pc
	 * @param name Current namespace 
	 * @param fn  Current function name in namespace
	 * @param blockNum Current program block within function
	 * @param instNum Current instruction within program block
	 * @param instID Current instruction unique identifier
	 * @param lineNum Current line number
	 */
	public DMLProgramCounter(String name, String fn, int blockNum, int instNum, long instID, int lineNum) {
		this(name, fn, blockNum, instNum, instID);
		lineNumber = lineNum;
	}
	
	/**
	 * Getter for namespace field
	 * @return Current pc's namespace
	 */
	public String getNamespace() {
		return namespace;
	}

	/**
	 * Getter for function name field
	 * @return Current pc's function name
	 */
	public String getFunctionName() {
		return fname;
	}
	
	/**
	 * Getter for program block number field
	 * @return Current pc's program block number
	 */
	public int getProgramBlockNumber() {
		return programBlockNumber;
	}

	/**
	 * Getter for instruction number field
	 * @return Current pc's instruction number
	 */
	public int getInstNumber() {
		return instNumber;
	}

	/**
	 * Getter for instruction unique identifier field
	 * @return Current pc's instruction ID
	 */
	public long getInstID() {
		return instID;
	}

	/**
	 * Getter for line number field
	 * @return Current pc's line number
	 */
	public int getLineNumber() {
		return lineNumber;
	}

	/**
	 * Setter for namespace field
	 * @param name Current pc's namespace
	 */
	public void setNamespace(String name) {
		namespace = name;
	}
	/**
	 * Setter for function name field
	 * @param fn Current pc's function name
	 */
	public void setFunctionName(String fn) {
		fname = fn;
	}

	/**
	 * Setter for program block number field
	 * @param blockNum Current pc's program block number 
	 */
	public void setProgramBlockNumber(int blockNum) {
		programBlockNumber = blockNum;
	}

	/**
	 * Setter for instruction number field
	 * @param instNum Current pc's instruction number
	 */
	public void setInstNumber(int instNum) {
		instNumber = instNum;
	}

	/**
	 * Setter for instruction unique identifier field
	 * @param instID Current pc's instruction unique ID
	 */
	public void setInstID(long instID) {
		this.instID = instID;
	}

	/**
	 * Setter for line number field
	 * @param lineNum Current pc's line number
	 */
	public void setLineNumber(int lineNum) {
		lineNumber = lineNum;
	}

	/**
	 * Displays a pretty-printed program counter
	 * @return Current pc
	 */
	public String toString() {
		StringBuffer sb = new StringBuffer();
		sb.append(DMLProgram.constructFunctionKey(this.namespace, this.fname));
		sb.append(" instID ");
		sb.append(this.instID);
		sb.append(": (line ");
		sb.append(this.lineNumber);
		sb.append(")");
		return sb.toString();
	}
	
	/**
	 * Displays a pretty-printed program counter without instructionID
	 * @return Current pc
	 */
	public String toStringWithoutInstructionID() {
		StringBuffer sb = new StringBuffer();
		sb.append(DMLProgram.constructFunctionKey(this.namespace, this.fname));
		// sb.append(" instID ");
		// sb.append(this.instID);
		sb.append(": (line ");
		sb.append(this.lineNumber);
		sb.append(")");
		return sb.toString();
	}
}
