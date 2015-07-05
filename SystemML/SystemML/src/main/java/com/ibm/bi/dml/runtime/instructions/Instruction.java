/**
 * IBM Confidential
 * OCO Source Materials
 * (C) Copyright IBM Corp. 2010, 2015
 * The source code for this program is not published or otherwise divested of its trade secrets, irrespective of what has been deposited with the U.S. Copyright Office.
 */

package com.ibm.bi.dml.runtime.instructions;

import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;

import com.ibm.bi.dml.lops.Lop;
import com.ibm.bi.dml.runtime.DMLRuntimeException;
import com.ibm.bi.dml.runtime.DMLUnsupportedOperationException;
import com.ibm.bi.dml.runtime.controlprogram.context.ExecutionContext;


public abstract class Instruction 
{
	@SuppressWarnings("unused")
	private static final String _COPYRIGHT = "Licensed Materials - Property of IBM\n(C) Copyright IBM Corp. 2010, 2015\n" +
                                             "US Government Users Restricted Rights - Use, duplication  disclosure restricted by GSA ADP Schedule Contract with IBM Corp.";
	
	public enum INSTRUCTION_TYPE { CONTROL_PROGRAM, MAPREDUCE, EXTERNAL_LIBRARY, MAPREDUCE_JOB, BREAKPOINT, SPARK };
	protected static final Log LOG = LogFactory.getLog(Instruction.class.getName());
	public static final String OPERAND_DELIM = Lop.OPERAND_DELIMITOR;
	public static final String DATATYPE_PREFIX = Lop.DATATYPE_PREFIX;
	public static final String VALUETYPE_PREFIX = Lop.VALUETYPE_PREFIX;
	public static final String LITERAL_PREFIX = Lop.LITERAL_PREFIX;
	public static final String INSTRUCTION_DELIM = Lop.INSTRUCTION_DELIMITOR;
	public static final String NAME_VALUE_SEPARATOR = Lop.NAME_VALUE_SEPARATOR;
	
	protected INSTRUCTION_TYPE type = null;
	protected String instString = null;
	protected String instOpcode = null;
	private int lineNum = -1;
	private long instID = -1;
	
	public void setType (INSTRUCTION_TYPE tp ) {
		type = tp;
	}
	
	public INSTRUCTION_TYPE getType() {
		return type;
	}
	
	/**
	 * Setter for instruction line number 
	 * @param ln Exact (or approximate) DML script line number
	 */
	public void setLineNum ( int ln ) {
		lineNum = ln;
	}
	
	/**
	 * Getter for instruction line number
	 * @return lineNum Instruction approximate DML script line number
	 */
	public int getLineNum() {
		return lineNum;
	}

	/**
	 * Setter for instruction unique identifier 
	 * @param id Instruction unique identifier
	 */
	public void setInstID ( long id ) {
		instID = id;
	}
		
	/**
	 * Getter for instruction unique identifier
	 * @return instID Instruction unique identifier
	 */
	public long getInstID() {
		return instID;
	}

	public void printMe() {
		LOG.debug(instString);
	}
	
	public String toString() {
		return instString;
	}
	
	public String getGraphString() {
		return null;
	}

	public String getOpcode()
	{
		return instOpcode;
	}
	
	/**
	 * 
	 * @return
	 */
	public boolean requiresLabelUpdate()
	{
		return instString.contains( Lop.VARIABLE_NAME_PLACEHOLDER );
	}	
	
	/**
	 * All instructions that have thread-specific filenames or names encoded in it
	 * should overwrite this method in order to update (1) the in-memory instruction
	 * and (2) the instruction string 
	 * 
	 * @param pattern
	 * @param replace
	 * @throws DMLRuntimeException 
	 */
	public void updateInstructionThreadID(String pattern, String replace) 
		throws DMLRuntimeException
	{
		//do nothing
	}
	
	/**
	 * This method should be used for any setup before executing this instruction.
	 * Overwriting methods should first call the super method and subsequently do
	 * their custom setup.
	 * 
	 * @param ec
	 * @return
	 * @throws DMLRuntimeException
	 * @throws DMLUnsupportedOperationException 
	 */
	public Instruction preprocessInstruction(ExecutionContext ec)
		throws DMLRuntimeException, DMLUnsupportedOperationException
	{
		//update debug status
		ec.updateDebugState( this );
		
		//return instruction ifself
		return this;
	}
	
	/**
	 * This method should be used to execute the instruction. 
	 * 
	 * @param ec
	 * @throws DMLRuntimeException
	 * @throws DMLUnsupportedOperationException
	 */
	public abstract void processInstruction(ExecutionContext ec) 
		throws DMLRuntimeException, DMLUnsupportedOperationException;
	
	/**
	 * This method should be used for any tear down after executing this instruction.
	 * Overwriting methods should first do their custom tear down and subsequently 
	 * call the super method.
	 * 
	 * @param ec
	 */
	public void postprocessInstruction(ExecutionContext ec)
		throws DMLRuntimeException
	{
		//do nothing
	}
}
