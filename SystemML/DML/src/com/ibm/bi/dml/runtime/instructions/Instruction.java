/**
 * IBM Confidential
 * OCO Source Materials
 * (C) Copyright IBM Corp. 2010, 2013
 * The source code for this program is not published or otherwise divested of its trade secrets, irrespective of what has been deposited with the U.S. Copyright Office.
 */

package com.ibm.bi.dml.runtime.instructions;

import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;

import com.ibm.bi.dml.lops.Lop;
import com.ibm.bi.dml.runtime.DMLRuntimeException;
import com.ibm.bi.dml.runtime.DMLUnsupportedOperationException;


public abstract class Instruction 
{
	@SuppressWarnings("unused")
	private static final String _COPYRIGHT = "Licensed Materials - Property of IBM\n(C) Copyright IBM Corp. 2010, 2013\n" +
                                             "US Government Users Restricted Rights - Use, duplication  disclosure restricted by GSA ADP Schedule Contract with IBM Corp.";
	
	public enum INSTRUCTION_TYPE { CONTROL_PROGRAM, MAPREDUCE, EXTERNAL_LIBRARY, MAPREDUCE_JOB };
	protected static final Log LOG = LogFactory.getLog(Instruction.class.getName());
	public static final String OPERAND_DELIM = Lop.OPERAND_DELIMITOR;
	public static final String DATATYPE_PREFIX = Lop.DATATYPE_PREFIX;
	public static final String VALUETYPE_PREFIX = Lop.VALUETYPE_PREFIX;
	public static final String INSTRUCTION_DELIM = Lop.INSTRUCTION_DELIMITOR;
	public static final String NAME_VALUE_SEPARATOR = Lop.NAME_VALUE_SEPARATOR;
	
	protected INSTRUCTION_TYPE type;
	protected String           instString;
	
	public void setType (INSTRUCTION_TYPE tp ) {
		type = tp;
	}
	
	public INSTRUCTION_TYPE getType() {
		return type;
	}
	
	protected static Instruction parseInstruction ( String str ) throws DMLRuntimeException, DMLUnsupportedOperationException{
		throw new DMLRuntimeException("parseInstruction(): should not be invoked from the base class.");
	}

	public abstract byte[] getInputIndexes() throws DMLRuntimeException;
	
	public abstract byte[] getAllIndexes() throws DMLRuntimeException;
	
	public void printMe() {
		LOG.debug(instString);
	}
	public String toString() {
		return instString;
	}
	
	public String getGraphString() {
		return null;
	}

	/**
	 * 
	 * @return
	 */
	public boolean requiresLabelUpdate()
	{
		return toString().contains( Lop.VARIABLE_NAME_PLACEHOLDER );
	}	
}
