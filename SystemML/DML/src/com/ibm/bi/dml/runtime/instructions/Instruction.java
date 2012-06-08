package com.ibm.bi.dml.runtime.instructions;

import com.ibm.bi.dml.lops.Lops;
import com.ibm.bi.dml.utils.DMLRuntimeException;
import com.ibm.bi.dml.utils.DMLUnsupportedOperationException;


public abstract class Instruction 
{
	public enum INSTRUCTION_TYPE { CONTROL_PROGRAM, MAPREDUCE, EXTERNAL_LIBRARY, MAPREDUCE_JOB };
	
	public static final String OPERAND_DELIM = Lops.OPERAND_DELIMITOR;
	public static final String DATATYPE_PREFIX = Lops.DATATYPE_PREFIX;
	public static final String VALUETYPE_PREFIX = Lops.VALUETYPE_PREFIX;
	public static final String INSTRUCTION_DELIM = Lops.INSTRUCTION_DELIMITOR;
	public static final String NAME_VALUE_SEPARATOR = Lops.NAME_VALUE_SEPARATOR;
	
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
		System.out.println(instString);
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
		return toString().contains( Lops.VARIABLE_NAME_PLACEHOLDER );
	}	
}
