package com.ibm.bi.dml.packagesupport;

import com.ibm.bi.dml.runtime.instructions.Instruction;
import com.ibm.bi.dml.utils.DMLRuntimeException;

/**
 * Class to maintain external function invocation instructions.
 * 
 * @author aghoting
 * 
 */

public class ExternalFunctionInvocationInstruction extends Instruction 
{
	public static final String ELEMENT_DELIM = ":";
	
	protected String className; // name of class that contains the function
	protected String configFile; // optional configuration file parameter
	protected String inputParams; // string representation of input parameters
	protected String outputParams; // string representation of output parameters
	protected String execLocation; // execution location for function

	public ExternalFunctionInvocationInstruction(String className,
			String configFile, String execLocation, String inputParams,
			String outputParams) 
	{
		this.className = className;
		this.configFile = configFile;
		this.inputParams = inputParams;
		this.outputParams = outputParams;
		this.execLocation = execLocation;
	}
	
	public String getClassName() {
		return className;
	}

	public String getConfigFile() {
		return configFile;
	}

	public String getInputParams() {
		return inputParams;
	}

	public String getOutputParams() {
		return outputParams;
	}

	public String getExecLocation() {
		return execLocation;
	}


	public String toString() {
		return className + ELEMENT_DELIM + 
		       configFile + ELEMENT_DELIM + 
		       inputParams + ELEMENT_DELIM + 
		       outputParams + ELEMENT_DELIM + 
		       execLocation;
	}

	@Override
	public byte[] getAllIndexes() throws DMLRuntimeException {

		return null;
	}

	@Override
	public byte[] getInputIndexes() throws DMLRuntimeException {

		return null;
	}

}
