package com.ibm.bi.dml.packagesupport;

import com.ibm.bi.dml.runtime.instructions.Instruction;
import com.ibm.bi.dml.utils.DMLRuntimeException;


/**
 * Class to maintain external function invokation instructions.
 * 
 * @author aghoting
 * 
 */

public class ExternalFunctionInvokationInstruction extends Instruction {

	String className; // name of class that contains the function
	String configFile; // optional configuration file parameter
	String inputParams; // string representation of input parameters
	String outputParams; // string representation of output parameters
	String execLocation; // execution location for function

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

	public ExternalFunctionInvokationInstruction(String className,
			String configFile, String execLocation, String inputParams,
			String outputParams) {
		this.className = className;
		this.configFile = configFile;
		this.inputParams = inputParams;
		this.outputParams = outputParams;
		this.execLocation = execLocation;
	}

	public String toString() {
		return className + ":" + configFile + ":" + inputParams + ":"
				+ outputParams + ":" + execLocation;
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
