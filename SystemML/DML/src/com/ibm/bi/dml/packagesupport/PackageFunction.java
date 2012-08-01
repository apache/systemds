package com.ibm.bi.dml.packagesupport;

import java.io.Serializable;
import java.util.ArrayList;

import org.nimble.control.DAGQueue;

import com.ibm.bi.dml.api.DMLScript;
import com.ibm.bi.dml.lops.Lops;
import com.ibm.bi.dml.runtime.controlprogram.parfor.util.ConfigurationManager;
import com.ibm.bi.dml.utils.configuration.DMLConfig;

/**
 * Abstract class that should be extended to implement a package function.
 * 
 * 
 * 
 */
public abstract class PackageFunction implements Serializable {

	private static final long serialVersionUID = 3274150928865462856L;

	private static String filePrefix = null;
	
	static
	{
		try
		{
			StringBuffer sb = new StringBuffer();
			DMLConfig conf = ConfigurationManager.getConfig();
			sb.append(conf.getTextValue(DMLConfig.SCRATCH_SPACE));
			sb.append(Lops.FILE_SEPARATOR);
			sb.append(Lops.PROCESS_PREFIX);
			sb.append(DMLScript.getUUID());
			sb.append(Lops.FILE_SEPARATOR);
			sb.append("PackageSupport");
			sb.append(Lops.FILE_SEPARATOR);
			
			filePrefix = sb.toString();
		}
		catch(Exception ex)
		{
			System.out.println("Warning: could not retrieve parameter " + DMLConfig.SCRATCH_SPACE + " from DMLConfig - using ./");
			filePrefix = "PackageSupport/";
		}
	}
	
	// function inputs
	ArrayList<FIO> function_inputs;

	// configuration file parameter that is provided during declaration
	String configurationFile;

	// DAG queue that can be used to spawn other tasks.
	DAGQueue dQueue;

	/**
	 * Constructor
	 */

	public PackageFunction() {
		function_inputs = new ArrayList<FIO>();
	}

	/**
	 * Method to get the number of inputs to this package function.
	 * 
	 * @return
	 */
	public final int getNumFunctionInputs() {
		if (function_inputs == null)
			throw new PackageRuntimeException("function inputs null");

		return (function_inputs.size());
	}

	/**
	 * Method to get a specific input to this package function.
	 * 
	 * @param pos
	 * @return
	 */
	public final FIO getFunctionInput(int pos) {
		if (function_inputs == null || function_inputs.size() <= pos)
			throw new PackageRuntimeException(
					"function inputs null or size <= pos");

		return (function_inputs.get(pos));
	}

	/**
	 * Method to get the number of outputs of this package function. This method
	 * should be implemented in the user's function.
	 * 
	 * @return
	 */

	public abstract int getNumFunctionOutputs();

	/**
	 * Method to get a specific output of this package function. This method
	 * should be implemented in the user's function.
	 * 
	 * @param pos
	 * @return
	 */
	public abstract FIO getFunctionOutput(int pos);

	/**
	 * Method to set the number of inputs for this package function
	 * 
	 * @param numInputs
	 */

	public final void setNumFunctionInputs(int numInputs) {
		if (function_inputs == null)
			throw new PackageRuntimeException("function inputs null");

		if (function_inputs.size() > numInputs)
			throw new PackageRuntimeException(
					"function inputs size > numInputs -- cannot reduce size");

		while (function_inputs.size() < numInputs)
			function_inputs.add(null);

	}

	/**
	 * Method to set a specific input for this package function
	 * 
	 * @param input
	 * @param pos
	 */

	public final void setInput(FIO input, int pos) {
		if (function_inputs == null || function_inputs.size() <= pos)
			throw new PackageRuntimeException(
					"function inputs null or size <= pos");

		function_inputs.set(pos, input);

	}

	/**
	 * Method to set the configuration file for this function.
	 * 
	 * @param fName
	 */

	public final void setConfiguration(String fName) {
		configurationFile = fName;
	}

	/**
	 * Method to get the configuration file name
	 * 
	 * @return
	 */

	public final String getConfiguration() {
		return configurationFile;
	}

	/**
	 * Method to return the DAG queue.
	 * 
	 * @return
	 */
	public final DAGQueue getDAGQueue() {
		return dQueue;
	}

	/**
	 * Method to set the DAG queue.
	 * 
	 * @param q
	 */

	public final void setDAGQueue(DAGQueue q) {
		dQueue = q;
	}
	
	public String getPackageSupportFilePrefix() 
	{
		return filePrefix;
	}

	/**
	 * Method that will be executed to perform this function.
	 */
	public abstract void execute();
	
}
