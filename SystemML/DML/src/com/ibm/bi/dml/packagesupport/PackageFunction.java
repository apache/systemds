package com.ibm.bi.dml.packagesupport;

import java.io.Serializable;
import java.util.ArrayList;

import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;
import org.nimble.control.DAGQueue;

import com.ibm.bi.dml.runtime.controlprogram.parfor.util.IDSequence;

/**
 * Abstract class that should be extended to implement a package function.
 * 
 * 
 */
public abstract class PackageFunction implements Serializable 
{
	protected static final Log LOG = LogFactory.getLog(PackageFunction.class.getName());
	
	private static final long serialVersionUID = 3274150928865462856L;
	
	private ArrayList<FIO> _function_inputs; // function inputs
	private String _configurationFile; // configuration file parameter that is provided during declaration
	private DAGQueue _dQueue; // DAG queue that can be used to spawn other tasks.
	
	private String _baseDir; // base dir for all created files of that external function
	private IDSequence _seq = null;
	
	/**
	 * Constructor
	 */

	public PackageFunction() 
	{
		_function_inputs = new ArrayList<FIO>();
		_seq = new IDSequence();
	}

	/**
	 * Method to get the number of inputs to this package function.
	 * 
	 * @return
	 */
	public final int getNumFunctionInputs() {
		if (_function_inputs == null)
			throw new PackageRuntimeException("function inputs null");

		return (_function_inputs.size());
	}

	/**
	 * Method to get a specific input to this package function.
	 * 
	 * @param pos
	 * @return
	 */
	public final FIO getFunctionInput(int pos) {
		if (_function_inputs == null || _function_inputs.size() <= pos)
			throw new PackageRuntimeException(
					"function inputs null or size <= pos");

		return (_function_inputs.get(pos));
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
		if (_function_inputs == null)
			throw new PackageRuntimeException("function inputs null");

		if (_function_inputs.size() > numInputs)
			throw new PackageRuntimeException(
					"function inputs size > numInputs -- cannot reduce size");

		while (_function_inputs.size() < numInputs)
			_function_inputs.add(null);

	}

	/**
	 * Method to set a specific input for this package function
	 * 
	 * @param input
	 * @param pos
	 */

	public final void setInput(FIO input, int pos) {
		if (_function_inputs == null || _function_inputs.size() <= pos)
			throw new PackageRuntimeException(
					"function inputs null or size <= pos");

		_function_inputs.set(pos, input);

	}

	/**
	 * Method to set the configuration file for this function.
	 * 
	 * @param fName
	 */

	public final void setConfiguration(String fName) {
		_configurationFile = fName;
	}

	/**
	 * Method to get the configuration file name
	 * 
	 * @return
	 */

	public final String getConfiguration() {
		return _configurationFile;
	}

	/**
	 * Method to return the DAG queue.
	 * 
	 * @return
	 */
	public final DAGQueue getDAGQueue() {
		return _dQueue;
	}

	/**
	 * Method to set the DAG queue.
	 * 
	 * @param q
	 */

	public final void setDAGQueue(DAGQueue q) {
		_dQueue = q;
	}
	
	/**
	 * 
	 * @param dir
	 */
	public void setBaseDir(String dir)
	{
		_baseDir = dir;
	}

	/**
	 * 
	 * @return
	 */
	public String getBaseDir()
	{
		return _baseDir;
	}
	
	public String createOutputFilePathAndName( String fname )
	{
		return _baseDir + fname + _seq.getNextID();
	}
	

	/**
	 * Method that will be executed to perform this function.
	 */
	public abstract void execute();
	
}
