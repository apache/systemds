package com.ibm.bi.dml.runtime.controlprogram.parfor;

import com.ibm.bi.dml.runtime.controlprogram.LocalVariableMap;

/**
 * Wrapper for job return of ParFor REMOTE for transferring statistics and result symbol table.
 * 
 */
public class RemoteParForJobReturn 
{
	private boolean _successful  = true;
	private int     _numTasks    = -1;
	private int     _numIters    = -1;
	private LocalVariableMap [] _variables = null;
		
	public RemoteParForJobReturn( boolean successful, int numTasks, int numIters, LocalVariableMap [] variables )
	{
		_successful = successful;
		_numTasks   = numTasks;
		_numIters   = numIters;
		_variables  = variables;
	}

	/**
	 * 
	 * @return
	 */
	public boolean isSuccessful()
	{
		return _successful;
	}
	
	/**
	 * 
	 * @return
	 */
	public int getNumExecutedTasks()
	{
		return _numTasks;
	}
	
	/**
	 * 
	 * @return
	 */
	public int getNumExecutedIterations()
	{
		return _numIters;
	}
	
	/**
	 * 
	 * @return
	 */
	public LocalVariableMap [] getVariables()
	{
		return _variables;
	}
	
	/**
	 * 
	 * @param variables
	 */
	public void setVariables (LocalVariableMap [] variables)
	{
		_variables = variables;
	}
}
