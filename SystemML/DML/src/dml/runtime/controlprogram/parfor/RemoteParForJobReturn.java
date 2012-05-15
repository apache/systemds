package dml.runtime.controlprogram.parfor;

import java.util.HashMap;

import dml.runtime.instructions.CPInstructions.Data;

/**
 * Wrapper for job return of ParFor REMOTE for transferring statistics and result symbol table.
 * 
 * @author mboehm
 */
public class RemoteParForJobReturn 
{
	private boolean _successful  = true;
	private int     _numTasks    = -1;
	private int     _numIters    = -1;
	private HashMap<String, Data>[] _variables = null;
		
	public RemoteParForJobReturn( boolean successful, int numTasks, int numIters, HashMap<String, Data>[] variables )
	{
		_successful = successful;
		_numTasks   = numTasks;
		_numIters   = numIters;
		_variables  = variables;
	}
	
	public boolean isSuccessful()
	{
		return _successful;
	}
	
	public int getNumExecutedTasks()
	{
		return _numTasks;
	}
	
	public int getNumExecutedIterations()
	{
		return _numIters;
	}
	
	public HashMap<String,Data>[] getVariables()
	{
		return _variables;
	}
	
	public void setVariables(HashMap<String,Data>[] variables)
	{
		_variables = variables;
	}
	
	
	
}
