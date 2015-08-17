/**
 * IBM Confidential
 * OCO Source Materials
 * (C) Copyright IBM Corp. 2010, 2013
 * The source code for this program is not published or otherwise divested of its trade secrets, irrespective of what has been deposited with the U.S. Copyright Office.
 */

package com.ibm.bi.dml.runtime.controlprogram.parfor;

import com.ibm.bi.dml.runtime.controlprogram.LocalVariableMap;

/**
 * Wrapper for job return of ParFor REMOTE for transferring statistics and result symbol table.
 * 
 */
public class RemoteParForJobReturn 
{
	@SuppressWarnings("unused")
	private static final String _COPYRIGHT = "Licensed Materials - Property of IBM\n(C) Copyright IBM Corp. 2010, 2013\n" +
                                             "US Government Users Restricted Rights - Use, duplication  disclosure restricted by GSA ADP Schedule Contract with IBM Corp.";
	
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
