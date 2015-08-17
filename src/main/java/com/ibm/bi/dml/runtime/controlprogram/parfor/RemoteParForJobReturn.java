/**
 * (C) Copyright IBM Corp. 2010, 2015
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * 
 */

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
