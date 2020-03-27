/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 * 
 *   http://www.apache.org/licenses/LICENSE-2.0
 * 
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

package org.apache.sysds.runtime.controlprogram.parfor;

import org.apache.sysds.runtime.controlprogram.LocalVariableMap;
import org.apache.sysds.runtime.lineage.Lineage;

/**
 * Wrapper for job return of ParFor REMOTE for transferring statistics and result symbol table.
 * 
 */
public class RemoteParForJobReturn 
{
	private final boolean _successful;
	private final int _numTasks;
	private final int _numIters;
	private final LocalVariableMap [] _variables;
	private final Lineage[] _lineages;
	
	public RemoteParForJobReturn( boolean successful, int numTasks,
		int numIters, LocalVariableMap [] variables) 
	{
		this(successful, numIters, numTasks, variables, null);
	}
	
	public RemoteParForJobReturn( boolean successful, int numTasks,
		int numIters, LocalVariableMap [] variables, Lineage [] lineages)
	{
		_successful = successful;
		_numTasks = numTasks;
		_numIters = numIters;
		_variables = variables;
		_lineages = lineages;
	}

	public boolean isSuccessful() {
		return _successful;
	}

	public int getNumExecutedTasks() {
		return _numTasks;
	}

	public int getNumExecutedIterations() {
		return _numIters;
	}

	public LocalVariableMap [] getVariables() {
		return _variables;
	}

	public Lineage [] getLineages() {
		return _lineages;
	}
}
