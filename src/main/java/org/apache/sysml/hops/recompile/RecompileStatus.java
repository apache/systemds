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

package org.apache.sysml.hops.recompile;

import java.util.ArrayList;
import java.util.HashMap;

import org.apache.sysml.parser.VariableSet;
import org.apache.sysml.runtime.matrix.MatrixCharacteristics;

public class RecompileStatus 
{
	
	private HashMap<String, MatrixCharacteristics> _lastTWrites = null; 
	
	public RecompileStatus()
	{
		_lastTWrites = new HashMap<String,MatrixCharacteristics>();
	}
	
	public HashMap<String, MatrixCharacteristics> getTWriteStats()
	{
		return _lastTWrites;
	}
	
	public void clearStatus()
	{
		_lastTWrites.clear();
	}
	
	public void clearStatus(VariableSet vars)
	{
		ArrayList<String> lvars = new ArrayList<String>(vars.getVariableNames());
		for( String var : lvars ) {
			_lastTWrites.remove(var);
		}
	}
	
	@Override
	public Object clone()
	{
		RecompileStatus ret = new RecompileStatus();
		ret._lastTWrites.putAll(this._lastTWrites);
		return ret;
	}
}
