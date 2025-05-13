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

package org.apache.sysds.parser;

import java.util.Arrays;
import java.util.HashMap;
import java.util.Set;

public class VariableSet 
{
	private HashMap<String,DataIdentifier> _variables;
	
	public VariableSet() {
		_variables = new HashMap<>();
	}
	
	public VariableSet( VariableSet vs ) {
		_variables = new HashMap<>();
		if (vs != null)
			_variables.putAll(vs.getVariables());
	}
	
	public void addVariable(String name, DataIdentifier id) {
		if( name != null ) // for robustness
			_variables.put(name,id);
	}
	
	public void addVariables(VariableSet vs) {
		if (vs != null)
			_variables.putAll(vs.getVariables());
	}
	
	public void removeVariables(VariableSet vs) {
		if( vs != null )
			for( String var : vs.getVariables().keySet() )
				_variables.remove(var);
	}

	public boolean containsVariable(String name){
		return _variables.containsKey(name);
	}
	
	public boolean containsAnyName(Set<String> names){
		return _variables.keySet().stream()
			.anyMatch(n -> names.contains(n));
	}
	
	public DataIdentifier getVariable(String name){
		return _variables.get(name);
	}
	
	public Set<String> getVariableNames(){
		return _variables.keySet();
	}
	
	public int getSize(){
		return _variables.size();
	}
	
	public HashMap<String,DataIdentifier> getVariables(){
		return _variables;
	}
	
	public boolean isMatrix(String name) {
		return _variables.containsKey(name)
			&& _variables.get(name).getDataType().isMatrix();
	}
	
	@Override
	public String toString() {
		return Arrays.toString(
			_variables.keySet().toArray());
	}
	
	public static VariableSet union(VariableSet vs1, VariableSet vs2) {
		VariableSet ret = new VariableSet(vs1);
		ret.addVariables(vs2);
		return ret;
	}
	
	public static VariableSet minus(VariableSet vs1, VariableSet vs2) {
		VariableSet ret = new VariableSet(vs1);
		ret.removeVariables(vs2);
		return ret;
	}
}
