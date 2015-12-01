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

package org.apache.sysml.parser;

import java.util.HashMap;
import java.util.Set;

public class VariableSet 
{
	
	private HashMap<String,DataIdentifier> _variables;
	
	public VariableSet(){
		_variables = new HashMap<String,DataIdentifier>();
	}
	
	public VariableSet( VariableSet vs )
	{
		_variables = new HashMap<String,DataIdentifier>();
		
		if (vs != null) {
			HashMap<String,DataIdentifier> vars = vs.getVariables();
			_variables.putAll(vars);
		}		
	}
	
	public void addVariable(String name, DataIdentifier id)
	{
		_variables.put(name,id);
	}
	
	public void addVariables(VariableSet vs)
	{
		if (vs != null) {
			HashMap<String,DataIdentifier> vars = vs.getVariables();
			_variables.putAll(vars);
		}
	}
	
	public void removeVariables(VariableSet vs)
	{
		if( vs != null ){
			Set<String> vars = vs.getVariables().keySet();
			for (String var : vars)
				_variables.remove(var);
		}
	}
	
	public void removeVariable(String name)
	{
		_variables.remove(name);
	}
	
	public boolean containsVariable(String name){
		return _variables.containsKey(name);
	}
	
	public DataIdentifier getVariable(String name){
		return _variables.get(name);
	}
	
	public Set<String> getVariableNames(){
		return _variables.keySet();
	}
	
	public HashMap<String,DataIdentifier> getVariables(){
		return _variables;
	}
	
	public String toString(){
		StringBuilder sb = new StringBuilder();
		for (String var : _variables.keySet()){
			sb.append(var + ",");
		}
		return sb.toString();
	}
	
}
