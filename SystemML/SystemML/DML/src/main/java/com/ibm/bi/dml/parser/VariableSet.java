/**
 * IBM Confidential
 * OCO Source Materials
 * (C) Copyright IBM Corp. 2010, 2014
 * The source code for this program is not published or otherwise divested of its trade secrets, irrespective of what has been deposited with the U.S. Copyright Office.
 */

package com.ibm.bi.dml.parser;

import java.util.HashMap;
import java.util.Set;

public class VariableSet 
{
	@SuppressWarnings("unused")
	private static final String _COPYRIGHT = "Licensed Materials - Property of IBM\n(C) Copyright IBM Corp. 2010, 2014\n" +
                                             "US Government Users Restricted Rights - Use, duplication  disclosure restricted by GSA ADP Schedule Contract with IBM Corp.";
	
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
		StringBuffer sb = new StringBuffer();
		for (String var : _variables.keySet()){
			sb.append(var + ",");
		}
		return sb.toString();
	}
	
}
