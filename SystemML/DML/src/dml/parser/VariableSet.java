package dml.parser;

import java.util.HashMap;
import java.util.Set;

public class VariableSet {

	HashMap<String,DataIdentifier> _variables;
	
	public VariableSet(){
		_variables = new HashMap<String,DataIdentifier>();
	}
	
	public void addVariable(String name, DataIdentifier id){
		_variables.put(name,id);
	}
	
	public void addVariables(VariableSet vs){
		if (vs == null)
			return;
		HashMap<String,DataIdentifier> vars = vs.getVariables();
		_variables.putAll(vars);
		return;
	}
	
	public void removeVariables(VariableSet vs){
		if (vs == null)
			return;
		Set<String> vars = vs.getVariables().keySet();
		for (String var : vars){
			_variables.remove(var);
		}
		
		return;
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
