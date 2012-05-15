package com.ibm.bi.dml.parser;

public class ConditionalPredicate {
	Expression _expr;
	
	public ConditionalPredicate(Expression expr){
		_expr = expr;
	}
	
	public Expression getPredicate(){
		return _expr;
	}
	
	public String toString(){
		return _expr.toString();
	}
	
	 
	public VariableSet variablesRead() {
		VariableSet result = new VariableSet();
		result.addVariables(_expr.variablesRead());
	 	return result;
	}

	 
	public VariableSet variablesUpdated() {
		VariableSet result = new VariableSet();
		result.addVariables(_expr.variablesUpdated());
	 	return result;
	}
	
}
