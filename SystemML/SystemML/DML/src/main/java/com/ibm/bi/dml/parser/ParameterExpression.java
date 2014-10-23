package com.ibm.bi.dml.parser;

public class ParameterExpression {
	private Expression 	_expr;
	private String 		_name;

	public ParameterExpression(String name, Expression val){
		_name 		= name;
		_expr 		= val;
	}
	
	public String getName(){
		return _name;
	}
	
	public Expression getExpr(){
		return _expr;
	}
	
	public void setName(String name){
		_name = name;
	}
	
	public void setExpr(Expression expr){
		_expr = expr;
	}
	
	public String toString(){
		String retVal = new String();
		if (_name != null)
			retVal += _name + "=";
		retVal +=_expr;
		
		return retVal;
	}
	
}
