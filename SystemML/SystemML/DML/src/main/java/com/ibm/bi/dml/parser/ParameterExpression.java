/**
 * IBM Confidential
 * OCO Source Materials
 * (C) Copyright IBM Corp. 2010, 2015
 * The source code for this program is not published or otherwise divested of its trade secrets, irrespective of what has been deposited with the U.S. Copyright Office.
 */

package com.ibm.bi.dml.parser;

public class ParameterExpression 
{
	@SuppressWarnings("unused")
	private static final String _COPYRIGHT = "Licensed Materials - Property of IBM\n(C) Copyright IBM Corp. 2010, 2015\n" +
                                             "US Government Users Restricted Rights - Use, duplication  disclosure restricted by GSA ADP Schedule Contract with IBM Corp.";
	
	
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
	
	@Override
	public String toString(){
		return _name + "=" + _expr;
	}
	
}
