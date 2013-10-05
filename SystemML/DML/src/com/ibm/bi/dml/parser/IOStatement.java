/**
 * IBM Confidential
 * OCO Source Materials
 * (C) Copyright IBM Corp. 2010, 2013
 * The source code for this program is not published or otherwise divested of its trade secrets, irrespective of what has been deposited with the U.S. Copyright Office.
 */

package com.ibm.bi.dml.parser;

import com.ibm.bi.dml.parser.Expression.DataOp;


public abstract class IOStatement extends Statement
{
	@SuppressWarnings("unused")
	private static final String _COPYRIGHT = "Licensed Materials - Property of IBM\n(C) Copyright IBM Corp. 2010, 2013\n" +
                                             "US Government Users Restricted Rights - Use, duplication  disclosure restricted by GSA ADP Schedule Contract with IBM Corp.";
		
	protected DataIdentifier _id;
		
	// data structures to store parameters as expressions

	protected DataExpression _paramsExpr;
	
	public IOStatement(){
		_id = null;
		_paramsExpr = new DataExpression();
	
	}
	
	IOStatement(DataIdentifier t, DataOp op){
		_id = t;
		_paramsExpr = new DataExpression(op);
	
	}
	
	
	public IOStatement (DataOp op){
		_id  = null;
		_paramsExpr = new DataExpression(op);
		
	}
	 
	public DataIdentifier getId(){
		return _id;
	}
	
	public void setIdentifier(DataIdentifier t) {
		_id = t;
	}
	
	public void setExprParam(String name, Expression value) {
		_paramsExpr.addVarParam(name, value);
	}
	
	public void setExprParams(DataExpression paramsExpr) {
		_paramsExpr = paramsExpr;
	}
	
	public void addExprParam(String name, Expression value, boolean fromMTDFile) throws ParseException
	{
		if (_paramsExpr.getVarParam(name) != null){
			throw new ParseException(value.printErrorLocation() + "attempted to add IOStatement parameter " + name + " more than once");
		}
		// verify parameter names for InputStatement
		if (this instanceof InputStatement && !InputStatement.isValidParamName(name, fromMTDFile)){
			throw new ParseException(value.printErrorLocation() + "attempted to add invalid read statement parameter " + name);
		}
		else if (this instanceof OutputStatement && !OutputStatement.isValidParamName(name)){
			throw new ParseException(value.printErrorLocation() + "attempted to add invalid write statement parameter: " + name);
		}
		_paramsExpr.addVarParam(name, value);
	}
	
	public Expression getExprParam(String name){
		return _paramsExpr.getVarParam(name);
	}
	
	public DataExpression getSource(){
		return _paramsExpr;
	}

	public DataIdentifier getIdentifier(){
		return _id;
	}
	
	public String getFormatName() {
		return(_paramsExpr.getVarParam(FORMAT_TYPE).toString());
	}
	
	@Override
	public boolean controlStatement() {
		return false;
	}
	
	public void initializeforwardLV(){}
	
	public VariableSet initializebackwardLV(VariableSet lo){
		return lo;
	}

	@Override
	public VariableSet variablesRead() {
		return null;
	}

	@Override
	public VariableSet variablesUpdated() {
		return null;
	}
}
