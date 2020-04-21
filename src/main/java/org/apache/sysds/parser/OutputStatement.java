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

import java.util.HashMap;

import org.antlr.v4.runtime.ParserRuleContext;
import org.apache.sysds.parser.Expression.DataOp;

public class OutputStatement extends Statement
{
		
	private DataIdentifier _id;
	private DataExpression _paramsExpr;
	
	public static final String[] WRITE_VALID_PARAM_NAMES = { 	DataExpression.IO_FILENAME, 
																DataExpression.FORMAT_TYPE, 
																DataExpression.DELIM_DELIMITER, 
																DataExpression.DELIM_HAS_HEADER_ROW, 
																DataExpression.DELIM_SPARSE,
																DataExpression.DESCRIPTIONPARAM};

	public DataIdentifier getIdentifier(){
		return _id;
	}
	
	public DataExpression getSource(){
		return _paramsExpr;
	}
	
	public void setIdentifier(DataIdentifier t) {
		_id = t;
	}

	public OutputStatement(ParserRuleContext ctx, DataIdentifier t, DataOp op, String filename) {
		_id = t;
		_paramsExpr = new DataExpression(ctx, op, new HashMap<String, Expression>(), filename);
		setCtxValuesAndFilename(ctx, filename);
	}

	public OutputStatement(DataIdentifier t, DataOp op, ParseInfo parseInfo) {
		_id = t;
		_paramsExpr = new DataExpression(op, new HashMap<String, Expression>(), parseInfo);
		setParseInfo(parseInfo);
	}

	public static boolean isValidParamName(String key){
		for (String paramName : WRITE_VALID_PARAM_NAMES)
			if (paramName.equals(key))
				return true;
			return false;
	}
	
	public void addExprParam(String name, Expression value, boolean fromMTDFile) {
		if( _paramsExpr.getVarParam(name) != null )
			raiseValidateError("attempted to add IOStatement parameter " + name + " more than once", false);
		if( !OutputStatement.isValidParamName(name) )
			raiseValidateError("attempted to add invalid write statement parameter: " + name, false);
		_paramsExpr.addVarParam(name, value);
	}
	
	// rewrites statement to support function inlining (create deep copy)
	@Override
	public Statement rewriteStatement(String prefix) {
		OutputStatement newStatement = new OutputStatement(null, Expression.DataOp.WRITE, this);
		// rewrite outputStatement variable name (creates deep copy)
		newStatement._id = (DataIdentifier)this._id.rewriteExpression(prefix);
		// rewrite parameter expressions (creates deep copy)
		DataOp op = _paramsExpr.getOpCode();
		HashMap<String,Expression> newExprParams = new HashMap<>();
		for (String key : _paramsExpr.getVarParams().keySet()){
			Expression newExpr = _paramsExpr.getVarParam(key).rewriteExpression(prefix);
			newExprParams.put(key, newExpr);
		}
		DataExpression newParamerizedExpr = new DataExpression(op, newExprParams, this);
		newStatement.setExprParams(newParamerizedExpr);
		return newStatement;
	}

	public void setExprParams(DataExpression newParamerizedExpr) {
		_paramsExpr = newParamerizedExpr;
	}

	public Expression getExprParam(String key){
		return _paramsExpr.getVarParam(key);
	}
	
	@Override
	public void initializeforwardLV(VariableSet activeIn){}
	@Override
	public VariableSet initializebackwardLV(VariableSet lo){return lo;}

	@Override
	public String toString() {
		StringBuilder sb = new StringBuilder();
		sb.append(Statement.OUTPUTSTATEMENT + "(");
		sb.append("id=" + _id.toString());
		for (String key : _paramsExpr.getVarParams().keySet()) {
			sb.append(", ");
			sb.append(key);
			sb.append("=");
			Expression exp = _paramsExpr.getVarParam(key);
			if (exp instanceof StringIdentifier) {
				sb.append("\"");
				sb.append(exp.toString());
				sb.append("\"");
			} else {
				sb.append(exp.toString());
			}
		}
		sb.append(");");
		return sb.toString();
	}

	@Override
	public VariableSet variablesRead() {
		VariableSet result = new VariableSet();
		
		// handle variable that is being written out
		result.addVariables(_id.variablesRead());
		
		// handle variables for output filename expression
		//result.addVariables(_filenameExpr.variablesRead());
		
		// add variables for parameter expressions 
		for (String key : _paramsExpr.getVarParams().keySet())
			result.addVariables(_paramsExpr.getVarParam(key).variablesRead()) ;
		
		return result;
	}

	@Override
	public VariableSet variablesUpdated() {
	  	return null;
	}
	
	@Override
	public boolean controlStatement() {
		Expression fmt = _paramsExpr.getVarParam(DataExpression.FORMAT_TYPE);
		return (fmt != null && fmt.toString().equalsIgnoreCase("csv"));
	}
}
