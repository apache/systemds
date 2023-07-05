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
import org.apache.sysds.common.Types.DataType;
import org.apache.sysds.common.Types.ValueType;


public class IterablePredicate extends Expression 
{
	private DataIdentifier _iterVar; // variable being iterated over
	private Expression _fromExpr;
	private Expression _toExpr;
	private Expression _incrementExpr;
	private HashMap<String,String> _parforParams;
	
	public IterablePredicate(ParserRuleContext ctx, DataIdentifier iterVar, Expression fromExpr, Expression toExpr,
			Expression incrementExpr, HashMap<String, String> parForParamValues, String filename) {
		_iterVar = iterVar;
		_fromExpr = fromExpr;
		_toExpr = toExpr;
		_incrementExpr = incrementExpr;
		
		_parforParams = parForParamValues;
		setCtxValuesAndFilename(ctx, filename);
	}
	
	@Override
	public VariableSet variablesRead() {
		VariableSet result = new VariableSet();
		result.addVariables(_fromExpr.variablesRead());
		result.addVariables(_toExpr.variablesRead());
		if( _incrementExpr != null )
			result.addVariables( _incrementExpr.variablesRead() );
		return result;
	}
	
	@Override
	public VariableSet variablesUpdated() {
		VariableSet result = new VariableSet();
		result.addVariable(_iterVar.getName(), _iterVar);
		return result;
	}
	
	@Override
	public Expression rewriteExpression(String prefix) {
		//DataIdentifier newIterVar = (DataIdentifier)_iterVar.rewriteExpression(prefix);
		//return new IterablePredicate(newIterVar, _from, _to, _increment);
		throw new LanguageException(this.printErrorLocation() + "rewriteExpression not supported for IterablePredicate");
	}
	
	@Override
	public void validateExpression(HashMap<String, DataIdentifier> ids, HashMap<String, ConstIdentifier> constVars, boolean conditional) 
	{
		//recursive validate
		if (_iterVar instanceof FunctionCallIdentifier
				|| _fromExpr instanceof FunctionCallIdentifier
				||	_toExpr instanceof FunctionCallIdentifier
				||	_incrementExpr instanceof FunctionCallIdentifier){
			raiseValidateError("user-defined function calls not supported for iterable predicates",
				false, LanguageException.LanguageErrorCodes.UNSUPPORTED_EXPRESSION);
		}
		
		//1) VALIDATE ITERATION VARIABLE (index)
		// check the variable has either 1) not been defined already OR 2) defined as integer scalar
		if (ids.containsKey(_iterVar.getName())){
			DataIdentifier otherDI = ids.get(_iterVar.getName());
			if( otherDI.getDataType() != DataType.SCALAR || otherDI.getValueType() != ValueType.INT64 ){
				raiseValidateError("iterable predicate in for loop '" + _iterVar.getName()
					+ "' must be a scalar integer", conditional);
			}
		}
		
		// set the values for DataIdentifer iterable variable
		_iterVar.setIntProperties();
		
		// add the iterVar to the variable set
		ids.put(_iterVar.getName(), _iterVar);
		
		//2) VALIDATE FOR PREDICATE in (from, to, increment)		
		// handle default increment if unspecified
		if (_incrementExpr == null && _fromExpr instanceof ConstIdentifier && _toExpr instanceof ConstIdentifier) {
			ConstIdentifier cFrom = (ConstIdentifier) _fromExpr;
			ConstIdentifier cTo = (ConstIdentifier) _toExpr;
			_incrementExpr = new IntIdentifier((cFrom.getLongValue() <= cTo.getLongValue()) ? 1 : -1, this);
		}
		
		//recursively validate the individual expression
		_fromExpr.validateExpression(ids, constVars, conditional);
		_toExpr.validateExpression(ids, constVars, conditional);
		if( _incrementExpr != null )
			_incrementExpr.validateExpression(ids, constVars, conditional);
		
		//check for scalar expression output
		checkNumericScalarOutput( _fromExpr );
		checkNumericScalarOutput( _toExpr );
		checkNumericScalarOutput( _incrementExpr );
	}
	
	public DataIdentifier getIterVar() {
		return _iterVar;
	}

	public void setIterVar(DataIdentifier iterVar) {
		_iterVar = iterVar;
	}

	public Expression getFromExpr() {
		return _fromExpr;
	}

	public void setFromExpr(Expression from) {
		_fromExpr = from;
	}

	public Expression getToExpr() {
		return _toExpr;
	}

	public void setToExpr(Expression to) {
		_toExpr = to;
	}

	public Expression getIncrementExpr() {
		return _incrementExpr;
	}

	public void setIncrementExpr(Expression increment) {
		_incrementExpr = increment;
	}
	
	public HashMap<String,String> getParForParams(){
		return _parforParams;
	}
	
	public void setParForParams(HashMap<String,String> params){
		_parforParams = params;
	}
	
	@Override
	public String toString()
	{
		StringBuilder sb = new StringBuilder();
		sb.append( "(" );
		sb.append( _iterVar.getName() );
		sb.append(" in seq(");
		sb.append(_fromExpr.toString());
		sb.append(",");
		sb.append(_toExpr.toString());
		if (_incrementExpr != null) {
			sb.append(",");
			sb.append(_incrementExpr.toString());
		}
		sb.append( ")" );
		if (_parforParams != null && _parforParams.size() > 0){
			for (String key : _parforParams.keySet())
			{
				sb.append( "," );
				sb.append( key );
				sb.append( "=" );
				sb.append( _parforParams.get(key) );
			}
		}
		sb.append( ")" );
		return sb.toString();
	}
	
	private void checkNumericScalarOutput( Expression expr ) {
		if( expr == null || expr.getOutput() == null )
			return;
		
		Identifier ident = expr.getOutput();
		if( ident.getDataType() == DataType.MATRIX ||
			(ident.getDataType() == DataType.SCALAR && (ident.getValueType() == ValueType.BITSET ||
			ident.getValueType() == ValueType.STRING)) )
		{
			throw new LanguageException(this.printErrorLocation() + "expression in iterable predicate in for loop '" + expr.toString() + "' must return a numeric scalar");
		}
	}
}
