/*
 * Modifications Copyright 2019 Graz University of Technology
 *
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
 
package org.tugraz.sysds.parser;

import java.util.HashMap;

import org.tugraz.sysds.common.Types.DataType;
import org.tugraz.sysds.common.Types.ValueType;


public class BooleanExpression extends Expression
{
		
	private Expression _left;
	private Expression _right;
	private BooleanOp _opcode;
	
	public BooleanExpression(BooleanOp bop){
		_opcode = bop;
		
		setFilename("MAIN SCRIPT");
		setBeginLine(0);
		setBeginColumn(0);
		setEndLine(0);
		setEndColumn(0);
		setText(null);
	}

	public BooleanExpression(BooleanOp bop, ParseInfo parseInfo) {
		_opcode = bop;
		setParseInfo(parseInfo);
	}

	public BooleanOp getOpCode(){
		return _opcode;
	}
	
	public void setLeft(Expression l){
		_left = l;
		
		// update script location information --> left expression is BEFORE in script
		if (_left != null){
			this.setParseInfo(_left);
		}
	}
	
	public void setRight(Expression r){
		_right = r;
		
		// update script location information --> right expression is AFTER in script
		if (_right != null){
			this.setParseInfo(_right);
		}
	}
	
	public Expression getLeft(){
		return _left;
	}
	
	public Expression getRight(){
		return _right;
	}

	@Override
	public Expression rewriteExpression(String prefix) {
		BooleanExpression newExpr = new BooleanExpression(this._opcode, this);
		newExpr.setLeft(_left.rewriteExpression(prefix));
		if (_right != null)
			newExpr.setRight(_right.rewriteExpression(prefix));
		return newExpr;
	}
	
	/**
	 * Validate parse tree : Process Boolean Expression  
	 */
	@Override
	public void validateExpression(HashMap<String,DataIdentifier> ids, HashMap<String, ConstIdentifier> constVars, boolean conditional) {
		//recursive validate
		getLeft().validateExpression(ids, constVars, conditional);
		if (_left instanceof FunctionCallIdentifier){
			raiseValidateError("user-defined function calls not supported in boolean expressions", 
				false, LanguageException.LanguageErrorCodes.UNSUPPORTED_EXPRESSION);
		}
		if (this.getRight() != null) {
			if (_right instanceof FunctionCallIdentifier){
				raiseValidateError("user-defined function calls not supported in boolean expressions", 
					false, LanguageException.LanguageErrorCodes.UNSUPPORTED_EXPRESSION);
			}
			this.getRight().validateExpression(ids, constVars, conditional);
		}
		
		String outputName = getTempName();
		DataIdentifier output = new DataIdentifier(outputName);
		output.setParseInfo(this);
		if( getLeft().getOutput().getDataType().isMatrix() 
			|| (getRight()!=null && getRight().getOutput().getDataType().isMatrix()) ) {
			output.setDataType((getRight()==null) ? DataType.MATRIX :
				computeDataType(this.getLeft(), this.getRight(), true));
			//since SystemDS only supports double matrices, the value type is forced to
			//double; once we support boolean matrices this needs to change
			output.setValueType(ValueType.FP64);
		}
		else {
			output.setBooleanProperties();
		}
		this.setOutput(output);
		
		if ((_opcode == Expression.BooleanOp.CONDITIONALAND) || (_opcode == Expression.BooleanOp.CONDITIONALOR)) {
			// always unconditional (because unsupported operation)
			if (_opcode == Expression.BooleanOp.CONDITIONALAND) {
				raiseValidateError("Conditional AND (&&) not supported.", false);
			} else if (_opcode == Expression.BooleanOp.CONDITIONALOR) {
				raiseValidateError("Conditional OR (||) not supported.", false);
			}
		}
	}

	@Override
	public String toString(){
		if (_opcode == BooleanOp.NOT) {
			return "(" + _opcode.toString() + " " + _left.toString() + ")";
		} else {
			return "(" + _left.toString() + " " + _opcode.toString() + " " + _right.toString() + ")";
		}
	}
	
	@Override
	public VariableSet variablesRead() {
		VariableSet result = new VariableSet();
		result.addVariables(_left.variablesRead());
		if (_right != null){
			result.addVariables(_right.variablesRead());
		}
		return result;
	}

	@Override
	public VariableSet variablesUpdated() {
		VariableSet result = new VariableSet();
		result.addVariables(_left.variablesUpdated());
		if (_right != null){
			result.addVariables(_right.variablesUpdated());
		}
		return result;
	}
}
