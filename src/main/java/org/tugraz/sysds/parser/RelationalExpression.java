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

import org.tugraz.sysds.common.Types.DataType;
import org.tugraz.sysds.common.Types.ValueType;
import org.tugraz.sysds.parser.LanguageException.LanguageErrorCodes;
import org.tugraz.sysds.runtime.meta.MatrixCharacteristics;

import java.util.HashMap;

public class RelationalExpression extends Expression
{
	private Expression _left;
	private Expression _right;
	private RelationalOp _opcode;
	
	public RelationalExpression(RelationalOp bop) {
		_opcode = bop;
		
		setFilename("MAIN SCRIPT");
		setBeginLine(0);
		setBeginColumn(0);
		setEndLine(0);
		setEndColumn(0);
		setText(null);
	}

	public RelationalExpression(RelationalOp bop, ParseInfo parseInfo) {
		_opcode = bop;
		setParseInfo(parseInfo);
	}

	@Override
	public Expression rewriteExpression(String prefix) {
		RelationalExpression newExpr = new RelationalExpression(this._opcode, this);
		newExpr.setLeft(_left.rewriteExpression(prefix));
		newExpr.setRight(_right.rewriteExpression(prefix));
		return newExpr;
	}

	public RelationalOp getOpCode(){
		return _opcode;
	}
	
	public void setLeft(Expression l){
		_left = l;
		
		// update script location information --> left expression is BEFORE in script
		if (_left != null) {
			setParseInfo(_left);
		}

	}
	
	public void setRight(Expression r){
		_right = r;
		
		// update script location information --> right expression is AFTER in script
		if (_right != null) {
			setParseInfo(_right);
		}
	}
	
	public Expression getLeft(){
		return _left;
	}
	
	public Expression getRight(){
		return _right;
	}

	/**
	 * Validate parse tree : Process Relational Expression
	 */
	@Override
	public void validateExpression(HashMap<String,DataIdentifier> ids, HashMap<String, ConstIdentifier> constVars, boolean conditional) 
	{
		//check for functions calls in expression
		if (_left instanceof FunctionCallIdentifier){
			raiseValidateError("user-defined function calls not supported in relational expressions", 
				false, LanguageException.LanguageErrorCodes.UNSUPPORTED_EXPRESSION);
		}
		if (_right instanceof FunctionCallIdentifier){
			raiseValidateError("user-defined function calls not supported in relational expressions", 
				false, LanguageException.LanguageErrorCodes.UNSUPPORTED_EXPRESSION);
		}
		
		// handle <NUMERIC> == <BOOLEAN> --> convert <BOOLEAN> to numeric value
		if ((_left != null && _left instanceof BooleanIdentifier)
				|| (_right != null && _right instanceof BooleanIdentifier)) {
			if ((_left instanceof IntIdentifier || _left instanceof DoubleIdentifier) || _right instanceof IntIdentifier
					|| _right instanceof DoubleIdentifier) {
				if (_left instanceof BooleanIdentifier) {
					if (((BooleanIdentifier) _left).getValue())
						this.setLeft(new IntIdentifier(1, _left));
					else
						this.setLeft(new IntIdentifier(0, _left));
				} else if (_right instanceof BooleanIdentifier) {
					if (((BooleanIdentifier) _right).getValue())
						this.setRight(new IntIdentifier(1, _right));
					else
						this.setRight(new IntIdentifier(0, _right));
				}
			}
		}

		//recursive validate
		_left.validateExpression(ids, constVars, conditional);
		if( _right !=null )
			_right.validateExpression(ids, constVars, conditional);
		
		//constant propagation (precondition for more complex constant folding rewrite)
		if( !conditional ) {
			if( _left instanceof DataIdentifier && constVars.containsKey(((DataIdentifier) _left).getName()) )
				_left = constVars.get(((DataIdentifier) _left).getName());
			if( _right instanceof DataIdentifier && constVars.containsKey(((DataIdentifier) _right).getName()) )
				_right = constVars.get(((DataIdentifier) _right).getName());
		}
		
		String outputName = getTempName();
		DataIdentifier output = new DataIdentifier(outputName);
		output.setParseInfo(this);
		
		boolean isLeftMatrix = (_left.getOutput() != null && _left.getOutput().getDataType() == DataType.MATRIX);
		boolean isRightMatrix = (_right.getOutput() != null && _right.getOutput().getDataType() == DataType.MATRIX); 
		if(isLeftMatrix || isRightMatrix) {
			// Added to support matrix relational comparison
			if(isLeftMatrix && isRightMatrix) {
				checkMatchingDimensions(_left, _right, true);
			}
			MatrixCharacteristics dims = getBinaryMatrixCharacteristics(_left, _right);
			output.setDataType(DataType.MATRIX);
			output.setDimensions(dims.getRows(), dims.getCols());
			output.setBlocksize(dims.getBlocksize());
			
			//since SystemDS only supports double matrices, the value type is forced to
			//double; once we support boolean matrices this needs to change
			output.setValueType(ValueType.FP64);
		}
		else {
			output.setBooleanProperties();
		}
		
		this.setOutput(output);
	}
	
	/**
	 * This is same as the function from BuiltinFunctionExpression which is called by ppred
	 * 
	 * @param expr1 expression 1
	 * @param expr2 expression 2
	 * @param allowsMV ?
	 */
	private void checkMatchingDimensions(Expression expr1, Expression expr2, boolean allowsMV) 
	{
		if (expr1 != null && expr2 != null) {
			
			// if any matrix has unknown dimensions, simply return
			if(  expr1.getOutput().getDim1() == -1 || expr2.getOutput().getDim1() == -1 
			   ||expr1.getOutput().getDim2() == -1 || expr2.getOutput().getDim2() == -1 ) 
			{
				return;
			}
			else if( (!allowsMV && expr1.getOutput().getDim1() != expr2.getOutput().getDim1())
				  || (allowsMV && expr1.getOutput().getDim1() != expr2.getOutput().getDim1() && expr2.getOutput().getDim1() != 1)
				  || (!allowsMV && expr1.getOutput().getDim2() != expr2.getOutput().getDim2()) 
				  || (allowsMV && expr1.getOutput().getDim2() != expr2.getOutput().getDim2() && expr2.getOutput().getDim2() != 1) ) 
			{
				raiseValidateError("Mismatch in matrix dimensions of parameters for function "
						+ this.getOpCode(), false, LanguageErrorCodes.INVALID_PARAMETERS);
			}
		}
	}

	@Override
	public String toString(){
		String leftString;
		String rightString;
		if (_left instanceof StringIdentifier) {
			leftString = "\"" + _left.toString() + "\"";
		} else {
			leftString = _left.toString();
		}
		if (_right instanceof StringIdentifier) {
			rightString = "\"" + _right.toString() + "\"";
		} else {
			rightString = _right.toString();
		}
		return "(" + leftString + " " + _opcode.toString() + " "
				+ rightString + ")";
	}

	@Override
	public VariableSet variablesRead() {
		VariableSet result = new VariableSet();
		result.addVariables(_left.variablesRead());
		result.addVariables(_right.variablesRead());
		return result;
	}

	@Override
	public VariableSet variablesUpdated() {
		VariableSet result = new VariableSet();
		result.addVariables(_left.variablesUpdated());
		result.addVariables(_right.variablesUpdated());
		return result;
	}
}
