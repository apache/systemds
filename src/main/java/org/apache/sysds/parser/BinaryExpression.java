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

import org.apache.sysds.common.Types.DataType;
import org.apache.sysds.common.Types.ValueType;
import org.apache.sysds.runtime.privacy.PrivacyPropagator;


public class BinaryExpression extends Expression 
{
	private Expression _left;
	private Expression _right;
	private BinaryOp _opcode;
	
	@Override
	public Expression rewriteExpression(String prefix) {
		BinaryExpression newExpr = new BinaryExpression(this._opcode, this);
		newExpr.setLeft(_left.rewriteExpression(prefix));
		newExpr.setRight(_right.rewriteExpression(prefix));
		return newExpr;
	}
	
	public BinaryExpression(BinaryOp bop) {
		_opcode = bop;
		
		setFilename("MAIN SCRIPT");
		setBeginLine(0);
		setBeginColumn(0);
		setEndLine(0);
		setEndColumn(0);
		setText(null);
	}

	public BinaryExpression(BinaryOp bop, ParseInfo parseInfo) {
		_opcode = bop;
		setParseInfo(parseInfo);
	}

	public BinaryOp getOpCode() {
		return _opcode;
	}

	public void setLeft(Expression l) {
		_left = l;
		
		// update script location information --> left expression is BEFORE in script
		if (_left != null){
			setParseInfo(_left);
		}
		
	}

	public void setRight(Expression r) {
		_right = r;
		
		// update script location information --> right expression is AFTER in script
		if (_right != null){
			setParseInfo(_right);
		}
	}

	public Expression getLeft() {
		return _left;
	}

	public Expression getRight() {
		return _right;
	}

	/**
	 * Validate parse tree : Process Binary Expression in an assignment
	 * statement
	 * 
	 */
	@Override
	public void validateExpression(HashMap<String, DataIdentifier> ids, HashMap<String, ConstIdentifier> constVars, boolean conditional)
	{
		//recursive validate
		if (_left instanceof FunctionCallIdentifier || _right instanceof FunctionCallIdentifier) {
			raiseValidateError("User-defined function calls not supported in binary expressions.", false,
					LanguageException.LanguageErrorCodes.UNSUPPORTED_EXPRESSION);
		}

		_left.validateExpression(ids, constVars, conditional);
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
		output.setDataType(computeDataType(this.getLeft(), this.getRight(), true));
		ValueType resultVT = computeValueType(this.getLeft(), this.getRight(), true);

		// Override the computed value type, if needed
		if (this.getOpCode() == Expression.BinaryOp.POW
				|| this.getOpCode() == Expression.BinaryOp.DIV) {
			resultVT = ValueType.FP64;
		}

		output.setValueType(resultVT);

		checkAndSetDimensions(output, conditional);
		if (getOpCode() == Expression.BinaryOp.MATMULT) {
			if ((getLeft().getOutput().getDataType() != DataType.MATRIX) || (getRight().getOutput().getDataType() != DataType.MATRIX)) {
		// remove exception for now
		//		throw new LanguageException(
		//				"Matrix multiplication not supported for scalars",
		//				LanguageException.LanguageErrorCodes.INVALID_PARAMETERS);
			}
			if (getLeft().getOutput().getDim2() != -1 && getRight().getOutput().getDim1() != -1
				&& getLeft().getOutput().getDim2() != getRight().getOutput().getDim1()) 
			{
				raiseValidateError("invalid dimensions for matrix multiplication (k1="
					+getLeft().getOutput().getDim2()+", k2="+getRight().getOutput().getDim1()+")", 
					conditional, LanguageException.LanguageErrorCodes.INVALID_PARAMETERS);
			}
			output.setDimensions(getLeft().getOutput().getDim1(),
				getRight().getOutput().getDim2());
		}

		// Set privacy of output
		output.setPrivacy(PrivacyPropagator.mergeBinary(
			getLeft().getOutput().getPrivacy(), getRight().getOutput().getPrivacy()));

		this.setOutput(output);
	}

	private void checkAndSetDimensions(DataIdentifier output, boolean conditional) {
		Identifier left = this.getLeft().getOutput();
		Identifier right = this.getRight().getOutput();
		Identifier pivot = null;
		Identifier aux = null;

		if (left.getDataType() == DataType.MATRIX) {
			pivot = left;
			if (right.getDataType() == DataType.MATRIX) {
				aux = right;
			}
		} else if (right.getDataType() == DataType.MATRIX) {
			pivot = right;
		}

		if ((pivot != null) && (aux != null)) {
			// check dimensions binary operations (if dims known)
			if (isSameDimensionBinaryOp(this.getOpCode()) && pivot.dimsKnown() && aux.dimsKnown()) {
				// number of rows must always be equivalent if not row vector
				// number of cols must be equivalent if not col vector
				if ((pivot.getDim1() != aux.getDim1() && aux.getDim1() > 1)
						|| (pivot.getDim2() != aux.getDim2() && aux.getDim2() > 1)) {
					raiseValidateError("Mismatch in dimensions for operation '" + this.getText() + "'. " + pivot
							+ " is " + pivot.getDim1() + "x" + pivot.getDim2() + " and " + aux + " is " + aux.getDim1()
							+ "x" + aux.getDim2() + ".", conditional);
				}
			}
		}

		//set dimension information
		if (pivot != null) {
			output.setDimensions(pivot.getDim1(), pivot.getDim2());
		}
	}
	
	@Override
	public String toString() {
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

	public static boolean isSameDimensionBinaryOp(BinaryOp op) {
		return (op == BinaryOp.PLUS) || (op == BinaryOp.MINUS)
				|| (op == BinaryOp.MULT) || (op == BinaryOp.DIV)
				|| (op == BinaryOp.MODULUS) || (op == BinaryOp.INTDIV)
				|| (op == BinaryOp.POW);
	}
}
