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

package org.apache.sysml.parser.antlr4;

import java.util.ArrayList;
import java.util.List;

import org.antlr.v4.runtime.Token;

import org.apache.sysml.parser.DMLProgram;
import org.apache.sysml.parser.antlr4.DmlParser.FunctionCallAssignmentStatementContext;
import org.apache.sysml.parser.antlr4.DmlParser.ParameterizedExpressionContext;
import org.apache.sysml.parser.antlr4.DmlSyntacticErrorListener.CustomDmlErrorListener;

public class DmlSyntacticValidatorHelper {
	
	private CustomDmlErrorListener _errorListener = null;
	
	public DmlSyntacticValidatorHelper(CustomDmlErrorListener errorListener) {
		this._errorListener = errorListener;
	}
	
	public void notifyErrorListeners(String message, int line, int charPositionInLine) {
		this._errorListener.validationError(line, charPositionInLine, message);
	}
	
	public void notifyErrorListeners(String message, Token op) {
		this._errorListener.validationError(op.getLine(), op.getCharPositionInLine(), message);
	}
	
	public void raiseWarning(String message, Token op) {
		this._errorListener.validationWarning(op.getLine(), op.getCharPositionInLine(), message);
	}
	
	public String getCurrentFileName() {
		return _errorListener.peekFileName();
	}
	
//	public static void setInfoForArithmeticOp(org.apache.sysml.parser.Expression current, 
//			org.apache.sysml.parser.Expression left, 
//			org.apache.sysml.parser.Expression right, String opStr) {
//		try {
//			// PLUS, MINUS, MULT, DIV, MODULUS, INTDIV, MATMULT, POW, INVALID
//			org.apache.sysml.parser.Expression.BinaryOp bop = org.apache.sysml.parser.Expression.getBinaryOp(opStr);
//			current = new org.apache.sysml.parser.BinaryExpression(bop);
//			((org.apache.sysml.parser.BinaryExpression)current).setLeft(left);
//			((org.apache.sysml.parser.BinaryExpression)current).setRight(right);
//			((org.apache.sysml.parser.BinaryExpression)current).setFilename(DmlSyntacticErrorListener.currentFileName.peek());
//		}
//		catch(Exception e) {
//			System.out.println("In setInfoForArithmeticOp>>");
//			e.printStackTrace();
//		}
//	}
	
//	public static void setInfoForBooleanOp(org.apache.sysml.parser.Expression current, 
//			org.apache.sysml.parser.Expression left, 
//			org.apache.sysml.parser.Expression right, String opStr) {
//		org.apache.sysml.parser.Expression.BooleanOp bop = org.apache.sysml.parser.Expression.getBooleanOp(opStr);
//		current = new org.apache.sysml.parser.BooleanExpression(bop);
//		((org.apache.sysml.parser.BooleanExpression)current).setLeft(left);
//		((org.apache.sysml.parser.BooleanExpression)current).setRight(right);
//		((org.apache.sysml.parser.BooleanExpression)current).setFilename(DmlSyntacticErrorListener.currentFileName.peek());
//	}
	
	public boolean validateBuiltinFunctions(FunctionCallAssignmentStatementContext ctx) {
		String functionName = ctx.name.getText().replaceAll(" ", "").trim();
		if(functionName.compareTo("write") == 0 || functionName.compareTo(DMLProgram.DEFAULT_NAMESPACE + "::write") == 0) {
			return validateBuiltinWriteFunction(ctx);
		}
		return true;
	}
	
	private boolean validateBuiltinWriteFunction(FunctionCallAssignmentStatementContext ctx) {
		
		return true;
	}
	
	public ArrayList<org.apache.sysml.parser.ParameterExpression> getParameterExpressionList(List<ParameterizedExpressionContext> paramExprs) {
		ArrayList<org.apache.sysml.parser.ParameterExpression> retVal = new ArrayList<org.apache.sysml.parser.ParameterExpression>();
		for(ParameterizedExpressionContext ctx : paramExprs) {
			String paramName = null;
			if(ctx.paramName != null && ctx.paramName.getText() != null && !ctx.paramName.getText().isEmpty()) {
				paramName = ctx.paramName.getText();
			}
			org.apache.sysml.parser.ParameterExpression myArg = new org.apache.sysml.parser.ParameterExpression(paramName, ctx.paramVal.info.expr);
			retVal.add(myArg);
		}
		return retVal;
	}
}
