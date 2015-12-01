/**
 * (C) Copyright IBM Corp. 2010, 2015
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * 
 */

package com.ibm.bi.dml.parser.python;

import java.util.ArrayList;
import java.util.List;

import org.antlr.v4.runtime.Token;

import com.ibm.bi.dml.parser.DMLProgram;
import com.ibm.bi.dml.parser.python.PydmlParser.FunctionCallAssignmentStatementContext;
import com.ibm.bi.dml.parser.python.PydmlParser.ParameterizedExpressionContext;
import com.ibm.bi.dml.parser.python.PydmlSyntacticErrorListener.CustomDmlErrorListener;


public class PydmlSyntacticValidatorHelper {
	
	private CustomDmlErrorListener _errorListener = null;
	public PydmlSyntacticValidatorHelper(CustomDmlErrorListener errorListener) {
		this._errorListener = errorListener;
	}
	public void notifyErrorListeners(String message, int line, int charPositionInLine) {
		this._errorListener.validationError(line, charPositionInLine, message);
	}
	
	public void notifyErrorListeners(String message, Token op) {
		this._errorListener.validationError(op.getLine(), op.getCharPositionInLine(), message);
	}
	
	public String getCurrentFileName() {
		return _errorListener.peekFileName();
	}
	
	// Returns list of two elements <namespace, function names>, else null
	public ArrayList<String> getQualifiedNames(String fullyQualifiedFunctionName) {
		String [] fnNames = fullyQualifiedFunctionName.split("\\."); // instead of ::
		String functionName = "";
		String namespace = "";
		if(fnNames.length == 1) {
			namespace = DMLProgram.DEFAULT_NAMESPACE;
			functionName = fnNames[0].trim();
		}
		else if(fnNames.length == 2) {
			namespace = fnNames[0].trim();
			functionName = fnNames[1].trim();
		}
		else
			return null;
		
		ArrayList<String> retVal = new ArrayList<String>();
		retVal.add(namespace);
		retVal.add(functionName);
		return retVal;
	}
	
//	public static void setInfoForArithmeticOp(com.ibm.bi.dml.parser.Expression current, 
//			com.ibm.bi.dml.parser.Expression left, 
//			com.ibm.bi.dml.parser.Expression right, String opStr) {
//		try {
//			// PLUS, MINUS, MULT, DIV, MODULUS, INTDIV, MATMULT, POW, INVALID
//			com.ibm.bi.dml.parser.Expression.BinaryOp bop = com.ibm.bi.dml.parser.Expression.getBinaryOp(opStr);
//			current = new com.ibm.bi.dml.parser.BinaryExpression(bop);
//			((com.ibm.bi.dml.parser.BinaryExpression)current).setLeft(left);
//			((com.ibm.bi.dml.parser.BinaryExpression)current).setRight(right);
//			((com.ibm.bi.dml.parser.BinaryExpression)current).setFilename(DmlSyntacticErrorListener.currentFileName.peek());
//		}
//		catch(Exception e) {
//			System.out.println("In setInfoForArithmeticOp>>");
//			e.printStackTrace();
//		}
//	}
	
//	public static void setInfoForBooleanOp(com.ibm.bi.dml.parser.Expression current, 
//			com.ibm.bi.dml.parser.Expression left, 
//			com.ibm.bi.dml.parser.Expression right, String opStr) {
//		com.ibm.bi.dml.parser.Expression.BooleanOp bop = com.ibm.bi.dml.parser.Expression.getBooleanOp(opStr);
//		current = new com.ibm.bi.dml.parser.BooleanExpression(bop);
//		((com.ibm.bi.dml.parser.BooleanExpression)current).setLeft(left);
//		((com.ibm.bi.dml.parser.BooleanExpression)current).setRight(right);
//		((com.ibm.bi.dml.parser.BooleanExpression)current).setFilename(DmlSyntacticErrorListener.currentFileName.peek());
//	}
	
	public boolean validateBuiltinFunctions(FunctionCallAssignmentStatementContext ctx) {
		String functionName = ctx.name.getText().replaceAll(" ", "").trim();
		if(functionName.compareTo("write") == 0 || functionName.compareTo(DMLProgram.DEFAULT_NAMESPACE + ".write") == 0) {
			return validateBuiltinWriteFunction(ctx);
		}
		return true;
	}
	
	private boolean validateBuiltinWriteFunction(FunctionCallAssignmentStatementContext ctx) {
		
		return true;
	}
	
	public ArrayList<com.ibm.bi.dml.parser.ParameterExpression> getParameterExpressionList(List<ParameterizedExpressionContext> paramExprs) {
		ArrayList<com.ibm.bi.dml.parser.ParameterExpression> retVal = new ArrayList<com.ibm.bi.dml.parser.ParameterExpression>();
		for(ParameterizedExpressionContext ctx : paramExprs) {
			String paramName = null;
			if(ctx.paramName != null && ctx.paramName.getText() != null && !ctx.paramName.getText().isEmpty()) {
				paramName = ctx.paramName.getText();
			}
			com.ibm.bi.dml.parser.ParameterExpression myArg = new com.ibm.bi.dml.parser.ParameterExpression(paramName, ctx.paramVal.info.expr);
			retVal.add(myArg);
		}
		return retVal;
	}
}
