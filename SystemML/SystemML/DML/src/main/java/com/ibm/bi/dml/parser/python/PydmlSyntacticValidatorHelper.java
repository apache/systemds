/**
 * IBM Confidential
 * OCO Source Materials
 * (C) Copyright IBM Corp. 2010, 2014
 * The source code for this program is not published or otherwise divested of its trade secrets, irrespective of what has been deposited with the U.S. Copyright Office.
 */

package com.ibm.bi.dml.parser.python;

import java.util.ArrayList;
import java.util.List;

import org.antlr.v4.runtime.Token;

import com.ibm.bi.dml.parser.DMLProgram;
import com.ibm.bi.dml.parser.python.PydmlParser.FunctionCallAssignmentStatementContext;
import com.ibm.bi.dml.parser.python.PydmlParser.ParameterizedExpressionContext;


public class PydmlSyntacticValidatorHelper {
	@SuppressWarnings("unused")
	private static final String _COPYRIGHT = "Licensed Materials - Property of IBM\n(C) Copyright IBM Corp. 2010, 2015\n" +
                                             "US Government Users Restricted Rights - Use, duplication  disclosure restricted by GSA ADP Schedule Contract with IBM Corp.";
	
	public void notifyErrorListeners(String message, int line, int charPositionInLine) {
		PyDMLParserWrapper.ERROR_LISTENER_INSTANCE.validationError(line, charPositionInLine, message);
	}
	
	public void notifyErrorListeners(String message, Token op) {
		PyDMLParserWrapper.ERROR_LISTENER_INSTANCE.validationError(op.getLine(), op.getCharPositionInLine(), message);
	}
	
	public String getCurrentFileName() {
		return PydmlSyntacticErrorListener.currentFileName.peek();
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
