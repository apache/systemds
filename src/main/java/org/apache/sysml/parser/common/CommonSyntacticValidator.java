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

package org.apache.sysml.parser.common;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.regex.Pattern;

import org.antlr.v4.runtime.ParserRuleContext;
import org.antlr.v4.runtime.Token;
import org.apache.sysml.api.DMLScript;
import org.apache.sysml.parser.AssignmentStatement;
import org.apache.sysml.parser.BinaryExpression;
import org.apache.sysml.parser.BooleanExpression;
import org.apache.sysml.parser.BooleanIdentifier;
import org.apache.sysml.parser.BuiltinFunctionExpression;
import org.apache.sysml.parser.ConstIdentifier;
import org.apache.sysml.parser.DMLProgram;
import org.apache.sysml.parser.DataExpression;
import org.apache.sysml.parser.DataIdentifier;
import org.apache.sysml.parser.DoubleIdentifier;
import org.apache.sysml.parser.Expression;
import org.apache.sysml.parser.Expression.DataOp;
import org.apache.sysml.parser.FunctionCallIdentifier;
import org.apache.sysml.parser.IntIdentifier;
import org.apache.sysml.parser.LanguageException;
import org.apache.sysml.parser.MultiAssignmentStatement;
import org.apache.sysml.parser.OutputStatement;
import org.apache.sysml.parser.ParameterExpression;
import org.apache.sysml.parser.ParameterizedBuiltinFunctionExpression;
import org.apache.sysml.parser.PrintStatement;
import org.apache.sysml.parser.RelationalExpression;
import org.apache.sysml.parser.Statement;
import org.apache.sysml.parser.StringIdentifier;
import org.apache.sysml.parser.dml.DmlSyntacticValidator;
import org.apache.sysml.parser.pydml.PydmlSyntacticValidator;
import org.apache.sysml.parser.BuiltinConstant;

/**
 * Contains fields and (helper) methods common to {@link DmlSyntacticValidator} and {@link PydmlSyntacticValidator}
 */
public abstract class CommonSyntacticValidator {

	protected final CustomErrorListener errorListener;
	protected final String currentFile;
	protected String _workingDir = ".";   //current working directory
	protected Map<String,String> argVals = null;
	protected String sourceNamespace = null;
	// Track imported scripts to prevent infinite recursion
	protected static ThreadLocal<HashMap<String, String>> _scripts = new ThreadLocal<HashMap<String, String>>() {
		@Override protected HashMap<String, String> initialValue() { return new HashMap<>(); }
	};
	// Map namespaces to full paths as defined only from source statements in this script (i.e., currentFile)
	protected HashMap<String, String> sources;
	// Names of new internal and external functions defined in this script (i.e., currentFile)
	protected Set<String> functions;
	
	public static void init() {
		_scripts.get().clear();
	}

	public CommonSyntacticValidator(CustomErrorListener errorListener, Map<String,String> argVals, String sourceNamespace, Set<String> prepFunctions) {
		this.errorListener = errorListener;
		currentFile = errorListener.getCurrentFileName();
		this.argVals = argVals;
		this.sourceNamespace = sourceNamespace;
		sources = new HashMap<>();
		functions = (null != prepFunctions) ? prepFunctions : new HashSet<>();
	}

	protected void notifyErrorListeners(String message, Token op) {
		if (!DMLScript.VALIDATOR_IGNORE_ISSUES) {
			errorListener.validationError(op.getLine(), op.getCharPositionInLine(), message);
		}
	}

	protected void raiseWarning(String message, Token op) {
		errorListener.validationWarning(op.getLine(), op.getCharPositionInLine(), message);
	}

	/**
	 * Obtain the namespace separator ({@code ::} for DML and {@code .} for
	 * PYDML) that is used to specify a namespace and a function in that
	 * namespace.
	 * 
	 * @return The namespace separator
	 */
	public abstract String namespaceResolutionOp();

	/**
	 * Obtain the namespace and the function name as a two-element array based
	 * on the fully-qualified function name. If no namespace is supplied in
	 * front of the function name, the default namespace will be used.
	 * 
	 * @param fullyQualifiedFunctionName
	 *            Namespace followed by separator ({@code ::} for DML and
	 *            {@code .} for PYDML) followed by function name (for example,
	 *            {@code mynamespace::myfunctionname}), or only function name if
	 *            the default namespace is used (for example,
	 *            {@code myfunctionname}).
	 * @return Two-element array consisting of namespace and function name, or
	 *         {@code null}.
	 */
	protected String[] getQualifiedNames(String fullyQualifiedFunctionName) {
		String splitStr = Pattern.quote(namespaceResolutionOp());
		String [] fnNames = fullyQualifiedFunctionName.split(splitStr);
		String functionName = "";
		String namespace = "";
		if(fnNames.length == 1) {
			namespace = DMLProgram.DEFAULT_NAMESPACE;
			functionName = fnNames[0].trim();
		}
		else if(fnNames.length == 2) {
			namespace = getQualifiedNamespace(fnNames[0].trim());
			functionName = fnNames[1].trim();
		}
		else
			return null;

		String[] retVal = new String[2];
		retVal[0] = namespace;
		retVal[1] = functionName;
		return retVal;
	}
	
	protected String getQualifiedNamespace(String namespace) {
		String path = sources.get(namespace);
		return (path != null && path.length() > 0) ? path : namespace;
	}

	protected void validateNamespace(String namespace, String filePath, ParserRuleContext ctx) {
		if (!sources.containsKey(namespace)) {
			sources.put(namespace, filePath);
		}
		else if (!sources.get(namespace).equals(filePath)) {
			// Only throw an exception if the filepath is different
			// If the filepath is same, ignore the statement. This is useful for repeated definition of common dml files such as source("nn/util.dml") as util
			notifyErrorListeners("Namespace Conflict: '" + namespace + "' already defined as " + sources.get(namespace), ctx.start);
		}
	}

	protected void setFileLineColumn(Expression expr, ParserRuleContext ctx) {
		expr.setCtxValuesAndFilename(ctx, currentFile);
	}

	protected void setFileLineColumn(Statement stmt, ParserRuleContext ctx) {
		stmt.setCtxValuesAndFilename(ctx, currentFile);
	}

	// For String literal "True/TRUE"
	public abstract String trueStringLiteral();

	// For String literal "False/FALSE"
	public abstract String falseStringLiteral();

	// --------------------------------------------------------------------
	//        HELPER METHODS FOR OVERRIDDEN VISITOR FUNCTIONS
	// --------------------------------------------------------------------

	protected void binaryExpressionHelper(ParserRuleContext ctx, ExpressionInfo left, ExpressionInfo right,
			ExpressionInfo me, String op) {
		if(left.expr != null && right.expr != null) {
			Expression.BinaryOp bop = Expression.getBinaryOp(op);
			BinaryExpression be = new BinaryExpression(bop);
			be = new BinaryExpression(bop);
			be.setLeft(left.expr);
			be.setRight(right.expr);
			me.expr = be;
			setFileLineColumn(me.expr, ctx);
		}
	}

	protected void relationalExpressionHelper(ParserRuleContext ctx, ExpressionInfo left, ExpressionInfo right,
			ExpressionInfo me, String op) {
		if(left.expr != null && right.expr != null) {
			Expression.RelationalOp rop = Expression.getRelationalOp(op);
			RelationalExpression re = new RelationalExpression(rop);
			re.setLeft(left.expr);
			re.setRight(right.expr);
			me.expr = re;
			setFileLineColumn(me.expr, ctx);
		}
	}

	protected void booleanExpressionHelper(ParserRuleContext ctx, ExpressionInfo left, ExpressionInfo right,
			ExpressionInfo me, String op) {
		if(left.expr != null && right.expr != null) {
			Expression.BooleanOp bop = Expression.getBooleanOp(op);
			BooleanExpression re = new BooleanExpression(bop);
			re.setLeft(left.expr);
			re.setRight(right.expr);
			me.expr = re;
			setFileLineColumn(me.expr, ctx);
		}
	}



	protected void unaryExpressionHelper(ParserRuleContext ctx, ExpressionInfo left, ExpressionInfo me, String op) {
		if(left.expr != null) {
			if(left.expr instanceof IntIdentifier) {
				if(op.equals("-")) {
					((IntIdentifier) left.expr).multiplyByMinusOne();
				}
				me.expr = left.expr;
			}
			else if(left.expr instanceof DoubleIdentifier) {
				if(op.equals("-")) {
					((DoubleIdentifier) left.expr).multiplyByMinusOne();
				}
				me.expr = left.expr;
			}
			else {
				Expression right = new IntIdentifier(ctx, 1, currentFile);
				if(op.equals("-")) {
					right = new IntIdentifier(ctx, -1, currentFile);
				}

				Expression.BinaryOp bop = Expression.getBinaryOp("*");
				BinaryExpression be = new BinaryExpression(bop);
				be.setLeft(left.expr);
				be.setRight(right);
				me.expr = be;
			}
			setFileLineColumn(me.expr, ctx);
		}
	}

	protected void unaryBooleanExpressionHelper(ParserRuleContext ctx, ExpressionInfo left, ExpressionInfo me,
			String op) {
		if(left.expr != null) {
			Expression.BooleanOp bop = Expression.getBooleanOp(op);
			BooleanExpression be = new BooleanExpression(bop);
			be.setLeft(left.expr);
			me.expr = be;
			setFileLineColumn(me.expr, ctx);
		}
	}


	protected void constDoubleIdExpressionHelper(ParserRuleContext ctx, ExpressionInfo me) {
		try {
			double val = Double.parseDouble(ctx.getText());
			me.expr = new DoubleIdentifier(ctx, val, currentFile);
		}
		catch(Exception e) {
			notifyErrorListeners("cannot parse the float value: \'" +  ctx.getText() + "\'", ctx.getStart());
			return;
		}
	}

	protected void constIntIdExpressionHelper(ParserRuleContext ctx, ExpressionInfo me) {
		try {
			long val = Long.parseLong(ctx.getText());
			me.expr = new IntIdentifier(ctx, val, currentFile);
		}
		catch(Exception e) {
			notifyErrorListeners("cannot parse the int value: \'" +  ctx.getText() + "\'", ctx.getStart());
			return;
		}
	}

	protected String extractStringInQuotes(String text, boolean inQuotes) {
		String val = null;
		if(inQuotes) {
			if(	(text.startsWith("\"") && text.endsWith("\"")) ||
				(text.startsWith("\'") && text.endsWith("\'"))) {
				if(text.length() > 2) {
					val = text.substring(1, text.length()-1)
						.replaceAll("\\\\b","\b")
						.replaceAll("\\\\t","\t")
						.replaceAll("\\\\n","\n")
						.replaceAll("\\\\f","\f")
						.replaceAll("\\\\r","\r")
						.replace("\\'","'")
						.replace("\\\"","\"");
				}
				else if(text.equals("\"\"") || text.equals("\'\'")) {
					val = "";
				}
			}
		}
		else {
			val = text.replaceAll("\\\\b","\b")
					.replaceAll("\\\\t","\t")
					.replaceAll("\\\\n","\n")
					.replaceAll("\\\\f","\f")
					.replaceAll("\\\\r","\r")
					.replace("\\'","'")
					.replace("\\\"","\"");
		}
		return val;
	}
	
	protected void constStringIdExpressionHelper(ParserRuleContext ctx, ExpressionInfo me) {
		String val = extractStringInQuotes(ctx.getText(), true);
		if(val == null) {
			notifyErrorListeners("incorrect string literal ", ctx.start);
			return;
		}

		me.expr = new StringIdentifier(ctx, val, currentFile);
	}

	protected void booleanIdentifierHelper(ParserRuleContext ctx, boolean val, ExpressionInfo info) {
		info.expr = new BooleanIdentifier(ctx, val, currentFile);
		setFileLineColumn(info.expr, ctx);
	}

	protected void exitDataIdExpressionHelper(ParserRuleContext ctx, ExpressionInfo me, ExpressionInfo dataInfo) {
		// inject builtin constant
		if (dataInfo.expr instanceof DataIdentifier) {
			DataIdentifier id = ((DataIdentifier) dataInfo.expr);
			if (BuiltinConstant.contains(id.getName())) { 
				dataInfo.expr = new DoubleIdentifier(BuiltinConstant.valueOf(id.getName()).get(), dataInfo.expr);
			}
		}
		me.expr = dataInfo.expr;
		// If "The parameter $X either needs to be passed through commandline or initialized to default value" validation
		// error occurs, then dataInfo.expr is null which would cause a null pointer exception with the following code.
		// Therefore, check for null so that parsing can continue so all parsing issues can be determined.
		if (me.expr != null) {
			me.expr.setCtxValuesAndFilename(ctx, currentFile);
		}
	}

	protected ConstIdentifier getConstIdFromString(ParserRuleContext ctx, String varValue) {
		// Compare to "True/TRUE"
		if(varValue.equals(trueStringLiteral()))
			return new BooleanIdentifier(ctx, true, currentFile);

		// Compare to "False/FALSE"
		if(varValue.equals(falseStringLiteral()))
			return new BooleanIdentifier(ctx, false, currentFile);

		// Check for long literal
		// NOTE: we use exception handling instead of Longs.tryParse for backwards compatibility with guava <14.1
		// Also the alternative of Ints.tryParse and falling back to double would not be lossless in all cases. 
		try {
			long lval = Long.parseLong(varValue);
			return new IntIdentifier(ctx, lval, currentFile);
		}
		catch(Exception ex) {
			//continue
		}
		
		// Check for double literal
		// NOTE: we use exception handling instead of Doubles.tryParse for backwards compatibility with guava <14.0
		try {
			double dval = Double.parseDouble(varValue);
			return new DoubleIdentifier(ctx, dval, currentFile);
		}
		catch(Exception ex) {
			//continue
		}
			
		// Otherwise it is a string literal (optionally enclosed within single or double quotes)
		String val = "";
		String text = varValue;
		if(	(text.startsWith("\"") && text.endsWith("\"")) || (text.startsWith("\'") && text.endsWith("\'"))) {
			if(text.length() > 2) {
				val = extractStringInQuotes(text, true);
			}
		}
		else {
			// the commandline parameters can be passed without any quotes
			val = extractStringInQuotes(text, false);
		}
		return new StringIdentifier(ctx, val, currentFile);
	}


	protected void fillExpressionInfoCommandLineParameters(ParserRuleContext ctx, String varName, ExpressionInfo dataInfo) {

		if(!varName.startsWith("$")) {
			notifyErrorListeners("commandline param does not start with $", ctx.start);
			return;
		}

		String varValue = null;
		for(Map.Entry<String, String> arg : this.argVals.entrySet()) {
			if(arg.getKey().equals(varName)) {
				if(varValue != null) {
					notifyErrorListeners("multiple values passed for the parameter " + varName + " via commandline", ctx.start);
					return;
				}
				else {
					varValue = arg.getValue();
				}
			}
		}

		if(varValue == null) {
			return;
		}

		// Command line param cannot be empty string
		// If you want to pass space, please quote it
		if(varValue.equals(""))
			return;

		dataInfo.expr = getConstIdFromString(ctx, varValue);
	}

	protected void exitAssignmentStatementHelper(ParserRuleContext ctx, String lhs, ExpressionInfo dataInfo,
			Token lhsStart, ExpressionInfo rhs, StatementInfo info) {
		if(lhs.startsWith("$")) {
			notifyErrorListeners("assignment of commandline parameters is not allowed. (Quickfix: try using someLocalVariable=ifdef(" + lhs + ", default value))", ctx.start);
			return;
		}

		DataIdentifier target = null;
		if(dataInfo.expr instanceof DataIdentifier) {
			target = (DataIdentifier) dataInfo.expr;
			Expression source = rhs.expr;
			try {
				info.stmt = new AssignmentStatement(ctx, target, source, currentFile);
			} catch (LanguageException e) {
				// TODO: extract more meaningful info from this exception.
				notifyErrorListeners("invalid assignment: " + e.getMessage(), lhsStart);
				return;
			}
		}
		else {
			notifyErrorListeners("incorrect lvalue in assignment statement", lhsStart);
			return;
		}
	}


	// -----------------------------------------------------------------
	// Helper Functions for exit*FunctionCall*AssignmentStatement
	// -----------------------------------------------------------------

	protected void setPrintStatement(ParserRuleContext ctx, String functionName,
			ArrayList<ParameterExpression> paramExpression, StatementInfo thisinfo) {
		if (DMLScript.VALIDATOR_IGNORE_ISSUES == true) { // create dummy print statement
			try {
				thisinfo.stmt = new PrintStatement(ctx, functionName, currentFile);
			} catch (LanguageException e) {
				e.printStackTrace();
			}
			return;
		}
		int numParams = paramExpression.size();
		if (numParams == 0) {
			notifyErrorListeners(functionName + "() must have more than 0 parameters", ctx.start);
			return;
		} else if (numParams == 1) {
			Expression expr = paramExpression.get(0).getExpr();
			if(expr == null) {
				notifyErrorListeners("cannot process " + functionName + "() function", ctx.start);
				return;
			}
			try {
				List<Expression> expList = new ArrayList<>();
				expList.add(expr);
				thisinfo.stmt = new PrintStatement(ctx, functionName, expList, currentFile);
			} catch (LanguageException e) {
				notifyErrorListeners("cannot process " + functionName + "() function", ctx.start);
				return;
			}
		} else if (numParams > 1) {
			if ("stop".equals(functionName)) {
				notifyErrorListeners("stop() function cannot have more than 1 parameter", ctx.start);
				return;
			}

			Expression firstExp = paramExpression.get(0).getExpr();
			if (firstExp == null) {
				notifyErrorListeners("cannot process " + functionName + "() function", ctx.start);
				return;
			}
			if (!(firstExp instanceof StringIdentifier)) {
				notifyErrorListeners("printf-style functionality requires first print parameter to be a string", ctx.start);
				return;
			}
			try {
				List<Expression> expressions = new ArrayList<>();
				for (ParameterExpression pe : paramExpression) {
					Expression expression = pe.getExpr();
					expressions.add(expression);
				}
				thisinfo.stmt = new PrintStatement(ctx, functionName, expressions, currentFile);
			} catch (LanguageException e) {
				notifyErrorListeners("cannot process " + functionName + "() function", ctx.start);
				return;
			}
		}
	}

	protected void setOutputStatement(ParserRuleContext ctx,
			ArrayList<ParameterExpression> paramExpression, StatementInfo info) {
		if(paramExpression.size() < 2){
			notifyErrorListeners("incorrect usage of write function (at least 2 arguments required)", ctx.start);
			return;
		}
		if(paramExpression.get(0).getExpr() instanceof DataIdentifier) {
			HashMap<String, Expression> varParams = new HashMap<>();
			varParams.put(DataExpression.IO_FILENAME, paramExpression.get(1).getExpr());
			for(int i = 2; i < paramExpression.size(); i++) {
				// DataExpression.FORMAT_TYPE, DataExpression.DELIM_DELIMITER, DataExpression.DELIM_HAS_HEADER_ROW,  DataExpression.DELIM_SPARSE
				varParams.put(paramExpression.get(i).getName(), paramExpression.get(i).getExpr());
			}

			DataExpression  dataExpression = new DataExpression(ctx, DataOp.WRITE, varParams, currentFile);
			info.stmt = new OutputStatement(ctx, (DataIdentifier) paramExpression.get(0).getExpr(), DataOp.WRITE,
					currentFile);
			((OutputStatement)info.stmt).setExprParams(dataExpression);
		}
		else {
			notifyErrorListeners("incorrect usage of write function", ctx.start);
		}
	}

	protected void setAssignmentStatement(ParserRuleContext ctx, StatementInfo info, DataIdentifier target, Expression expression) {
		try {
			info.stmt = new AssignmentStatement(ctx, target, expression, currentFile);
		} catch (LanguageException e) {
			// TODO: extract more meaningful info from this exception.
			notifyErrorListeners("invalid function call", ctx.start);
			return;
		}
	}

	/**
	 * Information about built in functions converted to a common format between
	 * PyDML and DML for the runtime.
	 */
	public static class ConvertedDMLSyntax {
		public final String namespace;
		public final String functionName;
		public final ArrayList<ParameterExpression> paramExpression;
		public ConvertedDMLSyntax(String namespace, String functionName,
				ArrayList<ParameterExpression> paramExpression) {
			this.namespace = namespace;
			this.functionName = functionName;
			this.paramExpression = paramExpression;
		}
	}

	/**
	 * Converts PyDML/DML built in functions to a common format for the runtime.
	 * @param ctx antlr rule context
	 * @param namespace Namespace of the function
	 * @param functionName Name of the builtin function
	 * @param paramExpression Array of parameter names and values
	 * @param fnName Token of the built in function identifier
	 * @return common syntax format for runtime
	 */
	protected abstract ConvertedDMLSyntax convertToDMLSyntax(ParserRuleContext ctx, String namespace, String functionName, ArrayList<ParameterExpression> paramExpression,
			Token fnName);

	/**
	 * Function overridden for DML &amp; PyDML that handles any language specific builtin functions
	 * @param ctx antlr rule context
	 * @param functionName Name of the builtin function
	 * @param paramExpressions Array of parameter names and values
	 * @return instance of {@link Expression}
	 */
	protected abstract Expression handleLanguageSpecificFunction(ParserRuleContext ctx, String functionName, ArrayList<ParameterExpression> paramExpressions);

	/** Creates a builtin function expression.
	 * 
	 * @param ctx antlr rule context
	 * @param functionName Name of the builtin function
	 * @param paramExpressions Array of parameter names and values
	 * @return expression if found otherwise null
	 */
	protected Expression buildForBuiltInFunction(ParserRuleContext ctx, String functionName, ArrayList<ParameterExpression> paramExpressions) {
		// In global namespace, so it can be a builtin function
		// Double verification: verify passed function name is a (non-parameterized) built-in function.
		try {
			if (functions.contains(functionName)) {
				// It is a user function definition (which takes precedence if name same as built-in)
				return null;
			}
			
			Expression lsf = handleLanguageSpecificFunction(ctx, functionName, paramExpressions);
			if (lsf != null) {
				setFileLineColumn(lsf, ctx);
				return lsf;
			}

			BuiltinFunctionExpression bife = BuiltinFunctionExpression.getBuiltinFunctionExpression(ctx, functionName, paramExpressions, currentFile);
			if (bife != null) {
				// It is a builtin function
				return bife;
			}

			ParameterizedBuiltinFunctionExpression pbife = ParameterizedBuiltinFunctionExpression
					.getParamBuiltinFunctionExpression(ctx, functionName, paramExpressions, currentFile);
			if (pbife != null){
				// It is a parameterized builtin function
				return pbife;
			}

			// built-in read, rand ...
			DataExpression dbife = DataExpression.getDataExpression(ctx, functionName, paramExpressions, currentFile, errorListener);
			if (dbife != null){
				return dbife;
			}
		} 
		catch(Exception e) {
			notifyErrorListeners("unable to process builtin function expression " + functionName  + ":" + e.getMessage(), ctx.start);
		}
		return null;
	}


	protected void functionCallAssignmentStatementHelper(final ParserRuleContext ctx,
			Set<String> printStatements, Set<String> outputStatements, final Expression dataInfo,
			final StatementInfo info, final Token nameToken, Token targetListToken, String namespace,
			String functionName, ArrayList<ParameterExpression> paramExpression, boolean hasLHS) {
		ConvertedDMLSyntax convertedSyntax = convertToDMLSyntax(ctx, namespace, functionName, paramExpression, nameToken);
		if(convertedSyntax == null) {
			return;
		}
		else {
			namespace = convertedSyntax.namespace;
			functionName = convertedSyntax.functionName;
			paramExpression = convertedSyntax.paramExpression;
		}

		// For builtin functions without LHS
		if(namespace.equals(DMLProgram.DEFAULT_NAMESPACE) && !functions.contains(functionName)) {
			if (printStatements.contains(functionName)){
				setPrintStatement(ctx, functionName, paramExpression, info);
				return;
			}
			else if (outputStatements.contains(functionName)){
				setOutputStatement(ctx, paramExpression, info);
				return;
			}
		}

		DataIdentifier target = null;
		if(dataInfo instanceof DataIdentifier) {
			target = (DataIdentifier) dataInfo;
		}
		else if (dataInfo != null) {
			notifyErrorListeners("incorrect lvalue for function call ", targetListToken);
			return;
		}

		// For builtin functions with LHS
		if(namespace.equals(DMLProgram.DEFAULT_NAMESPACE) && !functions.contains(functionName)){
			Expression e = buildForBuiltInFunction(ctx, functionName, paramExpression);
			if( e != null ) {
				setAssignmentStatement(ctx, info, target, e);
				return;
			}
		}

		// handle user-defined functions
		setAssignmentStatement(ctx, info, target,
			createFunctionCall(ctx, namespace, functionName, paramExpression));
	}
	
	protected FunctionCallIdentifier createFunctionCall(ParserRuleContext ctx,
		String namespace, String functionName, ArrayList<ParameterExpression> paramExpression) {
		FunctionCallIdentifier functCall = new FunctionCallIdentifier(paramExpression);
		functCall.setFunctionName(functionName);
		String inferNamespace = (sourceNamespace != null && sourceNamespace.length() > 0
			&& DMLProgram.DEFAULT_NAMESPACE.equals(namespace)) ? sourceNamespace : namespace;
		functCall.setFunctionNamespace(inferNamespace);
		functCall.setCtxValuesAndFilename(ctx, currentFile);
		return functCall;
	}

	protected void setMultiAssignmentStatement(ArrayList<DataIdentifier> target, Expression expression, ParserRuleContext ctx, StatementInfo info) {
		info.stmt = new MultiAssignmentStatement(target, expression);
		info.stmt.setCtxValuesAndFilename(ctx, currentFile);
	}

	// -----------------------------------------------------------------
	// End of Helper Functions for exit*FunctionCall*AssignmentStatement
	// -----------------------------------------------------------------

	/**
	 * Indicates if the given data type string is a valid data type. 
	 * 
	 * @param datatype data type (matrix, frame, or scalar)
	 * @param start antlr token
	 */
	protected void checkValidDataType(String datatype, Token start) {
		boolean validMatrixType = 
				datatype.equals("matrix") || datatype.equals("Matrix") || 
				datatype.equals("frame") || datatype.equals("Frame") ||
				datatype.equals("scalar") || datatype.equals("Scalar");
		if(!validMatrixType	) {
			notifyErrorListeners("incorrect datatype (expected matrix, frame or scalar)", start);
		}
	}
}
