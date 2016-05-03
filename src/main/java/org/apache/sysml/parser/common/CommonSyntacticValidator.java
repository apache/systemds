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
import java.util.Map;
import java.util.Set;
import java.util.regex.Pattern;

import org.antlr.v4.runtime.ParserRuleContext;
import org.antlr.v4.runtime.Token;
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
import org.apache.sysml.parser.IndexedIdentifier;
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
import org.apache.sysml.parser.dml.DmlParser.BuiltinFunctionExpressionContext;
import org.apache.sysml.parser.dml.DmlSyntacticValidator;
import org.apache.sysml.parser.pydml.PydmlSyntacticValidator;

/**
 * Contains fields and (helper) methods common to {@link DmlSyntacticValidator} and {@link PydmlSyntacticValidator}
 */
public abstract class CommonSyntacticValidator {

	protected final CustomErrorListener errorListener;
	protected final String currentFile;
	protected String _workingDir = ".";   //current working directory
	protected Map<String,String> argVals = null;
	protected String sourceNamespace = null;
	// track imported scripts to prevent infinite recursion
	protected static ThreadLocal<HashMap<String, String>> _scripts = new ThreadLocal<HashMap<String, String>>() {
		@Override protected HashMap<String, String> initialValue() { return new HashMap<String, String>(); }
	};
	// mapping of namespaces to full paths as defined only from source statements in this script (i.e., currentFile)
	protected HashMap<String, String> sources;
	
	public static void init() {
		_scripts.get().clear();
	}

	public CommonSyntacticValidator(CustomErrorListener errorListener, Map<String,String> argVals, String sourceNamespace) {
		this.errorListener = errorListener;
		currentFile = errorListener.getCurrentFileName();
		this.argVals = argVals;
		this.sourceNamespace = sourceNamespace;
		sources = new HashMap<String, String>();
	}

	protected void notifyErrorListeners(String message, int line, int charPositionInLine) {
		errorListener.validationError(line, charPositionInLine, message);
	}

	protected void notifyErrorListeners(String message, Token op) {
		errorListener.validationError(op.getLine(), op.getCharPositionInLine(), message);
	}

	protected void raiseWarning(String message, Token op) {
		errorListener.validationWarning(op.getLine(), op.getCharPositionInLine(), message);
	}

	// Different namespaces for DML (::) and PyDml (.)
	public abstract String namespaceResolutionOp();

	// Returns list of two elements <namespace, function names>, else null
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
		else {
			notifyErrorListeners("Namespace Conflict: '" + namespace + "' already defined as " + sources.get(namespace), ctx.start);
		}
	}
	
	protected boolean validateBuiltinFunctions(String function) {
		String functionName = function.replaceAll(" ", "").trim();
		if(functionName.equals("write") || functionName.equals(DMLProgram.DEFAULT_NAMESPACE + namespaceResolutionOp() + "write")) {
			return validateBuiltinWriteFunction(function);
		}
		return true;
	}

	protected boolean validateBuiltinWriteFunction(String function) {
		return true;
	}


	protected void setFileLineColumn(Expression expr, ParserRuleContext ctx) {
		String txt = ctx.getText();
		expr.setFilename(currentFile);
		expr.setBeginLine(ctx.start.getLine());
		expr.setBeginColumn(ctx.start.getCharPositionInLine());
		expr.setEndLine(ctx.stop.getLine());
		expr.setEndColumn(ctx.stop.getCharPositionInLine());
		if(expr.getBeginColumn() == expr.getEndColumn() && expr.getBeginLine() == expr.getEndLine() && txt.length() > 1) {
			expr.setEndColumn(expr.getBeginColumn() + txt.length() - 1);
		}
	}

	protected void setFileLineColumn(Statement stmt, ParserRuleContext ctx) {
		String txt = ctx.getText();
		stmt.setFilename(currentFile);
		stmt.setBeginLine(ctx.start.getLine());
		stmt.setBeginColumn(ctx.start.getCharPositionInLine());
		stmt.setEndLine(ctx.stop.getLine());
		stmt.setEndColumn(ctx.stop.getCharPositionInLine());
		if(stmt.getBeginColumn() == stmt.getEndColumn() && stmt.getBeginLine() == stmt.getEndLine() && txt.length() > 1) {
			stmt.setEndColumn(stmt.getBeginColumn() + txt.length() - 1);
		}
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
			Token start = ctx.start;
			String fileName = currentFile;
			int line = start.getLine();
			int col = start.getCharPositionInLine();

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
				Expression right = new IntIdentifier(1, fileName, line, col, line, col);
				if(op.equals("-")) {
					right = new IntIdentifier(-1, fileName, line, col, line, col);
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
			Token start = ctx.start;
			double val = Double.parseDouble(ctx.getText());
			int linePosition = start.getLine();
			int charPosition = start.getCharPositionInLine();
			me.expr = new DoubleIdentifier(val, currentFile, linePosition, charPosition, linePosition, charPosition);
			setFileLineColumn(me.expr, ctx);
		}
		catch(Exception e) {
			notifyErrorListeners("cannot parse the float value: \'" +  ctx.getText() + "\'", ctx.getStart());
			return;
		}
	}

	protected void constIntIdExpressionHelper(ParserRuleContext ctx, ExpressionInfo me) {
		try {
			Token start = ctx.start;
			long val = Long.parseLong(ctx.getText());
			int linePosition = start.getLine();
			int charPosition = start.getCharPositionInLine();
			me.expr = new IntIdentifier(val, currentFile, linePosition, charPosition, linePosition, charPosition);
			setFileLineColumn(me.expr, ctx);
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
						.replaceAll("\\\\r","\r");
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
					.replaceAll("\\\\r","\r");
		}
		return val;
	}
	
	protected void constStringIdExpressionHelper(ParserRuleContext ctx, ExpressionInfo me) {
		String val = extractStringInQuotes(ctx.getText(), true);
		if(val == null) {
			notifyErrorListeners("incorrect string literal ", ctx.start);
			return;
		}

		int linePosition = ctx.start.getLine();
		int charPosition = ctx.start.getCharPositionInLine();
		me.expr = new StringIdentifier(val, currentFile, linePosition, charPosition, linePosition, charPosition);
		setFileLineColumn(me.expr, ctx);
	}

	protected void booleanIdentifierHelper(ParserRuleContext ctx, boolean val, ExpressionInfo info) {
		int linePosition = ctx.start.getLine();
		int charPosition = ctx.start.getCharPositionInLine();
		info.expr = new BooleanIdentifier(val, currentFile, linePosition, charPosition, linePosition, charPosition);
		setFileLineColumn(info.expr, ctx);
	}

	protected void exitDataIdExpressionHelper(ParserRuleContext ctx, ExpressionInfo me, ExpressionInfo dataInfo) {
		me.expr = dataInfo.expr;
		// If "The parameter $X either needs to be passed through commandline or initialized to default value" validation
		// error occurs, then dataInfo.expr is null which would cause a null pointer exception with the following code.
		// Therefore, check for null so that parsing can continue so all parsing issues can be determined.
		if (me.expr != null) {
			int line = ctx.start.getLine();
			int col = ctx.start.getCharPositionInLine();
			me.expr.setAllPositions(currentFile, line, col, line, col);
			setFileLineColumn(me.expr, ctx);
		}
	}

	protected void exitIndexedExpressionHelper(ParserRuleContext ctx, String name, ExpressionInfo dataInfo,
			ExpressionInfo rowLower, ExpressionInfo rowUpper, ExpressionInfo colLower, ExpressionInfo colUpper) {
		dataInfo.expr = new IndexedIdentifier(name, false, false);
		setFileLineColumn(dataInfo.expr, ctx);
		boolean isRowLower = rowLower != null;
		boolean isRowUpper = rowUpper != null;
		boolean isColLower = colLower != null;
		boolean isColUpper = colUpper != null;
		try {
			ArrayList< ArrayList<Expression> > exprList = new ArrayList< ArrayList<Expression> >();

			ArrayList<Expression> rowIndices = new ArrayList<Expression>();
			ArrayList<Expression> colIndices = new ArrayList<Expression>();


			if(!isRowLower && !isRowUpper) {
				// both not set
				rowIndices.add(null); rowIndices.add(null);
			}
			else if(isRowLower && isRowUpper) {
				// both set
				rowIndices.add(incrementByOne(rowLower.expr, ctx));
				rowIndices.add(rowUpper.expr);
			}
			else if(isRowLower && !isRowUpper) {
				// only row set
				rowIndices.add(incrementByOne(rowLower.expr, ctx));
			}
			else {
				notifyErrorListeners("incorrect index expression for row", ctx.start);
				return;
			}

			if(!isColLower && !isColUpper) {
				// both not set
				colIndices.add(null); colIndices.add(null);
			}
			else if(isColLower && isColUpper) {
				colIndices.add(incrementByOne(colLower.expr, ctx));
				colIndices.add(colUpper.expr);
			}
			else if(isColLower && !isColUpper) {
				colIndices.add(incrementByOne(colLower.expr, ctx));
			}
			else {
				notifyErrorListeners("incorrect index expression for column", ctx.start);
				return;
			}
			exprList.add(rowIndices);
			exprList.add(colIndices);
			((IndexedIdentifier) dataInfo.expr).setIndices(exprList);
		}
		catch(Exception e) {
			notifyErrorListeners("cannot set the indices", ctx.start);
			return;
		}
	}

	private Expression incrementByOne(Expression expr, ParserRuleContext ctx) {
		// For maintaining semantic consistency, we have decided to keep 1-based indexing
		// If in future, PyDML becomes more popular than DML, this can be switched.
		return expr;
	}

	protected ConstIdentifier getConstIdFromString(String varValue, Token start) {

		int linePosition = start.getLine();
		int charPosition = start.getCharPositionInLine();

		// Compare to "True/TRUE"
		if(varValue.equals(trueStringLiteral()))
			return new BooleanIdentifier(true, currentFile, linePosition, charPosition, linePosition, charPosition);

		// Compare to "False/FALSE"
		if(varValue.equals(falseStringLiteral()))
			return new BooleanIdentifier(false, currentFile, linePosition, charPosition, linePosition, charPosition);

		// Check for long literal
		// NOTE: we use exception handling instead of Longs.tryParse for backwards compatibility with guava <14.1
		// Also the alternative of Ints.tryParse and falling back to double would not be lossless in all cases. 
		try {
			long lval = Long.parseLong(varValue);
			return new IntIdentifier(lval, currentFile, linePosition, charPosition, linePosition, charPosition);
		}
		catch(Exception ex) {
			//continue
		}
		
		// Check for double literal
		// NOTE: we use exception handling instead of Doubles.tryParse for backwards compatibility with guava <14.0
		try {
			double dval = Double.parseDouble(varValue);
			return new DoubleIdentifier(dval, currentFile, linePosition, charPosition, linePosition, charPosition);
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
		return new StringIdentifier(val, currentFile, linePosition, charPosition, linePosition, charPosition);
	}


	protected void fillExpressionInfoCommandLineParameters(String varName, ExpressionInfo dataInfo, Token start) {

		if(!varName.startsWith("$")) {
			notifyErrorListeners("commandline param doesnot start with $", start);
			return;
		}

		String varValue = null;
		for(Map.Entry<String, String> arg : this.argVals.entrySet()) {
			if(arg.getKey().equals(varName)) {
				if(varValue != null) {
					notifyErrorListeners("multiple values passed for the parameter " + varName + " via commandline", start);
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

		dataInfo.expr = getConstIdFromString(varValue, start);
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

			int line = ctx.start.getLine();
			int col = ctx.start.getCharPositionInLine();
			try {
				info.stmt = new AssignmentStatement(target, source, line, col, line, col);
				setFileLineColumn(info.stmt, ctx);
			} catch (LanguageException e) {
				// TODO: extract more meaningful info from this exception.
				notifyErrorListeners("invalid assignment", lhsStart);
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
		if(paramExpression.size() != 1) {
			notifyErrorListeners(functionName + "() has only one parameter", ctx.start);
			return;
		}
		Expression expr = paramExpression.get(0).getExpr();
		if(expr == null) {
			notifyErrorListeners("cannot process " + functionName + "() function", ctx.start);
			return;
		}
		try {
			int line = ctx.start.getLine();
			int col = ctx.start.getCharPositionInLine();
			thisinfo.stmt = new PrintStatement(functionName, expr, line, col, line, col);
			setFileLineColumn(thisinfo.stmt, ctx);
		} catch (LanguageException e) {
			notifyErrorListeners("cannot process " + functionName + "() function", ctx.start);
			return;
		}
	}

	protected void setOutputStatement(ParserRuleContext ctx,
			ArrayList<ParameterExpression> paramExpression, StatementInfo info) {
		if(paramExpression.size() < 2){
			notifyErrorListeners("incorrect usage of write function (at least 2 arguments required)", ctx.start);
			return;
		}
		if(paramExpression.get(0).getExpr() instanceof DataIdentifier) {
			String fileName = currentFile;
			int line = ctx.start.getLine();
			int col = ctx.start.getCharPositionInLine();
			HashMap<String, Expression> varParams = new HashMap<String, Expression>();
			varParams.put(DataExpression.IO_FILENAME, paramExpression.get(1).getExpr());
			for(int i = 2; i < paramExpression.size(); i++) {
				// DataExpression.FORMAT_TYPE, DataExpression.DELIM_DELIMITER, DataExpression.DELIM_HAS_HEADER_ROW,  DataExpression.DELIM_SPARSE
				varParams.put(paramExpression.get(i).getName(), paramExpression.get(i).getExpr());
			}

			DataExpression  dataExpression = new DataExpression(DataOp.WRITE, varParams, fileName, line, col, line, col);
			info.stmt = new  OutputStatement((DataIdentifier) paramExpression.get(0).getExpr(), DataOp.WRITE, fileName, line, col, line, col);
			setFileLineColumn(info.stmt, ctx);
			((OutputStatement)info.stmt).setExprParams(dataExpression);
		}
		else {
			notifyErrorListeners("incorrect usage of write function", ctx.start);
		}
	}

	protected void setAssignmentStatement(ParserRuleContext ctx, StatementInfo info, DataIdentifier target, Expression expression) {
		try {
			info.stmt = new AssignmentStatement(target, expression, ctx.start.getLine(), ctx.start.getCharPositionInLine(), ctx.start.getLine(), ctx.start.getCharPositionInLine());
			setFileLineColumn(info.stmt, ctx);
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
	};

	/**
	 * Converts PyDML/DML built in functions to a common format for the runtime.
	 * @param ctx
	 * @param namespace Namespace of the function
	 * @param functionName Name of the builtin function
	 * @param paramExpression Array of parameter names and values
	 * @param fnName Token of the built in function identifier
	 * @return
	 */
	protected abstract ConvertedDMLSyntax convertToDMLSyntax(ParserRuleContext ctx, String namespace, String functionName, ArrayList<ParameterExpression> paramExpression,
			Token fnName);

	/**
	 * Function overridden for DML & PyDML that handles any language specific builtin functions
	 * @param ctx
	 * @param functionName
	 * @param paramExpressions
	 * @return  instance of {@link Expression}
	 */
	protected abstract Expression handleLanguageSpecificFunction(ParserRuleContext ctx, String functionName, ArrayList<ParameterExpression> paramExpressions);

	/** Checks for builtin functions and does Action 'f'.
	 * <br/>
	 * Constructs the
	 * appropriate {@link AssignmentStatement} from
	 * {@link CommonSyntacticValidator#functionCallAssignmentStatementHelper(ParserRuleContext, Set, Set, Expression, StatementInfo, Token, Token, String, String, ArrayList, boolean)
	 * or Assign to {@link Expression} from
	 * {@link DmlSyntacticValidator#exitBuiltinFunctionExpression(BuiltinFunctionExpressionContext)}
	 *
	 * @param ctx
	 * @param functionName
	 * @param paramExpressions
	 * @return true if a builtin function was found
	 */
	protected boolean buildForBuiltInFunction(ParserRuleContext ctx, String functionName, ArrayList<ParameterExpression> paramExpressions, Action f) {
		// In global namespace, so it can be a builtin function
		// Double verification: verify passed function name is a (non-parameterized) built-in function.
		String fileName = currentFile;
		int line = ctx.start.getLine();
		int col = ctx.start.getCharPositionInLine();
		try {

			Expression lsf = handleLanguageSpecificFunction(ctx, functionName, paramExpressions);
			if (lsf != null){
				setFileLineColumn(lsf, ctx);
				f.execute(lsf);
				return true;
			}

			BuiltinFunctionExpression bife = BuiltinFunctionExpression.getBuiltinFunctionExpression(functionName, paramExpressions, fileName, line, col, line, col);
			if (bife != null){
				// It is a builtin function
				f.execute(bife);
				return true;
			}

			ParameterizedBuiltinFunctionExpression pbife = ParameterizedBuiltinFunctionExpression.getParamBuiltinFunctionExpression(functionName, paramExpressions, fileName, line, col, line, col);
			if (pbife != null){
				// It is a parameterized builtin function
				f.execute(pbife);
				return true;
			}

			// built-in read, rand ...
			DataExpression dbife = DataExpression.getDataExpression(functionName, paramExpressions, fileName, line, col, line, col);
			if (dbife != null){
				f.execute(dbife);
				return true;
			}
		} catch(Exception e) {
			notifyErrorListeners("unable to process builtin function expression " + functionName  + ":" + e.getMessage(), ctx.start);
			return true;
		}
		return false;
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
		if(namespace.equals(DMLProgram.DEFAULT_NAMESPACE)) {
			if (printStatements.contains(functionName)){
				setPrintStatement(ctx, functionName, paramExpression, info);
				return;
			}
			else if (outputStatements.contains(functionName)){
				setOutputStatement(ctx, paramExpression, info);
				return;
			}
		}

		if (!hasLHS){
			notifyErrorListeners("function call needs to have lvalue (Quickfix: change it to \'tmpVar = " + functionName + "(...)\')", nameToken);
			return;
		}

		DataIdentifier target = null;
		if(dataInfo instanceof DataIdentifier) {
			target = (DataIdentifier) dataInfo;
		}
		else {
			notifyErrorListeners("incorrect lvalue for function call ", targetListToken);
			return;
		}

		// For builtin functions with LHS
		if(namespace.equals(DMLProgram.DEFAULT_NAMESPACE)){
			final DataIdentifier ftarget = target;
			Action f = new Action() {
				@Override public void execute(Expression e) { setAssignmentStatement(ctx, info , ftarget, e); }
			};
			boolean validBIF = buildForBuiltInFunction(ctx, functionName, paramExpression, f);
			if (validBIF)
				return;
		}

		// If builtin functions weren't found...
		FunctionCallIdentifier functCall = new FunctionCallIdentifier(paramExpression);
		functCall.setFunctionName(functionName);
		// Override default namespace for imported non-built-in function
		String inferNamespace = (sourceNamespace != null && sourceNamespace.length() > 0 && DMLProgram.DEFAULT_NAMESPACE.equals(namespace)) ? sourceNamespace : namespace;
		functCall.setFunctionNamespace(inferNamespace);

		functCall.setAllPositions(currentFile, ctx.start.getLine(), ctx.start.getCharPositionInLine(), ctx.stop.getLine(), ctx.stop.getCharPositionInLine());

		setAssignmentStatement(ctx, info, target, functCall);
	}

	/**
	 * To allow for different actions in
	 * {@link CommonSyntacticValidator#functionCallAssignmentStatementHelper(ParserRuleContext, Set, Set, Expression, StatementInfo, Token, Token, String, String, ArrayList)}
	 */
	public static interface Action {
		public void execute(Expression e);
	}

	protected void setMultiAssignmentStatement(ArrayList<DataIdentifier> target, Expression expression, ParserRuleContext ctx, StatementInfo info) {
		info.stmt = new MultiAssignmentStatement(target, expression);
		info.stmt.setAllPositions(currentFile, ctx.start.getLine(), ctx.start.getCharPositionInLine(), ctx.start.getLine(), ctx.start.getCharPositionInLine());
		setFileLineColumn(info.stmt, ctx);
	}

	// -----------------------------------------------------------------
	// End of Helper Functions for exit*FunctionCall*AssignmentStatement
	// -----------------------------------------------------------------

}
