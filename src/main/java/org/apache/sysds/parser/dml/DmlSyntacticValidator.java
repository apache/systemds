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

package org.apache.sysds.parser.dml;

import java.io.File;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;
import java.util.Set;
import java.util.regex.Pattern;
import java.util.stream.Collectors;

import org.antlr.v4.runtime.ParserRuleContext;
import org.antlr.v4.runtime.Token;
import org.antlr.v4.runtime.tree.ErrorNode;
import org.antlr.v4.runtime.tree.TerminalNode;
import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;
import org.apache.sysds.api.DMLScript;
import org.apache.sysds.common.Builtins;
import org.apache.sysds.common.Opcodes;
import org.apache.sysds.common.Types.DataType;
import org.apache.sysds.common.Types.ValueType;
import org.apache.sysds.conf.CompilerConfig.ConfigType;
import org.apache.sysds.conf.ConfigurationManager;
import org.apache.sysds.parser.AssignmentStatement;
import org.apache.sysds.parser.BinaryExpression;
import org.apache.sysds.parser.BooleanExpression;
import org.apache.sysds.parser.BooleanIdentifier;
import org.apache.sysds.parser.BuiltinConstant;
import org.apache.sysds.parser.BuiltinFunctionExpression;
import org.apache.sysds.parser.ConditionalPredicate;
import org.apache.sysds.parser.ConstIdentifier;
import org.apache.sysds.parser.DMLProgram;
import org.apache.sysds.parser.DataExpression;
import org.apache.sysds.parser.DataIdentifier;
import org.apache.sysds.parser.DoubleIdentifier;
import org.apache.sysds.parser.Expression;
import org.apache.sysds.parser.Expression.DataOp;
import org.apache.sysds.parser.ExpressionList;
import org.apache.sysds.parser.ForStatement;
import org.apache.sysds.parser.FunctionCallIdentifier;
import org.apache.sysds.parser.FunctionDictionary;
import org.apache.sysds.parser.FunctionStatement;
import org.apache.sysds.parser.FunctionStatementBlock;
import org.apache.sysds.parser.IfStatement;
import org.apache.sysds.parser.ImportStatement;
import org.apache.sysds.parser.IndexedIdentifier;
import org.apache.sysds.parser.IntIdentifier;
import org.apache.sysds.parser.IterablePredicate;
import org.apache.sysds.parser.LanguageException;
import org.apache.sysds.parser.MultiAssignmentStatement;
import org.apache.sysds.parser.OutputStatement;
import org.apache.sysds.parser.ParForStatement;
import org.apache.sysds.parser.ParameterExpression;
import org.apache.sysds.parser.ParameterizedBuiltinFunctionExpression;
import org.apache.sysds.parser.ParseException;
import org.apache.sysds.parser.ParserWrapper;
import org.apache.sysds.parser.PathStatement;
import org.apache.sysds.parser.PrintStatement;
import org.apache.sysds.parser.RelationalExpression;
import org.apache.sysds.parser.Statement;
import org.apache.sysds.parser.StatementBlock;
import org.apache.sysds.parser.StringIdentifier;
import org.apache.sysds.parser.WhileStatement;
import org.apache.sysds.parser.dml.DmlParser.AccumulatorAssignmentStatementContext;
import org.apache.sysds.parser.dml.DmlParser.AddSubExpressionContext;
import org.apache.sysds.parser.dml.DmlParser.AssignmentStatementContext;
import org.apache.sysds.parser.dml.DmlParser.AtomicExpressionContext;
import org.apache.sysds.parser.dml.DmlParser.BooleanAndExpressionContext;
import org.apache.sysds.parser.dml.DmlParser.BooleanNotExpressionContext;
import org.apache.sysds.parser.dml.DmlParser.BooleanOrExpressionContext;
import org.apache.sysds.parser.dml.DmlParser.BuiltinFunctionExpressionContext;
import org.apache.sysds.parser.dml.DmlParser.CommandlineParamExpressionContext;
import org.apache.sysds.parser.dml.DmlParser.CommandlinePositionExpressionContext;
import org.apache.sysds.parser.dml.DmlParser.ConstDoubleIdExpressionContext;
import org.apache.sysds.parser.dml.DmlParser.ConstFalseExpressionContext;
import org.apache.sysds.parser.dml.DmlParser.ConstIntIdExpressionContext;
import org.apache.sysds.parser.dml.DmlParser.ConstStringIdExpressionContext;
import org.apache.sysds.parser.dml.DmlParser.ConstTrueExpressionContext;
import org.apache.sysds.parser.dml.DmlParser.DataIdExpressionContext;
import org.apache.sysds.parser.dml.DmlParser.DataIdentifierContext;
import org.apache.sysds.parser.dml.DmlParser.ExpressionContext;
import org.apache.sysds.parser.dml.DmlParser.ExternalFunctionDefExpressionContext;
import org.apache.sysds.parser.dml.DmlParser.ForStatementContext;
import org.apache.sysds.parser.dml.DmlParser.FunctionCallAssignmentStatementContext;
import org.apache.sysds.parser.dml.DmlParser.FunctionCallMultiAssignmentStatementContext;
import org.apache.sysds.parser.dml.DmlParser.FunctionStatementContext;
import org.apache.sysds.parser.dml.DmlParser.IfStatementContext;
import org.apache.sysds.parser.dml.DmlParser.IfdefAssignmentStatementContext;
import org.apache.sysds.parser.dml.DmlParser.ImportStatementContext;
import org.apache.sysds.parser.dml.DmlParser.IndexedExpressionContext;
import org.apache.sysds.parser.dml.DmlParser.InternalFunctionDefExpressionContext;
import org.apache.sysds.parser.dml.DmlParser.IterablePredicateColonExpressionContext;
import org.apache.sysds.parser.dml.DmlParser.IterablePredicateSeqExpressionContext;
import org.apache.sysds.parser.dml.DmlParser.MatrixDataTypeCheckContext;
import org.apache.sysds.parser.dml.DmlParser.MatrixMulExpressionContext;
import org.apache.sysds.parser.dml.DmlParser.Ml_typeContext;
import org.apache.sysds.parser.dml.DmlParser.ModIntDivExpressionContext;
import org.apache.sysds.parser.dml.DmlParser.MultDivExpressionContext;
import org.apache.sysds.parser.dml.DmlParser.MultiIdExpressionContext;
import org.apache.sysds.parser.dml.DmlParser.ParForStatementContext;
import org.apache.sysds.parser.dml.DmlParser.ParameterizedExpressionContext;
import org.apache.sysds.parser.dml.DmlParser.PathStatementContext;
import org.apache.sysds.parser.dml.DmlParser.PowerExpressionContext;
import org.apache.sysds.parser.dml.DmlParser.ProgramrootContext;
import org.apache.sysds.parser.dml.DmlParser.RelationalExpressionContext;
import org.apache.sysds.parser.dml.DmlParser.SimpleDataIdentifierExpressionContext;
import org.apache.sysds.parser.dml.DmlParser.StatementContext;
import org.apache.sysds.parser.dml.DmlParser.StrictParameterizedExpressionContext;
import org.apache.sysds.parser.dml.DmlParser.StrictParameterizedKeyValueStringContext;
import org.apache.sysds.parser.dml.DmlParser.TypedArgAssignContext;
import org.apache.sysds.parser.dml.DmlParser.TypedArgNoAssignContext;
import org.apache.sysds.parser.dml.DmlParser.UnaryExpressionContext;
import org.apache.sysds.parser.dml.DmlParser.ValueTypeContext;
import org.apache.sysds.parser.dml.DmlParser.WhileStatementContext;
import org.apache.sysds.runtime.DMLRuntimeException;
import org.apache.sysds.runtime.util.UtilFunctions;

public class DmlSyntacticValidator implements DmlListener {
	protected static final Log LOG = LogFactory.getLog(DmlSyntacticValidator.class.getName());

	private static final String DEF_WORK_DIR = ".";
	
	//externally loaded dml scripts filename (unmodified) / script
	protected static ThreadLocal<HashMap<String, String>> _tScripts = new ThreadLocal<>() {
		@Override protected HashMap<String, String> initialValue() { return new HashMap<>(); }
	};
	//imported scripts to prevent infinite recursion, modified filename / namespace
	protected static ThreadLocal<HashMap<String, String>> _f2NS = new ThreadLocal<>() {
		@Override protected HashMap<String, String> initialValue() { return new HashMap<>(); }
	};
	
	protected final CustomErrorListener errorListener;
	protected final String currentFile;
	protected String _workingDir = DEF_WORK_DIR;
	protected Map<String,String> argVals = null;
	protected String sourceNamespace = null;
	// Map namespaces to full paths as defined only from source statements in this script (i.e., currentFile)
	protected HashMap<String, String> sources;
	// Names of new internal and external functions defined in this script (i.e., currentFile)
	protected Set<String> functions;
	// DML-bodied builtin functions
	protected FunctionDictionary<FunctionStatementBlock> builtinFuns;
	// DML-bodied namespace functions (loaded via builtins)
	protected HashMap<String, FunctionDictionary<FunctionStatementBlock>> builtinFunsNs;
	
	public DmlSyntacticValidator(CustomErrorListener errorListener, Map<String,String> argVals, String sourceNamespace, Set<String> prepFunctions) {
		this.errorListener = errorListener;
		currentFile = errorListener.getCurrentFileName();
		this.argVals = argVals;
		this.sourceNamespace = sourceNamespace;
		sources = new HashMap<>();
		functions = (null != prepFunctions) ? prepFunctions : new HashSet<>();
		builtinFuns = new FunctionDictionary<>();
		builtinFunsNs = new HashMap<>();
	}


	/**
	 * Obtain the namespace separator ({@code ::} for DML 
	 * that is used to specify a namespace and a function in that namespace.
	 * 
	 * @return The namespace separator
	 */
	public String namespaceResolutionOp() {
		return "::";
	}
	
	public String trueStringLiteral() {
		return "TRUE";
	}
	public String falseStringLiteral() { 
		return "FALSE"; 
	}
	
	public FunctionDictionary<FunctionStatementBlock> getParsedBuiltinFunctions() {
		return builtinFuns;
	}
	
	public Map<String, FunctionDictionary<FunctionStatementBlock>> getParsedBuiltinFunctionsNs() {
		return builtinFunsNs;
	}
	
	protected ArrayList<ParameterExpression> getParameterExpressionList(List<ParameterizedExpressionContext> paramExprs) {
		ArrayList<ParameterExpression> retVal = new ArrayList<>();
		for(ParameterizedExpressionContext ctx : paramExprs) {
			String paramName = null;
			if(ctx.paramName != null && ctx.paramName.getText() != null && !ctx.paramName.getText().isEmpty()) {
				paramName = ctx.paramName.getText();
			}
			ParameterExpression myArg = new ParameterExpression(paramName, ctx.paramVal.info.expr);
			retVal.add(myArg);
		}
		return retVal;
	}

	@Override
	public void enterEveryRule(ParserRuleContext arg0) {
		if(arg0 instanceof StatementContext) {
			if(((StatementContext) arg0).info == null) {
				((StatementContext) arg0).info = new StatementInfo();
			}
		}
		if(arg0 instanceof FunctionStatementContext) {
			if(((FunctionStatementContext) arg0).info == null) {
				((FunctionStatementContext) arg0).info = new StatementInfo();
			}
		}
		if(arg0 instanceof ExpressionContext) {
			if(((ExpressionContext) arg0).info == null) {
				((ExpressionContext) arg0).info = new ExpressionInfo();
			}
		}
		if(arg0 instanceof DataIdentifierContext) {
			if(((DataIdentifierContext) arg0).dataInfo == null) {
				((DataIdentifierContext) arg0).dataInfo = new ExpressionInfo();
			}
		}
	}

	// -----------------------------------------------------------------
	// 			Binary, Unary & Relational Expressions
	// -----------------------------------------------------------------

	// For now do no type checking, let validation handle it.
	// This way parser doesn't have to open metadata file
	@Override
	public void exitAddSubExpression(AddSubExpressionContext ctx) {
		binaryExpressionHelper(ctx, ctx.left.info, ctx.right.info, ctx.info, ctx.op.getText());
	}

	@Override
	public void exitModIntDivExpression(ModIntDivExpressionContext ctx) {
		binaryExpressionHelper(ctx, ctx.left.info, ctx.right.info, ctx.info, ctx.op.getText());
	}

	@Override
	public void exitUnaryExpression(UnaryExpressionContext ctx) {
		unaryExpressionHelper(ctx, ctx.left.info, ctx.info, ctx.op.getText());
	}

	@Override
	public void exitMultDivExpression(MultDivExpressionContext ctx) {
		binaryExpressionHelper(ctx, ctx.left.info, ctx.right.info, ctx.info, ctx.op.getText());
	}
	@Override
	public void exitPowerExpression(PowerExpressionContext ctx) {
		binaryExpressionHelper(ctx, ctx.left.info, ctx.right.info, ctx.info, ctx.op.getText());
	}

	@Override
	public void exitMatrixMulExpression(MatrixMulExpressionContext ctx) {
		binaryExpressionHelper(ctx, ctx.left.info, ctx.right.info, ctx.info, ctx.op.getText());
	}

	@Override
	public void exitRelationalExpression(RelationalExpressionContext ctx) {
		relationalExpressionHelper(ctx, ctx.left.info, ctx.right.info, ctx.info, ctx.op.getText());
	}

	@Override
	public void exitBooleanAndExpression(BooleanAndExpressionContext ctx) {
		booleanExpressionHelper(ctx, ctx.left.info, ctx.right.info, ctx.info, ctx.op.getText());
	}

	@Override
	public void exitBooleanOrExpression(BooleanOrExpressionContext ctx) {
		booleanExpressionHelper(ctx, ctx.left.info, ctx.right.info, ctx.info, ctx.op.getText());
	}

	@Override
	public void exitBooleanNotExpression(BooleanNotExpressionContext ctx) {
		unaryBooleanExpressionHelper(ctx, ctx.left.info, ctx.info, ctx.op.getText());
	}

	@Override
	public void exitAtomicExpression(AtomicExpressionContext ctx) {
		ctx.info.expr = ctx.left.info.expr;
		setFileLineColumn(ctx.info.expr, ctx);
	}

	// -----------------------------------------------------------------
	// 			Constant Expressions
	// -----------------------------------------------------------------

	@Override
	public void exitConstFalseExpression(ConstFalseExpressionContext ctx) {
		booleanIdentifierHelper(ctx, false, ctx.info);
	}

	@Override
	public void exitConstTrueExpression(ConstTrueExpressionContext ctx) {
		booleanIdentifierHelper(ctx, true, ctx.info);
	}

	@Override
	public void exitConstDoubleIdExpression(ConstDoubleIdExpressionContext ctx) {
		constDoubleIdExpressionHelper(ctx, ctx.info);
	}

	@Override
	public void exitConstIntIdExpression(ConstIntIdExpressionContext ctx) {
		constIntIdExpressionHelper(ctx, ctx.info);
	}

	@Override
	public void exitConstStringIdExpression(ConstStringIdExpressionContext ctx) {
		constStringIdExpressionHelper(ctx, ctx.info);
	}


	// -----------------------------------------------------------------
	//          Identifier Based Expressions
	// -----------------------------------------------------------------

	@Override
	public void exitDataIdExpression(DataIdExpressionContext ctx) {
		exitDataIdExpressionHelper(ctx, ctx.info, ctx.dataIdentifier().dataInfo);
	}

	@Override
	public void exitSimpleDataIdentifierExpression(SimpleDataIdentifierExpressionContext ctx) {
		// This is either a function, or variable with namespace
		// By default, it assigns to a data type
		ctx.dataInfo.expr = new DataIdentifier(ctx.getText());
		setFileLineColumn(ctx.dataInfo.expr, ctx);
	}

	/**
	 * DML uses 1-based indexing.;
	 *
	 * @param ctx the parse tree
	 */
	@Override
	public void exitIndexedExpression(IndexedExpressionContext ctx) {
		boolean isRowLower = (ctx.rowLower != null && !ctx.rowLower.isEmpty() && (ctx.rowLower.info.expr != null));
		boolean isRowUpper = (ctx.rowUpper != null && !ctx.rowUpper.isEmpty() && (ctx.rowUpper.info.expr != null));
		boolean isColLower = (ctx.colLower != null && !ctx.colLower.isEmpty() && (ctx.colLower.info.expr != null));
		boolean isColUpper = (ctx.colUpper != null && !ctx.colUpper.isEmpty() && (ctx.colUpper.info.expr != null));
		ExpressionInfo rowLower = isRowLower ? ctx.rowLower.info : null;
		ExpressionInfo rowUpper = isRowUpper ? ctx.rowUpper.info : null;
		ExpressionInfo colLower = isColLower ? ctx.colLower.info : null;
		ExpressionInfo colUpper = isColUpper ? ctx.colUpper.info : null;

		ctx.dataInfo.expr = new IndexedIdentifier(ctx.name.getText(), false, false);
		setFileLineColumn(ctx.dataInfo.expr, ctx);

		try {
			ArrayList< ArrayList<Expression> > exprList = new ArrayList<>();

			ArrayList<Expression> rowIndices = new ArrayList<>();
			ArrayList<Expression> colIndices = new ArrayList<>();


			if(!isRowLower && !isRowUpper) {
				// both not set
				rowIndices.add(null); rowIndices.add(null);
			}
			else if(isRowLower && isRowUpper) {
				// both set
				rowIndices.add(rowLower.expr);
				rowIndices.add(rowUpper.expr);
			}
			else if(isRowLower && !isRowUpper) {
				// only row set
				rowIndices.add(rowLower.expr);
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
				colIndices.add(colLower.expr);
				colIndices.add(colUpper.expr);
			}
			else if(isColLower && !isColUpper) {
				colIndices.add(colLower.expr);
			}
			else {
				notifyErrorListeners("incorrect index expression for column", ctx.start);
				return;
			}
			exprList.add(rowIndices);
			exprList.add(colIndices);
			((IndexedIdentifier) ctx.dataInfo.expr).setIndices(exprList);
		}
		catch(Exception e) {
			notifyErrorListeners("cannot set the indices", ctx.start);
			return;
		}
	}


	// -----------------------------------------------------------------
	//          Command line parameters (begin with a '$')
	// -----------------------------------------------------------------

	@Override
	public void exitCommandlineParamExpression(CommandlineParamExpressionContext ctx) {
		handleCommandlineArgumentExpression(ctx);
	}

	@Override
	public void exitCommandlinePositionExpression(CommandlinePositionExpressionContext ctx) {
		handleCommandlineArgumentExpression(ctx);
	}

	private void handleCommandlineArgumentExpression(DataIdentifierContext ctx)
	{
		String varName = ctx.getText().trim();
		fillExpressionInfoCommandLineParameters(ctx, varName, ctx.dataInfo);

		if(ctx.dataInfo.expr == null) {
			if(!(ctx.parent instanceof IfdefAssignmentStatementContext)) {
				String msg = "The parameter " + varName + " either needs to be passed "
						+ "through commandline or initialized to default value.";
				if( ConfigurationManager.getCompilerConfigFlag(ConfigType.IGNORE_UNSPECIFIED_ARGS) ) {
					ctx.dataInfo.expr = getConstIdFromString(ctx, " ");
					if (!ConfigurationManager.getCompilerConfigFlag(ConfigType.MLCONTEXT)) {
						raiseWarning(msg, ctx.start);
					}
				}
				else {
					notifyErrorListeners(msg, ctx.start);
				}
			}
		}
	}


	// -----------------------------------------------------------------
	// 			"source" statement
	// -----------------------------------------------------------------

	@Override
	public void exitImportStatement(ImportStatementContext ctx) {
		String filePath = getWorkingFilePath(UtilFunctions.unquote(ctx.filePath.getText()));
		String namespace = getNamespaceSafe(ctx.namespace);
		setupContextInfo(ctx.info, namespace, filePath, ctx.filePath.getText(),
			parseAndAddImportedFunctions(namespace, filePath, ctx));
	}

	// -----------------------------------------------------------------
	// 			Assignment Statement
	// -----------------------------------------------------------------

	@Override
	public void exitAssignmentStatement(AssignmentStatementContext ctx) {
		if(ctx.targetList == null) {
			notifyErrorListeners("incorrect parsing for assignment", ctx.start);
			return;
		}
		exitAssignmentStatementHelper(ctx, ctx.targetList.getText(), ctx.targetList.dataInfo, ctx.targetList.start, ctx.source.info, ctx.info);
	}


	// -----------------------------------------------------------------
	// 			Control Statements - Guards & Loops
	// -----------------------------------------------------------------

	public ConvertedDMLSyntax convertToDMLSyntax(ParserRuleContext ctx, String namespace, String functionName, ArrayList<ParameterExpression> paramExpression, Token fnName) {
		return new ConvertedDMLSyntax(namespace, functionName, paramExpression);
	}

	protected Expression handleLanguageSpecificFunction(ParserRuleContext ctx, String functionName,
			ArrayList<ParameterExpression> paramExpressions) {
		return null;
	}


	@Override
	public void exitFunctionCallAssignmentStatement(FunctionCallAssignmentStatementContext ctx) {

		Set<String> printStatements = new  HashSet<>();
		printStatements.add(Opcodes.PRINT.toString());
		printStatements.add(Opcodes.STOP.toString());
		printStatements.add(Opcodes.ASSERT.toString());

		Set<String> outputStatements = new HashSet<>();
		outputStatements.add("write");

		String [] fnNames = getQualifiedNames (ctx.name.getText());
		if (fnNames == null) {
			String errorMsg = "incorrect function name (only namespace " + namespaceResolutionOp() + " functionName allowed. Hint: If you are trying to use builtin functions, you can skip the namespace)";
			notifyErrorListeners(errorMsg, ctx.name);
			return;
		}
		String namespace = fnNames[0];
		String functionName = fnNames[1];
		ArrayList<ParameterExpression> paramExpression = getParameterExpressionList(ctx.paramExprs);

		castAsScalarDeprecationCheck(functionName, ctx);

		boolean hasLHS = ctx.targetList != null;
		functionCallAssignmentStatementHelper(ctx, printStatements, outputStatements, hasLHS ? ctx.targetList.dataInfo.expr : null, ctx.info, ctx.name,
			hasLHS ? ctx.targetList.start : null, namespace, functionName, paramExpression, hasLHS);
	}

	// TODO: remove this when castAsScalar has been removed from DML/PYDML
	private void castAsScalarDeprecationCheck(String functionName, ParserRuleContext ctx) {
		if ("castAsScalar".equalsIgnoreCase(functionName)) {
			raiseWarning("castAsScalar() has been deprecated. Please use as.scalar().", ctx.start);
		}
	}

	@Override
	public void exitBuiltinFunctionExpression(BuiltinFunctionExpressionContext ctx) {
		// Double verification: verify passed function name is a (non-parameterized) built-in function.
		String[] names = getQualifiedNames(ctx.name.getText());
		if(names == null) {
			notifyErrorListeners("incorrect function name (only namespace " + namespaceResolutionOp() + " functionName allowed. Hint: If you are trying to use builtin functions, you can skip the namespace)", ctx.name);
			return;
		}
		String namespace = names[0];
		String functionName = names[1];

		ArrayList<ParameterExpression> paramExpression = getParameterExpressionList(ctx.paramExprs);
		castAsScalarDeprecationCheck(functionName, ctx);

		ConvertedDMLSyntax convertedSyntax = convertToDMLSyntax(ctx, namespace, functionName, paramExpression, ctx.name);
		if(convertedSyntax == null) {
			return;
		}
		else {
			functionName = convertedSyntax.functionName;
			paramExpression = convertedSyntax.paramExpression;
		}
		
		// handle built-in functions
		ctx.info.expr = buildForBuiltInFunction(ctx, functionName, paramExpression);
		if( ctx.info.expr != null )
			return;
		
		// handle user-defined functions
		ctx.info.expr = createFunctionCall(ctx, namespace, functionName, paramExpression);
	}

	@Override
	public void exitFunctionCallMultiAssignmentStatement(FunctionCallMultiAssignmentStatementContext ctx) {
		if( ctx.name == null )
			throw new ParseException("Missing name of multi-assignment function call (see parser issues above).");
		String[] names = getQualifiedNames(ctx.name.getText());
		if(names == null) {
			notifyErrorListeners("incorrect function name (only namespace::functionName allowed. Hint: If you are trying to use builtin functions, you can skip the namespace)", ctx.name);
			return;
		}
		String namespace = names[0];
		String functionName = names[1];

		ArrayList<ParameterExpression> paramExpression = getParameterExpressionList(ctx.paramExprs);
		ConvertedDMLSyntax convertedSyntax = convertToDMLSyntax(ctx, namespace, functionName, paramExpression, ctx.name);
		if(convertedSyntax == null) {
			return;
		}
		else {
			namespace = convertedSyntax.namespace;
			functionName = convertedSyntax.functionName;
			paramExpression = convertedSyntax.paramExpression;
		}

		FunctionCallIdentifier functCall = new FunctionCallIdentifier(paramExpression);
		functCall.setFunctionName(functionName);
		functCall.setFunctionNamespace(namespace);

		final ArrayList<DataIdentifier> targetList = new ArrayList<>();
		for(DataIdentifierContext dataCtx : ctx.targetList) {
			if(dataCtx.dataInfo.expr instanceof DataIdentifier) {
				targetList.add((DataIdentifier) dataCtx.dataInfo.expr);
			}
			else {
				notifyErrorListeners("incorrect type for variable ", dataCtx.start);
				return;
			}
		}

		if(namespace.equals(DMLProgram.DEFAULT_NAMESPACE)) {
			Expression e = buildForBuiltInFunction(ctx, functionName, paramExpression);
			if( e != null ) {
				setMultiAssignmentStatement(targetList, e, ctx, ctx.info);
				return;
			}
			handleDMLBodiedBuiltinFunction(functionName, DMLProgram.BUILTIN_NAMESPACE, ctx);
		}

		// Override default namespace for imported non-built-in function
		String inferNamespace = (sourceNamespace != null && sourceNamespace.length() > 0 
			&& DMLProgram.DEFAULT_NAMESPACE.equals(namespace)) ? sourceNamespace : namespace;
		functCall.setFunctionNamespace(inferNamespace);

		setMultiAssignmentStatement(targetList, functCall, ctx, ctx.info);
	}

	private void handleDMLBodiedBuiltinFunction(String functionName, String namespace, ParserRuleContext ctx) {
		if( Builtins.contains(functionName, true, false) 
			&& !builtinFuns.containsFunction(functionName) )
		{
			//load and add builtin DML-bodied functions
			String filePath = Builtins.getFilePath(functionName);
			DMLProgram tmpProg = parseAndAddImportedFunctions(namespace, filePath, ctx);
			if(tmpProg == null){
				//a parse error occured, which was logged in the error listener and will be shown in the error message
				return;
			}
			FunctionDictionary<FunctionStatementBlock> prog = tmpProg.getBuiltinFunctionDictionary();
			if( prog != null ) { //robustness for existing functions
				//add builtin functions
				for( Entry<String,FunctionStatementBlock> f : prog.getFunctions().entrySet() )
					builtinFuns.addFunction(f.getKey(), f.getValue());
				//add namespaces loaded by builtin functions (via source)
				tmpProg.getNamespaces().entrySet().stream()
					.filter(e -> !e.getKey().equals(DMLProgram.BUILTIN_NAMESPACE))
					.forEach(e -> {
						String ns = getQualifiedNamespace(e.getKey());
						if( builtinFunsNs.containsKey(ns) )
							builtinFunsNs.get(ns).merge(e.getValue());
						else
							builtinFunsNs.put(ns, e.getValue());
					});
			}
		}
	}

	public static Map<String,FunctionStatementBlock> loadAndParseBuiltinFunction(String name, String namespace, boolean forced) {
		if( !Builtins.contains(name, true, false) ) {
			throw new DMLRuntimeException("Function "
				+ DMLProgram.constructFunctionKey(namespace, name)+" is not a builtin function.");
		}
		//load and add builtin DML-bodied functions (via tmp validator instance)
		//including nested builtin function calls unless already loaded
		DmlSyntacticValidator tmp = new DmlSyntacticValidator(
			new CustomErrorListener(), new HashMap<>(), namespace, new HashSet<>());
		String filePath = Builtins.getFilePath(name);
		FunctionDictionary<FunctionStatementBlock> dict = tmp
			.parseAndAddImportedFunctions(namespace, filePath, null, forced)
			.getBuiltinFunctionDictionary();
		
		//construct output map of all functions
		if(dict == null)
			throw new RuntimeException("Failed function load: " + name + " " + namespace + "\n" + filePath);
		return dict.getFunctions();
	}

	// -----------------------------------------------------------------
	//   Control Statements - Guards & Loops
	// -----------------------------------------------------------------

	private static StatementBlock getStatementBlock(Statement current) {
		return ParserWrapper.getStatementBlock(current);
	}

	@Override
	public void exitIfStatement(IfStatementContext ctx) {
		IfStatement ifStmt = new IfStatement();
		ConditionalPredicate predicate = new ConditionalPredicate(ctx.predicate.info.expr);
		ifStmt.setConditionalPredicate(predicate);
		ifStmt.setCtxValuesAndFilename(ctx, currentFile);
		
		if(ctx.ifBody.size() > 0) {
			for(StatementContext stmtCtx : ctx.ifBody)
				ifStmt.addStatementBlockIfBody(getStatementBlock(stmtCtx.info.stmt));
			ifStmt.mergeStatementBlocksIfBody();
		}

		if(ctx.elseBody.size() > 0) {
			for(StatementContext stmtCtx : ctx.elseBody)
				ifStmt.addStatementBlockElseBody(getStatementBlock(stmtCtx.info.stmt));
			ifStmt.mergeStatementBlocksElseBody();
		}

		ctx.info.stmt = ifStmt;
		setFileLineColumn(ctx.info.stmt, ctx);
	}

	@Override
	public void exitWhileStatement(WhileStatementContext ctx) {
		WhileStatement whileStmt = new WhileStatement();
		ConditionalPredicate predicate = new ConditionalPredicate(ctx.predicate.info.expr);
		whileStmt.setPredicate(predicate);
		whileStmt.setCtxValuesAndFilename(ctx, currentFile);

		if(ctx.body.size() > 0) {
			for(StatementContext stmtCtx : ctx.body)
				whileStmt.addStatementBlock(getStatementBlock(stmtCtx.info.stmt));
			whileStmt.mergeStatementBlocks();
		}

		ctx.info.stmt = whileStmt;
		setFileLineColumn(ctx.info.stmt, ctx);
	}

	@Override
	public void exitForStatement(ForStatementContext ctx) {
		ForStatement forStmt = new ForStatement();

		DataIdentifier iterVar = new DataIdentifier(ctx.iterVar.getText());
		HashMap<String, String> parForParamValues = null;
		Expression incrementExpr = null; //1/-1
		if(ctx.iterPred.info.increment != null) {
			incrementExpr = ctx.iterPred.info.increment;
		}
		IterablePredicate predicate = new IterablePredicate(ctx, iterVar, ctx.iterPred.info.from, ctx.iterPred.info.to,
				incrementExpr, parForParamValues, currentFile);
		forStmt.setPredicate(predicate);

		if(ctx.body.size() > 0) {
			for(StatementContext stmtCtx : ctx.body)
				forStmt.addStatementBlock(getStatementBlock(stmtCtx.info.stmt));
			forStmt.mergeStatementBlocks();
		}
		ctx.info.stmt = forStmt;
	}

	@Override
	public void exitParForStatement(ParForStatementContext ctx) {
		ParForStatement parForStmt = new ParForStatement();

		DataIdentifier iterVar = new DataIdentifier(ctx.iterVar.getText());
		HashMap<String, String> parForParamValues = new HashMap<>();
		if(ctx.parForParams != null && ctx.parForParams.size() > 0) {
			for(StrictParameterizedExpressionContext parForParamCtx : ctx.parForParams) {
				String paramVal = parForParamCtx.paramVal.getText();
				if( argVals.containsKey(paramVal) )
					paramVal = argVals.get(paramVal);
				parForParamValues.put(parForParamCtx.paramName.getText(), paramVal);
			}
		}

		Expression incrementExpr = null; //1/-1
		if( ctx.iterPred.info.increment != null ) {
			incrementExpr = ctx.iterPred.info.increment;
		}
		IterablePredicate predicate = new IterablePredicate(ctx, iterVar, ctx.iterPred.info.from, ctx.iterPred.info.to,
				incrementExpr, parForParamValues, currentFile);
		parForStmt.setPredicate(predicate);
		if(ctx.body.size() > 0) {
			for(StatementContext stmtCtx : ctx.body)
				parForStmt.addStatementBlock(getStatementBlock(stmtCtx.info.stmt));
			parForStmt.mergeStatementBlocks();
		}
		ctx.info.stmt = parForStmt;
	}

	private ArrayList<DataIdentifier> getFunctionParametersNoAssign(List<TypedArgNoAssignContext> ctx) {
		ArrayList<DataIdentifier> retVal = new ArrayList<>(ctx.size());
		for(TypedArgNoAssignContext paramCtx : ctx) {
			DataIdentifier dataId = new DataIdentifier(paramCtx.paramName.getText());
			String dataType = (paramCtx.paramType == null || paramCtx.paramType.dataType() == null
				|| paramCtx.paramType.dataType().getText() == null || paramCtx.paramType.dataType().getText().isEmpty()) ?
				"scalar" : paramCtx.paramType.dataType().getText();
			String valueType = paramCtx.paramType.valueType().getText();
			
			//check and assign data type
			checkValidDataType(dataType, paramCtx.start);
			if( !setDataAndValueType(dataId, dataType, valueType, paramCtx.start, false, true) )
				return null;
			retVal.add(dataId);
		}
		return retVal;
	}
	
	private ArrayList<DataIdentifier> getFunctionParametersAssign(List<TypedArgAssignContext> ctx) {
		ArrayList<DataIdentifier> retVal = new ArrayList<>(ctx.size());
		for(TypedArgAssignContext paramCtx : ctx) {
			DataIdentifier dataId = new DataIdentifier(paramCtx.paramName.getText());
			String dataType = (paramCtx.paramType == null || paramCtx.paramType.dataType() == null
				|| paramCtx.paramType.dataType().getText() == null || paramCtx.paramType.dataType().getText().isEmpty()) ?
				"scalar" : paramCtx.paramType.dataType().getText();
			String valueType = paramCtx.paramType.valueType().getText();
			
			//check and assign data type
			checkValidDataType(dataType, paramCtx.start);
			if( !setDataAndValueType(dataId, dataType, valueType, paramCtx.start, false, true) )
				return null;
			retVal.add(dataId);
		}
		return retVal;
	}
	
	private static ArrayList<Expression> getFunctionDefaults(List<TypedArgAssignContext> ctx) {
		return new ArrayList<>(ctx.stream().map(arg -> 
			(arg.paramVal!=null)?arg.paramVal.info.expr:null).collect(Collectors.toList()));
	}

	@Override
	public void exitIterablePredicateColonExpression(IterablePredicateColonExpressionContext ctx) {
		ctx.info.from = ctx.from.info.expr;
		if( ctx.to == null ) {
			notifyErrorListeners("incorrect for/parfor loop bounds: \'" + ctx.info.from + " : "+ctx.info.to+"\'.", ctx.start);
			return;
		}
		ctx.info.to = ctx.to.info.expr;
		ctx.info.increment = null;
	}

	@Override
	public void exitIterablePredicateSeqExpression(IterablePredicateSeqExpressionContext ctx) {
		if(!ctx.ID().getText().equals("seq")) {
			notifyErrorListeners("incorrect function:\'" + ctx.ID().getText() + "\'. expected \'seq\'", ctx.start);
			return;
		}
		ctx.info.from = ctx.from.info.expr;
		ctx.info.to = ctx.to.info.expr;
		if(ctx.increment != null && ctx.increment.info != null)
			ctx.info.increment = ctx.increment.info.expr;
	}


	// -----------------------------------------------------------------
	//            Internal & External Functions Definitions
	// -----------------------------------------------------------------

	@Override
	public void exitInternalFunctionDefExpression(InternalFunctionDefExpressionContext ctx) {
		//populate function statement
		FunctionStatement functionStmt = new FunctionStatement();
		functionStmt.setName(ctx.name.getText());
		functionStmt.setInputParams(getFunctionParametersAssign(ctx.inputParams));
		functionStmt.setInputDefaults(getFunctionDefaults(ctx.inputParams));
		functionStmt.setOutputParams(getFunctionParametersNoAssign(ctx.outputParams));
		
		if(ctx.body.size() > 0) {
			// handle function body
			// Create arraylist of one statement block
			ArrayList<StatementBlock> body = new ArrayList<>();
			for(StatementContext stmtCtx : ctx.body) {
				body.add(getStatementBlock(stmtCtx.info.stmt));
			}
			functionStmt.setBody(body);
			functionStmt.mergeStatementBlocks();
		}
		else {
			notifyErrorListeners("functions with no statements are not allowed", ctx.start);
			return;
		}

		ctx.info.stmt = functionStmt;
		setFileLineColumn(ctx.info.stmt, ctx);
		ctx.info.functionName = ctx.name.getText();
	}
	
	@Override
	public void exitPathStatement(PathStatementContext ctx) {
		PathStatement stmt = new PathStatement(ctx.pathValue.getText());
		String filePath = UtilFunctions.unquote(ctx.pathValue.getText());
		_workingDir = filePath;
		ctx.info.stmt = stmt;
	}

	@Override
	public void exitIfdefAssignmentStatement(IfdefAssignmentStatementContext ctx) {
		if(!ctx.commandLineParam.getText().startsWith("$")) {
			notifyErrorListeners("the first argument of ifdef function should be a commandline argument parameter (which starts with $)", ctx.commandLineParam.start);
			return;
		}

		if(ctx.targetList == null) {
			notifyErrorListeners("ifdef assignment needs an lvalue ", ctx.start);
			return;
		}
		String targetListText = ctx.targetList.getText();
		if(targetListText.startsWith("$")) {
			notifyErrorListeners("lhs of ifdef function cannot be a commandline parameters. Use local variable instead", ctx.start);
			return;
		}

		DataIdentifier target = null;
		if(ctx.targetList.dataInfo.expr instanceof DataIdentifier) {
			target = (DataIdentifier) ctx.targetList.dataInfo.expr;
			Expression source = null;
			if(ctx.commandLineParam.dataInfo.expr != null) {
				// Since commandline parameter is set
				// The check of following is done in fillExpressionInfoCommandLineParameters:
				// Command line param cannot be empty string
				// If you want to pass space, please quote it
				source = ctx.commandLineParam.dataInfo.expr;
			}
			else {
				source = ctx.source.info.expr;
			}

			try {
				ctx.info.stmt = new AssignmentStatement(ctx, target, source, currentFile);
			} catch (LanguageException e) {
				notifyErrorListeners("invalid assignment for ifdef function", ctx.targetList.start);
				return;
			}

		}
		else {
			notifyErrorListeners("incorrect lvalue in ifdef function ", ctx.targetList.start);
			return;
		}
	}

	@Override
	public void exitAccumulatorAssignmentStatement(AccumulatorAssignmentStatementContext ctx) {
		if(ctx.targetList == null) {
			notifyErrorListeners("incorrect parsing for accumulator assignment", ctx.start);
			return;
		}
		//process as default assignment statement
		exitAssignmentStatementHelper(ctx, ctx.targetList.getText(),
			ctx.targetList.dataInfo, ctx.targetList.start, ctx.source.info, ctx.info);
		//mark as accumulator
		((AssignmentStatement)ctx.info.stmt).setAccumulator(true);
	}
	
	@Override
	public void exitMatrixDataTypeCheck(MatrixDataTypeCheckContext ctx) {
		checkValidDataType(ctx.ID().getText(), ctx.start);
	}


	// -----------------------------------------------------------------
	// 			Not overridden
	// -----------------------------------------------------------------

	@Override public void visitTerminal(TerminalNode node) {}

	@Override public void visitErrorNode(ErrorNode node) {}

	@Override public void exitEveryRule(ParserRuleContext ctx) {}

	@Override public void enterModIntDivExpression(ModIntDivExpressionContext ctx) {}

	@Override public void enterExternalFunctionDefExpression(ExternalFunctionDefExpressionContext ctx) {}

	@Override public void enterBooleanNotExpression(BooleanNotExpressionContext ctx) {}

	@Override public void enterPowerExpression(PowerExpressionContext ctx) {}

	@Override public void enterInternalFunctionDefExpression(InternalFunctionDefExpressionContext ctx) {}

	@Override public void enterBuiltinFunctionExpression(BuiltinFunctionExpressionContext ctx) {}

	@Override public void enterConstIntIdExpression(ConstIntIdExpressionContext ctx) {}

	@Override public void enterAtomicExpression(AtomicExpressionContext ctx) {}

	@Override public void enterIfdefAssignmentStatement(IfdefAssignmentStatementContext ctx) {}

	@Override public void enterAccumulatorAssignmentStatement(AccumulatorAssignmentStatementContext ctx) {}
	
	@Override public void enterConstStringIdExpression(ConstStringIdExpressionContext ctx) {}

	@Override public void enterConstTrueExpression(ConstTrueExpressionContext ctx) {}

	@Override public void enterParForStatement(ParForStatementContext ctx) {}

	@Override public void enterUnaryExpression(UnaryExpressionContext ctx) {}

	@Override public void enterImportStatement(ImportStatementContext ctx) {}

	@Override public void enterPathStatement(PathStatementContext ctx) {}

	@Override public void enterWhileStatement(WhileStatementContext ctx) {}

	@Override public void enterCommandlineParamExpression(CommandlineParamExpressionContext ctx) {}

	@Override public void enterFunctionCallAssignmentStatement(FunctionCallAssignmentStatementContext ctx) {}

	@Override public void enterAddSubExpression(AddSubExpressionContext ctx) {}

	@Override public void enterIfStatement(IfStatementContext ctx) {}

	@Override public void enterConstDoubleIdExpression(ConstDoubleIdExpressionContext ctx) {}

	@Override public void enterMatrixMulExpression(MatrixMulExpressionContext ctx) {}

	@Override public void enterMatrixDataTypeCheck(MatrixDataTypeCheckContext ctx) {}

	@Override public void enterCommandlinePositionExpression(CommandlinePositionExpressionContext ctx) {}

	@Override public void enterIterablePredicateColonExpression(IterablePredicateColonExpressionContext ctx) {}

	@Override public void enterAssignmentStatement(AssignmentStatementContext ctx) {}

	@Override public void enterValueType(ValueTypeContext ctx) {}

	@Override public void exitValueType(ValueTypeContext ctx) {}

	@Override public void enterMl_type(Ml_typeContext ctx) {}

	@Override public void exitMl_type(Ml_typeContext ctx) {}

	@Override public void enterBooleanAndExpression(BooleanAndExpressionContext ctx) {}

	@Override public void enterForStatement(ForStatementContext ctx) {}

	@Override public void enterRelationalExpression(RelationalExpressionContext ctx) {}

	@Override public void enterTypedArgNoAssign(TypedArgNoAssignContext ctx) {}

	@Override public void exitTypedArgNoAssign(TypedArgNoAssignContext ctx) {}

	@Override public void enterTypedArgAssign(TypedArgAssignContext ctx) {}

	@Override public void exitTypedArgAssign(TypedArgAssignContext ctx) {}

	@Override public void enterStrictParameterizedExpression(StrictParameterizedExpressionContext ctx) {}

	@Override public void exitStrictParameterizedExpression(StrictParameterizedExpressionContext ctx) {}

	@Override public void enterMultDivExpression(MultDivExpressionContext ctx) {}

	@Override public void enterConstFalseExpression(ConstFalseExpressionContext ctx) {}

	@Override public void enterStrictParameterizedKeyValueString(StrictParameterizedKeyValueStringContext ctx) {}

	@Override public void exitStrictParameterizedKeyValueString(StrictParameterizedKeyValueStringContext ctx) {}

	@Override public void enterProgramroot(ProgramrootContext ctx) {}

	@Override 
	public void exitProgramroot(ProgramrootContext ctx) {}

	@Override public void enterDataIdExpression(DataIdExpressionContext ctx) {}

	@Override public void enterIndexedExpression(IndexedExpressionContext ctx) {}

	@Override public void enterParameterizedExpression(ParameterizedExpressionContext ctx) {}

	@Override public void exitParameterizedExpression(ParameterizedExpressionContext ctx) {}

	@Override public void enterFunctionCallMultiAssignmentStatement(FunctionCallMultiAssignmentStatementContext ctx) {}

	@Override public void enterIterablePredicateSeqExpression(IterablePredicateSeqExpressionContext ctx) {}

	@Override public void enterSimpleDataIdentifierExpression(SimpleDataIdentifierExpressionContext ctx) {}

	@Override public void enterBooleanOrExpression(BooleanOrExpressionContext ctx) {}

	@Override
	public void enterMultiIdExpression(MultiIdExpressionContext ctx) { }

	@Override
	public void exitMultiIdExpression(MultiIdExpressionContext ctx) {
		ArrayList<Expression> values = new ArrayList<>();
		for(ExpressionContext elem : ctx.targetList) {
			values.add(elem.info.expr);
		}
		ctx.info.expr = new ExpressionList(values);
	}

	@Override
	public void exitExternalFunctionDefExpression(ExternalFunctionDefExpressionContext ctx) {
		// TODO Auto-generated method stub
		
	}

	// internal helpers
	
	public static void init() {
		_f2NS.get().clear();
	}
	
	public static void init(Map<String, String> scripts) {
		_f2NS.get().clear();
		_tScripts.get().clear();
		for( Entry<String,String> e : scripts.entrySet() )
			_tScripts.get().put(getDefWorkingFilePath(e.getKey()), e.getValue());
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
	
	public String getWorkingFilePath(String filePath) {
		return getWorkingFilePath(filePath, _workingDir);
	}
	
	public static String getDefWorkingFilePath(String filePath) {
		return getWorkingFilePath(filePath, DEF_WORK_DIR);
	}
	
	private static String getWorkingFilePath(String filePath, String workingDir) {
		//NOTE: the use of File.separator would lead to OS-specific inconsistencies,
		//which is problematic for second order functions such as eval or paramserv.
		//Since this is unnecessary, we now use "/" independent of the use OS.
		String prefix = workingDir + "/";
		return new File(filePath).isAbsolute() | filePath.startsWith(prefix) ?
			filePath : prefix + filePath;
	}
	
	public String getNamespaceSafe(Token ns) {
		return (ns != null && ns.getText() != null && !ns.getText().isEmpty()) ?
			ns.getText() : DMLProgram.DEFAULT_NAMESPACE;
	}

	protected void validateNamespace(String namespace, String filePath, ParserRuleContext ctx) {
		// error out if different scripts from different file paths are bound to the same namespace
		if( !DMLProgram.isInternalNamespace(namespace) ) {
			if( sources.containsKey(namespace) && !sources.get(namespace).equals(filePath) )
				notifyErrorListeners("Namespace Conflict: '" + namespace 
					+ "' already defined as " + sources.get(namespace), ctx.start);
			else
				sources.put(namespace, filePath);
		}
	}
	
	protected void setupContextInfo(StatementInfo info, String namespace, 
		String filePath, String filePath2, DMLProgram prog ) {
		info.namespaces = new HashMap<>();
		if(prog != null) {
			//add loaded namespaces (imported namespace already w/ correct name, not default)
			for( Entry<String, FunctionDictionary<FunctionStatementBlock>> e : prog.getNamespaces().entrySet() )
				info.namespaces.put(getQualifiedNamespace(e.getKey()), e.getValue());
			ImportStatement istmt = new ImportStatement();
			istmt.setCompletePath(filePath);
			istmt.setFilename(filePath2);
			istmt.setNamespace(namespace);
			info.stmt = istmt;
		}
	}

	protected void setFileLineColumn(Expression expr, ParserRuleContext ctx) {
		expr.setCtxValuesAndFilename(ctx, currentFile);
	}

	protected void setFileLineColumn(Statement stmt, ParserRuleContext ctx) {
		stmt.setCtxValuesAndFilename(ctx, currentFile);
	}

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
			if (Opcodes.STOP.toString().equals(functionName)) {
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
			handleDMLBodiedBuiltinFunction(functionName, DMLProgram.BUILTIN_NAMESPACE, ctx);
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
		boolean validMatrixType = datatype.equals("matrix") || datatype.equals("Matrix")
			|| datatype.equals("frame") || datatype.equals("Frame")
			|| datatype.equals("list") || datatype.equals("List")
			|| datatype.equals("scalar") || datatype.equals("Scalar");
		if( !validMatrixType )
			notifyErrorListeners("incorrect datatype (expected matrix, frame, list, or scalar)", start);
	}
	
	protected boolean setDataAndValueType(DataIdentifier dataId, String dataType, String valueType, Token start, boolean shortVt, boolean helpBool) {
		if( dataType.equalsIgnoreCase("matrix") )
			dataId.setDataType(DataType.MATRIX);
		else if( dataType.equalsIgnoreCase("frame") )
			dataId.setDataType(DataType.FRAME);
		else if( dataType.equalsIgnoreCase("list") )
			dataId.setDataType(DataType.LIST);
		else if( dataType.equalsIgnoreCase("scalar") )
			dataId.setDataType(DataType.SCALAR);

		if( (shortVt && valueType.equals("int"))
			|| valueType.equals("int") || valueType.equals("integer")
			|| valueType.equals("Int") || valueType.equals("Integer")) {
			dataId.setValueType(ValueType.INT64);
		}
		else if( (shortVt && valueType.equals("str"))
			|| valueType.equals("string") || valueType.equals("String")) {
			dataId.setValueType(ValueType.STRING);
		}
		else if( (shortVt && valueType.equals("bool"))
			|| valueType.equals("boolean") || valueType.equals("Boolean")) {
			dataId.setValueType(ValueType.BOOLEAN);
		}
		else if( (shortVt && valueType.equals("float") )
			|| valueType.equals("double") || valueType.equals("Double")) {
			dataId.setValueType(ValueType.FP64);
		}
		else if(valueType.equals("unknown") || (!shortVt && valueType.equals("Unknown"))) {
			dataId.setValueType(ValueType.UNKNOWN);
		}
		else if(helpBool && valueType.equals("bool")) {
			notifyErrorListeners("invalid valuetype " + valueType + " (Quickfix: use \'boolean\' instead)", start);
			return false;
		}
		else {
			notifyErrorListeners("invalid valuetype " + valueType, start);
			return false;
		}
		return true;
	}
	
	private DMLProgram parseAndAddImportedFunctions(String namespace, String filePath, ParserRuleContext ctx) {
		return parseAndAddImportedFunctions(namespace, filePath, ctx, false);
	}
	
	private DMLProgram parseAndAddImportedFunctions(String namespace, String filePath, ParserRuleContext ctx, boolean forced) {
		//validate namespace w/ awareness of dml-bodied builtin functions
		validateNamespace(namespace, filePath, ctx);
		
		//read and parse namespace files
		String scriptID = DMLProgram.constructFunctionKey(namespace, filePath);
		DMLProgram prog = null;
		if (forced || !_f2NS.get().containsKey(scriptID) ) {
			_f2NS.get().put(scriptID, namespace);
			try {
				prog = new DMLParserWrapper().doParse(filePath,
					_tScripts.get().get(filePath), getQualifiedNamespace(namespace), argVals);
			}
			catch (ParseException e) {
				notifyErrorListeners(e.getMessage(), ctx.start);
				return prog;
			}
			if(prog == null) {
				notifyErrorListeners("One or more errors found during importing a program from file " + filePath, ctx.start);
				return prog;
			}
		}
		else {
			// Skip redundant parsing (to prevent potential infinite recursion) and
			// create empty program for this context to allow processing to continue.
			prog = new DMLProgram();
		}
		return prog;
	}
}
