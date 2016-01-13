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

package org.apache.sysml.parser.python;

import java.io.File;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import org.antlr.v4.runtime.ParserRuleContext;
import org.antlr.v4.runtime.Token;
import org.antlr.v4.runtime.tree.ErrorNode;
import org.antlr.v4.runtime.tree.TerminalNode;
import org.apache.commons.lang.StringUtils;
import org.apache.sysml.parser.AssignmentStatement;
import org.apache.sysml.parser.BinaryExpression;
import org.apache.sysml.parser.BooleanExpression;
import org.apache.sysml.parser.BooleanIdentifier;
import org.apache.sysml.parser.BuiltinFunctionExpression;
import org.apache.sysml.parser.ConditionalPredicate;
import org.apache.sysml.parser.ConstIdentifier;
import org.apache.sysml.parser.DMLProgram;
import org.apache.sysml.parser.DataExpression;
import org.apache.sysml.parser.DataIdentifier;
import org.apache.sysml.parser.DoubleIdentifier;
import org.apache.sysml.parser.Expression;
import org.apache.sysml.parser.Expression.DataOp;
import org.apache.sysml.parser.Expression.DataType;
import org.apache.sysml.parser.Expression.ValueType;
import org.apache.sysml.parser.AParserWrapper;
import org.apache.sysml.parser.ExternalFunctionStatement;
import org.apache.sysml.parser.ForStatement;
import org.apache.sysml.parser.FunctionCallIdentifier;
import org.apache.sysml.parser.FunctionStatement;
import org.apache.sysml.parser.IfStatement;
import org.apache.sysml.parser.ImportStatement;
import org.apache.sysml.parser.IndexedIdentifier;
import org.apache.sysml.parser.IntIdentifier;
import org.apache.sysml.parser.IterablePredicate;
import org.apache.sysml.parser.LanguageException;
import org.apache.sysml.parser.MultiAssignmentStatement;
import org.apache.sysml.parser.OutputStatement;
import org.apache.sysml.parser.ParForStatement;
import org.apache.sysml.parser.ParameterExpression;
import org.apache.sysml.parser.ParameterizedBuiltinFunctionExpression;
import org.apache.sysml.parser.ParseException;
import org.apache.sysml.parser.PathStatement;
import org.apache.sysml.parser.PrintStatement;
import org.apache.sysml.parser.RelationalExpression;
import org.apache.sysml.parser.Statement;
import org.apache.sysml.parser.StatementBlock;
import org.apache.sysml.parser.StringIdentifier;
import org.apache.sysml.parser.WhileStatement;
//import org.apache.sysml.parser.antlr4.ExpressionInfo;
//import org.apache.sysml.parser.antlr4.StatementInfo;
import org.apache.sysml.parser.python.PydmlParser.AddSubExpressionContext;
import org.apache.sysml.parser.python.PydmlParser.AssignmentStatementContext;
import org.apache.sysml.parser.python.PydmlParser.AtomicExpressionContext;
import org.apache.sysml.parser.python.PydmlParser.BooleanAndExpressionContext;
import org.apache.sysml.parser.python.PydmlParser.BooleanNotExpressionContext;
import org.apache.sysml.parser.python.PydmlParser.BooleanOrExpressionContext;
import org.apache.sysml.parser.python.PydmlParser.BuiltinFunctionExpressionContext;
import org.apache.sysml.parser.python.PydmlParser.CommandlineParamExpressionContext;
import org.apache.sysml.parser.python.PydmlParser.CommandlinePositionExpressionContext;
import org.apache.sysml.parser.python.PydmlParser.ConstDoubleIdExpressionContext;
import org.apache.sysml.parser.python.PydmlParser.ConstFalseExpressionContext;
import org.apache.sysml.parser.python.PydmlParser.ConstIntIdExpressionContext;
import org.apache.sysml.parser.python.PydmlParser.ConstStringIdExpressionContext;
import org.apache.sysml.parser.python.PydmlParser.ConstTrueExpressionContext;
import org.apache.sysml.parser.python.PydmlParser.DataIdExpressionContext;
import org.apache.sysml.parser.python.PydmlParser.DataIdentifierContext;
import org.apache.sysml.parser.python.PydmlParser.ExpressionContext;
import org.apache.sysml.parser.python.PydmlParser.ExternalFunctionDefExpressionContext;
import org.apache.sysml.parser.python.PydmlParser.ForStatementContext;
import org.apache.sysml.parser.python.PydmlParser.FunctionCallAssignmentStatementContext;
import org.apache.sysml.parser.python.PydmlParser.FunctionCallMultiAssignmentStatementContext;
import org.apache.sysml.parser.python.PydmlParser.FunctionStatementContext;
import org.apache.sysml.parser.python.PydmlParser.IfStatementContext;
import org.apache.sysml.parser.python.PydmlParser.IfdefAssignmentStatementContext;
import org.apache.sysml.parser.python.PydmlParser.IgnoreNewLineContext;
import org.apache.sysml.parser.python.PydmlParser.ImportStatementContext;
import org.apache.sysml.parser.python.PydmlParser.IndexedExpressionContext;
import org.apache.sysml.parser.python.PydmlParser.InternalFunctionDefExpressionContext;
import org.apache.sysml.parser.python.PydmlParser.IterablePredicateColonExpressionContext;
import org.apache.sysml.parser.python.PydmlParser.IterablePredicateSeqExpressionContext;
import org.apache.sysml.parser.python.PydmlParser.MatrixDataTypeCheckContext;
import org.apache.sysml.parser.python.PydmlParser.Ml_typeContext;
import org.apache.sysml.parser.python.PydmlParser.ModIntDivExpressionContext;
import org.apache.sysml.parser.python.PydmlParser.MultDivExpressionContext;
import org.apache.sysml.parser.python.PydmlParser.ParForStatementContext;
import org.apache.sysml.parser.python.PydmlParser.ParameterizedExpressionContext;
import org.apache.sysml.parser.python.PydmlParser.PathStatementContext;
import org.apache.sysml.parser.python.PydmlParser.PmlprogramContext;
import org.apache.sysml.parser.python.PydmlParser.PowerExpressionContext;
import org.apache.sysml.parser.python.PydmlParser.RelationalExpressionContext;
import org.apache.sysml.parser.python.PydmlParser.SimpleDataIdentifierExpressionContext;
import org.apache.sysml.parser.python.PydmlParser.StatementContext;
import org.apache.sysml.parser.python.PydmlParser.StrictParameterizedExpressionContext;
import org.apache.sysml.parser.python.PydmlParser.StrictParameterizedKeyValueStringContext;
import org.apache.sysml.parser.python.PydmlParser.TypedArgNoAssignContext;
import org.apache.sysml.parser.python.PydmlParser.UnaryExpressionContext;
import org.apache.sysml.parser.python.PydmlParser.ValueDataTypeCheckContext;
import org.apache.sysml.parser.python.PydmlParser.WhileStatementContext;

/**
 * TODO: Refactor duplicated parser code dml/pydml (entire package).
 *
 */
public class PydmlSyntacticValidator implements PydmlListener
{	
	private PydmlSyntacticValidatorHelper helper = null;
	
	private String _workingDir = ".";   //current working directory
	private String _currentPath = null; //current file path
	private HashMap<String,String> argVals = null;
	
	public PydmlSyntacticValidator(PydmlSyntacticValidatorHelper helper, String currentPath, HashMap<String,String> argVals) {
		this.helper = helper;
		this.argVals = argVals;
		
		_currentPath = currentPath;
	}
	
	// Functions we have to implement but don't really need it
	@Override
	public void enterAddSubExpression(AddSubExpressionContext ctx) { }
	@Override
	public void enterAssignmentStatement(AssignmentStatementContext ctx) {}
	@Override
	public void enterAtomicExpression(AtomicExpressionContext ctx) { }
	@Override
	public void enterBooleanAndExpression(BooleanAndExpressionContext ctx) { }
	@Override
	public void enterBooleanNotExpression(BooleanNotExpressionContext ctx) { }
	@Override
	public void enterBooleanOrExpression(BooleanOrExpressionContext ctx) { }
	@Override
	public void enterCommandlineParamExpression(CommandlineParamExpressionContext ctx) { }
	@Override
	public void enterCommandlinePositionExpression(CommandlinePositionExpressionContext ctx) { }	
	@Override
	public void enterConstDoubleIdExpression(ConstDoubleIdExpressionContext ctx) { }
	@Override
	public void enterConstIntIdExpression(ConstIntIdExpressionContext ctx) { }
	@Override
	public void enterConstStringIdExpression(ConstStringIdExpressionContext ctx) { }
	@Override
	public void enterDataIdExpression(DataIdExpressionContext ctx) { }

	@Override
	public void enterIgnoreNewLine(IgnoreNewLineContext ctx) { }
	@Override
	public void enterPmlprogram(PmlprogramContext ctx) { }
	@Override
	public void exitPmlprogram(PmlprogramContext ctx) { }
	
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
	@Override
	public void enterExternalFunctionDefExpression(ExternalFunctionDefExpressionContext ctx) { }
	@Override
	public void enterForStatement(ForStatementContext ctx) {}
	@Override
	public void enterFunctionCallAssignmentStatement(FunctionCallAssignmentStatementContext ctx) { }
	@Override
	public void enterFunctionCallMultiAssignmentStatement(FunctionCallMultiAssignmentStatementContext ctx) { }
	@Override
	public void enterIfStatement(IfStatementContext ctx) { }
	@Override
	public void enterImportStatement(ImportStatementContext ctx) { }
	@Override
	public void enterIndexedExpression(IndexedExpressionContext ctx) { }
	@Override
	public void enterInternalFunctionDefExpression(InternalFunctionDefExpressionContext ctx) { }
	@Override
	public void enterMl_type(Ml_typeContext ctx) { }
	@Override
	public void enterModIntDivExpression(ModIntDivExpressionContext ctx) { }
	@Override
	public void enterMultDivExpression(MultDivExpressionContext ctx) { }
	@Override
	public void enterParameterizedExpression(ParameterizedExpressionContext ctx) { }
	@Override
	public void enterParForStatement(ParForStatementContext ctx) { }
	@Override
	public void enterPathStatement(PathStatementContext ctx) { }
	@Override
	public void enterPowerExpression(PowerExpressionContext ctx) { }
	@Override
	public void enterRelationalExpression(RelationalExpressionContext ctx) { }
	@Override
	public void enterSimpleDataIdentifierExpression(SimpleDataIdentifierExpressionContext ctx) { }
	@Override
	public void enterStrictParameterizedExpression(StrictParameterizedExpressionContext ctx) { }
	@Override
	public void enterTypedArgNoAssign(TypedArgNoAssignContext ctx) { }
	@Override
	public void enterUnaryExpression(UnaryExpressionContext ctx) { }
	@Override
	public void enterWhileStatement(WhileStatementContext ctx) { }
	
	@Override
	public void visitErrorNode(ErrorNode arg0) { }
	@Override
	public void visitTerminal(TerminalNode arg0) { }
	@Override
	public void exitEveryRule(ParserRuleContext arg0) {}
	// --------------------------------------------------------------------
	private void setFileLineColumn(Expression expr, ParserRuleContext ctx) {
		// expr.setFilename(helper.getCurrentFileName());
		String txt = ctx.getText();
		expr.setFilename(_currentPath);
		expr.setBeginLine(ctx.start.getLine());
		expr.setBeginColumn(ctx.start.getCharPositionInLine());
		expr.setEndLine(ctx.stop.getLine());
		expr.setEndColumn(ctx.stop.getCharPositionInLine());
		if(expr.getBeginColumn() == expr.getEndColumn() && expr.getBeginLine() == expr.getEndLine() && txt.length() > 1) {
			expr.setEndColumn(expr.getBeginColumn() + txt.length() - 1);
		}
	}
	
	private void setFileLineColumn(Statement stmt, ParserRuleContext ctx) {
		String txt = ctx.getText();
		stmt.setFilename(helper.getCurrentFileName());
		stmt.setBeginLine(ctx.start.getLine());
		stmt.setBeginColumn(ctx.start.getCharPositionInLine());
		stmt.setEndLine(ctx.stop.getLine());
		stmt.setEndColumn(ctx.stop.getCharPositionInLine());
		if(stmt.getBeginColumn() == stmt.getEndColumn() && stmt.getBeginLine() == stmt.getEndLine() && txt.length() > 1) {
			stmt.setEndColumn(stmt.getBeginColumn() + txt.length() - 1);
		}
	}
	
	// For now do no type checking, let validation handle it.
	// This way parser doesn't have to open metadata file
	@Override
	public void exitAddSubExpression(AddSubExpressionContext ctx) {
		if(ctx.left.info.expr != null && ctx.right.info.expr != null) {
			// Addition and subtraction operator same as DML
			Expression.BinaryOp bop = Expression.getBinaryOp(ctx.op.getText());
			ctx.info.expr = new BinaryExpression(bop);
			((BinaryExpression)ctx.info.expr).setLeft(ctx.left.info.expr);
			((BinaryExpression)ctx.info.expr).setRight(ctx.right.info.expr);
			setFileLineColumn(ctx.info.expr, ctx);
		}
	}
	
	
	
	@Override
	public void exitModIntDivExpression(ModIntDivExpressionContext ctx) {
		if(ctx.left.info.expr != null && ctx.right.info.expr != null) {
			String dmlOperator = "";
			if(ctx.op.getText().compareTo("//") == 0) {
				dmlOperator = "%/%";
			}
			else if(ctx.op.getText().compareTo("%") == 0) {
				dmlOperator = "%%";
			}
			else {
				helper.notifyErrorListeners("Incorrect operator (expected // or %)", ctx.op);
				return;
			}
			Expression.BinaryOp bop = Expression.getBinaryOp(dmlOperator);
			ctx.info.expr = new BinaryExpression(bop);
			((BinaryExpression)ctx.info.expr).setLeft(ctx.left.info.expr);
			((BinaryExpression)ctx.info.expr).setRight(ctx.right.info.expr);
			setFileLineColumn(ctx.info.expr, ctx);
		}
	}
	
	@Override
	public void exitUnaryExpression(UnaryExpressionContext ctx) {
		if(ctx.left.info.expr != null) {
			String fileName = helper.getCurrentFileName();
			int line = ctx.start.getLine();
			int col = ctx.start.getCharPositionInLine();
			
			if(ctx.left.info.expr instanceof IntIdentifier) {
				if(ctx.op.getText().compareTo("-") == 0) {
					((IntIdentifier) ctx.left.info.expr).multiplyByMinusOne();
				}
				ctx.info.expr = ctx.left.info.expr;
			}
			else if(ctx.left.info.expr instanceof DoubleIdentifier) {
				if(ctx.op.getText().compareTo("-") == 0) {
					((DoubleIdentifier) ctx.left.info.expr).multiplyByMinusOne();
				}
				ctx.info.expr = ctx.left.info.expr;
			}
			else {
				Expression right = new IntIdentifier(1, fileName, line, col, line, col);
				if(ctx.op.getText().compareTo("-") == 0) {
					right = new IntIdentifier(-1, fileName, line, col, line, col);
				}
				
				Expression.BinaryOp bop = Expression.getBinaryOp("*");
				ctx.info.expr = new BinaryExpression(bop);
				((BinaryExpression)ctx.info.expr).setLeft(ctx.left.info.expr);
				((BinaryExpression)ctx.info.expr).setRight(right);
			}
			setFileLineColumn(ctx.info.expr, ctx);
		}
	}
	
	@Override
	public void exitMultDivExpression(MultDivExpressionContext ctx) {
		Expression.BinaryOp bop = Expression.getBinaryOp(ctx.op.getText());
		ctx.info.expr = new BinaryExpression(bop);
		((BinaryExpression)ctx.info.expr).setLeft(ctx.left.info.expr);
		((BinaryExpression)ctx.info.expr).setRight(ctx.right.info.expr);
		setFileLineColumn(ctx.info.expr, ctx);
	}
	
	@Override
	public void exitPowerExpression(PowerExpressionContext ctx) {
		String dmlOperator = "";
		if(ctx.op.getText().compareTo("**") == 0) {
			dmlOperator = "^";
		}
		else {
			helper.notifyErrorListeners("Incorrect operator (expected **)", ctx.op);
			return;
		}
		
		Expression.BinaryOp bop = Expression.getBinaryOp(dmlOperator);
		ctx.info.expr = new BinaryExpression(bop);
		((BinaryExpression)ctx.info.expr).setLeft(ctx.left.info.expr);
		((BinaryExpression)ctx.info.expr).setRight(ctx.right.info.expr);
		setFileLineColumn(ctx.info.expr, ctx);
	}
	
	// TODO: 
//	@Override
//	public void exitMatrixMulExpression(MatrixMulExpressionContext ctx) {
//		Expression.BinaryOp bop = Expression.getBinaryOp(ctx.op.getText());
//		ctx.info.expr = new BinaryExpression(bop);
//		((BinaryExpression)ctx.info.expr).setLeft(ctx.left.info.expr);
//		((BinaryExpression)ctx.info.expr).setRight(ctx.right.info.expr);
//		setFileLineColumn(ctx.info.expr, ctx);
//	}

	// --------------------------------------------------------------------

	@Override
	public void exitRelationalExpression(RelationalExpressionContext ctx) {
		if(ctx.left.info.expr != null && ctx.right.info.expr != null) {
			Expression.RelationalOp rop = Expression.getRelationalOp(ctx.op.getText());
			ctx.info.expr = new RelationalExpression(rop);
			((RelationalExpression)ctx.info.expr).setLeft(ctx.left.info.expr);
			((RelationalExpression)ctx.info.expr).setRight(ctx.right.info.expr);
			setFileLineColumn(ctx.info.expr, ctx);
		}
	}
	
	// --------------------------------------------------------------------
	
	@Override
	public void exitBooleanAndExpression(BooleanAndExpressionContext ctx) {
		if(ctx.left.info.expr != null && ctx.right.info.expr != null) {
			String dmlOperator = "";
			if(ctx.op.getText().compareTo("&") == 0 || ctx.op.getText().compareTo("and") == 0) {
				dmlOperator = "&";
			}
			else {
				helper.notifyErrorListeners("Incorrect operator (expected &)", ctx.op);
				return;
			}
			
			Expression.BooleanOp bop = Expression.getBooleanOp(dmlOperator);
			ctx.info.expr = new BooleanExpression(bop);
			((BooleanExpression)ctx.info.expr).setLeft(ctx.left.info.expr);
			((BooleanExpression)ctx.info.expr).setRight(ctx.right.info.expr);
			setFileLineColumn(ctx.info.expr, ctx);
		}
	}
	
	@Override
	public void exitBooleanOrExpression(BooleanOrExpressionContext ctx) {
		if(ctx.left.info.expr != null && ctx.right.info.expr != null) {
			String dmlOperator = "";
			if(ctx.op.getText().compareTo("|") == 0 || ctx.op.getText().compareTo("or") == 0) {
				dmlOperator = "|";
			}
			else {
				helper.notifyErrorListeners("Incorrect operator (expected |)", ctx.op);
				return;
			}
			
			Expression.BooleanOp bop = Expression.getBooleanOp(dmlOperator);
			ctx.info.expr = new BooleanExpression(bop);
			((BooleanExpression)ctx.info.expr).setLeft(ctx.left.info.expr);
			((BooleanExpression)ctx.info.expr).setRight(ctx.right.info.expr);
			setFileLineColumn(ctx.info.expr, ctx);
		}
	}

	@Override
	public void exitBooleanNotExpression(BooleanNotExpressionContext ctx) {
		if(ctx.left.info.expr != null) {
			Expression.BooleanOp bop = Expression.getBooleanOp(ctx.op.getText());
			ctx.info.expr = new BooleanExpression(bop);
			((BooleanExpression)ctx.info.expr).setLeft(ctx.left.info.expr);
			setFileLineColumn(ctx.info.expr, ctx);
		}
	}
	
	// --------------------------------------------------------------------
	
	@Override
	public void exitAtomicExpression(AtomicExpressionContext ctx) {
		ctx.info.expr = ctx.left.info.expr;
		setFileLineColumn(ctx.info.expr, ctx);
	}
	
	@Override
	public void exitConstDoubleIdExpression(ConstDoubleIdExpressionContext ctx) {
		try {
			double val = Double.parseDouble(ctx.getText());
			int linePosition = ctx.start.getLine();
			int charPosition = ctx.start.getCharPositionInLine();
			ctx.info.expr = new DoubleIdentifier(val, helper.getCurrentFileName(), linePosition, charPosition, linePosition, charPosition);
			setFileLineColumn(ctx.info.expr, ctx);
		}
		catch(Exception e) {
			helper.notifyErrorListeners("cannot parse the float value: \'" +  ctx.getText() + "\'", ctx.getStart());
			return;
		}
	}

	@Override
	public void exitConstIntIdExpression(ConstIntIdExpressionContext ctx) {
		try {
			long val = Long.parseLong(ctx.getText());
			int linePosition = ctx.start.getLine();
			int charPosition = ctx.start.getCharPositionInLine();
			ctx.info.expr = new IntIdentifier(val, helper.getCurrentFileName(), linePosition, charPosition, linePosition, charPosition);
			setFileLineColumn(ctx.info.expr, ctx);
		}
		catch(Exception e) {
			helper.notifyErrorListeners("cannot parse the int value: \'" +  ctx.getText() + "\'", ctx.getStart());
			return;
		}
	}

	@Override
	public void exitConstStringIdExpression(ConstStringIdExpressionContext ctx) {
		String val = "";
		String text = ctx.getText();
		if(	(text.startsWith("\"") && text.endsWith("\"")) ||
			(text.startsWith("\'") && text.endsWith("\'"))) {
			if(text.length() > 2) {
				val = text.substring(1, text.length()-1);
			}
		}
		else {
			helper.notifyErrorListeners("something wrong while parsing string ... strange", ctx.start);
			return;
		}
			
		int linePosition = ctx.start.getLine();
		int charPosition = ctx.start.getCharPositionInLine();
		ctx.info.expr = new StringIdentifier(val, helper.getCurrentFileName(), linePosition, charPosition, linePosition, charPosition);
		setFileLineColumn(ctx.info.expr, ctx);
	}
	
	// --------------------------------------------------------------------
	
	@Override
	public void exitDataIdExpression(DataIdExpressionContext ctx) {
		ctx.info.expr = ctx.dataIdentifier().dataInfo.expr;
		int line = ctx.start.getLine();
		int col = ctx.start.getCharPositionInLine();
		ctx.info.expr.setAllPositions(helper.getCurrentFileName(), line, col, line, col);
		setFileLineColumn(ctx.info.expr, ctx);
	}
	
	@Override
	public void exitSimpleDataIdentifierExpression(SimpleDataIdentifierExpressionContext ctx) {
		// This is either a function, or variable with namespace
		// By default, it assigns to a data type
		ctx.dataInfo.expr = new DataIdentifier(ctx.getText());
		setFileLineColumn(ctx.dataInfo.expr, ctx);
	}
	
	
	private Expression incrementByOne(Expression expr, ParserRuleContext ctx) {
		// For maintaining semantic consistency, we have decided to keep 1-based indexing
		// If in future, PyDML becomes more popular than DML, this can be switched.
		return expr;
	}
	
	@Override
	public void exitIndexedExpression(IndexedExpressionContext ctx) {
		ctx.dataInfo.expr = new IndexedIdentifier(ctx.name.getText(), false, false);
		setFileLineColumn(ctx.dataInfo.expr, ctx);
		try {
			ArrayList< ArrayList<Expression> > exprList = new ArrayList< ArrayList<Expression> >();
			
			ArrayList<Expression> rowIndices = new ArrayList<Expression>();
			ArrayList<Expression> colIndices = new ArrayList<Expression>();
			
			boolean isRowLower = (ctx.rowLower != null && !ctx.rowLower.isEmpty() && (ctx.rowLower.info.expr != null));
			boolean isRowUpper = (ctx.rowUpper != null && !ctx.rowUpper.isEmpty() && (ctx.rowUpper.info.expr != null));
			boolean isColLower = (ctx.colLower != null && !ctx.colLower.isEmpty() && (ctx.colLower.info.expr != null));
			boolean isColUpper = (ctx.colUpper != null && !ctx.colUpper.isEmpty() && (ctx.colUpper.info.expr != null));
			
			if(!isRowLower && !isRowUpper) {
				// both not set
				rowIndices.add(null); rowIndices.add(null);
			}
			else if(isRowLower && isRowUpper) {
				// both set
				rowIndices.add(incrementByOne(ctx.rowLower.info.expr, ctx));
				rowIndices.add(ctx.rowUpper.info.expr);
			}
			else if(isRowLower && !isRowUpper) {
				// only row set
				rowIndices.add(incrementByOne(ctx.rowLower.info.expr, ctx));
			}
			else {
				helper.notifyErrorListeners("incorrect index expression for row", ctx.start);
				return;
			}
			
			if(!isColLower && !isColUpper) {
				// both not set
				colIndices.add(null); colIndices.add(null);
			}
			else if(isColLower && isColUpper) {
				colIndices.add(incrementByOne(ctx.colLower.info.expr, ctx));
				colIndices.add(ctx.colUpper.info.expr);
			}
			else if(isColLower && !isColUpper) {
				colIndices.add(incrementByOne(ctx.colLower.info.expr, ctx));
			}
			else {
				helper.notifyErrorListeners("incorrect index expression for column", ctx.start);
				return;
			}
			exprList.add(rowIndices);
			exprList.add(colIndices);
			((IndexedIdentifier) ctx.dataInfo.expr).setIndices(exprList);
		}
		catch(Exception e) {
			helper.notifyErrorListeners("cannot set the indices", ctx.start);
			return;
		}
	}
	
	private ConstIdentifier getConstIdFromString(String varValue, Token start) {
		// Both varName and varValue are correct
		int linePosition = start.getLine();
		int charPosition = start.getCharPositionInLine();
		try {
			long val = Long.parseLong(varValue);
			return new IntIdentifier(val, helper.getCurrentFileName(), linePosition, charPosition, linePosition, charPosition);
		}
		catch(Exception e) {
			try {
				double val = Double.parseDouble(varValue);
				return new DoubleIdentifier(val, helper.getCurrentFileName(), linePosition, charPosition, linePosition, charPosition);
			}
			catch(Exception e1) {
				try {
					if(varValue.compareTo("True") == 0 || varValue.compareTo("False") == 0) {
						boolean val = false;
						if(varValue.compareTo("True") == 0) {
							val = true;
						}
						return new BooleanIdentifier(val, helper.getCurrentFileName(), linePosition, charPosition, linePosition, charPosition);
					}
					else {
						String val = "";
						String text = varValue;
						if(	(text.startsWith("\"") && text.endsWith("\"")) ||
							(text.startsWith("\'") && text.endsWith("\'"))) {
							if(text.length() > 2) {
								val = text.substring(1, text.length()-1);
							}
						}
						else {
							// the commandline parameters can be passed without any quotes
							val = text;
						}
						return new StringIdentifier(val, helper.getCurrentFileName(), linePosition, charPosition, linePosition, charPosition);
					}
				}
				catch(Exception e3) {
					helper.notifyErrorListeners("unable to cast the commandline parameter into int/float/bool/str", start);
					return null;
				}
			}
		}			
	}
	
	private void fillExpressionInfoCommandLineParameters(String varName, ExpressionInfo dataInfo, Token start) {
		
		if(!varName.startsWith("$")) {
			helper.notifyErrorListeners("commandline param doesnot start with $ ... strange", start);
			return;
		}
		
		String varValue = null;
		for(Map.Entry<String, String> arg : this.argVals.entrySet()) {
			if(arg.getKey().trim().compareTo(varName) == 0) {
				if(varValue != null) {
					helper.notifyErrorListeners("multiple values passed for the parameter " + varName + " via commandline", start);
					return;
				}
				else {
					varValue = arg.getValue().trim();
				}
			}
		}
		
		if(varValue == null) {
			// helper.notifyErrorListeners("the parameter " + varName + " either needs to be passed through commandline or initialized to default value", start);
			return;
		}
		
		// Command line param cannot be empty string
		// If you want to pass space, please quote it
		if(varValue.trim().compareTo("") == 0)
			return;
		
		dataInfo.expr = getConstIdFromString(varValue, start);
	}
	
	@Override
	public void exitCommandlineParamExpression(CommandlineParamExpressionContext ctx) {
		handleCommandlineArgumentExpression(ctx);
	}

	@Override
	public void exitCommandlinePositionExpression(CommandlinePositionExpressionContext ctx) {
		handleCommandlineArgumentExpression(ctx);
	}
	
	/**
	 * 
	 * @param ctx
	 */
	private void handleCommandlineArgumentExpression(DataIdentifierContext ctx)
	{
		String varName = ctx.getText().trim();		
		fillExpressionInfoCommandLineParameters(varName, ctx.dataInfo, ctx.start);
		
		if(ctx.dataInfo.expr == null) {
			if(!(ctx.parent instanceof IfdefAssignmentStatementContext)) {
				String msg = "The parameter " + varName + " either needs to be passed "
						+ "through commandline or initialized to default value.";
				if( AParserWrapper.IGNORE_UNSPECIFIED_ARGS ) {
					ctx.dataInfo.expr = getConstIdFromString(" ", ctx.start);
					helper.raiseWarning(msg, ctx.start);
				}
				else {
					helper.notifyErrorListeners(msg, ctx.start);
				}
			}
		}
	}
	
	// --------------------------------------------------------------------
	
	@Override
	public void exitImportStatement(ImportStatementContext ctx)
	{
		//prepare import filepath
		String filePath = ctx.filePath.getText();
		String namespace = DMLProgram.DEFAULT_NAMESPACE;
		if(ctx.namespace != null && ctx.namespace.getText() != null && !ctx.namespace.getText().isEmpty()) { 
			namespace = ctx.namespace.getText();
		}
		if((filePath.startsWith("\"") && filePath.endsWith("\"")) || 
				filePath.startsWith("'") && filePath.endsWith("'")) {	
			filePath = filePath.substring(1, filePath.length()-1);
		}
		
		//concatenate working directory to filepath
		filePath = _workingDir + File.separator + filePath;
		
		DMLProgram prog = null;
		try {
			prog = (new PyDMLParserWrapper()).doParse(filePath, null, argVals);
		} catch (ParseException e) {
			helper.notifyErrorListeners("Exception found during importing a program from file " + filePath, ctx.start);
			return;
		}
        // Custom logic whether to proceed ahead or not. Better than the current exception handling mechanism
		if(prog == null) {
			helper.notifyErrorListeners("One or more errors found during importing a program from file " + filePath, ctx.start);
			return;
		}
		else {
			ctx.info.namespaces = new HashMap<String, DMLProgram>();
			ctx.info.namespaces.put(namespace, prog);
			ctx.info.stmt = new ImportStatement();
			((ImportStatement) ctx.info.stmt).setCompletePath(filePath);
			((ImportStatement) ctx.info.stmt).setFilePath(ctx.filePath.getText());
			((ImportStatement) ctx.info.stmt).setNamespace(namespace);
		}
	}
	
	@Override
	public void exitAssignmentStatement(AssignmentStatementContext ctx) {
		if(ctx.targetList == null || ctx.targetList.size() != 1) {
			helper.notifyErrorListeners("incorrect parsing for assignment", ctx.start);
			return;
		}
		String targetListText = ctx.targetList.get(0).getText(); 
		if(targetListText.startsWith("$")) {
			helper.notifyErrorListeners("assignment of commandline parameters is not allowed. (Quickfix: try using someLocalVariable=ifdef(" + targetListText + ", default value))", ctx.start);
			return;
		}
		
		DataIdentifier target = null; 
		if(ctx.targetList.get(0).dataInfo.expr instanceof DataIdentifier) {
			target = (DataIdentifier) ctx.targetList.get(0).dataInfo.expr;
			Expression source = ctx.source.info.expr;
			
			int line = ctx.start.getLine();
			int col = ctx.start.getCharPositionInLine();
			try {
				ctx.info.stmt = new AssignmentStatement(target, source, line, col, line, col);
				setFileLineColumn(ctx.info.stmt, ctx);
			} catch (LanguageException e) {
				// TODO: extract more meaningful info from this exception.
				helper.notifyErrorListeners("invalid assignment", ctx.targetList.get(0).start);
				return;
			} 
		}
		else {
			helper.notifyErrorListeners("incorrect lvalue ... strange", ctx.targetList.get(0).start);
			return;
		}
		
	}

	
	private void setAssignmentStatement(DataIdentifier target, Expression expression, StatementContext ctx) {
		try {
			ctx.info.stmt = new AssignmentStatement(target, expression, ctx.start.getLine(), ctx.start.getCharPositionInLine(), ctx.start.getLine(), ctx.start.getCharPositionInLine());
			setFileLineColumn(ctx.info.stmt, ctx);
		} catch (LanguageException e) {
			// TODO: extract more meaningful info from this exception.
			helper.notifyErrorListeners("invalid function call", ctx.start);
			return;
		}
	}
	
	private void setPrintStatement(FunctionCallAssignmentStatementContext ctx, String functionName) {
		ArrayList<ParameterExpression> paramExpression = helper.getParameterExpressionList(ctx.paramExprs);
		if(paramExpression.size() != 1) {
			helper.notifyErrorListeners(functionName + "() has only one parameter", ctx.start);
			return;
		}
		Expression expr = paramExpression.get(0).getExpr();
		if(expr == null) {
			helper.notifyErrorListeners("cannot process " + functionName + "() function", ctx.start);
			return;
		}
		try {
			int line = ctx.start.getLine();
			int col = ctx.start.getCharPositionInLine();
			ctx.info.stmt = new PrintStatement(functionName, expr, line, col, line, col);
		} catch (LanguageException e) {
			helper.notifyErrorListeners("cannot process " + functionName + "() function", ctx.start);
			return;
		}
	}
	
	private void setOutputStatement(FunctionCallAssignmentStatementContext ctx) {
		ArrayList<ParameterExpression> paramExpression = helper.getParameterExpressionList(ctx.paramExprs);
		if(paramExpression.size() < 2){
			helper.notifyErrorListeners("incorrect usage of load function (atleast 2 arguments required)", ctx.start);
			return;
		}
		if(paramExpression.get(0).getExpr() instanceof DataIdentifier) {
			String fileName = helper.getCurrentFileName();
			int line = ctx.start.getLine();
			int col = ctx.start.getCharPositionInLine();
			HashMap<String, Expression> varParams = new HashMap<String, Expression>();
			varParams.put(DataExpression.IO_FILENAME, paramExpression.get(1).getExpr());
			for(int i = 2; i < paramExpression.size(); i++) {
				// DataExpression.FORMAT_TYPE, DataExpression.DELIM_DELIMITER, DataExpression.DELIM_HAS_HEADER_ROW,  DataExpression.DELIM_SPARSE
				varParams.put(paramExpression.get(i).getName(), paramExpression.get(i).getExpr());
			}
			
			DataExpression  dataExpression = new DataExpression(DataOp.WRITE, varParams, fileName, line, col, line, col);
			ctx.info.stmt = new  OutputStatement((DataIdentifier) paramExpression.get(0).getExpr(), DataOp.WRITE, fileName, line, col, line, col);
			setFileLineColumn(ctx.info.stmt, ctx);
			((OutputStatement)ctx.info.stmt).setExprParams(dataExpression);
			return;
		}
		
		helper.notifyErrorListeners("incorrect usage of write function", ctx.start);
		return;
		
	}

	private boolean inDefaultNamespace(String namespace) {
		return namespace.compareTo(DMLProgram.DEFAULT_NAMESPACE) == 0;
	}
	
	// Returns 0, 1 or -1 (for error)
	private int getAxis(ParameterizedExpressionContext ctx) {
		if(ctx.paramName != null && ctx.paramName.getText() != null && !ctx.paramName.getText().isEmpty()) {
			if(ctx.paramName.getText().compareTo("axis") != 0) {
				return -1;
			}
		}
		
		String val = ctx.paramVal.getText();
		if(val != null && val.compareTo("0") == 0) {
			return 0;
		}
		else if(val != null && val.compareTo("1") == 0) {
			return 1;
		}
		
		return -1;
	}
	
	private String getPythonAggFunctionNames(String functionName, int axis) {
		if(axis != 0 && axis != 1) {
			return functionName;
		}
		// axis=0 maps to column-wise computation and axis=1 maps to row-wise computation
		
		if(functionName.compareTo("sum") == 0) {
			return axis == 0 ? "colSums" : "rowSums"; 
		}
		else if(functionName.compareTo("mean") == 0) {
			return axis == 0 ? "colMeans" : "rowMeans"; 
		}
		else if(functionName.compareTo("avg") == 0) {
			return axis == 0 ? "colMeans" : "rowMeans";
		}
		else if(functionName.compareTo("max") == 0) {
			return axis == 0 ? "colMaxs" : "rowMaxs";
		}
		else if(functionName.compareTo("min") == 0) {
			return axis == 0 ? "colMins" : "rowMins";
		}
		else if(functionName.compareTo("argmin") == 0) {
			return axis == 0 ? "Not Supported" : "rowIndexMin";
		}
		else if(functionName.compareTo("argmax") == 0) {
			return axis == 0 ? "Not Supported" : "rowIndexMax";
		}
		else if(functionName.compareTo("cumsum") == 0) {
			return axis == 0 ?  "cumsum" : "Not Supported";
		}
		else if(functionName.compareTo("transpose") == 0) {
			return axis == 0 ?  "Not Supported" : "Not Supported";
		}
		else if(functionName.compareTo("trace") == 0) {
			return axis == 0 ?  "Not Supported" : "Not Supported";
		}
		else {
			return functionName;
		}
	}
	
	private ConvertedDMLSyntax convertPythonBuiltinFunctionToDMLSyntax(String namespace, String functionName, ArrayList<ParameterExpression> paramExpression, 
		List<ParameterizedExpressionContext> paramCtx, Token fnName, String fileName, int line, int col) {
		// ===========================================================================================
		// Check function name, namespace, parameters (#params & possible values) and throw useful hints
		if(inDefaultNamespace(namespace) && functionName.compareTo("len") == 0) {
			if(paramExpression.size() != 1) {
				helper.notifyErrorListeners("The builtin function \'" + functionName + "\' accepts 1 arguments", fnName);
				return null;
			}
			functionName = "length";
		}
		else if(functionName.compareTo("sum") == 0 || functionName.compareTo("mean") == 0 || functionName.compareTo("avg") == 0 ||
				functionName.compareTo("min") == 0 || functionName.compareTo("max") == 0  || 
				functionName.compareTo("argmax") == 0 || functionName.compareTo("argmin") == 0 ||
				functionName.compareTo("cumsum") == 0 || functionName.compareTo("transpose") == 0 || functionName.compareTo("trace") == 0) {
			// 0 maps row-wise computation and 1 maps to column-wise computation
			
			// can mean sum of all cells or row-wise or columnwise sum
			if(inDefaultNamespace(namespace) && paramExpression.size() == 1) {
				// sum(x) => sum(x)
				// otherwise same function name
				if(functionName.compareTo("avg") == 0) {
					functionName = "mean";
				}
				else if(functionName.compareTo("transpose") == 0) {
					functionName = "t";
				}
				else if(functionName.compareTo("argmax") == 0 || functionName.compareTo("argmin") == 0 || functionName.compareTo("cumsum") == 0) {
					helper.notifyErrorListeners("The builtin function \'" + functionName + "\' for entire matrix is not supported", fnName);
					return null;
				}
			}
			else if(!inDefaultNamespace(namespace) && paramExpression.size() == 0) {
				// x.sum() => sum(x)
				paramExpression = new ArrayList<ParameterExpression>();
				paramExpression.add(new ParameterExpression(null, new DataIdentifier(namespace)));
				// otherwise same function name
				if(functionName.compareTo("avg") == 0) {
					functionName = "mean";
				}
				else if(functionName.compareTo("transpose") == 0) {
					functionName = "t";
				}
				else if(functionName.compareTo("argmax") == 0 || functionName.compareTo("argmin") == 0 || functionName.compareTo("cumsum") == 0) {
					helper.notifyErrorListeners("The builtin function \'" + functionName + "\' for entire matrix is not supported", fnName);
					return null;
				}
			}
			else if(inDefaultNamespace(namespace) && paramExpression.size() == 2) {
				// sum(x, axis=1) => rowSums(x)
				int axis = getAxis(paramCtx.get(1));
				if(axis == -1 && (functionName.compareTo("min") == 0 || functionName.compareTo("max") == 0 )) {
					// Do nothing
					// min(2, 3)
				}
				else if(axis == -1) {
					helper.notifyErrorListeners("The builtin function \'" + functionName + "\' for given arguments is not supported", fnName);
					return null;
				}
				else {
					ArrayList<ParameterExpression> temp = new ArrayList<ParameterExpression>();
					temp.add(paramExpression.get(0));
					paramExpression = temp;
					functionName = getPythonAggFunctionNames(functionName, axis);
					if(functionName.compareTo("Not Supported") == 0) {
						helper.notifyErrorListeners("The builtin function \'" + functionName + "\' for given arguments is not supported", fnName);
						return null;
					}
				}
			}
			else if(!inDefaultNamespace(namespace) && paramExpression.size() == 1) {
				// x.sum(axis=1) => rowSums(x)
				int axis = getAxis(paramCtx.get(0));
				 if(axis == -1) {
					 helper.notifyErrorListeners("The builtin function \'" + functionName + "\' for given arguments is not supported", fnName);
					 return null;
				 }
				 else {
					 paramExpression = new ArrayList<ParameterExpression>();
					 paramExpression.add(new ParameterExpression(null, new DataIdentifier(namespace)));
					 functionName = getPythonAggFunctionNames(functionName, axis);
					 if(functionName.compareTo("Not Supported") == 0) {
						 helper.notifyErrorListeners("The builtin function \'" + functionName + "\' for given arguments is not supported", fnName);
						 return null;
					 }
				 }
			}
			else {
				helper.notifyErrorListeners("Incorrect number of arguments for the builtin function \'" + functionName + "\'.", fnName);
				return null;
			}
			namespace = DMLProgram.DEFAULT_NAMESPACE;
		}
		else if(inDefaultNamespace(namespace) && functionName.compareTo("concatenate") == 0) {
			if(paramExpression.size() != 2) {
				helper.notifyErrorListeners("The builtin function \'" + functionName + "\' accepts 2 arguments (Note: concatenate append columns of two matrices)", fnName);
				return null;
			}
			functionName = "append";
			namespace = DMLProgram.DEFAULT_NAMESPACE;
		}
		else if(inDefaultNamespace(namespace) && functionName.compareTo("minimum") == 0) {
			if(paramExpression.size() != 2) {
				helper.notifyErrorListeners("The builtin function \'" + functionName + "\' accepts 2 arguments", fnName);
				return null;
			}
			functionName = "min";
			namespace = DMLProgram.DEFAULT_NAMESPACE;
		}
		else if(inDefaultNamespace(namespace) && functionName.compareTo("maximum") == 0) {
			if(paramExpression.size() != 2) {
				helper.notifyErrorListeners("The builtin function \'" + functionName + "\' accepts 2 arguments", fnName);
				return null;
			}
			functionName = "max";
			namespace = DMLProgram.DEFAULT_NAMESPACE;
		}
		else if(!inDefaultNamespace(namespace) && functionName.compareTo("shape") == 0) {
			if(paramExpression.size() != 1) {
				helper.notifyErrorListeners("The builtin function \'" + functionName + "\' accepts only 1 argument (0 or 1)", fnName);
				return null;
			}
			
			int axis = getAxis(paramCtx.get(0));
			if(axis == -1) {
				helper.notifyErrorListeners("The builtin function \'" + functionName + "\' accepts only 1 argument (0 or 1)", fnName);
				return null; 
			}
			paramExpression = new ArrayList<ParameterExpression>();
			paramExpression.add(new ParameterExpression(null, new DataIdentifier(namespace)));
			namespace = DMLProgram.DEFAULT_NAMESPACE;
			if(axis == 0) {
				functionName = "nrow";
			}
			else if(axis == 1) {
				functionName = "ncol";
			}
		}
		else if(inDefaultNamespace(namespace) && functionName.compareTo("random.normal") == 0) {
			if(paramExpression.size() != 3) {
				helper.notifyErrorListeners("The builtin function \'" + functionName + "\' accepts exactly 3 arguments (number of rows, number of columns, sparsity)", fnName);
				return null;
			}
			paramExpression.get(0).setName("rows");
			paramExpression.get(1).setName("cols");
			paramExpression.get(2).setName("sparsity");
			paramExpression.add(new org.apache.sysml.parser.ParameterExpression("pdf", new StringIdentifier("normal", fileName, line, col, line, col)));
			functionName = "rand";
			namespace = DMLProgram.DEFAULT_NAMESPACE;
		}
		else if(inDefaultNamespace(namespace) && functionName.compareTo("random.uniform") == 0) {
			if(paramExpression.size() != 5) {
				helper.notifyErrorListeners("The builtin function \'" + functionName + "\' accepts exactly 5 arguments (number of rows, number of columns, sparsity, min, max)", fnName);
				return null;
			}
			paramExpression.get(0).setName("rows");
			paramExpression.get(1).setName("cols");
			paramExpression.get(2).setName("sparsity");
			paramExpression.get(3).setName("min");
			paramExpression.get(4).setName("max");
			paramExpression.add(new org.apache.sysml.parser.ParameterExpression("pdf", new StringIdentifier("uniform", fileName, line, col, line, col)));
			functionName = "rand";
			namespace = DMLProgram.DEFAULT_NAMESPACE;
		}
		else if(inDefaultNamespace(namespace) && functionName.compareTo("full") == 0) {
			if(paramExpression.size() != 3) {
				helper.notifyErrorListeners("The builtin function \'" + functionName + "\' accepts exactly 3 arguments (constant float value, number of rows, number of columns)", fnName);
				return null;
			}
			paramExpression.get(1).setName("rows");
			paramExpression.get(2).setName("cols");
			functionName = "matrix";
			namespace = DMLProgram.DEFAULT_NAMESPACE;
		}
		else if(inDefaultNamespace(namespace) && functionName.compareTo("matrix") == 0) {
			// This can either be string initializer or as.matrix function
			if(paramExpression.size() != 1) {
				helper.notifyErrorListeners("The builtin function \'" + functionName + "\' accepts exactly 1 argument (either str or float value)", fnName);
				return null;
			}
			
			if(paramExpression.get(0).getExpr() instanceof StringIdentifier) {
				String initializerString = ((StringIdentifier)paramExpression.get(0).getExpr()).getValue().trim();
				if(!initializerString.startsWith("[") || !initializerString.endsWith("]")) {
					helper.notifyErrorListeners("Incorrect initializer string for builtin function \'" + functionName + "\' (Eg: matrix(\"[1 2 3; 4 5 6]\"))", fnName);
					return null;
				}
				int rows = StringUtils.countMatches(initializerString, ";") + 1;
				
				// Make sure user doesnot have pretty string
				initializerString = initializerString.replaceAll("; ", ";");
				initializerString = initializerString.replaceAll(" ;", ";");
				initializerString = initializerString.replaceAll("\\[ ", "\\[");
				initializerString = initializerString.replaceAll(" \\]", "\\]");
				
				// Each row has ncol-1 spaces
				// #spaces = nrow * (ncol-1)
				// ncol = (#spaces / nrow) + 1 
				int cols = (StringUtils.countMatches(initializerString, " ") / rows) + 1;
				
				initializerString = initializerString.replaceAll(";", " ");
				initializerString = initializerString.replaceAll("\\[", "");
				initializerString = initializerString.replaceAll("\\]", "");
				paramExpression = new ArrayList<ParameterExpression>();
				paramExpression.add(new ParameterExpression(null, new StringIdentifier(initializerString, fileName, line, col, line, col)));
				paramExpression.add(new ParameterExpression("rows", new IntIdentifier(rows, fileName, line, col, line, col)));
				paramExpression.add(new ParameterExpression("cols", new IntIdentifier(cols, fileName, line, col, line, col)));
			}
			else {
				functionName = "as.matrix";
			}
			namespace = DMLProgram.DEFAULT_NAMESPACE;
		}
		else if(inDefaultNamespace(namespace) && functionName.compareTo("scalar") == 0) {
			if(paramExpression.size() != 1) {
				helper.notifyErrorListeners("The builtin function \'" + functionName + "\' accepts exactly 1 argument", fnName);
				return null;
			}
			functionName = "as.scalar";
			namespace = DMLProgram.DEFAULT_NAMESPACE;
		}
		else if(inDefaultNamespace(namespace) && functionName.compareTo("float") == 0) {
			if(paramExpression.size() != 1) {
				helper.notifyErrorListeners("The builtin function \'" + functionName + "\' accepts exactly 1 argument", fnName);
				return null;
			}
			functionName = "as.double";
			namespace = DMLProgram.DEFAULT_NAMESPACE;
		}
		else if(inDefaultNamespace(namespace) && functionName.compareTo("int") == 0) {
			if(paramExpression.size() != 1) {
				helper.notifyErrorListeners("The builtin function \'" + functionName + "\' accepts exactly 1 argument", fnName);
				return null;
			}
			functionName = "as.integer";
			namespace = DMLProgram.DEFAULT_NAMESPACE;
		}
		else if(inDefaultNamespace(namespace) && functionName.compareTo("bool") == 0) {
			if(paramExpression.size() != 1) {
				helper.notifyErrorListeners("The builtin function \'" + functionName + "\' accepts exactly 1 argument", fnName);
				return null;
			}
			functionName = "as.logical";
			namespace = DMLProgram.DEFAULT_NAMESPACE;
		}
		else if(!inDefaultNamespace(namespace) && functionName.compareTo("reshape") == 0) {
			if(paramExpression.size() != 2) {
				helper.notifyErrorListeners("The builtin function \'" + functionName + "\' accepts exactly 2 arguments (number of rows, number of columns)", fnName);
				return null;
			}
			paramExpression.get(0).setName("rows");
			paramExpression.get(1).setName("cols");
			
			ArrayList<ParameterExpression> temp = new ArrayList<ParameterExpression>();
			temp.add(new ParameterExpression(null, new DataIdentifier(namespace)));
			temp.add(paramExpression.get(0));
			temp.add(paramExpression.get(1));
			paramExpression = temp;
			
			functionName = "matrix";
			namespace = DMLProgram.DEFAULT_NAMESPACE;
		}
		else if(inDefaultNamespace(namespace) && functionName.compareTo("removeEmpty") == 0) {
			if(paramExpression.size() != 2) {
				helper.notifyErrorListeners("The builtin function \'" + functionName + "\' accepts exactly 2 arguments (matrix, axis=0 or 1)", fnName);
				return null;
			}
			int axis = getAxis(paramCtx.get(1));
			if(axis == -1) {
				helper.notifyErrorListeners("The builtin function \'" + functionName + "\' accepts exactly 2 arguments (matrix, axis=0 or 1)", fnName);
				return null; 
			}
			StringIdentifier marginVal = null;
			if(axis == 0) {
				marginVal = new StringIdentifier("rows", fileName, line, col, line, col);
			}
			else {
				marginVal = new StringIdentifier("cols", fileName, line, col, line, col);
			}
			paramExpression.get(0).setName("target");
			paramExpression.get(1).setName("margin");
			paramExpression.get(1).setExpr(marginVal);
			functionName = "removeEmpty";
			namespace = DMLProgram.DEFAULT_NAMESPACE;
		}
		else if(inDefaultNamespace(namespace) && functionName.compareTo("replace") == 0) {
			if(paramExpression.size() != 3) {
				helper.notifyErrorListeners("The builtin function \'" + functionName + "\' accepts exactly 3 arguments (matrix, scalar value that should be replaced (pattern), scalar value (replacement))", fnName);
				return null;
			}
			paramExpression.get(0).setName("target");
			paramExpression.get(1).setName("pattern");
			paramExpression.get(2).setName("replacement");
			functionName = "replace";
			namespace = DMLProgram.DEFAULT_NAMESPACE;
		}
		else if(inDefaultNamespace(namespace) && functionName.compareTo("range") == 0) {
			if(paramExpression.size() != 3) {
				helper.notifyErrorListeners("The builtin function \'" + functionName + "\' accepts exactly 3 arguments (matrix, scalar value that should be replaced (pattern), scalar value (replacement))", fnName);
				return null;
			}
			functionName = "seq";
			namespace = DMLProgram.DEFAULT_NAMESPACE;
		}
		else if(inDefaultNamespace(namespace) && functionName.compareTo("norm.cdf") == 0) {
			if(paramExpression.size() != 3) {
				helper.notifyErrorListeners("The builtin function \'" + functionName + "\' accepts exactly 3 arguments (target, mean, sd)", fnName);
				return null;
			}
			functionName = "cumulativeProbability";
			paramExpression.get(0).setName("target");
			paramExpression.get(1).setName("mean");
			paramExpression.get(2).setName("sd");
			paramExpression.add(new ParameterExpression("dist", new StringIdentifier("normal", fileName, line, col, line, col)));
			namespace = DMLProgram.DEFAULT_NAMESPACE;
		}
		else if(inDefaultNamespace(namespace) && functionName.compareTo("expon.cdf") == 0) {
			if(paramExpression.size() != 2) {
				helper.notifyErrorListeners("The builtin function \'" + functionName + "\' accepts exactly 2 arguments (target, mean)", fnName);
				return null;
			}
			functionName = "cumulativeProbability";
			paramExpression.get(0).setName("target");
			paramExpression.get(1).setName("mean");
			paramExpression.add(new ParameterExpression("dist", new StringIdentifier("exp", fileName, line, col, line, col)));
			namespace = DMLProgram.DEFAULT_NAMESPACE;
		}
		else if(inDefaultNamespace(namespace) && functionName.compareTo("chi.cdf") == 0) {
			if(paramExpression.size() != 2) {
				helper.notifyErrorListeners("The builtin function \'" + functionName + "\' accepts exactly 2 arguments (target, df)", fnName);
				return null;
			}
			functionName = "cumulativeProbability";
			paramExpression.get(0).setName("target");
			paramExpression.get(1).setName("df");
			paramExpression.add(new ParameterExpression("dist", new StringIdentifier("chisq", fileName, line, col, line, col)));
			namespace = DMLProgram.DEFAULT_NAMESPACE;
		}
		else if(inDefaultNamespace(namespace) && functionName.compareTo("f.cdf") == 0) {
			if(paramExpression.size() != 3) {
				helper.notifyErrorListeners("The builtin function \'" + functionName + "\' accepts exactly 3 arguments (target, df1, df2)", fnName);
				return null;
			}
			functionName = "cumulativeProbability";
			paramExpression.get(0).setName("target");
			paramExpression.get(1).setName("df1");
			paramExpression.get(2).setName("df2");
			paramExpression.add(new ParameterExpression("dist", new StringIdentifier("f", fileName, line, col, line, col)));
			namespace = DMLProgram.DEFAULT_NAMESPACE;
		}
		else if(inDefaultNamespace(namespace) && functionName.compareTo("t.cdf") == 0) {
			if(paramExpression.size() != 2) {
				helper.notifyErrorListeners("The builtin function \'" + functionName + "\' accepts exactly 2 arguments (target, df)", fnName);
				return null;
			}
			functionName = "cumulativeProbability";
			paramExpression.get(0).setName("target");
			paramExpression.get(1).setName("df");
			paramExpression.add(new ParameterExpression("dist", new StringIdentifier("t", fileName, line, col, line, col)));
			namespace = DMLProgram.DEFAULT_NAMESPACE;
		}
		else if(inDefaultNamespace(namespace) && functionName.compareTo("percentile") == 0) {
			if(paramExpression.size() != 2 && paramExpression.size() != 3) {
				helper.notifyErrorListeners("The builtin function \'" + functionName + "\' accepts either 2 or 3 arguments", fnName);
				return null;
			}
			functionName = "quantile";
			namespace = DMLProgram.DEFAULT_NAMESPACE;
		}
		else if(inDefaultNamespace(namespace) && functionName.compareTo("arcsin") == 0) {
			functionName = "asin";
		}
		else if(inDefaultNamespace(namespace) && functionName.compareTo("arccos") == 0) {
			functionName = "acos";
		}
		else if(inDefaultNamespace(namespace) && functionName.compareTo("arctan") == 0) {
			functionName = "atan";
		}
		else if(inDefaultNamespace(namespace) && functionName.compareTo("load") == 0) {
			functionName = "read";
		}
		else if(inDefaultNamespace(namespace) && functionName.compareTo("eigen") == 0) {
			functionName = "eig";
		}
		else if(inDefaultNamespace(namespace) && functionName.compareTo("power") == 0) {
			if(paramExpression.size() != 2) {
				helper.notifyErrorListeners("The builtin function \'" + functionName + "\' accepts exactly 2 arguments", fnName);
				return null;
			}
		}
		else if(inDefaultNamespace(namespace) && functionName.compareTo("dot") == 0) {
			if(paramExpression.size() != 2) {
				helper.notifyErrorListeners("The builtin function \'" + functionName + "\' accepts exactly 2 arguments", fnName);
				return null;
			}
		}
		
		ConvertedDMLSyntax retVal = new ConvertedDMLSyntax();
		retVal.namespace = namespace;
		retVal.functionName = functionName;
		retVal.paramExpression = paramExpression;
		return retVal;
	}

	class ConvertedDMLSyntax {
		public String namespace;
		public String functionName;
		public ArrayList<ParameterExpression> paramExpression;
	};
	
	private Expression getOperatorExpression(String namespace, String functionName, ArrayList<ParameterExpression> paramExpression) {
		String dmlOperator = null;
		
		if(inDefaultNamespace(namespace) && functionName.compareTo("dot") == 0) {
			if(paramExpression.size() == 2) {
				dmlOperator = "%*%";
			}
		}

		if(dmlOperator != null) {
			Expression.BinaryOp bop = Expression.getBinaryOp(dmlOperator);
			Expression expr = new BinaryExpression(bop);
			((BinaryExpression)expr).setLeft(paramExpression.get(0).getExpr());
			((BinaryExpression)expr).setRight(paramExpression.get(1).getExpr());
			return expr;
		}
		
		return null;
	}
	
	@Override
	public void exitFunctionCallAssignmentStatement(FunctionCallAssignmentStatementContext ctx) {
		ArrayList<String> names = helper.getQualifiedNames(ctx.name.getText());
		if(names == null) {
			helper.notifyErrorListeners("incorrect function name (only namespace.functionName allowed. Hint: If you are trying to use builtin functions, you can skip the namespace)", ctx.name);
			return;
		}
		String namespace = names.get(0);
		String functionName = names.get(1);
		
		if((functionName.compareTo("print") == 0 || functionName.compareTo("stop") == 0 ) && namespace.compareTo(DMLProgram.DEFAULT_NAMESPACE) == 0) {
			setPrintStatement(ctx, functionName);
			return;
		}
		else if(functionName.compareTo("save") == 0 && namespace.compareTo(DMLProgram.DEFAULT_NAMESPACE) == 0) {
			setOutputStatement(ctx);
			return;
		}
		
		boolean ignoreLValue = false;
		if(ctx.targetList == null || ctx.targetList.size() == 0 || ctx.targetList.get(0).isEmpty()) {
			helper.notifyErrorListeners("function call needs to have lvalue (Quickfix: change it to \'tmpVar = " + functionName + "(...)\')", ctx.name);
			return;
		}
		String fileName = helper.getCurrentFileName();
		int line = ctx.start.getLine();
		int col = ctx.start.getCharPositionInLine();
		
		ArrayList<ParameterExpression> paramExpression = helper.getParameterExpressionList(ctx.paramExprs);
		
	 	ConvertedDMLSyntax convertedSyntax = convertPythonBuiltinFunctionToDMLSyntax(namespace, functionName, paramExpression, ctx.paramExprs, ctx.name, fileName, line, col);
		if(convertedSyntax == null) {
			return;
		}
		else {
			namespace = convertedSyntax.namespace;
			functionName = convertedSyntax.functionName;
			paramExpression = convertedSyntax.paramExpression;
		}
		
		// ===========================================================================================
		FunctionCallIdentifier functCall = new FunctionCallIdentifier(paramExpression);
		try {
			functCall.setFunctionName(functionName);
			functCall.setFunctionNamespace(namespace);
		} catch (ParseException e1) {
			helper.notifyErrorListeners("unable to process function " + functionName, ctx.start);
			 return;
		}
		
		DataIdentifier target = null; 
		if(!ignoreLValue) {
			if(ctx.targetList.get(0).dataInfo.expr instanceof DataIdentifier) {
				target = (DataIdentifier) ctx.targetList.get(0).dataInfo.expr;
			}
			else {
				helper.notifyErrorListeners("incorrect lvalue ... strange", ctx.targetList.get(0).start);
				//target = new DataIdentifier(); // so as not to avoid null pointer
				return;
			}
		}
		
		Expression operatorExpr = getOperatorExpression(namespace, functionName, paramExpression);
		if(operatorExpr != null) {
			setFileLineColumn(operatorExpr, ctx);
			setAssignmentStatement(target, operatorExpr, ctx);
			return;
		}
		
		//Note: In contrast to the dml parser, namespace and function names are separated by '.' not '::'.  
		//Hence, we have to include a whitelist of function names to handle builtins like 'as.scalar'.		
		String[] whitelist = new String[]{"as.matrix","as.scalar","as.double","as.integer","as.logical"};
		boolean isWhitelisted = Arrays.asList(whitelist).contains(functionName);
		
		if(    !functionName.contains(".") || isWhitelisted 
			|| functionName.startsWith(DMLProgram.DEFAULT_NAMESPACE) ) 
		{
			// In global namespace, so it can be a builtin function
			if(!helper.validateBuiltinFunctions(ctx)) {
				return; // it is a built-in function and validation failed, so donot proceed ahead.
			}
			// Double verification: verify passed function name is a (non-parameterized) built-in function.
			try 
			{
				// builtin functions
				BuiltinFunctionExpression bife = BuiltinFunctionExpression.getBuiltinFunctionExpression(functionName, functCall.getParamExprs(), fileName, line, col, line, col);
				if (bife != null) {
					setAssignmentStatement(target, bife, ctx);
					return;
				}
				
				// parameterized builtin functions
				ParameterizedBuiltinFunctionExpression pbife = ParameterizedBuiltinFunctionExpression.getParamBuiltinFunctionExpression(functionName, functCall.getParamExprs(), fileName, line, col, line, col);
				if (pbife != null) {
					setAssignmentStatement(target, pbife, ctx);
					return;
				}
				
				// built-in data expressions, e.g. read
				DataExpression dbife = DataExpression.getDataExpression(functionName, functCall.getParamExprs(), fileName, line, col, line, col);
				if (dbife != null){
					setAssignmentStatement(target, dbife, ctx);
					return;
				}
			} 
			catch(Exception e) {
				helper.notifyErrorListeners("unable to process builtin function expression " + functionName  + ":" + e.getMessage(), ctx.start);
				return ;
			}
		}
		
		setAssignmentStatement(target, functCall, ctx);
	}
	
	
	@Override
	public void exitBuiltinFunctionExpression(BuiltinFunctionExpressionContext ctx) {
		// Double verification: verify passed function name is a (non-parameterized) built-in function.
		ArrayList<String> names = helper.getQualifiedNames(ctx.name.getText());
		if(names == null) {
			helper.notifyErrorListeners("incorrect function name (only namespace.functionName allowed. Hint: If you are trying to use builtin functions, you can skip the namespace)", ctx.name);
			return;
		}
		String namespace = names.get(0);
		String functionName = names.get(1);
		
		String fileName = helper.getCurrentFileName();
		int line = ctx.start.getLine();
		int col = ctx.start.getCharPositionInLine();
		ArrayList<ParameterExpression> paramExpression = helper.getParameterExpressionList(ctx.paramExprs);

		ConvertedDMLSyntax convertedSyntax = convertPythonBuiltinFunctionToDMLSyntax(namespace, functionName, paramExpression, ctx.paramExprs, ctx.name, fileName, line, col);
		if(convertedSyntax == null) {
			return;
		}
		else {
			namespace = convertedSyntax.namespace;
			functionName = convertedSyntax.functionName;
			paramExpression = convertedSyntax.paramExpression;
			// System.out.println(ctx.name.getText() + ">>" + namespace + " " + functionName);
		}
		
		Expression operatorExpr = getOperatorExpression(namespace, functionName, paramExpression);
		if(operatorExpr != null) {
			ctx.info.expr = operatorExpr;
			setFileLineColumn(operatorExpr, ctx);
			return;
		}
		
		try {
			BuiltinFunctionExpression bife = BuiltinFunctionExpression.getBuiltinFunctionExpression(functionName, paramExpression, fileName, line, col, line, col);
			if (bife != null){
				// It is a builtin function
				ctx.info.expr = bife;
				return;
			}
			
			ParameterizedBuiltinFunctionExpression pbife = ParameterizedBuiltinFunctionExpression.getParamBuiltinFunctionExpression(functionName, paramExpression, fileName, line, col, line, col);
			if (pbife != null){
				// It is a parameterized builtin function
				ctx.info.expr = pbife;
				return;
			}
			
			// built-in read, rand ...
			DataExpression dbife = DataExpression.getDataExpression(functionName, paramExpression, fileName, line, col, line, col);
			if (dbife != null){
				ctx.info.expr = dbife;
				return;
			}
		} catch(Exception e) {
			helper.notifyErrorListeners("unable to process builtin function expression " + functionName + ":" + e.getMessage(), ctx.start);
			return ;
		}
		helper.notifyErrorListeners("only builtin functions allowed as part of expression", ctx.start);
	}
	
	private void setMultiAssignmentStatement(ArrayList<DataIdentifier> target, Expression expression, StatementContext ctx) {
		ctx.info.stmt = new MultiAssignmentStatement(target, expression);
		ctx.info.stmt.setAllPositions(helper.getCurrentFileName(), ctx.start.getLine(), ctx.start.getCharPositionInLine(), ctx.start.getLine(), ctx.start.getCharPositionInLine());
		setFileLineColumn(ctx.info.stmt, ctx);
	}

	@Override
	public void exitFunctionCallMultiAssignmentStatement(
			FunctionCallMultiAssignmentStatementContext ctx) {
		ArrayList<String> names = helper.getQualifiedNames(ctx.name.getText());
		if(names == null) {
			helper.notifyErrorListeners("incorrect function name (only namespace.functionName allowed. Hint: If you are trying to use builtin functions, you can skip the namespace)", ctx.name);
			return;
		}
		String namespace = names.get(0);
		String functionName = names.get(1);
		
		String fileName = helper.getCurrentFileName();
		int line = ctx.start.getLine();
		int col = ctx.start.getCharPositionInLine();
		
		ArrayList<ParameterExpression> paramExpression = helper.getParameterExpressionList(ctx.paramExprs);
		ConvertedDMLSyntax convertedSyntax = convertPythonBuiltinFunctionToDMLSyntax(namespace, functionName, paramExpression, ctx.paramExprs, ctx.name, fileName, line, col);
		if(convertedSyntax == null) {
			return;
		}
		else {
			namespace = convertedSyntax.namespace;
			functionName = convertedSyntax.functionName;
			paramExpression = convertedSyntax.paramExpression;
		}
		
		// No need to support dot() function since it will never return multi-assignment function
		
		FunctionCallIdentifier functCall = new FunctionCallIdentifier(paramExpression);
		try {
			functCall.setFunctionName(functionName);
			functCall.setFunctionNamespace(namespace);
		} catch (ParseException e1) {
			helper.notifyErrorListeners("unable to process function " + functionName, ctx.start);
			return;
		}
		
		ArrayList<DataIdentifier> targetList = new ArrayList<DataIdentifier>();
		for(DataIdentifierContext dataCtx : ctx.targetList) {
			if(dataCtx.dataInfo.expr instanceof DataIdentifier) {
				targetList.add((DataIdentifier) dataCtx.dataInfo.expr);
			}
			else {
				helper.notifyErrorListeners("incorrect lvalue ... strange", dataCtx.start);
				//target = new DataIdentifier(); // so as not to avoid null pointer
				return;
			}
		}
		
		if(!functionName.contains(".") || functionName.startsWith(DMLProgram.DEFAULT_NAMESPACE)) {
			// In global namespace, so it can be a builtin function
			// Double verification: verify passed function name is a (non-parameterized) built-in function.
			try {
				BuiltinFunctionExpression bife = BuiltinFunctionExpression.getBuiltinFunctionExpression(functionName, functCall.getParamExprs(), fileName, line, col, line, col);
				if (bife != null){
					// It is a builtin function
					setMultiAssignmentStatement(targetList, bife, ctx);
					return;
				}
				
				ParameterizedBuiltinFunctionExpression pbife = ParameterizedBuiltinFunctionExpression.getParamBuiltinFunctionExpression(functionName, functCall.getParamExprs(), fileName, line, col, line, col);
				if (pbife != null){
					// It is a parameterized builtin function
					setMultiAssignmentStatement(targetList, pbife, ctx);
					return;
				}
				
				// built-in read, rand ...
				DataExpression dbife = DataExpression.getDataExpression(functionName, functCall.getParamExprs(), fileName, line, col, line, col);
				if (dbife != null){
					setMultiAssignmentStatement(targetList, dbife, ctx);
					return;
				}
			} catch(Exception e) {
				helper.notifyErrorListeners("unable to process builtin function expression " + functionName  + ":" + e.getMessage(), ctx.start);
				return;
			}
		}
		
		setMultiAssignmentStatement(targetList, functCall, ctx);
	}
	
	private StatementBlock getStatementBlock(Statement current) {
		return PyDMLParserWrapper.getStatementBlock(current);
	}
	
	@Override
	public void exitIfStatement(IfStatementContext ctx) {
		IfStatement ifStmt = new IfStatement();
		ConditionalPredicate predicate = new ConditionalPredicate(ctx.predicate.info.expr);
		ifStmt.setConditionalPredicate(predicate);
		String fileName = helper.getCurrentFileName();
		int line = ctx.start.getLine();
		int col = ctx.start.getCharPositionInLine();
		ifStmt.setAllPositions(fileName, line, col, line, col);
		
		if(ctx.ifBody.size() > 0) {
			for(StatementContext stmtCtx : ctx.ifBody) {
				ifStmt.addStatementBlockIfBody(getStatementBlock(stmtCtx.info.stmt));
			}
			ifStmt.mergeStatementBlocksIfBody();
		}
		
		if(ctx.elseBody.size() > 0) {
			for(StatementContext stmtCtx : ctx.elseBody) {
				ifStmt.addStatementBlockElseBody(getStatementBlock(stmtCtx.info.stmt));
			}
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
		String fileName = helper.getCurrentFileName();
		int line = ctx.start.getLine();
		int col = ctx.start.getCharPositionInLine();
		whileStmt.setAllPositions(fileName, line, col, line, col);
		
		if(ctx.body.size() > 0) {
			for(StatementContext stmtCtx : ctx.body) {
				whileStmt.addStatementBlock(getStatementBlock(stmtCtx.info.stmt));
			}
			whileStmt.mergeStatementBlocks();
		}
		
		ctx.info.stmt = whileStmt;
		setFileLineColumn(ctx.info.stmt, ctx);
	}
	
	@Override
	public void exitForStatement(ForStatementContext ctx) {
		ForStatement forStmt = new ForStatement();
		String fileName = helper.getCurrentFileName();
		int line = ctx.start.getLine();
		int col = ctx.start.getCharPositionInLine();
		
		DataIdentifier iterVar = new DataIdentifier(ctx.iterVar.getText());
		HashMap<String, String> parForParamValues = null;
		Expression incrementExpr = new IntIdentifier(1, fileName, line, col, line, col);
		if(ctx.iterPred.info.increment != null) {
			incrementExpr = ctx.iterPred.info.increment;
		}
		IterablePredicate predicate = new IterablePredicate(iterVar, ctx.iterPred.info.from, ctx.iterPred.info.to, incrementExpr, parForParamValues, fileName, line, col, line, col);
		forStmt.setPredicate(predicate);
		
		if(ctx.body.size() > 0) {
			for(StatementContext stmtCtx : ctx.body) {
				forStmt.addStatementBlock(getStatementBlock(stmtCtx.info.stmt));
			}
			forStmt.mergeStatementBlocks();
		}
		ctx.info.stmt = forStmt;
		setFileLineColumn(ctx.info.stmt, ctx);
	}

	@Override
	public void exitParForStatement(ParForStatementContext ctx) {
		ParForStatement parForStmt = new ParForStatement();
		String fileName = helper.getCurrentFileName();
		int line = ctx.start.getLine();
		int col = ctx.start.getCharPositionInLine();
		
		DataIdentifier iterVar = new DataIdentifier(ctx.iterVar.getText());
		HashMap<String, String> parForParamValues = new HashMap<String, String>();
		if(ctx.parForParams != null && ctx.parForParams.size() > 0) {
			for(StrictParameterizedExpressionContext parForParamCtx : ctx.parForParams) {
				parForParamValues.put(parForParamCtx.paramName.getText(), parForParamCtx.paramVal.getText());
			}
		}
		
		Expression incrementExpr = new IntIdentifier(1, fileName, line, col, line, col);
		
		if( ctx.iterPred.info.increment != null ) {
			incrementExpr = ctx.iterPred.info.increment;
		}
		IterablePredicate predicate = new IterablePredicate(iterVar, ctx.iterPred.info.from, ctx.iterPred.info.to, incrementExpr, parForParamValues, fileName, line, col, line, col);
		parForStmt.setPredicate(predicate);
		if(ctx.body.size() > 0) {
			for(StatementContext stmtCtx : ctx.body) {
				parForStmt.addStatementBlock(getStatementBlock(stmtCtx.info.stmt));
			}
			parForStmt.mergeStatementBlocks();
		}
		ctx.info.stmt = parForStmt;
		setFileLineColumn(ctx.info.stmt, ctx);
	}
	
	
	

	// ----------------------------------------------------------------------
	@Override
	public void exitMl_type(Ml_typeContext ctx) { }
	
	private ArrayList<DataIdentifier> getFunctionParameters(List<TypedArgNoAssignContext> ctx) {
		ArrayList<DataIdentifier> retVal = new ArrayList<DataIdentifier>();
		for(TypedArgNoAssignContext paramCtx : ctx) {
			DataIdentifier dataId = new DataIdentifier(paramCtx.paramName.getText());
			String dataType = null;
			String valueType = null;
			
			if(paramCtx.paramType == null || paramCtx.paramType.dataType() == null 
					|| paramCtx.paramType.dataType().getText() == null || paramCtx.paramType.dataType().getText().isEmpty()) {
				dataType = "scalar";
			}
			else {
				dataType = paramCtx.paramType.dataType().getText();
			}
			
			if(dataType.compareTo("matrix") == 0) {
				// matrix
				dataId.setDataType(DataType.MATRIX);
			}
			else if(dataType.compareTo("scalar") == 0) {
				// scalar
				dataId.setDataType(DataType.SCALAR);
			}
			else {
				helper.notifyErrorListeners("invalid datatype " + dataType, paramCtx.start);
				return null;
			}
			
			valueType = paramCtx.paramType.valueType().getText();
			if(valueType.compareTo("int") == 0) {
				dataId.setValueType(ValueType.INT);
			}
			else if(valueType.compareTo("str") == 0) {
				dataId.setValueType(ValueType.STRING);
			}
			else if(valueType.compareTo("bool") == 0) {
				dataId.setValueType(ValueType.BOOLEAN);
			}
			else if(valueType.compareTo("float") == 0) {
				dataId.setValueType(ValueType.DOUBLE);
			}
			else {
				helper.notifyErrorListeners("invalid valuetype " + valueType, paramCtx.start);
				return null;
			}
			retVal.add(dataId);
		}
		return retVal;
	}
	
	@Override
	public void exitInternalFunctionDefExpression(InternalFunctionDefExpressionContext ctx) {
		FunctionStatement functionStmt = new FunctionStatement();
		
		ArrayList<DataIdentifier> functionInputs  = getFunctionParameters(ctx.inputParams);
		functionStmt.setInputParams(functionInputs);
		
		// set function outputs
		ArrayList<DataIdentifier> functionOutputs = getFunctionParameters(ctx.outputParams);
		functionStmt.setOutputParams(functionOutputs);
		
		// set function name
		functionStmt.setName(ctx.name.getText());
		
		
		if(ctx.body.size() > 0) {
			// handle function body
			// Create arraylist of one statement block
			ArrayList<StatementBlock> body = new ArrayList<StatementBlock>();
			for(StatementContext stmtCtx : ctx.body) {
				body.add(getStatementBlock(stmtCtx.info.stmt));
			}
			functionStmt.setBody(body);
			functionStmt.mergeStatementBlocks();
		}
		else {
			helper.notifyErrorListeners("functions with no statements are not allowed", ctx.start);
			return;
		}
		
		ctx.info.stmt = functionStmt;
		setFileLineColumn(ctx.info.stmt, ctx);
		ctx.info.functionName = ctx.name.getText();
	}
	
	@Override
	public void exitExternalFunctionDefExpression(ExternalFunctionDefExpressionContext ctx) {
		ExternalFunctionStatement functionStmt = new ExternalFunctionStatement();
		
		ArrayList<DataIdentifier> functionInputs  = getFunctionParameters(ctx.inputParams);
		functionStmt.setInputParams(functionInputs);
		
		// set function outputs
		ArrayList<DataIdentifier> functionOutputs = getFunctionParameters(ctx.outputParams);
		functionStmt.setOutputParams(functionOutputs);
		
		// set function name
		functionStmt.setName(ctx.name.getText());
		
		// set other parameters
		HashMap<String, String> otherParams = new HashMap<String,String>();
		boolean atleastOneClassName = false;
		for(StrictParameterizedKeyValueStringContext otherParamCtx : ctx.otherParams){
			String paramName = otherParamCtx.paramName.getText();
			String val = "";
			String text = otherParamCtx.paramVal.getText();
			// First unquote the string
			if(	(text.startsWith("\"") && text.endsWith("\"")) ||
				(text.startsWith("\'") && text.endsWith("\'"))) {
				if(text.length() > 2) {
					val = text.substring(1, text.length()-1);
				}
				// Empty value allowed
			}
			else {
				helper.notifyErrorListeners("the value of user parameter for external function should be of type str", ctx.start);
				return;
			}
			otherParams.put(paramName, val);
			if(paramName.compareTo("classname") == 0) {
				atleastOneClassName = true;
			}
		}
		functionStmt.setOtherParams(otherParams);
		if(!atleastOneClassName) {
			helper.notifyErrorListeners("the parameter \'className\' needs to be passed for defExternal", ctx.start);
			return;
		}
				
		ctx.info.stmt = functionStmt;
		setFileLineColumn(ctx.info.stmt, ctx);
		ctx.info.functionName = ctx.name.getText();
	}
	
	
	@Override
	public void exitPathStatement(PathStatementContext ctx) {
		PathStatement stmt = new PathStatement(ctx.pathValue.getText());
		String filePath = ctx.pathValue.getText();
		if((filePath.startsWith("\"") && filePath.endsWith("\"")) || 
				filePath.startsWith("'") && filePath.endsWith("'")) {	
			filePath = filePath.substring(1, filePath.length()-1);
		}
		
		_workingDir = filePath;
		ctx.info.stmt = stmt;
	}
	
	@Override
	public void exitIfdefAssignmentStatement(IfdefAssignmentStatementContext ctx) {
		if(!ctx.commandLineParam.getText().startsWith("$")) {
			helper.notifyErrorListeners("the first argument of ifdef function should be a commandline argument parameter (which starts with $)", ctx.commandLineParam.start);
			return;
		}
		
		if(ctx.targetList == null || ctx.targetList.size() != 1) {
			helper.notifyErrorListeners("incorrect parsing for ifdef function", ctx.start);
			return;
		}
		String targetListText = ctx.targetList.get(0).getText(); 
		if(targetListText.startsWith("$")) {
			helper.notifyErrorListeners("lhs of ifdef function cannot be a commandline parameters. Use local variable instead", ctx.start);
			return;
		}
		
		DataIdentifier target = null; 
		if(ctx.targetList.get(0).dataInfo.expr instanceof DataIdentifier) {
			target = (DataIdentifier) ctx.targetList.get(0).dataInfo.expr;
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
			
			int line = ctx.start.getLine();
			int col = ctx.start.getCharPositionInLine();
			try {
				ctx.info.stmt = new AssignmentStatement(target, source, line, col, line, col);
				setFileLineColumn(ctx.info.stmt, ctx);
			} catch (LanguageException e) {
				helper.notifyErrorListeners("invalid assignment for ifdef function", ctx.targetList.get(0).start);
				return;
			} 
			
		}
		else {
			helper.notifyErrorListeners("incorrect lvalue in ifdef function... strange", ctx.targetList.get(0).start);
			return;
		}
		
	}
	
	// ----------------------------------------------------------------------
	@Override
	public void exitParameterizedExpression(ParameterizedExpressionContext ctx) { }


	@Override
	public void exitStrictParameterizedExpression(StrictParameterizedExpressionContext ctx) { }

	@Override
	public void exitTypedArgNoAssign(TypedArgNoAssignContext ctx) { }
	@Override
	public void enterIfdefAssignmentStatement(IfdefAssignmentStatementContext ctx) { }
	@Override
	public void enterMatrixDataTypeCheck(MatrixDataTypeCheckContext ctx) { }
	@Override
	public void exitMatrixDataTypeCheck(MatrixDataTypeCheckContext ctx) {
		if(		ctx.ID().getText().compareTo("matrix") == 0 
				|| ctx.ID().getText().compareTo("scalar") == 0
				) {
			// Do nothing
		}
		else if(ctx.ID().getText().compareTo("Matrix") == 0)
			helper.notifyErrorListeners("incorrect datatype (Hint: use matrix instead of Matrix)", ctx.start);
		else if(ctx.ID().getText().compareTo("Scalar") == 0)
			helper.notifyErrorListeners("incorrect datatype (Hint: use scalar instead of Scalar)", ctx.start);
		else if(		ctx.ID().getText().compareTo("int") == 0 
				|| ctx.ID().getText().compareTo("str") == 0
				|| ctx.ID().getText().compareTo("bool") == 0
				|| ctx.ID().getText().compareTo("float") == 0
				) {
			helper.notifyErrorListeners("expected datatype but found a valuetype (Hint: use matrix or scalar instead of " + ctx.ID().getText() + ")", ctx.start);
		}
		else {
			helper.notifyErrorListeners("incorrect datatype (expected matrix or scalar)", ctx.start);
		}
	}
	
	@Override
	public void enterBuiltinFunctionExpression(BuiltinFunctionExpressionContext ctx) {}
	@Override
	public void enterStrictParameterizedKeyValueString(StrictParameterizedKeyValueStringContext ctx) { }
	@Override
	public void exitStrictParameterizedKeyValueString(StrictParameterizedKeyValueStringContext ctx) {}
	@Override
	public void enterIterablePredicateColonExpression(IterablePredicateColonExpressionContext ctx) {}
	@Override
	public void enterIterablePredicateSeqExpression(IterablePredicateSeqExpressionContext ctx) { }
	
	@Override
	public void exitIterablePredicateColonExpression(IterablePredicateColonExpressionContext ctx) {
		ctx.info.from = ctx.from.info.expr;
		ctx.info.to = ctx.to.info.expr;
		ctx.info.increment = null;
	}
	
	@Override
	public void exitIterablePredicateSeqExpression(IterablePredicateSeqExpressionContext ctx) {
		if(ctx.ID().getText().compareTo("range") != 0) {
			helper.notifyErrorListeners("incorrect function:\'" + ctx.ID().getText() + "\'. expected \'range\'", ctx.start);
			return;
		}
		ctx.info.from = ctx.from.info.expr;
		ctx.info.to = ctx.to.info.expr;
		ctx.info.increment = ctx.increment.info.expr;		
	}
	
	@Override
	public void enterConstFalseExpression(ConstFalseExpressionContext ctx) { }
	@Override
	public void enterConstTrueExpression(ConstTrueExpressionContext ctx) { }
	
	@Override
	public void exitConstFalseExpression(ConstFalseExpressionContext ctx) {
		boolean val = false;
		int linePosition = ctx.start.getLine();
		int charPosition = ctx.start.getCharPositionInLine();
		ctx.info.expr = new BooleanIdentifier(val, helper.getCurrentFileName(), linePosition, charPosition, linePosition, charPosition);
		setFileLineColumn(ctx.info.expr, ctx);
	}
	
	
	@Override
	public void exitConstTrueExpression(ConstTrueExpressionContext ctx) {
		boolean val = true;
		int linePosition = ctx.start.getLine();
		int charPosition = ctx.start.getCharPositionInLine();
		ctx.info.expr = new BooleanIdentifier(val, helper.getCurrentFileName(), linePosition, charPosition, linePosition, charPosition);
		setFileLineColumn(ctx.info.expr, ctx);
	}
	
	@Override
	public void exitIgnoreNewLine(IgnoreNewLineContext ctx) {
		// Introduce empty StatementInfo 
		// This is later ignored by PyDMLParserWrapper
		try {
			ctx.info.stmt = new AssignmentStatement(null, null, 0, 0, 0, 0);
			ctx.info.stmt.setEmptyNewLineStatement(true);
		} catch (LanguageException e) {
			e.printStackTrace();
		}
		
	}
	@Override
	public void enterValueDataTypeCheck(ValueDataTypeCheckContext ctx) { }
	@Override
	public void exitValueDataTypeCheck(ValueDataTypeCheckContext ctx) {
		if(		ctx.ID().getText().compareTo("int") == 0 
				|| ctx.ID().getText().compareTo("str") == 0
				|| ctx.ID().getText().compareTo("bool") == 0
				|| ctx.ID().getText().compareTo("float") == 0
				) {
			// Do nothing
		}
		else if(ctx.ID().getText().compareTo("integer") == 0)
			helper.notifyErrorListeners("incorrect valuetype (Hint: use int instead of integer)", ctx.start);
		else if(ctx.ID().getText().compareTo("double") == 0)
			helper.notifyErrorListeners("incorrect valuetype (Hint: use float instead of double)", ctx.start);
		else if(ctx.ID().getText().compareTo("boolean") == 0)
			helper.notifyErrorListeners("incorrect valuetype (Hint: use bool instead of boolean)", ctx.start);
		else if(ctx.ID().getText().compareTo("string") == 0)
			helper.notifyErrorListeners("incorrect valuetype (Hint: use str instead of string)", ctx.start);
		else {
			helper.notifyErrorListeners("incorrect valuetype (expected int, str, bool or float)", ctx.start);
		}
	}
}
