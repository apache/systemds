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
package org.apache.sysml.parser.dml;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Map;
import java.util.Random;
import java.util.Set;

import org.antlr.v4.runtime.ParserRuleContext;
import org.antlr.v4.runtime.Token;
import org.antlr.v4.runtime.TokenStreamRewriter;
import org.antlr.v4.runtime.tree.ErrorNode;
import org.antlr.v4.runtime.tree.TerminalNode;
import org.apache.sysml.parser.Expression;
import org.apache.sysml.parser.ParameterExpression;
import org.apache.sysml.parser.common.CommonSyntacticValidator;
import org.apache.sysml.parser.common.CustomErrorListener;
import org.apache.sysml.parser.dml.DmlParser.AccumulatorAssignmentStatementContext;
import org.apache.sysml.parser.dml.DmlParser.AddSubExpressionContext;
import org.apache.sysml.parser.dml.DmlParser.AssignmentStatementContext;
import org.apache.sysml.parser.dml.DmlParser.AtomicExpressionContext;
import org.apache.sysml.parser.dml.DmlParser.BooleanAndExpressionContext;
import org.apache.sysml.parser.dml.DmlParser.BooleanNotExpressionContext;
import org.apache.sysml.parser.dml.DmlParser.BooleanOrExpressionContext;
import org.apache.sysml.parser.dml.DmlParser.BuiltinFunctionExpressionContext;
import org.apache.sysml.parser.dml.DmlParser.CommandlineParamExpressionContext;
import org.apache.sysml.parser.dml.DmlParser.CommandlinePositionExpressionContext;
import org.apache.sysml.parser.dml.DmlParser.ConstDoubleIdExpressionContext;
import org.apache.sysml.parser.dml.DmlParser.ConstFalseExpressionContext;
import org.apache.sysml.parser.dml.DmlParser.ConstIntIdExpressionContext;
import org.apache.sysml.parser.dml.DmlParser.ConstStringIdExpressionContext;
import org.apache.sysml.parser.dml.DmlParser.ConstTrueExpressionContext;
import org.apache.sysml.parser.dml.DmlParser.DataIdExpressionContext;
import org.apache.sysml.parser.dml.DmlParser.ExternalFunctionDefExpressionContext;
import org.apache.sysml.parser.dml.DmlParser.ForStatementContext;
import org.apache.sysml.parser.dml.DmlParser.FunctionCallAssignmentStatementContext;
import org.apache.sysml.parser.dml.DmlParser.FunctionCallMultiAssignmentStatementContext;
import org.apache.sysml.parser.dml.DmlParser.IfStatementContext;
import org.apache.sysml.parser.dml.DmlParser.IfdefAssignmentStatementContext;
import org.apache.sysml.parser.dml.DmlParser.ImportStatementContext;
import org.apache.sysml.parser.dml.DmlParser.IndexedExpressionContext;
import org.apache.sysml.parser.dml.DmlParser.InternalFunctionDefExpressionContext;
import org.apache.sysml.parser.dml.DmlParser.IterablePredicateColonExpressionContext;
import org.apache.sysml.parser.dml.DmlParser.IterablePredicateSeqExpressionContext;
import org.apache.sysml.parser.dml.DmlParser.MatrixDataTypeCheckContext;
import org.apache.sysml.parser.dml.DmlParser.MatrixMulExpressionContext;
import org.apache.sysml.parser.dml.DmlParser.Ml_typeContext;
import org.apache.sysml.parser.dml.DmlParser.ModIntDivExpressionContext;
import org.apache.sysml.parser.dml.DmlParser.MultDivExpressionContext;
import org.apache.sysml.parser.dml.DmlParser.MultiIdExpressionContext;
import org.apache.sysml.parser.dml.DmlParser.ParForStatementContext;
import org.apache.sysml.parser.dml.DmlParser.ParameterizedExpressionContext;
import org.apache.sysml.parser.dml.DmlParser.PathStatementContext;
import org.apache.sysml.parser.dml.DmlParser.PowerExpressionContext;
import org.apache.sysml.parser.dml.DmlParser.ProgramrootContext;
import org.apache.sysml.parser.dml.DmlParser.RelationalExpressionContext;
import org.apache.sysml.parser.dml.DmlParser.SimpleDataIdentifierExpressionContext;
import org.apache.sysml.parser.dml.DmlParser.StatementContext;
import org.apache.sysml.parser.dml.DmlParser.StrictParameterizedExpressionContext;
import org.apache.sysml.parser.dml.DmlParser.StrictParameterizedKeyValueStringContext;
import org.apache.sysml.parser.dml.DmlParser.TypedArgAssignContext;
import org.apache.sysml.parser.dml.DmlParser.TypedArgNoAssignContext;
import org.apache.sysml.parser.dml.DmlParser.UnaryExpressionContext;
import org.apache.sysml.parser.dml.DmlParser.ValueTypeContext;
import org.apache.sysml.parser.dml.DmlParser.WhileStatementContext;

/**
 * This class is used to generate inline-able methods.
 * It does so in two phases:
 * - Phase 1. Rewriting phase where local variables are rewritten by adding a prefix.
 * - Phase 2. Capture the body of the functions using InlineableMethods class 
 */
public class InlineHelper extends CommonSyntacticValidator implements DmlListener {
	final static String ARG_PREFIX;
	static {
		Random rand = new Random();
		ARG_PREFIX = "INTERNAL_PREFIX_" + Math.abs(rand.nextLong()) + "_" + Math.abs(rand.nextLong()) + "_"; 
	}
	public HashMap<String, InlineableMethods> inlineMap = new HashMap<>();
	TokenStreamRewriter rewriter;
	
	// Set internally
	HashSet<String> variables = new HashSet<>();
	String currentFunction = null;
	boolean isRewritePhase;
	
	public InlineHelper(CustomErrorListener errorListener, Map<String, String> argVals, String sourceNamespace,
			Set<String> prepFunctions, TokenStreamRewriter rewriter1) {
		super(errorListener, argVals, sourceNamespace, prepFunctions);
		rewriter = rewriter1;
	}
	
	void setPhase(boolean isRewritePhase1) {
		isRewritePhase = isRewritePhase1;
	}
	

	@Override
	public void enterInternalFunctionDefExpression(InternalFunctionDefExpressionContext ctx) {
		currentFunction = ctx.name.getText();
		variables.clear();
	}
	
	@Override
	public void exitInternalFunctionDefExpression(InternalFunctionDefExpressionContext ctx) {
		if(!isRewritePhase) {
			StringBuilder sb = new StringBuilder();
			for(StatementContext stmt : ctx.body) {
				sb.append(stmt.getText());
				sb.append("\n");
			}
			ArrayList<String> inputArgs = new ArrayList<>(); 
			for(TypedArgAssignContext in : ctx.inputParams) {
				inputArgs.add(ARG_PREFIX + in.paramName.getText());
			}
			ArrayList<String> retVariables = new ArrayList<>();
			for(TypedArgNoAssignContext out : ctx.outputParams) {
				retVariables.add(ARG_PREFIX + out.paramName.getText());
			}
			
			inlineMap.put(currentFunction, new InlineableMethods(currentFunction, sb.toString(), variables, inputArgs, retVariables));
		}
		currentFunction = null;
		variables.clear();
	}
	
	@Override
	public void enterIndexedExpression(IndexedExpressionContext ctx) {
		if(currentFunction != null && isRewritePhase) {
			rewriter.insertBefore(ctx.name, " " + ARG_PREFIX);
		}
	}
	
	@Override
	public void exitIndexedExpression(IndexedExpressionContext ctx) {
		if(currentFunction != null)
			variables.add(ctx.name.getText());
	}
	
	@Override
	public void enterSimpleDataIdentifierExpression(SimpleDataIdentifierExpressionContext ctx) {
		if(currentFunction != null && isRewritePhase) {
			rewriter.insertBefore(ctx.start, " " + ARG_PREFIX);
			rewriter.insertAfter(ctx.stop, " ");
		}
	}
	
	@Override
	public void exitSimpleDataIdentifierExpression(SimpleDataIdentifierExpressionContext ctx) {
		if(currentFunction != null)
			variables.add(ctx.getText());
	}
	
	@Override
	public void enterForStatement(ForStatementContext ctx) {
		if(currentFunction != null && isRewritePhase) {
			rewriter.insertBefore(ctx.iterVar, " " + ARG_PREFIX);
			rewriter.insertAfter(ctx.iterVar, " ");
		}
	}
	
	@Override
	public void enterParForStatement(ParForStatementContext ctx) {
		if(currentFunction != null && isRewritePhase) {
			rewriter.insertBefore(ctx.iterVar, " " + ARG_PREFIX);
			rewriter.insertAfter(ctx.iterVar, " ");
		}
	}
	
	@Override
	public void exitForStatement(ForStatementContext ctx) {
		if(currentFunction != null)
			variables.add(ctx.iterVar.getText());
		if(currentFunction != null && isRewritePhase) {
			if(ctx.body != null && ctx.body.size() > 0)
			rewriter.insertBefore(ctx.body.get(0).start, "\n");
			rewriter.insertAfter(ctx.stop, "\n");
		}
	}
	
	@Override
	public void exitParForStatement(ParForStatementContext ctx) {
		if(currentFunction != null)
			variables.add(ctx.iterVar.getText());
		if(currentFunction != null && isRewritePhase) {
			if(ctx.body != null && ctx.body.size() > 0)
				rewriter.insertBefore(ctx.body.get(0).start, "\n");
			rewriter.insertAfter(ctx.stop, "\n");
		}
	}
	

	@Override
	protected ConvertedDMLSyntax convertToDMLSyntax(ParserRuleContext ctx, String namespace, String functionName,
			ArrayList<ParameterExpression> paramExpression, Token fnName) {
		
		return null;
	}

	@Override
	public void enterAccumulatorAssignmentStatement(AccumulatorAssignmentStatementContext ctx) {
		
		
	}

	@Override
	public void enterAddSubExpression(AddSubExpressionContext ctx) {
		
		
	}

	@Override
	public void enterAssignmentStatement(AssignmentStatementContext ctx) {
		
		
	}

	@Override
	public void enterAtomicExpression(AtomicExpressionContext ctx) {
		
		
	}

	@Override
	public void enterBooleanAndExpression(BooleanAndExpressionContext ctx) {
		
		
	}

	@Override
	public void enterBooleanNotExpression(BooleanNotExpressionContext ctx) {
		
		
	}

	@Override
	public void enterBooleanOrExpression(BooleanOrExpressionContext ctx) {
		
		
	}

	@Override
	public void enterBuiltinFunctionExpression(BuiltinFunctionExpressionContext ctx) {
		
		
	}

	@Override
	public void enterCommandlineParamExpression(CommandlineParamExpressionContext ctx) {
		
		
	}

	@Override
	public void enterCommandlinePositionExpression(CommandlinePositionExpressionContext ctx) {
		
		
	}

	@Override
	public void enterConstDoubleIdExpression(ConstDoubleIdExpressionContext ctx) {
		
		
	}

	@Override
	public void enterConstFalseExpression(ConstFalseExpressionContext ctx) {
		
		
	}

	@Override
	public void enterConstIntIdExpression(ConstIntIdExpressionContext ctx) {
		
		
	}

	@Override
	public void enterConstStringIdExpression(ConstStringIdExpressionContext ctx) {
		
		
	}

	@Override
	public void enterConstTrueExpression(ConstTrueExpressionContext ctx) {
		
		
	}

	@Override
	public void enterDataIdExpression(DataIdExpressionContext ctx) {
		
		
	}

	@Override
	public void enterEveryRule(ParserRuleContext arg0) {
		
		
	}

	@Override
	public void enterExternalFunctionDefExpression(ExternalFunctionDefExpressionContext ctx) {
		
		
	}
	

	@Override
	public void enterFunctionCallAssignmentStatement(FunctionCallAssignmentStatementContext ctx) {
		
		
	}

	@Override
	public void enterFunctionCallMultiAssignmentStatement(FunctionCallMultiAssignmentStatementContext ctx) {
		
		
	}

	@Override
	public void enterIfdefAssignmentStatement(IfdefAssignmentStatementContext ctx) {
		
		
	}

	@Override
	public void enterIfStatement(IfStatementContext ctx) {
		
		
	}

	@Override
	public void enterImportStatement(ImportStatementContext ctx) {
		
		
	}

	@Override
	public void enterIterablePredicateColonExpression(IterablePredicateColonExpressionContext ctx) {
		
		
	}

	@Override
	public void enterIterablePredicateSeqExpression(IterablePredicateSeqExpressionContext ctx) {
		
		
	}

	@Override
	public void enterMatrixDataTypeCheck(MatrixDataTypeCheckContext ctx) {
		
		
	}

	@Override
	public void enterMatrixMulExpression(MatrixMulExpressionContext ctx) {
		
		
	}

	@Override
	public void enterMl_type(Ml_typeContext ctx) {
		
		
	}

	@Override
	public void enterModIntDivExpression(ModIntDivExpressionContext ctx) {
		
		
	}

	@Override
	public void enterMultDivExpression(MultDivExpressionContext ctx) {
		
		
	}

	@Override
	public void enterMultiIdExpression(MultiIdExpressionContext ctx) {
		
		
	}

	@Override
	public void enterParameterizedExpression(ParameterizedExpressionContext ctx) {
		
		
	}

	@Override
	public void enterPathStatement(PathStatementContext ctx) {
		
		
	}

	@Override
	public void enterPowerExpression(PowerExpressionContext ctx) {
		
		
	}

	@Override
	public void enterProgramroot(ProgramrootContext ctx) {
		
		
	}

	@Override
	public void enterRelationalExpression(RelationalExpressionContext ctx) {
		
		
	}

	@Override
	public void enterStrictParameterizedExpression(StrictParameterizedExpressionContext ctx) {
		
		
	}

	@Override
	public void enterStrictParameterizedKeyValueString(StrictParameterizedKeyValueStringContext ctx) {
		
		
	}

	@Override
	public void enterTypedArgAssign(TypedArgAssignContext ctx) {
		
		
	}

	@Override
	public void enterTypedArgNoAssign(TypedArgNoAssignContext ctx) {
		
		
	}

	@Override
	public void enterUnaryExpression(UnaryExpressionContext ctx) {
		
		
	}

	@Override
	public void enterValueType(ValueTypeContext ctx) {
		
		
	}

	@Override
	public void enterWhileStatement(WhileStatementContext ctx) {
		
		
	}

	@Override
	public void exitAccumulatorAssignmentStatement(AccumulatorAssignmentStatementContext ctx) {
		if(currentFunction != null && isRewritePhase) {
			rewriter.insertAfter(ctx.stop, ";\n");
		}
		
	}

	@Override
	public void exitAddSubExpression(AddSubExpressionContext ctx) {
		
		
	}

	@Override
	public void exitAssignmentStatement(AssignmentStatementContext ctx) {
		if(currentFunction != null && isRewritePhase) {
			rewriter.insertAfter(ctx.stop, ";\n");
		}
		
	}

	@Override
	public void exitAtomicExpression(AtomicExpressionContext ctx) {
		
		
	}

	@Override
	public void exitBooleanAndExpression(BooleanAndExpressionContext ctx) {
		
		
	}

	@Override
	public void exitBooleanNotExpression(BooleanNotExpressionContext ctx) {
		
		
	}

	@Override
	public void exitBooleanOrExpression(BooleanOrExpressionContext ctx) {
		
		
	}

	@Override
	public void exitBuiltinFunctionExpression(BuiltinFunctionExpressionContext ctx) {
		
		
	}

	@Override
	public void exitCommandlineParamExpression(CommandlineParamExpressionContext ctx) {
		
		
	}

	@Override
	public void exitCommandlinePositionExpression(CommandlinePositionExpressionContext ctx) {
		
		
	}

	@Override
	public void exitConstDoubleIdExpression(ConstDoubleIdExpressionContext ctx) {
		
		
	}

	@Override
	public void exitConstFalseExpression(ConstFalseExpressionContext ctx) {
		
		
	}

	@Override
	public void exitConstIntIdExpression(ConstIntIdExpressionContext ctx) {
		
		
	}

	@Override
	public void exitConstStringIdExpression(ConstStringIdExpressionContext ctx) {
		
		
	}

	@Override
	public void exitConstTrueExpression(ConstTrueExpressionContext ctx) {
		
		
	}

	@Override
	public void exitDataIdExpression(DataIdExpressionContext ctx) {
		
		
	}

	@Override
	public void exitEveryRule(ParserRuleContext arg0) {
		
		
	}

	@Override
	public void exitExternalFunctionDefExpression(ExternalFunctionDefExpressionContext ctx) {
		
		
	}

	

	@Override
	public void exitFunctionCallAssignmentStatement(FunctionCallAssignmentStatementContext ctx) {
		if(currentFunction != null && isRewritePhase) {
			rewriter.insertAfter(ctx.stop, ";\n");
		}
		
	}

	@Override
	public void exitFunctionCallMultiAssignmentStatement(FunctionCallMultiAssignmentStatementContext ctx) {
		if(currentFunction != null && isRewritePhase) {
			rewriter.insertAfter(ctx.stop, ";\n");
		}
		
	}

	@Override
	public void exitIfdefAssignmentStatement(IfdefAssignmentStatementContext ctx) {
		if(currentFunction != null && isRewritePhase) {
			rewriter.insertAfter(ctx.stop, ";\n");
		}
		
	}

	@Override
	public void exitIfStatement(IfStatementContext ctx) {
		if(currentFunction != null && isRewritePhase) {
			if(ctx.ifBody != null && ctx.ifBody.size() > 0)
				rewriter.insertBefore(ctx.ifBody.get(0).start, "\n");
			if(ctx.elseBody != null && ctx.elseBody.size() > 0)
				rewriter.insertBefore(ctx.elseBody.get(0).start, "\n");
			rewriter.insertAfter(ctx.stop, "\n");
		}
		
	}

	@Override
	public void exitImportStatement(ImportStatementContext ctx) {
		if(currentFunction != null && isRewritePhase) {
			rewriter.insertAfter(ctx.stop, ";\n");
		}
		
	}

	@Override
	public void exitIterablePredicateColonExpression(IterablePredicateColonExpressionContext ctx) {
		
		
	}

	@Override
	public void exitIterablePredicateSeqExpression(IterablePredicateSeqExpressionContext ctx) {
		
		
	}

	@Override
	public void exitMatrixDataTypeCheck(MatrixDataTypeCheckContext ctx) {
		
		
	}

	@Override
	public void exitMatrixMulExpression(MatrixMulExpressionContext ctx) {
		
		
	}

	@Override
	public void exitMl_type(Ml_typeContext ctx) {
		
		
	}

	@Override
	public void exitModIntDivExpression(ModIntDivExpressionContext ctx) {
		
		
	}

	@Override
	public void exitMultDivExpression(MultDivExpressionContext ctx) {
		
		
	}

	@Override
	public void exitMultiIdExpression(MultiIdExpressionContext ctx) {
		
		
	}

	@Override
	public void exitParameterizedExpression(ParameterizedExpressionContext ctx) {
		
		
	}

	@Override
	public void exitPathStatement(PathStatementContext ctx) {
		if(currentFunction != null && isRewritePhase) {
			rewriter.insertAfter(ctx.stop, ";\n");
		}
		
	}

	@Override
	public void exitPowerExpression(PowerExpressionContext ctx) {
		
		
	}

	@Override
	public void exitProgramroot(ProgramrootContext ctx) {
		
		
	}

	@Override
	public void exitRelationalExpression(RelationalExpressionContext ctx) {
		
		
	}
	
	@Override
	public void exitStrictParameterizedExpression(StrictParameterizedExpressionContext ctx) {
		
		
	}

	@Override
	public void exitStrictParameterizedKeyValueString(StrictParameterizedKeyValueStringContext ctx) {
		
		
	}

	@Override
	public void exitTypedArgAssign(TypedArgAssignContext ctx) {
		
		
	}

	@Override
	public void exitTypedArgNoAssign(TypedArgNoAssignContext ctx) {
		
		
	}

	@Override
	public void exitUnaryExpression(UnaryExpressionContext ctx) {
		
		
	}

	@Override
	public void exitValueType(ValueTypeContext ctx) {
		
		
	}

	@Override
	public void exitWhileStatement(WhileStatementContext ctx) {
		if(currentFunction != null && isRewritePhase) {
			if(ctx.body != null && ctx.body.size() > 0)
				rewriter.insertBefore(ctx.body.get(0).start, "\n");
			rewriter.insertAfter(ctx.stop, "\n");
		}
	}

	@Override
	public String falseStringLiteral() {
		
		return null;
	}

	@Override
	protected Expression handleLanguageSpecificFunction(ParserRuleContext ctx, String functionName,
			ArrayList<ParameterExpression> paramExpressions) {
		
		return null;
	}

	@Override
	public String namespaceResolutionOp() {
		
		return null;
	}

	@Override
	public String trueStringLiteral() {
		
		return null;
	}

	@Override
	public void visitErrorNode(ErrorNode arg0) {
		
		
	}

	@Override
	public void visitTerminal(TerminalNode arg0) {
		
		
	}

}