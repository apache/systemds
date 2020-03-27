// Generated from org\tugraz\sysds\parser\dml\Dml.g4 by ANTLR 4.5.3
package org.tugraz.sysds.parser.dml;

/*
 * Modifications Copyright 2018 Graz University of Technology
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

import org.antlr.v4.runtime.tree.ParseTreeListener;

/**
 * This interface defines a complete listener for a parse tree produced by
 * {@link DmlParser}.
 */
public interface DmlListener extends ParseTreeListener {
	/**
	 * Enter a parse tree produced by {@link DmlParser#programroot}.
	 * @param ctx the parse tree
	 */
	void enterProgramroot(DmlParser.ProgramrootContext ctx);
	/**
	 * Exit a parse tree produced by {@link DmlParser#programroot}.
	 * @param ctx the parse tree
	 */
	void exitProgramroot(DmlParser.ProgramrootContext ctx);
	/**
	 * Enter a parse tree produced by the {@code ImportStatement}
	 * labeled alternative in {@link DmlParser#statement}.
	 * @param ctx the parse tree
	 */
	void enterImportStatement(DmlParser.ImportStatementContext ctx);
	/**
	 * Exit a parse tree produced by the {@code ImportStatement}
	 * labeled alternative in {@link DmlParser#statement}.
	 * @param ctx the parse tree
	 */
	void exitImportStatement(DmlParser.ImportStatementContext ctx);
	/**
	 * Enter a parse tree produced by the {@code PathStatement}
	 * labeled alternative in {@link DmlParser#statement}.
	 * @param ctx the parse tree
	 */
	void enterPathStatement(DmlParser.PathStatementContext ctx);
	/**
	 * Exit a parse tree produced by the {@code PathStatement}
	 * labeled alternative in {@link DmlParser#statement}.
	 * @param ctx the parse tree
	 */
	void exitPathStatement(DmlParser.PathStatementContext ctx);
	/**
	 * Enter a parse tree produced by the {@code FunctionCallAssignmentStatement}
	 * labeled alternative in {@link DmlParser#statement}.
	 * @param ctx the parse tree
	 */
	void enterFunctionCallAssignmentStatement(DmlParser.FunctionCallAssignmentStatementContext ctx);
	/**
	 * Exit a parse tree produced by the {@code FunctionCallAssignmentStatement}
	 * labeled alternative in {@link DmlParser#statement}.
	 * @param ctx the parse tree
	 */
	void exitFunctionCallAssignmentStatement(DmlParser.FunctionCallAssignmentStatementContext ctx);
	/**
	 * Enter a parse tree produced by the {@code FunctionCallMultiAssignmentStatement}
	 * labeled alternative in {@link DmlParser#statement}.
	 * @param ctx the parse tree
	 */
	void enterFunctionCallMultiAssignmentStatement(DmlParser.FunctionCallMultiAssignmentStatementContext ctx);
	/**
	 * Exit a parse tree produced by the {@code FunctionCallMultiAssignmentStatement}
	 * labeled alternative in {@link DmlParser#statement}.
	 * @param ctx the parse tree
	 */
	void exitFunctionCallMultiAssignmentStatement(DmlParser.FunctionCallMultiAssignmentStatementContext ctx);
	/**
	 * Enter a parse tree produced by the {@code IfdefAssignmentStatement}
	 * labeled alternative in {@link DmlParser#statement}.
	 * @param ctx the parse tree
	 */
	void enterIfdefAssignmentStatement(DmlParser.IfdefAssignmentStatementContext ctx);
	/**
	 * Exit a parse tree produced by the {@code IfdefAssignmentStatement}
	 * labeled alternative in {@link DmlParser#statement}.
	 * @param ctx the parse tree
	 */
	void exitIfdefAssignmentStatement(DmlParser.IfdefAssignmentStatementContext ctx);
	/**
	 * Enter a parse tree produced by the {@code AssignmentStatement}
	 * labeled alternative in {@link DmlParser#statement}.
	 * @param ctx the parse tree
	 */
	void enterAssignmentStatement(DmlParser.AssignmentStatementContext ctx);
	/**
	 * Exit a parse tree produced by the {@code AssignmentStatement}
	 * labeled alternative in {@link DmlParser#statement}.
	 * @param ctx the parse tree
	 */
	void exitAssignmentStatement(DmlParser.AssignmentStatementContext ctx);
	/**
	 * Enter a parse tree produced by the {@code AccumulatorAssignmentStatement}
	 * labeled alternative in {@link DmlParser#statement}.
	 * @param ctx the parse tree
	 */
	void enterAccumulatorAssignmentStatement(DmlParser.AccumulatorAssignmentStatementContext ctx);
	/**
	 * Exit a parse tree produced by the {@code AccumulatorAssignmentStatement}
	 * labeled alternative in {@link DmlParser#statement}.
	 * @param ctx the parse tree
	 */
	void exitAccumulatorAssignmentStatement(DmlParser.AccumulatorAssignmentStatementContext ctx);
	/**
	 * Enter a parse tree produced by the {@code IfStatement}
	 * labeled alternative in {@link DmlParser#statement}.
	 * @param ctx the parse tree
	 */
	void enterIfStatement(DmlParser.IfStatementContext ctx);
	/**
	 * Exit a parse tree produced by the {@code IfStatement}
	 * labeled alternative in {@link DmlParser#statement}.
	 * @param ctx the parse tree
	 */
	void exitIfStatement(DmlParser.IfStatementContext ctx);
	/**
	 * Enter a parse tree produced by the {@code ForStatement}
	 * labeled alternative in {@link DmlParser#statement}.
	 * @param ctx the parse tree
	 */
	void enterForStatement(DmlParser.ForStatementContext ctx);
	/**
	 * Exit a parse tree produced by the {@code ForStatement}
	 * labeled alternative in {@link DmlParser#statement}.
	 * @param ctx the parse tree
	 */
	void exitForStatement(DmlParser.ForStatementContext ctx);
	/**
	 * Enter a parse tree produced by the {@code ParForStatement}
	 * labeled alternative in {@link DmlParser#statement}.
	 * @param ctx the parse tree
	 */
	void enterParForStatement(DmlParser.ParForStatementContext ctx);
	/**
	 * Exit a parse tree produced by the {@code ParForStatement}
	 * labeled alternative in {@link DmlParser#statement}.
	 * @param ctx the parse tree
	 */
	void exitParForStatement(DmlParser.ParForStatementContext ctx);
	/**
	 * Enter a parse tree produced by the {@code WhileStatement}
	 * labeled alternative in {@link DmlParser#statement}.
	 * @param ctx the parse tree
	 */
	void enterWhileStatement(DmlParser.WhileStatementContext ctx);
	/**
	 * Exit a parse tree produced by the {@code WhileStatement}
	 * labeled alternative in {@link DmlParser#statement}.
	 * @param ctx the parse tree
	 */
	void exitWhileStatement(DmlParser.WhileStatementContext ctx);
	/**
	 * Enter a parse tree produced by the {@code IterablePredicateColonExpression}
	 * labeled alternative in {@link DmlParser#iterablePredicate}.
	 * @param ctx the parse tree
	 */
	void enterIterablePredicateColonExpression(DmlParser.IterablePredicateColonExpressionContext ctx);
	/**
	 * Exit a parse tree produced by the {@code IterablePredicateColonExpression}
	 * labeled alternative in {@link DmlParser#iterablePredicate}.
	 * @param ctx the parse tree
	 */
	void exitIterablePredicateColonExpression(DmlParser.IterablePredicateColonExpressionContext ctx);
	/**
	 * Enter a parse tree produced by the {@code IterablePredicateSeqExpression}
	 * labeled alternative in {@link DmlParser#iterablePredicate}.
	 * @param ctx the parse tree
	 */
	void enterIterablePredicateSeqExpression(DmlParser.IterablePredicateSeqExpressionContext ctx);
	/**
	 * Exit a parse tree produced by the {@code IterablePredicateSeqExpression}
	 * labeled alternative in {@link DmlParser#iterablePredicate}.
	 * @param ctx the parse tree
	 */
	void exitIterablePredicateSeqExpression(DmlParser.IterablePredicateSeqExpressionContext ctx);
	/**
	 * Enter a parse tree produced by the {@code InternalFunctionDefExpression}
	 * labeled alternative in {@link DmlParser#functionStatement}.
	 * @param ctx the parse tree
	 */
	void enterInternalFunctionDefExpression(DmlParser.InternalFunctionDefExpressionContext ctx);
	/**
	 * Exit a parse tree produced by the {@code InternalFunctionDefExpression}
	 * labeled alternative in {@link DmlParser#functionStatement}.
	 * @param ctx the parse tree
	 */
	void exitInternalFunctionDefExpression(DmlParser.InternalFunctionDefExpressionContext ctx);
	/**
	 * Enter a parse tree produced by the {@code ExternalFunctionDefExpression}
	 * labeled alternative in {@link DmlParser#functionStatement}.
	 * @param ctx the parse tree
	 */
	void enterExternalFunctionDefExpression(DmlParser.ExternalFunctionDefExpressionContext ctx);
	/**
	 * Exit a parse tree produced by the {@code ExternalFunctionDefExpression}
	 * labeled alternative in {@link DmlParser#functionStatement}.
	 * @param ctx the parse tree
	 */
	void exitExternalFunctionDefExpression(DmlParser.ExternalFunctionDefExpressionContext ctx);
	/**
	 * Enter a parse tree produced by the {@code IndexedExpression}
	 * labeled alternative in {@link DmlParser#dataIdentifier}.
	 * @param ctx the parse tree
	 */
	void enterIndexedExpression(DmlParser.IndexedExpressionContext ctx);
	/**
	 * Exit a parse tree produced by the {@code IndexedExpression}
	 * labeled alternative in {@link DmlParser#dataIdentifier}.
	 * @param ctx the parse tree
	 */
	void exitIndexedExpression(DmlParser.IndexedExpressionContext ctx);
	/**
	 * Enter a parse tree produced by the {@code SimpleDataIdentifierExpression}
	 * labeled alternative in {@link DmlParser#dataIdentifier}.
	 * @param ctx the parse tree
	 */
	void enterSimpleDataIdentifierExpression(DmlParser.SimpleDataIdentifierExpressionContext ctx);
	/**
	 * Exit a parse tree produced by the {@code SimpleDataIdentifierExpression}
	 * labeled alternative in {@link DmlParser#dataIdentifier}.
	 * @param ctx the parse tree
	 */
	void exitSimpleDataIdentifierExpression(DmlParser.SimpleDataIdentifierExpressionContext ctx);
	/**
	 * Enter a parse tree produced by the {@code CommandlineParamExpression}
	 * labeled alternative in {@link DmlParser#dataIdentifier}.
	 * @param ctx the parse tree
	 */
	void enterCommandlineParamExpression(DmlParser.CommandlineParamExpressionContext ctx);
	/**
	 * Exit a parse tree produced by the {@code CommandlineParamExpression}
	 * labeled alternative in {@link DmlParser#dataIdentifier}.
	 * @param ctx the parse tree
	 */
	void exitCommandlineParamExpression(DmlParser.CommandlineParamExpressionContext ctx);
	/**
	 * Enter a parse tree produced by the {@code CommandlinePositionExpression}
	 * labeled alternative in {@link DmlParser#dataIdentifier}.
	 * @param ctx the parse tree
	 */
	void enterCommandlinePositionExpression(DmlParser.CommandlinePositionExpressionContext ctx);
	/**
	 * Exit a parse tree produced by the {@code CommandlinePositionExpression}
	 * labeled alternative in {@link DmlParser#dataIdentifier}.
	 * @param ctx the parse tree
	 */
	void exitCommandlinePositionExpression(DmlParser.CommandlinePositionExpressionContext ctx);
	/**
	 * Enter a parse tree produced by the {@code ModIntDivExpression}
	 * labeled alternative in {@link DmlParser#expression}.
	 * @param ctx the parse tree
	 */
	void enterModIntDivExpression(DmlParser.ModIntDivExpressionContext ctx);
	/**
	 * Exit a parse tree produced by the {@code ModIntDivExpression}
	 * labeled alternative in {@link DmlParser#expression}.
	 * @param ctx the parse tree
	 */
	void exitModIntDivExpression(DmlParser.ModIntDivExpressionContext ctx);
	/**
	 * Enter a parse tree produced by the {@code RelationalExpression}
	 * labeled alternative in {@link DmlParser#expression}.
	 * @param ctx the parse tree
	 */
	void enterRelationalExpression(DmlParser.RelationalExpressionContext ctx);
	/**
	 * Exit a parse tree produced by the {@code RelationalExpression}
	 * labeled alternative in {@link DmlParser#expression}.
	 * @param ctx the parse tree
	 */
	void exitRelationalExpression(DmlParser.RelationalExpressionContext ctx);
	/**
	 * Enter a parse tree produced by the {@code BooleanNotExpression}
	 * labeled alternative in {@link DmlParser#expression}.
	 * @param ctx the parse tree
	 */
	void enterBooleanNotExpression(DmlParser.BooleanNotExpressionContext ctx);
	/**
	 * Exit a parse tree produced by the {@code BooleanNotExpression}
	 * labeled alternative in {@link DmlParser#expression}.
	 * @param ctx the parse tree
	 */
	void exitBooleanNotExpression(DmlParser.BooleanNotExpressionContext ctx);
	/**
	 * Enter a parse tree produced by the {@code PowerExpression}
	 * labeled alternative in {@link DmlParser#expression}.
	 * @param ctx the parse tree
	 */
	void enterPowerExpression(DmlParser.PowerExpressionContext ctx);
	/**
	 * Exit a parse tree produced by the {@code PowerExpression}
	 * labeled alternative in {@link DmlParser#expression}.
	 * @param ctx the parse tree
	 */
	void exitPowerExpression(DmlParser.PowerExpressionContext ctx);
	/**
	 * Enter a parse tree produced by the {@code BuiltinFunctionExpression}
	 * labeled alternative in {@link DmlParser#expression}.
	 * @param ctx the parse tree
	 */
	void enterBuiltinFunctionExpression(DmlParser.BuiltinFunctionExpressionContext ctx);
	/**
	 * Exit a parse tree produced by the {@code BuiltinFunctionExpression}
	 * labeled alternative in {@link DmlParser#expression}.
	 * @param ctx the parse tree
	 */
	void exitBuiltinFunctionExpression(DmlParser.BuiltinFunctionExpressionContext ctx);
	/**
	 * Enter a parse tree produced by the {@code ConstIntIdExpression}
	 * labeled alternative in {@link DmlParser#expression}.
	 * @param ctx the parse tree
	 */
	void enterConstIntIdExpression(DmlParser.ConstIntIdExpressionContext ctx);
	/**
	 * Exit a parse tree produced by the {@code ConstIntIdExpression}
	 * labeled alternative in {@link DmlParser#expression}.
	 * @param ctx the parse tree
	 */
	void exitConstIntIdExpression(DmlParser.ConstIntIdExpressionContext ctx);
	/**
	 * Enter a parse tree produced by the {@code AtomicExpression}
	 * labeled alternative in {@link DmlParser#expression}.
	 * @param ctx the parse tree
	 */
	void enterAtomicExpression(DmlParser.AtomicExpressionContext ctx);
	/**
	 * Exit a parse tree produced by the {@code AtomicExpression}
	 * labeled alternative in {@link DmlParser#expression}.
	 * @param ctx the parse tree
	 */
	void exitAtomicExpression(DmlParser.AtomicExpressionContext ctx);
	/**
	 * Enter a parse tree produced by the {@code ConstStringIdExpression}
	 * labeled alternative in {@link DmlParser#expression}.
	 * @param ctx the parse tree
	 */
	void enterConstStringIdExpression(DmlParser.ConstStringIdExpressionContext ctx);
	/**
	 * Exit a parse tree produced by the {@code ConstStringIdExpression}
	 * labeled alternative in {@link DmlParser#expression}.
	 * @param ctx the parse tree
	 */
	void exitConstStringIdExpression(DmlParser.ConstStringIdExpressionContext ctx);
	/**
	 * Enter a parse tree produced by the {@code ConstTrueExpression}
	 * labeled alternative in {@link DmlParser#expression}.
	 * @param ctx the parse tree
	 */
	void enterConstTrueExpression(DmlParser.ConstTrueExpressionContext ctx);
	/**
	 * Exit a parse tree produced by the {@code ConstTrueExpression}
	 * labeled alternative in {@link DmlParser#expression}.
	 * @param ctx the parse tree
	 */
	void exitConstTrueExpression(DmlParser.ConstTrueExpressionContext ctx);
	/**
	 * Enter a parse tree produced by the {@code UnaryExpression}
	 * labeled alternative in {@link DmlParser#expression}.
	 * @param ctx the parse tree
	 */
	void enterUnaryExpression(DmlParser.UnaryExpressionContext ctx);
	/**
	 * Exit a parse tree produced by the {@code UnaryExpression}
	 * labeled alternative in {@link DmlParser#expression}.
	 * @param ctx the parse tree
	 */
	void exitUnaryExpression(DmlParser.UnaryExpressionContext ctx);
	/**
	 * Enter a parse tree produced by the {@code MultDivExpression}
	 * labeled alternative in {@link DmlParser#expression}.
	 * @param ctx the parse tree
	 */
	void enterMultDivExpression(DmlParser.MultDivExpressionContext ctx);
	/**
	 * Exit a parse tree produced by the {@code MultDivExpression}
	 * labeled alternative in {@link DmlParser#expression}.
	 * @param ctx the parse tree
	 */
	void exitMultDivExpression(DmlParser.MultDivExpressionContext ctx);
	/**
	 * Enter a parse tree produced by the {@code ConstFalseExpression}
	 * labeled alternative in {@link DmlParser#expression}.
	 * @param ctx the parse tree
	 */
	void enterConstFalseExpression(DmlParser.ConstFalseExpressionContext ctx);
	/**
	 * Exit a parse tree produced by the {@code ConstFalseExpression}
	 * labeled alternative in {@link DmlParser#expression}.
	 * @param ctx the parse tree
	 */
	void exitConstFalseExpression(DmlParser.ConstFalseExpressionContext ctx);
	/**
	 * Enter a parse tree produced by the {@code DataIdExpression}
	 * labeled alternative in {@link DmlParser#expression}.
	 * @param ctx the parse tree
	 */
	void enterDataIdExpression(DmlParser.DataIdExpressionContext ctx);
	/**
	 * Exit a parse tree produced by the {@code DataIdExpression}
	 * labeled alternative in {@link DmlParser#expression}.
	 * @param ctx the parse tree
	 */
	void exitDataIdExpression(DmlParser.DataIdExpressionContext ctx);
	/**
	 * Enter a parse tree produced by the {@code AddSubExpression}
	 * labeled alternative in {@link DmlParser#expression}.
	 * @param ctx the parse tree
	 */
	void enterAddSubExpression(DmlParser.AddSubExpressionContext ctx);
	/**
	 * Exit a parse tree produced by the {@code AddSubExpression}
	 * labeled alternative in {@link DmlParser#expression}.
	 * @param ctx the parse tree
	 */
	void exitAddSubExpression(DmlParser.AddSubExpressionContext ctx);
	/**
	 * Enter a parse tree produced by the {@code ConstDoubleIdExpression}
	 * labeled alternative in {@link DmlParser#expression}.
	 * @param ctx the parse tree
	 */
	void enterConstDoubleIdExpression(DmlParser.ConstDoubleIdExpressionContext ctx);
	/**
	 * Exit a parse tree produced by the {@code ConstDoubleIdExpression}
	 * labeled alternative in {@link DmlParser#expression}.
	 * @param ctx the parse tree
	 */
	void exitConstDoubleIdExpression(DmlParser.ConstDoubleIdExpressionContext ctx);
	/**
	 * Enter a parse tree produced by the {@code MatrixMulExpression}
	 * labeled alternative in {@link DmlParser#expression}.
	 * @param ctx the parse tree
	 */
	void enterMatrixMulExpression(DmlParser.MatrixMulExpressionContext ctx);
	/**
	 * Exit a parse tree produced by the {@code MatrixMulExpression}
	 * labeled alternative in {@link DmlParser#expression}.
	 * @param ctx the parse tree
	 */
	void exitMatrixMulExpression(DmlParser.MatrixMulExpressionContext ctx);
	/**
	 * Enter a parse tree produced by the {@code MultiIdExpression}
	 * labeled alternative in {@link DmlParser#expression}.
	 * @param ctx the parse tree
	 */
	void enterMultiIdExpression(DmlParser.MultiIdExpressionContext ctx);
	/**
	 * Exit a parse tree produced by the {@code MultiIdExpression}
	 * labeled alternative in {@link DmlParser#expression}.
	 * @param ctx the parse tree
	 */
	void exitMultiIdExpression(DmlParser.MultiIdExpressionContext ctx);
	/**
	 * Enter a parse tree produced by the {@code BooleanAndExpression}
	 * labeled alternative in {@link DmlParser#expression}.
	 * @param ctx the parse tree
	 */
	void enterBooleanAndExpression(DmlParser.BooleanAndExpressionContext ctx);
	/**
	 * Exit a parse tree produced by the {@code BooleanAndExpression}
	 * labeled alternative in {@link DmlParser#expression}.
	 * @param ctx the parse tree
	 */
	void exitBooleanAndExpression(DmlParser.BooleanAndExpressionContext ctx);
	/**
	 * Enter a parse tree produced by the {@code BooleanOrExpression}
	 * labeled alternative in {@link DmlParser#expression}.
	 * @param ctx the parse tree
	 */
	void enterBooleanOrExpression(DmlParser.BooleanOrExpressionContext ctx);
	/**
	 * Exit a parse tree produced by the {@code BooleanOrExpression}
	 * labeled alternative in {@link DmlParser#expression}.
	 * @param ctx the parse tree
	 */
	void exitBooleanOrExpression(DmlParser.BooleanOrExpressionContext ctx);
	/**
	 * Enter a parse tree produced by {@link DmlParser#typedArgNoAssign}.
	 * @param ctx the parse tree
	 */
	void enterTypedArgNoAssign(DmlParser.TypedArgNoAssignContext ctx);
	/**
	 * Exit a parse tree produced by {@link DmlParser#typedArgNoAssign}.
	 * @param ctx the parse tree
	 */
	void exitTypedArgNoAssign(DmlParser.TypedArgNoAssignContext ctx);
	/**
	 * Enter a parse tree produced by {@link DmlParser#typedArgAssign}.
	 * @param ctx the parse tree
	 */
	void enterTypedArgAssign(DmlParser.TypedArgAssignContext ctx);
	/**
	 * Exit a parse tree produced by {@link DmlParser#typedArgAssign}.
	 * @param ctx the parse tree
	 */
	void exitTypedArgAssign(DmlParser.TypedArgAssignContext ctx);
	/**
	 * Enter a parse tree produced by {@link DmlParser#parameterizedExpression}.
	 * @param ctx the parse tree
	 */
	void enterParameterizedExpression(DmlParser.ParameterizedExpressionContext ctx);
	/**
	 * Exit a parse tree produced by {@link DmlParser#parameterizedExpression}.
	 * @param ctx the parse tree
	 */
	void exitParameterizedExpression(DmlParser.ParameterizedExpressionContext ctx);
	/**
	 * Enter a parse tree produced by {@link DmlParser#strictParameterizedExpression}.
	 * @param ctx the parse tree
	 */
	void enterStrictParameterizedExpression(DmlParser.StrictParameterizedExpressionContext ctx);
	/**
	 * Exit a parse tree produced by {@link DmlParser#strictParameterizedExpression}.
	 * @param ctx the parse tree
	 */
	void exitStrictParameterizedExpression(DmlParser.StrictParameterizedExpressionContext ctx);
	/**
	 * Enter a parse tree produced by {@link DmlParser#strictParameterizedKeyValueString}.
	 * @param ctx the parse tree
	 */
	void enterStrictParameterizedKeyValueString(DmlParser.StrictParameterizedKeyValueStringContext ctx);
	/**
	 * Exit a parse tree produced by {@link DmlParser#strictParameterizedKeyValueString}.
	 * @param ctx the parse tree
	 */
	void exitStrictParameterizedKeyValueString(DmlParser.StrictParameterizedKeyValueStringContext ctx);
	/**
	 * Enter a parse tree produced by {@link DmlParser#ml_type}.
	 * @param ctx the parse tree
	 */
	void enterMl_type(DmlParser.Ml_typeContext ctx);
	/**
	 * Exit a parse tree produced by {@link DmlParser#ml_type}.
	 * @param ctx the parse tree
	 */
	void exitMl_type(DmlParser.Ml_typeContext ctx);
	/**
	 * Enter a parse tree produced by {@link DmlParser#valueType}.
	 * @param ctx the parse tree
	 */
	void enterValueType(DmlParser.ValueTypeContext ctx);
	/**
	 * Exit a parse tree produced by {@link DmlParser#valueType}.
	 * @param ctx the parse tree
	 */
	void exitValueType(DmlParser.ValueTypeContext ctx);
	/**
	 * Enter a parse tree produced by the {@code MatrixDataTypeCheck}
	 * labeled alternative in {@link DmlParser#dataType}.
	 * @param ctx the parse tree
	 */
	void enterMatrixDataTypeCheck(DmlParser.MatrixDataTypeCheckContext ctx);
	/**
	 * Exit a parse tree produced by the {@code MatrixDataTypeCheck}
	 * labeled alternative in {@link DmlParser#dataType}.
	 * @param ctx the parse tree
	 */
	void exitMatrixDataTypeCheck(DmlParser.MatrixDataTypeCheckContext ctx);
}