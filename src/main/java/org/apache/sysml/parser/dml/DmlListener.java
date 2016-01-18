// Generated from org/apache/sysml/parser/dml/Dml.g4 by ANTLR 4.3
package org.apache.sysml.parser.dml;

	// Commenting the package name and explicitly passing it in build.xml to maintain compatibility with maven plugin
    // package org.apache.sysml.antlr4;

import org.antlr.v4.runtime.misc.NotNull;
import org.antlr.v4.runtime.tree.ParseTreeListener;

/**
 * This interface defines a complete listener for a parse tree produced by
 * {@link DmlParser}.
 */
public interface DmlListener extends ParseTreeListener {
	/**
	 * Enter a parse tree produced by {@link DmlParser#dmlprogram}.
	 * @param ctx the parse tree
	 */
	void enterDmlprogram(@NotNull DmlParser.DmlprogramContext ctx);
	/**
	 * Exit a parse tree produced by {@link DmlParser#dmlprogram}.
	 * @param ctx the parse tree
	 */
	void exitDmlprogram(@NotNull DmlParser.DmlprogramContext ctx);

	/**
	 * Enter a parse tree produced by the {@code ModIntDivExpression}
	 * labeled alternative in {@link DmlParser#expression}.
	 * @param ctx the parse tree
	 */
	void enterModIntDivExpression(@NotNull DmlParser.ModIntDivExpressionContext ctx);
	/**
	 * Exit a parse tree produced by the {@code ModIntDivExpression}
	 * labeled alternative in {@link DmlParser#expression}.
	 * @param ctx the parse tree
	 */
	void exitModIntDivExpression(@NotNull DmlParser.ModIntDivExpressionContext ctx);

	/**
	 * Enter a parse tree produced by the {@code ExternalFunctionDefExpression}
	 * labeled alternative in {@link DmlParser#functionStatement}.
	 * @param ctx the parse tree
	 */
	void enterExternalFunctionDefExpression(@NotNull DmlParser.ExternalFunctionDefExpressionContext ctx);
	/**
	 * Exit a parse tree produced by the {@code ExternalFunctionDefExpression}
	 * labeled alternative in {@link DmlParser#functionStatement}.
	 * @param ctx the parse tree
	 */
	void exitExternalFunctionDefExpression(@NotNull DmlParser.ExternalFunctionDefExpressionContext ctx);

	/**
	 * Enter a parse tree produced by the {@code BooleanNotExpression}
	 * labeled alternative in {@link DmlParser#expression}.
	 * @param ctx the parse tree
	 */
	void enterBooleanNotExpression(@NotNull DmlParser.BooleanNotExpressionContext ctx);
	/**
	 * Exit a parse tree produced by the {@code BooleanNotExpression}
	 * labeled alternative in {@link DmlParser#expression}.
	 * @param ctx the parse tree
	 */
	void exitBooleanNotExpression(@NotNull DmlParser.BooleanNotExpressionContext ctx);

	/**
	 * Enter a parse tree produced by the {@code PowerExpression}
	 * labeled alternative in {@link DmlParser#expression}.
	 * @param ctx the parse tree
	 */
	void enterPowerExpression(@NotNull DmlParser.PowerExpressionContext ctx);
	/**
	 * Exit a parse tree produced by the {@code PowerExpression}
	 * labeled alternative in {@link DmlParser#expression}.
	 * @param ctx the parse tree
	 */
	void exitPowerExpression(@NotNull DmlParser.PowerExpressionContext ctx);

	/**
	 * Enter a parse tree produced by the {@code InternalFunctionDefExpression}
	 * labeled alternative in {@link DmlParser#functionStatement}.
	 * @param ctx the parse tree
	 */
	void enterInternalFunctionDefExpression(@NotNull DmlParser.InternalFunctionDefExpressionContext ctx);
	/**
	 * Exit a parse tree produced by the {@code InternalFunctionDefExpression}
	 * labeled alternative in {@link DmlParser#functionStatement}.
	 * @param ctx the parse tree
	 */
	void exitInternalFunctionDefExpression(@NotNull DmlParser.InternalFunctionDefExpressionContext ctx);

	/**
	 * Enter a parse tree produced by the {@code BuiltinFunctionExpression}
	 * labeled alternative in {@link DmlParser#expression}.
	 * @param ctx the parse tree
	 */
	void enterBuiltinFunctionExpression(@NotNull DmlParser.BuiltinFunctionExpressionContext ctx);
	/**
	 * Exit a parse tree produced by the {@code BuiltinFunctionExpression}
	 * labeled alternative in {@link DmlParser#expression}.
	 * @param ctx the parse tree
	 */
	void exitBuiltinFunctionExpression(@NotNull DmlParser.BuiltinFunctionExpressionContext ctx);

	/**
	 * Enter a parse tree produced by the {@code ConstIntIdExpression}
	 * labeled alternative in {@link DmlParser#expression}.
	 * @param ctx the parse tree
	 */
	void enterConstIntIdExpression(@NotNull DmlParser.ConstIntIdExpressionContext ctx);
	/**
	 * Exit a parse tree produced by the {@code ConstIntIdExpression}
	 * labeled alternative in {@link DmlParser#expression}.
	 * @param ctx the parse tree
	 */
	void exitConstIntIdExpression(@NotNull DmlParser.ConstIntIdExpressionContext ctx);

	/**
	 * Enter a parse tree produced by the {@code AtomicExpression}
	 * labeled alternative in {@link DmlParser#expression}.
	 * @param ctx the parse tree
	 */
	void enterAtomicExpression(@NotNull DmlParser.AtomicExpressionContext ctx);
	/**
	 * Exit a parse tree produced by the {@code AtomicExpression}
	 * labeled alternative in {@link DmlParser#expression}.
	 * @param ctx the parse tree
	 */
	void exitAtomicExpression(@NotNull DmlParser.AtomicExpressionContext ctx);

	/**
	 * Enter a parse tree produced by the {@code IfdefAssignmentStatement}
	 * labeled alternative in {@link DmlParser#statement}.
	 * @param ctx the parse tree
	 */
	void enterIfdefAssignmentStatement(@NotNull DmlParser.IfdefAssignmentStatementContext ctx);
	/**
	 * Exit a parse tree produced by the {@code IfdefAssignmentStatement}
	 * labeled alternative in {@link DmlParser#statement}.
	 * @param ctx the parse tree
	 */
	void exitIfdefAssignmentStatement(@NotNull DmlParser.IfdefAssignmentStatementContext ctx);

	/**
	 * Enter a parse tree produced by the {@code ConstStringIdExpression}
	 * labeled alternative in {@link DmlParser#expression}.
	 * @param ctx the parse tree
	 */
	void enterConstStringIdExpression(@NotNull DmlParser.ConstStringIdExpressionContext ctx);
	/**
	 * Exit a parse tree produced by the {@code ConstStringIdExpression}
	 * labeled alternative in {@link DmlParser#expression}.
	 * @param ctx the parse tree
	 */
	void exitConstStringIdExpression(@NotNull DmlParser.ConstStringIdExpressionContext ctx);

	/**
	 * Enter a parse tree produced by the {@code ConstTrueExpression}
	 * labeled alternative in {@link DmlParser#expression}.
	 * @param ctx the parse tree
	 */
	void enterConstTrueExpression(@NotNull DmlParser.ConstTrueExpressionContext ctx);
	/**
	 * Exit a parse tree produced by the {@code ConstTrueExpression}
	 * labeled alternative in {@link DmlParser#expression}.
	 * @param ctx the parse tree
	 */
	void exitConstTrueExpression(@NotNull DmlParser.ConstTrueExpressionContext ctx);

	/**
	 * Enter a parse tree produced by the {@code ParForStatement}
	 * labeled alternative in {@link DmlParser#statement}.
	 * @param ctx the parse tree
	 */
	void enterParForStatement(@NotNull DmlParser.ParForStatementContext ctx);
	/**
	 * Exit a parse tree produced by the {@code ParForStatement}
	 * labeled alternative in {@link DmlParser#statement}.
	 * @param ctx the parse tree
	 */
	void exitParForStatement(@NotNull DmlParser.ParForStatementContext ctx);

	/**
	 * Enter a parse tree produced by the {@code UnaryExpression}
	 * labeled alternative in {@link DmlParser#expression}.
	 * @param ctx the parse tree
	 */
	void enterUnaryExpression(@NotNull DmlParser.UnaryExpressionContext ctx);
	/**
	 * Exit a parse tree produced by the {@code UnaryExpression}
	 * labeled alternative in {@link DmlParser#expression}.
	 * @param ctx the parse tree
	 */
	void exitUnaryExpression(@NotNull DmlParser.UnaryExpressionContext ctx);

	/**
	 * Enter a parse tree produced by the {@code ImportStatement}
	 * labeled alternative in {@link DmlParser#statement}.
	 * @param ctx the parse tree
	 */
	void enterImportStatement(@NotNull DmlParser.ImportStatementContext ctx);
	/**
	 * Exit a parse tree produced by the {@code ImportStatement}
	 * labeled alternative in {@link DmlParser#statement}.
	 * @param ctx the parse tree
	 */
	void exitImportStatement(@NotNull DmlParser.ImportStatementContext ctx);

	/**
	 * Enter a parse tree produced by the {@code PathStatement}
	 * labeled alternative in {@link DmlParser#statement}.
	 * @param ctx the parse tree
	 */
	void enterPathStatement(@NotNull DmlParser.PathStatementContext ctx);
	/**
	 * Exit a parse tree produced by the {@code PathStatement}
	 * labeled alternative in {@link DmlParser#statement}.
	 * @param ctx the parse tree
	 */
	void exitPathStatement(@NotNull DmlParser.PathStatementContext ctx);

	/**
	 * Enter a parse tree produced by the {@code WhileStatement}
	 * labeled alternative in {@link DmlParser#statement}.
	 * @param ctx the parse tree
	 */
	void enterWhileStatement(@NotNull DmlParser.WhileStatementContext ctx);
	/**
	 * Exit a parse tree produced by the {@code WhileStatement}
	 * labeled alternative in {@link DmlParser#statement}.
	 * @param ctx the parse tree
	 */
	void exitWhileStatement(@NotNull DmlParser.WhileStatementContext ctx);

	/**
	 * Enter a parse tree produced by the {@code CommandlineParamExpression}
	 * labeled alternative in {@link DmlParser#dataIdentifier}.
	 * @param ctx the parse tree
	 */
	void enterCommandlineParamExpression(@NotNull DmlParser.CommandlineParamExpressionContext ctx);
	/**
	 * Exit a parse tree produced by the {@code CommandlineParamExpression}
	 * labeled alternative in {@link DmlParser#dataIdentifier}.
	 * @param ctx the parse tree
	 */
	void exitCommandlineParamExpression(@NotNull DmlParser.CommandlineParamExpressionContext ctx);

	/**
	 * Enter a parse tree produced by the {@code FunctionCallAssignmentStatement}
	 * labeled alternative in {@link DmlParser#statement}.
	 * @param ctx the parse tree
	 */
	void enterFunctionCallAssignmentStatement(@NotNull DmlParser.FunctionCallAssignmentStatementContext ctx);
	/**
	 * Exit a parse tree produced by the {@code FunctionCallAssignmentStatement}
	 * labeled alternative in {@link DmlParser#statement}.
	 * @param ctx the parse tree
	 */
	void exitFunctionCallAssignmentStatement(@NotNull DmlParser.FunctionCallAssignmentStatementContext ctx);

	/**
	 * Enter a parse tree produced by the {@code AddSubExpression}
	 * labeled alternative in {@link DmlParser#expression}.
	 * @param ctx the parse tree
	 */
	void enterAddSubExpression(@NotNull DmlParser.AddSubExpressionContext ctx);
	/**
	 * Exit a parse tree produced by the {@code AddSubExpression}
	 * labeled alternative in {@link DmlParser#expression}.
	 * @param ctx the parse tree
	 */
	void exitAddSubExpression(@NotNull DmlParser.AddSubExpressionContext ctx);

	/**
	 * Enter a parse tree produced by the {@code IfStatement}
	 * labeled alternative in {@link DmlParser#statement}.
	 * @param ctx the parse tree
	 */
	void enterIfStatement(@NotNull DmlParser.IfStatementContext ctx);
	/**
	 * Exit a parse tree produced by the {@code IfStatement}
	 * labeled alternative in {@link DmlParser#statement}.
	 * @param ctx the parse tree
	 */
	void exitIfStatement(@NotNull DmlParser.IfStatementContext ctx);

	/**
	 * Enter a parse tree produced by the {@code ConstDoubleIdExpression}
	 * labeled alternative in {@link DmlParser#expression}.
	 * @param ctx the parse tree
	 */
	void enterConstDoubleIdExpression(@NotNull DmlParser.ConstDoubleIdExpressionContext ctx);
	/**
	 * Exit a parse tree produced by the {@code ConstDoubleIdExpression}
	 * labeled alternative in {@link DmlParser#expression}.
	 * @param ctx the parse tree
	 */
	void exitConstDoubleIdExpression(@NotNull DmlParser.ConstDoubleIdExpressionContext ctx);

	/**
	 * Enter a parse tree produced by the {@code MatrixMulExpression}
	 * labeled alternative in {@link DmlParser#expression}.
	 * @param ctx the parse tree
	 */
	void enterMatrixMulExpression(@NotNull DmlParser.MatrixMulExpressionContext ctx);
	/**
	 * Exit a parse tree produced by the {@code MatrixMulExpression}
	 * labeled alternative in {@link DmlParser#expression}.
	 * @param ctx the parse tree
	 */
	void exitMatrixMulExpression(@NotNull DmlParser.MatrixMulExpressionContext ctx);

	/**
	 * Enter a parse tree produced by the {@code MatrixDataTypeCheck}
	 * labeled alternative in {@link DmlParser#dataType}.
	 * @param ctx the parse tree
	 */
	void enterMatrixDataTypeCheck(@NotNull DmlParser.MatrixDataTypeCheckContext ctx);
	/**
	 * Exit a parse tree produced by the {@code MatrixDataTypeCheck}
	 * labeled alternative in {@link DmlParser#dataType}.
	 * @param ctx the parse tree
	 */
	void exitMatrixDataTypeCheck(@NotNull DmlParser.MatrixDataTypeCheckContext ctx);

	/**
	 * Enter a parse tree produced by the {@code CommandlinePositionExpression}
	 * labeled alternative in {@link DmlParser#dataIdentifier}.
	 * @param ctx the parse tree
	 */
	void enterCommandlinePositionExpression(@NotNull DmlParser.CommandlinePositionExpressionContext ctx);
	/**
	 * Exit a parse tree produced by the {@code CommandlinePositionExpression}
	 * labeled alternative in {@link DmlParser#dataIdentifier}.
	 * @param ctx the parse tree
	 */
	void exitCommandlinePositionExpression(@NotNull DmlParser.CommandlinePositionExpressionContext ctx);

	/**
	 * Enter a parse tree produced by the {@code IterablePredicateColonExpression}
	 * labeled alternative in {@link DmlParser#iterablePredicate}.
	 * @param ctx the parse tree
	 */
	void enterIterablePredicateColonExpression(@NotNull DmlParser.IterablePredicateColonExpressionContext ctx);
	/**
	 * Exit a parse tree produced by the {@code IterablePredicateColonExpression}
	 * labeled alternative in {@link DmlParser#iterablePredicate}.
	 * @param ctx the parse tree
	 */
	void exitIterablePredicateColonExpression(@NotNull DmlParser.IterablePredicateColonExpressionContext ctx);

	/**
	 * Enter a parse tree produced by the {@code AssignmentStatement}
	 * labeled alternative in {@link DmlParser#statement}.
	 * @param ctx the parse tree
	 */
	void enterAssignmentStatement(@NotNull DmlParser.AssignmentStatementContext ctx);
	/**
	 * Exit a parse tree produced by the {@code AssignmentStatement}
	 * labeled alternative in {@link DmlParser#statement}.
	 * @param ctx the parse tree
	 */
	void exitAssignmentStatement(@NotNull DmlParser.AssignmentStatementContext ctx);

	/**
	 * Enter a parse tree produced by {@link DmlParser#valueType}.
	 * @param ctx the parse tree
	 */
	void enterValueType(@NotNull DmlParser.ValueTypeContext ctx);
	/**
	 * Exit a parse tree produced by {@link DmlParser#valueType}.
	 * @param ctx the parse tree
	 */
	void exitValueType(@NotNull DmlParser.ValueTypeContext ctx);

	/**
	 * Enter a parse tree produced by {@link DmlParser#ml_type}.
	 * @param ctx the parse tree
	 */
	void enterMl_type(@NotNull DmlParser.Ml_typeContext ctx);
	/**
	 * Exit a parse tree produced by {@link DmlParser#ml_type}.
	 * @param ctx the parse tree
	 */
	void exitMl_type(@NotNull DmlParser.Ml_typeContext ctx);

	/**
	 * Enter a parse tree produced by the {@code BooleanAndExpression}
	 * labeled alternative in {@link DmlParser#expression}.
	 * @param ctx the parse tree
	 */
	void enterBooleanAndExpression(@NotNull DmlParser.BooleanAndExpressionContext ctx);
	/**
	 * Exit a parse tree produced by the {@code BooleanAndExpression}
	 * labeled alternative in {@link DmlParser#expression}.
	 * @param ctx the parse tree
	 */
	void exitBooleanAndExpression(@NotNull DmlParser.BooleanAndExpressionContext ctx);

	/**
	 * Enter a parse tree produced by the {@code ForStatement}
	 * labeled alternative in {@link DmlParser#statement}.
	 * @param ctx the parse tree
	 */
	void enterForStatement(@NotNull DmlParser.ForStatementContext ctx);
	/**
	 * Exit a parse tree produced by the {@code ForStatement}
	 * labeled alternative in {@link DmlParser#statement}.
	 * @param ctx the parse tree
	 */
	void exitForStatement(@NotNull DmlParser.ForStatementContext ctx);

	/**
	 * Enter a parse tree produced by the {@code RelationalExpression}
	 * labeled alternative in {@link DmlParser#expression}.
	 * @param ctx the parse tree
	 */
	void enterRelationalExpression(@NotNull DmlParser.RelationalExpressionContext ctx);
	/**
	 * Exit a parse tree produced by the {@code RelationalExpression}
	 * labeled alternative in {@link DmlParser#expression}.
	 * @param ctx the parse tree
	 */
	void exitRelationalExpression(@NotNull DmlParser.RelationalExpressionContext ctx);

	/**
	 * Enter a parse tree produced by {@link DmlParser#typedArgNoAssign}.
	 * @param ctx the parse tree
	 */
	void enterTypedArgNoAssign(@NotNull DmlParser.TypedArgNoAssignContext ctx);
	/**
	 * Exit a parse tree produced by {@link DmlParser#typedArgNoAssign}.
	 * @param ctx the parse tree
	 */
	void exitTypedArgNoAssign(@NotNull DmlParser.TypedArgNoAssignContext ctx);

	/**
	 * Enter a parse tree produced by {@link DmlParser#strictParameterizedExpression}.
	 * @param ctx the parse tree
	 */
	void enterStrictParameterizedExpression(@NotNull DmlParser.StrictParameterizedExpressionContext ctx);
	/**
	 * Exit a parse tree produced by {@link DmlParser#strictParameterizedExpression}.
	 * @param ctx the parse tree
	 */
	void exitStrictParameterizedExpression(@NotNull DmlParser.StrictParameterizedExpressionContext ctx);

	/**
	 * Enter a parse tree produced by the {@code MultDivExpression}
	 * labeled alternative in {@link DmlParser#expression}.
	 * @param ctx the parse tree
	 */
	void enterMultDivExpression(@NotNull DmlParser.MultDivExpressionContext ctx);
	/**
	 * Exit a parse tree produced by the {@code MultDivExpression}
	 * labeled alternative in {@link DmlParser#expression}.
	 * @param ctx the parse tree
	 */
	void exitMultDivExpression(@NotNull DmlParser.MultDivExpressionContext ctx);

	/**
	 * Enter a parse tree produced by the {@code ConstFalseExpression}
	 * labeled alternative in {@link DmlParser#expression}.
	 * @param ctx the parse tree
	 */
	void enterConstFalseExpression(@NotNull DmlParser.ConstFalseExpressionContext ctx);
	/**
	 * Exit a parse tree produced by the {@code ConstFalseExpression}
	 * labeled alternative in {@link DmlParser#expression}.
	 * @param ctx the parse tree
	 */
	void exitConstFalseExpression(@NotNull DmlParser.ConstFalseExpressionContext ctx);

	/**
	 * Enter a parse tree produced by {@link DmlParser#strictParameterizedKeyValueString}.
	 * @param ctx the parse tree
	 */
	void enterStrictParameterizedKeyValueString(@NotNull DmlParser.StrictParameterizedKeyValueStringContext ctx);
	/**
	 * Exit a parse tree produced by {@link DmlParser#strictParameterizedKeyValueString}.
	 * @param ctx the parse tree
	 */
	void exitStrictParameterizedKeyValueString(@NotNull DmlParser.StrictParameterizedKeyValueStringContext ctx);

	/**
	 * Enter a parse tree produced by the {@code DataIdExpression}
	 * labeled alternative in {@link DmlParser#expression}.
	 * @param ctx the parse tree
	 */
	void enterDataIdExpression(@NotNull DmlParser.DataIdExpressionContext ctx);
	/**
	 * Exit a parse tree produced by the {@code DataIdExpression}
	 * labeled alternative in {@link DmlParser#expression}.
	 * @param ctx the parse tree
	 */
	void exitDataIdExpression(@NotNull DmlParser.DataIdExpressionContext ctx);

	/**
	 * Enter a parse tree produced by the {@code IndexedExpression}
	 * labeled alternative in {@link DmlParser#dataIdentifier}.
	 * @param ctx the parse tree
	 */
	void enterIndexedExpression(@NotNull DmlParser.IndexedExpressionContext ctx);
	/**
	 * Exit a parse tree produced by the {@code IndexedExpression}
	 * labeled alternative in {@link DmlParser#dataIdentifier}.
	 * @param ctx the parse tree
	 */
	void exitIndexedExpression(@NotNull DmlParser.IndexedExpressionContext ctx);

	/**
	 * Enter a parse tree produced by {@link DmlParser#parameterizedExpression}.
	 * @param ctx the parse tree
	 */
	void enterParameterizedExpression(@NotNull DmlParser.ParameterizedExpressionContext ctx);
	/**
	 * Exit a parse tree produced by {@link DmlParser#parameterizedExpression}.
	 * @param ctx the parse tree
	 */
	void exitParameterizedExpression(@NotNull DmlParser.ParameterizedExpressionContext ctx);

	/**
	 * Enter a parse tree produced by the {@code FunctionCallMultiAssignmentStatement}
	 * labeled alternative in {@link DmlParser#statement}.
	 * @param ctx the parse tree
	 */
	void enterFunctionCallMultiAssignmentStatement(@NotNull DmlParser.FunctionCallMultiAssignmentStatementContext ctx);
	/**
	 * Exit a parse tree produced by the {@code FunctionCallMultiAssignmentStatement}
	 * labeled alternative in {@link DmlParser#statement}.
	 * @param ctx the parse tree
	 */
	void exitFunctionCallMultiAssignmentStatement(@NotNull DmlParser.FunctionCallMultiAssignmentStatementContext ctx);

	/**
	 * Enter a parse tree produced by the {@code IterablePredicateSeqExpression}
	 * labeled alternative in {@link DmlParser#iterablePredicate}.
	 * @param ctx the parse tree
	 */
	void enterIterablePredicateSeqExpression(@NotNull DmlParser.IterablePredicateSeqExpressionContext ctx);
	/**
	 * Exit a parse tree produced by the {@code IterablePredicateSeqExpression}
	 * labeled alternative in {@link DmlParser#iterablePredicate}.
	 * @param ctx the parse tree
	 */
	void exitIterablePredicateSeqExpression(@NotNull DmlParser.IterablePredicateSeqExpressionContext ctx);

	/**
	 * Enter a parse tree produced by the {@code SimpleDataIdentifierExpression}
	 * labeled alternative in {@link DmlParser#dataIdentifier}.
	 * @param ctx the parse tree
	 */
	void enterSimpleDataIdentifierExpression(@NotNull DmlParser.SimpleDataIdentifierExpressionContext ctx);
	/**
	 * Exit a parse tree produced by the {@code SimpleDataIdentifierExpression}
	 * labeled alternative in {@link DmlParser#dataIdentifier}.
	 * @param ctx the parse tree
	 */
	void exitSimpleDataIdentifierExpression(@NotNull DmlParser.SimpleDataIdentifierExpressionContext ctx);

	/**
	 * Enter a parse tree produced by the {@code BooleanOrExpression}
	 * labeled alternative in {@link DmlParser#expression}.
	 * @param ctx the parse tree
	 */
	void enterBooleanOrExpression(@NotNull DmlParser.BooleanOrExpressionContext ctx);
	/**
	 * Exit a parse tree produced by the {@code BooleanOrExpression}
	 * labeled alternative in {@link DmlParser#expression}.
	 * @param ctx the parse tree
	 */
	void exitBooleanOrExpression(@NotNull DmlParser.BooleanOrExpressionContext ctx);
}