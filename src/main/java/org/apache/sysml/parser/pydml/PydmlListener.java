// Generated from org/apache/sysml/parser/pydml/Pydml.g4 by ANTLR 4.3
package org.apache.sysml.parser.pydml;

    // package org.apache.sysml.python;
    //import org.apache.sysml.parser.dml.StatementInfo;
    //import org.apache.sysml.parser.dml.ExpressionInfo;

import org.antlr.v4.runtime.misc.NotNull;
import org.antlr.v4.runtime.tree.ParseTreeListener;

/**
 * This interface defines a complete listener for a parse tree produced by
 * {@link PydmlParser}.
 */
public interface PydmlListener extends ParseTreeListener {
	/**
	 * Enter a parse tree produced by the {@code ModIntDivExpression}
	 * labeled alternative in {@link PydmlParser#expression}.
	 * @param ctx the parse tree
	 */
	void enterModIntDivExpression(@NotNull PydmlParser.ModIntDivExpressionContext ctx);
	/**
	 * Exit a parse tree produced by the {@code ModIntDivExpression}
	 * labeled alternative in {@link PydmlParser#expression}.
	 * @param ctx the parse tree
	 */
	void exitModIntDivExpression(@NotNull PydmlParser.ModIntDivExpressionContext ctx);

	/**
	 * Enter a parse tree produced by the {@code ExternalFunctionDefExpression}
	 * labeled alternative in {@link PydmlParser#functionStatement}.
	 * @param ctx the parse tree
	 */
	void enterExternalFunctionDefExpression(@NotNull PydmlParser.ExternalFunctionDefExpressionContext ctx);
	/**
	 * Exit a parse tree produced by the {@code ExternalFunctionDefExpression}
	 * labeled alternative in {@link PydmlParser#functionStatement}.
	 * @param ctx the parse tree
	 */
	void exitExternalFunctionDefExpression(@NotNull PydmlParser.ExternalFunctionDefExpressionContext ctx);

	/**
	 * Enter a parse tree produced by the {@code BooleanNotExpression}
	 * labeled alternative in {@link PydmlParser#expression}.
	 * @param ctx the parse tree
	 */
	void enterBooleanNotExpression(@NotNull PydmlParser.BooleanNotExpressionContext ctx);
	/**
	 * Exit a parse tree produced by the {@code BooleanNotExpression}
	 * labeled alternative in {@link PydmlParser#expression}.
	 * @param ctx the parse tree
	 */
	void exitBooleanNotExpression(@NotNull PydmlParser.BooleanNotExpressionContext ctx);

	/**
	 * Enter a parse tree produced by the {@code PowerExpression}
	 * labeled alternative in {@link PydmlParser#expression}.
	 * @param ctx the parse tree
	 */
	void enterPowerExpression(@NotNull PydmlParser.PowerExpressionContext ctx);
	/**
	 * Exit a parse tree produced by the {@code PowerExpression}
	 * labeled alternative in {@link PydmlParser#expression}.
	 * @param ctx the parse tree
	 */
	void exitPowerExpression(@NotNull PydmlParser.PowerExpressionContext ctx);

	/**
	 * Enter a parse tree produced by the {@code InternalFunctionDefExpression}
	 * labeled alternative in {@link PydmlParser#functionStatement}.
	 * @param ctx the parse tree
	 */
	void enterInternalFunctionDefExpression(@NotNull PydmlParser.InternalFunctionDefExpressionContext ctx);
	/**
	 * Exit a parse tree produced by the {@code InternalFunctionDefExpression}
	 * labeled alternative in {@link PydmlParser#functionStatement}.
	 * @param ctx the parse tree
	 */
	void exitInternalFunctionDefExpression(@NotNull PydmlParser.InternalFunctionDefExpressionContext ctx);

	/**
	 * Enter a parse tree produced by the {@code BuiltinFunctionExpression}
	 * labeled alternative in {@link PydmlParser#expression}.
	 * @param ctx the parse tree
	 */
	void enterBuiltinFunctionExpression(@NotNull PydmlParser.BuiltinFunctionExpressionContext ctx);
	/**
	 * Exit a parse tree produced by the {@code BuiltinFunctionExpression}
	 * labeled alternative in {@link PydmlParser#expression}.
	 * @param ctx the parse tree
	 */
	void exitBuiltinFunctionExpression(@NotNull PydmlParser.BuiltinFunctionExpressionContext ctx);

	/**
	 * Enter a parse tree produced by the {@code ConstIntIdExpression}
	 * labeled alternative in {@link PydmlParser#expression}.
	 * @param ctx the parse tree
	 */
	void enterConstIntIdExpression(@NotNull PydmlParser.ConstIntIdExpressionContext ctx);
	/**
	 * Exit a parse tree produced by the {@code ConstIntIdExpression}
	 * labeled alternative in {@link PydmlParser#expression}.
	 * @param ctx the parse tree
	 */
	void exitConstIntIdExpression(@NotNull PydmlParser.ConstIntIdExpressionContext ctx);

	/**
	 * Enter a parse tree produced by the {@code AtomicExpression}
	 * labeled alternative in {@link PydmlParser#expression}.
	 * @param ctx the parse tree
	 */
	void enterAtomicExpression(@NotNull PydmlParser.AtomicExpressionContext ctx);
	/**
	 * Exit a parse tree produced by the {@code AtomicExpression}
	 * labeled alternative in {@link PydmlParser#expression}.
	 * @param ctx the parse tree
	 */
	void exitAtomicExpression(@NotNull PydmlParser.AtomicExpressionContext ctx);

	/**
	 * Enter a parse tree produced by {@link PydmlParser#pmlprogram}.
	 * @param ctx the parse tree
	 */
	void enterPmlprogram(@NotNull PydmlParser.PmlprogramContext ctx);
	/**
	 * Exit a parse tree produced by {@link PydmlParser#pmlprogram}.
	 * @param ctx the parse tree
	 */
	void exitPmlprogram(@NotNull PydmlParser.PmlprogramContext ctx);

	/**
	 * Enter a parse tree produced by the {@code IfdefAssignmentStatement}
	 * labeled alternative in {@link PydmlParser#statement}.
	 * @param ctx the parse tree
	 */
	void enterIfdefAssignmentStatement(@NotNull PydmlParser.IfdefAssignmentStatementContext ctx);
	/**
	 * Exit a parse tree produced by the {@code IfdefAssignmentStatement}
	 * labeled alternative in {@link PydmlParser#statement}.
	 * @param ctx the parse tree
	 */
	void exitIfdefAssignmentStatement(@NotNull PydmlParser.IfdefAssignmentStatementContext ctx);

	/**
	 * Enter a parse tree produced by the {@code ConstStringIdExpression}
	 * labeled alternative in {@link PydmlParser#expression}.
	 * @param ctx the parse tree
	 */
	void enterConstStringIdExpression(@NotNull PydmlParser.ConstStringIdExpressionContext ctx);
	/**
	 * Exit a parse tree produced by the {@code ConstStringIdExpression}
	 * labeled alternative in {@link PydmlParser#expression}.
	 * @param ctx the parse tree
	 */
	void exitConstStringIdExpression(@NotNull PydmlParser.ConstStringIdExpressionContext ctx);

	/**
	 * Enter a parse tree produced by the {@code ConstTrueExpression}
	 * labeled alternative in {@link PydmlParser#expression}.
	 * @param ctx the parse tree
	 */
	void enterConstTrueExpression(@NotNull PydmlParser.ConstTrueExpressionContext ctx);
	/**
	 * Exit a parse tree produced by the {@code ConstTrueExpression}
	 * labeled alternative in {@link PydmlParser#expression}.
	 * @param ctx the parse tree
	 */
	void exitConstTrueExpression(@NotNull PydmlParser.ConstTrueExpressionContext ctx);

	/**
	 * Enter a parse tree produced by the {@code ValueDataTypeCheck}
	 * labeled alternative in {@link PydmlParser#valueType}.
	 * @param ctx the parse tree
	 */
	void enterValueDataTypeCheck(@NotNull PydmlParser.ValueDataTypeCheckContext ctx);
	/**
	 * Exit a parse tree produced by the {@code ValueDataTypeCheck}
	 * labeled alternative in {@link PydmlParser#valueType}.
	 * @param ctx the parse tree
	 */
	void exitValueDataTypeCheck(@NotNull PydmlParser.ValueDataTypeCheckContext ctx);

	/**
	 * Enter a parse tree produced by the {@code ParForStatement}
	 * labeled alternative in {@link PydmlParser#statement}.
	 * @param ctx the parse tree
	 */
	void enterParForStatement(@NotNull PydmlParser.ParForStatementContext ctx);
	/**
	 * Exit a parse tree produced by the {@code ParForStatement}
	 * labeled alternative in {@link PydmlParser#statement}.
	 * @param ctx the parse tree
	 */
	void exitParForStatement(@NotNull PydmlParser.ParForStatementContext ctx);

	/**
	 * Enter a parse tree produced by the {@code UnaryExpression}
	 * labeled alternative in {@link PydmlParser#expression}.
	 * @param ctx the parse tree
	 */
	void enterUnaryExpression(@NotNull PydmlParser.UnaryExpressionContext ctx);
	/**
	 * Exit a parse tree produced by the {@code UnaryExpression}
	 * labeled alternative in {@link PydmlParser#expression}.
	 * @param ctx the parse tree
	 */
	void exitUnaryExpression(@NotNull PydmlParser.UnaryExpressionContext ctx);

	/**
	 * Enter a parse tree produced by the {@code ImportStatement}
	 * labeled alternative in {@link PydmlParser#statement}.
	 * @param ctx the parse tree
	 */
	void enterImportStatement(@NotNull PydmlParser.ImportStatementContext ctx);
	/**
	 * Exit a parse tree produced by the {@code ImportStatement}
	 * labeled alternative in {@link PydmlParser#statement}.
	 * @param ctx the parse tree
	 */
	void exitImportStatement(@NotNull PydmlParser.ImportStatementContext ctx);

	/**
	 * Enter a parse tree produced by the {@code PathStatement}
	 * labeled alternative in {@link PydmlParser#statement}.
	 * @param ctx the parse tree
	 */
	void enterPathStatement(@NotNull PydmlParser.PathStatementContext ctx);
	/**
	 * Exit a parse tree produced by the {@code PathStatement}
	 * labeled alternative in {@link PydmlParser#statement}.
	 * @param ctx the parse tree
	 */
	void exitPathStatement(@NotNull PydmlParser.PathStatementContext ctx);

	/**
	 * Enter a parse tree produced by the {@code WhileStatement}
	 * labeled alternative in {@link PydmlParser#statement}.
	 * @param ctx the parse tree
	 */
	void enterWhileStatement(@NotNull PydmlParser.WhileStatementContext ctx);
	/**
	 * Exit a parse tree produced by the {@code WhileStatement}
	 * labeled alternative in {@link PydmlParser#statement}.
	 * @param ctx the parse tree
	 */
	void exitWhileStatement(@NotNull PydmlParser.WhileStatementContext ctx);

	/**
	 * Enter a parse tree produced by the {@code CommandlineParamExpression}
	 * labeled alternative in {@link PydmlParser#dataIdentifier}.
	 * @param ctx the parse tree
	 */
	void enterCommandlineParamExpression(@NotNull PydmlParser.CommandlineParamExpressionContext ctx);
	/**
	 * Exit a parse tree produced by the {@code CommandlineParamExpression}
	 * labeled alternative in {@link PydmlParser#dataIdentifier}.
	 * @param ctx the parse tree
	 */
	void exitCommandlineParamExpression(@NotNull PydmlParser.CommandlineParamExpressionContext ctx);

	/**
	 * Enter a parse tree produced by the {@code FunctionCallAssignmentStatement}
	 * labeled alternative in {@link PydmlParser#statement}.
	 * @param ctx the parse tree
	 */
	void enterFunctionCallAssignmentStatement(@NotNull PydmlParser.FunctionCallAssignmentStatementContext ctx);
	/**
	 * Exit a parse tree produced by the {@code FunctionCallAssignmentStatement}
	 * labeled alternative in {@link PydmlParser#statement}.
	 * @param ctx the parse tree
	 */
	void exitFunctionCallAssignmentStatement(@NotNull PydmlParser.FunctionCallAssignmentStatementContext ctx);

	/**
	 * Enter a parse tree produced by the {@code AddSubExpression}
	 * labeled alternative in {@link PydmlParser#expression}.
	 * @param ctx the parse tree
	 */
	void enterAddSubExpression(@NotNull PydmlParser.AddSubExpressionContext ctx);
	/**
	 * Exit a parse tree produced by the {@code AddSubExpression}
	 * labeled alternative in {@link PydmlParser#expression}.
	 * @param ctx the parse tree
	 */
	void exitAddSubExpression(@NotNull PydmlParser.AddSubExpressionContext ctx);

	/**
	 * Enter a parse tree produced by the {@code IfStatement}
	 * labeled alternative in {@link PydmlParser#statement}.
	 * @param ctx the parse tree
	 */
	void enterIfStatement(@NotNull PydmlParser.IfStatementContext ctx);
	/**
	 * Exit a parse tree produced by the {@code IfStatement}
	 * labeled alternative in {@link PydmlParser#statement}.
	 * @param ctx the parse tree
	 */
	void exitIfStatement(@NotNull PydmlParser.IfStatementContext ctx);

	/**
	 * Enter a parse tree produced by the {@code IgnoreNewLine}
	 * labeled alternative in {@link PydmlParser#statement}.
	 * @param ctx the parse tree
	 */
	void enterIgnoreNewLine(@NotNull PydmlParser.IgnoreNewLineContext ctx);
	/**
	 * Exit a parse tree produced by the {@code IgnoreNewLine}
	 * labeled alternative in {@link PydmlParser#statement}.
	 * @param ctx the parse tree
	 */
	void exitIgnoreNewLine(@NotNull PydmlParser.IgnoreNewLineContext ctx);

	/**
	 * Enter a parse tree produced by the {@code ConstDoubleIdExpression}
	 * labeled alternative in {@link PydmlParser#expression}.
	 * @param ctx the parse tree
	 */
	void enterConstDoubleIdExpression(@NotNull PydmlParser.ConstDoubleIdExpressionContext ctx);
	/**
	 * Exit a parse tree produced by the {@code ConstDoubleIdExpression}
	 * labeled alternative in {@link PydmlParser#expression}.
	 * @param ctx the parse tree
	 */
	void exitConstDoubleIdExpression(@NotNull PydmlParser.ConstDoubleIdExpressionContext ctx);

	/**
	 * Enter a parse tree produced by the {@code MatrixDataTypeCheck}
	 * labeled alternative in {@link PydmlParser#dataType}.
	 * @param ctx the parse tree
	 */
	void enterMatrixDataTypeCheck(@NotNull PydmlParser.MatrixDataTypeCheckContext ctx);
	/**
	 * Exit a parse tree produced by the {@code MatrixDataTypeCheck}
	 * labeled alternative in {@link PydmlParser#dataType}.
	 * @param ctx the parse tree
	 */
	void exitMatrixDataTypeCheck(@NotNull PydmlParser.MatrixDataTypeCheckContext ctx);

	/**
	 * Enter a parse tree produced by the {@code CommandlinePositionExpression}
	 * labeled alternative in {@link PydmlParser#dataIdentifier}.
	 * @param ctx the parse tree
	 */
	void enterCommandlinePositionExpression(@NotNull PydmlParser.CommandlinePositionExpressionContext ctx);
	/**
	 * Exit a parse tree produced by the {@code CommandlinePositionExpression}
	 * labeled alternative in {@link PydmlParser#dataIdentifier}.
	 * @param ctx the parse tree
	 */
	void exitCommandlinePositionExpression(@NotNull PydmlParser.CommandlinePositionExpressionContext ctx);

	/**
	 * Enter a parse tree produced by the {@code IterablePredicateColonExpression}
	 * labeled alternative in {@link PydmlParser#iterablePredicate}.
	 * @param ctx the parse tree
	 */
	void enterIterablePredicateColonExpression(@NotNull PydmlParser.IterablePredicateColonExpressionContext ctx);
	/**
	 * Exit a parse tree produced by the {@code IterablePredicateColonExpression}
	 * labeled alternative in {@link PydmlParser#iterablePredicate}.
	 * @param ctx the parse tree
	 */
	void exitIterablePredicateColonExpression(@NotNull PydmlParser.IterablePredicateColonExpressionContext ctx);

	/**
	 * Enter a parse tree produced by the {@code AssignmentStatement}
	 * labeled alternative in {@link PydmlParser#statement}.
	 * @param ctx the parse tree
	 */
	void enterAssignmentStatement(@NotNull PydmlParser.AssignmentStatementContext ctx);
	/**
	 * Exit a parse tree produced by the {@code AssignmentStatement}
	 * labeled alternative in {@link PydmlParser#statement}.
	 * @param ctx the parse tree
	 */
	void exitAssignmentStatement(@NotNull PydmlParser.AssignmentStatementContext ctx);

	/**
	 * Enter a parse tree produced by {@link PydmlParser#ml_type}.
	 * @param ctx the parse tree
	 */
	void enterMl_type(@NotNull PydmlParser.Ml_typeContext ctx);
	/**
	 * Exit a parse tree produced by {@link PydmlParser#ml_type}.
	 * @param ctx the parse tree
	 */
	void exitMl_type(@NotNull PydmlParser.Ml_typeContext ctx);

	/**
	 * Enter a parse tree produced by the {@code BooleanAndExpression}
	 * labeled alternative in {@link PydmlParser#expression}.
	 * @param ctx the parse tree
	 */
	void enterBooleanAndExpression(@NotNull PydmlParser.BooleanAndExpressionContext ctx);
	/**
	 * Exit a parse tree produced by the {@code BooleanAndExpression}
	 * labeled alternative in {@link PydmlParser#expression}.
	 * @param ctx the parse tree
	 */
	void exitBooleanAndExpression(@NotNull PydmlParser.BooleanAndExpressionContext ctx);

	/**
	 * Enter a parse tree produced by the {@code ForStatement}
	 * labeled alternative in {@link PydmlParser#statement}.
	 * @param ctx the parse tree
	 */
	void enterForStatement(@NotNull PydmlParser.ForStatementContext ctx);
	/**
	 * Exit a parse tree produced by the {@code ForStatement}
	 * labeled alternative in {@link PydmlParser#statement}.
	 * @param ctx the parse tree
	 */
	void exitForStatement(@NotNull PydmlParser.ForStatementContext ctx);

	/**
	 * Enter a parse tree produced by the {@code RelationalExpression}
	 * labeled alternative in {@link PydmlParser#expression}.
	 * @param ctx the parse tree
	 */
	void enterRelationalExpression(@NotNull PydmlParser.RelationalExpressionContext ctx);
	/**
	 * Exit a parse tree produced by the {@code RelationalExpression}
	 * labeled alternative in {@link PydmlParser#expression}.
	 * @param ctx the parse tree
	 */
	void exitRelationalExpression(@NotNull PydmlParser.RelationalExpressionContext ctx);

	/**
	 * Enter a parse tree produced by {@link PydmlParser#typedArgNoAssign}.
	 * @param ctx the parse tree
	 */
	void enterTypedArgNoAssign(@NotNull PydmlParser.TypedArgNoAssignContext ctx);
	/**
	 * Exit a parse tree produced by {@link PydmlParser#typedArgNoAssign}.
	 * @param ctx the parse tree
	 */
	void exitTypedArgNoAssign(@NotNull PydmlParser.TypedArgNoAssignContext ctx);

	/**
	 * Enter a parse tree produced by {@link PydmlParser#strictParameterizedExpression}.
	 * @param ctx the parse tree
	 */
	void enterStrictParameterizedExpression(@NotNull PydmlParser.StrictParameterizedExpressionContext ctx);
	/**
	 * Exit a parse tree produced by {@link PydmlParser#strictParameterizedExpression}.
	 * @param ctx the parse tree
	 */
	void exitStrictParameterizedExpression(@NotNull PydmlParser.StrictParameterizedExpressionContext ctx);

	/**
	 * Enter a parse tree produced by the {@code MultDivExpression}
	 * labeled alternative in {@link PydmlParser#expression}.
	 * @param ctx the parse tree
	 */
	void enterMultDivExpression(@NotNull PydmlParser.MultDivExpressionContext ctx);
	/**
	 * Exit a parse tree produced by the {@code MultDivExpression}
	 * labeled alternative in {@link PydmlParser#expression}.
	 * @param ctx the parse tree
	 */
	void exitMultDivExpression(@NotNull PydmlParser.MultDivExpressionContext ctx);

	/**
	 * Enter a parse tree produced by the {@code ConstFalseExpression}
	 * labeled alternative in {@link PydmlParser#expression}.
	 * @param ctx the parse tree
	 */
	void enterConstFalseExpression(@NotNull PydmlParser.ConstFalseExpressionContext ctx);
	/**
	 * Exit a parse tree produced by the {@code ConstFalseExpression}
	 * labeled alternative in {@link PydmlParser#expression}.
	 * @param ctx the parse tree
	 */
	void exitConstFalseExpression(@NotNull PydmlParser.ConstFalseExpressionContext ctx);

	/**
	 * Enter a parse tree produced by {@link PydmlParser#strictParameterizedKeyValueString}.
	 * @param ctx the parse tree
	 */
	void enterStrictParameterizedKeyValueString(@NotNull PydmlParser.StrictParameterizedKeyValueStringContext ctx);
	/**
	 * Exit a parse tree produced by {@link PydmlParser#strictParameterizedKeyValueString}.
	 * @param ctx the parse tree
	 */
	void exitStrictParameterizedKeyValueString(@NotNull PydmlParser.StrictParameterizedKeyValueStringContext ctx);

	/**
	 * Enter a parse tree produced by the {@code DataIdExpression}
	 * labeled alternative in {@link PydmlParser#expression}.
	 * @param ctx the parse tree
	 */
	void enterDataIdExpression(@NotNull PydmlParser.DataIdExpressionContext ctx);
	/**
	 * Exit a parse tree produced by the {@code DataIdExpression}
	 * labeled alternative in {@link PydmlParser#expression}.
	 * @param ctx the parse tree
	 */
	void exitDataIdExpression(@NotNull PydmlParser.DataIdExpressionContext ctx);

	/**
	 * Enter a parse tree produced by the {@code IndexedExpression}
	 * labeled alternative in {@link PydmlParser#dataIdentifier}.
	 * @param ctx the parse tree
	 */
	void enterIndexedExpression(@NotNull PydmlParser.IndexedExpressionContext ctx);
	/**
	 * Exit a parse tree produced by the {@code IndexedExpression}
	 * labeled alternative in {@link PydmlParser#dataIdentifier}.
	 * @param ctx the parse tree
	 */
	void exitIndexedExpression(@NotNull PydmlParser.IndexedExpressionContext ctx);

	/**
	 * Enter a parse tree produced by {@link PydmlParser#parameterizedExpression}.
	 * @param ctx the parse tree
	 */
	void enterParameterizedExpression(@NotNull PydmlParser.ParameterizedExpressionContext ctx);
	/**
	 * Exit a parse tree produced by {@link PydmlParser#parameterizedExpression}.
	 * @param ctx the parse tree
	 */
	void exitParameterizedExpression(@NotNull PydmlParser.ParameterizedExpressionContext ctx);

	/**
	 * Enter a parse tree produced by the {@code FunctionCallMultiAssignmentStatement}
	 * labeled alternative in {@link PydmlParser#statement}.
	 * @param ctx the parse tree
	 */
	void enterFunctionCallMultiAssignmentStatement(@NotNull PydmlParser.FunctionCallMultiAssignmentStatementContext ctx);
	/**
	 * Exit a parse tree produced by the {@code FunctionCallMultiAssignmentStatement}
	 * labeled alternative in {@link PydmlParser#statement}.
	 * @param ctx the parse tree
	 */
	void exitFunctionCallMultiAssignmentStatement(@NotNull PydmlParser.FunctionCallMultiAssignmentStatementContext ctx);

	/**
	 * Enter a parse tree produced by the {@code IterablePredicateSeqExpression}
	 * labeled alternative in {@link PydmlParser#iterablePredicate}.
	 * @param ctx the parse tree
	 */
	void enterIterablePredicateSeqExpression(@NotNull PydmlParser.IterablePredicateSeqExpressionContext ctx);
	/**
	 * Exit a parse tree produced by the {@code IterablePredicateSeqExpression}
	 * labeled alternative in {@link PydmlParser#iterablePredicate}.
	 * @param ctx the parse tree
	 */
	void exitIterablePredicateSeqExpression(@NotNull PydmlParser.IterablePredicateSeqExpressionContext ctx);

	/**
	 * Enter a parse tree produced by the {@code SimpleDataIdentifierExpression}
	 * labeled alternative in {@link PydmlParser#dataIdentifier}.
	 * @param ctx the parse tree
	 */
	void enterSimpleDataIdentifierExpression(@NotNull PydmlParser.SimpleDataIdentifierExpressionContext ctx);
	/**
	 * Exit a parse tree produced by the {@code SimpleDataIdentifierExpression}
	 * labeled alternative in {@link PydmlParser#dataIdentifier}.
	 * @param ctx the parse tree
	 */
	void exitSimpleDataIdentifierExpression(@NotNull PydmlParser.SimpleDataIdentifierExpressionContext ctx);

	/**
	 * Enter a parse tree produced by the {@code BooleanOrExpression}
	 * labeled alternative in {@link PydmlParser#expression}.
	 * @param ctx the parse tree
	 */
	void enterBooleanOrExpression(@NotNull PydmlParser.BooleanOrExpressionContext ctx);
	/**
	 * Exit a parse tree produced by the {@code BooleanOrExpression}
	 * labeled alternative in {@link PydmlParser#expression}.
	 * @param ctx the parse tree
	 */
	void exitBooleanOrExpression(@NotNull PydmlParser.BooleanOrExpressionContext ctx);
}