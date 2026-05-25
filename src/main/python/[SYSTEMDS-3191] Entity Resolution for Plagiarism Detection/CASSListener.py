# Generated from CASS.g4 by ANTLR 4.13.2
from antlr4 import *
if "." in __name__:
    from .CASSParser import CASSParser
else:
    from CASSParser import CASSParser

# This class defines a complete listener for a parse tree produced by CASSParser.
class CASSListener(ParseTreeListener):

    # Enter a parse tree produced by CASSParser#prog.
    def enterProg(self, ctx:CASSParser.ProgContext):
        pass

    # Exit a parse tree produced by CASSParser#prog.
    def exitProg(self, ctx:CASSParser.ProgContext):
        pass


    # Enter a parse tree produced by CASSParser#functionDefinition.
    def enterFunctionDefinition(self, ctx:CASSParser.FunctionDefinitionContext):
        pass

    # Exit a parse tree produced by CASSParser#functionDefinition.
    def exitFunctionDefinition(self, ctx:CASSParser.FunctionDefinitionContext):
        pass


    # Enter a parse tree produced by CASSParser#compoundStatement.
    def enterCompoundStatement(self, ctx:CASSParser.CompoundStatementContext):
        pass

    # Exit a parse tree produced by CASSParser#compoundStatement.
    def exitCompoundStatement(self, ctx:CASSParser.CompoundStatementContext):
        pass


    # Enter a parse tree produced by CASSParser#statement.
    def enterStatement(self, ctx:CASSParser.StatementContext):
        pass

    # Exit a parse tree produced by CASSParser#statement.
    def exitStatement(self, ctx:CASSParser.StatementContext):
        pass


    # Enter a parse tree produced by CASSParser#declarationStatement.
    def enterDeclarationStatement(self, ctx:CASSParser.DeclarationStatementContext):
        pass

    # Exit a parse tree produced by CASSParser#declarationStatement.
    def exitDeclarationStatement(self, ctx:CASSParser.DeclarationStatementContext):
        pass


    # Enter a parse tree produced by CASSParser#forBlockStatement.
    def enterForBlockStatement(self, ctx:CASSParser.ForBlockStatementContext):
        pass

    # Exit a parse tree produced by CASSParser#forBlockStatement.
    def exitForBlockStatement(self, ctx:CASSParser.ForBlockStatementContext):
        pass


    # Enter a parse tree produced by CASSParser#forSingleStatement.
    def enterForSingleStatement(self, ctx:CASSParser.ForSingleStatementContext):
        pass

    # Exit a parse tree produced by CASSParser#forSingleStatement.
    def exitForSingleStatement(self, ctx:CASSParser.ForSingleStatementContext):
        pass


    # Enter a parse tree produced by CASSParser#conditionClause.
    def enterConditionClause(self, ctx:CASSParser.ConditionClauseContext):
        pass

    # Exit a parse tree produced by CASSParser#conditionClause.
    def exitConditionClause(self, ctx:CASSParser.ConditionClauseContext):
        pass


    # Enter a parse tree produced by CASSParser#whileBlockStatement.
    def enterWhileBlockStatement(self, ctx:CASSParser.WhileBlockStatementContext):
        pass

    # Exit a parse tree produced by CASSParser#whileBlockStatement.
    def exitWhileBlockStatement(self, ctx:CASSParser.WhileBlockStatementContext):
        pass


    # Enter a parse tree produced by CASSParser#whileSingleStatement.
    def enterWhileSingleStatement(self, ctx:CASSParser.WhileSingleStatementContext):
        pass

    # Exit a parse tree produced by CASSParser#whileSingleStatement.
    def exitWhileSingleStatement(self, ctx:CASSParser.WhileSingleStatementContext):
        pass


    # Enter a parse tree produced by CASSParser#ifBlockStatement.
    def enterIfBlockStatement(self, ctx:CASSParser.IfBlockStatementContext):
        pass

    # Exit a parse tree produced by CASSParser#ifBlockStatement.
    def exitIfBlockStatement(self, ctx:CASSParser.IfBlockStatementContext):
        pass


    # Enter a parse tree produced by CASSParser#ifSingleStatement.
    def enterIfSingleStatement(self, ctx:CASSParser.IfSingleStatementContext):
        pass

    # Exit a parse tree produced by CASSParser#ifSingleStatement.
    def exitIfSingleStatement(self, ctx:CASSParser.IfSingleStatementContext):
        pass


    # Enter a parse tree produced by CASSParser#elseClause.
    def enterElseClause(self, ctx:CASSParser.ElseClauseContext):
        pass

    # Exit a parse tree produced by CASSParser#elseClause.
    def exitElseClause(self, ctx:CASSParser.ElseClauseContext):
        pass


    # Enter a parse tree produced by CASSParser#switchStatement.
    def enterSwitchStatement(self, ctx:CASSParser.SwitchStatementContext):
        pass

    # Exit a parse tree produced by CASSParser#switchStatement.
    def exitSwitchStatement(self, ctx:CASSParser.SwitchStatementContext):
        pass


    # Enter a parse tree produced by CASSParser#caseStatement.
    def enterCaseStatement(self, ctx:CASSParser.CaseStatementContext):
        pass

    # Exit a parse tree produced by CASSParser#caseStatement.
    def exitCaseStatement(self, ctx:CASSParser.CaseStatementContext):
        pass


    # Enter a parse tree produced by CASSParser#functionCall.
    def enterFunctionCall(self, ctx:CASSParser.FunctionCallContext):
        pass

    # Exit a parse tree produced by CASSParser#functionCall.
    def exitFunctionCall(self, ctx:CASSParser.FunctionCallContext):
        pass


    # Enter a parse tree produced by CASSParser#arrayDeclarator.
    def enterArrayDeclarator(self, ctx:CASSParser.ArrayDeclaratorContext):
        pass

    # Exit a parse tree produced by CASSParser#arrayDeclarator.
    def exitArrayDeclarator(self, ctx:CASSParser.ArrayDeclaratorContext):
        pass


    # Enter a parse tree produced by CASSParser#listInitializer.
    def enterListInitializer(self, ctx:CASSParser.ListInitializerContext):
        pass

    # Exit a parse tree produced by CASSParser#listInitializer.
    def exitListInitializer(self, ctx:CASSParser.ListInitializerContext):
        pass


    # Enter a parse tree produced by CASSParser#emptyInitializer.
    def enterEmptyInitializer(self, ctx:CASSParser.EmptyInitializerContext):
        pass

    # Exit a parse tree produced by CASSParser#emptyInitializer.
    def exitEmptyInitializer(self, ctx:CASSParser.EmptyInitializerContext):
        pass


    # Enter a parse tree produced by CASSParser#nullptr.
    def enterNullptr(self, ctx:CASSParser.NullptrContext):
        pass

    # Exit a parse tree produced by CASSParser#nullptr.
    def exitNullptr(self, ctx:CASSParser.NullptrContext):
        pass


    # Enter a parse tree produced by CASSParser#argumentList.
    def enterArgumentList(self, ctx:CASSParser.ArgumentListContext):
        pass

    # Exit a parse tree produced by CASSParser#argumentList.
    def exitArgumentList(self, ctx:CASSParser.ArgumentListContext):
        pass


    # Enter a parse tree produced by CASSParser#returnStatement.
    def enterReturnStatement(self, ctx:CASSParser.ReturnStatementContext):
        pass

    # Exit a parse tree produced by CASSParser#returnStatement.
    def exitReturnStatement(self, ctx:CASSParser.ReturnStatementContext):
        pass


    # Enter a parse tree produced by CASSParser#expressionStatement.
    def enterExpressionStatement(self, ctx:CASSParser.ExpressionStatementContext):
        pass

    # Exit a parse tree produced by CASSParser#expressionStatement.
    def exitExpressionStatement(self, ctx:CASSParser.ExpressionStatementContext):
        pass


    # Enter a parse tree produced by CASSParser#includeStatement.
    def enterIncludeStatement(self, ctx:CASSParser.IncludeStatementContext):
        pass

    # Exit a parse tree produced by CASSParser#includeStatement.
    def exitIncludeStatement(self, ctx:CASSParser.IncludeStatementContext):
        pass


    # Enter a parse tree produced by CASSParser#parameterList.
    def enterParameterList(self, ctx:CASSParser.ParameterListContext):
        pass

    # Exit a parse tree produced by CASSParser#parameterList.
    def exitParameterList(self, ctx:CASSParser.ParameterListContext):
        pass


    # Enter a parse tree produced by CASSParser#parameter.
    def enterParameter(self, ctx:CASSParser.ParameterContext):
        pass

    # Exit a parse tree produced by CASSParser#parameter.
    def exitParameter(self, ctx:CASSParser.ParameterContext):
        pass


    # Enter a parse tree produced by CASSParser#typeSpec.
    def enterTypeSpec(self, ctx:CASSParser.TypeSpecContext):
        pass

    # Exit a parse tree produced by CASSParser#typeSpec.
    def exitTypeSpec(self, ctx:CASSParser.TypeSpecContext):
        pass


    # Enter a parse tree produced by CASSParser#expression.
    def enterExpression(self, ctx:CASSParser.ExpressionContext):
        pass

    # Exit a parse tree produced by CASSParser#expression.
    def exitExpression(self, ctx:CASSParser.ExpressionContext):
        pass


    # Enter a parse tree produced by CASSParser#defaultExpression.
    def enterDefaultExpression(self, ctx:CASSParser.DefaultExpressionContext):
        pass

    # Exit a parse tree produced by CASSParser#defaultExpression.
    def exitDefaultExpression(self, ctx:CASSParser.DefaultExpressionContext):
        pass


    # Enter a parse tree produced by CASSParser#breakExpression.
    def enterBreakExpression(self, ctx:CASSParser.BreakExpressionContext):
        pass

    # Exit a parse tree produced by CASSParser#breakExpression.
    def exitBreakExpression(self, ctx:CASSParser.BreakExpressionContext):
        pass


    # Enter a parse tree produced by CASSParser#assignmentExpression.
    def enterAssignmentExpression(self, ctx:CASSParser.AssignmentExpressionContext):
        pass

    # Exit a parse tree produced by CASSParser#assignmentExpression.
    def exitAssignmentExpression(self, ctx:CASSParser.AssignmentExpressionContext):
        pass


    # Enter a parse tree produced by CASSParser#unaryExpression.
    def enterUnaryExpression(self, ctx:CASSParser.UnaryExpressionContext):
        pass

    # Exit a parse tree produced by CASSParser#unaryExpression.
    def exitUnaryExpression(self, ctx:CASSParser.UnaryExpressionContext):
        pass


    # Enter a parse tree produced by CASSParser#comparingExpression.
    def enterComparingExpression(self, ctx:CASSParser.ComparingExpressionContext):
        pass

    # Exit a parse tree produced by CASSParser#comparingExpression.
    def exitComparingExpression(self, ctx:CASSParser.ComparingExpressionContext):
        pass


    # Enter a parse tree produced by CASSParser#primaryExpression.
    def enterPrimaryExpression(self, ctx:CASSParser.PrimaryExpressionContext):
        pass

    # Exit a parse tree produced by CASSParser#primaryExpression.
    def exitPrimaryExpression(self, ctx:CASSParser.PrimaryExpressionContext):
        pass


    # Enter a parse tree produced by CASSParser#pointerExpression.
    def enterPointerExpression(self, ctx:CASSParser.PointerExpressionContext):
        pass

    # Exit a parse tree produced by CASSParser#pointerExpression.
    def exitPointerExpression(self, ctx:CASSParser.PointerExpressionContext):
        pass


    # Enter a parse tree produced by CASSParser#assignmentOperator.
    def enterAssignmentOperator(self, ctx:CASSParser.AssignmentOperatorContext):
        pass

    # Exit a parse tree produced by CASSParser#assignmentOperator.
    def exitAssignmentOperator(self, ctx:CASSParser.AssignmentOperatorContext):
        pass


    # Enter a parse tree produced by CASSParser#logicalOrExpression.
    def enterLogicalOrExpression(self, ctx:CASSParser.LogicalOrExpressionContext):
        pass

    # Exit a parse tree produced by CASSParser#logicalOrExpression.
    def exitLogicalOrExpression(self, ctx:CASSParser.LogicalOrExpressionContext):
        pass


    # Enter a parse tree produced by CASSParser#logicalAndExpression.
    def enterLogicalAndExpression(self, ctx:CASSParser.LogicalAndExpressionContext):
        pass

    # Exit a parse tree produced by CASSParser#logicalAndExpression.
    def exitLogicalAndExpression(self, ctx:CASSParser.LogicalAndExpressionContext):
        pass


    # Enter a parse tree produced by CASSParser#equalityExpression.
    def enterEqualityExpression(self, ctx:CASSParser.EqualityExpressionContext):
        pass

    # Exit a parse tree produced by CASSParser#equalityExpression.
    def exitEqualityExpression(self, ctx:CASSParser.EqualityExpressionContext):
        pass


    # Enter a parse tree produced by CASSParser#relationalExpression.
    def enterRelationalExpression(self, ctx:CASSParser.RelationalExpressionContext):
        pass

    # Exit a parse tree produced by CASSParser#relationalExpression.
    def exitRelationalExpression(self, ctx:CASSParser.RelationalExpressionContext):
        pass


    # Enter a parse tree produced by CASSParser#additiveExpression.
    def enterAdditiveExpression(self, ctx:CASSParser.AdditiveExpressionContext):
        pass

    # Exit a parse tree produced by CASSParser#additiveExpression.
    def exitAdditiveExpression(self, ctx:CASSParser.AdditiveExpressionContext):
        pass


    # Enter a parse tree produced by CASSParser#multiplicativeExpression.
    def enterMultiplicativeExpression(self, ctx:CASSParser.MultiplicativeExpressionContext):
        pass

    # Exit a parse tree produced by CASSParser#multiplicativeExpression.
    def exitMultiplicativeExpression(self, ctx:CASSParser.MultiplicativeExpressionContext):
        pass


    # Enter a parse tree produced by CASSParser#operationExpression.
    def enterOperationExpression(self, ctx:CASSParser.OperationExpressionContext):
        pass

    # Exit a parse tree produced by CASSParser#operationExpression.
    def exitOperationExpression(self, ctx:CASSParser.OperationExpressionContext):
        pass



del CASSParser