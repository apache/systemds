from CASSVisitor import CASSVisitor
from CASSParser import CASSParser
from CASSNode import CassNode 

"""
Original implementation by IntelLabs: https://github.com/IntelLabs/MICSAS/tree/master

Cass strings in the original implementation come with a configuration setup. This is the setup (most likely) used in this project and decides on whether a node gets a label. 
  
    annot_mode : Annotations
    compound_mode: Compound Statements
    gvar_mode : Global Variable 
    gfun_mode : Global Function  
    fsig_mode : Function Signatures

Unfortunately we don't know for sure how labels influence the Cass string as changing the configuration inside the original implementation cloned from GitHub has no effect on the final output string.

A Node according to the original implementation consists of 3 parts:
   1) Node type :
        -> I : Internal
        -> N / C / S : Number / Char / String literal
        -> F : Global function
        -> f : Local function
        -> V : Global variable
        -> v : Local variable
        -> S : Function signature
        -> E : Error * not implemented

    2) Annotation : #compound_statement#, #init_declarator# etc.
    3) Labels : #VAR, #GVAR, #GFUN, {#} ? 

In this project we focused on replicating the Cass string building logic of the original implementation to be able to then vectorize it and calculate a similarity score. Our nodes include node types, annotations, observed suffixes representing a binary operation or the number of immediate children nodes using dollar signs "$=$", {$$$} as well as variable/function names. While building the string, nodes are seperated using '\t' followed with either number of immediate children nodes like '2' or the number (id) of the node that a local variable/function has been previously used it, and the node number where it will next be used it (current node numer).

Here's an example of a Cass string for a simple program obtained from the original implementation showing the structure we were trying to implement :

int summation(int start_val, int end_val) { 
    int sum = 0; 
    for (int i = start_val; i <= end_val; ++i) {
        sum += i;
    } 
    return sum;
}

['0,0,6,1\t23\tS#FS#1_2\tI#compound_statement#{$$$}\t3\tI#declaration#int$;\t1\tI#init_declarator#$=$\t2\tvsum\t-1\t19\tN0\tI#for_statement#for($$;$)$\t4\tI#declaration#int$;\t1\tI#init_declarator#$=$\t2\tvi\t-1\t12\tvstart_val\t-1\t-1\tI#binary_expression#$<=$\t2\tvi\t9\t15\tvend_val\t-1\t-1\tI#update_expression#++$\t1\tvi\t12\t20\tI#compound_statement#{$}\t1\tI#expression_statement#$;\t1\tI#assignment_expression#$+=$\t2\tvsum\t4\t22\tvi\t15\t-1\tI#return_statement#return$;\t1\tvsum\t19\t-1\t']

-> 0,0,6,1 being the source range (start_row, start_col, end_row, end_col)
-> first number is the total number of nodes, here 23
"""



class MyCassVisitor(CASSVisitor):

    def __init__(self):
        super().__init__()
        self.scopes = []  # keeping track of scopes to differenciate between local/global
        self.function_nesting_level = 0

    def visitProg(self, ctx: CASSParser.ProgContext):

        root = CassNode("removed")  # Root node to remove later

        for statement in ctx.statement():
            node = self.visit(statement)

            if isinstance(node, CassNode):
                # If the node is a top-level function, treat it separately
                if node.label.startswith("S#FS#"):
                    root.add_child(node)
                else:
                    # Other statements or nested functions are added normally
                    if root.children:
                        root.children[-1].add_child(node)
                    else:
                        root.add_child(node)

        return root
    
    def visitFunctionDefinition(self, ctx: CASSParser.FunctionDefinitionContext):
        """
        If we're at the top (most global) level, we produce an "S#FS#..." node.
        If it's nested (within another function), we produce an "I#function_definition#...".
        But in both cases, we push a scope so that parameters and local variables become local.
        """

        self.function_nesting_level += 1
        # 1) Push a new scope so that parameters/locals are recognized as local
        self.scopes.append(set())
        in_global_scope = (self.function_nesting_level == 1)

        # 2) Build the function node label
        func_type_text = ctx.typeSpec().getText()
        params_num = 0
        if ctx.parameterList():
            params_num = len(ctx.parameterList().parameter())

        if in_global_scope:
            # This is the very first function => produce "S#FS#..." style
            func_type = 0 if func_type_text == 'void' else 1
            # Possibly clamp param count
            if params_num > 2:
                params_num = 2
            node = CassNode(f"S#FS#{func_type}_{params_num}")

            start_line = ctx.start.line -1
            start_col = ctx.start.column
            end_line = ctx.stop.line -1
            end_col = ctx.stop.column +1

            node.source_range = (start_line, start_col, end_line, end_col)
        else:
            # Nested function => produce "I#function_definition#..." style
            node = CassNode(f"I#function_definition#{func_type_text}$$")

       
        if not in_global_scope:
            # Create "I#function_declarator#$$" or similar
            if params_num == 0:
                decl_label = "I#function_declarator#$()"
            else:
                decl_label = "I#function_declarator#$$"
            decl_node = CassNode(decl_label)

            # The function name (the grammar has "primaryExpression" after typeSpec)
            func_name = ctx.primaryExpression().getText()
            decl_node.add_child(CassNode(f"v{func_name}"))

            # If there are parameters, build the param list (which also adds them to scope)
            if ctx.parameterList():
                param_list_node = self.visitParameterList(ctx.parameterList())
                decl_node.add_child(param_list_node)

            node.add_child(decl_node)
        else:
           
            if ctx.parameterList():
                self.visitParameterList(ctx.parameterList())  # Adds param names to scope

        # 4) Visit the compound statement so local declarations become part of the scope
        block_node = self.visit(ctx.compoundStatement())
        node.add_child(block_node)

        # 5) Pop the scope after finishing
        self.scopes.pop()

        self.function_nesting_level -= 1

        return node

    def visitParameterList(self, ctx: CASSParser.ParameterListContext):
        
        num_params = len(ctx.parameter())
        placeholders = ",".join(["$"] * num_params)
        node = CassNode(f'I#parameter_list#({placeholders})')
        #node.add_child(CassNode(f"{num_params}"))
        for p in ctx.parameter():
            node.add_child(self.visit(p))
        return node

    def visitParameter(self, ctx: CASSParser.ParameterContext):
        
        param_type = ctx.typeSpec().getText()
        param_name = ctx.primaryExpression().getText()

        if self.scopes:
            self.scopes[-1].add(param_name)

        node = CassNode(f"I#parameter_declaration#{param_type}$")
        #node.add_child(CassNode("1"))
        node.add_child(self.visit(ctx.primaryExpression()))
        
        return node

    def visitCompoundStatement(self, ctx: CASSParser.CompoundStatementContext):
        # 1) Push a new empty set for local declarations in this block
        self.scopes.append(set())
    
        # Count the number of direct statements (children) in the compound statement
        num_children = len(ctx.statement())
        dollar_signs = "$" * num_children  # Create the correct number of $ placeholders
        block_node = CassNode(f"I#compound_statement#{{{dollar_signs}}}")
        #block_node.add_child(CassNode(F"{num_children}"))
        block_node.is_in_comp_stmt = True

        # Add each statement as a direct child
        for st in ctx.statement():
            stmt_node = self.visit(st)
            #stmt_node.is_in_comp_stmt = True
            block_node.add_child(stmt_node)

        # 4) Pop the scope after leaving this block    
        self.scopes.pop()
        return block_node
    
    def visitIncludeStatement(self, ctx: CASSParser.IncludeStatementContext):
        return CassNode("removed")

    def visitDeclarationStatement(self, ctx: CASSParser.DeclarationStatementContext):
        
        type_label = ctx.typeSpec().getText()

        decl_node = CassNode(f"I#declaration#{type_label}$;")

        #Array handling
        if ctx.arrayDeclarator():
            
            if ctx.emptyInitializer() or ctx.nullptr() or ctx.expression() :
                
                placeholder = '$'

                array_decl = self.visit(ctx.arrayDeclarator())

                if(ctx.emptyInitializer()):
                    placeholder = '{}'

                if(ctx.nullptr()):
                    placeholder = "nullptr"

                if ctx.expression():
                    helperNode = self.visit(ctx.expression())
                    array_decl.add_child(helperNode)
                    
        
                init_decl = CassNode(f"I#init_declarator#$={placeholder}")
                init_decl.add_child(array_decl)
                decl_node.add_child(init_decl)
            
            else:

                decl_node.add_child(self.visit(ctx.arrayDeclarator()))

            return decl_node


        if ctx.primaryExpression():
            
            # Mark this variable as local in the top scope
            var_name = ctx.primaryExpression().getText()

            if len(self.scopes) > 0:
                self.scopes[-1].add(var_name)
    
            pointer_node = CassNode("I#pointer_declarator#*$")

            if ctx.POINTER() and not(ctx.nullptr()) and not(ctx.emptyInitializer()) and not(ctx.expression()) :

                pointer_node.add_child(self.visit(ctx.primaryExpression()))
                decl_node.add_child(pointer_node)
                return decl_node
            
            if ctx.expression() or ctx.nullptr() or ctx.emptyInitializer():

                placeholder = '$'

                if(ctx.emptyInitializer()):
                    placeholder = '{}'

                if(ctx.nullptr()):
                    placeholder = "nullptr"
                
                if ctx.expression():
                    helperNode = self.visit(ctx.expression())

                assign_node = CassNode(f"I#init_declarator#$={placeholder}")
                
                if(ctx.POINTER()):

                    pointer_node.add_child(self.visit(ctx.primaryExpression()))
                    
                    if ctx.expression():
                        pointer_node.add_child(helperNode)
                    
                    assign_node.add_child(pointer_node)

                else :
                    
                    assign_node.add_child(self.visit(ctx.primaryExpression()))
                    
                    if ctx.expression():
                        assign_node.add_child(helperNode)

                
                decl_node.add_child(assign_node)
            
            else:

                decl_node.add_child(self.visit(ctx.primaryExpression()))

            return decl_node
    
    
    def visitListInitializer(self, ctx: CASSParser.ListInitializerContext):

        placeholders = ",".join(["$"] * len(ctx.primaryExpression()))
        list_init = CassNode(f"I#initializer_list#{{{placeholders}}}")
        for c in ctx.primaryExpression():
            list_init.add_child(self.visit(c))

        return list_init

    def visitForBlockStatement(self, ctx: CASSParser.ForBlockStatementContext):
    
        for_node = CassNode(f"I#for_statement#for($$;$)$")
        #for_node.add_child(CassNode("4"))

        # Initialization (forInit)
        if ctx.declarationStatement():
            init_node = self.visit(ctx.declarationStatement())
        else:
            init_node = self.visit(ctx.assignmentExpression())
            
        for_node.add_child(init_node)
       
        cond_node = self.visit(ctx.logicalOrExpression())
        for_node.add_child(cond_node)
       
        # Update (forUpdate)
        if ctx.unaryExpression():
            update_node = self.visit(ctx.unaryExpression())
        else:
            for_node.add_child(CassNode("EMPTY_UPDATE"))

        for_node.add_child(update_node)


        # Body (multiple statements in the block)
        for_node.add_child(self.visit(ctx.compoundStatement()))

        return for_node

    
    def visitForSingleStatement(self, ctx: CASSParser.ForSingleStatementContext):
        for_node = CassNode("I#for_statement#for($$;$)$")
        #for_node.add_child(CassNode("4"))

        # Initialization (forInit)
        if ctx.forInit():
            init_node = self.visit(ctx.forInit())
            for_node.add_child(init_node)
        else:
            for_node.add_child(CassNode("EMPTY_INIT"))

        # Condition
        if ctx.expression():
            cond_node = self.visit(ctx.expression())
            for_node.add_child(cond_node)
        else:
            for_node.add_child(CassNode("EMPTY_COND"))

        # Update (forUpdate)
        if ctx.forUpdate():
            update_node = self.visit(ctx.forUpdate())
            for_node.add_child(update_node)
        else:
            for_node.add_child(CassNode("EMPTY_UPDATE"))

        # Body (single statement)
        body_node = self.visit(ctx.statement())
        for_node.add_child(body_node)


        return for_node

    def visitConditionClause(self, ctx: CASSParser.ConditionClauseContext):
        node = CassNode("I#condition_clause#($)")
    
        if ctx.logicalOrExpression():
            node.add_child(self.visit(ctx.logicalOrExpression()))
        
        return node

    
    def visitWhileBlockStatement(self, ctx: CASSParser.WhileBlockStatementContext):

        while_node = CassNode("I#while_statement#while$$")

        # Condition
        cond_node = self.visit(ctx.conditionClause())
        while_node.add_child(cond_node)
        while_node.add_child(self.visit(ctx.compoundStatement()))

        return while_node
    
    def visitWhileSingleStatement(self, ctx: CASSParser.WhileSingleStatementContext):

        while_node = CassNode("I#while_statement#while$$")

        # Condition
        cond_node = self.visit(ctx.conditionClause())
        while_node.add_child(cond_node)

        # Single body statement
        body_node = self.visit(ctx.statement())
        while_node.add_child(body_node)

        return while_node
    
    def visitIfBlockStatement(self, ctx: CASSParser.IfBlockStatementContext):

        num_children = 0
        if ctx.conditionClause():
            num_children += 1
        
        if ctx.compoundStatement():
            num_children += 1

        if ctx.elseClause():
            num_children += 1

        dollar_signs = "$" * num_children  # Create the correct number of $ placeholders
        
        # Create a node for the "if" statement
        if_node = CassNode(f"I#if_statement#if{dollar_signs}")

        cond_node = self.visit(ctx.conditionClause())
        if_node.add_child(cond_node)

        # Separate "if" and "else" blocks
        # Visit the 'if' body (compoundStatement) and add as a child
        if_body_node = self.visit(ctx.compoundStatement())
        if_node.add_child(if_body_node)

        # Handle 'else' clause if present
        if ctx.elseClause():
            else_clause_node = self.visit(ctx.elseClause())
            if_node.add_child(else_clause_node)

        return if_node
    
    def visitElseClause(self, ctx: CASSParser.ElseClauseContext):
        else_node = CassNode("I#else_clause#else$")

        if ctx.ifBlockStatement():
            nested_if_node = self.visit(ctx.ifBlockStatement())
            else_node.add_child(nested_if_node)
        elif ctx.compoundStatement():
            else_body_node = self.visit(ctx.compoundStatement())
            else_node.add_child(else_body_node)
        else:
            # It's a simple 'else' -> visit the statement
            else_body_node = self.visit(ctx.statement())
            else_node.add_child(else_body_node)

        return else_node


    
    def visitIfSingleStatement(self, ctx: CASSParser.IfSingleStatementContext):
        num_children = 0
        if ctx.conditionClause():
            num_children += 1
        
        if ctx.statement():
            num_children += 1

        if ctx.elseClause():
            num_children += 1

        dollar_signs = "$" * num_children  # Create the correct number of $ placeholders
        
        # Create a node for the "if" statement
        if_node = CassNode(f"I#if_statement#if{dollar_signs}")

        # Condition
        cond_node = self.visit(ctx.conditionClause())
        if_node.add_child(cond_node)

        # Single "if" body statement
        if_body_node = self.visit(ctx.statement())
        if_node.add_child(if_body_node)

        # Optional "else"
        if ctx.elseClause():
            else_node = self.visit(ctx.elseClause())
            if_node.add_child(else_node)

        return if_node
    
    def visitSwitchStatement(self, ctx: CASSParser.SwitchStatementContext):

        switch_node = CassNode("I#switch_statement#switch$$")
        switch_node.add_child(self.visit(ctx.conditionClause()))
        switch_node.add_child(self.visit(ctx.compoundStatement()))
    
        return switch_node
    
    def visitCaseStatement(self, ctx: CASSParser.CaseStatementContext):

        case_name = 'case$'
        has_break = ''

        if ctx.breakExpression():
            has_break = 'break'

        if ctx.defaultExpression():
            case_name = 'default'

        num_statement = len(ctx.statement())
        placeholder = "$" * num_statement

        case_node = CassNode(f"I#case_statement#{case_name}:{placeholder}{has_break}")
        
        if(ctx.primaryExpression()):
            case_node.add_child(self.visit(ctx.primaryExpression()))

        for c in ctx.statement():
            case_node.add_child(self.visit(c))

        return case_node

    def visitLogicalOrExpression(self, ctx: CASSParser.LogicalOrExpressionContext):
        if len(ctx.logicalAndExpression()) == 1:
            # If there is only one logicalAndExpression, visit it directly
            return self.visit(ctx.logicalAndExpression(0))
        
        # Otherwise, create a node to represent the OR operation
        node = CassNode("I#binary_expression#$||$")
        
        
        for expr in ctx.logicalAndExpression():
            node.add_child(self.visit(expr))
        
        return node

    def visitLogicalAndExpression(self, ctx: CASSParser.LogicalAndExpressionContext):
        if len(ctx.equalityExpression()) == 1:
            return self.visit(ctx.equalityExpression(0))
        
        node = CassNode("I#binary_expression#$&&$")
        
        
        for expr in ctx.equalityExpression():
            node.add_child(self.visit(expr))
        
        return node
    
    def visitEqualityExpression(self, ctx: CASSParser.EqualityExpressionContext):
        if len(ctx.relationalExpression()) == 1:
            return self.visit(ctx.relationalExpression(0))
        
        node = CassNode(f"I#binary_expression#${ctx.getChild(1).getText()}$")
        
        
        lhs = self.visit(ctx.relationalExpression(0))  # Left operand
        rhs = self.visit(ctx.relationalExpression(1))  # Right operand
        
        node.add_child(lhs)
        node.add_child(rhs)
        
        return node
    
    def visitRelationalExpression(self, ctx: CASSParser.RelationalExpressionContext):
    # If there's only one child additiveExpression, just pass it up the chain
        if len(ctx.children) == 1:
            return self.visit(ctx.additiveExpression(0))

        # If there's an operator like "<=" or ">" ...
        left = self.visit(ctx.additiveExpression(0))
        op = ctx.getChild(1).getText()  # e.g. "<="
        right = self.visit(ctx.additiveExpression(1))

        # Create a node labeled "$<=$" (or "$>$" etc.)
        node = CassNode(f"I#binary_expression#${op}$")
        node.add_child(left)
        node.add_child(right)
        return node
    
    def visitAdditiveExpression(self, ctx: CASSParser.AdditiveExpressionContext):
        # If there's only one child, pass it up the chain (e.g., "a")
        if len(ctx.children) == 1:
            return self.visit(ctx.multiplicativeExpression(0))

        # If there are multiple operands, create a node for each operator and operand
        operands = ctx.multiplicativeExpression()
        result = self.visit(operands[0])  # Start with the first operand

        for i in range(1, len(operands)):
            operator = ctx.getChild(2 * i - 1).getText()  # Get "+" or "-"
            next_operand = self.visit(operands[i])
            operator_node = CassNode(f"I#binary_expression#${operator}$")
            #operator_node.add_child(CassNode("2"))
            operator_node.add_child(result)
            operator_node.add_child(next_operand)
            result = operator_node  # Update the result to the new operator node

        return result
    
    def visitMultiplicativeExpression(self, ctx: CASSParser.MultiplicativeExpressionContext):
        # If there's only one child, pass it up the chain (e.g., "a")
        if len(ctx.children) == 1:
            return self.visit(ctx.unaryExpression(0))

        # If there are multiple operands, create a node for each operator and operand
        operands = ctx.unaryExpression()
        result = self.visit(operands[0])  # Start with the first operand

        for i in range(1, len(operands)):
            operator = ctx.getChild(2 * i - 1).getText()  # Get "*" or "/"
            next_operand = self.visit(operands[i])
            operator_node = CassNode(f"I#binary_expression#${operator}$")
            #operator_node.add_child(CassNode("2"))
            operator_node.add_child(result)
            operator_node.add_child(next_operand)
            result = operator_node  # Update the result to the new operator node

        return result
        

    
    def visitFunctionCall(self, ctx: CASSParser.FunctionCallContext):
       
        # 1) The function name is the ID
        func_name = ctx.ID().getText()  # e.g. "init"

        # call_expression always has 2 children: name and parameter list
        call_node = CassNode("I#call_expression#$$")

        # 3) First child = "F<funcName>", e.g. Finit
        func_node = CassNode(f"F{func_name}")
        call_node.add_child(func_node)

        # 4) Second child = the argument list (which might be empty)
        if ctx.argumentList():
            arg_list_node = self.visit(ctx.argumentList())
            call_node.add_child(arg_list_node)
        else:
            # No arguments => #argument_list#() with zero placeholders
            empty_args = CassNode("I#argument_list#()")
            call_node.add_child(empty_args)

        return call_node


    def visitArgumentList(self, ctx: CASSParser.ArgumentListContext):
        """
        Grammar snippet:
            argumentList
                : expression (',' expression)*  # ArgumentList
                ;
        """
        # Count how many arguments we have
        num_args = len(ctx.expression())

        # Create a label like #argument_list#($,$,$...) with as many $ as arguments
        placeholders = ",".join(["$"] * num_args)  # Join $ with commas if more than one
        arg_list_node = CassNode(f"I#argument_list#({placeholders})")
        

        # For each expression argument, visit it and add as a child
        for expr_ctx in ctx.expression():
            arg_node = self.visit(expr_ctx)
            arg_list_node.add_child(arg_node)

        return arg_list_node
    

    def visitReturnStatement(self, ctx: CASSParser.ReturnStatementContext):
        # e.g. "return sum;"
        node = CassNode("I#return_statement#return$;")
        if ctx.expression():
            expr_node = self.visit(ctx.expression())
            node.add_child(expr_node)
        return node

    def visitExpressionStatement(self, ctx: CASSParser.ExpressionStatementContext):

        statement_node = CassNode("I#expression_statement#$;")

        # 2) Visit the expression, which might yield something like "$+=$"
        expr_node = self.visit(ctx.expression())

        # 3) Add it as a child
        statement_node.add_child(expr_node)

        return statement_node


    # ---------------------
    # Expression Collapsing
    # ---------------------
    def visitExpression(self, ctx: CASSParser.ExpressionContext):

        if ctx.assignmentExpression():
            return self.visit(ctx.assignmentExpression())
        
        return None


    def visitAssignmentExpression(self, ctx: CASSParser.AssignmentExpressionContext):
        # Distinguish between:
        #   unaryExpression assignmentOperator assignmentExpression
        # vs
        #   logicalOrExpression

        if ctx.assignmentOperator():
            # e.g. b = b + 1
            op_text = ctx.assignmentOperator().getText()  # '=' or '+=' or ...
            
            placeholder = '$'

            if ctx.nullptr():
                placeholder = 'nullptr'

            if ctx.emptyInitializer():
                placeholder = '{}'

            # Use a node labeled #assignment_expression#$<op_text>
            # For a simple '=' you might produce '#assignment_expression#$=$'
            # For '+=' maybe '#assignment_expression#$+=$', etc.
            node = CassNode(f"I#assignment_expression#$" + op_text + placeholder)

            lhs = self.visit(ctx.unaryExpression())  # e.g. b
            node.add_child(lhs)

            if ctx.assignmentExpression():
                rhs = self.visit(ctx.assignmentExpression())  # e.g. b + 1
                node.add_child(rhs)

            
            return node
        else:
            # No assignment operator => just pass logicalOrExpression up
            return self.visit(ctx.logicalOrExpression())


    def visitUnaryExpression(self, ctx: CASSParser.UnaryExpressionContext):
       
        if ctx.listInitializer():
            return self.visit(ctx.listInitializer())
        
        if ctx.pointerExpression():
            return self.visit(ctx.pointerExpression())
        
        # If it's prefix like ++i
        if ctx.unaryExpression():
            op = ''.join('$' if x not in ('+', '-') else x for x in ctx.getText())
            node = CassNode(f"I#update_expression#{op}")
            node.add_child(self.visit(ctx.unaryExpression()))
            return node
        else:

            return self.visit(ctx.primaryExpression())
        
    def isLocal(self, var_name: str) -> bool:
        # Search from the top of the stack downward
        for scope_set in reversed(self.scopes):
            if var_name in scope_set:
                return True
        return False

    def visitPointerExpression(self, ctx: CASSParser.PointerExpressionContext): 
    
        var_text = ctx.primaryExpression().getText()
        sign = ctx.getText()[0]

        if self.isLocal(var_text):
            ptr_node = CassNode(f"I#pointer_expression#{sign}$")
            ptr_node.add_child(self.visit(ctx.primaryExpression()))
        else:
            ptr_node = CassNode(f"I#pointer_expression#{sign}$")
            ptr_node.add_child(self.visit(ctx.primaryExpression()))
        
        return ptr_node
    
    def visitArrayDeclarator(self, ctx: CASSParser.ArrayDeclaratorContext):


        if len(ctx.primaryExpression()) > 1:

            var_name = ctx.primaryExpression(0).getText()

            if len(self.scopes) > 0:
                self.scopes[-1].add(var_name)

            arr_dclr = CassNode("I#array_declarator#$[$]")
            arr_dclr.add_child(self.visit(ctx.primaryExpression(0)))
            arr_dclr.add_child(self.visit(ctx.primaryExpression(1)))
        
        elif len(ctx.primaryExpression()) == 1:

            var_name = ctx.primaryExpression(0).getText()
            
            if len(self.scopes) > 0:
                self.scopes[-1].add(var_name)

            arr_dclr = CassNode("I#array_declarator#$[]")
            arr_dclr.add_child(self.visit(ctx.primaryExpression(0)))
        
        return arr_dclr
    

    def visitPrimaryExpression(self, ctx: CASSParser.PrimaryExpressionContext):
        # Case 1: It's an identifier
        if ctx.ID():

            var_text = ctx.ID().getText()

            # Check if var_text is declared in the current or any parent scope
            if self.isLocal(var_text):
                return CassNode(f"v{var_text}")
            else:
                return CassNode(f"V{var_text}")

        
        # Case 2: It's an integer literal
        elif ctx.INT():
            lit_text = ctx.INT().getText()
            return CassNode(f"N{lit_text}")
        
        # Case 3: It's a float literal
        elif ctx.FLOAT():
            lit_text = ctx.FLOAT().getText()
            return CassNode(f"N{lit_text}")
        
        elif ctx.CHAR():
            lit_text = ctx.CHAR().getText()
            return CassNode(f"C{lit_text}")
        
        elif ctx.STRING():
            str_text = ctx.STRING().getText()
            return CassNode(f"S{str_text}")
        
        # Case 4: It's parentheses => ( expression )
        elif ctx.expression():
            # 1) Visit the sub-expression
            subexpr_node = self.visit(ctx.expression())

            # 2) Check if subexpr_node is an additive expression
            #    For example, if your additive visitor produces "$+$" or "$-$" as the label:
            if subexpr_node and subexpr_node.label in {"I#binary_expression#$+$", "I#binary_expression#$-$", "I#binary_expression#$*$", "I#binary_expression#$/$", "I#binary_expression#$%$"}:
                # Create a paren node
                paren_node = CassNode("I#parenthesized_expression#($)")
                paren_node.add_child(subexpr_node)
                return paren_node
            else:
                # If not additive, just return the inner expression without special wrapping
                return subexpr_node
            
        elif ctx.functionCall():
            return self.visit(ctx.functionCall())
        
        # Fallback if something unexpected
        else:
            return CassNode("???")

