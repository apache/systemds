grammar CASS;

// --------------------------
// 1) Top-Level Structure
// --------------------------

// Parse one or more function definitions.
prog
    : statement+ EOF
    ;

functionDefinition
    : typeSpec primaryExpression '(' parameterList? ')' compoundStatement
    ;

// A block of statements in braces
compoundStatement
    : '{' statement* '}'
    ;


// --------------------------
// 2) Declarations & Statements
// --------------------------

statement
   
    : declarationStatement
    | forBlockStatement          
    | forSingleStatement         
    | whileBlockStatement       
    | whileSingleStatement
    | ifBlockStatement           
    | ifSingleStatement    
    | returnStatement
    | switchStatement
    | caseStatement
    | expressionStatement
    | functionDefinition  
    | includeStatement
    ;

declarationStatement
    : typeSpec POINTER* (primaryExpression || arrayDeclarator) ('=' (expression || nullptr || emptyInitializer))? ';'?
    ;

forBlockStatement
    : 'for' '(' (declarationStatement || assignmentExpression) ';' logicalOrExpression ';' unaryExpression ')' compoundStatement
    ;

forSingleStatement
    : 'for' '(' (declarationStatement?|| assignmentExpression) ';' logicalOrExpression ';' unaryExpression ')' statement
    ;

conditionClause
    : logicalOrExpression
    ;

whileBlockStatement
    : 'while' '(' conditionClause ')' compoundStatement
    ;

whileSingleStatement
    : 'while' '(' conditionClause ')' statement
    ;

ifBlockStatement
    : 'if' '(' conditionClause ')' compoundStatement elseClause? 
    ;

ifSingleStatement
    : 'if' '(' conditionClause ')' statement elseClause? 
    ;

elseClause
    : 'else' (compoundStatement | ifBlockStatement | statement)
    ;

switchStatement
    : 'switch' '(' conditionClause ')' compoundStatement
    ;

caseStatement
    : ('case' | defaultExpression) primaryExpression? ':' statement* breakExpression?
    ;

functionCall
    : ID '(' argumentList? ')'  
    ;

arrayDeclarator
    : primaryExpression '[' primaryExpression? ']'
    ;

listInitializer
    : '{' primaryExpression (',' primaryExpression)* '}'
    ;

emptyInitializer 
    : '{' '}'
    ;

nullptr
    : 'nullptr'
    ;

argumentList
    : expression (',' expression)*  
    ;

returnStatement
    : 'return' expression? ';'
    ;

expressionStatement
    : expression ';'
    ;

includeStatement
    : 'include' STRING
    ;
// --------------------------
// 3) Parameters & Types
// --------------------------

parameterList
    : parameter (',' parameter)*
    ;

parameter
    : typeSpec primaryExpression
    ;

typeSpec
    : 'int' 
    | 'float'
    | 'double'
    | 'void'
    ;

// --------------------------
// 4) Expressions
// --------------------------

// For simplicity, we let "expression" wrap typical C operator precedences.


expression
    : assignmentExpression
    | functionCall
    ;

defaultExpression
    : 'default'
    ;

breakExpression
    : 'break' ';'
    ;

assignmentExpression
    : unaryExpression assignmentOperator assignmentExpression nullptr? emptyInitializer?
    | logicalOrExpression
    ;

unaryExpression
    : ('++' | '--') unaryExpression
    | unaryExpression ('++' | '--')
    | pointerExpression
    | primaryExpression
    | listInitializer
    ;

comparingExpression
    : '>'
    | '<'
    | '<='
    | '>='
    ;

primaryExpression
    : ID                     
    | INT                    
    | FLOAT                  
    | CHAR                   
    | STRING
    | BOOL                 
    | functionCall           
    | '(' expression ')'     
    ;

pointerExpression
    : '&' primaryExpression
    | '*' primaryExpression
    ;

assignmentOperator
    : '='
    | '+='
    | '-='
    | '*='
    | '/='
    ;

logicalOrExpression
    : logicalAndExpression ('||' logicalAndExpression)*
    ;

logicalAndExpression
    : equalityExpression ('&&' equalityExpression)*
    ;

equalityExpression
    : relationalExpression (( '==' | '!=' ) relationalExpression)*
    ;

relationalExpression
    : additiveExpression (( '<' | '>' | '<=' | '>=' ) additiveExpression)*
    ;

additiveExpression
    : multiplicativeExpression (( '+' | '-' ) multiplicativeExpression)*
    ;

multiplicativeExpression
    : unaryExpression (( '*' | '/' | '%' ) unaryExpression)*
    ;

operationExpression
    : additiveExpression  // Handles '+' and '-' precedence
    | multiplicativeExpression  // Handles '*' and '/'
    ;

// --------------------------
// 5) Lexer Rules
// --------------------------

SL_COMMENT
    : '//' ~[\r\n]* -> skip
    ;

ML_COMMENT
    : '/*' .*? '*/' -> skip
    ;

ID
    : [a-zA-Z_] [a-zA-Z0-9_]*
    ;

INT
    : '-'? [0-9]+
    ;

BOOL
    : 'true'
    | 'false'
    ;

FLOAT
    : [0-9]+ '.' [0-9]+ ([eE] [+-]? [0-9]+)?
    | '.' [0-9]+ ([eE] [+-]? [0-9]+)?
    | [0-9]+ ([eE] [+-]? [0-9]+)
    ;

CHAR 
    : '"'[a-zA-Z] '"'
    ;

POINTER
    : '*'
    ;


STRING
    : '"' (ESC_SEQ | ~["\\])* '"'  // A string starts and ends with double quotes
    ;

fragment ESC_SEQ
    : '\\' [btnfr"'\\]  // Escape sequences for backslash, single quote, double quote, etc.
    ;

// Skip whitespace and newlines
WS
    : [ \t\r\n]+ -> skip
    ;
