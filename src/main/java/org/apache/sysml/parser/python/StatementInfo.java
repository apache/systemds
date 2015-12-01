package org.apache.sysml.parser.python;

/**
 * This class exists solely to prevent compiler warnings.
 * 
 * <p>
 * The ExpressionInfo and StatementInfo classes are shared among both parsers
 * (R-like and Python-like dialects), and Antlr-generated code assumes that
 * these classes are present in the parser's namespace.
 */
class StatementInfo extends org.apache.sysml.parser.antlr4.StatementInfo {

}
