package com.ibm.bi.dml.test.components.parser;

import static org.junit.Assert.*;

import org.junit.Test;

import com.ibm.bi.dml.hops.Hops.FileFormatTypes;
import com.ibm.bi.dml.parser.Expression;
import com.ibm.bi.dml.parser.Statement;
import com.ibm.bi.dml.parser.Expression.BinaryOp;
import com.ibm.bi.dml.parser.Expression.BooleanOp;
import com.ibm.bi.dml.parser.Expression.RelationalOp;


public class ExpressionTest {
    
    @Test
    public void testGetBinaryOp() {
        assertEquals(BinaryOp.PLUS, Expression.getBinaryOp("+"));
        assertEquals(BinaryOp.MINUS, Expression.getBinaryOp("-"));
        assertEquals(BinaryOp.MULT, Expression.getBinaryOp("*"));
        assertEquals(BinaryOp.DIV, Expression.getBinaryOp("/"));
        assertEquals(BinaryOp.MATMULT, Expression.getBinaryOp("%*%"));
        assertEquals(BinaryOp.INVALID, Expression.getBinaryOp("wrong"));
    }
    
    @Test
    public void testGetRelationalOp() {
        assertEquals(RelationalOp.LESS, Expression.getRelationalOp("<"));
        assertEquals(RelationalOp.LESSEQUAL, Expression.getRelationalOp("<="));
        assertEquals(RelationalOp.GREATER, Expression.getRelationalOp(">"));
        assertEquals(RelationalOp.GREATEREQUAL, Expression.getRelationalOp(">="));
        assertEquals(RelationalOp.EQUAL, Expression.getRelationalOp("=="));
        assertEquals(RelationalOp.NOTEQUAL, Expression.getRelationalOp("!="));
        assertEquals(RelationalOp.INVALID, Expression.getRelationalOp("wrong"));
    }
    
    @Test
    public void testGetBooleanOp() {
        assertEquals(BooleanOp.CONDITIONALAND, Expression.getBooleanOp("&&"));
        assertEquals(BooleanOp.LOGICALAND, Expression.getBooleanOp("&"));
        assertEquals(BooleanOp.CONDITIONALOR, Expression.getBooleanOp("||"));
        assertEquals(BooleanOp.LOGICALOR, Expression.getBooleanOp("|"));
        assertEquals(BooleanOp.INVALID, Expression.getBooleanOp("wrong"));
    }
    
    @Test
    public void testConvertFormatType() {
        assertEquals(FileFormatTypes.TEXT, Expression.convertFormatType(null));
        assertEquals(FileFormatTypes.TEXT, Expression.convertFormatType(Statement.FORMAT_TYPE_VALUE_TEXT));
        assertEquals(FileFormatTypes.BINARY, Expression.convertFormatType(Statement.FORMAT_TYPE_VALUE_BINARY));
        assertEquals(FileFormatTypes.TEXT, Expression.convertFormatType("wrong"));
    }

}
