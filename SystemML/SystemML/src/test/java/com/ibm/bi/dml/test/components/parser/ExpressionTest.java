/**
 * IBM Confidential
 * OCO Source Materials
 * (C) Copyright IBM Corp. 2010, 2013
 * The source code for this program is not published or otherwise divested of its trade secrets, irrespective of what has been deposited with the U.S. Copyright Office.
 */

package com.ibm.bi.dml.test.components.parser;

import static org.junit.Assert.*;

import org.junit.Test;

import com.ibm.bi.dml.hops.Hop.FileFormatTypes;
import com.ibm.bi.dml.parser.DataExpression;
import com.ibm.bi.dml.parser.Expression;
import com.ibm.bi.dml.parser.Statement;
import com.ibm.bi.dml.parser.Expression.BinaryOp;
import com.ibm.bi.dml.parser.Expression.BooleanOp;
import com.ibm.bi.dml.parser.Expression.RelationalOp;


public class ExpressionTest
{
	@SuppressWarnings("unused")
	private static final String _COPYRIGHT = "Licensed Materials - Property of IBM\n(C) Copyright IBM Corp. 2010, 2013\n" +
                                             "US Government Users Restricted Rights - Use, duplication  disclosure restricted by GSA ADP Schedule Contract with IBM Corp.";
	    
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
        assertEquals(FileFormatTypes.TEXT, Expression.convertFormatType(DataExpression.FORMAT_TYPE_VALUE_TEXT));
        assertEquals(FileFormatTypes.BINARY, Expression.convertFormatType(DataExpression.FORMAT_TYPE_VALUE_BINARY));
        assertEquals(FileFormatTypes.TEXT, Expression.convertFormatType("wrong"));
    }

}
