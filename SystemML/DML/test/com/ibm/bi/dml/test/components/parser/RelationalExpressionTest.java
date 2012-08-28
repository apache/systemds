package com.ibm.bi.dml.test.components.parser;

import static org.junit.Assert.*;

import java.util.HashMap;
import java.io.IOException;

import org.junit.Test;

import com.ibm.bi.dml.parser.ConstIdentifier;
import com.ibm.bi.dml.parser.DataIdentifier;
import com.ibm.bi.dml.parser.RelationalExpression;
import com.ibm.bi.dml.parser.Expression.RelationalOp;
import com.ibm.bi.dml.parser.Expression.ValueType;
import com.ibm.bi.dml.utils.LanguageException;


public class RelationalExpressionTest {

    @Test
    public void testValidateExpression() throws LanguageException, IOException {
        HashMap<String, DataIdentifier> ids = new HashMap<String, DataIdentifier>();
        DataIdentifier left = new DataIdentifier("left");
        left.setDimensions(100, 101);
        DataIdentifier right = new DataIdentifier("right");
        right.setDimensions(102, 103);
        ids.put("left", left);
        ids.put("right", right);
        
        RelationalExpression beToTest = new RelationalExpression(RelationalOp.EQUAL);
        beToTest.setLeft(new DataIdentifier("left"));
        beToTest.setRight(new DataIdentifier("right"));
        beToTest.validateExpression(ids, new HashMap<String,ConstIdentifier>());
        assertEquals(ValueType.BOOLEAN, beToTest.getOutput().getValueType());
        
        ids = new HashMap<String, DataIdentifier>();
        ids.put("right", right);
        try {
            beToTest.validateExpression(ids, new HashMap<String,ConstIdentifier>());
            fail("left expression not validated");
        } catch(Exception e) { }
        
        ids = new HashMap<String, DataIdentifier>();
        ids.put("left", left);
        try {
        // TODO: investigate
        //    beToTest.validateExpression(ids);
        //    fail("right expression not validated");
        } catch(RuntimeException e) { }
    }
    
    @Test
    public void testVariablesRead() {
        RelationalExpression beToTest = new RelationalExpression(RelationalOp.EQUAL);
        DataIdentifier left = new DataIdentifier("left");
        DataIdentifier right = new DataIdentifier("right");
        beToTest.setLeft(left);
        beToTest.setRight(right);
        HashMap<String, DataIdentifier> variablesRead = beToTest.variablesRead().getVariables();
        assertEquals(2, variablesRead.size());
        assertTrue("no left variable", variablesRead.containsKey("left"));
        assertTrue("no right variable", variablesRead.containsKey("right"));
        assertEquals(left, variablesRead.get("left"));
        assertEquals(right, variablesRead.get("right"));
    }

}
