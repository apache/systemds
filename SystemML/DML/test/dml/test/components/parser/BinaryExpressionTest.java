package dml.test.components.parser;

import static org.junit.Assert.*;

import java.util.HashMap;

import org.junit.Test;

import dml.parser.BinaryExpression;
import dml.parser.DataIdentifier;
import dml.parser.Expression.BinaryOp;
import dml.utils.LanguageException;

public class BinaryExpressionTest {

    @Test
    public void testValidateExpression() throws LanguageException {
        HashMap<String, DataIdentifier> ids = new HashMap<String, DataIdentifier>();
        DataIdentifier left = new DataIdentifier("left");
        left.setDimensions(100, 101);
        DataIdentifier right = new DataIdentifier("right");
        right.setDimensions(102, 103);
        ids.put("left", left);
        ids.put("right", right);
        
        BinaryExpression beToTest = new BinaryExpression(BinaryOp.PLUS);
        beToTest.setLeft(new DataIdentifier("left"));
        beToTest.setRight(new DataIdentifier("right"));
        beToTest.validateExpression(ids);
        assertEquals(-1, beToTest.getOutput().getDim1());
        assertEquals(-1, beToTest.getOutput().getDim2());
        
        beToTest = new BinaryExpression(BinaryOp.MATMULT);
        beToTest.setLeft(new DataIdentifier("left"));
        beToTest.setLeft(new DataIdentifier("left"));
        beToTest.setRight(new DataIdentifier("right"));
        try {
            beToTest.validateExpression(ids);
            fail("dimensions do not match for matrix multiplication");
        } catch(Exception e) { }
        
        right.setDimensions(101, 102);
        beToTest = new BinaryExpression(BinaryOp.MATMULT);
        beToTest.setLeft(new DataIdentifier("left"));
        beToTest.setLeft(new DataIdentifier("left"));
        beToTest.setRight(new DataIdentifier("right"));
        beToTest.validateExpression(ids);
        assertEquals(100, beToTest.getOutput().getDim1());
        assertEquals(102, beToTest.getOutput().getDim2());
        
        ids = new HashMap<String, DataIdentifier>();
        ids.put("right", right);
        try {
            beToTest.validateExpression(ids);
            fail("left expression not validated");
        } catch(Exception e) { }
        
        ids = new HashMap<String, DataIdentifier>();
        ids.put("left", left);
        try {
            beToTest.validateExpression(ids);
            fail("right expression not validated");
        } catch(Exception e) { }
    }

    @Test
    public void testVariablesRead() {
        BinaryExpression beToTest = new BinaryExpression(BinaryOp.PLUS);
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
