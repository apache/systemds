package dml.test.components.parser;

import static org.junit.Assert.*;

import java.io.IOException;
import java.util.HashMap;

import org.junit.Test;

import dml.parser.DataIdentifier;
import dml.parser.InputStatement;
import dml.parser.Expression.ValueType;
import dml.utils.LanguageException;

public class InputStatementTest {

    @Test
    public void testVariablesRead() {
        DataIdentifier target = new DataIdentifier("target");
        InputStatement isToTest = new InputStatement(target, "filename");
        
        HashMap<String, DataIdentifier> variablesRead = isToTest.variablesRead().getVariables();
        
        assertEquals(0, variablesRead.size());
    }

    @Test
    public void testVariablesUpdated() {
        DataIdentifier target = new DataIdentifier("target");
        InputStatement isToTest = new InputStatement(target, "filename");
        
        HashMap<String, DataIdentifier> variablesUpdated = isToTest.variablesUpdated().getVariables();
        
        assertEquals(1, variablesUpdated.size());
        assertTrue("target is missing", variablesUpdated.containsKey("target"));
        assertEquals(target, variablesUpdated.get("target"));
    }

    @Test
    public void testProcessParams() throws LanguageException, IOException {
        DataIdentifier target = new DataIdentifier("target");
        InputStatement isToTest = new InputStatement(target, "filename");
        
        isToTest.addStringParam("rows", "100");
        isToTest.addStringParam("cols", "101");
        
        
        isToTest.addStringParam("rows_in_block", "102");
        isToTest.addStringParam("columns_in_block", "103");
        isToTest.addStringParam("value_type", "double");
        
        isToTest.processParams(false);
        assertEquals(100, isToTest.getIdentifier().getDim1());
        assertEquals(101, isToTest.getIdentifier().getDim2());
        assertEquals(-1, isToTest.getIdentifier().getRowsInBlock());
        assertEquals(-1, isToTest.getIdentifier().getColumnsInBlock());
        assertEquals(ValueType.DOUBLE, isToTest.getIdentifier().getValueType());
        
        isToTest.addStringParam("value_type", "string");
        isToTest.processParams(false);
        assertEquals(ValueType.STRING, isToTest.getIdentifier().getValueType());
        
        isToTest.addStringParam("value_type", "int");
        isToTest.processParams(false);
        assertEquals(ValueType.INT, isToTest.getIdentifier().getValueType());
        
        isToTest.addStringParam("value_type", "other");
        try {
        	isToTest.processParams(false);
        	assertEquals(ValueType.UNKNOWN, isToTest.getIdentifier().getValueType());
        } catch (LanguageException e) {}
        	

        target = new DataIdentifier("target");
        isToTest = new InputStatement(target, "filename");
        isToTest.addStringParam("rows", "104");
        isToTest.addStringParam("cols", "105");
        isToTest.processParams(false);
        assertEquals(104, isToTest.getIdentifier().getDim1());
        assertEquals(105, isToTest.getIdentifier().getDim2());

        target = new DataIdentifier("target");
        isToTest = new InputStatement(target, "filename");
        isToTest.addStringParam("cols", "105");
        try {
        	isToTest.processParams(false);
        } catch (LanguageException e){}
        assertEquals(-1, isToTest.getIdentifier().getDim1());
        assertEquals(-1, isToTest.getIdentifier().getDim2());

    }

}
