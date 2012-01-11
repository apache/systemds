package dml.test.components.parser;

import static org.junit.Assert.*;

import java.util.HashMap;

import org.junit.Test;

import dml.parser.ConstIdentifier;
import dml.parser.DataIdentifier;
import dml.parser.DoubleIdentifier;
import dml.parser.IntIdentifier;
import dml.parser.ParseException;
import dml.parser.RandStatement;
import dml.parser.Expression.DataType;
import dml.parser.Expression.FormatType;
import dml.parser.Expression.ValueType;
import dml.test.utils.TestUtils;

public class RandStatementTest {
    
    @Test
    public void testVariablesUpdated() {
        String idName = "A";
        DataIdentifier id = new DataIdentifier(idName);
        RandStatement rs = new RandStatement(id);
        HashMap<String, DataIdentifier> variables = rs.variablesUpdated().getVariables();
        assertEquals(1, variables.size());
        assertTrue(variables.containsKey(idName));
        assertEquals(id, variables.get(idName));
    }

    @Test
    public void testAddLongParam() {
        RandStatement rs = getRandStatementInstance();
        try {
            rs.addLongParam("rows", 1);
            assertEquals(1, rs.getRows());
            rs.addLongParam("cols", 2);
            assertEquals(2, rs.getCols());
            rs.addLongParam("min", 5);
            assertEquals(5, rs.getMinValue(), 0);
            rs.addLongParam("max", 6);
            assertEquals(6, rs.getMaxValue(), 0);
        } catch(ParseException e) {
            fail("Unable to add parameter");
        }
        try {
            rs.addLongParam("wrong", 0);
            fail("Possible to add wrong parameter");
        } catch(ParseException e) { }
    }

    @Test
    public void testAddDoubleParam() {
        RandStatement rs = getRandStatementInstance();
        try {
            rs.addDoubleParam("min", 0.1);
            assertEquals(0.1, rs.getMinValue(), 0);
            rs.addDoubleParam("max", 0.2);
            assertEquals(0.2, rs.getMaxValue(), 0);
            rs.addDoubleParam("sparsity", 0.3);
            assertEquals(0.3, rs.getSparsity(), 0);
        } catch(ParseException e) {
            fail("Unable to add parameter");
        }
        try {
            rs.addDoubleParam("wrong", 0.0);
            fail("Possible to add wrong parameter");
        } catch(ParseException e) { }
    }

    @Test
    public void testAddStringParam() {
        RandStatement rs = getRandStatementInstance();
        try {
            rs.addStringParam("rows", "1");
            assertEquals(1, rs.getRows());
            rs.addStringParam("cols", "2");
            assertEquals(2, rs.getCols());
            rs.addStringParam("min", "5");
            assertEquals(5, rs.getMinValue(), 0);
            rs.addStringParam("max", "6");
            assertEquals(6, rs.getMaxValue(), 0);
            rs.addStringParam("sparsity", "0.7");
            assertEquals(0.7, rs.getSparsity(), 0);
            rs.addStringParam("pdf", "uniform");
            assertEquals("uniform", rs.getProbabilityDensityFunction());
        } catch(ParseException e) {
            fail("Unable to add parameter");
        }
        try {
            rs.addStringParam("wrong", "");
            fail("Possible to add wrong parameter");
        } catch(ParseException e) { }
    }
    
    @Test
    public void testAddVarParam() {
        RandStatement rs = getRandStatementInstance();
        rs.addVarParam("rows", new DataIdentifier("rowsID"));
        String[] requiredVariables = rs.getRequiredVariables();
        TestUtils.assertInterchangedArraysEquals(new String[] { "rowsID" }, requiredVariables);
        
        rs = getRandStatementInstance();
        rs.addVarParam("cols", new DataIdentifier("colsID"));
        rs.addVarParam("sparsity", new DataIdentifier("sparsityID"));
        requiredVariables = rs.getRequiredVariables();
        TestUtils.assertInterchangedArraysEquals(new String[] { "colsID", "sparsityID" }, requiredVariables);
        
        rs = getRandStatementInstance();
        rs.addVarParam("min", new DataIdentifier("minMaxID"));
        rs.addVarParam("max", new DataIdentifier("minMaxID"));
        requiredVariables = rs.getRequiredVariables();
        TestUtils.assertInterchangedArraysEquals(new String[] { "minMaxID" }, requiredVariables);
    }

    @Test
    public void testValidateFunctionCall() {
        RandStatement rs = getRandStatementInstance();
        try {
            rs.validateFunctionCall();
            rs.addLongParam("rows", 10);
            rs.validateFunctionCall();
            rs.addLongParam("cols", 10);
            rs.validateFunctionCall();
            rs.addLongParam("rows", 1);
            rs.validateFunctionCall();
            rs.addDoubleParam("sparsity", 0.5);
            rs.validateFunctionCall();
        } catch(ParseException e) {
            fail("Validation failed");
        }
        rs = getRandStatementInstance();
        try {
            rs.addLongParam("rows", 0);
            rs.validateFunctionCall();
            fail("Validation successful although rows=0");
        } catch(ParseException e) { }
        rs = getRandStatementInstance();
        try {
            rs.addLongParam("cols", 0);
            rs.validateFunctionCall();
            fail("Validation successful although cols=0");
        } catch(ParseException e) { }
        rs = getRandStatementInstance();
        try {
            rs.addLongParam("rows", 0);
            rs.addLongParam("cols", 0);
            rs.validateFunctionCall();
            fail("Validation successful although rows=0 and cols=0");
        } catch(ParseException e) { }
        try {
            rs.addDoubleParam("sparsity", -0.1);
            rs.validateFunctionCall();
            fail("Validation successful although sparsity<0");
        } catch(ParseException e) { }
        try {
            rs.addDoubleParam("sparsity", 1.1);
            rs.validateFunctionCall();
            fail("Validation successful although sparsity>1");
        } catch(ParseException e) { }
    }

    @Test
    public void testSetIdentifierProperties() throws ParseException {
        RandStatement rs = getRandStatementInstance();
        rs.setIdentifierProperties();
        DataIdentifier id = rs.getIdentifier();
        assertEquals(FormatType.BINARY, id.getFormatType());
        assertEquals(ValueType.DOUBLE, id.getValueType());
        assertEquals(DataType.SCALAR, id.getDataType());
        assertEquals(1, id.getDim1());
        assertEquals(1, id.getDim2());
        
        rs = getRandStatementInstance();
        try {
            rs.addLongParam("rows", 10);
        } catch(ParseException e) { }
        rs.setIdentifierProperties();
        id = rs.getIdentifier();
        assertEquals(FormatType.BINARY, id.getFormatType());
        assertEquals(ValueType.DOUBLE, id.getValueType());
        assertEquals(DataType.MATRIX, id.getDataType());
        assertEquals(10, id.getDim1());
        assertEquals(1, id.getDim2());
        
        rs = getRandStatementInstance();
        try {
            rs.addLongParam("rows", 10);
            rs.addLongParam("cols", 10);
        } catch(ParseException e) { }
        rs.setIdentifierProperties();
        id = rs.getIdentifier();
        assertEquals(FormatType.BINARY, id.getFormatType());
        assertEquals(ValueType.DOUBLE, id.getValueType());
        assertEquals(DataType.MATRIX, id.getDataType());
        assertEquals(10, id.getDim1());
        assertEquals(10, id.getDim2());
    }
    
    @Test
    public void testGetRequiredVariables() {
        RandStatement rs = getRandStatementInstance();
        rs.addVarParam("rows", new DataIdentifier("rowsID"));
        String[] requiredVariables = rs.getRequiredVariables();
        TestUtils.assertInterchangedArraysEquals(new String[] { "rowsID" }, requiredVariables);
        
        rs = getRandStatementInstance();
        rs.addVarParam("cols", new DataIdentifier("colsID"));
        rs.addVarParam("sparsity", new DataIdentifier("sparsityID"));
        requiredVariables = rs.getRequiredVariables();
        TestUtils.assertInterchangedArraysEquals(new String[] { "colsID", "sparsityID" }, requiredVariables);
        
        rs = getRandStatementInstance();
        rs.addVarParam("min", new DataIdentifier("minMaxID"));
        rs.addVarParam("max", new DataIdentifier("minMaxID"));
        requiredVariables = rs.getRequiredVariables();
        TestUtils.assertInterchangedArraysEquals(new String[] { "minMaxID" }, requiredVariables);
    }
    
    @Test
    public void testUpdateVariables() {
        try {
            RandStatement rs = getRandStatementInstance();
            rs.addVarParam("rows", new DataIdentifier("rowsID"));
            HashMap<String, ConstIdentifier> variables = new HashMap<String, ConstIdentifier>();
            variables.put("rowsID", new IntIdentifier(100));
            rs.updateVariables(variables);
            assertEquals(100, rs.getRows());
            assertEquals(100, rs.getIdentifier().getDim1());
            
            rs = getRandStatementInstance();
            rs.addVarParam("cols", new DataIdentifier("colsID"));
            rs.addVarParam("sparsity", new DataIdentifier("sparsityID"));
            variables = new HashMap<String, ConstIdentifier>();
            variables.put("colsID", new IntIdentifier(100));
            variables.put("sparsityID", new DoubleIdentifier(0.5));
            rs.updateVariables(variables);
            assertEquals(100, rs.getCols());
            assertEquals(100, rs.getIdentifier().getDim2());
            assertEquals(0.5, rs.getSparsity(), 0);
            
            rs = getRandStatementInstance();
            rs.addVarParam("min", new DataIdentifier("minMaxID"));
            rs.addVarParam("max", new DataIdentifier("minMaxID"));
            variables = new HashMap<String, ConstIdentifier>();
            variables.put("minMaxID", new DoubleIdentifier(2.5));
            rs.updateVariables(variables);
            assertEquals(2.5, rs.getMinValue(), 0);
            assertEquals(2.5, rs.getMaxValue(), 0);
        } catch(ParseException e) {
            fail("failed to update variables");
        }
    }
    
    private RandStatement getRandStatementInstance()
    {
        String idName = "A";
        DataIdentifier id = new DataIdentifier(idName);
        return new RandStatement(id);
    }
    
}
