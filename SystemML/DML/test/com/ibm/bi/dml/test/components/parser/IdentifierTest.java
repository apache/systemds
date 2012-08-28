package com.ibm.bi.dml.test.components.parser;

import static org.junit.Assert.*;

import java.util.HashMap;
import java.io.IOException;

import org.junit.Test;

import com.ibm.bi.dml.parser.ConstIdentifier;
import com.ibm.bi.dml.parser.DataIdentifier;
import com.ibm.bi.dml.parser.Identifier;
import com.ibm.bi.dml.parser.ParseException;
import com.ibm.bi.dml.parser.Expression.DataType;
import com.ibm.bi.dml.parser.Expression.FormatType;
import com.ibm.bi.dml.parser.Expression.ValueType;
import com.ibm.bi.dml.utils.LanguageException;


public class IdentifierTest {

    @Test
    public void testSetProperties() {
        DataIdentifier sourceId = new DataIdentifier("sourceId");
        sourceId.setDimensions(10, 11);
        sourceId.setBlockDimensions(12, 13);
        sourceId.setNnz(14);
        sourceId.setDataType(DataType.MATRIX);
        sourceId.setValueType(ValueType.DOUBLE);
        sourceId.setFormatType(FormatType.BINARY);
        
        DataIdentifier idToTest = new DataIdentifier("idToTest");
        idToTest.setProperties(sourceId);
        
        checkIdentifierProperties(idToTest, 10, 11, 12, 13, 14, DataType.MATRIX, ValueType.DOUBLE, FormatType.BINARY);
    }

    @Test
    public void testSetNullProperties() {
        DataIdentifier sourceId = new DataIdentifier("sourceId");
        sourceId.setDimensions(10, 11);
        sourceId.setBlockDimensions(12, 13);
        sourceId.setNnz(14);
        sourceId.setDataType(DataType.MATRIX);
        sourceId.setValueType(ValueType.DOUBLE);
        sourceId.setFormatType(FormatType.BINARY);
        
        DataIdentifier idToTest = new DataIdentifier("idToTest");
        idToTest.setDimensions(15, 16);
        //idToTest.setNullProperties(sourceId);
        //checkIdentifierProperties(idToTest, 15, 16, 12, 13, 14, DataType.MATRIX, ValueType.DOUBLE, FormatType.BINARY);
        
        idToTest = new DataIdentifier("idToTest");
        idToTest.setBlockDimensions(15, 16);
        //idToTest.setNullProperties(sourceId);
        //checkIdentifierProperties(idToTest, 10, 11, 15, 16, 14, DataType.MATRIX, ValueType.DOUBLE, FormatType.BINARY);
        
        idToTest = new DataIdentifier("idToTest");
        idToTest.setNnz(15);
        //idToTest.setNullProperties(sourceId);
        //checkIdentifierProperties(idToTest, 10, 11, 12, 13, 15, DataType.MATRIX, ValueType.DOUBLE, FormatType.BINARY);
        
        idToTest = new DataIdentifier("idToTest");
        idToTest.setDataType(DataType.MATRIX);
        //idToTest.setNullProperties(sourceId);
        //checkIdentifierProperties(idToTest, 10, 11, 12, 13, 14, DataType.MATRIX, ValueType.DOUBLE, FormatType.BINARY);
        
        idToTest = new DataIdentifier("idToTest");
        idToTest.setValueType(ValueType.INT);
        //idToTest.setNullProperties(sourceId);
        //checkIdentifierProperties(idToTest, 10, 11, 12, 13, 14, DataType.MATRIX, ValueType.INT, FormatType.BINARY);
        
        idToTest = new DataIdentifier("idToTest");
        idToTest.setFormatType(FormatType.TEXT);
        //idToTest.setNullProperties(sourceId);
        //checkIdentifierProperties(idToTest, 10, 11, 12, 13, 14, DataType.MATRIX, ValueType.DOUBLE, FormatType.TEXT);
        
        idToTest = new DataIdentifier("idToTest");
        idToTest.setDimensions(15, 16);
        idToTest.setBlockDimensions(17, 18);
        idToTest.setNnz(19);
        idToTest.setDataType(DataType.MATRIX);
        idToTest.setValueType(ValueType.INT);
        idToTest.setFormatType(FormatType.TEXT);
        //idToTest.setNullProperties(sourceId);
        //checkIdentifierProperties(idToTest, 15, 16, 17, 18, 19, DataType.MATRIX, ValueType.INT, FormatType.TEXT);
    }
    
    @Test
    public void setBooleanProperties() {
        DataIdentifier idToTest = new DataIdentifier("idToTest");
        idToTest.setBooleanProperties();
        assertEquals(DataType.SCALAR, idToTest.getDataType());
        assertEquals(ValueType.BOOLEAN, idToTest.getValueType());
        assertEquals(0, idToTest.getDim1());
        assertEquals(0, idToTest.getDim2());
        assertEquals(0, idToTest.getRowsInBlock());
        assertEquals(0, idToTest.getColumnsInBlock());
        assertEquals(-1, idToTest.getNnz());
        assertNull(idToTest.getFormatType());
    }

    @Test
    public void testValidateExpression() throws LanguageException, IOException {
        DataIdentifier idToTest = new DataIdentifier("idToTest");
        HashMap<String, DataIdentifier> ids = new HashMap<String, DataIdentifier>();
        DataIdentifier originalId = new DataIdentifier("idToTest");
        originalId.setDimensions(100, 101);
        ids.put("idToTest", originalId);
        HashMap<String,ConstIdentifier> dummyConst = new HashMap<String,ConstIdentifier>();
        
        try {
			idToTest.validateExpression(ids, dummyConst);
		} catch (LanguageException e) {
			fail("unable to validate expression");
		}
        assertEquals(100, idToTest.getDim1());
        assertEquals(101, idToTest.getDim2());
        
        try {
            idToTest.validateExpression(new HashMap<String, DataIdentifier>(), dummyConst);
            fail("validation should have failed");
        } catch(NullPointerException e) {
        	
        } catch(LanguageException e) {
        	
        }
        
        try {
            idToTest.validateExpression(null, dummyConst);
            fail("validation should have failed");
        } catch(NullPointerException e) {
        	
        } catch(LanguageException e) {
        	
        }
    }
    
    @Test
    public void testComputeDataType() throws ParseException {
        DataIdentifier idToTest = new DataIdentifier("idToTest");
        
        idToTest.setDimensions(0, 0);
        idToTest.computeDataType();
        assertEquals(DataType.SCALAR, idToTest.getDataType());
       
        idToTest.setDimensions(1, 2);
        idToTest.computeDataType();
        assertEquals(DataType.MATRIX, idToTest.getDataType());
        
        idToTest.setDimensions(-1, -1);
        idToTest.computeDataType();
        assertEquals(DataType.UNKNOWN, idToTest.getDataType());
          
    }
    
    private void checkIdentifierProperties(Identifier idToTest, long rows, long cols, long rowsInBlock, long colsInBlock,
            long nnzs, DataType dataType, ValueType valueType, FormatType formatType) {
        assertEquals(rows, idToTest.getDim1());
        assertEquals(cols, idToTest.getDim2());
        assertEquals(rowsInBlock, idToTest.getRowsInBlock());
        assertEquals(colsInBlock, idToTest.getColumnsInBlock());
        assertEquals(nnzs, idToTest.getNnz());
        assertEquals(dataType, idToTest.getDataType());
        assertEquals(valueType, idToTest.getValueType());
        assertEquals(formatType, idToTest.getFormatType());
    }

}
