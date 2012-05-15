package com.ibm.bi.dml.test.components.parser;

import static org.junit.Assert.*;

import java.util.HashMap;

import org.junit.Test;

import com.ibm.bi.dml.parser.DataIdentifier;


public class DataIdentifierTest {

    @Test
    public void testVariablesRead() {
        DataIdentifier idToTest = new DataIdentifier("idToTest");
        HashMap<String, DataIdentifier> variablesRead = idToTest.variablesRead().getVariables();
        assertEquals(1, variablesRead.size());
        assertTrue(variablesRead.containsKey("idToTest"));
        assertEquals(idToTest, variablesRead.get("idToTest"));
    }

    @Test
    public void testVariablesUpdated() {
        DataIdentifier idToTest = new DataIdentifier("idToTest");
        assertNull(idToTest.variablesUpdated());
    }

}
