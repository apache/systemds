package com.ibm.bi.dml.test.components.parser;

import static org.junit.Assert.*;

import java.util.HashMap;
import java.util.Set;

import org.junit.Test;

import com.ibm.bi.dml.parser.DataIdentifier;
import com.ibm.bi.dml.parser.VariableSet;


public class VariableSetTest {

    @Test
    public void testAddVariable() {
        VariableSet vsToTest = new VariableSet();
        DataIdentifier var = new DataIdentifier("intern");
        vsToTest.addVariable("extern", var);
        HashMap<String, DataIdentifier> variables = vsToTest.getVariables();
        assertEquals(1, variables.size());
        assertTrue("var missing", variables.containsKey("extern"));
        assertEquals(var, variables.get("extern"));
    }

    @Test
    public void testAddVariables() {
        VariableSet vsToTest = new VariableSet();
        DataIdentifier var1 =  new DataIdentifier("var1");
        vsToTest.addVariable("var1", var1);
        DataIdentifier var2 =  new DataIdentifier("var2");
        vsToTest.addVariable("var2", var2);
        VariableSet vsToAdd = new VariableSet();
        DataIdentifier var3 =  new DataIdentifier("var3");
        vsToAdd.addVariable("var3", var3);
        DataIdentifier var4 =  new DataIdentifier("var4");
        vsToAdd.addVariable("var4", var4);
        vsToTest.addVariables(vsToAdd);
        HashMap<String, DataIdentifier> variables = vsToTest.getVariables();
        assertEquals(4, variables.size());
        assertTrue("var 1 is missing", variables.containsKey("var1"));
        assertTrue("var 2 is missing", variables.containsKey("var2"));
        assertTrue("var 3 is missing", variables.containsKey("var3"));
        assertTrue("var 4 is missing", variables.containsKey("var4"));
        assertEquals(var1, variables.get("var1"));
        assertEquals(var2, variables.get("var2"));
        assertEquals(var3, variables.get("var3"));
        assertEquals(var4, variables.get("var4"));
        
        vsToTest = new VariableSet();
        vsToTest.addVariable("var1", var1);
        vsToAdd = new VariableSet();
        vsToAdd.addVariable("var1", var2);
        vsToTest.addVariables(vsToAdd);
        variables = vsToTest.getVariables();
        assertEquals(1, variables.size());
        assertTrue("var is missing", variables.containsKey("var1"));
        assertEquals(var2, variables.get("var1"));
    }

    @Test
    public void testRemoveVariables() {
        VariableSet vsToTest = new VariableSet();
        vsToTest.addVariable("var 1", new DataIdentifier("var 1"));
        vsToTest.addVariable("var 2", new DataIdentifier("var 2"));
        vsToTest.addVariable("var 3", new DataIdentifier("var 3"));
        vsToTest.addVariable("var 4", new DataIdentifier("var 4"));
        
        VariableSet vsToRemove = new VariableSet();
        vsToRemove.addVariable("var 5", new DataIdentifier("var 4"));
        vsToRemove.addVariable("var 2", new DataIdentifier("var 6"));
        vsToRemove.addVariable("var 3", new DataIdentifier("var 1"));
        
        vsToTest.removeVariables(vsToRemove);
        HashMap<String, DataIdentifier> variables = vsToTest.getVariables();
        
        assertEquals(2, variables.size());
        assertTrue("var 1 missing", variables.containsKey("var 1"));
        assertTrue("var 4 missing", variables.containsKey("var 4"));
    }

    @Test
    public void testContainsVariable() {
        VariableSet vsToTest = new VariableSet();
        vsToTest.addVariable("var", new DataIdentifier("var"));
        
        assertTrue("var missing", vsToTest.containsVariable("var"));
        assertFalse("wrongVar shouldn't be in there", vsToTest.containsVariable("wrongVar"));
    }

    @Test
    public void testGetVariable() {
        VariableSet vsToTest = new VariableSet();
        DataIdentifier var = new DataIdentifier("var");
        vsToTest.addVariable("var", var);
        
        assertEquals(var, vsToTest.getVariable("var"));
        assertNull("wrongVar shouldn't be in there", vsToTest.getVariable("wrongVar"));
    }

    @Test
    public void testGetVariableNames() {
        VariableSet vsToTest = new VariableSet();
        vsToTest.addVariable("var 1", new DataIdentifier("var 1"));
        vsToTest.addVariable("var 2", new DataIdentifier("var 2"));
        vsToTest.addVariable("var 3", new DataIdentifier("var 3"));
        
        Set<String> variableNames = vsToTest.getVariableNames();
        assertEquals(3, variableNames.size());
        assertTrue("var 1 missing", variableNames.contains("var 1"));
        assertTrue("var 2 missing", variableNames.contains("var 1"));
        assertTrue("var 3 missing", variableNames.contains("var 1"));
    }

}
