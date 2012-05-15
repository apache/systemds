package com.ibm.bi.dml.test.components.parser;

import static org.junit.Assert.*;

import java.util.HashMap;

import org.junit.Test;

import com.ibm.bi.dml.parser.AssignmentStatement;
import com.ibm.bi.dml.parser.DataIdentifier;


public class AssignmentStatementTest {

    @Test
    public void testVariablesRead() {
        DataIdentifier target = new DataIdentifier("target");
        DataIdentifier source = new DataIdentifier("source");
        AssignmentStatement as = new AssignmentStatement(target, source);
        HashMap<String, DataIdentifier> variables = as.variablesRead().getVariables();
        assertEquals(1, variables.size());
        assertTrue(variables.containsKey("source"));
        assertEquals(source, variables.get("source"));
    }

    @Test
    public void testVariablesUpdated() {
        DataIdentifier target = new DataIdentifier("target");
        DataIdentifier source = new DataIdentifier("source");
        AssignmentStatement as = new AssignmentStatement(target, source);
        HashMap<String, DataIdentifier> variables = as.variablesUpdated().getVariables();
        assertEquals(1, variables.size());
        assertTrue(variables.containsKey("target"));
        assertEquals(target, variables.get("target"));
    }

}
