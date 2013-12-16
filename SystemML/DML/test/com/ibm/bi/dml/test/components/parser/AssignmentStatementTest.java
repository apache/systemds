/**
 * IBM Confidential
 * OCO Source Materials
 * (C) Copyright IBM Corp. 2010, 2013
 * The source code for this program is not published or otherwise divested of its trade secrets, irrespective of what has been deposited with the U.S. Copyright Office.
 */

package com.ibm.bi.dml.test.components.parser;

import static org.junit.Assert.*;

import java.util.HashMap;

import org.junit.Test;

import com.ibm.bi.dml.parser.AssignmentStatement;
import com.ibm.bi.dml.parser.DataIdentifier;
import com.ibm.bi.dml.parser.LanguageException;


public class AssignmentStatementTest 
{
	@SuppressWarnings("unused")
	private static final String _COPYRIGHT = "Licensed Materials - Property of IBM\n(C) Copyright IBM Corp. 2010, 2013\n" +
                                             "US Government Users Restricted Rights - Use, duplication  disclosure restricted by GSA ADP Schedule Contract with IBM Corp.";
	
    @Test
    public void testVariablesRead() throws LanguageException {
        DataIdentifier target = new DataIdentifier("target");
        DataIdentifier source = new DataIdentifier("source");
        AssignmentStatement as = new AssignmentStatement(target, source,0,0,0,0);
        HashMap<String, DataIdentifier> variables = as.variablesRead().getVariables();
        assertEquals(1, variables.size());
        assertTrue(variables.containsKey("source"));
        assertEquals(source, variables.get("source"));
    }

    @Test
    public void testVariablesUpdated() throws LanguageException {
        DataIdentifier target = new DataIdentifier("target");
        DataIdentifier source = new DataIdentifier("source");
        AssignmentStatement as = new AssignmentStatement(target, source,0,0,0,0);
        HashMap<String, DataIdentifier> variables = as.variablesUpdated().getVariables();
        assertEquals(1, variables.size());
        assertTrue(variables.containsKey("target"));
        assertEquals(target, variables.get("target"));
    }

}
