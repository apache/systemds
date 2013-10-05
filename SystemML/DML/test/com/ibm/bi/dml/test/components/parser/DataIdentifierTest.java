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

import com.ibm.bi.dml.parser.DataIdentifier;


public class DataIdentifierTest 
{
	@SuppressWarnings("unused")
	private static final String _COPYRIGHT = "Licensed Materials - Property of IBM\n(C) Copyright IBM Corp. 2010, 2013\n" +
                                             "US Government Users Restricted Rights - Use, duplication  disclosure restricted by GSA ADP Schedule Contract with IBM Corp.";
	
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
