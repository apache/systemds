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

import com.ibm.bi.dml.parser.ConditionalPredicate;
import com.ibm.bi.dml.parser.DataIdentifier;


public class ConditionalPredicateTest 
{
	@SuppressWarnings("unused")
	private static final String _COPYRIGHT = "Licensed Materials - Property of IBM\n(C) Copyright IBM Corp. 2010, 2013\n" +
                                             "US Government Users Restricted Rights - Use, duplication  disclosure restricted by GSA ADP Schedule Contract with IBM Corp.";
	
    @Test
    public void testVariablesRead() {
        DataIdentifier var = new DataIdentifier("var");
        ConditionalPredicate cpToTest = new ConditionalPredicate(var);
        HashMap<String, DataIdentifier> variablesRead = cpToTest.variablesRead().getVariables();
        
        assertEquals(1, variablesRead.size());
        assertTrue("var is missing", variablesRead.containsKey("var"));
        assertEquals(var, variablesRead.get("var"));
    }

}
