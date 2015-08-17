/**
 * IBM Confidential
 * OCO Source Materials
 * (C) Copyright IBM Corp. 2010, 2013
 * The source code for this program is not published or otherwise divested of its trade secrets, irrespective of what has been deposited with the U.S. Copyright Office.
 */

package com.ibm.bi.dml.test.integration.functions.unary.scalar;

import org.junit.Test;

import com.ibm.bi.dml.test.integration.AutomatedTestBase;
import com.ibm.bi.dml.test.integration.TestConfiguration;



public class BooleanTest extends AutomatedTestBase 
{
	@SuppressWarnings("unused")
	private static final String _COPYRIGHT = "Licensed Materials - Property of IBM\n(C) Copyright IBM Corp. 2010, 2013\n" +
                                             "US Government Users Restricted Rights - Use, duplication  disclosure restricted by GSA ADP Schedule Contract with IBM Corp.";
	
	private static final String TEST_DIR = "functions/unary/scalar/";
	
	
	@Override
	public void setUp() {
		
		// positive tests
		addTestConfiguration("WhileTest", new TestConfiguration(TEST_DIR, "BooleanWhileTest",
				new String[] { "true", "false" }));
		
		// negative tests
	}
	
	@Test
	public void testWhile() {
		TestConfiguration config = getTestConfiguration("WhileTest");
		loadTestConfiguration(config);
		
		createHelperMatrix();
		
		writeExpectedHelperMatrix("true", 2);
		writeExpectedHelperMatrix("false", 1);
		
		runTest();
		
		compareResults();
	}

}
