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



/**
 * <p><b>Positive tests:</b></p>
 * <ul>
 * 	<li>negation (int, negative int, double, negative double)</li>
 * </ul>
 * <p><b>Negative tests:</b></p>
 * 
 * 
 */
public class NegationTest extends AutomatedTestBase 
{
	@SuppressWarnings("unused")
	private static final String _COPYRIGHT = "Licensed Materials - Property of IBM\n(C) Copyright IBM Corp. 2010, 2013\n" +
                                             "US Government Users Restricted Rights - Use, duplication  disclosure restricted by GSA ADP Schedule Contract with IBM Corp.";
	
	@Override
	public void setUp() {
		baseDirectory = SCRIPT_DIR + "functions/unary/scalar/";
		
		// positive tests
		availableTestConfigurations.put("NegationTest", new TestConfiguration("NegationTest",
				new String[] { }));
		
		// negative tests
	}
	
	@Test
	public void testNegation() {
		int intValue = 2;
		int negativeIntValue = -2;
		double doubleValue = 2;
		double negativeDoubleValue = -2;
		
		TestConfiguration config = availableTestConfigurations.get("NegationTest");
		config.addVariable("int", intValue);
		config.addVariable("negativeint", negativeIntValue);
		config.addVariable("double", doubleValue);
		config.addVariable("negativedouble", negativeDoubleValue);
		
		loadTestConfiguration("NegationTest");
		
		createHelperMatrix();
		writeExpectedHelperMatrix("int", -intValue);
		writeExpectedHelperMatrix("negative_int", -negativeIntValue);
		writeExpectedHelperMatrix("double", -doubleValue);
		writeExpectedHelperMatrix("negative_double", -negativeDoubleValue);
		
		runTest();
		
		compareResults();
	}
	
}
