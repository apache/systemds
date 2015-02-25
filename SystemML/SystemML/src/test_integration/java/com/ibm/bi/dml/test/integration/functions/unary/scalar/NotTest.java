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
 * 	<li>not (!true == true, !true == false, !false == false, !false == true)</li>
 * </ul>
 * <p><b>Negative tests:</b></p>
 * 
 * 
 */
public class NotTest extends AutomatedTestBase 
{
	@SuppressWarnings("unused")
	private static final String _COPYRIGHT = "Licensed Materials - Property of IBM\n(C) Copyright IBM Corp. 2010, 2013\n" +
                                             "US Government Users Restricted Rights - Use, duplication  disclosure restricted by GSA ADP Schedule Contract with IBM Corp.";
	
	@Override
	public void setUp() {
		baseDirectory = SCRIPT_DIR + "functions/unary/scalar/";
		
		// positive tests
		availableTestConfigurations.put("NotTest", new TestConfiguration("NotTest",
				new String[] { "true_true", "true_false", "false_false", "false_true" }));
		
		// negative tests
	}
	
	@Test
	public void testNot() {
		loadTestConfiguration("NotTest");
		
		createHelperMatrix();
		writeExpectedHelperMatrix("true_true", 1);
		writeExpectedHelperMatrix("true_false", 2);
		writeExpectedHelperMatrix("false_false", 1);
		writeExpectedHelperMatrix("false_true", 2);
		
		runTest();
		
		compareResults();
	}
	
}
