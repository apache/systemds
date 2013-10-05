/**
 * IBM Confidential
 * OCO Source Materials
 * (C) Copyright IBM Corp. 2010, 2013
 * The source code for this program is not published or otherwise divested of its trade secrets, irrespective of what has been deposited with the U.S. Copyright Office.
 */

package com.ibm.bi.dml.test.integration.functions.terms;

import org.junit.Test;

import com.ibm.bi.dml.test.integration.AutomatedTestBase;
import com.ibm.bi.dml.test.integration.TestConfiguration;


public class ScalarToMatrixInLoopTest extends AutomatedTestBase 
{
	@SuppressWarnings("unused")
	private static final String _COPYRIGHT = "Licensed Materials - Property of IBM\n(C) Copyright IBM Corp. 2010, 2013\n" +
                                             "US Government Users Restricted Rights - Use, duplication  disclosure restricted by GSA ADP Schedule Contract with IBM Corp.";
	
	@SuppressWarnings("deprecation")
	@Override
	public void setUp() {
		baseDirectory = SCRIPT_DIR + "functions/terms/";

		availableTestConfigurations.put("ScalarToMatrixInLoop", new TestConfiguration("TestScalarToMatrixInLoop", new String[] {}));
	}

	@Test
	public void testScalarToMatrixInLoop() {
		int rows = 5, cols = 5;

		TestConfiguration config = getTestConfiguration("ScalarToMatrixInLoop");
		config.addVariable("rows", rows);
		config.addVariable("cols", cols);

		loadTestConfiguration(config);

		runTest();
	}
}
