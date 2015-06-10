/**
 * IBM Confidential
 * OCO Source Materials
 * (C) Copyright IBM Corp. 2010, 2015
 * The source code for this program is not published or otherwise divested of its trade secrets, irrespective of what has been deposited with the U.S. Copyright Office.
 */

package com.ibm.bi.dml.test.integration.functions.external;

import org.junit.Test;

import com.ibm.bi.dml.test.integration.AutomatedTestBase;
import com.ibm.bi.dml.test.integration.TestConfiguration;

/**
 * JUnit test for SGD based MF
 */

public class SGDMFTest extends AutomatedTestBase 
{
	@SuppressWarnings("unused")
	private static final String _COPYRIGHT = "Licensed Materials - Property of IBM\n(C) Copyright IBM Corp. 2010, 2015\n" +
                                             "US Government Users Restricted Rights - Use, duplication  disclosure restricted by GSA ADP Schedule Contract with IBM Corp.";
	
	private static final String TEST_DIR = "functions/external/";
	
	@Override
	public void setUp() {
		addTestConfiguration("SGDMFTest", new TestConfiguration(TEST_DIR, "SGDMFTest", new String[] { "W", "tH" }));
	}

	@Test
	public void testSGDMFTest() {
		
		int rows = 100;
		int cols = 50;
		
		TestConfiguration config = availableTestConfigurations.get("SGDMFTest");
		config.addVariable("rows", rows);
		config.addVariable("cols", cols);
		loadTestConfiguration(config);

		double[][] V = getRandomMatrix(rows, cols, 1, 5, 0.05, -1);
		
		writeInputMatrix("V", V);

		runTest();
		checkForResultExistence();
	}
}
