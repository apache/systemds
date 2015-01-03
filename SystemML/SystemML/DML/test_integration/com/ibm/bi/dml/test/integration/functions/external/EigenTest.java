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
import com.ibm.bi.dml.test.utils.TestUtils;

public class EigenTest extends AutomatedTestBase 
{
	@SuppressWarnings("unused")
	private static final String _COPYRIGHT = "Licensed Materials - Property of IBM\n(C) Copyright IBM Corp. 2010, 2015\n" +
                                             "US Government Users Restricted Rights - Use, duplication  disclosure restricted by GSA ADP Schedule Contract with IBM Corp.";
	
	private static String TEST_DIR = "functions/external/";
	
	@Override
	public void setUp() {
		TestUtils.clearAssertionInformation();
		addTestConfiguration("EigenTest", new TestConfiguration(TEST_DIR, "EigenTest", new String[] { "val", "vec" }));
	}

	@Test
	public void testEigen() {
		
		int rows = 3;
		int cols = rows;

		TestConfiguration config = availableTestConfigurations.get("EigenTest");
		config.addVariable("rows", rows);
		config.addVariable("cols", cols);

		double[][] A = {{0, 1, -1},{1, 1, 0},{-1, 0, 1}};
		
		
		writeInputMatrix("A", A);
		
		loadTestConfiguration(config);

		runTest();
		
		double [][] val = {{-1}, {1}, {2}};
		double [][] vec = {
				{0.81649, 0, -0.57735},
				{-0.4082, 0.7071, -0.57735},
				{0.4082, 0.7071, 0.57735}
				};

		writeExpectedMatrix("val", val);
		writeExpectedMatrix("vec", vec);
		compareResults(0.001);
		
	}
}
