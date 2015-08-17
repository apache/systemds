/**
 * IBM Confidential
 * OCO Source Materials
 * (C) Copyright IBM Corp. 2010, 2013
 * The source code for this program is not published or otherwise divested of its trade secrets, irrespective of what has been deposited with the U.S. Copyright Office.
 */

package com.ibm.bi.dml.test.integration.functions.data;

import org.junit.Test;

import com.ibm.bi.dml.test.integration.AutomatedTestBase;
import com.ibm.bi.dml.test.integration.TestConfiguration;


/**
 * <p>
 * <b>Positive tests:</b>
 * </p>
 * <ul>
 * <li>random matrix generation (rows, cols, min, max)</li>
 * <li>random scalar generation (min, max)</li>
 * </ul>
 * <p>
 * <b>Negative tests:</b>
 * </p>
 * 
 * 
 */
public class RandTest3 extends AutomatedTestBase 
{
	@SuppressWarnings("unused")
	private static final String _COPYRIGHT = "Licensed Materials - Property of IBM\n(C) Copyright IBM Corp. 2010, 2013\n" +
                                             "US Government Users Restricted Rights - Use, duplication  disclosure restricted by GSA ADP Schedule Contract with IBM Corp.";

	private static final String TEST_DIR = "functions/data/";
	
	@Override
	public void setUp() {
	
		// positive tests
		addTestConfiguration("MatrixTest", new TestConfiguration(TEST_DIR, "RandTest3", new String[] { "rand" }));
		
		// negative tests
	}

	@Test
	public void testMatrix() {
		int rows = 10;
		int cols = 10;
		double min = -1;
		double max = 1;

		TestConfiguration config = availableTestConfigurations.get("MatrixTest");
		config.addVariable("rows", rows);
		config.addVariable("cols", cols);
		config.addVariable("min", min);
		config.addVariable("max", max);

		loadTestConfiguration(config);

		runTest();

		checkResults(rows, cols, min, max);
	}


}
