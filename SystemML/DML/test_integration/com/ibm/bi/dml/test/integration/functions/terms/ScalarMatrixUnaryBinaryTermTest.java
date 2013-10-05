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
import com.ibm.bi.dml.test.utils.TestUtils;


public class ScalarMatrixUnaryBinaryTermTest extends AutomatedTestBase 
{
	@SuppressWarnings("unused")
	private static final String _COPYRIGHT = "Licensed Materials - Property of IBM\n(C) Copyright IBM Corp. 2010, 2013\n" +
                                             "US Government Users Restricted Rights - Use, duplication  disclosure restricted by GSA ADP Schedule Contract with IBM Corp.";
	
	@SuppressWarnings("deprecation")
	@Override
	public void setUp() {
		baseDirectory = SCRIPT_DIR + "functions/terms/";

		availableTestConfigurations.put("TestTerm1", new TestConfiguration("TestTerm1", new String[] {}));
	}

	@Test
	public void testTerm1() {
		int rows = 5, cols = 5;

		TestConfiguration config = getTestConfiguration("TestTerm1");
		config.addVariable("rows", rows);
		config.addVariable("cols", cols);

		loadTestConfiguration(config);

		double[][] a = createNonRandomMatrixValues(rows, cols);
		writeInputMatrix("a", a);

		double[][] w = new double[rows][cols];
		for (int i = 0; i < rows; i++) {
			for (int j = 0; j < cols; j++) {
				w[i][j] = 1 + a[i][j];
			}
		}
		w = TestUtils.performMatrixMultiplication(w, w);
		writeExpectedMatrix("w", w);

		runTest();

		compareResults();
	}
}
