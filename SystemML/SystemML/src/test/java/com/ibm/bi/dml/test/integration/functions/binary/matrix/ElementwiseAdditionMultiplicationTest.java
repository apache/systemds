/**
 * IBM Confidential
 * OCO Source Materials
 * (C) Copyright IBM Corp. 2010, 2014
 * The source code for this program is not published or otherwise divested of its trade secrets, irrespective of what has been deposited with the U.S. Copyright Office.
 */

package com.ibm.bi.dml.test.integration.functions.binary.matrix;

import java.util.Date;

import org.junit.Test;

import com.ibm.bi.dml.test.integration.AutomatedTestBase;
import com.ibm.bi.dml.test.integration.TestConfiguration;


public class ElementwiseAdditionMultiplicationTest extends AutomatedTestBase 
{
	@SuppressWarnings("unused")
	private static final String _COPYRIGHT = "Licensed Materials - Property of IBM\n(C) Copyright IBM Corp. 2010, 2014\n" +
                                             "US Government Users Restricted Rights - Use, duplication  disclosure restricted by GSA ADP Schedule Contract with IBM Corp.";
	
	private String TEST_DIR = "functions/binary/matrix/";
	
	@Override
	public void setUp() {

		// positive tests
		addTestConfiguration("Term1", new TestConfiguration(TEST_DIR, "ElementwiseAdditionMultiplicationTerm1",
				new String[] { "result" }));
		addTestConfiguration("Term2", new TestConfiguration(TEST_DIR, "ElementwiseAdditionMultiplicationTerm2",
				new String[] { "result" }));
		addTestConfiguration("Term3", new TestConfiguration(TEST_DIR, "ElementwiseAdditionMultiplicationTerm3",
				new String[] { "result" }));
	}

	@Test
	public void testTerm1() {
		int rows = 5;
		int cols = 4;
		TestConfiguration config = availableTestConfigurations.get("Term1");
		config.addVariable("rows", rows);
		config.addVariable("cols", cols);
		loadTestConfiguration(config);

		double[][] a = createRandomMatrix("a", rows, cols, 0, 10, 1, new Date().getTime());
		double[][] b = createRandomMatrix("b", rows, cols, 0, 10, 1, new Date().getTime() + 1);
		double[][] c = createRandomMatrix("c", rows, cols, 0, 10, 1, new Date().getTime() + 2);
		double[][] d = createRandomMatrix("d", rows, cols, 0, 10, 1, new Date().getTime() + 4);
		writeInputMatrixWithMTD("a", a, false);
		writeInputMatrixWithMTD("b", b, false);
		writeInputMatrixWithMTD("c", c, false);
		writeInputMatrixWithMTD("d", d, false);
		
		double[][] result = new double[rows][cols];
		for (int i = 0; i < rows; i++) {
			for (int j = 0; j < cols; j++) {
				result[i][j] = a[i][j] * b[i][j] + c[i][j] * d[i][j];
			}
		}

		writeExpectedMatrix("result", result);

		runTest(6);

		compareResults();
	}

	@Test
	public void testTerm2() {
		int rows = 5;
		int cols = 4;
		TestConfiguration config = availableTestConfigurations.get("Term2");
		config.addVariable("rows", rows);
		config.addVariable("cols", cols);
		loadTestConfiguration(config);

		double[][] a = createRandomMatrix("a", rows, cols, 0, 10, 1, new Date().getTime());
		double[][] b = createRandomMatrix("b", rows, cols, 0, 10, 1, new Date().getTime() + 1);
		double[][] c = createRandomMatrix("c", rows, cols, 0, 10, 1, new Date().getTime() + 4);
		writeInputMatrixWithMTD("a", a, false);
		writeInputMatrixWithMTD("b", b, false);
		writeInputMatrixWithMTD("c", c, false);
		
		double[][] result = new double[rows][cols];
		for (int i = 0; i < rows; i++) {
			for (int j = 0; j < cols; j++) {
				result[i][j] = a[i][j] * b[i][j] + a[i][j] * c[i][j];
			}
		}

		writeExpectedMatrix("result", result);

		runTest();

		compareResults();
	}

	@Test
	public void testTerm3() {
		int rows = 5;
		int cols = 4;
		TestConfiguration config = availableTestConfigurations.get("Term3");
		config.addVariable("rows", rows);
		config.addVariable("cols", cols);
		loadTestConfiguration(config);

		double[][] a = createRandomMatrix("a", rows, cols, 0, 10, 1, new Date().getTime());
		double[][] b = createRandomMatrix("b", rows, cols, 0, 10, 1, new Date().getTime() + 1);
		double[][] c = createRandomMatrix("c", rows, cols, 0, 10, 1, new Date().getTime() + 2);
		writeInputMatrixWithMTD("a", a, false);
		writeInputMatrixWithMTD("b", b, false);
		writeInputMatrixWithMTD("c", c, false);
		
		double[][] result = new double[rows][cols];
		for (int i = 0; i < rows; i++) {
			for (int j = 0; j < cols; j++) {
				result[i][j] = a[i][j] * b[i][j] * c[i][j];
			}
		}

		writeExpectedMatrix("result", result);

		runTest();

		compareResults();
	}
}
