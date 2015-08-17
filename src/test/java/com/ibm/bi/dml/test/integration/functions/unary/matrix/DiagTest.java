/**
 * IBM Confidential
 * OCO Source Materials
 * (C) Copyright IBM Corp. 2010, 2015
 * The source code for this program is not published or otherwise divested of its trade secrets, irrespective of what has been deposited with the U.S. Copyright Office.
 */

package com.ibm.bi.dml.test.integration.functions.unary.matrix;

import org.junit.Test;

import com.ibm.bi.dml.api.DMLException;
import com.ibm.bi.dml.test.integration.AutomatedTestBase;
import com.ibm.bi.dml.test.integration.TestConfiguration;



/**
 * <p><b>Positive tests:</b></p>
 * <ul>
 * 	<li>matrix to vector & matrix 2 vector</li>
 * </ul>
 * <p><b>Negative tests:</b></p>
 * <ul>
 * 	<li>wrong dimensions</li>
 * </ul>
 * 
 * 
 */
public class DiagTest extends AutomatedTestBase 
{
	@SuppressWarnings("unused")
	private static final String _COPYRIGHT = "Licensed Materials - Property of IBM\n(C) Copyright IBM Corp. 2010, 2015\n" +
                                             "US Government Users Restricted Rights - Use, duplication  disclosure restricted by GSA ADP Schedule Contract with IBM Corp.";

	private static final String TEST_DIR = "functions/unary/matrix/";
	
	@Override
	public void setUp() {
		
		// positive tests
		addTestConfiguration("DiagTest", new TestConfiguration(TEST_DIR, "DiagTest", new String[] { "b", "d" }));
		
		// negative tests
		addTestConfiguration("WrongDimensionsTest", new TestConfiguration(TEST_DIR, "DiagSingleTest",
				new String[] { "b" }));
	}
	
	@Test
	public void testDiag() {
		int rowsCols = 10;
		
		TestConfiguration config = availableTestConfigurations.get("DiagTest");
		config.addVariable("rows", rowsCols);
		config.addVariable("cols", rowsCols);
		
		loadTestConfiguration(config);
		
		double[][] a = getRandomMatrix(rowsCols, rowsCols, -1, 1, 0.5, -1);
		writeInputMatrix("a", a);
		
		double[][] b = new double[rowsCols][1];
		for(int i = 0; i < rowsCols; i++) {
			b[i][0] = a[i][i];
		}
		writeExpectedMatrix("b", b);
		
		double[][] c = getRandomMatrix(rowsCols, 1, -1, 1, 0.5, -1);
		writeInputMatrix("c", c);
		
		double[][] d = new double[rowsCols][rowsCols];
		for(int i = 0; i < rowsCols; i++) {
			d[i][i] = c[i][0];
		}
		writeExpectedMatrix("d", d);
		
		runTest();
		
		compareResults();
	}
	
	@Test
	public void testWrongDimensions() {
		int rows = 10;
		int cols = 9;
		
		TestConfiguration config = availableTestConfigurations.get("WrongDimensionsTest");
		config.addVariable("rows", rows);
		config.addVariable("cols", cols);
		
		loadTestConfiguration(config);
		
		createRandomMatrix("a", rows, cols, -1, 1, 0.5, -1);
		
		runTest(true, DMLException.class);
	}

}
