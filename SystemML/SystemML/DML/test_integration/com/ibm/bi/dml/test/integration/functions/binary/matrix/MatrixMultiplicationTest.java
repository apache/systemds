/**
 * IBM Confidential
 * OCO Source Materials
 * (C) Copyright IBM Corp. 2010, 2014
 * The source code for this program is not published or otherwise divested of its trade secrets, irrespective of what has been deposited with the U.S. Copyright Office.
 */

package com.ibm.bi.dml.test.integration.functions.binary.matrix;

import java.util.HashMap;

import org.junit.Test;

import com.ibm.bi.dml.api.DMLException;
import com.ibm.bi.dml.runtime.matrix.io.MatrixValue.CellIndex;
import com.ibm.bi.dml.test.integration.AutomatedTestBase;
import com.ibm.bi.dml.test.integration.TestConfiguration;
import com.ibm.bi.dml.test.utils.TestUtils;


public class MatrixMultiplicationTest extends AutomatedTestBase 
{
	@SuppressWarnings("unused")
	private static final String _COPYRIGHT = "Licensed Materials - Property of IBM\n(C) Copyright IBM Corp. 2010, 2014\n" +
                                             "US Government Users Restricted Rights - Use, duplication  disclosure restricted by GSA ADP Schedule Contract with IBM Corp.";
	
	@Override
	public void setUp() {
		baseDirectory = SCRIPT_DIR + "functions/binary/matrix/";

		// positive tests
		availableTestConfigurations.put("MatrixMultiplicationTest", new TestConfiguration("MatrixMultiplicationTest",
				new String[] { "c" }));
		availableTestConfigurations.put("WrongDimensionsTest", new TestConfiguration("MatrixMultiplicationTest",
				new String[] { "c" }));
		availableTestConfigurations.put("AMultASpecial1Test", new TestConfiguration("AMultASpecial1Test",
				new String[] { "a" }));
		availableTestConfigurations.put("AMultBSpecial2Test", new TestConfiguration("AMultBSpecial2Test",
				new String[] { "e" }));

		// negative tests
	}

	@Test
	public void testMatrixMultiplication() {
		int m = 20;
		int n = 20;
		int k = 20;

		TestConfiguration config = availableTestConfigurations.get("MatrixMultiplicationTest");
		config.addVariable("m", m);
		config.addVariable("n1", n);
		config.addVariable("n2", n);
		config.addVariable("k", k);

		loadTestConfiguration("MatrixMultiplicationTest");

		double[][] a = getRandomMatrix(m, n, -1, 1, 1, -1);
		double[][] b = getRandomMatrix(n, k, -1, 1, 1, -1);
		double[][] c = TestUtils.performMatrixMultiplication(a, b);

		writeInputMatrix("a", a);
		writeInputMatrix("b", b);
		writeExpectedMatrix("c", c);

		runTest();

		compareResults(0.00000000001);
	}

	@Test
	public void testWrongDimensions() {
		int m = 6;
		int n1 = 8;
		int n2 = 10;
		int k = 12;

		TestConfiguration config = availableTestConfigurations.get("WrongDimensionsTest");
		config.addVariable("m", m);
		config.addVariable("n1", n1);
		config.addVariable("n2", n2);
		config.addVariable("k", k);

		loadTestConfiguration("WrongDimensionsTest");

		createRandomMatrix("a", m, n1, -1, 1, 0.5, -1);
		createRandomMatrix("b", n2, k, -1, 1, 0.5, -1);

		runTest(true, DMLException.class);
	}

	@Test
	public void testAMultASpecial1() {
		int rows = 10;
		int cols = 10;

		TestConfiguration config = availableTestConfigurations.get("AMultASpecial1Test");
		config.addVariable("rows", rows);
		config.addVariable("cols", cols);
		loadTestConfiguration(config);

		double[][] a = createNonRandomMatrixValues(rows, cols);
		writeInputMatrix("a", a);

		a = TestUtils.performMatrixMultiplication(a, a);
		a = TestUtils.performMatrixMultiplication(a, a);

		writeExpectedMatrix("a", a);

		runTest();

		compareResults();
	}

	@Test
	public void testAMultBSpecial2() {
		int rows = 10;
		int cols = 10;

		TestConfiguration config = availableTestConfigurations.get("AMultBSpecial2Test");
		config.addVariable("rows", rows);
		config.addVariable("cols", cols);
		loadTestConfiguration(config);

		double[][] a = createNonRandomMatrixValues(rows, cols);
		writeInputMatrix("a", a);
		double[][] b = createNonRandomMatrixValues(rows, cols);
		writeInputMatrix("b", b);
		double[][] d = createNonRandomMatrixValues(rows, cols);
		writeInputMatrix("d", d);

		double[][] c = TestUtils.performMatrixMultiplication(a, b);
		double[][] e = TestUtils.performMatrixMultiplication(c, d);

		writeExpectedMatrix("e", e);
	
		runTest();
		
		HashMap<CellIndex, Double> hmDMLJ = TestUtils.convert2DDoubleArrayToHashMap(e);
		HashMap<CellIndex, Double> hmDMLE = readDMLMatrixFromHDFS("e");
		TestUtils.compareMatrices(hmDMLJ, hmDMLE, 0, "hmDMLJ","hmDMLE");
		
		TestUtils.displayAssertionBuffer();
	}
}
