/**
 * IBM Confidential
 * OCO Source Materials
 * (C) Copyright IBM Corp. 2010, 2013
 * The source code for this program is not published or otherwise divested of its trade secrets, irrespective of what has been deposited with the U.S. Copyright Office.
 */

package com.ibm.bi.dml.test.integration.functions.binary.matrix;

import org.junit.Test;

import com.ibm.bi.dml.runtime.DMLRuntimeException;
import com.ibm.bi.dml.test.integration.AutomatedTestBase;
import com.ibm.bi.dml.test.integration.TestConfiguration;


public class ElementwiseMultiplicationTest extends AutomatedTestBase 
{
	@SuppressWarnings("unused")
	private static final String _COPYRIGHT = "Licensed Materials - Property of IBM\n(C) Copyright IBM Corp. 2010, 2013\n" +
                                             "US Government Users Restricted Rights - Use, duplication  disclosure restricted by GSA ADP Schedule Contract with IBM Corp.";
	
	@Override
	public void setUp() {
		baseDirectory = SCRIPT_DIR + "functions/binary/matrix/";

		// positive tests
		availableTestConfigurations.put("DenseTest", new TestConfiguration("ElementwiseMultiplicationTest",
				new String[] { "c" }));
		availableTestConfigurations.put("SparseTest", new TestConfiguration("ElementwiseMultiplicationTest",
				new String[] { "c" }));
		availableTestConfigurations.put("EmptyTest", new TestConfiguration("ElementwiseMultiplicationTest",
				new String[] { "c" }));
		availableTestConfigurations.put("WrongDimensionLessRowsTest", new TestConfiguration(
				"ElementwiseMultiplicationVariableDimensionsTest", new String[] { "c" }));
		availableTestConfigurations.put("WrongDimensionMoreRowsTest", new TestConfiguration(
				"ElementwiseMultiplicationVariableDimensionsTest", new String[] { "c" }));
		availableTestConfigurations.put("WrongDimensionLessColsTest", new TestConfiguration(
				"ElementwiseMultiplicationVariableDimensionsTest", new String[] { "c" }));
		availableTestConfigurations.put("WrongDimensionMoreColsTest", new TestConfiguration(
				"ElementwiseMultiplicationVariableDimensionsTest", new String[] { "c" }));
		availableTestConfigurations.put("WrongDimensionLessRowsLessColsTest", new TestConfiguration(
				"ElementwiseMultiplicationVariableDimensionsTest", new String[] { "c" }));
		availableTestConfigurations.put("WrongDimensionMoreRowsMoreColsTest", new TestConfiguration(
				"ElementwiseMultiplicationVariableDimensionsTest", new String[] { "c" }));
		availableTestConfigurations.put("WrongDimensionLessRowsMoreColsTest", new TestConfiguration(
				"ElementwiseMultiplicationVariableDimensionsTest", new String[] { "c" }));
		availableTestConfigurations.put("WrongDimensionMoreRowsLessColsTest", new TestConfiguration(
				"ElementwiseMultiplicationVariableDimensionsTest", new String[] { "c" }));

		// negative tests
	}

	@Test
	public void testDense() {
		int rows = 10;
		int cols = 10;

		TestConfiguration config = availableTestConfigurations.get("DenseTest");
		config.addVariable("rows", rows);
		config.addVariable("cols", cols);

		loadTestConfiguration("DenseTest");

		double[][] a = getRandomMatrix(rows, cols, -1, 1, 1, -1);
		double[][] b = getRandomMatrix(rows, cols, -1, 1, 1, -1);
		double[][] c = new double[rows][cols];
		for (int i = 0; i < rows; i++) {
			for (int j = 0; j < cols; j++) {
				c[i][j] = a[i][j] * b[i][j];
			}
		}

		writeInputMatrix("a", a);
		writeInputMatrix("b", b);
		writeExpectedMatrix("c", c);

		runTest();

		compareResults();
	}

	@Test
	public void testSparse() {
		int rows = 50;
		int cols = 50;

		TestConfiguration config = availableTestConfigurations.get("SparseTest");
		config.addVariable("rows", rows);
		config.addVariable("cols", cols);

		loadTestConfiguration("SparseTest");

		double[][] a = getRandomMatrix(rows, cols, -1, 1, 0.05, -1);
		double[][] b = getRandomMatrix(rows, cols, -1, 1, 0.05, -1);
		double[][] c = new double[rows][cols];
		for (int i = 0; i < rows; i++) {
			for (int j = 0; j < cols; j++) {
				c[i][j] = a[i][j] * b[i][j];
			}
		}

		writeInputMatrix("a", a);
		writeInputMatrix("b", b);
		writeExpectedMatrix("c", c);

		runTest();

		compareResults();
	}

	@Test
	public void testWrongDimensionsLessRows() {
		int rows1 = 8;
		int cols1 = 10;
		int rows2 = 10;
		int cols2 = 10;

		TestConfiguration config = availableTestConfigurations.get("WrongDimensionLessRowsTest");
		config.addVariable("rows1", rows1);
		config.addVariable("cols1", cols1);
		config.addVariable("rows2", rows2);
		config.addVariable("cols2", cols2);

		loadTestConfiguration("WrongDimensionLessRowsTest");

		runTest(true, DMLRuntimeException.class);
	}

	@Test
	public void testWrongDimensionsMoreRows() {
		int rows1 = 12;
		int cols1 = 10;
		int rows2 = 10;
		int cols2 = 10;

		TestConfiguration config = availableTestConfigurations.get("WrongDimensionMoreRowsTest");
		config.addVariable("rows1", rows1);
		config.addVariable("cols1", cols1);
		config.addVariable("rows2", rows2);
		config.addVariable("cols2", cols2);

		loadTestConfiguration("WrongDimensionMoreRowsTest");

		runTest(true, DMLRuntimeException.class);
	}

	@Test
	public void testWrongDimensionsLessCols() {
		int rows1 = 10;
		int cols1 = 8;
		int rows2 = 10;
		int cols2 = 10;

		TestConfiguration config = availableTestConfigurations.get("WrongDimensionLessColsTest");
		config.addVariable("rows1", rows1);
		config.addVariable("cols1", cols1);
		config.addVariable("rows2", rows2);
		config.addVariable("cols2", cols2);

		loadTestConfiguration("WrongDimensionLessColsTest");

		runTest(true, DMLRuntimeException.class);
	}

	@Test
	public void testWrongDimensionsMoreCols() {
		int rows1 = 10;
		int cols1 = 12;
		int rows2 = 10;
		int cols2 = 10;

		TestConfiguration config = availableTestConfigurations.get("WrongDimensionMoreColsTest");
		config.addVariable("rows1", rows1);
		config.addVariable("cols1", cols1);
		config.addVariable("rows2", rows2);
		config.addVariable("cols2", cols2);

		loadTestConfiguration("WrongDimensionMoreColsTest");

		runTest(true, DMLRuntimeException.class);
	}

	@Test
	public void testWrongDimensionsLessRowsLessCols() {
		int rows1 = 8;
		int cols1 = 8;
		int rows2 = 10;
		int cols2 = 10;

		TestConfiguration config = availableTestConfigurations.get("WrongDimensionLessRowsLessColsTest");
		config.addVariable("rows1", rows1);
		config.addVariable("cols1", cols1);
		config.addVariable("rows2", rows2);
		config.addVariable("cols2", cols2);

		loadTestConfiguration("WrongDimensionLessRowsLessColsTest");

		runTest(true, DMLRuntimeException.class);
	}

	@Test
	public void testWrongDimensionsMoreRowsMoreCols() {
		int rows1 = 12;
		int cols1 = 12;
		int rows2 = 10;
		int cols2 = 10;

		TestConfiguration config = availableTestConfigurations.get("WrongDimensionMoreRowsMoreColsTest");
		config.addVariable("rows1", rows1);
		config.addVariable("cols1", cols1);
		config.addVariable("rows2", rows2);
		config.addVariable("cols2", cols2);

		loadTestConfiguration("WrongDimensionMoreRowsMoreColsTest");

		runTest(true, DMLRuntimeException.class);
	}

	@Test
	public void testWrongDimensionsLessRowsMoreCols() {
		int rows1 = 8;
		int cols1 = 12;
		int rows2 = 10;
		int cols2 = 10;

		TestConfiguration config = availableTestConfigurations.get("WrongDimensionLessRowsMoreColsTest");
		config.addVariable("rows1", rows1);
		config.addVariable("cols1", cols1);
		config.addVariable("rows2", rows2);
		config.addVariable("cols2", cols2);

		loadTestConfiguration("WrongDimensionLessRowsMoreColsTest");

		runTest(true, DMLRuntimeException.class);
	}

	@Test
	public void testWrongDimensionsMoreRowsLessCols() {
		int rows1 = 12;
		int cols1 = 8;
		int rows2 = 10;
		int cols2 = 10;

		TestConfiguration config = availableTestConfigurations.get("WrongDimensionMoreRowsLessColsTest");
		config.addVariable("rows1", rows1);
		config.addVariable("cols1", cols1);
		config.addVariable("rows2", rows2);
		config.addVariable("cols2", cols2);

		loadTestConfiguration("WrongDimensionMoreRowsLessColsTest");

		runTest(true, DMLRuntimeException.class);
	}
}
