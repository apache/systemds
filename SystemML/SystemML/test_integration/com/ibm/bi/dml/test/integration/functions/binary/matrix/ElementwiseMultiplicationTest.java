/**
 * IBM Confidential
 * OCO Source Materials
 * (C) Copyright IBM Corp. 2010, 2015
 * The source code for this program is not published or otherwise divested of its trade secrets, irrespective of what has been deposited with the U.S. Copyright Office.
 */

package com.ibm.bi.dml.test.integration.functions.binary.matrix;

import org.junit.Test;

import com.ibm.bi.dml.api.DMLException;
import com.ibm.bi.dml.test.integration.AutomatedTestBase;
import com.ibm.bi.dml.test.integration.TestConfiguration;
import com.ibm.bi.dml.test.utils.TestUtils;


public class ElementwiseMultiplicationTest extends AutomatedTestBase 
{
	@SuppressWarnings("unused")
	private static final String _COPYRIGHT = "Licensed Materials - Property of IBM\n(C) Copyright IBM Corp. 2010, 2015\n" +
                                             "US Government Users Restricted Rights - Use, duplication  disclosure restricted by GSA ADP Schedule Contract with IBM Corp.";
	
	private static final String TEST_DIR = "functions/binary/matrix/";
	
	@Override
	public void setUp() {
		TestUtils.clearAssertionInformation();

		// positive tests
		addTestConfiguration("DenseTest", new TestConfiguration(TEST_DIR,"ElementwiseMultiplicationTest",
				new String[] { "c" }));
		addTestConfiguration("SparseTest", new TestConfiguration(TEST_DIR,"ElementwiseMultiplicationTest",
				new String[] { "c" }));
		addTestConfiguration("EmptyTest", new TestConfiguration(TEST_DIR,"ElementwiseMultiplicationTest",
				new String[] { "c" }));
		addTestConfiguration("WrongDimensionLessRowsTest", new TestConfiguration(TEST_DIR,
				"ElementwiseMultiplicationVariableDimensionsTest", new String[] { "c" }));
		addTestConfiguration("WrongDimensionMoreRowsTest", new TestConfiguration(TEST_DIR,
				"ElementwiseMultiplicationVariableDimensionsTest", new String[] { "c" }));
		addTestConfiguration("WrongDimensionLessColsTest", new TestConfiguration(TEST_DIR,
				"ElementwiseMultiplicationVariableDimensionsTest", new String[] { "c" }));
		addTestConfiguration("WrongDimensionMoreColsTest", new TestConfiguration(TEST_DIR,
				"ElementwiseMultiplicationVariableDimensionsTest", new String[] { "c" }));
		addTestConfiguration("WrongDimensionLessRowsLessColsTest", new TestConfiguration(TEST_DIR,
				"ElementwiseMultiplicationVariableDimensionsTest", new String[] { "c" }));
		addTestConfiguration("WrongDimensionMoreRowsMoreColsTest", new TestConfiguration(TEST_DIR,
				"ElementwiseMultiplicationVariableDimensionsTest", new String[] { "c" }));
		addTestConfiguration("WrongDimensionLessRowsMoreColsTest", new TestConfiguration(TEST_DIR,
				"ElementwiseMultiplicationVariableDimensionsTest", new String[] { "c" }));
		addTestConfiguration("WrongDimensionMoreRowsLessColsTest", new TestConfiguration(TEST_DIR,
				"ElementwiseMultiplicationVariableDimensionsTest", new String[] { "c" }));

		// negative tests
	}

	@Test
	public void testDense() {
		int rows = 10;
		int cols = 10;

		TestConfiguration config = getTestConfiguration("DenseTest");
		config.addVariable("rows", rows);
		config.addVariable("cols", cols);

		loadTestConfiguration(config);

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

		TestConfiguration config = getTestConfiguration("SparseTest");
		config.addVariable("rows", rows);
		config.addVariable("cols", cols);

		loadTestConfiguration(config);

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

		TestConfiguration config = getTestConfiguration("WrongDimensionLessRowsTest");
		config.addVariable("rows1", rows1);
		config.addVariable("cols1", cols1);
		config.addVariable("rows2", rows2);
		config.addVariable("cols2", cols2);

		loadTestConfiguration(config);

		runTest(true, DMLException.class);
	}

	@Test
	public void testWrongDimensionsMoreRows() {
		int rows1 = 12;
		int cols1 = 10;
		int rows2 = 10;
		int cols2 = 10;

		TestConfiguration config = getTestConfiguration("WrongDimensionMoreRowsTest");
		config.addVariable("rows1", rows1);
		config.addVariable("cols1", cols1);
		config.addVariable("rows2", rows2);
		config.addVariable("cols2", cols2);

		loadTestConfiguration(config);

		runTest(true, DMLException.class);
	}

	@Test
	public void testWrongDimensionsLessCols() {
		int rows1 = 10;
		int cols1 = 8;
		int rows2 = 10;
		int cols2 = 10;

		TestConfiguration config = getTestConfiguration("WrongDimensionLessColsTest");
		config.addVariable("rows1", rows1);
		config.addVariable("cols1", cols1);
		config.addVariable("rows2", rows2);
		config.addVariable("cols2", cols2);

		loadTestConfiguration(config);

		runTest(true, DMLException.class);
	}

	@Test
	public void testWrongDimensionsMoreCols() {
		int rows1 = 10;
		int cols1 = 12;
		int rows2 = 10;
		int cols2 = 10;

		TestConfiguration config = getTestConfiguration("WrongDimensionMoreColsTest");
		config.addVariable("rows1", rows1);
		config.addVariable("cols1", cols1);
		config.addVariable("rows2", rows2);
		config.addVariable("cols2", cols2);

		loadTestConfiguration(config);

		runTest(true, DMLException.class);
	}

	@Test
	public void testWrongDimensionsLessRowsLessCols() {
		int rows1 = 8;
		int cols1 = 8;
		int rows2 = 10;
		int cols2 = 10;

		TestConfiguration config = getTestConfiguration("WrongDimensionLessRowsLessColsTest");
		config.addVariable("rows1", rows1);
		config.addVariable("cols1", cols1);
		config.addVariable("rows2", rows2);
		config.addVariable("cols2", cols2);

		loadTestConfiguration(config);

		runTest(true, DMLException.class);
	}

	@Test
	public void testWrongDimensionsMoreRowsMoreCols() {
		int rows1 = 12;
		int cols1 = 12;
		int rows2 = 10;
		int cols2 = 10;

		TestConfiguration config = getTestConfiguration("WrongDimensionMoreRowsMoreColsTest");
		config.addVariable("rows1", rows1);
		config.addVariable("cols1", cols1);
		config.addVariable("rows2", rows2);
		config.addVariable("cols2", cols2);

		loadTestConfiguration(config);

		runTest(true, DMLException.class);
	}

	@Test
	public void testWrongDimensionsLessRowsMoreCols() {
		int rows1 = 8;
		int cols1 = 12;
		int rows2 = 10;
		int cols2 = 10;

		TestConfiguration config = getTestConfiguration("WrongDimensionLessRowsMoreColsTest");
		config.addVariable("rows1", rows1);
		config.addVariable("cols1", cols1);
		config.addVariable("rows2", rows2);
		config.addVariable("cols2", cols2);

		loadTestConfiguration(config);

		runTest(true, DMLException.class);
	}

	@Test
	public void testWrongDimensionsMoreRowsLessCols() {
		int rows1 = 12;
		int cols1 = 8;
		int rows2 = 10;
		int cols2 = 10;

		TestConfiguration config = getTestConfiguration("WrongDimensionMoreRowsLessColsTest");
		config.addVariable("rows1", rows1);
		config.addVariable("cols1", cols1);
		config.addVariable("rows2", rows2);
		config.addVariable("cols2", cols2);

		loadTestConfiguration(config);

		runTest(true, DMLException.class);
	}
}
