package com.ibm.bi.dml.test.integration.functions.binary.matrix;

import java.util.Date;

import org.junit.Test;

import com.ibm.bi.dml.test.integration.AutomatedTestBase;
import com.ibm.bi.dml.test.integration.TestConfiguration;


public class ElementwiseAdditionMultiplicationTest extends AutomatedTestBase {

	@SuppressWarnings("deprecation")
	@Override
	public void setUp() {
		baseDirectory = SCRIPT_DIR + "functions/binary/matrix/";

		// positive tests
		availableTestConfigurations.put("Term1", new TestConfiguration("ElementwiseAdditionMultiplicationTerm1",
				new String[] { "result" }));
		availableTestConfigurations.put("Term2", new TestConfiguration("ElementwiseAdditionMultiplicationTerm2",
				new String[] { "result" }));
		availableTestConfigurations.put("Term3", new TestConfiguration("ElementwiseAdditionMultiplicationTerm3",
				new String[] { "result" }));
	}

	@Test
	public void testTerm1() {
		int rows = 5;
		int cols = 4;
		TestConfiguration config = availableTestConfigurations.get("Term1");
		config.addVariable("rows", rows);
		config.addVariable("cols", cols);
		loadTestConfiguration("Term1");

		double[][] a = createRandomMatrix("a", rows, cols, 0, 10, 1, new Date().getTime());
		double[][] b = createRandomMatrix("b", rows, cols, 0, 10, 1, new Date().getTime() + 1);
		double[][] c = createRandomMatrix("c", rows, cols, 0, 10, 1, new Date().getTime() + 2);
		double[][] d = createRandomMatrix("d", rows, cols, 0, 10, 1, new Date().getTime() + 4);

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
		loadTestConfiguration("Term2");

		double[][] a = createRandomMatrix("a", rows, cols, 0, 10, 1, new Date().getTime());
		double[][] b = createRandomMatrix("b", rows, cols, 0, 10, 1, new Date().getTime() + 1);
		double[][] c = createRandomMatrix("c", rows, cols, 0, 10, 1, new Date().getTime() + 4);

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
		loadTestConfiguration("Term3");

		double[][] a = createRandomMatrix("a", rows, cols, 0, 10, 1, new Date().getTime());
		double[][] b = createRandomMatrix("b", rows, cols, 0, 10, 1, new Date().getTime() + 1);
		double[][] c = createRandomMatrix("c", rows, cols, 0, 10, 1, new Date().getTime() + 2);

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
