package com.ibm.bi.dml.test.integration.functions.terms;

import org.junit.Test;

import com.ibm.bi.dml.test.integration.AutomatedTestBase;
import com.ibm.bi.dml.test.integration.TestConfiguration;
import com.ibm.bi.dml.test.utils.TestUtils;


public class ScalarMatrixUnaryBinaryTermTest extends AutomatedTestBase {
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
