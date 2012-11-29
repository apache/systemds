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
 * </ul>
 * <p>
 * <b>Negative tests:</b>
 * </p>
 * 
 * 
 */
public class RandTest4 extends AutomatedTestBase {

	@Override
	public void setUp() {
		baseDirectory = SCRIPT_DIR + "functions/data/";

		// positive tests
		availableTestConfigurations.put("MatrixTest", new TestConfiguration("RandTest4", new String[] { "rand" }));
		
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
		config.addVariable("format", "text");

		loadTestConfiguration(config);

		double[][] a = getRandomMatrix(rows, cols, 0, 1, 0.5, 7);
		writeInputMatrix("a", a);
		double sum = 0;
		for (int i = 0; i< rows; i++){
			for (int j = 0; j < cols; j++){
				sum += a[i][j];
			}
		}
		runTest();

		checkResults((int)sum, cols, min, max);
	}


}
