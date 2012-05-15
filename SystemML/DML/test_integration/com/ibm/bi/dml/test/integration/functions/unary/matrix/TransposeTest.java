package com.ibm.bi.dml.test.integration.functions.unary.matrix;

import org.junit.Test;

import com.ibm.bi.dml.test.integration.AutomatedTestBase;
import com.ibm.bi.dml.test.integration.TestConfiguration;



public class TransposeTest extends AutomatedTestBase {

	@Override
	public void setUp() {
		baseDirectory = SCRIPT_DIR + "functions/unary/matrix/";
		availableTestConfigurations.put("TransposeTest",
				new TestConfiguration("TransposeTest", new String[] { "vector", "matrix" }));
	}
	
	@Test
	public void testTranspose() {
		int rows = 10;
		int cols = 10;
		
		TestConfiguration config = availableTestConfigurations.get("TransposeTest");
		config.addVariable("rows", rows);
		config.addVariable("cols", cols);
		
		loadTestConfiguration("TransposeTest");
		
		double[][] vector = getRandomMatrix(rows, 1, 0, 1, 1, -1);
		double[][] transposedVector = new double[1][rows];
		for(int i = 0; i < rows; i++) {
			transposedVector[0][i] = vector[i][0];
		}
		writeInputMatrix("vector", vector);
		writeExpectedMatrix("vector", transposedVector);
		
		double[][] matrix = getRandomMatrix(rows, cols, 0, 1, 1, -1);
		double[][] transposedMatrix = new double[cols][rows];
		for(int i = 0; i < rows; i++) {
			for(int j = 0; j < cols; j++) {
				transposedMatrix[j][i] = matrix[i][j];
			}
		}
		writeInputMatrix("matrix", matrix);
		writeExpectedMatrix("matrix", transposedMatrix);
		
		runTest();
		
		compareResults();
	}
	
}
