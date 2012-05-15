package com.ibm.bi.dml.test.integration.functions.unary.matrix;

import org.junit.Test;

import com.ibm.bi.dml.test.integration.AutomatedTestBase;
import com.ibm.bi.dml.test.integration.TestConfiguration;
import com.ibm.bi.dml.utils.DMLRuntimeException;



public class SqrtTest extends AutomatedTestBase {

	@Override
	public void setUp() {
		baseDirectory = SCRIPT_DIR + "functions/unary/matrix/";
		availableTestConfigurations.put("PositiveTest",
				new TestConfiguration("SqrtTest", new String[] { "vector", "matrix" }));
		availableTestConfigurations.put("NegativeVectorTest",
				new TestConfiguration("SqrtSingleTest", new String[] { "out" }));
		availableTestConfigurations.put("NegativeMatrixTest",
				new TestConfiguration("SqrtSingleTest", new String[] { "out" }));
	}
	
	@Test
	public void testPositive() {
		int rows = 10;
		int cols = 10;
		
		TestConfiguration config = availableTestConfigurations.get("PositiveTest");
		config.addVariable("rows", rows);
		config.addVariable("cols", cols);
		
		loadTestConfiguration("PositiveTest");
		
		double[][] vector = getRandomMatrix(rows, 1, 0, 1, 1, -1);
		double[][] SqrtVector = new double[rows][1];
		for(int i = 0; i < rows; i++) {
			SqrtVector[i][0] = Math.sqrt(vector[i][0]);
		}
		writeInputMatrix("vector", vector);
		writeExpectedMatrix("vector", SqrtVector);
		
		double[][] matrix = getRandomMatrix(rows, cols, 0, 1, 1, -1);
		double[][] absMatrix = new double[rows][cols];
		for(int i = 0; i < rows; i++) {
			for(int j = 0; j < cols; j++) {
				absMatrix[i][j] = Math.sqrt(matrix[i][j]);
			}
		}
		writeInputMatrix("matrix", matrix);
		writeExpectedMatrix("matrix", absMatrix);
		
		runTest();
		
		compareResults();
	}
	
	@Test
	public void testNegativeVector() {
		int rows = 10;
		int cols = 1;
		
		TestConfiguration config = availableTestConfigurations.get("NegativeVectorTest");
		config.addVariable("rows", rows);
		config.addVariable("cols", cols);
		
		loadTestConfiguration("NegativeVectorTest");
		
		createRandomMatrix("in", rows, cols, -1, 0, 1, -1);
		
		runTest(true, DMLRuntimeException.class);
	}
	
	@Test
	public void testNegativeMatrix() {
		int rows = 10;
		int cols = 10;
		
		TestConfiguration config = availableTestConfigurations.get("NegativeMatrixTest");
		config.addVariable("rows", rows);
		config.addVariable("cols", cols);
		
		loadTestConfiguration("NegativeMatrixTest");
		
		createRandomMatrix("in", rows, cols, -1, 0, 1, -1);
		
		runTest(true, DMLRuntimeException.class);
	}
	
}
