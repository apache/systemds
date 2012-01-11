package dml.test.integration.functions.unary.matrix;

import org.junit.Test;

import dml.test.integration.AutomatedTestBase;
import dml.test.integration.TestConfiguration;


public class NegationTest extends AutomatedTestBase {

	@Override
	public void setUp() {
		baseDirectory = SCRIPT_DIR + "functions/unary/matrix/";
		availableTestConfigurations.put("PositiveTest",
				new TestConfiguration("NegationTest", new String[] { "vector", "matrix" }));
		availableTestConfigurations.put("NegativeTest",
				new TestConfiguration("NegationTest", new String[] { "vector", "matrix" }));
		availableTestConfigurations.put("RandomTest",
				new TestConfiguration("NegationTest", new String[] { "vector", "matrix" }));
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
		double[][] negativeVector = new double[rows][1];
		for(int i = 0; i < rows; i++) {
			negativeVector[i][0] = -vector[i][0];
		}
		writeInputMatrix("vector", vector);
		writeExpectedMatrix("vector", negativeVector);
		
		double[][] matrix = getRandomMatrix(rows, cols, 0, 1, 1, -1);
		double[][] negativeMatrix = new double[rows][cols];
		for(int i = 0; i < rows; i++) {
			for(int j = 0; j < cols; j++) {
				negativeMatrix[i][j] = -matrix[i][j];
			}
		}
		writeInputMatrix("matrix", matrix);
		writeExpectedMatrix("matrix", negativeMatrix);
		
		runTest();
		
		compareResults();
	}
	
	@Test
	public void testNegative() {
		int rows = 10;
		int cols = 10;
		
		TestConfiguration config = availableTestConfigurations.get("NegativeTest");
		config.addVariable("rows", rows);
		config.addVariable("cols", cols);
		
		loadTestConfiguration("NegativeTest");
		
		double[][] vector = getRandomMatrix(rows, 1, -1, 0, 1, -1);
		double[][] negativeVector = new double[rows][1];
		for(int i = 0; i < rows; i++) {
			negativeVector[i][0] = -vector[i][0];
		}
		writeInputMatrix("vector", vector);
		writeExpectedMatrix("vector", negativeVector);
		
		double[][] matrix = getRandomMatrix(rows, cols, -1, 0, 1, -1);
		double[][] negativeMatrix = new double[rows][cols];
		for(int i = 0; i < rows; i++) {
			for(int j = 0; j < cols; j++) {
				negativeMatrix[i][j] = -matrix[i][j];
			}
		}
		writeInputMatrix("matrix", matrix);
		writeExpectedMatrix("matrix", negativeMatrix);
		
		runTest();
		
		compareResults();
	}
	
	@Test
	public void testRandom() {
		int rows = 10;
		int cols = 10;
		
		TestConfiguration config = availableTestConfigurations.get("RandomTest");
		config.addVariable("rows", rows);
		config.addVariable("cols", cols);
		
		loadTestConfiguration("RandomTest");
		
		double[][] vector = getRandomMatrix(rows, 1, -1, 1, 1, -1);
		double[][] negativeVector = new double[rows][1];
		for(int i = 0; i < rows; i++) {
			negativeVector[i][0] = -vector[i][0];
		}
		writeInputMatrix("vector", vector);
		writeExpectedMatrix("vector", negativeVector);
		
		double[][] matrix = getRandomMatrix(rows, cols, -1, 1, 1, -1);
		double[][] negativeMatrix = new double[rows][cols];
		for(int i = 0; i < rows; i++) {
			for(int j = 0; j < cols; j++) {
				negativeMatrix[i][j] = -matrix[i][j];
			}
		}
		writeInputMatrix("matrix", matrix);
		writeExpectedMatrix("matrix", negativeMatrix);
		
		runTest();
		
		compareResults();
	}
	
}
