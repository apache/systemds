package com.ibm.bi.dml.test.integration.functions.binary.matrix;

import org.junit.Test;


import com.ibm.bi.dml.runtime.functionobjects.Modulus;
import com.ibm.bi.dml.test.integration.AutomatedTestBase;
import com.ibm.bi.dml.test.integration.TestConfiguration;

import static com.ibm.bi.dml.runtime.util.UtilFunctions.toInt;

public class ScalarModulusTest extends AutomatedTestBase {
	
	@Override
	public void setUp() {
		baseDirectory = SCRIPT_DIR + "functions/binary/matrix/";
		
		// positive tests
		availableTestConfigurations.put("IntConstTest", new TestConfiguration("ScalarModulusTest",
				new String[] { "vector_left", "vector_right", "matrix_left", "matrix_right" }));
		availableTestConfigurations.put("IntVarTest", new TestConfiguration("ScalarModulusTest",
				new String[] { "vector_left", "vector_right", "matrix_left", "matrix_right" }));
		availableTestConfigurations.put("DoubleConstTest", new TestConfiguration("ScalarModulusTest",
				new String[] { "vector_left", "vector_right", "matrix_left", "matrix_right" }));
		availableTestConfigurations.put("DoubleVarTest", new TestConfiguration("ScalarModulusTest",
				new String[] { "vector_left", "vector_right", "matrix_left", "matrix_right" }));
		availableTestConfigurations.put("SparseTest", new TestConfiguration("ScalarModulusTest",
				new String[] { "vector_left", "vector_right", "matrix_left", "matrix_right" }));
		availableTestConfigurations.put("EmptyTest", new TestConfiguration("ScalarModulusTest",
				new String[] { "vector_left", "vector_right", "matrix_left", "matrix_right" }));
		availableTestConfigurations.put("DivisionByZeroTest", new TestConfiguration("ScalarModulusTest",
				new String[] { "vector_left", "vector_right", "matrix_left", "matrix_right" }));
		
		// negative tests
	}
	
	@Test
	public void testIntConst() {
		int rows = 10;
		int cols = 10;
		int divisor = 20;
		int dividend = 20;
		
		TestConfiguration config = availableTestConfigurations.get("IntConstTest");
		config.addVariable("rows", rows);
		config.addVariable("cols", cols);
		config.addVariable("vardeclaration", "");
		config.addVariable("divisor", divisor);
		config.addVariable("dividend", dividend);
		
		loadTestConfiguration("IntConstTest");
		
		double[][] vector = getNonZeroRandomMatrix(rows, 1, 1, 5, -1);
		double[][] computedVectorLeft = new double[rows][1];
		double[][] computedVectorRight = new double[rows][1];
		for(int i = 0; i < rows; i++) {
			computedVectorLeft[i][0] = toInt(vector[i][0]) % divisor;
			computedVectorRight[i][0] = dividend % toInt(vector[i][0]);
		}
		writeInputMatrix("vector", vector);
		writeExpectedMatrix("vector_left", computedVectorLeft);
		writeExpectedMatrix("vector_right", computedVectorRight);
		
		double[][] matrix = getNonZeroRandomMatrix(rows, cols, 1, 5, -1);
		double[][] computedMatrixLeft = new double[rows][cols];
		double[][] computedMatrixRight = new double[rows][cols];
		for(int i = 0; i < rows; i++) {
			for(int j = 0; j < cols; j++) {
				computedMatrixLeft[i][j] = toInt(matrix[i][j]) % divisor;
				computedMatrixRight[i][j] = dividend % toInt(matrix[i][j]);
			}
		}
		writeInputMatrix("matrix", matrix);
		writeExpectedMatrix("matrix_left", computedMatrixLeft);
		writeExpectedMatrix("matrix_right", computedMatrixRight);
		
		runTest();
		
		compareResults();
	}
	
	@Test
	public void testIntVar() {
		int rows = 10;
		int cols = 10;
		int divisor = 20;
		int dividend = 20;
		
		TestConfiguration config = availableTestConfigurations.get("IntVarTest");
		config.addVariable("rows", rows);
		config.addVariable("cols", cols);
		config.addVariable("vardeclaration", "divisor = " + divisor + "; dividend = " + dividend + ";");
		config.addVariable("divisor", "divisor");
		config.addVariable("dividend", "dividend");
		
		loadTestConfiguration("IntVarTest");
		
		double[][] vector = getNonZeroRandomMatrix(rows, 1, 1, 5, -1);
		double[][] computedVectorLeft = new double[rows][1];
		double[][] computedVectorRight = new double[rows][1];
		for(int i = 0; i < rows; i++) {
			computedVectorLeft[i][0] = toInt(vector[i][0]) % divisor;
			computedVectorRight[i][0] = dividend % toInt(vector[i][0]);
		}
		writeInputMatrix("vector", vector);
		writeExpectedMatrix("vector_left", computedVectorLeft);
		writeExpectedMatrix("vector_right", computedVectorRight);
		
		double[][] matrix = getNonZeroRandomMatrix(rows, cols, 1, 5, -1);
		double[][] computedMatrixLeft = new double[rows][cols];
		double[][] computedMatrixRight = new double[rows][cols];
		for(int i = 0; i < rows; i++) {
			for(int j = 0; j < cols; j++) {
				computedMatrixLeft[i][j] = toInt(matrix[i][j]) % divisor;
				computedMatrixRight[i][j] = dividend % toInt(matrix[i][j]);
			}
		}
		writeInputMatrix("matrix", matrix);
		writeExpectedMatrix("matrix_left", computedMatrixLeft);
		writeExpectedMatrix("matrix_right", computedMatrixRight);
		
		runTest();
		
		compareResults();
	}
	
	@Test
	public void testDoubleConst() {
		int rows = 10;
		int cols = 10;
		int divisor = 20;
		int dividend = 20;
		
		TestConfiguration config = availableTestConfigurations.get("DoubleConstTest");
		config.addVariable("rows", rows);
		config.addVariable("cols", cols);
		config.addVariable("vardeclaration", "");
		config.addVariable("divisor", divisor);
		config.addVariable("dividend", dividend);
		
		loadTestConfiguration("DoubleConstTest");
		
		double[][] vector = getNonZeroRandomMatrix(rows, 1, 1, 5, -1);
		double[][] computedVectorLeft = new double[rows][1];
		double[][] computedVectorRight = new double[rows][1];
		for(int i = 0; i < rows; i++) {
			computedVectorLeft[i][0] = toInt(vector[i][0]) % divisor;
			computedVectorRight[i][0] = dividend % toInt(vector[i][0]);
		}
		writeInputMatrix("vector", vector);
		writeExpectedMatrix("vector_left", computedVectorLeft);
		writeExpectedMatrix("vector_right", computedVectorRight);
		
		double[][] matrix = getNonZeroRandomMatrix(rows, cols, 1, 5, -1);
		double[][] computedMatrixLeft = new double[rows][cols];
		double[][] computedMatrixRight = new double[rows][cols];
		for(int i = 0; i < rows; i++) {
			for(int j = 0; j < cols; j++) {
				computedMatrixLeft[i][j] = toInt(matrix[i][j]) % divisor;
				computedMatrixRight[i][j] = dividend % toInt(matrix[i][j]);
			}
		}
		writeInputMatrix("matrix", matrix);
		writeExpectedMatrix("matrix_left", computedMatrixLeft);
		writeExpectedMatrix("matrix_right", computedMatrixRight);
		
		runTest();
		
		compareResults();
	}
	
	@Test
	public void testDoubleVar() {
		int rows = 10;
		int cols = 10;
		int divisor = 20;
		int dividend = 20;
		
		TestConfiguration config = availableTestConfigurations.get("DoubleVarTest");
		config.addVariable("rows", rows);
		config.addVariable("cols", cols);
		config.addVariable("vardeclaration", "divisor = " + divisor + "; dividend = " + dividend + ";");
		config.addVariable("divisor", "divisor");
		config.addVariable("dividend", "dividend");
		
		loadTestConfiguration("DoubleVarTest");
		
		double[][] vector = getNonZeroRandomMatrix(rows, 1, 1, 5, -1);
		double[][] computedVectorLeft = new double[rows][1];
		double[][] computedVectorRight = new double[rows][1];
		for(int i = 0; i < rows; i++) {
			computedVectorLeft[i][0] = toInt(vector[i][0]) % divisor;
			computedVectorRight[i][0] = dividend % toInt(vector[i][0]);
		}
		writeInputMatrix("vector", vector);
		writeExpectedMatrix("vector_left", computedVectorLeft);
		writeExpectedMatrix("vector_right", computedVectorRight);
		
		double[][] matrix = getNonZeroRandomMatrix(rows, cols, 1, 5, -1);
		double[][] computedMatrixLeft = new double[rows][cols];
		double[][] computedMatrixRight = new double[rows][cols];
		for(int i = 0; i < rows; i++) {
			for(int j = 0; j < cols; j++) {
				computedMatrixLeft[i][j] = toInt(matrix[i][j]) % divisor;
				computedMatrixRight[i][j] = dividend % toInt(matrix[i][j]);
			}
		}
		writeInputMatrix("matrix", matrix);
		writeExpectedMatrix("matrix_left", computedMatrixLeft);
		writeExpectedMatrix("matrix_right", computedMatrixRight);
		
		runTest();
		
		compareResults();
	}
	
	@Test
	public void testSparse() {
		int rows = 100;
		int cols = 50;
		int divisor = 20;
		int dividend = 20;
		
		TestConfiguration config = availableTestConfigurations.get("SparseTest");
		config.addVariable("rows", rows);
		config.addVariable("cols", cols);
		config.addVariable("vardeclaration", "");
		config.addVariable("divisor", divisor);
		config.addVariable("dividend", dividend);
		
		loadTestConfiguration("SparseTest");
		
		double[][] vector = getRandomMatrix(rows, 1, 1, 5, 0.05, -1);
		double[][] computedVectorLeft = new double[rows][1];
		double[][] computedVectorRight = new double[rows][1];
		for(int i = 0; i < rows; i++) {
			computedVectorLeft[i][0] = Modulus.modOrNaN(toInt(vector[i][0]), divisor);
			computedVectorRight[i][0] = Modulus.modOrNaN(dividend, toInt(vector[i][0]));
		}
		writeInputMatrix("vector", vector);
		writeExpectedMatrix("vector_left", computedVectorLeft);
		writeExpectedMatrix("vector_right", computedVectorRight);
		
		double[][] matrix = getRandomMatrix(rows, cols, -1, 1, 0.05, -1);
		double[][] computedMatrixLeft = new double[rows][cols];
		double[][] computedMatrixRight = new double[rows][cols];
		for(int i = 0; i < rows; i++) {
			for(int j = 0; j < cols; j++) {
				computedMatrixLeft[i][j] = Modulus.modOrNaN(toInt(matrix[i][j]), divisor);
				computedMatrixRight[i][j] = Modulus.modOrNaN(dividend, toInt(matrix[i][j]));
			}
		}
		writeInputMatrix("matrix", matrix);
		writeExpectedMatrix("matrix_left", computedMatrixLeft);
		writeExpectedMatrix("matrix_right", computedMatrixRight);
		
		runTest();
		
		compareResults();
	}
	
	@Test
	public void testDivisionByZero() {
		int rows = 10;
		int cols = 10;
		int divisor = 0;
		int dividend = 0;
		
		TestConfiguration config = availableTestConfigurations.get("DivisionByZeroTest");
		config.addVariable("rows", rows);
		config.addVariable("cols", cols);
		config.addVariable("vardeclaration", "");
		config.addVariable("divisor", divisor);
		config.addVariable("dividend", dividend);
		
		loadTestConfiguration("DivisionByZeroTest");
		
		double[][] vector = getRandomMatrix(rows, 1, 1, 5, 0.5, -1);
		double[][] computedVectorLeft = new double[rows][1];
		double[][] computedVectorRight = new double[rows][1];
		for(int i = 0; i < rows; i++) {
			computedVectorLeft[i][0] = Modulus.modOrNaN(toInt(vector[i][0]), divisor);
			computedVectorRight[i][0] = Modulus.modOrNaN(dividend, toInt(vector[i][0]));
		}
		writeInputMatrix("vector", vector);
		writeExpectedMatrix("vector_left", computedVectorLeft);
		writeExpectedMatrix("vector_right", computedVectorRight);
		
		double[][] matrix = getRandomMatrix(rows, cols, -1, 1, 0.5, -1);
		double[][] computedMatrixLeft = new double[rows][cols];
		double[][] computedMatrixRight = new double[rows][cols];
		for(int i = 0; i < rows; i++) {
			for(int j = 0; j < cols; j++) {
				computedMatrixLeft[i][j] = Modulus.modOrNaN(toInt(matrix[i][j]), divisor);
				computedMatrixRight[i][j] = Modulus.modOrNaN(dividend, toInt(matrix[i][j]));
			}
		}
		writeInputMatrix("matrix", matrix);
		writeExpectedMatrix("matrix_left", computedMatrixLeft);
		writeExpectedMatrix("matrix_right", computedMatrixRight);
		
		runTest();
		
		compareResults();
	}	

}
