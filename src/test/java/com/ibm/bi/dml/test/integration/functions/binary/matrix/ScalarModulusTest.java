/**
 * IBM Confidential
 * OCO Source Materials
 * (C) Copyright IBM Corp. 2010, 2015
 * The source code for this program is not published or otherwise divested of its trade secrets, irrespective of what has been deposited with the U.S. Copyright Office.
 */

package com.ibm.bi.dml.test.integration.functions.binary.matrix;

import org.junit.Test;


import com.ibm.bi.dml.runtime.functionobjects.Modulus;
import com.ibm.bi.dml.test.integration.AutomatedTestBase;
import com.ibm.bi.dml.test.integration.TestConfiguration;


public class ScalarModulusTest extends AutomatedTestBase 
{
	@SuppressWarnings("unused")
	private static final String _COPYRIGHT = "Licensed Materials - Property of IBM\n(C) Copyright IBM Corp. 2010, 2015\n" +
                                             "US Government Users Restricted Rights - Use, duplication  disclosure restricted by GSA ADP Schedule Contract with IBM Corp.";
		
	private static final String TEST_DIR = "functions/binary/matrix/";
	
	@Override
	public void setUp() {
		// positive tests
		addTestConfiguration("IntConstTest", new TestConfiguration(TEST_DIR, "ScalarModulusTest",
				new String[] { "vector_left", "vector_right", "matrix_left", "matrix_right" }));
		addTestConfiguration("IntVarTest", new TestConfiguration(TEST_DIR, "ScalarModulusTest",
				new String[] { "vector_left", "vector_right", "matrix_left", "matrix_right" }));
		addTestConfiguration("DoubleConstTest", new TestConfiguration(TEST_DIR, "ScalarModulusTest",
				new String[] { "vector_left", "vector_right", "matrix_left", "matrix_right" }));
		addTestConfiguration("DoubleVarTest", new TestConfiguration(TEST_DIR, "ScalarModulusTest",
				new String[] { "vector_left", "vector_right", "matrix_left", "matrix_right" }));
		addTestConfiguration("SparseTest", new TestConfiguration(TEST_DIR, "ScalarModulusTest",
				new String[] { "vector_left", "vector_right", "matrix_left", "matrix_right" }));
		addTestConfiguration("EmptyTest", new TestConfiguration(TEST_DIR, "ScalarModulusTest",
				new String[] { "vector_left", "vector_right", "matrix_left", "matrix_right" }));
		addTestConfiguration("DivisionByZeroTest", new TestConfiguration(TEST_DIR, "ScalarModulusTest",
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
		
		loadTestConfiguration(config);
		
		double[][] vector = getNonZeroRandomMatrix(rows, 1, 1, 5, -1);
		double[][] computedVectorLeft = new double[rows][1];
		double[][] computedVectorRight = new double[rows][1];
		Modulus fnmod = Modulus.getModulusFnObject();
		for(int i = 0; i < rows; i++) {
			computedVectorLeft[i][0] = fnmod.execute(vector[i][0], divisor);
			computedVectorRight[i][0] = fnmod.execute(dividend, vector[i][0]);
		}
		writeInputMatrix("vector", vector);
		writeExpectedMatrix("vector_left", computedVectorLeft);
		writeExpectedMatrix("vector_right", computedVectorRight);
		
		double[][] matrix = getNonZeroRandomMatrix(rows, cols, 1, 5, -1);
		double[][] computedMatrixLeft = new double[rows][cols];
		double[][] computedMatrixRight = new double[rows][cols];
		for(int i = 0; i < rows; i++) {
			for(int j = 0; j < cols; j++) {
				computedMatrixLeft[i][j] = fnmod.execute(matrix[i][j], divisor);
				computedMatrixRight[i][j] = fnmod.execute(dividend, matrix[i][j]);
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
		
		loadTestConfiguration(config);
		
		double[][] vector = getNonZeroRandomMatrix(rows, 1, 1, 5, -1);
		double[][] computedVectorLeft = new double[rows][1];
		double[][] computedVectorRight = new double[rows][1];
		Modulus fnmod = Modulus.getModulusFnObject();
		for(int i = 0; i < rows; i++) {
			computedVectorLeft[i][0] = fnmod.execute(vector[i][0], divisor);
			computedVectorRight[i][0] = fnmod.execute(dividend, vector[i][0]);
		}
		writeInputMatrix("vector", vector);
		writeExpectedMatrix("vector_left", computedVectorLeft);
		writeExpectedMatrix("vector_right", computedVectorRight);
		
		double[][] matrix = getNonZeroRandomMatrix(rows, cols, 1, 5, -1);
		double[][] computedMatrixLeft = new double[rows][cols];
		double[][] computedMatrixRight = new double[rows][cols];
		for(int i = 0; i < rows; i++) {
			for(int j = 0; j < cols; j++) {
				computedMatrixLeft[i][j] = fnmod.execute(matrix[i][j], divisor);
				computedMatrixRight[i][j] = fnmod.execute(dividend, matrix[i][j]);
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
		
		loadTestConfiguration(config);
		
		double[][] vector = getNonZeroRandomMatrix(rows, 1, 1, 5, -1);
		double[][] computedVectorLeft = new double[rows][1];
		double[][] computedVectorRight = new double[rows][1];
		Modulus fnmod = Modulus.getModulusFnObject();
		for(int i = 0; i < rows; i++) {
			computedVectorLeft[i][0] = fnmod.execute(vector[i][0], divisor);
			computedVectorRight[i][0] = fnmod.execute(dividend, vector[i][0]);
		}
		writeInputMatrix("vector", vector);
		writeExpectedMatrix("vector_left", computedVectorLeft);
		writeExpectedMatrix("vector_right", computedVectorRight);
		
		double[][] matrix = getNonZeroRandomMatrix(rows, cols, 1, 5, -1);
		double[][] computedMatrixLeft = new double[rows][cols];
		double[][] computedMatrixRight = new double[rows][cols];
		for(int i = 0; i < rows; i++) {
			for(int j = 0; j < cols; j++) {
				computedMatrixLeft[i][j] = fnmod.execute(matrix[i][j], divisor);
				computedMatrixRight[i][j] = fnmod.execute(dividend, matrix[i][j]);
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
		
		loadTestConfiguration(config);
		
		double[][] vector = getNonZeroRandomMatrix(rows, 1, 1, 5, -1);
		double[][] computedVectorLeft = new double[rows][1];
		double[][] computedVectorRight = new double[rows][1];
		Modulus fnmod = Modulus.getModulusFnObject();
		for(int i = 0; i < rows; i++) {
			computedVectorLeft[i][0] = fnmod.execute(vector[i][0], divisor);
			computedVectorRight[i][0] = fnmod.execute(dividend, vector[i][0]);
		}
		writeInputMatrix("vector", vector);
		writeExpectedMatrix("vector_left", computedVectorLeft);
		writeExpectedMatrix("vector_right", computedVectorRight);
		
		double[][] matrix = getNonZeroRandomMatrix(rows, cols, 1, 5, -1);
		double[][] computedMatrixLeft = new double[rows][cols];
		double[][] computedMatrixRight = new double[rows][cols];
		for(int i = 0; i < rows; i++) {
			for(int j = 0; j < cols; j++) {
				computedMatrixLeft[i][j] = fnmod.execute(matrix[i][j], divisor);
				computedMatrixRight[i][j] = fnmod.execute(dividend, matrix[i][j]);
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
		
		loadTestConfiguration(config);
		
		double[][] vector = getRandomMatrix(rows, 1, 1, 5, 0.05, -1);
		double[][] computedVectorLeft = new double[rows][1];
		double[][] computedVectorRight = new double[rows][1];
		Modulus fnmod = Modulus.getModulusFnObject();
		for(int i = 0; i < rows; i++) {
			computedVectorLeft[i][0] = fnmod.execute(vector[i][0], divisor);
			computedVectorRight[i][0] = fnmod.execute(dividend, vector[i][0]);
		}
		writeInputMatrix("vector", vector);
		writeExpectedMatrix("vector_left", computedVectorLeft);
		writeExpectedMatrix("vector_right", computedVectorRight);
		
		double[][] matrix = getRandomMatrix(rows, cols, -1, 1, 0.05, -1);
		double[][] computedMatrixLeft = new double[rows][cols];
		double[][] computedMatrixRight = new double[rows][cols];
		for(int i = 0; i < rows; i++) {
			for(int j = 0; j < cols; j++) {
				computedMatrixLeft[i][j] = fnmod.execute(matrix[i][j], divisor);
				computedMatrixRight[i][j] = fnmod.execute(dividend, matrix[i][j]);
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
		
		loadTestConfiguration(config);
		
		double[][] vector = getRandomMatrix(rows, 1, 1, 5, 0.5, -1);
		double[][] computedVectorLeft = new double[rows][1];
		double[][] computedVectorRight = new double[rows][1];
		Modulus fnmod = Modulus.getModulusFnObject();
		for(int i = 0; i < rows; i++) {
			computedVectorLeft[i][0] = fnmod.execute(vector[i][0], divisor);
			computedVectorRight[i][0] = fnmod.execute(dividend, vector[i][0]);
		}
		writeInputMatrix("vector", vector);
		writeExpectedMatrix("vector_left", computedVectorLeft);
		writeExpectedMatrix("vector_right", computedVectorRight);
		
		double[][] matrix = getRandomMatrix(rows, cols, -1, 1, 0.5, -1);
		double[][] computedMatrixLeft = new double[rows][cols];
		double[][] computedMatrixRight = new double[rows][cols];
		for(int i = 0; i < rows; i++) {
			for(int j = 0; j < cols; j++) {
				computedMatrixLeft[i][j] = fnmod.execute(matrix[i][j], divisor);
				computedMatrixRight[i][j] = fnmod.execute(dividend, matrix[i][j]);
			}
		}
		writeInputMatrix("matrix", matrix);
		writeExpectedMatrix("matrix_left", computedMatrixLeft);
		writeExpectedMatrix("matrix_right", computedMatrixRight);
		
		runTest();
		
		compareResults();
	}	

}
