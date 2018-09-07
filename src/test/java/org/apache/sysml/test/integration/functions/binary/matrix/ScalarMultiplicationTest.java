/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 * 
 *   http://www.apache.org/licenses/LICENSE-2.0
 * 
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

package org.apache.sysml.test.integration.functions.binary.matrix;

import org.junit.Test;

import org.apache.sysml.test.integration.AutomatedTestBase;
import org.apache.sysml.test.integration.TestConfiguration;
import org.apache.sysml.test.utils.TestUtils;


public class ScalarMultiplicationTest extends AutomatedTestBase 
{
	
	private static final String TEST_DIR = "functions/binary/matrix/";
	private static final String TEST_CLASS_DIR = TEST_DIR + ScalarMultiplicationTest.class.getSimpleName() + "/";
	
	@Override
	public void setUp() {
		TestUtils.clearAssertionInformation();
		
		// positive tests
		addTestConfiguration("IntConstTest", new TestConfiguration(TEST_CLASS_DIR,"ScalarMultiplicationTest",
				new String[] { "vector_left", "vector_right", "matrix_left", "matrix_right" }));
		addTestConfiguration("IntVarTest", new TestConfiguration(TEST_CLASS_DIR,"ScalarMultiplicationTest",
				new String[] { "vector_left", "vector_right", "matrix_left", "matrix_right" }));
		addTestConfiguration("DoubleConstTest", new TestConfiguration(TEST_CLASS_DIR,"ScalarMultiplicationTest",
				new String[] { "vector_left", "vector_right", "matrix_left", "matrix_right" }));
		addTestConfiguration("DoubleVarTest", new TestConfiguration(TEST_CLASS_DIR,"ScalarMultiplicationTest",
				new String[] { "vector_left", "vector_right", "matrix_left", "matrix_right" }));
		addTestConfiguration("SparseTest", new TestConfiguration(TEST_CLASS_DIR,"ScalarMultiplicationTest",
				new String[] { "vector_left", "vector_right", "matrix_left", "matrix_right" }));
		addTestConfiguration("EmptyTest", new TestConfiguration(TEST_CLASS_DIR,"ScalarMultiplicationTest",
				new String[] { "vector_left", "vector_right", "matrix_left", "matrix_right" }));
		
		// negative tests
	}
	
	@Test
	public void testIntConst() {
		int rows = 10;
		int cols = 10;
		int factor = 2;
		
		TestConfiguration config = availableTestConfigurations.get("IntConstTest");
		config.addVariable("rows", rows);
		config.addVariable("cols", cols);
		config.addVariable("vardeclaration", "");
		config.addVariable("factor", factor);
		
		loadTestConfiguration(config);
		
		double[][] vector = getRandomMatrix(rows, 1, 0, 1, 1, -1);
		double[][] computedVector = new double[rows][1];
		for(int i = 0; i < rows; i++) {
			computedVector[i][0] = vector[i][0] * factor;
		}
		writeInputMatrix("vector", vector);
		writeExpectedMatrix("vector_left", computedVector);
		writeExpectedMatrix("vector_right", computedVector);
		
		double[][] matrix = getRandomMatrix(rows, cols, 0, 1, 1, -1);
		double[][] computedMatrix = new double[rows][cols];
		for(int i = 0; i < rows; i++) {
			for(int j = 0; j < cols; j++) {
				computedMatrix[i][j] = matrix[i][j] * factor;
			}
		}
		writeInputMatrix("matrix", matrix);
		writeExpectedMatrix("matrix_left", computedMatrix);
		writeExpectedMatrix("matrix_right", computedMatrix);
		
		runTest();
		
		compareResults();
	}
	
	@Test
	public void testIntVar() {
		int rows = 10;
		int cols = 10;
		int factor = 2;
		
		TestConfiguration config = availableTestConfigurations.get("IntVarTest");
		config.addVariable("rows", rows);
		config.addVariable("cols", cols);
		config.addVariable("vardeclaration", "Factor = " + factor);
		config.addVariable("factor", "Factor");
		
		loadTestConfiguration(config);
		
		double[][] vector = getRandomMatrix(rows, 1, 0, 1, 1, -1);
		double[][] computedVector = new double[rows][1];
		for(int i = 0; i < rows; i++) {
			computedVector[i][0] = vector[i][0] * factor;
		}
		writeInputMatrix("vector", vector);
		writeExpectedMatrix("vector_left", computedVector);
		writeExpectedMatrix("vector_right", computedVector);
		
		double[][] matrix = getRandomMatrix(rows, cols, 0, 1, 1, -1);
		double[][] computedMatrix = new double[rows][cols];
		for(int i = 0; i < rows; i++) {
			for(int j = 0; j < cols; j++) {
				computedMatrix[i][j] = matrix[i][j] * factor;
			}
		}
		writeInputMatrix("matrix", matrix);
		writeExpectedMatrix("matrix_left", computedMatrix);
		writeExpectedMatrix("matrix_right", computedMatrix);
		
		runTest();
		
		compareResults();
	}
	
	@Test
	public void testDoubleConst() {
		int rows = 10;
		int cols = 10;
		double factor = 2;
		
		TestConfiguration config = availableTestConfigurations.get("DoubleConstTest");
		config.addVariable("rows", rows);
		config.addVariable("cols", cols);
		config.addVariable("vardeclaration", "");
		config.addVariable("factor", factor);
		
		loadTestConfiguration(config);
		
		double[][] vector = getRandomMatrix(rows, 1, 0, 1, 1, -1);
		double[][] computedVector = new double[rows][1];
		for(int i = 0; i < rows; i++) {
			computedVector[i][0] = vector[i][0] * factor;
		}
		writeInputMatrix("vector", vector);
		writeExpectedMatrix("vector_left", computedVector);
		writeExpectedMatrix("vector_right", computedVector);
		
		double[][] matrix = getRandomMatrix(rows, cols, 0, 1, 1, -1);
		double[][] computedMatrix = new double[rows][cols];
		for(int i = 0; i < rows; i++) {
			for(int j = 0; j < cols; j++) {
				computedMatrix[i][j] = matrix[i][j] * factor;
			}
		}
		writeInputMatrix("matrix", matrix);
		writeExpectedMatrix("matrix_left", computedMatrix);
		writeExpectedMatrix("matrix_right", computedMatrix);
		
		runTest();
		
		compareResults();
	}
	
	@Test
	public void testDoubleVar() {
		int rows = 10;
		int cols = 10;
		double factor = 2;
		
		TestConfiguration config = availableTestConfigurations.get("DoubleVarTest");
		config.addVariable("rows", rows);
		config.addVariable("cols", cols);
		config.addVariable("vardeclaration", "Factor = " + factor);
		config.addVariable("factor", "Factor");
		
		loadTestConfiguration(config);
		
		double[][] vector = getRandomMatrix(rows, 1, 0, 1, 1, -1);
		double[][] computedVector = new double[rows][1];
		for(int i = 0; i < rows; i++) {
			computedVector[i][0] = vector[i][0] * factor;
		}
		writeInputMatrix("vector", vector);
		writeExpectedMatrix("vector_left", computedVector);
		writeExpectedMatrix("vector_right", computedVector);
		
		double[][] matrix = getRandomMatrix(rows, cols, 0, 1, 1, -1);
		double[][] computedMatrix = new double[rows][cols];
		for(int i = 0; i < rows; i++) {
			for(int j = 0; j < cols; j++) {
				computedMatrix[i][j] = matrix[i][j] * factor;
			}
		}
		writeInputMatrix("matrix", matrix);
		writeExpectedMatrix("matrix_left", computedMatrix);
		writeExpectedMatrix("matrix_right", computedMatrix);
		
		runTest();
		
		compareResults();
	}
	
	@Test
	public void testSparse() {
		int rows = 100;
		int cols = 50;
		int factor = 2;
		
		TestConfiguration config = availableTestConfigurations.get("SparseTest");
		config.addVariable("rows", rows);
		config.addVariable("cols", cols);
		config.addVariable("vardeclaration", "");
		config.addVariable("factor", factor);
		
		loadTestConfiguration(config);
		
		double[][] vector = getRandomMatrix(rows, 1, -1, 1, 0.05, -1);
		double[][] computedVector = new double[rows][1];
		for(int i = 0; i < rows; i++) {
			computedVector[i][0] = vector[i][0] * factor;
		}
		writeInputMatrix("vector", vector);
		writeExpectedMatrix("vector_left", computedVector);
		writeExpectedMatrix("vector_right", computedVector);
		
		double[][] matrix = getRandomMatrix(rows, cols, -1, 1, 0.05, -1);
		double[][] computedMatrix = new double[rows][cols];
		for(int i = 0; i < rows; i++) {
			for(int j = 0; j < cols; j++) {
				computedMatrix[i][j] = matrix[i][j] * factor;
			}
		}
		writeInputMatrix("matrix", matrix);
		writeExpectedMatrix("matrix_left", computedMatrix);
		writeExpectedMatrix("matrix_right", computedMatrix);
		
		runTest();
		
		compareResults();
	}
	
}
