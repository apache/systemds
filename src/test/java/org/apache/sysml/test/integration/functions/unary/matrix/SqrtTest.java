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

package org.apache.sysml.test.integration.functions.unary.matrix;

import org.junit.Test;

import org.apache.sysml.runtime.DMLRuntimeException;
import org.apache.sysml.test.integration.AutomatedTestBase;
import org.apache.sysml.test.integration.TestConfiguration;
import org.apache.sysml.test.utils.TestUtils;



public class SqrtTest extends AutomatedTestBase 
{
	
	private static String TEST_DIR = "functions/unary/matrix/";
	private static final String TEST_CLASS_DIR = TEST_DIR + SqrtTest.class.getSimpleName() + "/";
	
	@Override
	public void setUp() {
		TestUtils.clearAssertionInformation();
		addTestConfiguration("PositiveTest",
				new TestConfiguration(TEST_CLASS_DIR, "SqrtTest", new String[] { "vector", "matrix" }));
		addTestConfiguration("NegativeVectorTest",
				new TestConfiguration(TEST_CLASS_DIR, "SqrtSingleTest", new String[] { "out" }));
		addTestConfiguration("NegativeMatrixTest",
				new TestConfiguration(TEST_CLASS_DIR, "SqrtSingleTest", new String[] { "out" }));
	}
	
	@Test
	public void testPositive() {
		int rows = 10;
		int cols = 10;
		
		TestConfiguration config = availableTestConfigurations.get("PositiveTest");
		config.addVariable("rows", rows);
		config.addVariable("cols", cols);
		
		loadTestConfiguration(config);
		
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
		
		loadTestConfiguration(config);
		
		createRandomMatrix("in", rows, cols, -1, 0, 1, -1);
		
		runTest(false, DMLRuntimeException.class);
	}
	
	@Test
	public void testNegativeMatrix() {
		int rows = 10;
		int cols = 10;
		
		TestConfiguration config = availableTestConfigurations.get("NegativeMatrixTest");
		config.addVariable("rows", rows);
		config.addVariable("cols", cols);
		
		loadTestConfiguration(config);
		
		createRandomMatrix("in", rows, cols, -1, 0, 1, -1);
		
		runTest(false, DMLRuntimeException.class);
	}
	
}
