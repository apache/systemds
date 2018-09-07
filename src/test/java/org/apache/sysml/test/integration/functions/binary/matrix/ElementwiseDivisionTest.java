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

import org.apache.sysml.api.DMLException;
import org.apache.sysml.test.integration.AutomatedTestBase;
import org.apache.sysml.test.integration.TestConfiguration;



public class ElementwiseDivisionTest extends AutomatedTestBase 
{
		
	private final static String TEST_DIR = "functions/binary/matrix/";
	private final static String TEST_CLASS_DIR = TEST_DIR + ElementwiseDivisionTest.class.getSimpleName() + "/";
	
	@Override
	public void setUp() {
		
		// positive tests
		addTestConfiguration("DenseTest",
			new TestConfiguration(TEST_CLASS_DIR, "ElementwiseDivisionTest", new String[] { "c" }));
		addTestConfiguration("SparseTest",
			new TestConfiguration(TEST_CLASS_DIR, "ElementwiseDivisionTest", new String[] { "c" }));
		addTestConfiguration("EmptyTest",
			new TestConfiguration(TEST_CLASS_DIR, "ElementwiseDivisionTest", new String[] { "c" }));
		addTestConfiguration("WrongDimensionLessRowsTest",
			new TestConfiguration(TEST_CLASS_DIR, "ElementwiseDivisionVariableDimensionsTest", new String[] { "c" }));
		addTestConfiguration("WrongDimensionMoreRowsTest",
			new TestConfiguration(TEST_CLASS_DIR, "ElementwiseDivisionVariableDimensionsTest", new String[] { "c" }));
		addTestConfiguration("WrongDimensionLessColsTest",
			new TestConfiguration(TEST_CLASS_DIR, "ElementwiseDivisionVariableDimensionsTest", new String[] { "c" }));
		addTestConfiguration("WrongDimensionMoreColsTest",
			new TestConfiguration(TEST_CLASS_DIR, "ElementwiseDivisionVariableDimensionsTest", new String[] { "c" }));
		addTestConfiguration("WrongDimensionLessRowsLessColsTest",
			new TestConfiguration(TEST_CLASS_DIR, "ElementwiseDivisionVariableDimensionsTest", new String[] { "c" }));
		addTestConfiguration("WrongDimensionMoreRowsMoreColsTest",
			new TestConfiguration(TEST_CLASS_DIR, "ElementwiseDivisionVariableDimensionsTest", new String[] { "c" }));
		addTestConfiguration("WrongDimensionLessRowsMoreColsTest",
			new TestConfiguration(TEST_CLASS_DIR, "ElementwiseDivisionVariableDimensionsTest", new String[] { "c" }));
		addTestConfiguration("WrongDimensionMoreRowsLessColsTest",
			new TestConfiguration(TEST_CLASS_DIR, "ElementwiseDivisionVariableDimensionsTest", new String[] { "c" }));
		addTestConfiguration("DivisionByZeroTest",
			new TestConfiguration(TEST_CLASS_DIR, "ElementwiseDivisionTest", new String[] { "c" }));
		
		// negative tests
	}

	@Test
	public void testDense() {
		int rows = 10;
		int cols = 10;
		
		TestConfiguration config = availableTestConfigurations.get("DenseTest");
		config.addVariable("rows", rows);
		config.addVariable("cols", cols);
		
		loadTestConfiguration(config);
		
		double[][] a = getRandomMatrix(rows, cols, -1, 1, 1, -1);
		double[][] b = getNonZeroRandomMatrix(rows, cols, -1, 1, -1);
		double[][] c = new double[rows][cols];
		for(int i = 0; i < rows; i++) {
			for(int j = 0; j < cols; j++) {
				c[i][j] = a[i][j] / b[i][j];
			}
		}
		
		writeInputMatrix("a", a);
		writeInputMatrix("b", b);
		writeExpectedMatrix("c", c);
		
		runTest();
		
		compareResults();
	}

	@Test
	public void testSparse() {
		int rows = 50;
		int cols = 50;
		
		TestConfiguration config = availableTestConfigurations.get("SparseTest");
		config.addVariable("rows", rows);
		config.addVariable("cols", cols);
		
		loadTestConfiguration(config);
		
		double[][] a = getRandomMatrix(rows, cols, -1, 1, 0.05, -1);
		double[][] b = getNonZeroRandomMatrix(rows, cols, -1, 0.05, -1);
		double[][] c = new double[rows][cols];
		for(int i = 0; i < rows; i++) {
			for(int j = 0; j < cols; j++) {
				c[i][j] = a[i][j] / b[i][j];
			}
		}
		
		writeInputMatrix("a", a);
		writeInputMatrix("b", b);
		writeExpectedMatrix("c", c);
		
		runTest();
		
		compareResults();
	}
	
	@Test
	public void testWrongDimensionsLessRows() {
		int rows1 = 8;
		int cols1 = 10;
		int rows2 = 10;
		int cols2 = 10;
		
		TestConfiguration config = availableTestConfigurations.get("WrongDimensionLessRowsTest");
		config.addVariable("rows1", rows1);
		config.addVariable("cols1", cols1);
		config.addVariable("rows2", rows2);
		config.addVariable("cols2", cols2);
		
		loadTestConfiguration(config);
		
		runTest(true, DMLException.class);
	}
	
	@Test
	public void testWrongDimensionsMoreRows() {
		int rows1 = 12;
		int cols1 = 10;
		int rows2 = 10;
		int cols2 = 10;
		
		TestConfiguration config = availableTestConfigurations.get("WrongDimensionMoreRowsTest");
		config.addVariable("rows1", rows1);
		config.addVariable("cols1", cols1);
		config.addVariable("rows2", rows2);
		config.addVariable("cols2", cols2);
		
		loadTestConfiguration(config);
		
		runTest(true, DMLException.class);
	}
	
	@Test
	public void testWrongDimensionsLessCols() {
		int rows1 = 10;
		int cols1 = 8;
		int rows2 = 10;
		int cols2 = 10;
		
		TestConfiguration config = availableTestConfigurations.get("WrongDimensionLessColsTest");
		config.addVariable("rows1", rows1);
		config.addVariable("cols1", cols1);
		config.addVariable("rows2", rows2);
		config.addVariable("cols2", cols2);
		
		loadTestConfiguration(config);
		
		runTest(true, DMLException.class);
	}
	
	@Test
	public void testWrongDimensionsMoreCols() {
		int rows1 = 10;
		int cols1 = 12;
		int rows2 = 10;
		int cols2 = 10;
		
		TestConfiguration config = availableTestConfigurations.get("WrongDimensionMoreColsTest");
		config.addVariable("rows1", rows1);
		config.addVariable("cols1", cols1);
		config.addVariable("rows2", rows2);
		config.addVariable("cols2", cols2);
		
		loadTestConfiguration(config);
		
		runTest(true, DMLException.class);
	}
	
	@Test
	public void testWrongDimensionsLessRowsLessCols() {
		int rows1 = 8;
		int cols1 = 8;
		int rows2 = 10;
		int cols2 = 10;
		
		TestConfiguration config = availableTestConfigurations.get("WrongDimensionLessRowsLessColsTest");
		config.addVariable("rows1", rows1);
		config.addVariable("cols1", cols1);
		config.addVariable("rows2", rows2);
		config.addVariable("cols2", cols2);
		
		loadTestConfiguration(config);
		
		runTest(true, DMLException.class);
	}
	
	@Test
	public void testWrongDimensionsMoreRowsMoreCols() {
		int rows1 = 12;
		int cols1 = 12;
		int rows2 = 10;
		int cols2 = 10;
		
		TestConfiguration config = availableTestConfigurations.get("WrongDimensionMoreRowsMoreColsTest");
		config.addVariable("rows1", rows1);
		config.addVariable("cols1", cols1);
		config.addVariable("rows2", rows2);
		config.addVariable("cols2", cols2);
		
		loadTestConfiguration(config);
		
		runTest(true, DMLException.class);
	}
	
	@Test
	public void testWrongDimensionsLessRowsMoreCols() {
		int rows1 = 8;
		int cols1 = 12;
		int rows2 = 10;
		int cols2 = 10;
		
		TestConfiguration config = availableTestConfigurations.get("WrongDimensionLessRowsMoreColsTest");
		config.addVariable("rows1", rows1);
		config.addVariable("cols1", cols1);
		config.addVariable("rows2", rows2);
		config.addVariable("cols2", cols2);
		
		loadTestConfiguration(config);
		
		runTest(true, DMLException.class);
	}
	
	@Test
	public void testWrongDimensionsMoreRowsLessCols() {
		int rows1 = 12;
		int cols1 = 8;
		int rows2 = 10;
		int cols2 = 10;
		
		TestConfiguration config = availableTestConfigurations.get("WrongDimensionMoreRowsLessColsTest");
		config.addVariable("rows1", rows1);
		config.addVariable("cols1", cols1);
		config.addVariable("rows2", rows2);
		config.addVariable("cols2", cols2);
		
		loadTestConfiguration(config);
		
		runTest(true, DMLException.class);
	}
	
	@Test
	public void testDivisionByZero() {
		int rows = 10;
		int cols = 10;
		
		TestConfiguration config = availableTestConfigurations.get("DivisionByZeroTest");
		config.addVariable("rows", rows);
		config.addVariable("cols", cols);
		
		loadTestConfiguration(config);
		
		double[][] a = getRandomMatrix(rows, cols, -1, 1, 0.5, -1);
		double[][] b = getRandomMatrix(rows, cols, -1, 1, 0.5, -1);
		double[][] c = new double[rows][cols];
		for(int i = 0; i < rows; i++) {
			for(int j = 0; j < cols; j++) {
				c[i][j] = a[i][j] / b[i][j];
			}
		}
		writeInputMatrix("a", a);
		writeInputMatrix("b", b);
		writeExpectedMatrix("c", c);
		
		runTest();
		
		compareResults();
	}

}
