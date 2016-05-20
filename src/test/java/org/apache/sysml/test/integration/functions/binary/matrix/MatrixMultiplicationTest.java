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

import java.util.HashMap;

import org.junit.Test;

import org.apache.sysml.api.DMLException;
import org.apache.sysml.runtime.matrix.data.MatrixValue.CellIndex;
import org.apache.sysml.test.integration.AutomatedTestBase;
import org.apache.sysml.test.integration.TestConfiguration;
import org.apache.sysml.test.utils.TestUtils;


public class MatrixMultiplicationTest extends AutomatedTestBase 
{
	
	private static final String TEST_DIR = "functions/binary/matrix/";
	private final static String TEST_CLASS_DIR = TEST_DIR + MatrixMultiplicationTest.class.getSimpleName() + "/";
	
	@Override
	public void setUp() {
		// positive tests
		addTestConfiguration("MatrixMultiplicationTest", 
			new TestConfiguration(TEST_CLASS_DIR, "MatrixMultiplicationTest", new String[] { "c" }));
		addTestConfiguration("WrongDimensionsTest", 
			new TestConfiguration(TEST_CLASS_DIR, "MatrixMultiplicationTest", new String[] { "c" }));
		addTestConfiguration("AMultASpecial1Test", 
			new TestConfiguration(TEST_CLASS_DIR, "AMultASpecial1Test", new String[] { "a" }));
		addTestConfiguration("AMultBSpecial2Test", 
			new TestConfiguration(TEST_CLASS_DIR, "AMultBSpecial2Test", new String[] { "e" }));

		// negative tests
	}

	@Test
	public void testMatrixMultiplication() {
		int m = 20;
		int n = 20;
		int k = 20;

		TestConfiguration config = availableTestConfigurations.get("MatrixMultiplicationTest");
		config.addVariable("m", m);
		config.addVariable("n1", n);
		config.addVariable("n2", n);
		config.addVariable("k", k);

		loadTestConfiguration(config);

		double[][] a = getRandomMatrix(m, n, -1, 1, 1, -1);
		double[][] b = getRandomMatrix(n, k, -1, 1, 1, -1);
		double[][] c = TestUtils.performMatrixMultiplication(a, b);

		writeInputMatrix("a", a);
		writeInputMatrix("b", b);
		writeExpectedMatrix("c", c);

		runTest();

		compareResults(0.00000000001);
	}
	
	@Test
	public void testSparseMatrixMultiplication() {
		int m = 40;
		int n = 10;
		int k = 30;

		TestConfiguration config = availableTestConfigurations.get("MatrixMultiplicationTest");
		config.addVariable("m", m);
		config.addVariable("n1", n);
		config.addVariable("n2", n);
		config.addVariable("k", k);

		loadTestConfiguration(config);

		double[][] a = getRandomMatrix(m, n, -1, 1, 0.1, -1);
		double[][] b = getRandomMatrix(n, k, -1, 1, 0.1, -1);
		double[][] c = TestUtils.performMatrixMultiplication(a, b);

		writeInputMatrix("a", a);
		writeInputMatrix("b", b);
		writeExpectedMatrix("c", c);

		runTest();

		compareResults(0.00000000001);
	}

	@Test
	public void testWrongDimensions() {
		int m = 6;
		int n1 = 8;
		int n2 = 10;
		int k = 12;

		TestConfiguration config = availableTestConfigurations.get("WrongDimensionsTest");
		config.addVariable("m", m);
		config.addVariable("n1", n1);
		config.addVariable("n2", n2);
		config.addVariable("k", k);

		loadTestConfiguration(config);

		createRandomMatrix("a", m, n1, -1, 1, 0.5, -1);
		createRandomMatrix("b", n2, k, -1, 1, 0.5, -1);

		runTest(true, DMLException.class);
	}

	@Test
	public void testAMultASpecial1() {
		int rows = 10;
		int cols = 10;

		TestConfiguration config = availableTestConfigurations.get("AMultASpecial1Test");
		config.addVariable("rows", rows);
		config.addVariable("cols", cols);
		loadTestConfiguration(config);

		double[][] a = createNonRandomMatrixValues(rows, cols);
		writeInputMatrix("a", a);

		a = TestUtils.performMatrixMultiplication(a, a);
		a = TestUtils.performMatrixMultiplication(a, a);

		writeExpectedMatrix("a", a);

		runTest();

		compareResults();
	}

	@Test
	public void testAMultBSpecial2() {
		int rows = 10;
		int cols = 10;

		TestConfiguration config = availableTestConfigurations.get("AMultBSpecial2Test");
		config.addVariable("rows", rows);
		config.addVariable("cols", cols);
		loadTestConfiguration(config);

		double[][] a = createNonRandomMatrixValues(rows, cols);
		writeInputMatrix("a", a);
		double[][] b = createNonRandomMatrixValues(rows, cols);
		writeInputMatrix("b", b);
		double[][] d = createNonRandomMatrixValues(rows, cols);
		writeInputMatrix("d", d);

		double[][] c = TestUtils.performMatrixMultiplication(a, b);
		double[][] e = TestUtils.performMatrixMultiplication(c, d);

		writeExpectedMatrix("e", e);
	
		runTest();
		
		HashMap<CellIndex, Double> hmDMLJ = TestUtils.convert2DDoubleArrayToHashMap(e);
		HashMap<CellIndex, Double> hmDMLE = readDMLMatrixFromHDFS("e");
		TestUtils.compareMatrices(hmDMLJ, hmDMLE, 0, "hmDMLJ","hmDMLE");
		
		TestUtils.displayAssertionBuffer();
	}
}
