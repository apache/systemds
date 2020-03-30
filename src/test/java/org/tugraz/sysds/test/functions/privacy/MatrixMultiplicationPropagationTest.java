/*
 * Copyright 2020 Graz University of Technology
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 */

package org.tugraz.sysds.test.functions.privacy;

import org.junit.Test;
import org.tugraz.sysds.api.DMLException;
import org.tugraz.sysds.test.AutomatedTestBase;
import org.tugraz.sysds.test.TestConfiguration;
import org.tugraz.sysds.test.TestUtils;


public class MatrixMultiplicationPropagationTest extends AutomatedTestBase 
{
	
	private static final String TEST_DIR = "functions/privacy/";
	private final static String TEST_CLASS_DIR = TEST_DIR + MatrixMultiplicationPropagationTest.class.getSimpleName() + "/";
	
	@Override
	public void setUp() {
		addTestConfiguration("MatrixMultiplicationPropagationTest", 
			new TestConfiguration(TEST_CLASS_DIR, "MatrixMultiplicationPropagationTest", new String[] { "c" }));
		addTestConfiguration("WrongDimensionsTest", 
			new TestConfiguration(TEST_CLASS_DIR, "MatrixMultiplicationPropagationTest", new String[] { "c" }));
	}

	@Test
	public void testMatrixMultiplication() {
		int m = 20;
		int n = 20;
		int k = 20;

		TestConfiguration config = availableTestConfigurations.get("MatrixMultiplicationPropagationTest");
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

		TestConfiguration config = availableTestConfigurations.get("MatrixMultiplicationPropagationTest");
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
}
