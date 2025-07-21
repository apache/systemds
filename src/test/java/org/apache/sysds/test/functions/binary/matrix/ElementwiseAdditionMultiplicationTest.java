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

package org.apache.sysds.test.functions.binary.matrix;

import java.util.Date;

import org.junit.Test;
import org.apache.sysds.test.AutomatedTestBase;
import org.apache.sysds.test.TestConfiguration;


public class ElementwiseAdditionMultiplicationTest extends AutomatedTestBase 
{
	
	private final static String TEST_DIR = "functions/binary/matrix/";
	private final static String TEST_CLASS_DIR = TEST_DIR + ElementwiseAdditionMultiplicationTest.class.getSimpleName() + "/";
	
	@Override
	public void setUp() {

		// positive tests
		addTestConfiguration("Term1", new TestConfiguration(TEST_CLASS_DIR, "ElementwiseAdditionMultiplicationTerm1",
				new String[] { "result" }));
		addTestConfiguration("Term2", new TestConfiguration(TEST_CLASS_DIR, "ElementwiseAdditionMultiplicationTerm2",
				new String[] { "result" }));
		addTestConfiguration("Term3", new TestConfiguration(TEST_CLASS_DIR, "ElementwiseAdditionMultiplicationTerm3",
				new String[] { "result" }));
	}

	@Test
	public void testTerm1() {
		int rows = 5;
		int cols = 4;
		TestConfiguration config = availableTestConfigurations.get("Term1");
		config.addVariable("rows", rows);
		config.addVariable("cols", cols);
		loadTestConfiguration(config);

		cleanupDir(baseDirectory + INPUT_DIR, true);
		double[][] a = getRandomMatrix( rows, cols, 0, 10, 1, new Date().getTime());
		double[][] b = getRandomMatrix(rows, cols, 0, 10, 1, new Date().getTime() + 1);
		double[][] c = getRandomMatrix(rows, cols, 0, 10, 1, new Date().getTime() + 2);
		double[][] d = getRandomMatrix(rows, cols, 0, 10, 1, new Date().getTime() + 4);
		writeInputMatrixWithMTD("a", a, false);
		writeInputMatrixWithMTD("b", b, false);
		writeInputMatrixWithMTD("c", c, false);
		writeInputMatrixWithMTD("d", d, false);
		
		double[][] result = new double[rows][cols];
		for (int i = 0; i < rows; i++) {
			for (int j = 0; j < cols; j++) {
				result[i][j] = a[i][j] * b[i][j] + c[i][j] * d[i][j];
			}
		}

		writeExpectedMatrix("result", result);

		runTest(6);

		compareResults();
	}

	@Test
	public void testTerm2() {
		int rows = 5;
		int cols = 4;
		TestConfiguration config = availableTestConfigurations.get("Term2");
		config.addVariable("rows", rows);
		config.addVariable("cols", cols);
		loadTestConfiguration(config);

		double[][] a = getRandomMatrix(rows, cols, 0, 10, 1, new Date().getTime());
		double[][] b = getRandomMatrix(rows, cols, 0, 10, 1, new Date().getTime() + 1);
		double[][] c = getRandomMatrix(rows, cols, 0, 10, 1, new Date().getTime() + 4);
		writeInputMatrixWithMTD("a", a, false);
		writeInputMatrixWithMTD("b", b, false);
		writeInputMatrixWithMTD("c", c, false);
		
		double[][] result = new double[rows][cols];
		for (int i = 0; i < rows; i++) {
			for (int j = 0; j < cols; j++) {
				result[i][j] = a[i][j] * b[i][j] + a[i][j] * c[i][j];
			}
		}

		writeExpectedMatrix("result", result);

		runTest();

		compareResults();
	}

	@Test
	public void testTerm3() {
		int rows = 5;
		int cols = 4;
		TestConfiguration config = availableTestConfigurations.get("Term3");
		config.addVariable("rows", rows);
		config.addVariable("cols", cols);
		loadTestConfiguration(config);

		double[][] a = getRandomMatrix(rows, cols, 0, 10, 1, new Date().getTime());
		double[][] b = getRandomMatrix(rows, cols, 0, 10, 1, new Date().getTime() + 1);
		double[][] c = getRandomMatrix(rows, cols, 0, 10, 1, new Date().getTime() + 2);
		writeInputMatrixWithMTD("a", a, false);
		writeInputMatrixWithMTD("b", b, false);
		writeInputMatrixWithMTD("c", c, false);
		
		double[][] result = new double[rows][cols];
		for (int i = 0; i < rows; i++) {
			for (int j = 0; j < cols; j++) {
				result[i][j] = a[i][j] * b[i][j] * c[i][j];
			}
		}

		writeExpectedMatrix("result", result);

		runTest();

		compareResults(1e-10);
	}
}
