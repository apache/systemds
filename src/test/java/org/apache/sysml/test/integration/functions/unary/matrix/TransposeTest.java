/**
 * (C) Copyright IBM Corp. 2010, 2015
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * 
 */

package org.apache.sysml.test.integration.functions.unary.matrix;

import org.junit.Test;

import org.apache.sysml.test.integration.AutomatedTestBase;
import org.apache.sysml.test.integration.TestConfiguration;
import org.apache.sysml.test.utils.TestUtils;



public class TransposeTest extends AutomatedTestBase 
{

	private static final String TEST_DIR = "functions/unary/matrix/";
	private static final String TEST_CLASS_DIR = TEST_DIR + TransposeTest.class.getSimpleName() + "/";
	
	@Override
	public void setUp() {
		TestUtils.clearAssertionInformation();
		addTestConfiguration("TransposeTest",
				new TestConfiguration(TEST_CLASS_DIR,"TransposeTest", new String[] { "vector", "matrix" }));
	}
	
	@Test
	public void testTranspose() {
		int rows = 10;
		int cols = 10;
		
		TestConfiguration config = getTestConfiguration("TransposeTest");
		config.addVariable("rows", rows);
		config.addVariable("cols", cols);
		
		loadTestConfiguration(config);
		
		double[][] vector = getRandomMatrix(rows, 1, 0, 1, 1, -1);
		double[][] transposedVector = new double[1][rows];
		for(int i = 0; i < rows; i++) {
			transposedVector[0][i] = vector[i][0];
		}
		writeInputMatrix("vector", vector);
		writeExpectedMatrix("vector", transposedVector);
		
		double[][] matrix = getRandomMatrix(rows, cols, 0, 1, 1, -1);
		double[][] transposedMatrix = new double[cols][rows];
		for(int i = 0; i < rows; i++) {
			for(int j = 0; j < cols; j++) {
				transposedMatrix[j][i] = matrix[i][j];
			}
		}
		writeInputMatrix("matrix", matrix);
		writeExpectedMatrix("matrix", transposedMatrix);
		
		runTest();		
		compareResults();
	}
	
}
