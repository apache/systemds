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

import org.apache.sysml.api.DMLException;
import org.apache.sysml.test.integration.AutomatedTestBase;
import org.apache.sysml.test.integration.TestConfiguration;



/**
 * <p><b>Positive tests:</b></p>
 * <ul>
 * 	<li>matrix to vector & matrix 2 vector</li>
 * </ul>
 * <p><b>Negative tests:</b></p>
 * <ul>
 * 	<li>wrong dimensions</li>
 * </ul>
 * 
 * 
 */
public class DiagTest extends AutomatedTestBase 
{

	private static final String TEST_DIR = "functions/unary/matrix/";
	private static final String TEST_CLASS_DIR = TEST_DIR + DiagTest.class.getSimpleName() + "/";
	
	@Override
	public void setUp() {
		
		// positive tests
		addTestConfiguration("DiagTest", new TestConfiguration(TEST_CLASS_DIR, "DiagTest", new String[] { "b", "d" }));
		
		// negative tests
		addTestConfiguration("WrongDimensionsTest", new TestConfiguration(TEST_CLASS_DIR, "DiagSingleTest",
				new String[] { "b" }));
	}
	
	@Test
	public void testDiag() {
		int rowsCols = 10;
		
		TestConfiguration config = availableTestConfigurations.get("DiagTest");
		config.addVariable("rows", rowsCols);
		config.addVariable("cols", rowsCols);
		
		loadTestConfiguration(config);
		
		double[][] a = getRandomMatrix(rowsCols, rowsCols, -1, 1, 0.5, -1);
		writeInputMatrix("a", a);
		
		double[][] b = new double[rowsCols][1];
		for(int i = 0; i < rowsCols; i++) {
			b[i][0] = a[i][i];
		}
		writeExpectedMatrix("b", b);
		
		double[][] c = getRandomMatrix(rowsCols, 1, -1, 1, 0.5, -1);
		writeInputMatrix("c", c);
		
		double[][] d = new double[rowsCols][rowsCols];
		for(int i = 0; i < rowsCols; i++) {
			d[i][i] = c[i][0];
		}
		writeExpectedMatrix("d", d);
		
		runTest();
		
		compareResults();
	}
	
	@Test
	public void testWrongDimensions() {
		int rows = 10;
		int cols = 9;
		
		TestConfiguration config = availableTestConfigurations.get("WrongDimensionsTest");
		config.addVariable("rows", rows);
		config.addVariable("cols", cols);
		
		loadTestConfiguration(config);
		
		createRandomMatrix("a", rows, cols, -1, 1, 0.5, -1);
		
		runTest(true, DMLException.class);
	}

}
