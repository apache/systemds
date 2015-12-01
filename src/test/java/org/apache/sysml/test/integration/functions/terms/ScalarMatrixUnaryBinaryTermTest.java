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

package com.ibm.bi.dml.test.integration.functions.terms;

import org.junit.Test;

import com.ibm.bi.dml.test.integration.AutomatedTestBase;
import com.ibm.bi.dml.test.integration.TestConfiguration;
import com.ibm.bi.dml.test.utils.TestUtils;


public class ScalarMatrixUnaryBinaryTermTest extends AutomatedTestBase 
{
	private static final String TEST_DIR = "functions/terms/";
	
	@Override
	public void setUp() {
		addTestConfiguration("TestTerm1", new TestConfiguration(TEST_DIR, "TestTerm1", new String[] {}));
	}

	@Test
	public void testTerm1() {
		int rows = 5, cols = 5;

		TestConfiguration config = getTestConfiguration("TestTerm1");
		config.addVariable("rows", rows);
		config.addVariable("cols", cols);

		loadTestConfiguration(config);

		double[][] a = createNonRandomMatrixValues(rows, cols);
		writeInputMatrix("a", a);

		double[][] w = new double[rows][cols];
		for (int i = 0; i < rows; i++) {
			for (int j = 0; j < cols; j++) {
				w[i][j] = 1 + a[i][j];
			}
		}
		w = TestUtils.performMatrixMultiplication(w, w);
		writeExpectedMatrix("w", w);

		runTest();

		compareResults();
	}
}
