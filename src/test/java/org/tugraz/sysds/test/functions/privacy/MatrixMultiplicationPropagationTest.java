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

import static org.junit.Assert.assertEquals;

import org.apache.wink.json4j.JSONException;
import org.junit.Test;
import org.tugraz.sysds.parser.DataExpression;
import org.tugraz.sysds.runtime.meta.MatrixCharacteristics;
import org.tugraz.sysds.runtime.privacy.PrivacyConstraint;
import org.tugraz.sysds.test.AutomatedTestBase;
import org.tugraz.sysds.test.TestConfiguration;
import org.tugraz.sysds.test.TestUtils;

public class MatrixMultiplicationPropagationTest extends AutomatedTestBase {

	private static final String TEST_DIR = "functions/privacy/";
	private final static String TEST_CLASS_DIR = TEST_DIR + MatrixMultiplicationPropagationTest.class.getSimpleName()
			+ "/";

	@Override
	public void setUp() {
		addTestConfiguration("MatrixMultiplicationPropagationTest",
				new TestConfiguration(TEST_CLASS_DIR, "MatrixMultiplicationPropagationTest", new String[] { "c" }));
	}

	@Test
	public void testMatrixMultiplicationPropagation() throws JSONException {
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
		
		PrivacyConstraint privacyConstraint = new PrivacyConstraint(true);
		MatrixCharacteristics dataCharacteristics = new MatrixCharacteristics(m,n,k,k);
		
		writeInputMatrixWithMTD("a", a, false, dataCharacteristics, privacyConstraint);
		writeInputMatrix("b", b);
		writeExpectedMatrix("c", c);

		runTest();

		String actualPrivacyValue = readDMLMetaDataValue("c", OUTPUT_DIR, DataExpression.PRIVACY);
		assertEquals(String.valueOf(true), actualPrivacyValue);
	}


	@Test
	public void testMatrixMultiplicationPrivacyInputTrue() throws JSONException {
		testMatrixMultiplicationPrivacyInput(true);
	}

	@Test
	public void testMatrixMultiplicationPrivacyInputFalse() throws JSONException {
		testMatrixMultiplicationPrivacyInput(false);
	}

	private void testMatrixMultiplicationPrivacyInput(boolean privacy) throws JSONException {
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
		
		PrivacyConstraint privacyConstraint = new PrivacyConstraint();
		privacyConstraint.setPrivacy(privacy);
		MatrixCharacteristics dataCharacteristics = new MatrixCharacteristics(m,n,k,k);
		
		writeInputMatrixWithMTD("a", a, false, dataCharacteristics, privacyConstraint);

		String actualPrivacyValue = readDMLMetaDataValue("a", INPUT_DIR, DataExpression.PRIVACY);
		assertEquals(String.valueOf(privacy), actualPrivacyValue);
	}
}
