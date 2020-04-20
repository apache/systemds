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
import static org.junit.Assert.fail;

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
	private int m = 20;
	private int n = 20;
	private int k = 20;

	@Override
	public void setUp() {
		TestConfiguration config = new TestConfiguration(TEST_CLASS_DIR, "MatrixMultiplicationPropagationTest", new String[] { "c" });
		config.addVariable("m", m);
		config.addVariable("n1", n);
		config.addVariable("n2", n);
		config.addVariable("k", k);
		addTestConfiguration("MatrixMultiplicationPropagationTest",config);
	}

	@Test
	public void testMatrixMultiplicationPropagation() throws JSONException {
		matrixMultiplicationPropagation(true);
	}

	@Test
	public void testMatrixMultiplicationPropagationFalse() throws JSONException {
		matrixMultiplicationPropagation(false);
	}

	private void matrixMultiplicationPropagation(boolean privacy) throws JSONException {

		TestConfiguration config = availableTestConfigurations.get("MatrixMultiplicationPropagationTest");
		loadTestConfiguration(config);

		double[][] a = getRandomMatrix(m, n, -1, 1, 1, -1);
		double[][] b = getRandomMatrix(n, k, -1, 1, 1, -1);
		double[][] c = TestUtils.performMatrixMultiplication(a, b);
		
		PrivacyConstraint privacyConstraint = new PrivacyConstraint(privacy);
		MatrixCharacteristics dataCharacteristics = new MatrixCharacteristics(m,n,k,k);
		
		writeInputMatrixWithMTD("a", a, false, dataCharacteristics, privacyConstraint);
		writeInputMatrix("b", b);
		writeExpectedMatrix("c", c);

		runTest();

		String actualPrivacyValue = readDMLMetaDataValue("c", OUTPUT_DIR, DataExpression.PRIVACY);
		assertEquals(String.valueOf(privacy), actualPrivacyValue);
	}

	@Test
	public void testMatrixMultiplicationPropagationSecondOperand() throws JSONException {
		matrixMultiplicationPropagationSecondOperand(true);
	}

	@Test
	public void testMatrixMultiplicationPropagationSecondOperandFalse() throws JSONException {
		matrixMultiplicationPropagationSecondOperand(false);
	}

	private void matrixMultiplicationPropagationSecondOperand(boolean privacy) throws JSONException {

		TestConfiguration config = availableTestConfigurations.get("MatrixMultiplicationPropagationTest");
		loadTestConfiguration(config);

		double[][] a = getRandomMatrix(m, n, -1, 1, 1, -1);
		double[][] b = getRandomMatrix(n, k, -1, 1, 1, -1);
		double[][] c = TestUtils.performMatrixMultiplication(a, b);
		
		PrivacyConstraint privacyConstraint = new PrivacyConstraint(privacy);
		MatrixCharacteristics dataCharacteristics = new MatrixCharacteristics(n,k,k,k);
		
		writeInputMatrix("a", a);
		writeInputMatrixWithMTD("b", b, false, dataCharacteristics, privacyConstraint);
		writeExpectedMatrix("c", c);

		runTest();

		String actualPrivacyValue = readDMLMetaDataValue("c", OUTPUT_DIR, DataExpression.PRIVACY);
		assertEquals(String.valueOf(privacy), actualPrivacyValue);
	}

	@Test
	public void testMatrixMultiplicationNoPropagation() {
		matrixMultiplicationNoPropagation();
	}

	private void matrixMultiplicationNoPropagation() {

		TestConfiguration config = availableTestConfigurations.get("MatrixMultiplicationPropagationTest");
		loadTestConfiguration(config);

		double[][] a = getRandomMatrix(m, n, -1, 1, 1, -1);
		double[][] b = getRandomMatrix(n, k, -1, 1, 1, -1);
		double[][] c = TestUtils.performMatrixMultiplication(a, b);
		
		
		writeInputMatrix("a", a);
		writeInputMatrix("b", b);
		writeExpectedMatrix("c", c);

		runTest();

		boolean JSONExceptionThrown = false;
		try{
			readDMLMetaDataValue("c", OUTPUT_DIR, DataExpression.PRIVACY);
		} catch (JSONException e){
			JSONExceptionThrown = true;
		} catch (Exception e){
			fail("Exception occured, but JSONException was expected. The exception thrown is: " + e.getMessage());
			e.printStackTrace();
		}
		assert(JSONExceptionThrown);
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

		TestConfiguration config = availableTestConfigurations.get("MatrixMultiplicationPropagationTest");
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
