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

package org.apache.sysds.test.functions.privacy;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.fail;

import org.apache.sysds.parser.DataExpression;
import org.apache.sysds.runtime.meta.MatrixCharacteristics;
import org.apache.sysds.runtime.privacy.PrivacyConstraint;
import org.apache.sysds.runtime.privacy.PrivacyConstraint.PrivacyLevel;
import org.apache.sysds.test.AutomatedTestBase;
import org.apache.sysds.test.TestConfiguration;
import org.apache.sysds.test.TestUtils;
import org.apache.wink.json4j.JSONException;
import org.junit.Test;

public class MatrixRuntimePropagationTest extends AutomatedTestBase
{
	private static final String TEST_DIR = "functions/privacy/";
	private final static String TEST_CLASS_DIR = TEST_DIR + MatrixMultiplicationPropagationTest.class.getSimpleName() + "/";
	private final int m = 20;
	private final int n = 20;
	private final int k = 20;

	@Override
	public void setUp() {
		addTestConfiguration("MatrixRuntimePropagationTest",
			new TestConfiguration(TEST_CLASS_DIR, "MatrixRuntimePropagationTest", new String[]{"c"}));
	}

	@Test
	public void testRuntimePropagationPrivate() throws JSONException {
		conditionalPropagation(PrivacyLevel.Private);
	}

	@Test
	public void testRuntimePropagationNone() throws JSONException {
		conditionalPropagation(PrivacyLevel.None);
	}

	@Test
	public void testRuntimePropagationPrivateAggregation() throws JSONException {
		conditionalPropagation(PrivacyLevel.PrivateAggregation);
	}
	
	private void conditionalPropagation(PrivacyLevel privacyLevel) throws JSONException {

		TestConfiguration config = availableTestConfigurations.get("MatrixRuntimePropagationTest");
		loadTestConfiguration(config);
		fullDMLScriptName = SCRIPT_DIR + TEST_DIR + config.getTestScript() + ".dml";

		double[][] a = getRandomMatrix(m, n, -1, 1, 1, -1);
		double[][] b = getRandomMatrix(n, k, -1, 1, 1, -1);
		double sum;

		PrivacyConstraint privacyConstraint = new PrivacyConstraint(privacyLevel);
		MatrixCharacteristics dataCharacteristics = new MatrixCharacteristics(m,n,k,k);

		writeInputMatrixWithMTD("a", a, false, dataCharacteristics, privacyConstraint);
		writeInputMatrix("b", b);
		if ( privacyLevel.equals(PrivacyLevel.Private) || privacyLevel.equals(PrivacyLevel.PrivateAggregation) ){
			writeExpectedMatrix("c", a);
			sum = TestUtils.sum(a, m, n) + 1;
		} else {
			writeExpectedMatrix("c", b);
			sum = TestUtils.sum(a, m, n) - 1;
		}

		programArgs = new String[]{"-nvargs",
		"a=" + input("a"), "b=" + input("b"), "c=" + output("c"),
		"m=" + m, "n=" + n, "k=" + k, "s=" + sum };
		
		runTest(true,false,null,-1);

		// Check that the output data is correct
		compareResults(1e-9);

		// Check that the output metadata is correct
		if ( privacyLevel.equals(PrivacyLevel.Private) ) {
			String actualPrivacyValue = readDMLMetaDataValue("c", OUTPUT_DIR, DataExpression.PRIVACY);
			assertEquals(PrivacyLevel.Private.name(), actualPrivacyValue);
		}
		else if ( privacyLevel.equals(PrivacyLevel.PrivateAggregation) ){
			String actualPrivacyValue = readDMLMetaDataValue("c", OUTPUT_DIR, DataExpression.PRIVACY);
			assertEquals(PrivacyLevel.PrivateAggregation.name(), actualPrivacyValue);
		} 
		else {
			// Check that a JSONException is thrown
			// or that privacy level is set to none 
			// because no privacy metadata should be written to c
			// except if the privacy written is set to private
			boolean JSONExceptionThrown = false;
			String actualPrivacyValue = null;
			try{
				actualPrivacyValue = readDMLMetaDataValue("c", OUTPUT_DIR, DataExpression.PRIVACY);
			} catch (JSONException e){
				JSONExceptionThrown = true;
			} catch (Exception e){
				fail("Exception occured, but JSONException was expected. The exception thrown is: " + e.getMessage());
				e.printStackTrace();
			}
			assert(JSONExceptionThrown || (PrivacyLevel.None.name().equals(actualPrivacyValue)));
		}
	}
}
