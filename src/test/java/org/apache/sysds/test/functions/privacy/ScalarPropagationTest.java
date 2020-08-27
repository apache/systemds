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
import static org.junit.Assert.assertFalse;

import java.util.HashMap;

import org.junit.Test;
import org.apache.sysds.parser.DataExpression;
import org.apache.sysds.runtime.matrix.data.MatrixValue.CellIndex;
import org.apache.sysds.runtime.privacy.PrivacyConstraint;
import org.apache.sysds.runtime.privacy.PrivacyConstraint.PrivacyLevel;
import org.apache.sysds.test.AutomatedTestBase;
import org.apache.sysds.test.TestConfiguration;
import org.apache.sysds.test.TestUtils;
import org.apache.wink.json4j.JSONObject;

public class ScalarPropagationTest extends AutomatedTestBase 
{
	
	private final static String TEST_NAME = "ScalarPropagationTest";
	private final static String TEST_DIR = "functions/privacy/";
	private final static String TEST_CLASS_DIR = TEST_DIR + ScalarPropagationTest.class.getSimpleName() + "/";
	private final static String TEST_CLASS_DIR_2 = TEST_DIR + ScalarPropagationTest.class.getSimpleName() + "2/";

	@Override
	public void setUp() {
		TestUtils.clearAssertionInformation();
		addTestConfiguration(TEST_NAME, new TestConfiguration(TEST_CLASS_DIR, TEST_NAME, new String[] { "scalar" }));
		addTestConfiguration(TEST_NAME+"2", new TestConfiguration(TEST_CLASS_DIR_2, TEST_NAME+"2", new String[] { "scalar" }));
	}
	
	@Test
	public void testCastAndRound() {
		TestConfiguration conf = getAndLoadTestConfiguration(TEST_NAME);

		String HOME = SCRIPT_DIR + TEST_DIR;
		fullDMLScriptName = HOME + conf.getTestScript() + ".dml";
		programArgs = new String[]{"-args", input("A"), output("scalar") };

		double scalar = 10.7;
		double[][] A = {{scalar}};
		writeInputMatrixWithMTD("A", A, true, new PrivacyConstraint(PrivacyLevel.Private));
		
		double roundScalar = Math.round(scalar);

		writeExpectedScalar("scalar", roundScalar);
		
		runTest(true, false, null, -1);
		
		HashMap<CellIndex, Double> map = readDMLScalarFromHDFS("scalar");
		double dmlvalue = map.get(new CellIndex(1,1));
		
		assertEquals("Values mismatch: DMLvalue " + dmlvalue + " != ExpectedValue " + roundScalar, 
			roundScalar, dmlvalue, 0.001);

		String actualPrivacyValue = readDMLMetaDataValueCatchException("scalar", "out/", DataExpression.PRIVACY);
		assertEquals(String.valueOf(PrivacyLevel.Private), actualPrivacyValue);
	}

	@Test
	public void testCastAndMultiplyPrivatePrivate(){
		testCastAndMultiply(PrivacyLevel.Private, PrivacyLevel.Private, PrivacyLevel.Private);
	}

	@Test
	public void testCastAndMultiplyPrivatePrivateAggregation(){
		testCastAndMultiply(PrivacyLevel.Private, PrivacyLevel.PrivateAggregation, PrivacyLevel.Private);
	}

	@Test
	public void testCastAndMultiplyPrivateAggregationPrivate(){
		testCastAndMultiply(PrivacyLevel.PrivateAggregation, PrivacyLevel.Private, PrivacyLevel.Private);
	}

	@Test
	public void testCastAndMultiplyPrivateAggregationPrivateAggregation(){
		testCastAndMultiply(PrivacyLevel.PrivateAggregation, PrivacyLevel.PrivateAggregation, PrivacyLevel.PrivateAggregation);
	}

	@Test
	public void testCastAndMultiplyPrivateNone(){
		testCastAndMultiply(PrivacyLevel.Private, PrivacyLevel.None, PrivacyLevel.Private);
	}

	@Test
	public void testCastAndMultiplyNoneNone(){
		testCastAndMultiply(PrivacyLevel.None, PrivacyLevel.None, PrivacyLevel.None);
	}

	public void testCastAndMultiply(PrivacyLevel privacyLevelA, PrivacyLevel privacyLevelB, PrivacyLevel expectedPrivacyLevel) {
		TestConfiguration conf = getAndLoadTestConfiguration(TEST_NAME+"2");

		String HOME = SCRIPT_DIR + TEST_DIR;
		fullDMLScriptName = HOME + conf.getTestScript()+ ".dml";
		programArgs = new String[]{"-args", input("A"), input("B"), output("scalar") };

		double scalarA = 10.7;
		double scalarB = 20.1;
		writeInputScalar(scalarA, "A", privacyLevelA);
		writeInputScalar(scalarB, "B", privacyLevelB);

		double expectedScalar = scalarA * scalarB;
		writeExpectedScalar("scalar", expectedScalar);
		
		runTest(true, false, null, -1);
		
		HashMap<CellIndex, Double> map = readDMLScalarFromHDFS("scalar");
		double actualScalar = map.get(new CellIndex(1,1));
		
		assertEquals("Values mismatch: DMLvalue " + actualScalar + " != ExpectedValue " + expectedScalar, 
			expectedScalar, actualScalar, 0.001);

		if ( expectedPrivacyLevel != PrivacyLevel.None ){
			String actualPrivacyValue = readDMLMetaDataValueCatchException("scalar", "out/", DataExpression.PRIVACY);
			assertEquals(String.valueOf(expectedPrivacyLevel), actualPrivacyValue);
		} else {
			JSONObject meta = getMetaDataJSON("scalar", "out/");
			assertFalse( "Metadata found for output scalar with privacy constraint set, but input privacy level is none", meta != null && meta.has(DataExpression.PRIVACY) );
		}
	}

	private void writeInputScalar(double value, String name, PrivacyLevel privacyLevel){
		double[][] M = {{value}};
		writeInputMatrixWithMTD(name, M, true, new PrivacyConstraint(privacyLevel));
	}
}
