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

package com.ibm.bi.dml.test.integration.functions.unary.scalar;

import org.junit.Test;

import com.ibm.bi.dml.test.integration.AutomatedTestBase;
import com.ibm.bi.dml.test.integration.TestConfiguration;
import com.ibm.bi.dml.test.utils.TestUtils;



/**
 * <p><b>Positive tests:</b></p>
 * <ul>
 * 	<li>sqrt (int, double)</li>
 * 	<li>random int</li>
 * 	<li>random double</li>
 * </ul>
 * <p><b>Negative tests:</b></p>
 * <ul>
 * 	<li>negative int</li>
 * 	<li>negative double</li>
 * 	<li>random int</li>
 * 	<li>random double</li>
 * </ul>
 * 
 * 
 */
public class SqrtTest extends AutomatedTestBase 
{

	
	private static final String TEST_DIR = "functions/unary/scalar/";
	
	@Override
	public void setUp() {
		
		// positive tests
		addTestConfiguration("PositiveTest",
				new TestConfiguration(TEST_DIR, "SqrtTest", new String[] { "int", "double" }));
		
		// random tests
		addTestConfiguration("RandomIntTest",
				new TestConfiguration(TEST_DIR, "SqrtSingleTest", new String[] { "computed" }));
		addTestConfiguration("RandomDoubleTest",
				new TestConfiguration(TEST_DIR, "SqrtSingleTest", new String[] { "computed" }));
		
		// negative tests
		addTestConfiguration("NegativeIntTest",
				new TestConfiguration(TEST_DIR, "SqrtSingleTest", new String[] { "computed" }));
		addTestConfiguration("NegativeDoubleTest",
				new TestConfiguration(TEST_DIR, "SqrtSingleTest", new String[] { "computed" }));
	}
	
	@Test
	public void testPositive() {
		int intValue = 5;
		double doubleValue = 5.0;
		
		TestConfiguration config = availableTestConfigurations.get("PositiveTest");
		config.addVariable("int", intValue);
		config.addVariable("double", doubleValue);
		
		loadTestConfiguration(config);
		
		double computedIntValue = Math.sqrt(intValue);
		double computedDoubleValue = Math.sqrt(doubleValue);
		
		createHelperMatrix();
		writeExpectedHelperMatrix("int", computedIntValue);
		writeExpectedHelperMatrix("double", computedDoubleValue);
		
		runTest();
		
		compareResults();
	}
	
	@Test
	public void testRandomInt() {
		int intValue = TestUtils.getRandomInt();
		
		TestConfiguration config = availableTestConfigurations.get("RandomIntTest");
		config.addVariable("value", intValue);
		
		loadTestConfiguration(config);
		
		createHelperMatrix();
		
		double computedIntValue = Math.sqrt(intValue);
		writeExpectedHelperMatrix("computed", computedIntValue);
		runTest();
		compareResults();
	}
	
	@Test
	public void testRandomDouble() {
		double doubleValue = TestUtils.getRandomDouble();
		
		TestConfiguration config = availableTestConfigurations.get("RandomDoubleTest");
		config.addVariable("value", doubleValue);
		
		loadTestConfiguration(config);
		
		createHelperMatrix();
		
		double computedDoubleValue = Math.sqrt(doubleValue);
		writeExpectedHelperMatrix("computed", computedDoubleValue);
		runTest();
		compareResults();
	}
	
	@Test
	public void testNegativeInt() {
		int intValue = -5;
		
		TestConfiguration config = availableTestConfigurations.get("NegativeIntTest");
		config.addVariable("value", intValue);
		
		loadTestConfiguration(config);
		
		createHelperMatrix();
		
		runTest(false);
	}
	
	@Test
	public void testNegativeDouble() {
		double doubleValue = -5.0;
		
		TestConfiguration config = availableTestConfigurations.get("NegativeDoubleTest");
		config.addVariable("value", doubleValue);
		
		loadTestConfiguration(config);
		
		createHelperMatrix();
		
		runTest(false);
	}
}
