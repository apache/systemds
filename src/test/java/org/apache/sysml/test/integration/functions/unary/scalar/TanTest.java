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

package org.apache.sysml.test.integration.functions.unary.scalar;

import org.junit.Test;

import org.apache.sysml.test.integration.AutomatedTestBase;
import org.apache.sysml.test.integration.TestConfiguration;
import org.apache.sysml.test.utils.TestUtils;


/**
 * <p><b>Positive tests:</b></p>
 * <ul>
 * 	<li>positive scalar (int, double)</li>
 * 	<li>negative scalar (int, double)</li>
 * 	<li>random scalar (int, double)</li>
 * </ul>
 * <p><b>Negative tests:</b></p>
 * 
 * 
 */
public class TanTest extends AutomatedTestBase 
{
		
	private static final String TEST_DIR = "functions/unary/scalar/";
	private static final String TEST_CLASS_DIR = TEST_DIR + TanTest.class.getSimpleName() + "/";
	
	@Override
	public void setUp() {
		TestUtils.clearAssertionInformation();
		
		// positive tests
		addTestConfiguration("PositiveTest",
				new TestConfiguration(TEST_CLASS_DIR, "TanTest", new String[] { "int", "double" }));
		addTestConfiguration("NegativeTest",
				new TestConfiguration(TEST_CLASS_DIR, "TanTest", new String[] { "int", "double" }));
		addTestConfiguration("RandomTest",
				new TestConfiguration(TEST_CLASS_DIR, "TanTest", new String[] { "int", "double" }));
		
		// negative tests
	}
	
	@Test
	public void testPositive() {
		int intValue = 5;
		double doubleValue = 5.0;
		
		TestConfiguration config = getTestConfiguration("PositiveTest");
		config.addVariable("int", intValue);
		config.addVariable("double", doubleValue);
		
		loadTestConfiguration(config);
		
		double computedIntValue = Math.tan(intValue);
		double computedDoubleValue = Math.tan(doubleValue);
		
		createHelperMatrix();
		writeExpectedHelperMatrix("int", computedIntValue);
		writeExpectedHelperMatrix("double", computedDoubleValue);
		
		runTest();
		
		compareResults(1e-15);
	}
	
	@Test
	public void testNegative() {
		int intValue = -5;
		double doubleValue = -5.0;
		
		TestConfiguration config = getTestConfiguration("NegativeTest");
		config.addVariable("int", intValue);
		config.addVariable("double", doubleValue);
		
		loadTestConfiguration(config);
		
		double computedIntValue = Math.tan(intValue);
		double computedDoubleValue = Math.tan(doubleValue);
		
		createHelperMatrix();
		writeExpectedHelperMatrix("int", computedIntValue);
		writeExpectedHelperMatrix("double", computedDoubleValue);
		
		runTest();
		
		compareResults(1e-15);
	}
	
	@Test
	public void testRandom() {
		int intValue = TestUtils.getRandomInt();
		double doubleValue = TestUtils.getRandomDouble();
		
		TestConfiguration config = getTestConfiguration("RandomTest");
		config.addVariable("int", intValue);
		config.addVariable("double", doubleValue);
		
		loadTestConfiguration(config);
		
		double computedIntValue = Math.tan(intValue);
		double computedDoubleValue = Math.tan(doubleValue);
		
		createHelperMatrix();
		writeExpectedHelperMatrix("int", computedIntValue);
		writeExpectedHelperMatrix("double", computedDoubleValue);
		
		runTest();
		
		compareResults(1e-15);
	}

}
