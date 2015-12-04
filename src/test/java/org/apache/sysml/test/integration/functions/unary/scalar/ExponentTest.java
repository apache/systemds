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

import org.apache.sysml.api.DMLException;
import org.apache.sysml.test.integration.AutomatedTestBase;
import org.apache.sysml.test.integration.TestConfiguration;
import org.apache.sysml.test.utils.TestUtils;



/**
 * <p><b>Positive tests:</b></p>
 * <ul>
 * 	<li>constant (int, double)</li>
 * 	<li>variable (int, double)</li>
 * 	<li>random constant (int, double)</li>
 * 	<li>random variable (int, double)</li>
 * 	<li>negative constant (int, double)</li>
 * </ul>
 * <p><b>Negative tests:</b></p>
 * <ul>
 * 	<li>two parameters</li>
 * </ul>
 * 
 * 
 */
public class ExponentTest extends AutomatedTestBase 
{
		
	private static final String TEST_DIR = "functions/unary/scalar/";
	
	@Override
	public void setUp() {
		// positive tests
		addTestConfiguration("ConstTest", new TestConfiguration(TEST_DIR, "ExponentTest",
				new String[] { "int", "double" }));
		addTestConfiguration("VarTest", new TestConfiguration(TEST_DIR, "ExponentTest",
				new String[] { "int", "double" }));
		addTestConfiguration("RandomConstTest", new TestConfiguration(TEST_DIR, "ExponentTest",
				new String[] { "int", "double" }));
		addTestConfiguration("RandomVarTest", new TestConfiguration(TEST_DIR, "ExponentTest",
				new String[] { "int", "double" }));
		addTestConfiguration("NegativeTest", new TestConfiguration(TEST_DIR, "ExponentTest",
				new String[] { "int", "double" }));
		
		// negative tests
		addTestConfiguration("TwoParametersTest", new TestConfiguration(TEST_DIR, "ExponentBinaryTest",
				new String[] { "computed" }));
	}
	
	@Test
	public void testConst() {
		int intValue = 2;
		double doubleValue = 2.5;
		
		TestConfiguration config = availableTestConfigurations.get("ConstTest");
		config.addVariable("intvardeclaration", "");
		config.addVariable("intop", intValue);
		config.addVariable("doubledeclaration", "");
		config.addVariable("doubleop", doubleValue);
		
		loadTestConfiguration(config);
		
		double computedIntValue = Math.exp(intValue);
		double computedDoubleValue = Math.exp(doubleValue);
		
		createHelperMatrix();
		writeExpectedHelperMatrix("int", computedIntValue);
		writeExpectedHelperMatrix("double", computedDoubleValue);
		
		runTest();
		
		compareResults();
	}
	
	@Test
	public void testVar() {
		int intValue = 2;
		double doubleValue = 2.5;
		
		TestConfiguration config = availableTestConfigurations.get("VarTest");
		config.addVariable("intvardeclaration", "intValue = " + intValue + ";");
		config.addVariable("intop", "intValue");
		config.addVariable("doublevardeclaration", "doubleValue = " +
				TestUtils.getStringRepresentationForDouble(doubleValue) + ";");
		config.addVariable("doubleop", "doubleValue");
		
		loadTestConfiguration(config);
		
		double computedIntValue = Math.exp(intValue);
		double computedDoubleValue = Math.exp(doubleValue);
		
		createHelperMatrix();
		writeExpectedHelperMatrix("int", computedIntValue);
		writeExpectedHelperMatrix("double", computedDoubleValue);
		
		runTest();
		
		compareResults();
	}
	
	@Test
	public void testRandomConst() {
		int intValue = TestUtils.getRandomInt();
		double doubleValue = TestUtils.getRandomDouble();
		
		TestConfiguration config = availableTestConfigurations.get("RandomConstTest");
		config.addVariable("intvardeclaration", "");
		config.addVariable("intop", intValue);
		config.addVariable("doubledeclaration", "");
		config.addVariable("doubleop", doubleValue);
		
		loadTestConfiguration(config);
		
		double computedIntValue = Math.exp(intValue);
		double computedDoubleValue = Math.exp(doubleValue);
		
		createHelperMatrix();
		writeExpectedHelperMatrix("int", computedIntValue);
		writeExpectedHelperMatrix("double", computedDoubleValue);
		
		runTest();
		
		compareResults();
	}
	
	@Test
	public void testRandomVar() {
		int intValue = TestUtils.getRandomInt();
		double doubleValue = TestUtils.getRandomDouble();
		
		TestConfiguration config = availableTestConfigurations.get("RandomVarTest");
		config.addVariable("intvardeclaration", "intValue = " + intValue + ";");
		config.addVariable("intop", "intValue");
		config.addVariable("doublevardeclaration", "doubleValue = " +
				TestUtils.getStringRepresentationForDouble(doubleValue) + ";");
		config.addVariable("doubleop", "doubleValue");
		
		loadTestConfiguration(config);
		
		double computedIntValue = Math.exp(intValue);
		double computedDoubleValue = Math.exp(doubleValue);
		
		createHelperMatrix();
		writeExpectedHelperMatrix("int", computedIntValue);
		writeExpectedHelperMatrix("double", computedDoubleValue);
		
		runTest();
		
		compareResults();
	}
	
	@Test
	public void testNegative() {
		int intValue = -2;
		double doubleValue = -2.5;
		
		TestConfiguration config = availableTestConfigurations.get("NegativeTest");
		config.addVariable("intvardeclaration", "");
		config.addVariable("intop", intValue);
		config.addVariable("doublevardeclaration", "");
		config.addVariable("doubleop", doubleValue);
		
		loadTestConfiguration(config);
		
		double computedIntValue = Math.exp(intValue);
		double computedDoubleValue = Math.exp(doubleValue);
		
		createHelperMatrix();
		writeExpectedHelperMatrix("int", computedIntValue);
		writeExpectedHelperMatrix("double", computedDoubleValue);
		
		runTest();
		
		compareResults();
	}
	
	@Test
	public void testTwoParameters() {
		TestConfiguration config = availableTestConfigurations.get("TwoParametersTest");
		config.addVariable("vardeclaration", "");
		config.addVariable("op1", 1);
		config.addVariable("op2", 2);
		
		loadTestConfiguration(config);
		
		createHelperMatrix();
		
		runTest(true, DMLException.class);
	}
	
}
