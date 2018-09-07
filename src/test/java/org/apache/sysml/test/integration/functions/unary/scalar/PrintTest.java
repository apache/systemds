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


/**
 * Tests the print function
 */
public class PrintTest extends AutomatedTestBase 
{
	
	private static final String TEST_DIR = "functions/unary/scalar/";
	private static final String TEST_CLASS_DIR = TEST_DIR + PrintTest.class.getSimpleName() + "/";
	
	@Override
	public void setUp() {
		
		addTestConfiguration("PrintTest", new TestConfiguration(TEST_CLASS_DIR, "PrintTest", new String[] {}));
		addTestConfiguration("PrintTest2", new TestConfiguration(TEST_CLASS_DIR, "PrintTest2", new String[] {}));
		addTestConfiguration("PrintTest3", new TestConfiguration(TEST_CLASS_DIR, "PrintTest3", new String[] {}));
	}

	@Test
	public void testInt() {
		int value = 0;

		TestConfiguration config = availableTestConfigurations.get("PrintTest");
		config.addVariable("value", value);

		loadTestConfiguration(config);

		setExpectedStdOut("X= " + value);
		runTest();
	}
	
	@Test
	public void testDouble() {
		double value = 1337.3;

		TestConfiguration config = availableTestConfigurations.get("PrintTest");
		config.addVariable("value", value);

		loadTestConfiguration(config);
		
		setExpectedStdOut("X= " + value);
		runTest();
	}
	
	@Test
	public void testBoolean() {
		String value = "TRUE";

		TestConfiguration config = availableTestConfigurations.get("PrintTest");
		config.addVariable("value", value);

		loadTestConfiguration(config);

		setExpectedStdOut("X= " + value);
		runTest();
	}
	
	@Test
	public void testString() {
		String value = "\"Hello World!\"";

		TestConfiguration config = availableTestConfigurations.get("PrintTest");
		config.addVariable("value", value);

		loadTestConfiguration(config);

		setExpectedStdOut("X= " + value.substring(1, value.length()-1));
		runTest();
	}
	
	@Test
	public void testStringWithoutMsg() {
		String value = "\"Hello World!\"";

		TestConfiguration config = availableTestConfigurations.get("PrintTest2");
		config.addVariable("value", value);

		loadTestConfiguration(config);

		setExpectedStdOut(value.substring(1, value.length()-1));
		runTest();
	}

	@Test
	public void testPrint() {
		TestConfiguration config = availableTestConfigurations.get("PrintTest3");

		loadTestConfiguration(config);

		String value = "fooboo, 0.0";
		setExpectedStdOut(value.substring(1, value.length()-1));
		runTest();
	}
}
