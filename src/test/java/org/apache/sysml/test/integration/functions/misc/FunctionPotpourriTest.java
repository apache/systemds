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

package org.apache.sysml.test.integration.functions.misc;


import org.junit.Test;
import org.apache.sysml.api.DMLException;
import org.apache.sysml.test.integration.AutomatedTestBase;
import org.apache.sysml.test.integration.TestConfiguration;
import org.apache.sysml.test.utils.TestUtils;

public class FunctionPotpourriTest extends AutomatedTestBase 
{
	private final static String TEST_NAME1 = "FunPotpourriNoReturn";
	private final static String TEST_NAME2 = "FunPotpourriComments";
	
	private final static String TEST_DIR = "functions/misc/";
	private final static String TEST_CLASS_DIR = TEST_DIR + FunctionPotpourriTest.class.getSimpleName() + "/";
	
	@Override
	public void setUp() {
		TestUtils.clearAssertionInformation();
		addTestConfiguration( TEST_NAME1, new TestConfiguration(TEST_CLASS_DIR, TEST_NAME1, new String[] { "R" }) );
		addTestConfiguration( TEST_NAME2, new TestConfiguration(TEST_CLASS_DIR, TEST_NAME2, new String[] { "R" }) );
	}

	@Test
	public void testFunctionNoReturn() {
		runFunctionTest( TEST_NAME1, false );
	}
	
	@Test
	public void testFunctionComments() {
		runFunctionTest( TEST_NAME2, false );
	}
	
	private void runFunctionTest(String testName, boolean error) {
		TestConfiguration config = getTestConfiguration(testName);
		loadTestConfiguration(config);
		
		String HOME = SCRIPT_DIR + TEST_DIR;
		fullDMLScriptName = HOME + testName + ".dml";
		programArgs = new String[]{"-explain", "-stats"};

		//run script and compare output
		runTest(true, error, DMLException.class, -1); 
	}
}
