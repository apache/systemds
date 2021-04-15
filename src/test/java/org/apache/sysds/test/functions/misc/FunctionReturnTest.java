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

package org.apache.sysds.test.functions.misc;

import org.junit.Test;
import org.apache.sysds.hops.OptimizerUtils;
import org.apache.sysds.test.AutomatedTestBase;
import org.apache.sysds.test.TestConfiguration;

public class FunctionReturnTest extends AutomatedTestBase 
{
	private final static String TEST_DIR = "functions/misc/";
	private final static String TEST_NAME1 = "FunctionReturnInt";
	private final static String TEST_NAME2 = "FunctionReturnBoolean";
	private final static String TEST_CLASS_DIR = TEST_DIR + FunctionReturnTest.class.getSimpleName() + "/";
	
	@Override
	public void setUp() {
		addTestConfiguration( TEST_NAME1, new TestConfiguration(TEST_CLASS_DIR, TEST_NAME1, new String[] { "Rout" }) );
		addTestConfiguration( TEST_NAME2, new TestConfiguration(TEST_CLASS_DIR, TEST_NAME2, new String[] { "Rout" }) );
	}

	@Test
	public void testFunctionReturnInt() {
		runFunctionReturnTest(TEST_NAME1, false);
	}
	
	@Test
	public void testFunctionReturnBool() {
		runFunctionReturnTest(TEST_NAME2, false);
	}
	
	@Test
	public void testFunctionReturnIntIPA() {
		runFunctionReturnTest(TEST_NAME1, true);
	}
	
	@Test
	public void testFunctionReturnBoolIPA() {
		runFunctionReturnTest(TEST_NAME2, true);
	}

	private void runFunctionReturnTest( String testname, boolean IPA ) {
		boolean oldIPA = OptimizerUtils.ALLOW_INTER_PROCEDURAL_ANALYSIS;
		OptimizerUtils.ALLOW_INTER_PROCEDURAL_ANALYSIS = IPA;
		try {
			TestConfiguration config = getTestConfiguration(testname);
			loadTestConfiguration(config);
			
			String HOME = SCRIPT_DIR + TEST_DIR;
			fullDMLScriptName = HOME + testname + ".dml";
			programArgs = new String[]{};
	
			runTest(true, false, null, -1); 
		}
		finally {
			OptimizerUtils.ALLOW_INTER_PROCEDURAL_ANALYSIS = oldIPA;
		}
	}
}
