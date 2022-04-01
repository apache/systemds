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

package org.apache.sysds.test.functions.parfor.misc;

import org.junit.Test;
import org.apache.sysds.test.AutomatedTestBase;
import org.apache.sysds.test.TestConfiguration;

public class ParForEvalBuiltinTest extends AutomatedTestBase 
{
	private final static String TEST_NAME1 = "parfor_eval_local";
	private final static String TEST_NAME2 = "parfor_eval_remote";
	private final static String TEST_NAME3 = "parfor_eval_remote2";
	private final static String TEST_DIR = "functions/parfor/";
	private final static String TEST_CLASS_DIR = TEST_DIR + ParForEvalBuiltinTest.class.getSimpleName() + "/";
	
	private final static int rows = 20;
	private final static int cols = 10;
	
	@Override
	public void setUp() {
		addTestConfiguration(TEST_NAME1, new TestConfiguration(TEST_CLASS_DIR, TEST_NAME1, new String[] {"Rout"}) );
		addTestConfiguration(TEST_NAME2, new TestConfiguration(TEST_CLASS_DIR, TEST_NAME2, new String[] {"Rout"}) );
		addTestConfiguration(TEST_NAME3, new TestConfiguration(TEST_CLASS_DIR, TEST_NAME3, new String[] {"Rout"}) );
	}

	@Test
	public void testParForEvalLocal() {
		runFunctionTest(TEST_NAME1);
	}
	
	@Test
	public void testParForEvalRemote() {
		runFunctionTest(TEST_NAME2);
	}
	
	@Test
	public void testParForEvalRemote2() {
		runFunctionTest(TEST_NAME3);
	}
	
	private void runFunctionTest( String testName ) {
		TestConfiguration config = getTestConfiguration(testName);
		config.addVariable("rows", rows);
		config.addVariable("cols", cols);
		loadTestConfiguration(config);
		
		String HOME = SCRIPT_DIR + TEST_DIR;
		fullDMLScriptName = HOME + testName + ".dml";
		programArgs = new String[]{"-args", 
			Integer.toString(rows), Integer.toString(cols)};

		//run without errors on function loading
		runTest(true, false, null, -1);
	}
}
