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

package org.apache.sysds.test.functions.builtin;

import org.apache.sysds.parser.LanguageException;
import org.apache.sysds.test.AutomatedTestBase;
import org.apache.sysds.test.TestConfiguration;
import org.junit.Test;

import java.util.ArrayList;
import java.util.List;

public class BuiltinErrorHandlingTest extends AutomatedTestBase {

	private final static String TEST_NAME1 = "dummy1";
	private final static String TEST_NAME2 = "dummy2";
	private final static String TEST_DIR = "functions/builtin/";
	private static final String TEST_CLASS_DIR = TEST_DIR + BuiltinErrorHandlingTest.class.getSimpleName() + "/";

	@Override
	public void setUp() {
		addTestConfiguration(TEST_NAME1,new TestConfiguration(TEST_CLASS_DIR, TEST_NAME1, new String[]{"B"}));
		addTestConfiguration(TEST_NAME2,new TestConfiguration(TEST_CLASS_DIR, TEST_NAME2, new String[]{"B"}));
	}

	@Test
	public void runDummyTest1Return() {
		runDummytest(TEST_NAME1);
	}
	
	@Test
	public void runDummyTest2Return2() {
		runDummytest(TEST_NAME2);
	}
	
	private void runDummytest(String testname) {
		loadTestConfiguration(getTestConfiguration(testname));
		String HOME = SCRIPT_DIR + TEST_DIR;
		fullDMLScriptName = HOME + testname + ".dml";
		List<String> proArgs = new ArrayList<>();
		proArgs.add("-args");
		proArgs.add(input("X"));
		proArgs.add(output("Y"));
		programArgs = proArgs.toArray(new String[proArgs.size()]);

		double[][] X = getRandomMatrix(10, 5, 0, 1, 0.8, -1);
		writeInputMatrixWithMTD("X", X, true);
		
		runTest(true, true, LanguageException.class, "typo", -1);
	}
}
