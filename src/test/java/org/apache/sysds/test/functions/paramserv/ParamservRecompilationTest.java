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

package org.apache.sysds.test.functions.paramserv;

import org.junit.Test;
import org.apache.sysds.test.AutomatedTestBase;
import org.apache.sysds.test.TestConfiguration;

public class ParamservRecompilationTest extends AutomatedTestBase {

	private static final String TEST_NAME1 = "paramserv-large-parallelism";

	private static final String TEST_DIR = "functions/paramserv/";
	private static final String TEST_CLASS_DIR = TEST_DIR + ParamservRecompilationTest.class.getSimpleName() + "/";

	private final String HOME = SCRIPT_DIR + TEST_DIR;

	@Override
	public void setUp() {
		addTestConfiguration(TEST_NAME1, new TestConfiguration(TEST_CLASS_DIR, TEST_NAME1, new String[] {}));
	}

	@Test
	public void testParamservLargeParallelism() {
		runDMLTest(TEST_NAME1, false, null, null);
	}

	private void runDMLTest(String testname, boolean exceptionExpected, Class<?> exceptionClass, String errmsg) {
		TestConfiguration config = getTestConfiguration(testname);
		loadTestConfiguration(config);
		programArgs = new String[] { "-explain" };
		fullDMLScriptName = HOME + testname + ".dml";
		runTest(true, exceptionExpected, exceptionClass, errmsg, -1);
	}
}
