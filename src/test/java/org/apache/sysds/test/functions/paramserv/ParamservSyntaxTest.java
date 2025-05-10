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

import org.apache.sysds.parser.LanguageException;
import org.apache.sysds.runtime.DMLRuntimeException;
import org.apache.sysds.test.AutomatedTestBase;
import org.apache.sysds.test.TestConfiguration;
import org.junit.Test;

@net.jcip.annotations.NotThreadSafe
public class ParamservSyntaxTest extends AutomatedTestBase {

	private static final String TEST_NAME1 = "paramserv-all-args";
	private static final String TEST_NAME2 = "paramserv-without-optional-args";
	private static final String TEST_NAME3 = "paramserv-miss-args";
	private static final String TEST_NAME4 = "paramserv-wrong-type-args";
	private static final String TEST_NAME5 = "paramserv-wrong-args";
	private static final String TEST_NAME6 = "paramserv-wrong-args2";
	private static final String TEST_NAME7 = "paramserv-minimum-version";

	private static final String TEST_DIR = "functions/paramserv/";
	private static final String TEST_CLASS_DIR = TEST_DIR + ParamservSyntaxTest.class.getSimpleName() + "/";

	private final String HOME = SCRIPT_DIR + TEST_DIR;

	@Override
	public void setUp() {
		addTestConfiguration(TEST_NAME1, new TestConfiguration(TEST_CLASS_DIR, TEST_NAME1, new String[] {}));
		addTestConfiguration(TEST_NAME2, new TestConfiguration(TEST_CLASS_DIR, TEST_NAME2, new String[] {}));
		addTestConfiguration(TEST_NAME3, new TestConfiguration(TEST_CLASS_DIR, TEST_NAME3, new String[] {}));
		addTestConfiguration(TEST_NAME4, new TestConfiguration(TEST_CLASS_DIR, TEST_NAME4, new String[] {}));
		addTestConfiguration(TEST_NAME5, new TestConfiguration(TEST_CLASS_DIR, TEST_NAME5, new String[] {}));
		addTestConfiguration(TEST_NAME6, new TestConfiguration(TEST_CLASS_DIR, TEST_NAME6, new String[] {}));
		addTestConfiguration(TEST_NAME7, new TestConfiguration(TEST_CLASS_DIR, TEST_NAME7, new String[] {}));
	}

	@Test
	public void testParamservWithAllArgs() {
		runDMLTest(TEST_NAME1, false, null, null);
	}

	@Test
	public void testParamservWithoutOptionalArgs() {
		runDMLTest(TEST_NAME2, false, null, null);
	}

	@Test
	public void testParamservMissArgs() {
		final String errmsg = "Named parameter 'features' missing. Please specify the input.";
		runDMLTest(TEST_NAME3, true, LanguageException.class, errmsg);
	}

	@Test
	public void testParamservWrongTypeArgs() {
		final String errmsg = "Input to PARAMSERV::model must be of type 'LIST'. It should not be of type 'MATRIX'";
		runDMLTest(TEST_NAME4, true, LanguageException.class, errmsg);
	}

	@Test
	public void testParamservWrongArgs() {
		final String errmsg = "Paramserv function: not support update type 'NSP'.";
		runDMLTest(TEST_NAME5, true, DMLRuntimeException.class, errmsg);
	}

	@Test
	public void testParamservWrongArgs2() {
		final String errmsg = "Invalid parameters for PARAMSERV: [modelList, val_featur=X_val]";
		runDMLTest(TEST_NAME6, true, LanguageException.class, errmsg);
	}

	@Test
	public void testParamservMinimumVersion() {
		runDMLTest(TEST_NAME7, false, null, null);
	}

	private void runDMLTest(String testname, boolean exceptionExpected, Class<?> exceptionClass, String errmsg) {
		TestConfiguration config = getTestConfiguration(testname);
		setOutputBuffering(true);
		loadTestConfiguration(config);
		programArgs = new String[] { "-explain" };
		fullDMLScriptName = HOME + testname + ".dml";
		runTest(true, exceptionExpected, exceptionClass, errmsg, -1);
	}
}
