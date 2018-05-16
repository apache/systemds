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

package org.apache.sysml.test.integration.functions.paramserv;

import org.apache.sysml.api.DMLException;
import org.apache.sysml.test.integration.AutomatedTestBase;
import org.apache.sysml.test.integration.TestConfiguration;
import org.junit.Test;

public class ParamservFuncTest extends AutomatedTestBase {

	private static final String TEST_NAME1 = "paramserv-all-args";
	private static final String TEST_NAME2 = "paramserv-without-optional-args";
	private static final String TEST_NAME3 = "paramserv-miss-args";
	private static final String TEST_NAME4 = "paramserv-miss-args2";
	private static final String TEST_NAME5 = "paramserv-wrong-type-args";
	private static final String TEST_NAME6 = "paramserv-wrong-named-args";
	private static final String TEST_NAME7 = "paramserv-wrong-args";

	private static final String TEST_DIR = "functions/paramserv/";
	private static final String TEST_CLASS_DIR = TEST_DIR + ParamservFuncTest.class.getSimpleName() + "/";

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
		runDMLTest(TEST_NAME1, true, false, null, null);
	}

	@Test
	public void testParamservWithoutOptionalArgs() {
		runDMLTest(TEST_NAME2, true, false, null, null);
	}

	@Test
	public void testParamservMissArgs() {
		final String errmsg = "Named parameter 'features' missing. Please specify the input.";
		runDMLTest(TEST_NAME3, true, true, DMLException.class, errmsg);
	}

	@Test
	public void testParamservMissArgs2() {
		final String errmsg = "Parameter 'model' missing. Please specify the input.";
		runDMLTest(TEST_NAME4, true, true, DMLException.class, errmsg);
	}

	@Test
	public void testParamservWrongTypeArgs() {
		final String errmsg = "Input to PARAMSERV::model must be of type 'LIST'. It should not be of type 'MATRIX'";
		runDMLTest(TEST_NAME5, true, true, DMLException.class, errmsg);
	}

	@Test
	public void testParamservWrongNamedArgs() {
		final String errmsg = "Invalid parameters for PARAMSERV: [val_label]";
		runDMLTest(TEST_NAME6, true, true, DMLException.class, errmsg);
	}

	@Test
	public void testParamservWrongArgs() {
		final String errmsg = "Function PARAMSERV does not support value 'NSP' as the 'utype' parameter.";
		runDMLTest(TEST_NAME7, true, true, DMLException.class, errmsg);
	}

	private void runDMLTest(String testname, boolean newWay, boolean exceptionExpected, Class<?> exceptionClass,
			String errmsg) {
		TestConfiguration config = getTestConfiguration(testname);
		loadTestConfiguration(config);
		programArgs = new String[] { "-explain" };
		fullDMLScriptName = HOME + testname + ".dml";
		runTest(newWay, exceptionExpected, exceptionClass, errmsg, -1);
	}
}
