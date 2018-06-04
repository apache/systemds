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
	private static final String TEST_NAME4 = "paramserv-wrong-type-args";
	private static final String TEST_NAME5 = "paramserv-wrong-args";
	private static final String TEST_NAME6 = "paramserv-wrong-args2";
	private static final String TEST_NAME7 = "paramserv-nn-bsp-batch";
	private static final String TEST_NAME8 = "paramserv-minimum-version";
	private static final String TEST_NAME9 = "paramserv-worker-failed";
	private static final String TEST_NAME10 = "paramserv-agg-service-failed";
	private static final String TEST_NAME11 = "paramserv-large-parallelism";
	private static final String TEST_NAME12 = "paramserv-wrong-aggregate-func";
	private static final String TEST_NAME13 = "paramserv-nn-asp-batch";
	private static final String TEST_NAME14 = "paramserv-nn-bsp-epoch";
	private static final String TEST_NAME15 = "paramserv-nn-asp-epoch";

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
		addTestConfiguration(TEST_NAME8, new TestConfiguration(TEST_CLASS_DIR, TEST_NAME8, new String[] {}));
		addTestConfiguration(TEST_NAME9, new TestConfiguration(TEST_CLASS_DIR, TEST_NAME9, new String[] {}));
		addTestConfiguration(TEST_NAME10, new TestConfiguration(TEST_CLASS_DIR, TEST_NAME10, new String[] {}));
		addTestConfiguration(TEST_NAME11, new TestConfiguration(TEST_CLASS_DIR, TEST_NAME11, new String[] {}));
		addTestConfiguration(TEST_NAME12, new TestConfiguration(TEST_CLASS_DIR, TEST_NAME12, new String[] {}));
		addTestConfiguration(TEST_NAME13, new TestConfiguration(TEST_CLASS_DIR, TEST_NAME13, new String[] {}));
		addTestConfiguration(TEST_NAME14, new TestConfiguration(TEST_CLASS_DIR, TEST_NAME14, new String[] {}));
		addTestConfiguration(TEST_NAME15, new TestConfiguration(TEST_CLASS_DIR, TEST_NAME15, new String[] {}));
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
		runDMLTest(TEST_NAME3, true, DMLException.class, errmsg);
	}

	@Test
	public void testParamservWrongTypeArgs() {
		final String errmsg = "Input to PARAMSERV::model must be of type 'LIST'. It should not be of type 'MATRIX'";
		runDMLTest(TEST_NAME4, true, DMLException.class, errmsg);
	}

	@Test
	public void testParamservWrongArgs() {
		final String errmsg = "Paramserv function: not support update type 'NSP'.";
		runDMLTest(TEST_NAME5, true, DMLException.class, errmsg);
	}

	@Test
	public void testParamservWrongArgs2() {
		final String errmsg = "Invalid parameters for PARAMSERV: [modelList, val_featur=X_val]";
		runDMLTest(TEST_NAME6, true, DMLException.class, errmsg);
	}

	@Test
	public void testParamservNNBspBatchTest() {
		runDMLTest(TEST_NAME7, false, null, null);
	}

	@Test
	public void testParamservMinimumVersionTest() {
		runDMLTest(TEST_NAME8, false, null, null);
	}

	@Test
	public void testParamservWorkerFailedTest() {
		runDMLTest(TEST_NAME9, true, DMLException.class, "Invalid lookup by name in unnamed list: worker_err.");
	}

	@Test
	public void testParamservAggServiceFailedTest() {
		runDMLTest(TEST_NAME10, true, DMLException.class, "Invalid lookup by name in unnamed list: agg_service_err");
	}

	@Test
	public void testParamservLargeParallelismTest() {
		runDMLTest(TEST_NAME11, false, null, null);
	}

	@Test
	public void testParamservWrongAggregateFuncTest() {
		runDMLTest(TEST_NAME12, true, DMLException.class,
				"The 'gradients' function should provide an input of 'MATRIX' type named 'labels'.");
	}

	@Test
	public void testParamservASPTest() {
		runDMLTest(TEST_NAME13, false, null, null);
	}

	@Test
	public void testParamservBSPEpochTest() {
		runDMLTest(TEST_NAME14, false, null, null);
	}

	@Test
	public void testParamservASPEpochTest() {
		runDMLTest(TEST_NAME15, false, null, null);
	}

	private void runDMLTest(String testname, boolean exceptionExpected, Class<?> exceptionClass, String errmsg) {
		TestConfiguration config = getTestConfiguration(testname);
		loadTestConfiguration(config);
		programArgs = new String[] { "-explain" };
		fullDMLScriptName = HOME + testname + ".dml";
		runTest(true, exceptionExpected, exceptionClass, errmsg, -1);
	}
}
