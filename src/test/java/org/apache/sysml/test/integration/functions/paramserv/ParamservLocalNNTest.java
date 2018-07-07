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

import org.apache.sysml.test.integration.AutomatedTestBase;
import org.apache.sysml.test.integration.TestConfiguration;
import org.junit.Test;

public class ParamservLocalNNTest extends AutomatedTestBase {

	private static final String TEST_NAME1 = "paramserv-nn-bsp-batch-dc";
	private static final String TEST_NAME2 = "paramserv-nn-asp-batch";
	private static final String TEST_NAME3 = "paramserv-nn-bsp-epoch";
	private static final String TEST_NAME4 = "paramserv-nn-asp-epoch";
	private static final String TEST_NAME5 = "paramserv-nn-bsp-batch-drr";
	private static final String TEST_NAME6 = "paramserv-nn-bsp-batch-dr";
	private static final String TEST_NAME7 = "paramserv-nn-bsp-batch-or";

	private static final String TEST_DIR = "functions/paramserv/";
	private static final String TEST_CLASS_DIR = TEST_DIR + ParamservLocalNNTest.class.getSimpleName() + "/";

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
	public void testParamservBSPBatchDisjointContiguous() {
		runDMLTest(TEST_NAME1);
	}

	@Test
	public void testParamservASPBatch() {
		runDMLTest(TEST_NAME2);
	}

	@Test
	public void testParamservBSPEpoch() {
		runDMLTest(TEST_NAME3);
	}

	@Test
	public void testParamservASPEpoch() {
		runDMLTest(TEST_NAME4);
	}

	@Test
	public void testParamservBSPBatchDisjointRoundRobin() {
		runDMLTest(TEST_NAME5);
	}

	@Test
	public void testParamservBSPBatchDisjointRandom() {
		runDMLTest(TEST_NAME6);
	}

	@Test
	public void testParamservBSPBatchOverlapReshuffle() {
		runDMLTest(TEST_NAME7);
	}

	private void runDMLTest(String testname) {
		TestConfiguration config = getTestConfiguration(testname);
		loadTestConfiguration(config);
		programArgs = new String[] { "-explain" };
		fullDMLScriptName = HOME + testname + ".dml";
		runTest(true, false, null, null, -1);
	}
}
