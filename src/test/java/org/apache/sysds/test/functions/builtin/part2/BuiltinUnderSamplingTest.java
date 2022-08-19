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

package org.apache.sysds.test.functions.builtin.part2;

import org.apache.sysds.common.Types;
import org.apache.sysds.test.AutomatedTestBase;
import org.apache.sysds.test.TestConfiguration;
import org.apache.sysds.test.TestUtils;
import org.junit.Assert;
import org.junit.Test;

public class BuiltinUnderSamplingTest extends AutomatedTestBase {
	private final static String TEST_NAME = "underSamplingTest";
	private final static String TEST_DIR = "functions/builtin/";
	private final static String TEST_CLASS_DIR = TEST_DIR + BuiltinUnderSamplingTest.class.getSimpleName() + "/";

	@Override
	public void setUp() {
		TestUtils.clearAssertionInformation();
		addTestConfiguration(TEST_NAME, new TestConfiguration(TEST_CLASS_DIR, TEST_NAME, new String[] {"B",}));
	}

	@Test
	public void test_CP1() {

		runUnderSamplingTest(0.3, Types.ExecType.CP);

	}

	@Test
	public void test_CP2() {

		runUnderSamplingTest(0.4, Types.ExecType.CP);

	}

	@Test
	public void test_Spark() {
		runUnderSamplingTest(0.4,Types.ExecType.SPARK);
	}

	private void runUnderSamplingTest(double ratio, Types.ExecType instType) {
		Types.ExecMode platformOld = setExecMode(instType);

		try {
			setOutputBuffering(true);

			loadTestConfiguration(getTestConfiguration(TEST_NAME));

			String HOME = SCRIPT_DIR + TEST_DIR;

			fullDMLScriptName = HOME + TEST_NAME + ".dml";
			programArgs = new String[] {"-args", String.valueOf(ratio)};

			String out = runTest(null).toString();
			Assert.assertTrue(out.contains("TRUE"));
		}
		finally {
			rtplatform = platformOld;
		}
	}
}
