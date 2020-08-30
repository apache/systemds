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

import org.apache.sysds.common.Types;
import org.apache.sysds.lops.LopProperties;
import org.apache.sysds.test.AutomatedTestBase;
import org.apache.sysds.test.TestConfiguration;
import org.apache.sysds.test.TestUtils;
import org.junit.Assert;
import org.junit.Test;

public class BuiltinBanditTest extends AutomatedTestBase {
	private final static String TEST_NAME = "banditTest";
	private final static String TEST_DIR = "functions/builtin/";
	private final static String TEST_CLASS_DIR = TEST_DIR + BuiltinBanditTest.class.getSimpleName() + "/";

	private final static int rows = 30;
	private final static int cols = 20;
	protected static final String DATA_DIR = "./scripts/staging/pipelines/";
	private final static String DATASET = DATA_DIR+ "airbnb.csv";

	@Override
	public void setUp() {
		addTestConfiguration(TEST_NAME,new TestConfiguration(TEST_CLASS_DIR, TEST_NAME,new String[]{"R"}));
	}

	@Test
	public void testBandit1CP() {
		runBanditTest(50, 5, LopProperties.ExecType.CP);
	}

	@Test
	public void testBandit3CP() {
		runBanditTest(30, 2, LopProperties.ExecType.CP);
	}

	private void runBanditTest(Integer resources, Integer k, LopProperties.ExecType et) {
		Types.ExecMode modeOld = setExecMode(et);
		setOutputBuffering(false);
		try {
			loadTestConfiguration(getTestConfiguration(TEST_NAME));
			String HOME = SCRIPT_DIR + TEST_DIR;

			fullDMLScriptName = HOME + TEST_NAME + ".dml";
			programArgs = new String[] {"-stats", "-args", DATASET,
				String.valueOf(resources), String.valueOf(k), output("R")};
			double[][] X = getRandomMatrix(rows, cols, 0, 1, 1, -1);
			double[][] y = getRandomMatrix(rows, 1, 0, 3, 1, -1);
			writeInputMatrixWithMTD("X", X, true);
			writeInputMatrixWithMTD("y", y, true);
			TestUtils.round(y);
			runTest(true, EXCEPTION_NOT_EXPECTED, null, -1);

			//expected loss smaller than default invocation
			Assert.assertTrue(TestUtils.readDMLBoolean(output("R")));
		}
		finally {
			resetExecMode(modeOld);
		}
	}
}
