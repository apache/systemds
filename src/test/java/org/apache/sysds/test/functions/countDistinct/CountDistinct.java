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

package org.apache.sysds.test.functions.countDistinct;

import static org.junit.Assert.assertTrue;

import org.apache.sysds.common.Types;
import org.apache.sysds.lops.LopProperties;
import org.apache.sysds.test.AutomatedTestBase;
import org.apache.sysds.test.TestConfiguration;
import org.apache.sysds.test.TestUtils;
import org.junit.Test;

public class CountDistinct extends AutomatedTestBase {

	private final static String TEST_NAME = "countDistinct";
	private final static String TEST_DIR = "functions/countDistinct/";
	private static final String TEST_CLASS_DIR = TEST_DIR + CountDistinct.class.getSimpleName() + "/";

	private static String[] esT = new String[] {
			// The different types of Estimators
			"count", // EstimatorType.NUM_DISTINCT_COUNT,
			// EstimatorType.NUM_DISTINCT_KMV,
			// EstimatorType.NUM_DISTINCT_HYPER_LOG_LOG
	};

	@Override
	public void setUp() {
		TestUtils.clearAssertionInformation();
		addTestConfiguration(TEST_NAME, new TestConfiguration(TEST_CLASS_DIR, TEST_NAME, new String[] { "A.scalar" }));
	}

	@Test
	public void testSimple1by1() {
		// test simple 1 by 1.
		LopProperties.ExecType ex = LopProperties.ExecType.CP;
		for (String type : esT) {
			countDistinctTest(1, 1, 1, ex, type);
		}
	}

	@Test
	public void testSmall() {
		LopProperties.ExecType ex = LopProperties.ExecType.CP;
		for (String type : esT) {
			countDistinctTest(50, 50, 50, ex, type);
		}
	}

	@Test
	public void testLarge() {
		LopProperties.ExecType ex = LopProperties.ExecType.CP;
		for (String type : esT) {
			countDistinctTest(1000, 1000, 1000, ex, type);
		}
	}

	private void countDistinctTest(int numberDistinct, int cols, int rows, LopProperties.ExecType instType,
			String type) {
		Types.ExecMode platformOld = setExecMode(instType);
		try {
			loadTestConfiguration(getTestConfiguration(TEST_NAME));
			String HOME = SCRIPT_DIR + TEST_DIR;
			fullDMLScriptName = HOME + TEST_NAME + ".dml";
			String out = output("A");
			System.out.println(out);
			programArgs = new String[] { "-args", String.valueOf(numberDistinct), String.valueOf(rows),
					String.valueOf(cols), out, type };

			runTest(true, false, null, -1);
			writeExpectedScalar("A", numberDistinct);
			compareResults(0.001);
		} catch (Exception e) {
			e.printStackTrace();
			assertTrue("Exception in execution: " + e.getMessage(), false);
		} finally {
			rtplatform = platformOld;
		}
	}
}