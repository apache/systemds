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

package org.apache.sysds.test.functions.aggregate;

import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.Parameterized;
import org.junit.runners.Parameterized.Parameters;
import org.apache.sysds.api.DMLScript;
import org.apache.sysds.common.Types.ExecMode;
import org.apache.sysds.lops.LopProperties;
import org.apache.sysds.test.AutomatedTestBase;
import org.apache.sysds.test.TestConfiguration;

import java.util.Arrays;
import java.util.Collection;

@RunWith(value = Parameterized.class)
@net.jcip.annotations.NotThreadSafe
public class TensorSumTest extends AutomatedTestBase
{
	private final static String TEST_DIR = "functions/aggregate/";
	private final static String TEST_NAME = "TensorSum";
	private final static String TEST_CLASS_DIR = TEST_DIR + TensorSumTest.class.getSimpleName() + "/";

	private String value;
	private int[] dimensions;

	public TensorSumTest(int[] dims, String v) {
		dimensions = dims;
		value = v;
	}
	
	@Parameters
	public static Collection<Object[]> data() {
		Object[][] data = new Object[][] {
				{new int[]{3, 4, 5}, "3"},
				{new int[]{1, 1}, "8"},
				{new int[]{7, 1, 1}, "0.5"},
				{new int[]{10, 2, 4}, "1"},
				{new int[]{1003, 5, 50, 10}, "3"},
				{new int[]{10000, 2}, "8"},
				{new int[]{1020, 1, 30}, "0.5"},
				{new int[]{1, 1, 1, 2, 1, 1, 1000}, "1"},
				};
		return Arrays.asList(data);
	}
	
	@Override
	public void setUp() {
		addTestConfiguration(TEST_NAME,new TestConfiguration(TEST_CLASS_DIR, TEST_NAME,new String[]{"A.scalar"}));
	}

	@Test
	public void tensorSumTestCP() {
		testTensorSum(TEST_NAME, LopProperties.ExecType.CP);
	}

	// Sp instructions not supported for tensors.
	// TODO: make support for spark
	// @Test
	// public void tensorSumTestSpark() {
	// 	testTensorSum(TEST_NAME, LopProperties.ExecType.SPARK);
	// }

	private void testTensorSum(String testName, LopProperties.ExecType platform) {
		ExecMode platformOld = rtplatform;
		if (platform == LopProperties.ExecType.SPARK) {
			rtplatform = ExecMode.SPARK;
		}
		else {
			rtplatform = ExecMode.SINGLE_NODE;
		}

		boolean sparkConfigOld = DMLScript.USE_LOCAL_SPARK_CONFIG;
		if (rtplatform == ExecMode.SPARK) {
			DMLScript.USE_LOCAL_SPARK_CONFIG = true;
		}
		try {
			getAndLoadTestConfiguration(TEST_NAME);

			String HOME = SCRIPT_DIR + TEST_DIR;

			fullDMLScriptName = HOME + TEST_NAME + ".dml";
			String dimensionsString = Arrays.toString(dimensions).replace("[", "")
					.replace(",", "").replace("]", "");
			programArgs = new String[]{"-explain", "-args",
					value, dimensionsString, output("A")};

			try {
				writeExpectedScalar("A", Arrays.stream(dimensions).reduce(1, (a, b) -> a * b) * Long.parseLong(value));
			}
			catch (NumberFormatException e) {
				writeExpectedScalar("A", Arrays.stream(dimensions).reduce(1, (a, b) -> a * b) * Double.parseDouble(value));
			}

			runTest(true, false, null, -1);

			compareResults();
		}
		finally {
			rtplatform = platformOld;
			DMLScript.USE_LOCAL_SPARK_CONFIG = sparkConfigOld;
		}
	}
}
