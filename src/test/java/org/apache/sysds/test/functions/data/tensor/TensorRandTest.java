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

package org.apache.sysds.test.functions.data.tensor;

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
public class TensorRandTest extends AutomatedTestBase {

	private final static String TEST_DIR = "functions/data/";
	private final static String TEST_NAME = "RandTensorTest";
	private final static String TEST_CLASS_DIR = TEST_DIR + TensorRandTest.class.getSimpleName() + "/";

	private String min, max;
	private int seed;
	private long[] dimensions;

	public TensorRandTest(long[] dims, String min, String max, int seed) {
		this.dimensions = dims;
		this.min = min;
		this.max = max;
		this.seed = seed;
	}

	@Parameters
	public static Collection<Object[]> data() {
		Object[][] data = new Object[][]{
				{new long[]{3, 4, 5}, "3", "50", 1},
				{new long[]{1, 1}, "8", "100", 10},
				{new long[]{7, 1, 1}, "0.5", "1", 43},
				{new long[]{7, 8, 8}, "0.5", "0.6", 42},
				{new long[]{1030, 600, 2}, "0.5", "0.6", 4},
		};
		return Arrays.asList(data);
	}

	@Override
	public void setUp() {
		addTestConfiguration(TEST_NAME, new TestConfiguration(TEST_CLASS_DIR, TEST_NAME, null));
	}

	@Test
	public void tensorRandTestCP() {
		testTensorConstruction(TEST_NAME, LopProperties.ExecType.CP);
	}

	@Test
	public void tensorRandTestSpark() {
		testTensorConstruction(TEST_NAME, LopProperties.ExecType.SPARK);
	}

	private void testTensorConstruction(String testName, LopProperties.ExecType platform) {
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
			//TODO test correctness
			//assertTrue("the test is not done, needs comparison, of result.", false);
			getAndLoadTestConfiguration(testName);
			String HOME = SCRIPT_DIR + TEST_DIR;

			fullDMLScriptName = HOME + testName + ".dml";
			String dimensionsString = Arrays.toString(dimensions).replace(",", "");
			dimensionsString = dimensionsString.substring(1, dimensionsString.length() - 1);

			programArgs = new String[]{"-explain", "-args", dimensionsString, min, max, Integer.toString(seed)};

			// TODO check tensors (write not implemented yet, so not possible)
			runTest(true, false, null, -1);
		} finally {
			rtplatform = platformOld;
			DMLScript.USE_LOCAL_SPARK_CONFIG = sparkConfigOld;
		}
	}
}
