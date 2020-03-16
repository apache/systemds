/*
 * Copyright 2019 Graz University of Technology
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package org.tugraz.sysds.test.functions.data;

import org.apache.commons.lang.ArrayUtils;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.Parameterized;
import org.junit.runners.Parameterized.Parameters;
import org.tugraz.sysds.api.DMLScript;
import org.tugraz.sysds.common.Types.ExecMode;
import org.tugraz.sysds.lops.LopProperties;
import org.tugraz.sysds.test.AutomatedTestBase;
import org.tugraz.sysds.test.TestConfiguration;

import java.util.Arrays;
import java.util.Collection;

@RunWith(value = Parameterized.class)
public class TensorConstructionTest extends AutomatedTestBase {

	private final static String TEST_DIR = "functions/data/";
	private final static String TEST_NAME = "TensorConstruction";
	private final static String TEST_CLASS_DIR = TEST_DIR + TensorConstructionTest.class.getSimpleName() + "/";

	private String value;
	private long[] dimensions;

	public TensorConstructionTest(long[] dims, String v) {
		dimensions = dims;
		value = v;
	}

	@Parameters
	public static Collection<Object[]> data() {
		Object[][] data = new Object[][]{
				{new long[]{1024, 600, 2}, "3"},
				{new long[]{1, 1}, "8"},
				{new long[]{7, 1, 1}, "0.5"},
				{new long[]{10, 2, 4}, "TRUE"},
				{new long[]{30, 40, 50}, "FALSE"},
				{new long[]{1000, 20}, "0"},
				{new long[]{100, 10, 10, 10, 10}, "1.0"},
				{new long[]{1, 1, 1, 1, 1, 1, 100}, "1"},
		};
		return Arrays.asList(data);
	}

	@Override
	public void setUp() {
		addTestConfiguration(TEST_NAME, new TestConfiguration(TEST_CLASS_DIR, TEST_NAME, new String[]{"A.scalar"}));
	}

	@Test
	public void tensorConstructionTestCP() {
		testTensorConstruction(TEST_NAME, LopProperties.ExecType.CP);
	}

	@Test
	public void tensorConstructionTestSpark() {
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

			long length = Arrays.stream(dimensions).reduce(1, (a, b) -> a * b);
			StringBuilder values = new StringBuilder();
			for (long i = 0; i < length; i++) {
				values.append(i).append(" ");
			}
			fullDMLScriptName = HOME + testName + ".dml";
			StringBuilder dimensionsStringBuilder = new StringBuilder();
			long[] dims = Arrays.copyOf(dimensions, dimensions.length);
			Arrays.stream(dims).forEach((dim) -> dimensionsStringBuilder.append(dim).append(" "));
			String dimensionsString = dimensionsStringBuilder.toString();

			StringBuilder reverseDimsStrBuilder = new StringBuilder();
			ArrayUtils.reverse(dims);
			Arrays.stream(dims).forEach((dim) -> reverseDimsStrBuilder.append(dim).append(" "));
			String reversedDimStr = reverseDimsStrBuilder.toString();

			programArgs = new String[]{"-explain", "-args",
					dimensionsString, Integer.toString(dims.length), value, values.toString(),
					reversedDimStr};

			// TODO check tensors (write not implemented yet, so not possible)
			runTest(true, false, null, -1);
		} finally {
			rtplatform = platformOld;
			DMLScript.USE_LOCAL_SPARK_CONFIG = sparkConfigOld;
		}
	}
}
