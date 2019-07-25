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

package org.tugraz.sysds.test.functions.aggregate;

import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.Parameterized;
import org.junit.runners.Parameterized.Parameters;
import org.tugraz.sysds.common.Types.ExecMode;
import org.tugraz.sysds.test.AutomatedTestBase;
import org.tugraz.sysds.test.TestConfiguration;

import java.util.Arrays;
import java.util.Collection;

@RunWith(value = Parameterized.class)
public class TensorSumTest extends AutomatedTestBase
{
	private final static String TEST_DIR = "functions/aggregate/";
	private final static String TEST_NAME = "TensorSum";
	private final static String TEST_CLASS_DIR = TEST_DIR + TensorSumTest.class.getSimpleName() + "/";

	private double value;
	private int[] dimensions;

	public TensorSumTest(int[] dims, double v) {
		dimensions = dims;
		value = v;
	}
	
	@Parameters
	public static Collection<Object[]> data() {
		Object[][] data = new Object[][] { 
				{new int[]{3, 4, 5}, 3},
				{new int[]{1, 1}, 8},
				{new int[]{7, 1, 1}, 0.5},
				{new int[]{10, 2, 4}, 1},
				{new int[]{1000, 100, 100, 10}, 3},
				{new int[]{10000000, 2}, 8},
				{new int[]{100000, 1, 1000}, 0.5},
				{new int[]{1, 1, 1, 2, 1, 1, 1000}, 1},
				};
		return Arrays.asList(data);
	}
	
	@Override
	public void setUp() {
		addTestConfiguration(TEST_NAME,new TestConfiguration(TEST_CLASS_DIR, TEST_NAME,new String[]{"A.scalar"}));
	}

	@Test
	public void tensorSumTest() {
		ExecMode platformOld = rtplatform;
		try {
			getAndLoadTestConfiguration(TEST_NAME);
			
			String HOME = SCRIPT_DIR + TEST_DIR;

			fullDMLScriptName = HOME + TEST_NAME + ".dml";
			String dimensionsString = Arrays.toString(dimensions).replace("[", "")
					.replace(",", "").replace("]", "");
			programArgs = new String[]{"-explain", "-args",
				Double.toString(value), dimensionsString, output("A") };

			// Generate Data in CP
			rtplatform = ExecMode.SINGLE_NODE;
			writeExpectedScalar("A", Arrays.stream(dimensions).reduce(1, (a, b) -> a*b) * value);

			runTest(true, false, null, -1);

			compareResults();
		}
		finally {
			rtplatform = platformOld;
		}
	}
}
