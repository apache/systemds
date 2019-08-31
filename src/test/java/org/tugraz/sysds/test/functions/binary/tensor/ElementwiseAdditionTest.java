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
 *
 */

package org.tugraz.sysds.test.functions.binary.tensor;

import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.Parameterized;
import org.junit.runners.Parameterized.Parameters;
import org.tugraz.sysds.api.DMLScript;
import org.tugraz.sysds.common.Types.ExecMode;
import org.tugraz.sysds.hops.BinaryOp;
import org.tugraz.sysds.lops.LopProperties;
import org.tugraz.sysds.test.AutomatedTestBase;
import org.tugraz.sysds.test.TestConfiguration;

import java.util.Arrays;
import java.util.Collection;

@RunWith(value = Parameterized.class)
public class ElementwiseAdditionTest extends AutomatedTestBase {
	private final static String TEST_DIR = "functions/binary/tensor/";
	private final static String TEST_NAME = "ElementwiseAdditionTest";
	private final static String TEST_CLASS_DIR = TEST_DIR + ElementwiseAdditionTest.class.getSimpleName() + "/";

	private String _lvalue, _rvalue;
	private int[] _dimsLeft, _dimsRight;

	public ElementwiseAdditionTest(int[] dimsLeft, int[] dimsRight, String lv, String rv) {
		_dimsLeft = dimsLeft;
		_dimsRight = dimsRight;
		_lvalue = lv;
		_rvalue = rv;
	}

	@Parameters
	public static Collection<Object[]> data() {
		Object[][] data = new Object[][]{
			{new int[]{3, 4, 5}, new int[]{3, 4, 5}, "3", "-2"},
			{new int[]{1, 1, 1, 1, 1}, new int[]{1, 1, 1, 1, 1}, "2", "30000000000.0"},
			{new int[]{4000, 4000}, new int[]{4000, 4000}, "3.0", "-2.0"},
			{new int[]{4000, 4000}, new int[]{4000, 1}, "3.0", "-2.0"},
			{new int[]{4000, 4000}, new int[]{1, 1}, "3.0", "-2"},
		};
		return Arrays.asList(data);
	}

	@Override
	public void setUp() {
		addTestConfiguration(TEST_NAME, new TestConfiguration(TEST_CLASS_DIR, TEST_NAME, new String[]{"A.scalar"}));
	}

	@Test
	public void elementwiseAdditionTestCP() {
		testElementwiseAddition(TEST_NAME, LopProperties.ExecType.CP);
	}

	@Test
	public void elementwiseAdditionTestSpark() {
		BinaryOp.FORCED_BINARY_METHOD = null;
		testElementwiseAddition(TEST_NAME, LopProperties.ExecType.SPARK);
	}

	@Test
	public void elementwiseAdditionTestBroadcastSpark() {
		BinaryOp.FORCED_BINARY_METHOD = BinaryOp.MMBinaryMethod.MR_BINARY_M;
		testElementwiseAddition(TEST_NAME, LopProperties.ExecType.SPARK);
	}

	private void testElementwiseAddition(String testName, LopProperties.ExecType platform) {
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
			String ldimString = Arrays.toString(_dimsLeft).replace("[", "")
					.replace(",", "").replace("]", "");
			String rdimString = Arrays.toString(_dimsRight).replace("[", "")
					.replace(",", "").replace("]", "");
			programArgs = new String[]{"-explain", "-args",
					ldimString, rdimString, Integer.toString(_dimsLeft.length), _lvalue, _rvalue, output("A")};

			runTest(true, false, null, -1);
			//TODO test correctness
		}
		finally {
			rtplatform = platformOld;
			DMLScript.USE_LOCAL_SPARK_CONFIG = sparkConfigOld;
		}
	}
}
