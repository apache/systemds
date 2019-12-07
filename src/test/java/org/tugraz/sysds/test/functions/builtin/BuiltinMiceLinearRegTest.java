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

package org.tugraz.sysds.test.functions.builtin;

import org.junit.Test;
import org.tugraz.sysds.common.Types;
import org.tugraz.sysds.lops.LopProperties;
import org.tugraz.sysds.test.AutomatedTestBase;
import org.tugraz.sysds.test.TestConfiguration;

public class BuiltinMiceLinearRegTest extends AutomatedTestBase {
	private final static String TEST_NAME = "mice_lm";
	private final static String TEST_DIR = "functions/builtin/";
	private static final String TEST_CLASS_DIR = TEST_DIR + BuiltinMiceLinearRegTest.class.getSimpleName() + "/";

	private final static int rows = 50;
	private final static int cols = 30;
	private final static int iter = 3;
	private final static int com = 2;

	@Override
	public void setUp() {
		addTestConfiguration(TEST_NAME, new TestConfiguration(TEST_CLASS_DIR, TEST_NAME, new String[]{"B"}));
	}

	@Test
	public void testMatrixSparseCP() {
		runLmTest(0.7, LopProperties.ExecType.CP);
	}

	@Test
	public void testMatrixDenseCP() {
		runLmTest(0.3, LopProperties.ExecType.CP);
	}

	@Test
	public void testMatrixSparseSpark() {
		runLmTest(0.7, LopProperties.ExecType.SPARK);
	}

	@Test
	public void testMatrixDenseSpark() {
		runLmTest(0.3, LopProperties.ExecType.SPARK);
	}

	private void runLmTest(double sparseVal, LopProperties.ExecType instType) {
		Types.ExecMode platformOld = setExecMode(instType);
		try {
			loadTestConfiguration(getTestConfiguration(TEST_NAME));
			String HOME = SCRIPT_DIR + TEST_DIR;
			fullDMLScriptName = HOME + TEST_NAME + ".dml";
			programArgs = new String[]{"-nvargs", "X=" + input("A"), "iteration=" + iter, "com=" + com, "data=" + output("B")};

			//generate actual dataset
			double[][] A = getRandomMatrix(rows, cols, 0, 1, sparseVal, 7);
			writeInputMatrixWithMTD("A", A, true);

			runTest(true, false, null, -1);
		}
		finally {
			rtplatform = platformOld;
		}
	}
}
