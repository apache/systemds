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

import java.util.HashMap;

import org.apache.sysds.common.Types;
import org.apache.sysds.lops.LopProperties;
import org.apache.sysds.lops.LopProperties.ExecType;
import org.apache.sysds.runtime.DMLScriptException;
import org.apache.sysds.runtime.matrix.data.MatrixValue;
import org.apache.sysds.test.AutomatedTestBase;
import org.apache.sysds.test.TestConfiguration;
import org.apache.sysds.test.TestUtils;
import org.junit.Test;

public class BuiltinToOneHotTest extends AutomatedTestBase {
	private final static String TEST_NAME = "toOneHot";
	private final static String TEST_DIR = "functions/builtin/";
	private static final String TEST_CLASS_DIR = TEST_DIR + BuiltinToOneHotTest.class.getSimpleName() + "/";

	private final static double eps = 0;
	private final static int rows = 10;
	private final static int cols = 1;
	private final static int numClasses = 10;

	@Override
	public void setUp() {
		addTestConfiguration(TEST_NAME,new TestConfiguration(TEST_CLASS_DIR, TEST_NAME, new String[]{"B"}));
	}

	@Test
	public void runSimpleTest() {
		runToOneHotTest(false, false, LopProperties.ExecType.CP, false);
	}

	@Test
	public void runFailingSimpleTest() {
		runToOneHotTest(false, false, ExecType.CP, true);
	}

	private void runToOneHotTest(boolean scalar, boolean sparse, ExecType instType, boolean shouldFail) {
		Types.ExecMode platformOld = setExecMode(instType);

		try
		{
			loadTestConfiguration(getTestConfiguration(TEST_NAME));

			//generate actual dataset
			double[][] A = TestUtils.round(getRandomMatrix(rows, cols, 1, numClasses, 1, 7));
			int max = -1;
			for(int i = 0; i < rows; i++)
				max = Math.max(max, (int) A[i][0]);
			writeInputMatrixWithMTD("A", A, false);

			// script fails if numClasses provided is smaller than maximum value in A
			int numClassesPassed = shouldFail ? max - 1 : max;

			String HOME = SCRIPT_DIR + TEST_DIR;
			fullDMLScriptName = HOME + TEST_NAME + ".dml";
			programArgs = new String[]{"-args", input("A"),
				String.format("%d", numClassesPassed), output("B") };

			runTest(true, shouldFail, shouldFail ? DMLScriptException.class : null, -1);

			if(!shouldFail) {
				HashMap<MatrixValue.CellIndex, Double> expected = computeExpectedResult(A);
				HashMap<MatrixValue.CellIndex, Double> result = readDMLMatrixFromHDFS("B");
				TestUtils.compareMatrices(result, expected, eps, "Stat-DML", "Stat-Java");
			}
		}
		finally {
			rtplatform = platformOld;
		}
	}

	private static HashMap<MatrixValue.CellIndex, Double> computeExpectedResult(double[][] a) {
		HashMap<MatrixValue.CellIndex, Double> expected = new HashMap<>();
		for(int i = 0; i < a.length; i++) {
			for(int j = 0; j < a[i].length; j++) {
				// indices start with 1 here
				expected.put(new MatrixValue.CellIndex(i + 1, (int) a[i][j]), 1.0);
			}
		}
		return expected;
	}
}
