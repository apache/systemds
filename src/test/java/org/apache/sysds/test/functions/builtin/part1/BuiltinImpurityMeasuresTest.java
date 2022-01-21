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

package org.apache.sysds.test.functions.builtin.part1;

import java.util.HashMap;

import org.apache.sysds.common.Types;
import org.apache.sysds.common.Types.ExecType;
import org.apache.sysds.runtime.matrix.data.MatrixValue;
import org.apache.sysds.test.AutomatedTestBase;
import org.apache.sysds.test.TestConfiguration;
import org.apache.sysds.test.TestUtils;
import org.junit.Test;

public class BuiltinImpurityMeasuresTest extends AutomatedTestBase {
	private final static String TEST_NAME = "impurityMeasures";
	private final static String TEST_DIR = "functions/builtin/";
	private static final String TEST_CLASS_DIR = TEST_DIR + BuiltinImpurityMeasuresTest.class.getSimpleName() + "/";

	private final static double eps = 1e-10;

	@Override
	public void setUp() {
		addTestConfiguration(TEST_NAME, new TestConfiguration(TEST_CLASS_DIR, TEST_NAME, new String[] {"C"}));
	}

	@Test
	public void GiniTest1() {
		double[][] X = {{1, 1}, {2, 2}};
		double[][] Y = {{1}, {0}};
		double[][] R = {{2, 2}};
		HashMap<MatrixValue.CellIndex, Double> expected_m = new HashMap<>();
		expected_m.put(new MatrixValue.CellIndex(1, 1), 0.5);
		expected_m.put(new MatrixValue.CellIndex(1, 2), 0.5);
		String method = "gini";

		runImpurityMeasuresTest(ExecType.SPARK, X, Y, R, method, expected_m);
	}

	@Test
	public void GiniTest2() {
		double[][] X = {{1},{1},{1},{1},{1},{1},{2},{2},{2},{2}};
		double[][] Y = {{0}, {0}, {0}, {0}, {0}, {1}, {1}, {1}, {1}, {1}};
		double[][] R = {{2}};
		HashMap<MatrixValue.CellIndex, Double> expected_m = new HashMap<>();
		expected_m.put(new MatrixValue.CellIndex(1, 1), 0.3333333333);
		String method = "gini";

		runImpurityMeasuresTest(ExecType.SPARK, X, Y, R, method, expected_m);
	}

	@Test
	public void GiniTest3() {
		double[][] X = {{1,1,2,1}, {1,3,1,2}, {2,1,1,2}, {3,2,1,1}, {1,3,2,1}};
		double[][] Y = {{0}, {0}, {1}, {1}, {1}};
		double[][] R = {{3, 3, 2, 2}};
		HashMap<MatrixValue.CellIndex, Double> expected_m = new HashMap<>();
		expected_m.put(new MatrixValue.CellIndex(1, 1), 0.2133333333);
		expected_m.put(new MatrixValue.CellIndex(1, 2), 0.0799999999);
		expected_m.put(new MatrixValue.CellIndex(1, 3), 0.0133333333);
		expected_m.put(new MatrixValue.CellIndex(1, 4), 0.0133333333);
		String method = "gini";

		runImpurityMeasuresTest(ExecType.SPARK, X, Y, R, method, expected_m);
	}

	@Test
	public void GiniWithContinuousValues1() {
		double[][] X = {{1.5}, {12.6}, {3.4}, {14.2}};
		double[][] Y = {{0}, {1}, {0}, {1}};
		double[][] R = {{1}};
		HashMap<MatrixValue.CellIndex, Double> expected_m = new HashMap<>();
		expected_m.put(new MatrixValue.CellIndex(1, 1), 0.5);
		String method = "gini";

		runImpurityMeasuresTest(ExecType.SPARK, X, Y, R, method, expected_m);
	}

	@Test
	public void GiniWithContinuousValues2() {
		double[][] X = {{1.5}, {12.6}, {3.4}, {14.2}};
		double[][] Y = {{0}, {1}, {0}, {1}};
		double[][] R = {{1}};
		HashMap<MatrixValue.CellIndex, Double> expected_m = new HashMap<>();
		expected_m.put(new MatrixValue.CellIndex(1, 1), 0.5);
		String method = "gini";

		runImpurityMeasuresTest(ExecType.SPARK, X, Y, R, method, expected_m);
	}

	// comparing with values from https://planetcalc.com/8421/
	@Test
	public void EntropyTest1() {
		double[][] X = {{1, 1}, {2, 2}};
		double[][] Y = {{1}, {0}};
		double[][] R = {{2, 2}};
		HashMap<MatrixValue.CellIndex, Double> expected_m = new HashMap<>();
		expected_m.put(new MatrixValue.CellIndex(1, 1), 1.0);
		expected_m.put(new MatrixValue.CellIndex(1, 2), 1.0);
		String method = "entropy";

		runImpurityMeasuresTest(ExecType.SPARK, X, Y, R, method, expected_m);
	}

	@Test
	public void EntropyTest2() {
		double[][] X = {{1},{1},{1},{1},{1},{1},{2},{2},{2},{2}};
		double[][] Y = {{0},{0},{0},{0},{0},{1},{1},{1},{1},{1}};
		double[][] R = {{2}};
		HashMap<MatrixValue.CellIndex, Double> expected_m = new HashMap<>();
		expected_m.put(new MatrixValue.CellIndex(1, 1), 0.6099865470);
		String method = "entropy";

		runImpurityMeasuresTest(ExecType.SPARK, X, Y, R, method, expected_m);
	}

	@Test
	public void EntropyTest3() {
		double[][] X = {{1,1,2,1}, {1,3,1,2}, {2,1,1,2}, {3,2,1,1}, {1,3,2,1}};
		double[][] Y = {{0}, {0}, {1}, {1}, {1}};
		double[][] R = {{3, 3, 2, 2}};
		HashMap<MatrixValue.CellIndex, Double> expected_m = new HashMap<>();
		expected_m.put(new MatrixValue.CellIndex(1, 1), 0.4199730940);
		expected_m.put(new MatrixValue.CellIndex(1, 2), 0.1709505945);
		expected_m.put(new MatrixValue.CellIndex(1, 3), 0.0199730940);
		expected_m.put(new MatrixValue.CellIndex(1, 4), 0.0199730940);
		String method = "entropy";

		runImpurityMeasuresTest(ExecType.SPARK, X, Y, R, method, expected_m);
	}

	private void runImpurityMeasuresTest(ExecType exec_type, double[][] X, double[][] Y, double[][] R, String method, HashMap<MatrixValue.CellIndex, Double> expected_m) {
		Types.ExecMode platform_old = setExecMode(exec_type);

		try {
			loadTestConfiguration(getTestConfiguration(TEST_NAME));

			String HOME = SCRIPT_DIR + TEST_DIR;
			fullDMLScriptName = HOME + TEST_NAME + ".dml";
			programArgs = new String[] {"-args", input("X"), input("Y"), input("R"), method, output("impurity_measures")};

			writeInputMatrixWithMTD("X", X, true);
			writeInputMatrixWithMTD("Y", Y, true);
			writeInputMatrixWithMTD("R", R, true);

			runTest(true, false, null, -1);

			HashMap<MatrixValue.CellIndex, Double> actual_measures = readDMLMatrixFromOutputDir("impurity_measures");

			System.out.println(actual_measures);
			System.out.println(expected_m);
			TestUtils.compareMatrices(expected_m, actual_measures, eps, "Expected measures", "Actual measures");
		}
		finally {
			rtplatform = platform_old;
		}
	}
}
