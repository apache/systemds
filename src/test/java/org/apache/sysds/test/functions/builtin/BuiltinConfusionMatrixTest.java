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

import org.apache.sysds.common.Types.ExecMode;
import org.apache.sysds.lops.LopProperties;
import org.apache.sysds.lops.LopProperties.ExecType;
import org.apache.sysds.runtime.DMLScriptException;
import org.apache.sysds.runtime.matrix.data.MatrixValue;
import org.apache.sysds.runtime.matrix.data.MatrixValue.CellIndex;
import org.apache.sysds.test.AutomatedTestBase;
import org.apache.sysds.test.TestConfiguration;
import org.apache.sysds.test.TestUtils;
import org.junit.Test;

public class BuiltinConfusionMatrixTest extends AutomatedTestBase {
	private final static String TEST_NAME = "confusionMatrix";
	private final static String TEST_DIR = "functions/builtin/";
	private final static String TEST_CLASS_DIR = TEST_DIR + BuiltinConfusionMatrixTest.class.getSimpleName() + "/";

	@Override
	public void setUp() {
		TestUtils.clearAssertionInformation();
		addTestConfiguration(TEST_NAME, new TestConfiguration(TEST_CLASS_DIR, TEST_NAME, new String[] {"B",}));
	}

	public double eps = 0.00001;

	@Test
	public void test_01() {
		double[][] y;
		double[][] p;
		HashMap<MatrixValue.CellIndex, Double> res = new HashMap<>();

		// Classification is 100% accurate in all classes if Y == P.
		y = TestUtils.round(getRandomMatrix(1000, 1, 1, 2, 1.0, 7));
		p = y;

		res.put(new CellIndex(1, 1), 1.0);
		res.put(new CellIndex(2, 2), 1.0);

		for(LopProperties.ExecType ex : new ExecType[] {LopProperties.ExecType.CP, LopProperties.ExecType.SPARK}) {
			runConfusionMatrixTest(y, p, res, ex);
		}
	}

	@Test
	public void test_02() {
		HashMap<MatrixValue.CellIndex, Double> res = new HashMap<>();
		res.put(new CellIndex(2, 2), 1.0);
		runConfusionMatrixTest(new double[][] {{2}}, new double[][] {{2}}, res, LopProperties.ExecType.CP);
	}

	@Test
	public void test_03() {
		HashMap<MatrixValue.CellIndex, Double> res = new HashMap<>();
		res.put(new CellIndex(2, 1), 1.0);
		runConfusionMatrixTest(new double[][] {{1}}, new double[][] {{2}}, res, LopProperties.ExecType.CP);
	}

	@Test
	public void test_04() {
		HashMap<MatrixValue.CellIndex, Double> res = new HashMap<>();
		res.put(new CellIndex(6, 1), 1.0);
		runConfusionMatrixTest(new double[][] {{1}}, new double[][] {{6}}, res, LopProperties.ExecType.CP);
	}

	@Test
	public void test_05() {
		HashMap<MatrixValue.CellIndex, Double> res = new HashMap<>();
		res.put(new CellIndex(1, 9), 1.0);
		runConfusionMatrixTest(new double[][] {{9}}, new double[][] {{1}}, res, LopProperties.ExecType.CP);
	}

	@Test
	public void test_06() {
		HashMap<MatrixValue.CellIndex, Double> res = new HashMap<>();
		double[][] y = new double[][] {{1}, {1}, {1}, {1}};
		double[][] p = new double[][] {{1}, {2}, {3}, {4}};
		res.put(new CellIndex(1, 1), 0.25);
		res.put(new CellIndex(2, 1), 0.25);
		res.put(new CellIndex(3, 1), 0.25);
		res.put(new CellIndex(4, 1), 0.25);
		runConfusionMatrixTest(y, p, res, LopProperties.ExecType.CP);
	}

	@Test
	public void test_07() {
		HashMap<MatrixValue.CellIndex, Double> res = new HashMap<>();
		double[][] y = new double[][] {{1}, {2}, {3}, {4}};
		double[][] p = new double[][] {{1}, {1}, {1}, {1}};
		res.put(new CellIndex(1, 1), 1.0);
		res.put(new CellIndex(1, 2), 1.0);
		res.put(new CellIndex(1, 3), 1.0);
		res.put(new CellIndex(1, 4), 1.0);
		runConfusionMatrixTest(y, p, res, LopProperties.ExecType.CP);
	}

	private void runConfusionMatrixTest(double[][] y, double[][] p, HashMap<MatrixValue.CellIndex, Double> res,
		ExecType instType) {
		ExecMode platformOld = setExecMode(instType);

		try {
			loadTestConfiguration(getTestConfiguration(TEST_NAME));

			String HOME = SCRIPT_DIR + TEST_DIR;

			fullDMLScriptName = HOME + TEST_NAME + ".dml";
			programArgs = new String[] {"-nvargs", "P=" + input("P"), "Y=" + input("Y"), "out_file=" + output("B")};
			writeInputMatrixWithMTD("P", p, false);
			writeInputMatrixWithMTD("Y", y, false);
			runTest(true, false, null, -1);

			HashMap<MatrixValue.CellIndex, Double> dmlResult = readDMLMatrixFromHDFS("B");
			TestUtils.compareMatrices(dmlResult, res, eps, "DML_Result", "Expected");
		}
		finally {
			rtplatform = platformOld;
		}
	}

	// TODO Future, does it make sense to save an empty matrix, since we have ways to make an empty matrix?
	// @Test
	// public void test_invalid_01(){
	// // Test if the script fails with input containing no values.
	// runConfusionMatrixExceptionTest(new double[][]{}, new double[][]{});
	// }

	@Test
	public void test_invalid_02() {
		// Test if the script fails with input contain multiple columns
		runConfusionMatrixExceptionTest(new double[][] {{1, 2}}, new double[][] {{1, 2}});
		runConfusionMatrixExceptionTest(new double[][] {{1}}, new double[][] {{1, 2}});
		runConfusionMatrixExceptionTest(new double[][] {{1, 2}}, new double[][] {{1}});
	}

	@Test
	public void test_invalid_03() {
		// Test if the script fails with input contains different amount of rows
		runConfusionMatrixExceptionTest(new double[][] {{1}, {1}}, new double[][] {{1}});
		runConfusionMatrixExceptionTest(new double[][] {{1}}, new double[][] {{1}, {1}});
	}

	private void runConfusionMatrixExceptionTest(double[][] y, double[][] p) {
		ExecMode platformOld = setExecMode(LopProperties.ExecType.CP);

		try {
			loadTestConfiguration(getTestConfiguration(TEST_NAME));
			String HOME = SCRIPT_DIR + TEST_DIR;

			fullDMLScriptName = HOME + TEST_NAME + ".dml";
			programArgs = new String[] {"-nvargs", "P=" + input("P"), "Y=" + input("Y"), "out_file=" + output("B")};
			writeInputMatrixWithMTD("P", p, false);
			writeInputMatrixWithMTD("Y", y, false);

			runTest(DMLScriptException.class);

		}
		finally {
			rtplatform = platformOld;
		}
	}
}
