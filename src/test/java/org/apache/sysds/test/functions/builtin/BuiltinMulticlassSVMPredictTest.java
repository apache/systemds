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

public class BuiltinMulticlassSVMPredictTest extends AutomatedTestBase {
	private final static String TEST_NAME = "multisvmPredict";
	private final static String TEST_DIR = "functions/builtin/";
	private final static String TEST_CLASS_DIR = TEST_DIR + BuiltinConfusionMatrixTest.class.getSimpleName() + "/";

	@Override
	public void setUp() {
		TestUtils.clearAssertionInformation();
		addTestConfiguration(TEST_NAME, new TestConfiguration(TEST_CLASS_DIR, TEST_NAME, new String[] {"YRaw", "Y"}));
	}

	public double eps = 0.00001;

	@Test
	public void test_01() {

		double[][] x = new double[][] {{0.4, 0.0}, {0.0, 0.2}};
		double[][] w = new double[][] {{1.0, 0.0}, {0.0, 1.0}};

		HashMap<MatrixValue.CellIndex, Double> res_Y = new HashMap<>();
		res_Y.put(new CellIndex(1, 1), 1.0);
		res_Y.put(new CellIndex(2, 1), 2.0);

		HashMap<MatrixValue.CellIndex, Double> res_YRaw = new HashMap<>();
		res_YRaw.put(new CellIndex(1, 1), 0.4);
		res_YRaw.put(new CellIndex(2, 2), 0.2);

		for(LopProperties.ExecType ex : new ExecType[] {LopProperties.ExecType.CP, LopProperties.ExecType.SPARK}) {
			runMSVMPredict(x, w, res_YRaw, res_Y, ex);
		}
	}

	@Test
	public void test_02() {
		double[][] x = new double[][] {{0.4, 0.1}};
		double[][] w = new double[][] {{1.0, 0.5}, {0.2, 1.0}};

		HashMap<MatrixValue.CellIndex, Double> res_Y = new HashMap<>();
		res_Y.put(new CellIndex(1, 1), 1.0);
		
		HashMap<MatrixValue.CellIndex, Double> res_YRaw = new HashMap<>();
		res_YRaw.put(new CellIndex(1, 1), 0.42);
		res_YRaw.put(new CellIndex(1, 2), 0.3);

		for(LopProperties.ExecType ex : new ExecType[] {LopProperties.ExecType.CP, LopProperties.ExecType.SPARK}) {
			runMSVMPredict(x, w, res_YRaw, res_Y, ex);
		}
	}

	@Test
	public void test_03() {
		// Add bios column
		double[][] x = new double[][] {{0.4, 0.1}};
		double[][] w = new double[][] {{1.0, 0.5}, {0.2, 1.0}, {1.0,0.5}};

		HashMap<MatrixValue.CellIndex, Double> res_Y = new HashMap<>();
		res_Y.put(new CellIndex(1, 1), 1.0);

		HashMap<MatrixValue.CellIndex, Double> res_YRaw = new HashMap<>();
		res_YRaw.put(new CellIndex(1, 1), 1.42);
		res_YRaw.put(new CellIndex(1, 2), 0.8);

		for(LopProperties.ExecType ex : new ExecType[] {LopProperties.ExecType.CP, LopProperties.ExecType.SPARK}) {
			runMSVMPredict(x, w, res_YRaw, res_Y, ex);
		}
	}

	private void runMSVMPredict(double[][] x, double[][] w, HashMap<MatrixValue.CellIndex, Double> YRaw,
		HashMap<MatrixValue.CellIndex, Double> Y, ExecType instType) {
		ExecMode platformOld = setExecMode(instType);

		try {
			loadTestConfiguration(getTestConfiguration(TEST_NAME));

			String HOME = SCRIPT_DIR + TEST_DIR;

			fullDMLScriptName = HOME + TEST_NAME + ".dml";
			programArgs = new String[] {"-nvargs", "X=" + input("X"), "W=" + input("W"), "YRaw=" + output("YRaw"),
				"Y=" + output("Y")};
			writeInputMatrixWithMTD("X", x, false);
			writeInputMatrixWithMTD("W", w, false);
			runTest(true, false, null, -1);

			HashMap<MatrixValue.CellIndex, Double> YRaw_res = readDMLMatrixFromHDFS("YRaw");
			HashMap<MatrixValue.CellIndex, Double> Y_res = readDMLMatrixFromHDFS("Y");

			TestUtils.compareMatrices(YRaw_res, YRaw, eps, "DML_Result", "Expected");
			TestUtils.compareMatrices(Y_res, Y, eps, "DML_Result", "Expected");
		}
		finally {
			rtplatform = platformOld;
		}
	}

	@Test
	public void test_invalid_01() {
		// Test if the script fails with input contain incorrect number of columns
		double[][] x = new double[][] {{1, 2, 3}};
		double[][] w = new double[][] {{1, -1}, {-1, 1}};
		runMSVMPredictionExceptionTest(x, w);
	}

	@Test
	public void test_invalid_02() {
		// Test if the script fails with input contain incorrect number of rows vs columns
		double[][] x = new double[][] {{1, 2, 3}};
		double[][] w = new double[][] {{1, -1, 1, 3 ,3}, {-1, 1, 1, 1 ,12}};
		runMSVMPredictionExceptionTest(x, w);
	}

	@Test
	public void test_invalid_03() {
		// Add one column more than the bios column.
		double[][] x = new double[][] {{1, 2, 3}};
		double[][] w = new double[][] {{1.0, 0.5}, {0.2, 1.0}, {1.0,0.5},{1.0,0.5},{1.0,0.5}};
		runMSVMPredictionExceptionTest(x, w);
	}

	private void runMSVMPredictionExceptionTest(double[][] x, double[][] w) {
		ExecMode platformOld = setExecMode(LopProperties.ExecType.CP);

		try {
			loadTestConfiguration(getTestConfiguration(TEST_NAME));
			String HOME = SCRIPT_DIR + TEST_DIR;

			fullDMLScriptName = HOME + TEST_NAME + ".dml";
			programArgs = new String[] {"-nvargs", "X=" + input("X"), "W=" + input("W"), "YRaw=" + output("YRaw"),
				"Y=" + output("Y")};
			writeInputMatrixWithMTD("X", x, false);
			writeInputMatrixWithMTD("W", w, false);

			runTest(DMLScriptException.class);

		}
		finally {
			rtplatform = platformOld;
		}
	}
}
