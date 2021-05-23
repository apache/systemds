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
package org.apache.sysds.test.functions.pipelines;

import org.apache.sysds.common.Types.ExecMode;
import org.apache.sysds.common.Types.ExecType;
import org.apache.sysds.runtime.matrix.data.MatrixValue;
import org.apache.sysds.test.AutomatedTestBase;
import org.apache.sysds.test.TestConfiguration;
import org.apache.sysds.test.TestUtils;
import org.junit.Test;

import java.util.HashMap;

public class BuiltinImageTranslateTest extends AutomatedTestBase {
	private final static String TEST_NAME = "image_translate";
	private final static String TEST_DIR = "functions/builtin/";
	private final static String TEST_CLASS_DIR = TEST_DIR + BuiltinImageTranslateTest.class.getSimpleName() + "/";

	private final static double eps = 1e-10;
	private final static int rows = 3;
	private final static int cols = 3;
	private final static double spSparse = 0.1;
	private final static double spDense = 0.9;
	private final static double offset_x = 34;
	private final static double offset_y = -111;

	@Override
	public void setUp() {
		addTestConfiguration(TEST_NAME, new TestConfiguration(TEST_CLASS_DIR, TEST_NAME, new String[] {"B"}));
	}

	@Test
	public void testImageTranslateMatrixDenseCP() { runImageTranslateTest(false, ExecType.CP); }

	@Test
	public void testImageTranslateMatrixSparseCP() {
		runImageTranslateTest(true, ExecType.CP);
	}

	@Test
	public void testImageTranslateMatrixDenseSP() {
		runImageTranslateTest(false, ExecType.SPARK);
	}

	@Test
	public void testImageTranslateMatrixSparseSP() {
		runImageTranslateTest(false, ExecType.SPARK);
	}

	@Test
	public void testImageTranslatePillow() throws Exception {
		loadTestConfiguration(getTestConfiguration(TEST_NAME));
		final int w = 500, h = 135, out_w = 510, out_h = 125, offset_x = 39, offset_y = -13;
		final double fill_value = 128.0;
		double[][] input = TestUtils.readExpectedResource("ImageTransformInput.csv", h, w);
		double[][] reference = TestUtils.readExpectedResource("ImageTransformTranslated.csv", out_h, out_w);
		String HOME = SCRIPT_DIR + TEST_DIR;
		fullDMLScriptName = HOME + TEST_NAME + ".dml";
		programArgs = new String[] {"-nvargs", "in_file=" + input("A"), "out_file=" + output("B"), "width=" + w,
				"height=" + h, "offset_x=" + offset_x, "offset_y=" + offset_y, "out_w=" + out_w, "out_h=" + out_h,
				"fill_value=" + fill_value};
		writeInputMatrixWithMTD("A", input, true);
		runTest(true, false, null, -1);

		HashMap<MatrixValue.CellIndex, Double> dmlfile = readDMLMatrixFromOutputDir("B");
		double[][] dml_res = TestUtils.convertHashMapToDoubleArray(dmlfile, out_h, out_w);
		TestUtils.compareMatrices(reference, dml_res, eps, "Pillow vs. DML");
	}

	private void runImageTranslateTest(boolean sparse, ExecType instType) {
		ExecMode platformOld = setExecMode(instType);
		disableOutAndExpectedDeletion();

		try {
			loadTestConfiguration(getTestConfiguration(TEST_NAME));
			double sparsity = sparse ? spSparse : spDense;

			String HOME = SCRIPT_DIR + TEST_DIR;
			fullDMLScriptName = HOME + TEST_NAME + ".dml";
			programArgs = new String[] {"-nvargs", "in_file=" + input("A"), "out_file=" + output("B"), "width=" + cols,
				"height=" + rows, "offset_x=" + offset_x, "offset_y=" + offset_y, "out_w=" + cols, "out_h=" + rows};

			fullRScriptName = HOME + TEST_NAME + ".R";
			rCmd = "Rscript" + " " + fullRScriptName + " " + inputDir() + " " + expectedDir() + " " + cols + " " + rows + " " + offset_x + " " + offset_y;

			//generate actual dataset
			double[][] A = getRandomMatrix(rows, cols, 0, 255, sparsity, 7);
			writeInputMatrixWithMTD("A", A, true);

			runTest(true, false, null, -1);
			runRScript(true);

			//compare matrices
			HashMap<MatrixValue.CellIndex, Double> dmlfile = readDMLMatrixFromOutputDir("B");
			HashMap<MatrixValue.CellIndex, Double> rfile = readRMatrixFromExpectedDir("B");
			TestUtils.compareMatrices(dmlfile, rfile, eps, "Stat-DML", "Stat-R");
		}
		finally {
			rtplatform = platformOld;
		}
	}
}
