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

import java.util.Arrays;
import java.util.Collection;
import java.util.HashMap;

import org.apache.sysds.common.Types.ExecMode;
import org.apache.sysds.common.Types.ExecType;
import org.apache.sysds.runtime.matrix.data.MatrixValue;
import org.apache.sysds.test.AutomatedTestBase;
import org.apache.sysds.test.TestConfiguration;
import org.apache.sysds.test.TestUtils;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.Parameterized;

@RunWith(Parameterized.class)
@net.jcip.annotations.NotThreadSafe

public class BuiltinImageCutoutLinTest extends AutomatedTestBase {
	private final static String TEST_NAME = "image_cutout_linearized";
	private final static String TEST_DIR = "functions/builtin/";
	private final static String TEST_CLASS_DIR = TEST_DIR + BuiltinImageCutoutLinTest.class.getSimpleName() + "/";

	private final static double eps = 1e-10;
	private final static double spSparse = 0.1;
	private final static double spDense = 0.9;

	@Parameterized.Parameter(0)
	public int s_rows;
	@Parameterized.Parameter(1)
	public int s_cols;
	@Parameterized.Parameter(2)
	public int x;
	@Parameterized.Parameter(3)
	public int y;
	@Parameterized.Parameter(4)
	public int width;
	@Parameterized.Parameter(5)
	public int height;
	@Parameterized.Parameter(6)
	public int fill_color;
	@Parameterized.Parameter(7)
	public int n_imgs;

	@Parameterized.Parameters
	public static Collection<Object[]> data() {
		return Arrays.asList(new Object[][] {
			{12, 12, 7, 5, 6, 2, 0, 512}, 
			{13, 11, 10, 7, 2, 3, 32, 175},
			{32, 32, 2, 11, 1, 60, 64, 4}, 
			{64, 64, 50, 17, 10, 109, 96, 5}, 
			{64, 61, 33, 20, 30, 10, 128, 32},
			{128, 128, 2, 3, 2, 9, 192, 5}, 
			{123, 128, 83, 70, 50, 2, 225, 12},});
	}

	@Override
	public void setUp() {
		addTestConfiguration(TEST_NAME, new TestConfiguration(TEST_CLASS_DIR, TEST_NAME, new String[] {"B"}));
	}

	@Test
	public void testImageTranslateMatrixDenseCP() {
		runImageCutoutLinTest(false, ExecType.CP);
	}

	@Test
	public void testImageTranslateMatrixSparseCP() {
		runImageCutoutLinTest(true, ExecType.CP);
	}

	@Test
	public void testImageTranslateMatrixDenseSP() {
		runImageCutoutLinTest(false, ExecType.SPARK);
	}

	@Test
	public void testImageTranslateMatrixSparseSP() {
		runImageCutoutLinTest(false, ExecType.SPARK);
	}

	private void runImageCutoutLinTest(boolean sparse, ExecType instType) {
		ExecMode platformOld = setExecMode(instType);
		disableOutAndExpectedDeletion();

		setOutputBuffering(true);

		try {
			loadTestConfiguration(getTestConfiguration(TEST_NAME));
			double sparsity = sparse ? spSparse : spDense;

			String HOME = SCRIPT_DIR + TEST_DIR;
			fullDMLScriptName = HOME + TEST_NAME + ".dml";
			programArgs = new String[] {"-nvargs", "in_file=" + input("A"), "out_file=" + output("B"),
				"width=" + (s_cols * s_rows), "height=" + n_imgs, "x=" + (x + 1), "y=" + (y + 1), "w=" + width,
				"h=" + height, "fill_color=" + fill_color, "s_cols=" + s_cols, "s_rows=" + s_rows};

			double[][] A = getRandomMatrix(n_imgs, s_cols * s_rows, 0, 255, sparsity, 7);
			writeInputMatrixWithMTD("A", A, true);

			double[][] ref = new double[n_imgs][s_cols * s_rows];
			for(int i = 0; i < n_imgs; i++) {
				for(int j = 0; j < s_cols * s_rows; j++) {
					ref[i][j] = A[i][j];
					if(y <= (int) Math.floor(j / s_cols) && (int) Math.floor(j / s_cols) < y + height && x <= (j % s_cols) &&
						(j % s_cols) < x + width) {
						ref[i][j] = fill_color;
					}
				}
			}

			runTest(true, false, null, -1);

			HashMap<MatrixValue.CellIndex, Double> dmlfile = readDMLMatrixFromOutputDir("B");
			double[][] dml_res = TestUtils.convertHashMapToDoubleArray(dmlfile, n_imgs, (s_cols * s_rows));

			TestUtils.compareMatrices(ref, dml_res, eps, "Java vs. DML");

		}
		finally {
			rtplatform = platformOld;
		}
	}
}
