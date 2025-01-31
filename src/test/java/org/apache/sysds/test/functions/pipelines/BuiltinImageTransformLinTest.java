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

import java.util.Arrays;
import java.util.Collection;
import java.util.HashMap;

import org.apache.sysds.common.Types.ExecMode;
import org.apache.sysds.common.Types.ExecType;
import org.apache.sysds.runtime.matrix.data.MatrixValue;
import org.apache.sysds.test.AutomatedTestBase;
import org.apache.sysds.test.TestConfiguration;
import org.apache.sysds.test.TestUtils;
import org.junit.Ignore;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.Parameterized;

@RunWith(Parameterized.class)
@net.jcip.annotations.NotThreadSafe
public class BuiltinImageTransformLinTest extends AutomatedTestBase {
	private final static String TEST_NAME = "image_transform_linearized";
	private final static String TEST_DIR = "functions/builtin/";
	private final static String TEST_CLASS_DIR = TEST_DIR + BuiltinImageTransformLinTest.class.getSimpleName() + "/";

	private final static double eps = 1e-10;

	private final static double spSparse = 0.1;
	private final static double spDense = 0.9;
	private final static double fill_value = 0.0;

	@Parameterized.Parameter(0)
	public int s_rows;
	@Parameterized.Parameter(1)
	public int s_cols;
	@Parameterized.Parameter(2)
	public int n_imgs;

	public double a;
	public double b;
	public double c;
	public double d;
	public double e;
	public double f;

	@Parameterized.Parameters
	public static Collection<Object[]> data() {
		return Arrays.asList(new Object[][] {{16, 15, 50}, {32, 31, 100}, {64, 64, 200},{127, 128, 100}, {256, 256, 200}, {500, 135, 100}});
	}

	@Override
	public void setUp() {
		// rotate 30 degrees around the center
		a = Math.sqrt(3) / 2;
		b = -1.0 / 2.0;
		c = s_cols / 4.0 * (3 - Math.sqrt(3));
		d = 1.0 / 2.0;
		e = Math.sqrt(3) / 2;
		f = s_rows / 4.0 * (1 - Math.sqrt(3));

		addTestConfiguration(TEST_NAME, new TestConfiguration(TEST_CLASS_DIR, TEST_NAME, new String[] {"B"}));
	}

	@Test
	public void testImageTransformLinMatrixDenseCP() {
		runImageTransformLinTest(false, ExecType.CP);
	}

	@Test
	public void testImageTransformLinMatrixSparseCP() {
		runImageTransformLinTest(true, ExecType.CP);
	}

	@Ignore
	@Test
	public void testImageTransformLinMatrixDenseSP() {
		runImageTransformLinTest(false, ExecType.SPARK);
	}

	@Ignore
	@Test
	public void testImageTransformLinMatrixSparseSP() {
		runImageTransformLinTest(false, ExecType.SPARK);
	}

	private void runImageTransformLinTest(boolean sparse, ExecType instType) {
		ExecMode platformOld = setExecMode(instType);
		disableOutAndExpectedDeletion();
		setOutputBuffering(true);

		try {
			loadTestConfiguration(getTestConfiguration(TEST_NAME));
			double sparsity = sparse ? spSparse : spDense;

			String HOME = SCRIPT_DIR + TEST_DIR;
			fullDMLScriptName = HOME + TEST_NAME + ".dml";
			programArgs = new String[] {"-nvargs", "in_file=" + input("A"), "out_file=" + output("B"),
				"width=" + s_cols * s_rows, "height=" + n_imgs, "out_w=" + s_cols, "out_h=" + Math.floor(s_rows * 1.2), "a=" + a,
				"b=" + b, "c=" + c, "d=" + d, "e=" + e, "f=" + f, "fill_value=" + fill_value, "s_cols=" + s_cols,
				"s_rows=" + s_rows};

			fullRScriptName = HOME + TEST_NAME + ".R";
			rCmd = "Rscript" + " " + fullRScriptName + " " + inputDir() + " " + expectedDir() + " " + s_cols * s_rows + " "
				+ n_imgs + " " + s_cols + " " + Math.floor(s_rows * 1.2) + " " + a + " " + b + " " + c + " " + d + " " + e + " " + f
				+ " " + fill_value + " " + s_cols + " " + s_rows;

			// generate actual dataset
			double[][] A = getRandomMatrix(n_imgs, s_cols * s_rows, 0, 255, sparsity, 7);
			writeInputMatrixWithMTD("A", A, true);

			runTest(true, false, null, -1);
			runRScript(true);

			// compare matrices

			HashMap<MatrixValue.CellIndex, Double> dmlfile = readDMLMatrixFromOutputDir("B");
			HashMap<MatrixValue.CellIndex, Double> rfile = readRMatrixFromExpectedDir("B");
			TestUtils.compareMatrices(dmlfile, rfile, eps, "Stat-DML", "Stat-R");
		}
		finally {
			rtplatform = platformOld;
		}
	}
}
