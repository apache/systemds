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

import org.apache.sysds.runtime.matrix.data.MatrixValue;
import org.apache.sysds.test.AutomatedTestBase;
import org.apache.sysds.test.TestConfiguration;
import org.apache.sysds.test.TestUtils;
import org.junit.Test;

import java.util.HashMap;

public class BuiltinImageRotateLinTest extends AutomatedTestBase {
	private final static String TEST_NAME = "image_rotate_linearized";
	private final static String TEST_DIR = "functions/builtin/";
	private final static String TEST_CLASS_DIR = TEST_DIR + BuiltinImageRotateLinTest.class.getSimpleName() + "/";

	private final static double eps = 1e-10;
	private final static int s_rows = 135;
	private final static int s_cols = 500;
	private final static int n_imgs = 1;

	private final static double angle = 42 * (Math.PI / 180);

	@Override
	public void setUp() {
		addTestConfiguration(TEST_NAME, new TestConfiguration(TEST_CLASS_DIR, TEST_NAME, new String[] { "B" }));
	}

	@Test
	public void testImageRotateLinZero() throws Exception {
		loadTestConfiguration(getTestConfiguration(TEST_NAME));
		setOutputBuffering(true);
		final int w = 500, h = 135;
		final double fill_value = 128.0;
		double[][] input = TestUtils.readExpectedResource("ImageTransformLinInput.csv", n_imgs, h * w);
		double[][] reference = input;
		String HOME = SCRIPT_DIR + TEST_DIR;
		fullDMLScriptName = HOME + TEST_NAME + ".dml";
		programArgs = new String[] { "-nvargs", "in_file=" + input("A"), "out_file=" + output("B"),
				"width=" + s_rows * s_cols,
				"height=" + n_imgs, "angle=" + 0, "fill_value=" + fill_value, "s_cols=" + s_cols, "s_rows=" + s_rows };
		writeInputMatrixWithMTD("A", input, true);
		runTest(true, false, null, -1);

		HashMap<MatrixValue.CellIndex, Double> dmlfile = readDMLMatrixFromOutputDir("B");
		double[][] dml_res = TestUtils.convertHashMapToDoubleArray(dmlfile, n_imgs, h * w);
		TestUtils.compareMatrices(reference, dml_res, eps, "Input vs. DML");
	}

	@Test
	public void testImageRotateLinComplete() throws Exception {
		loadTestConfiguration(getTestConfiguration(TEST_NAME));
		setOutputBuffering(true);
		final int w = 500, h = 135;
		final double fill_value = 128.0;
		double[][] input = TestUtils.readExpectedResource("ImageTransformLinInput.csv", n_imgs, h * w);
		double[][] reference = input;
		String HOME = SCRIPT_DIR + TEST_DIR;
		fullDMLScriptName = HOME + TEST_NAME + ".dml";
		programArgs = new String[] { "-nvargs", "in_file=" + input("A"), "out_file=" + output("B"),
				"width=" + s_rows * s_cols,
				"height=" + n_imgs, "angle=" + (2 * Math.PI), "fill_value=" + fill_value, "s_cols=" + s_cols,
				"s_rows=" + s_rows };
		writeInputMatrixWithMTD("A", input, true);
		runTest(true, false, null, -1);

		HashMap<MatrixValue.CellIndex, Double> dmlfile = readDMLMatrixFromOutputDir("B");
		double[][] dml_res = TestUtils.convertHashMapToDoubleArray(dmlfile, n_imgs, h * w);
		TestUtils.compareMatrices(reference, dml_res, eps, "Input vs. DML");
	}

	@Test
	public void testImageRotateLinPillow() throws Exception {
		loadTestConfiguration(getTestConfiguration(TEST_NAME));
		setOutputBuffering(true);
		final int w = 500, h = 135;
		final double fill_value = 128.0;
		double[][] input = TestUtils.readExpectedResource("ImageTransformLinInput.csv", n_imgs, h * w);
		double[][] reference = TestUtils.readExpectedResource("ImageTransformLinRotated.csv", n_imgs, h * w);
		String HOME = SCRIPT_DIR + TEST_DIR;
		fullDMLScriptName = HOME + TEST_NAME + ".dml";
		programArgs = new String[] { "-nvargs", "in_file=" + input("A"), "out_file=" + output("B"),
				"width=" + s_rows * s_cols,
				"height=" + n_imgs, "angle=" + angle, "fill_value=" + fill_value, "s_cols=" + s_cols,
				"s_rows=" + s_rows };
		writeInputMatrixWithMTD("A", input, true);
		runTest(true, false, null, -1);

		HashMap<MatrixValue.CellIndex, Double> dmlfile = readDMLMatrixFromOutputDir("B");
		double[][] dml_res = TestUtils.convertHashMapToDoubleArray(dmlfile, n_imgs, h * w);
		TestUtils.compareMatrices(reference, dml_res, eps, "Pillow vs. DML");
		// The error here seems to be in the pillow implementation. It calculates that
		// the source coordinates of
		// (339, 14) are (351, 87) using fixed point arithmetic, while the floating
		// point implementation returns
		// (351.975384 88.000514) rounded down to (351, 88). (All indices here start at
		// 0)
	}
}
