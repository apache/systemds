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

public class BuiltinImageShearTest extends AutomatedTestBase {
	private final static String TEST_NAME = "image_shear";
	private final static String TEST_DIR = "functions/builtin/";
	private final static String TEST_CLASS_DIR = TEST_DIR + BuiltinImageShearTest.class.getSimpleName() + "/";

	private final static double eps = 1e-10;
	private final static int rows = 135;
	private final static int cols = 500;
	private final static double shearX = 0.1;
	private final static double shearY = -0.3;
	private final static double fill_value = 128.0;

	@Override
	public void setUp() {
		addTestConfiguration(TEST_NAME, new TestConfiguration(TEST_CLASS_DIR, TEST_NAME, new String[] {"B"}));
	}

	@Test
	public void testImageShearZero() throws Exception {
		loadTestConfiguration(getTestConfiguration(TEST_NAME));
		double[][] input = TestUtils.readExpectedResource("ImageTransformInput.csv", rows, cols);
		double[][] reference = input;
		String HOME = SCRIPT_DIR + TEST_DIR;
		fullDMLScriptName = HOME + TEST_NAME + ".dml";
		programArgs = new String[] {"-nvargs", "in_file=" + input("A"), "out_file=" + output("B"), "width=" + cols,
				"height=" + rows, "shear_x=" + 0, "shear_y=" + 0, "fill_value=" + fill_value};
		writeInputMatrixWithMTD("A", input, true);
		runTest(true, false, null, -1);

		HashMap<MatrixValue.CellIndex, Double> dmlfile = readDMLMatrixFromOutputDir("B");
		double[][] dml_res = TestUtils.convertHashMapToDoubleArray(dmlfile, rows, cols);
		TestUtils.compareMatrices(reference, dml_res, eps, "Input vs. DML");
	}

	@Test
	public void testImageShearPillowX() throws Exception {
		loadTestConfiguration(getTestConfiguration(TEST_NAME));
		final double fill_value = 128.0;
		double[][] input = TestUtils.readExpectedResource("ImageTransformInput.csv", rows, cols);
		double[][] reference = TestUtils.readExpectedResource("ImageTransformShearedX.csv", rows, cols);
		String HOME = SCRIPT_DIR + TEST_DIR;
		fullDMLScriptName = HOME + TEST_NAME + ".dml";
		programArgs = new String[] {"-nvargs", "in_file=" + input("A"), "out_file=" + output("B"), "width=" + cols,
				"height=" + rows, "shear_x=" + shearX, "shear_y=" + 0, "fill_value=" + fill_value};
		writeInputMatrixWithMTD("A", input, true);
		runTest(true, false, null, -1);

		HashMap<MatrixValue.CellIndex, Double> dmlfile = readDMLMatrixFromOutputDir("B");
		double[][] dml_res = TestUtils.convertHashMapToDoubleArray(dmlfile, rows, cols);
		TestUtils.compareMatrices(reference, dml_res, eps, "Pillow vs. DML");
	}

	@Test
	public void testImageShearPillowY() throws Exception {
		loadTestConfiguration(getTestConfiguration(TEST_NAME));
		final double fill_value = 128.0;
		double[][] input = TestUtils.readExpectedResource("ImageTransformInput.csv", rows, cols);
		double[][] reference = TestUtils.readExpectedResource("ImageTransformShearedY.csv", rows, cols);
		String HOME = SCRIPT_DIR + TEST_DIR;
		fullDMLScriptName = HOME + TEST_NAME + ".dml";
		programArgs = new String[] {"-nvargs", "in_file=" + input("A"), "out_file=" + output("B"), "width=" + cols,
				"height=" + rows, "shear_x=" + 0, "shear_y=" + shearY, "fill_value=" + fill_value};
		writeInputMatrixWithMTD("A", input, true);
		runTest(true, false, null, -1);

		HashMap<MatrixValue.CellIndex, Double> dmlfile = readDMLMatrixFromOutputDir("B");
		double[][] dml_res = TestUtils.convertHashMapToDoubleArray(dmlfile, rows, cols);
		TestUtils.compareMatrices(reference, dml_res, eps, "Pillow vs. DML");
	}
}
