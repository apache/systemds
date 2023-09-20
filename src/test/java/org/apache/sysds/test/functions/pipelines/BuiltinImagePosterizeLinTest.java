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
import java.util.Random;

public class BuiltinImagePosterizeLinTest extends AutomatedTestBase {
	private final static String TEST_NAME = "image_posterize_linearized"; // IS A NEW INTERFACE REQUIRED? maybe bits as
																			// vector in the future?
	private final static String TEST_DIR = "functions/builtin/";
	private final static String TEST_CLASS_DIR = TEST_DIR + BuiltinImagePosterizeLinTest.class.getSimpleName() + "/";

	private final static double eps = 1e-10;
	private final static double spSparse = 0.1;
	private final static double spDense = 0.9;
	private final static Random random = new Random();

	@Override
	public void setUp() {
		addTestConfiguration(TEST_NAME, new TestConfiguration(TEST_CLASS_DIR, TEST_NAME, new String[] { "B" }));
	}

	@Test
	public void testImagePosterizeLinMatrixDenseCP() {
		runImagePosterizeLinTest(false, ExecType.CP);
	}

	@Test
	public void testImagePosterizeLinMatrixSparseCP() {
		runImagePosterizeLinTest(true, ExecType.CP);
	}

	@Test
	public void testImagePosterizeLinMatrixDenseSP() {
		runImagePosterizeLinTest(false, ExecType.SPARK);
	}

	@Test
	public void testImagePosterizeLinMatrixSparseSP() {
		runImagePosterizeLinTest(false, ExecType.SPARK);
	}

	private void runImagePosterizeLinTest(boolean sparse, ExecType instType) {
		ExecMode platformOld = setExecMode(instType);
		disableOutAndExpectedDeletion();

		setOutputBuffering(true);
		int n_imgs = random.nextInt(1000) + 1; // n_imgs
		int w = random.nextInt(100) + 1; // w*h
		int h = random.nextInt(100) + 1; // w*h
		int bits = random.nextInt(7) + 1;

		try {
			loadTestConfiguration(getTestConfiguration(TEST_NAME));
			double sparsity = sparse ? spSparse : spDense;

			String HOME = SCRIPT_DIR + TEST_DIR;
			fullDMLScriptName = HOME + TEST_NAME + ".dml";
			programArgs = new String[] { "-nvargs", "in_file=" + input("A"), "out_file=" + output("B"),
					"width=" + w * h,
					"height=" + n_imgs, "bits=" + bits };

			// generate actual dataset
			double[][] A = getRandomMatrix(n_imgs, w * h, 0, 255, sparsity, 7);
			writeInputMatrixWithMTD("A", A, true);

			double[][] ref = new double[n_imgs][w * h];
			for (int i = 0; i < n_imgs; i++) {
				for (int j = 0; j < w * h; j++) {
					ref[i][j] = (int) (A[i][j] / (1 << (8 - bits))) * (1 << (8 - bits));
				}
			}

			runTest(true, false, null, -1);

			// compare matrices
			HashMap<MatrixValue.CellIndex, Double> dmlfile = readDMLMatrixFromOutputDir("B");
			double[][] dml_res = TestUtils.convertHashMapToDoubleArray(dmlfile, n_imgs, w * h);
			TestUtils.compareMatrices(ref, dml_res, eps, "Java vs. DML");
		} finally {
			rtplatform = platformOld;
		}
	}
}
