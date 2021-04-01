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

import org.apache.sysds.common.Types.ExecMode;
import org.apache.sysds.lops.LopProperties.ExecType;
import org.apache.sysds.runtime.matrix.data.MatrixValue;
import org.apache.sysds.test.AutomatedTestBase;
import org.apache.sysds.test.TestConfiguration;
import org.apache.sysds.test.TestUtils;
import org.junit.Test;

import java.util.HashMap;

public class BuiltinImageTransformTest extends AutomatedTestBase {
	private final static String TEST_NAME = "image_transform";
	private final static String TEST_DIR = "functions/builtin/";
	private final static String TEST_CLASS_DIR = TEST_DIR + BuiltinImageTransformTest.class.getSimpleName() + "/";

	private final static double eps = 1e-10;
	private final static int rows = 512;
	private final static int cols = 512;
	private final static double spSparse = 0.1;
	private final static double spDense = 0.9;
	// rotate 45 degrees around the center
	private final static double a = 1 / Math.sqrt(2);
	private final static double b = -1 / Math.sqrt(2);
	private final static double c = cols / 2;
	private final static double d = 1 / Math.sqrt(2);
	private final static double e = 1 / Math.sqrt(2);
	private final static double f = rows / 2 * (1 - Math.sqrt(2));

	@Override public void setUp() {
		addTestConfiguration(TEST_NAME, new TestConfiguration(TEST_CLASS_DIR, TEST_NAME, new String[] {"B"}));
	}

	@Test public void testImageTransformMatrixDenseCP() {
		runImageTransformTest(false, ExecType.CP);
	}

	@Test public void testImageTransformMatrixSparseCP() {
		runImageTransformTest(true, ExecType.CP);
	}

	@Test public void testImageTransformMatrixDenseSP() {
		runImageTransformTest(false, ExecType.SPARK);
	}

	@Test public void testImageTransformMatrixSparseSP() {
		runImageTransformTest(false, ExecType.SPARK);
	}

	private void runImageTransformTest(boolean sparse, ExecType instType) {
		ExecMode platformOld = setExecMode(instType);
		disableOutAndExpectedDeletion();

		try {
			loadTestConfiguration(getTestConfiguration(TEST_NAME));
			double sparsity = sparse ? spSparse : spDense;

			String HOME = SCRIPT_DIR + TEST_DIR;
			fullDMLScriptName = HOME + TEST_NAME + ".dml";
			programArgs = new String[] {"-nvargs", "in_file=" + input("A"), "out_file=" + output("B"), "width=" + cols,
				"height=" + rows, "out_w=" + cols, "out_h=" + rows,
				"a=" + a, "b=" + b, "c=" + c, "d=" + d, "e=" + e, "f=" + f};

			fullRScriptName = HOME + TEST_NAME + ".R";
			rCmd = "Rscript" + " " + fullRScriptName + " " + inputDir() + " " + expectedDir() + " " + cols + " " + rows + " " + cols + " " + rows + " " + a + " " + b + " " + c + " " + d + " " + e + " " + f;

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
