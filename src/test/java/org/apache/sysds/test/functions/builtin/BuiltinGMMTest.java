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

import org.apache.sysds.common.Types;
import org.apache.sysds.hops.OptimizerUtils;
import org.apache.sysds.lops.LopProperties;
import org.apache.sysds.runtime.matrix.data.MatrixValue;
import org.apache.sysds.test.AutomatedTestBase;
import org.apache.sysds.test.TestConfiguration;
import org.apache.sysds.test.TestUtils;
import org.junit.Test;

import java.util.HashMap;

public class BuiltinGMMTest extends AutomatedTestBase {
	private final static String TEST_NAME = "GMM";
	private final static String TEST_DIR = "functions/builtin/";
	private static final String TEST_CLASS_DIR = TEST_DIR + BuiltinGMMTest.class.getSimpleName() + "/";

	private final static double eps = 1;
	private final static double tol = 1e-3;
	private final static double tol1 = 1e-4;
	private final static double tol2 = 1e-5;
	//private final static int rows = 100;
	//private final static double spDense = 0.99;
	private final static String DATASET = SCRIPT_DIR + "functions/transform/input/iris/iris.csv";

	@Override
	public void setUp() {
		addTestConfiguration(TEST_NAME, new TestConfiguration(TEST_CLASS_DIR, TEST_NAME, new String[] {"B"}));
	}

	@Test
	public void testGMMM1() { runGMMTest(3, "VVV", "random", 10, 0.0000001, tol,true, LopProperties.ExecType.CP); }

	@Test
	public void testGMMM2() {
		runGMMTest(3, "EEE", "random", 150, 0.000001, tol1,true, LopProperties.ExecType.CP);
	}

	@Test
	public void testGMMM3() {
		runGMMTest(3, "VVI", "random", 10, 0.000000001, tol,true, LopProperties.ExecType.CP);
	}

	@Test
	public void testGMMM4() {
		runGMMTest(3, "VII", "random", 50, 0.000001, tol2,true, LopProperties.ExecType.CP);
	}

	@Test
	public void testGMMM1Kmean() {
		runGMMTest(3, "VVV", "kmeans", 10, 0.0000001, tol,true, LopProperties.ExecType.CP);
	}

	@Test
	public void testGMMM2Kmean() {
		runGMMTest(3, "EEE", "kmeans", 150, 0.000001, tol,true, LopProperties.ExecType.CP);
	}

	@Test
	public void testGMMM3Kmean() {
		runGMMTest(3, "VVI", "kmeans", 10, 0.00000001, tol1,true, LopProperties.ExecType.CP);
	}

	@Test
	public void testGMMM4Kmean() {
		runGMMTest(3, "VII", "kmeans", 50, 0.000001, tol2,true, LopProperties.ExecType.CP);
	}

	@Test
	public void testGMMM1Spark() { runGMMTest(3, "VVV", "random", 10, 0.0000001, tol,true, LopProperties.ExecType.SPARK); }

	@Test
	public void testGMMM2Spark() {
		runGMMTest(3, "EEE", "random", 50, 0.0000001, tol,true, LopProperties.ExecType.CP);
	}

	@Test
	public void testGMMMS3Spark() {
		runGMMTest(3, "VVI", "random", 100, 0.000001, tol,true, LopProperties.ExecType.CP);
	}

	@Test
	public void testGMMM4Spark() {
		runGMMTest(3, "VII", "random", 100, 0.000001, tol1,true, LopProperties.ExecType.CP);
	}

	@Test
	public void testGMMM1KmeanSpark() {
		runGMMTest(3, "VVV", "kmeans", 100, 0.000001, tol2,false, LopProperties.ExecType.SPARK);
	}

	@Test
	public void testGMMM2KmeanSpark() {
		runGMMTest(3, "EEE", "kmeans", 50, 0.00000001, tol1,false, LopProperties.ExecType.SPARK);
	}

	@Test
	public void testGMMM3KmeanSpark() {
		runGMMTest(3, "VVI", "kmeans", 100, 0.000001, tol,false, LopProperties.ExecType.SPARK);
	}

	@Test
	public void testGMMM4KmeanSpark() {
		runGMMTest(3, "VII", "kmeans", 100, 0.000001, tol,false, LopProperties.ExecType.SPARK);
	}

	private void runGMMTest(int G_mixtures, String model, String init_param, int iter, double reg, double tol, boolean rewrite,
			LopProperties.ExecType instType) {
		Types.ExecMode platformOld = setExecMode(instType);
		OptimizerUtils.ALLOW_ALGEBRAIC_SIMPLIFICATION = rewrite;

		try {
			loadTestConfiguration(getTestConfiguration(TEST_NAME));
			String HOME = SCRIPT_DIR + TEST_DIR;
			fullDMLScriptName = HOME + TEST_NAME + ".dml";
			programArgs = new String[] {"-args", DATASET, String.valueOf(G_mixtures), model, init_param, String.valueOf(iter),
					String.valueOf(reg), String.valueOf(tol), output("B"), output("O")};

			fullRScriptName = HOME + TEST_NAME + ".R";
			rCmd = "Rscript" + " " + fullRScriptName + " " + DATASET + " " + String
					.valueOf(G_mixtures) + " " + model + " " + expectedDir();

			runTest(true, false, null, -1);
			runRScript(true);

			//compare matrices
			HashMap<MatrixValue.CellIndex, Double> dmlfile = readDMLMatrixFromHDFS("O");
			HashMap<MatrixValue.CellIndex, Double> rfile = readRMatrixFromFS("O");
			System.out.println(dmlfile.values().iterator().next().doubleValue());
			TestUtils.compareMatrices(dmlfile, rfile, eps, "Stat-DML", "Stat-R");
		}
		finally {
			rtplatform = platformOld;
		}
	}
}
