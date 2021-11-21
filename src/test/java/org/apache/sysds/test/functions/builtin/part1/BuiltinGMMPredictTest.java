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

import org.apache.sysds.common.Types;
import org.apache.sysds.common.Types.ExecType;
import org.apache.sysds.hops.OptimizerUtils;
import org.apache.sysds.test.AutomatedTestBase;
import org.apache.sysds.test.TestConfiguration;
import org.apache.sysds.test.TestUtils;
import org.junit.Assert;
import org.junit.Test;

public class BuiltinGMMPredictTest extends AutomatedTestBase {
	private final static String TEST_NAME = "GMM_Predict";
	private final static String TEST_DIR = "functions/builtin/";
	private static final String TEST_CLASS_DIR = TEST_DIR + BuiltinGMMPredictTest.class.getSimpleName() + "/";

	private final static double eps = 2;
	private final static double tol = 1e-3;
	private final static double tol2 = 1e-5;

	private final static String DATASET = DATASET_DIR+"iris/iris.csv";

	@Override
	public void setUp() {
		addTestConfiguration(TEST_NAME, new TestConfiguration(TEST_CLASS_DIR, TEST_NAME, new String[] {"B"}));
	}

	@Test
	public void testGMMMPredictCP1() {
		runGMMPredictTest(3, "VVI", "random", 10,
			0.000000001, tol,42,true, ExecType.CP);
	}

	@Test
	public void testGMMMPredictCP2() {
		runGMMPredictTest(3, "VII", "random", 50,
			0.000001, tol2,42,true, ExecType.CP);
	}

	@Test
	public void testGMMMPredictCPKmean1() {
		runGMMPredictTest(3, "VVV", "kmeans", 10,
			0.0000001, tol,42,true, ExecType.CP);
	}

	@Test
	public void testGMMMPredictCPKmean2() {
		runGMMPredictTest(3, "EEE", "kmeans", 150,
			0.000001, tol,42,true, ExecType.CP);
	}

	@Test
	public void testGMMMPredictCPKmean3() {
		runGMMPredictTest(3, "VII", "kmeans", 50,
			0.000001, tol2,42,true, ExecType.CP);
	}

//	@Test
//	public void testGMMM1Spark() {
//		runGMMPredictTest(3, "VVV", "random", 10,
//		0.0000001, tol,42,true, ExecType.SPARK); }
//
//	@Test
//	public void testGMMM2Spark() {
//		runGMMPredictTest(3, "EEE", "random", 50,
//			0.0000001, tol,42,true, ExecType.CP);
//	}
//
//	@Test
//	public void testGMMMS3Spark() {
//		runGMMPredictTest(3, "VVI", "random", 100,
//			0.000001, tol,42,true, ExecType.CP);
//	}
//
//	@Test
//	public void testGMMM4Spark() {
//		runGMMPredictTest(3, "VII", "random", 100,
//			0.000001, tol1,42,true, ExecType.CP);
//	}
//
//	@Test
//	public void testGMMM1KmeanSpark() {
//		runGMMPredictTest(3, "VVV", "kmeans", 100,
//			0.000001, tol2,42,false, ExecType.SPARK);
//	}
//
//	@Test
//	public void testGMMM2KmeanSpark() {
//		runGMMPredictTest(3, "EEE", "kmeans", 50,
//			0.00000001, tol1,42,false, ExecType.SPARK);
//	}
//
//	@Test
//	public void testGMMM3KmeanSpark() {
//		runGMMPredictTest(3, "VVI", "kmeans", 100,
//			0.000001, tol,42,false, ExecType.SPARK);
//	}
//
//	@Test
//	public void testGMMM4KmeanSpark() {
//		runGMMPredictTest(3, "VII", "kmeans", 100,
//			0.000001, tol,42,false, ExecType.SPARK);
//	}

	private void runGMMPredictTest(int G_mixtures, String model, String init_param, int iter,
		double reg, double tol, int seed, boolean rewrite, ExecType instType) {

		Types.ExecMode platformOld = setExecMode(instType);
		boolean rewriteOld = OptimizerUtils.ALLOW_ALGEBRAIC_SIMPLIFICATION;
		OptimizerUtils.ALLOW_ALGEBRAIC_SIMPLIFICATION = rewrite;
		try {
			loadTestConfiguration(getTestConfiguration(TEST_NAME));
			String HOME = SCRIPT_DIR + TEST_DIR;
			fullDMLScriptName = HOME + TEST_NAME + ".dml";
			String outFile = output("O");
			System.out.println(outFile);
			programArgs = new String[] {"-args", DATASET,
				String.valueOf(G_mixtures), model, init_param, String.valueOf(iter), String.valueOf(reg),
				String.valueOf(tol), String.valueOf(seed), outFile};

			runTest(true, false, null, -1);
			// compare results
			double accuracy = TestUtils.readDMLScalar(outFile);
			Assert.assertEquals(1, accuracy, eps);
		}
		finally {
			resetExecMode(platformOld);
			OptimizerUtils.ALLOW_ALGEBRAIC_SIMPLIFICATION = rewriteOld;
		}
	}
}
