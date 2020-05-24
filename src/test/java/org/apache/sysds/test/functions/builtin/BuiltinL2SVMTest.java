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

import org.junit.Test;
import org.apache.sysds.api.DMLScript;
import org.apache.sysds.common.Types;
import org.apache.sysds.hops.OptimizerUtils;
import org.apache.sysds.lops.LopProperties;
import org.apache.sysds.runtime.matrix.data.MatrixValue;
import org.apache.sysds.test.AutomatedTestBase;
import org.apache.sysds.test.TestConfiguration;
import org.apache.sysds.test.TestUtils;


public class BuiltinL2SVMTest extends AutomatedTestBase {
	private final static String TEST_NAME = "l2svm";
	private final static String TEST_DIR = "functions/builtin/";
	private static final String TEST_CLASS_DIR = TEST_DIR + BuiltinL2SVMTest.class.getSimpleName() + "/";


	private final static double eps = 0.001;
	private final static int rows = 1000;
	private final static int colsX = 200;
	private final static double spSparse = 0.01;
	private final static double spDense = 0.7;
	private final static int max_iter = 10;


	@Override
	public void setUp() {
		TestUtils.clearAssertionInformation();
		addTestConfiguration(TEST_NAME,new TestConfiguration(TEST_CLASS_DIR, TEST_NAME,new String[]{"C"}));
	}

	@Test
	public void testL2SVMDense() {
		runL2SVMTest(false, false, eps, 1.0, max_iter, LopProperties.ExecType.CP);
	}
	@Test
	public void testL2SVMSparse() {
		runL2SVMTest(true, false, eps, 1.0, max_iter, LopProperties.ExecType.CP);
	}

	@Test
	public void testL2SVMIntercept() {
		runL2SVMTest(true,true, eps, 1.0, max_iter, LopProperties.ExecType.SPARK);
	}

	@Test
	public void testL2SVMDenseIntercept() {
		 runL2SVMTest(false,true, 1, 1.0, max_iter, LopProperties.ExecType.CP);
	}

	@Test
	public void testL2SVMSparseLambda2() {
		runL2SVMTest(true,true, 1, 2.0, max_iter, LopProperties.ExecType.CP);
	}

	@Test
	public void testL2SVMSparseLambda100CP() {
		runL2SVMTest(true,true, 1, 100, max_iter, LopProperties.ExecType.CP);
	}
	@Test
	public void testL2SVMSparseLambda100Spark() {
		runL2SVMTest(true,true, 1, 100, max_iter, LopProperties.ExecType.SPARK);
	}

	@Test
	public void testL2SVMIteration() {
		runL2SVMTest(true,true, 1, 2.0, 100, LopProperties.ExecType.CP);
	}

	private void runL2SVMTest(boolean sparse, boolean intercept, double eps,
		double lambda, int run, LopProperties.ExecType instType)
	{
		Types.ExecMode platformOld = setExecMode(instType);

		boolean oldFlag = OptimizerUtils.ALLOW_ALGEBRAIC_SIMPLIFICATION;
		boolean sparkConfigOld = DMLScript.USE_LOCAL_SPARK_CONFIG;

		try
		{
			loadTestConfiguration(getTestConfiguration(TEST_NAME));

			double sparsity = sparse ? spSparse : spDense;

			String HOME = SCRIPT_DIR + TEST_DIR;

			fullDMLScriptName = HOME + TEST_NAME + ".dml";
			programArgs = new String[]{
					"-nvargs", "X=" + input("X"), "Y=" + input("Y"), "model=" + output("model"),
					"inc=" + String.valueOf(intercept).toUpperCase(),"eps=" + eps, "lam=" + lambda, "max=" + run};


			fullRScriptName = HOME + TEST_NAME + ".R";
			rCmd = getRCmd(inputDir(), Boolean.toString(intercept), Double.toString(eps),
					Double.toString(lambda), Integer.toString(run), expectedDir());

			//generate actual datasets
			double[][] X = getRandomMatrix(rows, colsX, 0, 100, sparsity, 10);
			double[][] Y= getRandomMatrix(rows, 1, -1, 1, 1, -1);

			Y = TestUtils.round(Y);

			writeInputMatrixWithMTD("X", X, true);
			writeInputMatrixWithMTD("Y", Y, true);

			runTest(true, false, null, -1);
			runRScript(true);

			HashMap<MatrixValue.CellIndex, Double> dmlfile = readDMLMatrixFromHDFS("model");
			HashMap<MatrixValue.CellIndex, Double> rfile  = readRMatrixFromFS("model");
			TestUtils.compareMatrices(dmlfile, rfile, eps, "Stat-DML", "Stat-R");

		}
		finally {
			rtplatform = platformOld;
			DMLScript.USE_LOCAL_SPARK_CONFIG = sparkConfigOld;
			OptimizerUtils.ALLOW_ALGEBRAIC_SIMPLIFICATION = oldFlag;
			OptimizerUtils.ALLOW_AUTO_VECTORIZATION = true;
			OptimizerUtils.ALLOW_OPERATOR_FUSION = true;
		}
	}
}
