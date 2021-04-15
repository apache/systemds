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

import org.apache.sysds.api.DMLScript;
import org.apache.sysds.common.Types;
import org.apache.sysds.hops.OptimizerUtils;
import org.apache.sysds.lops.LopProperties;
import org.apache.sysds.runtime.matrix.data.MatrixValue;
import org.apache.sysds.test.AutomatedTestBase;
import org.apache.sysds.test.TestConfiguration;
import org.apache.sysds.test.TestUtils;
import org.junit.Assert;
import org.junit.Test;

import java.util.HashMap;

public class BuiltinSmoteTest extends AutomatedTestBase {

	private final static String TEST_NAME = "smote";
	private final static String TEST_DIR = "functions/builtin/";
	private static final String TEST_CLASS_DIR = TEST_DIR + BuiltinSmoteTest.class.getSimpleName() + "/";

	private final static int rows = 20;
	private final static int colsX = 20;

	@Override
	public void setUp() {
		TestUtils.clearAssertionInformation();
		addTestConfiguration(TEST_NAME, new TestConfiguration(TEST_CLASS_DIR, TEST_NAME, new String[] {"C"}));
	}

	@Test
	public void testSmote0CP() {
		double[][] mask =  new double[][]{{1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0}};
		runSmoteTest(100, 3, mask, LopProperties.ExecType.CP);
	}

	@Test
	public void testSmote1CP() {
		double[][] mask =  new double[][]{{1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1}};
		runSmoteTest(300, 10, mask, LopProperties.ExecType.CP);
	}

	@Test
	public void testSmote2CP() {
		double[][] mask =  new double[][]{{1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0}};
		runSmoteTest(400, 5, mask, LopProperties.ExecType.CP);
	}

	@Test
	public void testSmote3CP() {
		double[][] mask =  new double[][]{{1,0,0,0,0,0,0,1,1,1,0,0,0,0,0,0,0,0,0,0}};
		runSmoteTest(300, 3, mask, LopProperties.ExecType.CP);
	}

	@Test
	public void testSmote4CP() {
		double[][] mask =  new double[][]{{0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0}};
		runSmoteTest(400, 5, mask, LopProperties.ExecType.CP);	}

	public void testSmote3Spark() {
		double[][] mask =  new double[][]{{1,0,0,0,0,0,0,1,1,1,0,0,0,0,0,0,0,0,0,0}};
		runSmoteTest(300, 3, mask, LopProperties.ExecType.SPARK);
	}

	@Test
	public void testSmote4Spark() {
		double[][] mask =  new double[][]{{0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0}};
		runSmoteTest(400, 5, mask, LopProperties.ExecType.SPARK);	}
		

	private void runSmoteTest(int sample, int nn, double[][] mask, LopProperties.ExecType instType) {
		Types.ExecMode platformOld = setExecMode(instType);

		boolean oldFlag = OptimizerUtils.ALLOW_ALGEBRAIC_SIMPLIFICATION;
		boolean sparkConfigOld = DMLScript.USE_LOCAL_SPARK_CONFIG;
		OptimizerUtils.ALLOW_ALGEBRAIC_SIMPLIFICATION = false;
		try {
			loadTestConfiguration(getTestConfiguration(TEST_NAME));
			String HOME = SCRIPT_DIR + TEST_DIR;
			fullDMLScriptName = HOME + TEST_NAME + ".dml";
			programArgs = new String[] {"-nvargs", "X=" + input("X"), "S=" + sample, "M="+input("M"),
				"K=" + nn , "Z="+output("Sum"), "T="+input("T")};

			double[][] X = getRandomMatrix(rows, colsX, 1, 10, 1, 1);
			X = TestUtils.round(X);
			writeInputMatrixWithMTD("X", X, true);
			writeInputMatrixWithMTD("M", mask, true);

			double[][] T = getRandomMatrix(rows, colsX, 20, 30, 1, 3);
			T = TestUtils.round(T);

			writeInputMatrixWithMTD("T", T, true);

			runTest(true, false, null, -1);
			HashMap<MatrixValue.CellIndex, Double> value = readDMLMatrixFromOutputDir("Sum");
			Assert.assertEquals("synthetic samples does not fall into minority class cluster",1,
				value.get(new MatrixValue.CellIndex(1,1)), 0.000001);
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

