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

import org.junit.Test;
import org.apache.sysds.api.DMLScript;
import org.apache.sysds.common.Types;
import org.apache.sysds.hops.OptimizerUtils;
import org.apache.sysds.lops.LopProperties;
import org.apache.sysds.test.AutomatedTestBase;
import org.apache.sysds.test.TestConfiguration;
import org.apache.sysds.test.TestUtils;

public class BuiltinMultiLogRegTest extends AutomatedTestBase {

	private final static String TEST_NAME = "MultiLogReg";
	private final static String TEST_DIR = "functions/builtin/";
	private static final String TEST_CLASS_DIR = TEST_DIR + BuiltinMultiLogRegTest.class.getSimpleName() + "/";

	private final static double tol = 0.1;
	private final static int rows = 1500;
	private final static int colsX = 300;
	private final static double sparse = 0.3;
	private final static int maxIter = 10;
	private final static int maxInnerIter = 2;


	@Override
	public void setUp() {
		TestUtils.clearAssertionInformation();
		addTestConfiguration(TEST_NAME, new TestConfiguration(TEST_CLASS_DIR, TEST_NAME, new String[]{"C"}));
	}

	@Test
	public void testMultiLogRegInterceptCP0() {
		runMultiLogeRegTest( 0, tol, 1.0, maxIter, maxInnerIter, LopProperties.ExecType.CP);
	}
	@Test
	public void testMultiLogRegInterceptCP1() {
		runMultiLogeRegTest( 1, tol, 1.0, maxIter, maxInnerIter, LopProperties.ExecType.CP);
	}
	@Test
	public void testMultiLogRegInterceptCP2() {
		runMultiLogeRegTest( 2, tol, 1.0, maxIter, maxInnerIter, LopProperties.ExecType.CP);
	}
	@Test
	public void testMultiLogRegInterceptSpark0() {
		runMultiLogeRegTest( 0, tol, 1.0, maxIter, maxInnerIter, LopProperties.ExecType.SPARK);
	}
	@Test
	public void testMultiLogRegInterceptSpark1() {
		runMultiLogeRegTest( 1, tol, 1.0, maxIter, maxInnerIter, LopProperties.ExecType.SPARK);
	}
	@Test
	public void testMultiLogRegInterceptSpark2() {
		runMultiLogeRegTest(2, tol, 1.0, maxIter, maxInnerIter, LopProperties.ExecType.SPARK);
	}

	private void runMultiLogeRegTest( int inc, double tol, double reg, int maxOut, int maxIn, LopProperties.ExecType instType) {
		Types.ExecMode platformOld = setExecMode(instType);

		boolean oldFlag = OptimizerUtils.ALLOW_ALGEBRAIC_SIMPLIFICATION;
		boolean sparkConfigOld = DMLScript.USE_LOCAL_SPARK_CONFIG;

		try {
			loadTestConfiguration(getTestConfiguration(TEST_NAME));

			String HOME = SCRIPT_DIR + TEST_DIR;
			fullDMLScriptName = HOME + TEST_NAME + ".dml";

			programArgs = new String[]{"-nvargs", "X=" + input("X"), "Y=" + input("Y"), "output=" + output("betas"),
					"inc=" + String.valueOf(inc).toUpperCase(), "tol=" + tol, "reg=" + reg, "maxOut=" + maxOut, "maxIn="+maxIn, "verbose=FALSE"};

			double[][] X = getRandomMatrix(rows, colsX, 0, 1, sparse, -1);
			double[][] Y = getRandomMatrix(rows, 1, 0, 5, 1, -1);
			Y = TestUtils.round(Y);

			writeInputMatrixWithMTD("X", X, true);
			writeInputMatrixWithMTD("Y", Y, true);
			runTest(true, false, null, -1);
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
