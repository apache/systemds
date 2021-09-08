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

package org.apache.sysds.test.functions.rewrite;

import static org.junit.Assert.fail;

import org.apache.sysds.api.DMLScript;
import org.apache.sysds.common.Types.ExecMode;
import org.apache.sysds.common.Types.ExecType;
import org.apache.sysds.hops.OptimizerUtils;
import org.apache.sysds.test.AutomatedTestBase;
import org.apache.sysds.test.TestConfiguration;
import org.apache.sysds.test.TestUtils;
import org.junit.Test;

/**
 * from:
 * 
 * res = cbind((X %*% Y), matrix (0, nrow(X), 1));
 * 
 * to:
 * 
 * res = X %*% (cbind(Y, matrix(0, nrow(Y), 1)))
 * 
 * 
 * if the X has many rows, the allocation of x is expensive, to cbind. the case where this is applicable is mLogReg.
 * 
 */
public class RewriteMMCBindZeroVector extends AutomatedTestBase {
	// private static final Log LOG = LogFactory.getLog(RewriteMMCBindZeroVector.class.getName());

	private static final String TEST_NAME1 = "RewritMMCBindZeroVectorOp";
	private static final String TEST_DIR = "functions/rewrite/";
	private static final String TEST_CLASS_DIR = TEST_DIR + RewriteMMCBindZeroVector.class.getSimpleName() + "/";

	@Override
	public void setUp() {
		TestUtils.clearAssertionInformation();
		addTestConfiguration(TEST_NAME1, new TestConfiguration(TEST_CLASS_DIR, TEST_NAME1, new String[] {"R"}));
	}

	@Test
	public void testNoRewritesCP() {
		testRewrite(TEST_NAME1, false, ExecType.CP, 100, 3, 10, false);
	}

	@Test
	public void testNoRewritesSP() {
		testRewrite(TEST_NAME1, false, ExecType.SPARK, 100, 3, 10, false);
	}

	@Test
	public void testRewritesCP() {
		testRewrite(TEST_NAME1, true, ExecType.CP, 100, 3, 10, true);
	}

	@Test
	public void testRewritesSP() {
		testRewrite(TEST_NAME1, true, ExecType.SPARK, 100, 3, 10, true);
	}

	@Test
	public void testRewritesCP_ButToSmall() {
		testRewrite(TEST_NAME1, true, ExecType.CP, 100, 10, 55, false);
	}

	@Test
	public void testRewritesSP_ButToSmall() {
		testRewrite(TEST_NAME1, true, ExecType.SPARK, 100, 10, 55, false);
	}

	private void testRewrite(String testname, boolean rewrites, ExecType et, int leftRows, int rightCols, int shared, boolean rewriteShouldBeExecuted) {
		ExecMode platformOld = rtplatform;
		switch(et) {
			case SPARK:
				rtplatform = ExecMode.SPARK;
				break;
			default:
				rtplatform = ExecMode.HYBRID;
				break;
		}

		boolean sparkConfigOld = DMLScript.USE_LOCAL_SPARK_CONFIG;
		if(rtplatform == ExecMode.SPARK || rtplatform == ExecMode.HYBRID)
			DMLScript.USE_LOCAL_SPARK_CONFIG = true;

		boolean rewritesOld = OptimizerUtils.ALLOW_ALGEBRAIC_SIMPLIFICATION;
		OptimizerUtils.ALLOW_ALGEBRAIC_SIMPLIFICATION = rewrites;

		try {
			TestConfiguration config = getTestConfiguration(testname);
			loadTestConfiguration(config);

			String HOME = SCRIPT_DIR + TEST_DIR;
			fullDMLScriptName = HOME + testname + ".dml";
			programArgs = new String[] {"-explain", "hops","-stats", "-args", input("X"), input("Y"),
				output("R")};
			fullRScriptName = HOME + testname + ".R";
			rCmd = getRCmd(inputDir(), expectedDir());

			double[][] X = getRandomMatrix(leftRows, shared, -1, 1, 0.97d, 7);
			double[][] Y = getRandomMatrix(shared, rightCols, -1, 1, 0.9d, 3);
			writeInputMatrixWithMTD("X", X, false);
			writeInputMatrixWithMTD("Y", Y, false);

			// execute tests
			String out = runTest(null).toString();

			for(String line : out.split("\n")) {
				if(rewrites && rewriteShouldBeExecuted) {
					if(line.contains("b(cbind)"))
						break;
					else if(line.contains("ba(+*)"))
						fail(
							"invalid execution matrix multiplication is done before b(cbind), therefore the rewrite did not tricker.\n\n"
								+ out);
				}
				else {
					if(line.contains("ba(+*)"))
						break;
					else if(line.contains("b(cbind)"))
						fail(
							"invalid execution b(cbind) was done before multiplication, therefore the rewrite did tricker when not allowed.\n\n"
								+ out);
				}

			}
			// compare matrices
			// HashMap<CellIndex, Double> dmlfile = readDMLMatrixFromOutputDir("R");

		}
		finally {
			OptimizerUtils.ALLOW_ALGEBRAIC_SIMPLIFICATION = rewritesOld;
			rtplatform = platformOld;
			DMLScript.USE_LOCAL_SPARK_CONFIG = sparkConfigOld;
		}
	}
}
