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

import org.junit.Assert;
import org.junit.Test;
import org.apache.sysds.api.DMLScript;
import org.apache.sysds.common.Types;
import org.apache.sysds.hops.OptimizerUtils;
import org.apache.sysds.lops.LopProperties.ExecType;
import org.apache.sysds.test.AutomatedTestBase;
import org.apache.sysds.test.TestConfiguration;
import org.apache.sysds.test.TestUtils;
import org.apache.sysds.utils.Statistics;

public class BuiltinFactorizationTest extends AutomatedTestBase
{
	private final static String TEST_NAME1 = "GNMF";
	private final static String TEST_NAME2 = "PNMF";
	private final static String TEST_DIR = "functions/builtin/";
	private static final String TEST_CLASS_DIR = TEST_DIR + BuiltinFactorizationTest.class.getSimpleName() + "/";

	private final static int rows = 1210;
	private final static int cols = 2012;
	private final static int rank = 50;
	private final static double sparsity = 0.01;
	private final static double max_iter = 10;

	@Override
	public void setUp() {
		TestUtils.clearAssertionInformation();
		addTestConfiguration(TEST_NAME1,new TestConfiguration(TEST_CLASS_DIR, TEST_NAME1,new String[]{"U","V"}));
		addTestConfiguration(TEST_NAME2,new TestConfiguration(TEST_CLASS_DIR, TEST_NAME2,new String[]{"U","V"}));
	}

	@Test
	public void testGNMFRewritesCP() {
		runFactorizationTest(TEST_NAME1, true, ExecType.CP);
	}

	@Test
	public void testGNMFNoRewritesCP() {
		runFactorizationTest(TEST_NAME1, false, ExecType.CP);
	}

// Note: too expensive in forced spark mode (lazy evaluation across iterations)
//	@Test
//	public void testGNMFRewritesSpark() {
//		runFactorizationTest(TEST_NAME1, true, ExecType.SPARK);
//	}
//
//	@Test
//	public void testGNMFNoRewritesSpark() {
//		runFactorizationTest(TEST_NAME1, false, ExecType.SPARK);
//	}
//	
	@Test
	public void testPNMFRewritesCP() {
		runFactorizationTest(TEST_NAME2, true, ExecType.CP);
	}

	@Test
	public void testPNMFNoRewritesCP() {
		runFactorizationTest(TEST_NAME2, false, ExecType.CP);
	}
	
	@Test
	public void testPNMFRewritesSpark() {
		runFactorizationTest(TEST_NAME2, true, ExecType.SPARK);
	}

	@Test
	public void testPNMFNoRewritesSpark() {
		runFactorizationTest(TEST_NAME2, false, ExecType.SPARK);
	}
	
	private void runFactorizationTest(String testname, boolean rewrites, ExecType instType)
	{
		Types.ExecMode platformOld = setExecMode(instType);

		boolean oldFlag = OptimizerUtils.ALLOW_ALGEBRAIC_SIMPLIFICATION;
		boolean sparkConfigOld = DMLScript.USE_LOCAL_SPARK_CONFIG;

		try
		{
			loadTestConfiguration(getTestConfiguration(testname));

			String HOME = SCRIPT_DIR + TEST_DIR;

			fullDMLScriptName = HOME + testname + ".dml";
			programArgs = new String[]{ "-stats",
				"-args", input("X"), output("W"), output("H"),
				String.valueOf(rank), String.valueOf(max_iter)};

			OptimizerUtils.ALLOW_ALGEBRAIC_SIMPLIFICATION = rewrites;
			
			//generate input and write incl meta data
			double[][] Xa = TestUtils.generateTestMatrix(rows, cols, 1.0, 10.0, sparsity, 7);
			writeInputMatrixWithMTD("X", Xa, true);
			
			//run test case
			runTest(true, false, null, -1);
			
			if( (testname.equals(TEST_NAME1) || testname.equals(TEST_NAME2))
				&& instType==ExecType.CP) {
				Assert.assertTrue(Statistics.getCPHeavyHitterCount("/")>=20);
			}
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
