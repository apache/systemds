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

import java.util.HashMap;

import org.apache.sysds.common.Opcodes;
import org.junit.Assert;
import org.junit.Test;
import org.apache.sysds.common.Types.ExecMode;
import org.apache.sysds.hops.OptimizerUtils;
import org.apache.sysds.common.Types.ExecType;
import org.apache.sysds.runtime.matrix.data.MatrixValue.CellIndex;
import org.apache.sysds.test.AutomatedTestBase;
import org.apache.sysds.test.TestConfiguration;
import org.apache.sysds.test.TestUtils;
import org.apache.sysds.utils.Statistics;

public class RewriteHoistingLoopInvariantOpsTest extends AutomatedTestBase
{
	private static final String TEST_NAME1 = "RewriteCodeMotionFor";
	private static final String TEST_NAME2 = "RewriteCodeMotionWhile";
	
	private static final String TEST_DIR = "functions/rewrite/";
	private static final String TEST_CLASS_DIR = TEST_DIR + RewriteHoistingLoopInvariantOpsTest.class.getSimpleName() + "/";
	
	private static final int rows = 265;
	private static final int cols = 132;
	private static final int iters = 10;
	private static final double sparsity = 0.1;
	private static final double eps = Math.pow(10, -10);
	
	@Override
	public void setUp() {
		TestUtils.clearAssertionInformation();
		addTestConfiguration( TEST_NAME1, new TestConfiguration(TEST_CLASS_DIR, TEST_NAME1, new String[] {"R"}) );
		addTestConfiguration( TEST_NAME2, new TestConfiguration(TEST_CLASS_DIR, TEST_NAME2, new String[] {"R"}) );
	}

	@Test
	public void testCodeMotionForCP() {
		testRewriteCodeMotion(TEST_NAME1, false, ExecType.CP);
	}
	
	@Test
	public void testCodeMotionForRewriteCP() {
		testRewriteCodeMotion(TEST_NAME1, true, ExecType.CP);
	}

	@Test
	public void testCodeMotionWhileCP() {
		testRewriteCodeMotion(TEST_NAME2, false, ExecType.CP);
	}
	
	@Test
	public void testCodeMotionWhileRewriteCP() {
		testRewriteCodeMotion(TEST_NAME2, true, ExecType.CP);
	}

	private void testRewriteCodeMotion(String testname, boolean rewrites, ExecType et)
	{
		ExecMode platformOld = setExecMode(et);
		boolean rewritesOld = OptimizerUtils.ALLOW_CODE_MOTION;
		OptimizerUtils.ALLOW_CODE_MOTION = rewrites;
		
		try
		{
			TestConfiguration config = getTestConfiguration(testname);
			loadTestConfiguration(config);
			
			String HOME = SCRIPT_DIR + TEST_DIR;
			fullDMLScriptName = HOME + testname + ".dml";
			programArgs = new String[] { "-explain", "hops", "-stats", "-args",
				input("X"), String.valueOf(iters), output("R") };
			fullRScriptName = HOME + testname + ".R";
			rCmd = getRCmd(inputDir(), String.valueOf(iters), expectedDir());

			double[][] X = getRandomMatrix(rows, cols, -1, 1, sparsity, 7);
			writeInputMatrixWithMTD("X", X, true);

			//execute tests
			runTest(true, false, null, -1); 
			runRScript(true); 
			
			//compare matrices 
			HashMap<CellIndex, Double> dmlfile = readDMLMatrixFromOutputDir("R");
			HashMap<CellIndex, Double> rfile  = readRMatrixFromExpectedDir("R");
			TestUtils.compareMatrices(dmlfile, rfile, eps, "Stat-DML", "Stat-R");
			
			//check applied code motion rewrites (moved sum and - from 10 to 1)
			Assert.assertEquals(rewrites?1:10, Statistics.getCPHeavyHitterCount(Opcodes.UAKP.toString()));
			Assert.assertEquals(rewrites?1:10, Statistics.getCPHeavyHitterCount("-"));
		}
		finally {
			OptimizerUtils.ALLOW_CODE_MOTION = rewritesOld;
			resetExecMode(platformOld);
		}
	}
}
