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
import org.apache.sysds.hops.OptimizerUtils;
import org.apache.sysds.runtime.matrix.data.MatrixValue.CellIndex;
import org.apache.sysds.test.AutomatedTestBase;
import org.apache.sysds.test.TestConfiguration;
import org.apache.sysds.test.TestUtils;
import org.apache.sysds.utils.Statistics;

/**
 * Regression test for function recompile-once issue with literal replacement.
 * 
 */
public class RewriteSimplifyRowColSumMVMultTest extends AutomatedTestBase 
{
	private static final String TEST_NAME1 = "RewriteRowSumsMVMult";
	private static final String TEST_NAME2 = "RewriteRowSumsMVMult";
	private static final String TEST_DIR = "functions/rewrite/";
	private static final String TEST_CLASS_DIR = TEST_DIR + RewriteSimplifyRowColSumMVMultTest.class.getSimpleName() + "/";
	
	private static final int rows = 1234;
	private static final int cols = 567;
	private static final double eps = Math.pow(10, -10);
	
	@Override
	public void setUp() {
		TestUtils.clearAssertionInformation();
		addTestConfiguration( TEST_NAME1, new TestConfiguration(TEST_CLASS_DIR, TEST_NAME1, new String[] { "R" }) );
		addTestConfiguration( TEST_NAME2, new TestConfiguration(TEST_CLASS_DIR, TEST_NAME2, new String[] { "R" }) );
	}

	@Test
	public void testMultiScalarToBinaryNoRewrite() {
		testRewriteRowColSumsMVMult( TEST_NAME1, false );
	}
	
	@Test
	public void testMultiScalarToBinaryRewrite() {
		testRewriteRowColSumsMVMult( TEST_NAME1, true );
	}
	
	@Test
	public void testMultiBinaryToScalarNoRewrite() {
		testRewriteRowColSumsMVMult( TEST_NAME2, false );
	}
	
	@Test
	public void testMultiBinaryToScalarRewrite() {
		testRewriteRowColSumsMVMult( TEST_NAME2, true );
	}
	
	private void testRewriteRowColSumsMVMult( String testname, boolean rewrites )
	{
		boolean oldFlag = OptimizerUtils.ALLOW_ALGEBRAIC_SIMPLIFICATION;
		
		try
		{
			TestConfiguration config = getTestConfiguration(testname);
			loadTestConfiguration(config);
			
			String HOME = SCRIPT_DIR + TEST_DIR;
			fullDMLScriptName = HOME + testname + ".dml";
			programArgs = new String[]{ "-stats","-args", input("X"), output("R") };
			
			fullRScriptName = HOME + testname + ".R";
			rCmd = getRCmd(inputDir(), expectedDir());

			OptimizerUtils.ALLOW_ALGEBRAIC_SIMPLIFICATION = rewrites;

			double[][] X = getRandomMatrix(rows, cols, -1, 1, 0.56d, 7);
			writeInputMatrixWithMTD("X", X, true);
			
			runTest(true, false, null, -1); 
			runRScript(true); 
			
			//compare matrices 
			HashMap<CellIndex, Double> dmlfile = readDMLMatrixFromOutputDir("R");
			HashMap<CellIndex, Double> rfile  = readRMatrixFromExpectedDir("R");
			TestUtils.compareMatrices(dmlfile, rfile, eps, "Stat-DML", "Stat-R");
			
			//check matrix mult existence
			String gpuBa = "gpu_ba+*";
			String ba = Opcodes.MMULT.toString();
			boolean isMatmultPresent = Statistics.getCPHeavyHitterOpCodes().contains(ba) ||  Statistics.getCPHeavyHitterOpCodes().contains(gpuBa);
			Assert.assertTrue( isMatmultPresent == rewrites );
		}
		finally {
			OptimizerUtils.ALLOW_ALGEBRAIC_SIMPLIFICATION = oldFlag;
		}
	}
}
