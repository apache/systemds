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

package org.apache.sysds.test.functions.aggregate;

import java.util.HashMap;

import org.junit.AfterClass;
import org.junit.Assert;
import org.junit.BeforeClass;
import org.junit.Test;
import org.apache.sysds.common.Types.ExecMode;
import org.apache.sysds.common.Types.ExecType;
import org.apache.sysds.runtime.matrix.data.MatrixValue.CellIndex;
import org.apache.sysds.test.AutomatedTestBase;
import org.apache.sysds.test.TestConfiguration;
import org.apache.sysds.test.TestUtils;
import org.apache.sysds.utils.Statistics;

/**
 * 
 */
public class PushdownSumBinaryTest extends AutomatedTestBase 
{
	private final static String TEST_NAME1 = "PushdownSum1"; //+
	private final static String TEST_NAME2 = "PushdownSum2"; //-
	
	private final static String TEST_DIR = "functions/aggregate/";
	private static final String TEST_CLASS_DIR = TEST_DIR + PushdownSumBinaryTest.class.getSimpleName() + "/";
	private final static double eps = 1e-10;
	
	private final static int rows = 1765;
	private final static int cols = 19;
	private final static double sparsity = 0.1;
	
	
	@Override
	public void setUp() 
	{
		addTestConfiguration(TEST_NAME1, new TestConfiguration(TEST_CLASS_DIR, TEST_NAME1, new String[]{"C"})); 
		addTestConfiguration(TEST_NAME2, new TestConfiguration(TEST_CLASS_DIR, TEST_NAME2, new String[]{"C"})); 
		TestUtils.clearAssertionInformation();

		if (TEST_CACHE_ENABLED) {
			setOutAndExpectedDeletionDisabled(true);
		}
	}

	@BeforeClass
	public static void init()
	{
		TestUtils.clearDirectory(TEST_DATA_DIR + TEST_CLASS_DIR);
	}

	@AfterClass
	public static void cleanUp()
	{
		if (TEST_CACHE_ENABLED) {
			TestUtils.clearDirectory(TEST_DATA_DIR + TEST_CLASS_DIR);
		}
	}

	@Test
	public void testPushDownSumPlusSP() {
		runPushdownSumOnBinaryTest(TEST_NAME1, true, ExecType.SPARK);
	}
	
	@Test
	public void testPushDownSumMinusSP() {
		runPushdownSumOnBinaryTest(TEST_NAME2, true, ExecType.SPARK);
	}
	
	@Test
	public void testPushDownSumPlusBroadcastSP() {
		runPushdownSumOnBinaryTest(TEST_NAME1, false, ExecType.SPARK);
	}
	
	@Test
	public void testPushDownSumMinusBroadcastSP() {
		runPushdownSumOnBinaryTest(TEST_NAME2, false, ExecType.SPARK);
	}
	
	private void runPushdownSumOnBinaryTest( String testname, boolean equiDims, ExecType instType) 
	{
		//rtplatform for MR
		ExecMode platformOld = setExecMode(instType);
			
		try
		{
			//determine script and function name
			String TEST_NAME = testname;
			String TEST_CACHE_DIR = TEST_CACHE_ENABLED ? TEST_NAME + "_" + String.valueOf(equiDims) + "/" : "";
			
			TestConfiguration config = getTestConfiguration(TEST_NAME);
			loadTestConfiguration(config, TEST_CACHE_DIR);
			
			// This is for running the junit test the new way, i.e., construct the arguments directly
			String HOME = SCRIPT_DIR + TEST_DIR;
			fullDMLScriptName = HOME + TEST_NAME + ".dml";
			programArgs = new String[]{"-explain","-stats","-args", input("A"), input("B"), output("C") };
			fullRScriptName = HOME + TEST_NAME + ".R";
			rCmd = "Rscript" + " " + fullRScriptName + " " + inputDir() + " " + expectedDir();
	
			//generate actual dataset 
			double[][] A = getRandomMatrix(rows, cols, -1, 1, sparsity, 7); 
			writeInputMatrixWithMTD("A", A, true);
			double[][] B = getRandomMatrix(rows, equiDims ? cols : 1, -1, 1, sparsity, 73); 
			writeInputMatrixWithMTD("B", B, true);
			
			//run tests
			runTest(true, false, null, -1); 
			runRScript(true); 
			
			//compare output matrices
			HashMap<CellIndex, Double> dmlfile = readDMLMatrixFromOutputDir("C");
			HashMap<CellIndex, Double> rfile  = readRMatrixFromExpectedDir("C");
			TestUtils.compareMatrices(dmlfile, rfile, eps, "Stat-DML", "Stat-R");
			
			String lopcode = TEST_NAME.equals(TEST_NAME1) ? "+" : "-";
			Assert.assertTrue("Non-applied rewrite", Statistics.getCPHeavyHitterOpCodes().contains(lopcode));
		}
		finally {
			resetExecMode(platformOld);
		}
	}
}
