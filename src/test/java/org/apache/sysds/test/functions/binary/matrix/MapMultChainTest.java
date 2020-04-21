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

package org.apache.sysds.test.functions.binary.matrix;

import java.util.HashMap;

import org.junit.AfterClass;
import org.junit.BeforeClass;
import org.junit.Test;
import org.apache.sysds.common.Types.ExecMode;
import org.apache.sysds.hops.OptimizerUtils;
import org.apache.sysds.lops.LopProperties.ExecType;
import org.apache.sysds.runtime.matrix.data.MatrixValue.CellIndex;
import org.apache.sysds.test.AutomatedTestBase;
import org.apache.sysds.test.TestConfiguration;
import org.apache.sysds.test.TestUtils;

public class MapMultChainTest extends AutomatedTestBase 
{
	private final static String TEST_NAME1 = "MapMultChain";
	private final static String TEST_NAME2 = "MapMultChainWeights";
	private final static String TEST_NAME3 = "MapMultChainWeights2";
	private final static String TEST_DIR = "functions/binary/matrix/";
	private final static String TEST_CLASS_DIR = TEST_DIR + MapMultChainTest.class.getSimpleName() + "/";
	private final static double eps = 1e-8;
	
	private final static int rowsX = 3471; //mod 8 = 7
	private final static int colsX = 567;
	
	private final static double sparsity1 = 0.7;
	private final static double sparsity2 = 0.1;
	
	
	@Override
	public void setUp() 
	{
		TestUtils.clearAssertionInformation();
		addTestConfiguration(TEST_NAME1, 
			new TestConfiguration(TEST_CLASS_DIR, TEST_NAME1, new String[] { "R" })); 
		addTestConfiguration(TEST_NAME2, 
			new TestConfiguration(TEST_CLASS_DIR, TEST_NAME2, new String[] { "R" })); 
		addTestConfiguration(TEST_NAME3, 
				new TestConfiguration(TEST_CLASS_DIR, TEST_NAME3, new String[] { "R" })); 
		
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
	public static void cleanUp() {
		if (TEST_CACHE_ENABLED) {
			TestUtils.clearDirectory(TEST_DATA_DIR + TEST_CLASS_DIR);
		}
	}
	
	@Test
	public void testMapMultChainNoRewriteDenseCP() {
		runMapMultChainTest(TEST_NAME1, false, false, ExecType.CP);
	}
	
	@Test
	public void testMapMultChainWeightsNoRewriteDenseCP() {
		runMapMultChainTest(TEST_NAME2, false, false, ExecType.CP);
	}
	
	@Test
	public void testMapMultChainWeights2NoRewriteDenseCP() {
		runMapMultChainTest(TEST_NAME3, false, false, ExecType.CP);
	}
	
	@Test
	public void testMapMultChainNoRewriteSparseCP() {
		runMapMultChainTest(TEST_NAME1, true, false, ExecType.CP);
	}
	
	@Test
	public void testMapMultChainWeightsNoRewriteSparseCP() {
		runMapMultChainTest(TEST_NAME2, true, false, ExecType.CP);
	}
	
	@Test
	public void testMapMultChainWeights2NoRewriteSparseCP() {
		runMapMultChainTest(TEST_NAME3, true, false, ExecType.CP);
	}
	
	@Test
	public void testMapMultChainRewriteDenseCP() {
		runMapMultChainTest(TEST_NAME1, false, true, ExecType.CP);
	}
	
	@Test
	public void testMapMultChainWeightsRewriteDenseCP() {
		runMapMultChainTest(TEST_NAME2, false, true, ExecType.CP);
	}
	
	@Test
	public void testMapMultChainWeights2RewriteDenseCP() {
		runMapMultChainTest(TEST_NAME3, false, true, ExecType.CP);
	}
	
	@Test
	public void testMapMultChainRewriteSparseCP() {
		runMapMultChainTest(TEST_NAME1, true, true, ExecType.CP);
	}
	
	@Test
	public void testMapMultChainWeightsRewriteSparseCP() {
		runMapMultChainTest(TEST_NAME2, true, true, ExecType.CP);
	}
	
	@Test
	public void testMapMultChainWeights2RewriteSparseCP() {
		runMapMultChainTest(TEST_NAME3, true, true, ExecType.CP);
	}
	
	@Test
	public void testMapMultChainNoRewriteDenseSpark() {
		runMapMultChainTest(TEST_NAME1, false, false, ExecType.SPARK);
	}
	
	@Test
	public void testMapMultChainWeightsNoRewriteDenseSpark() {
		runMapMultChainTest(TEST_NAME2, false, false, ExecType.SPARK);
	}
	
	@Test
	public void testMapMultChainWeights2NoRewriteDenseSpark() {
		runMapMultChainTest(TEST_NAME3, false, false, ExecType.SPARK);
	}
	
	@Test
	public void testMapMultChainNoRewriteSparseSpark() {
		runMapMultChainTest(TEST_NAME1, true, false, ExecType.SPARK);
	}
	
	@Test
	public void testMapMultChainWeightsNoRewriteSparseSpark() {
		runMapMultChainTest(TEST_NAME2, true, false, ExecType.SPARK);
	}
	
	@Test
	public void testMapMultChainWeights2NoRewriteSparseSpark() {
		runMapMultChainTest(TEST_NAME3, true, false, ExecType.SPARK);
	}
	
	@Test
	public void testMapMultChainRewriteDenseSpark() {
		runMapMultChainTest(TEST_NAME1, false, true, ExecType.SPARK);
	}
	
	@Test
	public void testMapMultChainWeightsRewriteDenseSpark() {
		runMapMultChainTest(TEST_NAME2, false, true, ExecType.SPARK);
	}
	
	@Test
	public void testMapMultChainWeights2RewriteDenseSpark() {
		runMapMultChainTest(TEST_NAME3, false, true, ExecType.SPARK);
	}
	
	@Test
	public void testMapMultChainRewriteSparseSpark() {
		runMapMultChainTest(TEST_NAME1, true, true, ExecType.SPARK);
	}
	
	@Test
	public void testMapMultChainWeightsRewriteSparseSpark() {
		runMapMultChainTest(TEST_NAME2, true, true, ExecType.SPARK);
	}
	
	@Test
	public void testMapMultChainWeights2RewriteSparseSpark() {
		runMapMultChainTest(TEST_NAME3, true, true, ExecType.SPARK);
	}
	
	private void runMapMultChainTest( String testname, boolean sparse, boolean sumProductRewrites, ExecType instType)
	{
		ExecMode platformOld = setExecMode(instType);
		
		//rewrite
		boolean rewritesOld = OptimizerUtils.ALLOW_SUM_PRODUCT_REWRITES;
		OptimizerUtils.ALLOW_SUM_PRODUCT_REWRITES = sumProductRewrites;
		
		try
		{
			String TEST_NAME = testname;
			TestConfiguration config = getTestConfiguration(TEST_NAME);
			String TEST_CACHE_DIR = (TEST_CACHE_ENABLED)? TEST_NAME + "_" + sparse + "/" : "";
			loadTestConfiguration(config, TEST_CACHE_DIR);
			
			String HOME = SCRIPT_DIR + TEST_DIR;
			fullDMLScriptName = HOME + TEST_NAME + ".dml";
			programArgs = new String[]{"-stats","-args", 
				input("X"), input("v"), input("w"), output("R")};
			
			fullRScriptName = HOME + TEST_NAME + ".R";
			rCmd = "Rscript" + " " + fullRScriptName + " " + inputDir() + " " + expectedDir();
	
			//generate actual datasets
			double[][] X = getRandomMatrix(rowsX, colsX, 0, 1, sparse?sparsity2:sparsity1, 7);
			writeInputMatrixWithMTD("X", X, true);
			double[][] v = getRandomMatrix(colsX, 1, 0, 1, sparsity1, 3);
			writeInputMatrixWithMTD("v", v, true);
			if( !TEST_NAME.equals(TEST_NAME1) ){
				double[][] w = getRandomMatrix(rowsX, 1, 0, 1, sparse?sparsity2:sparsity1, 10);
				writeInputMatrixWithMTD("w", w, true);
			}
			
			runTest(true, false, null, -1); 
			runRScript(true); 
			
			//compare matrices 
			HashMap<CellIndex, Double> dmlfile = readDMLMatrixFromHDFS("R");
			HashMap<CellIndex, Double> rfile  = readRMatrixFromFS("R");
			TestUtils.compareMatrices(dmlfile, rfile, eps, "Stat-DML", "Stat-R");
			
			//check compiled/executed jobs
			int numInputs = testname.equals(TEST_NAME1) ? 2 : 3;
			int expectedNumCompiled = numInputs + ((instType==ExecType.SPARK) ? 
				(numInputs + (sumProductRewrites?2:((numInputs==2)?4:5))):0);
			checkNumCompiledSparkInst(expectedNumCompiled); 
			checkNumExecutedSparkInst(expectedNumCompiled 
				- ((instType==ExecType.CP)?numInputs:0));
		}
		finally {
			rtplatform = platformOld;
			OptimizerUtils.ALLOW_SUM_PRODUCT_REWRITES = rewritesOld;
		}
	}
}
