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

package org.apache.sysml.test.integration.functions.binary.matrix;

import java.util.HashMap;

import org.junit.Assert;
import org.junit.Test;

import org.apache.sysml.api.DMLScript;
import org.apache.sysml.api.DMLScript.RUNTIME_PLATFORM;
import org.apache.sysml.hops.OptimizerUtils;
import org.apache.sysml.lops.MapMultChain;
import org.apache.sysml.lops.LopProperties.ExecType;
import org.apache.sysml.runtime.matrix.data.MatrixValue.CellIndex;
import org.apache.sysml.test.integration.AutomatedTestBase;
import org.apache.sysml.test.integration.TestConfiguration;
import org.apache.sysml.test.utils.TestUtils;
import org.apache.sysml.utils.Statistics;

public class MapMultChainTest extends AutomatedTestBase 
{
	
	private final static String TEST_NAME1 = "MapMultChain";
	private final static String TEST_NAME2 = "MapMultChainWeights";
	private final static String TEST_DIR = "functions/binary/matrix/";
	private final static String TEST_CLASS_DIR = TEST_DIR + MapMultChainTest.class.getSimpleName() + "/";
	private final static double eps = 1e-8;
	
	private final static int rowsX = 3468;
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
	}

	@Test
	public void testMapMultChainNoRewriteDenseCP() 
	{
		runMapMultChainTest(TEST_NAME1, false, false, ExecType.CP);
	}
	
	@Test
	public void testMapMultChainWeightsNoRewriteDenseCP() 
	{
		runMapMultChainTest(TEST_NAME2, false, false, ExecType.CP);
	}
	
	@Test
	public void testMapMultChainNoRewriteSparseCP() 
	{
		runMapMultChainTest(TEST_NAME1, true, false, ExecType.CP);
	}
	
	@Test
	public void testMapMultChainWeightsNoRewriteSparseCP() 
	{
		runMapMultChainTest(TEST_NAME2, true, false, ExecType.CP);
	}
	
	@Test
	public void testMapMultChainRewriteDenseCP() 
	{
		runMapMultChainTest(TEST_NAME1, false, true, ExecType.CP);
	}
	
	@Test
	public void testMapMultChainWeightsRewriteDenseCP() 
	{
		runMapMultChainTest(TEST_NAME2, false, true, ExecType.CP);
	}
	
	@Test
	public void testMapMultChainRewriteSparseCP() 
	{
		runMapMultChainTest(TEST_NAME1, true, true, ExecType.CP);
	}
	
	@Test
	public void testMapMultChainWeightsRewriteSparseCP() 
	{
		runMapMultChainTest(TEST_NAME2, true, true, ExecType.CP);
	}
	
	@Test
	public void testMapMultChainNoRewriteDenseMR() 
	{
		runMapMultChainTest(TEST_NAME1, false, false, ExecType.MR);
	}
	
	@Test
	public void testMapMultChainWeightsNoRewriteDenseMR() 
	{
		runMapMultChainTest(TEST_NAME2, false, false, ExecType.MR);
	}
	
	@Test
	public void testMapMultChainNoRewriteSparseMR() 
	{
		runMapMultChainTest(TEST_NAME1, true, false, ExecType.MR);
	}
	
	@Test
	public void testMapMultChainWeightsNoRewriteSparseMR() 
	{
		runMapMultChainTest(TEST_NAME2, true, false, ExecType.MR);
	}
	
	@Test
	public void testMapMultChainRewriteDenseMR() 
	{
		runMapMultChainTest(TEST_NAME1, false, true, ExecType.MR);
	}
	
	@Test
	public void testMapMultChainWeightsRewriteDenseMR() 
	{
		runMapMultChainTest(TEST_NAME2, false, true, ExecType.MR);
	}
	
	@Test
	public void testMapMultChainRewriteSparseMR() 
	{
		runMapMultChainTest(TEST_NAME1, true, true, ExecType.MR);
	}
	
	@Test
	public void testMapMultChainWeightsRewriteSparseMR() 
	{
		runMapMultChainTest(TEST_NAME2, true, true, ExecType.MR);
	}
	
	@Test
	public void testMapMultChainNoRewriteDenseSpark() 
	{
		runMapMultChainTest(TEST_NAME1, false, false, ExecType.SPARK);
	}
	
	@Test
	public void testMapMultChainWeightsNoRewriteDenseSpark() 
	{
		runMapMultChainTest(TEST_NAME2, false, false, ExecType.SPARK);
	}
	
	@Test
	public void testMapMultChainNoRewriteSparseSpark() 
	{
		runMapMultChainTest(TEST_NAME1, true, false, ExecType.SPARK);
	}
	
	@Test
	public void testMapMultChainWeightsNoRewriteSparseSpark() 
	{
		runMapMultChainTest(TEST_NAME2, true, false, ExecType.SPARK);
	}
	
	@Test
	public void testMapMultChainRewriteDenseSpark() 
	{
		runMapMultChainTest(TEST_NAME1, false, true, ExecType.SPARK);
	}
	
	@Test
	public void testMapMultChainWeightsRewriteDenseSpark() 
	{
		runMapMultChainTest(TEST_NAME2, false, true, ExecType.SPARK);
	}
	
	@Test
	public void testMapMultChainRewriteSparseSpark() 
	{
		runMapMultChainTest(TEST_NAME1, true, true, ExecType.SPARK);
	}
	
	@Test
	public void testMapMultChainWeightsRewriteSparseSpark() 
	{
		runMapMultChainTest(TEST_NAME2, true, true, ExecType.SPARK);
	}

	/**
	 * 
	 * @param sparseM1
	 * @param sparseM2
	 * @param instType
	 */
	private void runMapMultChainTest( String testname, boolean sparse, boolean sumProductRewrites, ExecType instType)
	{
		//rtplatform for MR
		RUNTIME_PLATFORM platformOld = rtplatform;
		switch( instType ){
			case MR: rtplatform = RUNTIME_PLATFORM.HADOOP; break;
			case SPARK: rtplatform = RUNTIME_PLATFORM.SPARK; break;
			default: rtplatform = RUNTIME_PLATFORM.HYBRID; break;
		}
	
		boolean sparkConfigOld = DMLScript.USE_LOCAL_SPARK_CONFIG;
		if( rtplatform == RUNTIME_PLATFORM.SPARK )
			DMLScript.USE_LOCAL_SPARK_CONFIG = true;
		
		//rewrite
		boolean rewritesOld = OptimizerUtils.ALLOW_SUM_PRODUCT_REWRITES;
		OptimizerUtils.ALLOW_SUM_PRODUCT_REWRITES = sumProductRewrites;
		
		try
		{
			String TEST_NAME = testname;
			getAndLoadTestConfiguration(TEST_NAME);
			
			/* This is for running the junit test the new way, i.e., construct the arguments directly */
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
			if( TEST_NAME.equals(TEST_NAME2) ){
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
			//changed 07/2015: by disabling the mm-transpose rewrite in forced mr/spark, the write is packed 
			//into the GMR for mapmult because the additional CP r' does not create a cut anymore.
			int expectedNumCompiled = (sumProductRewrites)?2:3; //GMR Reblock, 2x(GMR mapmult, incl write) -> GMR Reblock, GMR mapmultchain+write
			int expectedNumExecuted = expectedNumCompiled;
			if( instType != ExecType.MR ) {
				expectedNumCompiled = (instType==ExecType.SPARK)?0:1; //REBLOCK in CP
				expectedNumExecuted = 0;
			}
			checkNumCompiledMRJobs(expectedNumCompiled); 
			checkNumExecutedMRJobs(expectedNumExecuted); 
			
			//check compiled mmchain instructions (cp/spark)
			if( instType != ExecType.MR ){
				String opcode = (instType==ExecType.CP)?MapMultChain.OPCODE_CP:"sp_"+MapMultChain.OPCODE;
				Assert.assertEquals(sumProductRewrites, Statistics.getCPHeavyHitterOpCodes().contains(opcode));
			}
		}
		finally
		{
			rtplatform = platformOld;
			DMLScript.USE_LOCAL_SPARK_CONFIG = sparkConfigOld;
			OptimizerUtils.ALLOW_SUM_PRODUCT_REWRITES = rewritesOld;
		}
	}

}