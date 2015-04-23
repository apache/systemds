/**
 * IBM Confidential
 * OCO Source Materials
 * (C) Copyright IBM Corp. 2010, 2014
 * The source code for this program is not published or otherwise divested of its trade secrets, irrespective of what has been deposited with the U.S. Copyright Office.
 */

package com.ibm.bi.dml.test.integration.functions.binary.matrix;

import java.util.HashMap;

import org.junit.Assert;
import org.junit.Test;

import com.ibm.bi.dml.api.DMLScript;
import com.ibm.bi.dml.api.DMLScript.RUNTIME_PLATFORM;
import com.ibm.bi.dml.hops.OptimizerUtils;
import com.ibm.bi.dml.lops.MapMultChain;
import com.ibm.bi.dml.lops.LopProperties.ExecType;
import com.ibm.bi.dml.runtime.matrix.data.MatrixValue.CellIndex;
import com.ibm.bi.dml.test.integration.AutomatedTestBase;
import com.ibm.bi.dml.test.integration.TestConfiguration;
import com.ibm.bi.dml.test.utils.TestUtils;
import com.ibm.bi.dml.utils.Statistics;

public class MapMultChainTest extends AutomatedTestBase 
{
	@SuppressWarnings("unused")
	private static final String _COPYRIGHT = "Licensed Materials - Property of IBM\n(C) Copyright IBM Corp. 2010, 2014\n" +
                                             "US Government Users Restricted Rights - Use, duplication  disclosure restricted by GSA ADP Schedule Contract with IBM Corp.";
	
	private final static String TEST_NAME1 = "MapMultChain";
	private final static String TEST_NAME2 = "MapMultChainWeights";
	private final static String TEST_DIR = "functions/binary/matrix/";
	private final static double eps = 1e-8;
	
	private final static int rowsX = 3468;
	private final static int colsX = 567;
	
	private final static double sparsity1 = 0.7;
	private final static double sparsity2 = 0.1;
	
	
	@Override
	public void setUp() 
	{
		TestUtils.clearAssertionInformation();
		addTestConfiguration(TEST_NAME1, new TestConfiguration(TEST_DIR, TEST_NAME1, new String[] { "R" })); 
		addTestConfiguration(TEST_NAME2, new TestConfiguration(TEST_DIR, TEST_NAME2, new String[] { "R" })); 
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
			TestConfiguration config = getTestConfiguration(TEST_NAME);
			
			/* This is for running the junit test the new way, i.e., construct the arguments directly */
			String HOME = SCRIPT_DIR + TEST_DIR;
			fullDMLScriptName = HOME + TEST_NAME + ".dml";
			programArgs = new String[]{"-stats","-args", HOME + INPUT_DIR + "X",
					                            HOME + INPUT_DIR + "v",
					                            HOME + INPUT_DIR + "w",
					                            HOME + OUTPUT_DIR + "R"};
			fullRScriptName = HOME + TEST_NAME + ".R";
			rCmd = "Rscript" + " " + fullRScriptName + " " + 
			       HOME + INPUT_DIR + " " + HOME + EXPECTED_DIR;
			
			loadTestConfiguration(config);
	
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
			int expectedNumCompiled = (sumProductRewrites)?2:4; //GMR Reblock, 2x(GMR mapmult), GMR write -> GMR Reblock, GMR mapmultchain+write
			int expectedNumExecuted = expectedNumCompiled;
			if( instType != ExecType.MR ) {
				expectedNumCompiled = 1; //REBLOCK
				expectedNumExecuted = 0;
			}
			checkNumCompiledMRJobs(expectedNumCompiled); 
			checkNumExecutedMRJobs(expectedNumExecuted); 
			
			//check compiled mmchain instructions (cp/spark)
			if( instType != ExecType.MR ){
				String opcode = (instType==ExecType.CP)?MapMultChain.OPCODE_CP:MapMultChain.OPCODE;
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