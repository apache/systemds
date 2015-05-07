/**
 * IBM Confidential
 * OCO Source Materials
 * (C) Copyright IBM Corp. 2010, 2015
 * The source code for this program is not published or otherwise divested of its trade secrets, irrespective of what has been deposited with the U.S. Copyright Office.
 */

package com.ibm.bi.dml.test.integration.functions.quaternary;

import java.util.HashMap;

import org.junit.Assert;
import org.junit.Test;

import com.ibm.bi.dml.api.DMLScript.RUNTIME_PLATFORM;
import com.ibm.bi.dml.hops.OptimizerUtils;
import com.ibm.bi.dml.lops.LopProperties.ExecType;
import com.ibm.bi.dml.lops.WeightedSquaredLoss;
import com.ibm.bi.dml.runtime.matrix.data.MatrixValue.CellIndex;
import com.ibm.bi.dml.test.integration.AutomatedTestBase;
import com.ibm.bi.dml.test.integration.TestConfiguration;
import com.ibm.bi.dml.test.utils.TestUtils;
import com.ibm.bi.dml.utils.Statistics;

/**
 * 
 * 
 */
public class WeightedSquaredLossTest extends AutomatedTestBase 
{
	@SuppressWarnings("unused")
	private static final String _COPYRIGHT = "Licensed Materials - Property of IBM\n(C) Copyright IBM Corp. 2010, 2015\n" +
                                             "US Government Users Restricted Rights - Use, duplication  disclosure restricted by GSA ADP Schedule Contract with IBM Corp.";
	
	private final static String TEST_NAME1 = "WeightedSquaredLossPost";
	private final static String TEST_NAME2 = "WeightedSquaredLossPre";
	private final static String TEST_NAME3 = "WeightedSquaredLossNo";

	
	private final static String TEST_DIR = "functions/quaternary/";
	
	private final static double eps = 1e-6;
	
	private final static int rows = 1201;
	private final static int cols = 1103;
	private final static int rank = 10;
	private final static double spSparse = 0.001;
	private final static double spDense = 0.6;
	
	@Override
	public void setUp() 
	{
		addTestConfiguration(TEST_NAME1,new TestConfiguration(TEST_DIR, TEST_NAME1,new String[]{"R"}));
		addTestConfiguration(TEST_NAME2,new TestConfiguration(TEST_DIR, TEST_NAME2,new String[]{"R"}));
		addTestConfiguration(TEST_NAME3,new TestConfiguration(TEST_DIR, TEST_NAME3,new String[]{"R"}));
	}

	
	@Test
	public void testSquaredLossDensePostWeightsNoRewritesCP() 
	{
		runMLUnaryBuiltinTest(TEST_NAME1, false, false, ExecType.CP);
	}
	
	@Test
	public void testSquaredLossDensePreWeightsNoRewritesCP() 
	{
		runMLUnaryBuiltinTest(TEST_NAME2, false, false, ExecType.CP);
	}
	
	@Test
	public void testSquaredLossDenseNoWeightsNoRewritesCP() 
	{
		runMLUnaryBuiltinTest(TEST_NAME3, false, false, ExecType.CP);
	}
	
	@Test
	public void testSquaredLossSparsePostWeightsNoRewritesCP() 
	{
		runMLUnaryBuiltinTest(TEST_NAME1, true, false, ExecType.CP);
	}
	
	@Test
	public void testSquaredLossSparsePreWeightsNoRewritesCP() 
	{
		runMLUnaryBuiltinTest(TEST_NAME2, true, false, ExecType.CP);
	}
	
	@Test
	public void testSquaredLossSparseNoWeightsNoRewritesCP() 
	{
		runMLUnaryBuiltinTest(TEST_NAME3, true, false, ExecType.CP);
	}
	
	@Test
	public void testSquaredLossDensePostWeightsNoRewritesMR() 
	{
		runMLUnaryBuiltinTest(TEST_NAME1, false, false, ExecType.MR);
	}
	
	@Test
	public void testSquaredLossDensePreWeightsNoRewritesMR() 
	{
		runMLUnaryBuiltinTest(TEST_NAME2, false, false, ExecType.MR);
	}
	
	@Test
	public void testSquaredLossDenseNoWeightsNoRewritesMR() 
	{
		runMLUnaryBuiltinTest(TEST_NAME3, false, false, ExecType.MR);
	}
	
	@Test
	public void testSquaredLossSparsePostWeightsNoRewritesMR() 
	{
		runMLUnaryBuiltinTest(TEST_NAME1, true, false, ExecType.MR);
	}
	
	@Test
	public void testSquaredLossSparsePreWeightsNoRewritesMR() 
	{
		runMLUnaryBuiltinTest(TEST_NAME2, true, false, ExecType.MR);
	}
	
	@Test
	public void testSquaredLossSparseNoWeightsNoRewritesMR() 
	{
		runMLUnaryBuiltinTest(TEST_NAME3, true, false, ExecType.MR);
	}
	
	//with rewrites
	
	@Test
	public void testSquaredLossDensePostWeightsRewritesCP() 
	{
		runMLUnaryBuiltinTest(TEST_NAME1, false, true, ExecType.CP);
	}
	
	@Test
	public void testSquaredLossDensePreWeightsRewritesCP() 
	{
		runMLUnaryBuiltinTest(TEST_NAME2, false, true, ExecType.CP);
	}
	
	@Test
	public void testSquaredLossDenseNoWeightsRewritesCP() 
	{
		runMLUnaryBuiltinTest(TEST_NAME3, false, true, ExecType.CP);
	}
	
	@Test
	public void testSquaredLossSparsePostWeightsRewritesCP() 
	{
		runMLUnaryBuiltinTest(TEST_NAME1, true, true, ExecType.CP);
	}
	
	@Test
	public void testSquaredLossSparsePreWeightsRewritesCP() 
	{
		runMLUnaryBuiltinTest(TEST_NAME2, true, true, ExecType.CP);
	}
	
	@Test
	public void testSquaredLossSparseNoWeightsRewritesCP() 
	{
		runMLUnaryBuiltinTest(TEST_NAME3, true, true, ExecType.CP);
	}


	@Test
	public void testSquaredLossDensePostWeightsRewritesMR() 
	{
		runMLUnaryBuiltinTest(TEST_NAME1, false, true, ExecType.MR);
	}
	
	@Test
	public void testSquaredLossDensePreWeightsRewritesMR() 
	{
		runMLUnaryBuiltinTest(TEST_NAME2, false, true, ExecType.MR);
	}
	
	@Test
	public void testSquaredLossDenseNoWeightsRewritesMR() 
	{
		runMLUnaryBuiltinTest(TEST_NAME3, false, true, ExecType.MR);
	}
	
	@Test
	public void testSquaredLossSparsePostWeightsRewritesMR() 
	{
		runMLUnaryBuiltinTest(TEST_NAME1, true, true, ExecType.MR);
	}
	
	@Test
	public void testSquaredLossSparsePreWeightsRewritesMR() 
	{
		runMLUnaryBuiltinTest(TEST_NAME2, true, true, ExecType.MR);
	}
	
	@Test
	public void testSquaredLossSparseNoWeightsRewritesMR() 
	{
		runMLUnaryBuiltinTest(TEST_NAME3, true, true, ExecType.MR);
	}

	
	
	
	/**
	 * 
	 * @param sparseM1
	 * @param sparseM2
	 * @param instType
	 */
	private void runMLUnaryBuiltinTest( String testname, boolean sparse, boolean rewrites, ExecType instType)
	{
		//keep old flags
		RUNTIME_PLATFORM platformOld = rtplatform;
		boolean rewritesOld = OptimizerUtils.ALLOW_ALGEBRAIC_SIMPLIFICATION;
		
		//set test-specific flags
		rtplatform = (instType==ExecType.MR) ? RUNTIME_PLATFORM.HADOOP : RUNTIME_PLATFORM.HYBRID;
	    OptimizerUtils.ALLOW_ALGEBRAIC_SIMPLIFICATION = rewrites;
		
		try
		{
			double sparsity = (sparse) ? spSparse : spDense;
			String TEST_NAME = testname;
			
			TestConfiguration config = getTestConfiguration(TEST_NAME);
			
			// This is for running the junit test the new way, i.e., construct the arguments directly
			String HOME = SCRIPT_DIR + TEST_DIR;
			fullDMLScriptName = HOME + TEST_NAME + ".dml";
			programArgs = new String[]{"-stats", "-explain", "runtime",
					                   "-args", 
					                        HOME + INPUT_DIR + "X",
					                        HOME + INPUT_DIR + "U",
					                        HOME + INPUT_DIR + "V",
					                        HOME + INPUT_DIR + "W",
					                        HOME + OUTPUT_DIR + "R"    };
			fullRScriptName = HOME + TEST_NAME + ".R";
			rCmd = "Rscript" + " " + fullRScriptName + " " + 
			       HOME + INPUT_DIR + " " + HOME + EXPECTED_DIR;
			
			loadTestConfiguration(config);
	
			//generate actual dataset 
			double[][] X = getRandomMatrix(rows, cols, 0, 1, sparsity, 7); 
			writeInputMatrixWithMTD("X", X, true);
			double[][] U = getRandomMatrix(rows, rank, 0, 1, 1.0, 213); 
			writeInputMatrixWithMTD("U", U, true);
			double[][] V = getRandomMatrix(cols, rank, 0, 1, 1.0, 312); 
			writeInputMatrixWithMTD("V", V, true);
			double[][] W = getRandomMatrix(rows, cols, 1, 1, sparsity, 1467); 
			writeInputMatrixWithMTD("W", W, true);
			
			runTest(true, false, null, -1); 
			runRScript(true); 
		
			//compare matrices 
			HashMap<CellIndex, Double> dmlfile = readDMLMatrixFromHDFS("R");
			HashMap<CellIndex, Double> rfile  = readRMatrixFromFS("R");
			TestUtils.compareMatrices(dmlfile, rfile, eps, "Stat-DML", "Stat-R");
			
			//check statistics for right operator in cp
			if( instType == ExecType.CP && rewrites )
				Assert.assertTrue(Statistics.getCPHeavyHitterOpCodes().contains(WeightedSquaredLoss.OPCODE_CP));
		}
		finally
		{
			rtplatform = platformOld;
			OptimizerUtils.ALLOW_ALGEBRAIC_SIMPLIFICATION = rewritesOld;
		}
	}	
}