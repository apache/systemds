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

import com.ibm.bi.dml.api.DMLScript;
import com.ibm.bi.dml.api.DMLScript.RUNTIME_PLATFORM;
import com.ibm.bi.dml.hops.OptimizerUtils;
import com.ibm.bi.dml.hops.QuaternaryOp;
import com.ibm.bi.dml.lops.LopProperties.ExecType;
import com.ibm.bi.dml.lops.WeightedSigmoid;
import com.ibm.bi.dml.runtime.matrix.data.MatrixValue.CellIndex;
import com.ibm.bi.dml.test.integration.AutomatedTestBase;
import com.ibm.bi.dml.test.integration.TestConfiguration;
import com.ibm.bi.dml.test.utils.TestUtils;
import com.ibm.bi.dml.utils.Statistics;

/**
 * 
 * 
 */
public class WeightedSigmoidTest extends AutomatedTestBase 
{
	@SuppressWarnings("unused")
	private static final String _COPYRIGHT = "Licensed Materials - Property of IBM\n(C) Copyright IBM Corp. 2010, 2015\n" +
                                             "US Government Users Restricted Rights - Use, duplication  disclosure restricted by GSA ADP Schedule Contract with IBM Corp.";
	
	private final static String TEST_NAME1 = "WeightedSigmoidP1";
	private final static String TEST_NAME2 = "WeightedSigmoidP2";
	private final static String TEST_NAME3 = "WeightedSigmoidP3";
	private final static String TEST_NAME4 = "WeightedSigmoidP4";

	private final static String TEST_DIR = "functions/quaternary/";
	
	private final static double eps = 1e-10;
	
	private final static int rows = 1201;
	private final static int cols = 1103;
	private final static int rank = 10;
	private final static double spSparse = 0.001;
	private final static double spDense = 0.6;
	
	@Override
	public void setUp() 
	{
		TestUtils.clearAssertionInformation();
		addTestConfiguration(TEST_NAME1,new TestConfiguration(TEST_DIR, TEST_NAME1,new String[]{"R"}));
		addTestConfiguration(TEST_NAME2,new TestConfiguration(TEST_DIR, TEST_NAME2,new String[]{"R"}));
		addTestConfiguration(TEST_NAME3,new TestConfiguration(TEST_DIR, TEST_NAME3,new String[]{"R"}));
		addTestConfiguration(TEST_NAME4,new TestConfiguration(TEST_DIR, TEST_NAME4,new String[]{"R"}));
	}

	
	@Test
	public void testSigmoidDenseBasicNoRewritesCP() 
	{
		runMLUnaryBuiltinTest(TEST_NAME1, false, false, ExecType.CP);
	}
	
	@Test
	public void testSigmoidDenseLogNoRewritesCP() 
	{
		runMLUnaryBuiltinTest(TEST_NAME2, false, false, ExecType.CP);
	}
	
	@Test
	public void testSigmoidDenseMinusNoRewritesCP() 
	{
		runMLUnaryBuiltinTest(TEST_NAME3, false, false, ExecType.CP);
	}
	
	@Test
	public void testSigmoidDenseLogMinusNoRewritesCP() 
	{
		runMLUnaryBuiltinTest(TEST_NAME4, false, false, ExecType.CP);
	}

	@Test
	public void testSigmoidSparseBasicNoRewritesCP() 
	{
		runMLUnaryBuiltinTest(TEST_NAME1, true, false, ExecType.CP);
	}
	
	@Test
	public void testSigmoidSparseLogNoRewritesCP() 
	{
		runMLUnaryBuiltinTest(TEST_NAME2, true, false, ExecType.CP);
	}
	
	@Test
	public void testSigmoidSparseMinusNoRewritesCP() 
	{
		runMLUnaryBuiltinTest(TEST_NAME3, true, false, ExecType.CP);
	}
	
	@Test
	public void testSigmoidSparseLogMinusNoRewritesCP() 
	{
		runMLUnaryBuiltinTest(TEST_NAME4, true, false, ExecType.CP);
	}

	@Test
	public void testSigmoidDenseBasicNoRewritesMR() 
	{
		runMLUnaryBuiltinTest(TEST_NAME1, false, false, ExecType.MR);
	}
	
	@Test
	public void testSigmoidDenseLogNoRewritesMR() 
	{
		runMLUnaryBuiltinTest(TEST_NAME2, false, false, ExecType.MR);
	}
	
	@Test
	public void testSigmoidDenseMinusNoRewritesMR() 
	{
		runMLUnaryBuiltinTest(TEST_NAME3, false, false, ExecType.MR);
	}
	
	@Test
	public void testSigmoidDenseLogMinusNoRewritesMR() 
	{
		runMLUnaryBuiltinTest(TEST_NAME4, false, false, ExecType.MR);
	}

	@Test
	public void testSigmoidSparseBasicNoRewritesMR() 
	{
		runMLUnaryBuiltinTest(TEST_NAME1, true, false, ExecType.MR);
	}
	
	@Test
	public void testSigmoidSparseLogNoRewritesMR() 
	{
		runMLUnaryBuiltinTest(TEST_NAME2, true, false, ExecType.MR);
	}
	
	@Test
	public void testSigmoidSparseMinusNoRewritesMR() 
	{
		runMLUnaryBuiltinTest(TEST_NAME3, true, false, ExecType.MR);
	}
	
	@Test
	public void testSigmoidSparseLogMinusNoRewritesMR() 
	{
		runMLUnaryBuiltinTest(TEST_NAME4, true, false, ExecType.MR);
	}
	
	//with rewrites

	@Test
	public void testSigmoidDenseBasicRewritesCP() 
	{
		runMLUnaryBuiltinTest(TEST_NAME1, false, true, ExecType.CP);
	}
	
	@Test
	public void testSigmoidDenseLogRewritesCP() 
	{
		runMLUnaryBuiltinTest(TEST_NAME2, false, true, ExecType.CP);
	}
	
	@Test
	public void testSigmoidDenseMinusRewritesCP() 
	{
		runMLUnaryBuiltinTest(TEST_NAME3, false, true, ExecType.CP);
	}
	
	@Test
	public void testSigmoidDenseLogMinusRewritesCP() 
	{
		runMLUnaryBuiltinTest(TEST_NAME4, false, true, ExecType.CP);
	}

	@Test
	public void testSigmoidSparseBasicRewritesCP() 
	{
		runMLUnaryBuiltinTest(TEST_NAME1, true, true, ExecType.CP);
	}
	
	@Test
	public void testSigmoidSparseLogRewritesCP() 
	{
		runMLUnaryBuiltinTest(TEST_NAME2, true, true, ExecType.CP);
	}
	
	@Test
	public void testSigmoidSparseMinusRewritesCP() 
	{
		runMLUnaryBuiltinTest(TEST_NAME3, true, true, ExecType.CP);
	}
	
	@Test
	public void testSigmoidSparseLogMinusRewritesCP() 
	{
		runMLUnaryBuiltinTest(TEST_NAME4, true, true, ExecType.CP);
	}

	@Test
	public void testSigmoidDenseBasicRewritesMR() 
	{
		runMLUnaryBuiltinTest(TEST_NAME1, false, true, ExecType.MR);
	}
	
	@Test
	public void testSigmoidDenseLogRewritesMR() 
	{
		runMLUnaryBuiltinTest(TEST_NAME2, false, true, ExecType.MR);
	}
	
	@Test
	public void testSigmoidDenseMinusRewritesMR() 
	{
		runMLUnaryBuiltinTest(TEST_NAME3, false, true, ExecType.MR);
	}
	
	@Test
	public void testSigmoidDenseLogMinusRewritesMR() 
	{
		runMLUnaryBuiltinTest(TEST_NAME4, false, true, ExecType.MR);
	}

	@Test
	public void testSigmoidSparseBasicRewritesMR() 
	{
		runMLUnaryBuiltinTest(TEST_NAME1, true, true, ExecType.MR);
	}
	
	@Test
	public void testSigmoidSparseLogRewritesMR() 
	{
		runMLUnaryBuiltinTest(TEST_NAME2, true, true, ExecType.MR);
	}
	
	@Test
	public void testSigmoidSparseMinusRewritesMR() 
	{
		runMLUnaryBuiltinTest(TEST_NAME3, true, true, ExecType.MR);
	}
	
	@Test
	public void testSigmoidSparseLogMinusRewritesMR() 
	{
		runMLUnaryBuiltinTest(TEST_NAME4, true, true, ExecType.MR);
	}

	@Test
	public void testSigmoidDenseBasicRewritesSP() 
	{
		runMLUnaryBuiltinTest(TEST_NAME1, false, true, ExecType.SPARK);
	}
	
	@Test
	public void testSigmoidDenseLogRewritesSP() 
	{
		runMLUnaryBuiltinTest(TEST_NAME2, false, true, ExecType.SPARK);
	}
	
	@Test
	public void testSigmoidDenseMinusRewritesSP() 
	{
		runMLUnaryBuiltinTest(TEST_NAME3, false, true, ExecType.SPARK);
	}
	
	@Test
	public void testSigmoidDenseLogMinusRewritesSP() 
	{
		runMLUnaryBuiltinTest(TEST_NAME4, false, true, ExecType.SPARK);
	}

	@Test
	public void testSigmoidSparseBasicRewritesSP() 
	{
		runMLUnaryBuiltinTest(TEST_NAME1, true, true, ExecType.SPARK);
	}
	
	@Test
	public void testSigmoidSparseLogRewritesSP() 
	{
		runMLUnaryBuiltinTest(TEST_NAME2, true, true, ExecType.SPARK);
	}
	
	@Test
	public void testSigmoidSparseMinusRewritesSP() 
	{
		runMLUnaryBuiltinTest(TEST_NAME3, true, true, ExecType.SPARK);
	}
	
	@Test
	public void testSigmoidSparseLogMinusRewritesSP() 
	{
		runMLUnaryBuiltinTest(TEST_NAME4, true, true, ExecType.SPARK);
	}
	
	
	//the following tests force the replication based mr operator because
	//otherwise we would always choose broadcasts for this small input data
	
	@Test
	public void testSigmoidSparseBasicRewritesRepMR() 
	{
		runMLUnaryBuiltinTest(TEST_NAME1, true, true, true, ExecType.MR);
	}
	
	@Test
	public void testSigmoidSparseLogRewritesRepMR() 
	{
		runMLUnaryBuiltinTest(TEST_NAME2, true, true, true, ExecType.MR);
	}
	
	@Test
	public void testSigmoidSparseMinusRewritesRepMR() 
	{
		runMLUnaryBuiltinTest(TEST_NAME3, true, true, true, ExecType.MR);
	}
	
	@Test
	public void testSigmoidSparseLogMinusRewritesRepMR() 
	{
		runMLUnaryBuiltinTest(TEST_NAME4, true, true, true, ExecType.MR);
	}
	
	@Test
	public void testSigmoidDenseBasicRewritesRepMR() 
	{
		runMLUnaryBuiltinTest(TEST_NAME1, false, true, true, ExecType.MR);
	}
	
	@Test
	public void testSigmoidDenseLogRewritesRepMR() 
	{
		runMLUnaryBuiltinTest(TEST_NAME2, false, true, true, ExecType.MR);
	}
	
	@Test
	public void testSigmoidDenseMinusRewritesRepMR() 
	{
		runMLUnaryBuiltinTest(TEST_NAME3, false, true, true, ExecType.MR);
	}
	
	@Test
	public void testSigmoidDenseLogMinusRewritesRepMR() 
	{
		runMLUnaryBuiltinTest(TEST_NAME4, false, true, true, ExecType.MR);
	}
	

	@Test
	public void testSigmoidSparseBasicRewritesRepSP() 
	{
		runMLUnaryBuiltinTest(TEST_NAME1, true, true, true, ExecType.SPARK);
	}
	
	@Test
	public void testSigmoidSparseLogRewritesRepSP() 
	{
		runMLUnaryBuiltinTest(TEST_NAME2, true, true, true, ExecType.SPARK);
	}
	
	@Test
	public void testSigmoidSparseMinusRewritesRepSP() 
	{
		runMLUnaryBuiltinTest(TEST_NAME3, true, true, true, ExecType.SPARK);
	}
	
	@Test
	public void testSigmoidSparseLogMinusRewritesRepSP() 
	{
		runMLUnaryBuiltinTest(TEST_NAME4, true, true, true, ExecType.SPARK);
	}
	
	@Test
	public void testSigmoidDenseBasicRewritesRepSP() 
	{
		runMLUnaryBuiltinTest(TEST_NAME1, false, true, true, ExecType.SPARK);
	}
	
	@Test
	public void testSigmoidDenseLogRewritesRepSP() 
	{
		runMLUnaryBuiltinTest(TEST_NAME2, false, true, true, ExecType.SPARK);
	}
	
	@Test
	public void testSigmoidDenseMinusRewritesRepSP() 
	{
		runMLUnaryBuiltinTest(TEST_NAME3, false, true, true, ExecType.SPARK);
	}
	
	@Test
	public void testSigmoidDenseLogMinusRewritesRepSP() 
	{
		runMLUnaryBuiltinTest(TEST_NAME4, false, true, true, ExecType.SPARK);
	}
	
	
	/**
	 * 
	 * @param testname
	 * @param sparse
	 * @param rewrites
	 * @param instType
	 */
	private void runMLUnaryBuiltinTest( String testname, boolean sparse, boolean rewrites, ExecType instType)
	{
		runMLUnaryBuiltinTest(testname, sparse, rewrites, false, instType);
	}
	
	/**
	 * 
	 * @param sparseM1
	 * @param sparseM2
	 * @param instType
	 */
	private void runMLUnaryBuiltinTest( String testname, boolean sparse, boolean rewrites, boolean rep, ExecType instType)
	{
		RUNTIME_PLATFORM platformOld = rtplatform;
		switch( instType ){
			case MR: rtplatform = RUNTIME_PLATFORM.HADOOP; break;
			case SPARK: rtplatform = RUNTIME_PLATFORM.SPARK; break;
			default: rtplatform = RUNTIME_PLATFORM.HYBRID; break;
		}
	
		boolean sparkConfigOld = DMLScript.USE_LOCAL_SPARK_CONFIG;
		if( rtplatform == RUNTIME_PLATFORM.SPARK )
			DMLScript.USE_LOCAL_SPARK_CONFIG = true;

		boolean rewritesOld = OptimizerUtils.ALLOW_ALGEBRAIC_SIMPLIFICATION;
		boolean forceOld = QuaternaryOp.FORCE_REPLICATION;
		
		OptimizerUtils.ALLOW_ALGEBRAIC_SIMPLIFICATION = rewrites;
		QuaternaryOp.FORCE_REPLICATION = rep;
	    
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
			
			runTest(true, false, null, -1); 
			runRScript(true); 
		
			//compare matrices 
			HashMap<CellIndex, Double> dmlfile = readDMLMatrixFromHDFS("R");
			HashMap<CellIndex, Double> rfile  = readRMatrixFromFS("R");
			TestUtils.compareMatrices(dmlfile, rfile, eps, "Stat-DML", "Stat-R");
			
			//check statistics for right operator in cp
			if( instType == ExecType.CP && rewrites )
				Assert.assertTrue(Statistics.getCPHeavyHitterOpCodes().contains(WeightedSigmoid.OPCODE_CP));
		}
		finally
		{
			rtplatform = platformOld;
			DMLScript.USE_LOCAL_SPARK_CONFIG = sparkConfigOld;
			OptimizerUtils.ALLOW_ALGEBRAIC_SIMPLIFICATION = rewritesOld;
			QuaternaryOp.FORCE_REPLICATION = forceOld;
		}
	}	
}