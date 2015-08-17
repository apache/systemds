/**
 * (C) Copyright IBM Corp. 2010, 2015
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * 
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
import com.ibm.bi.dml.lops.WeightedSquaredLoss;
import com.ibm.bi.dml.runtime.matrix.MatrixCharacteristics;
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
	
	@Test
	public void testSquaredLossDensePostWeightsRewritesSP() 
	{
		runMLUnaryBuiltinTest(TEST_NAME1, false, true, ExecType.SPARK);
	}
	
	@Test
	public void testSquaredLossDensePreWeightsRewritesSP() 
	{
		runMLUnaryBuiltinTest(TEST_NAME2, false, true, ExecType.SPARK);
	}
	
	@Test
	public void testSquaredLossDenseNoWeightsRewritesSP() 
	{
		runMLUnaryBuiltinTest(TEST_NAME3, false, true, ExecType.SPARK);
	}
	
	@Test
	public void testSquaredLossSparsePostWeightsRewritesSP() 
	{
		runMLUnaryBuiltinTest(TEST_NAME1, true, true, ExecType.SPARK);
	}
	
	@Test
	public void testSquaredLossSparsePreWeightsRewritesSP() 
	{
		runMLUnaryBuiltinTest(TEST_NAME2, true, true, ExecType.SPARK);
	}
	
	@Test
	public void testSquaredLossSparseNoWeightsRewritesSP() 
	{
		runMLUnaryBuiltinTest(TEST_NAME3, true, true, ExecType.SPARK);
	}
	
	//the following tests force the replication based mr operator because
	//otherwise we would always choose broadcasts for this small input data
	
	@Test
	public void testSquaredLossSparsePostWeightsRewritesRepMR() 
	{
		runMLUnaryBuiltinTest(TEST_NAME1, true, true, true, ExecType.MR);
	}
	
	@Test
	public void testSquaredLossSparsePreWeightsRewritesRepMR() 
	{
		runMLUnaryBuiltinTest(TEST_NAME2, true, true, true, ExecType.MR);
	}
	
	@Test
	public void testSquaredLossSparseNoWeightsRewritesRepMR() 
	{
		runMLUnaryBuiltinTest(TEST_NAME3, true, true, true, ExecType.MR);
	}
	
	@Test
	public void testSquaredLossDensePostWeightsRewritesRepMR() 
	{
		runMLUnaryBuiltinTest(TEST_NAME1, false, true, true, ExecType.MR);
	}
	
	@Test
	public void testSquaredLossDensePreWeightsRewritesRepMR() 
	{
		runMLUnaryBuiltinTest(TEST_NAME2, false, true, true, ExecType.MR);
	}
	
	@Test
	public void testSquaredLossDenseNoWeightsRewritesRepMR() 
	{
		runMLUnaryBuiltinTest(TEST_NAME3, false, true, true, ExecType.MR);
	}
	
	@Test
	public void testSquaredLossSparsePostWeightsRewritesRepSP() 
	{
		runMLUnaryBuiltinTest(TEST_NAME1, true, true, true, ExecType.SPARK);
	}
	
	@Test
	public void testSquaredLossSparsePreWeightsRewritesRepSP() 
	{
		runMLUnaryBuiltinTest(TEST_NAME2, true, true, true, ExecType.SPARK);
	}
	
	@Test
	public void testSquaredLossSparseNoWeightsRewritesRepSP() 
	{
		runMLUnaryBuiltinTest(TEST_NAME3, true, true, true, ExecType.SPARK);
	}
	
	@Test
	public void testSquaredLossDensePostWeightsRewritesRepSP() 
	{
		runMLUnaryBuiltinTest(TEST_NAME1, false, true, true, ExecType.SPARK);
	}
	
	@Test
	public void testSquaredLossDensePreWeightsRewritesRepSP() 
	{
		runMLUnaryBuiltinTest(TEST_NAME2, false, true, true, ExecType.SPARK);
	}
	
	@Test
	public void testSquaredLossDenseNoWeightsRewritesRepSP() 
	{
		runMLUnaryBuiltinTest(TEST_NAME3, false, true, true, ExecType.SPARK);
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
			if( !TEST_NAME.equals(TEST_NAME3) ) {
				double[][] W = getRandomMatrix(rows, cols, 0, 1, sparsity, 1467); 
				writeInputMatrixWithMTD("W", W, true);
			}
			
			runTest(true, false, null, -1); 
			runRScript(true); 
		
			//compare matrices 
			HashMap<CellIndex, Double> dmlfile = readDMLMatrixFromHDFS("R");
			HashMap<CellIndex, Double> rfile  = readRMatrixFromFS("R");
			TestUtils.compareMatrices(dmlfile, rfile, eps, "Stat-DML", "Stat-R");
			checkDMLMetaDataFile("R", new MatrixCharacteristics(1,1,1,1));

			//check statistics for right operator in cp
			if( instType == ExecType.CP && rewrites )
				Assert.assertTrue(Statistics.getCPHeavyHitterOpCodes().contains(WeightedSquaredLoss.OPCODE_CP));
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