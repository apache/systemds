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
import com.ibm.bi.dml.lops.WeightedDivMM;
import com.ibm.bi.dml.lops.WeightedDivMMR;
import com.ibm.bi.dml.runtime.instructions.Instruction;
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
public class WeightedDivMatrixMultTest extends AutomatedTestBase 
{
	private final static String TEST_NAME1 = "WeightedDivMMLeft";
	private final static String TEST_NAME2 = "WeightedDivMMRight";

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
		TestUtils.clearAssertionInformation();
		addTestConfiguration(TEST_NAME1,new TestConfiguration(TEST_DIR, TEST_NAME1,new String[]{"R"}));
		addTestConfiguration(TEST_NAME2,new TestConfiguration(TEST_DIR, TEST_NAME2,new String[]{"R"}));
	}

	
	@Test
	public void testWeightedDivMMLeftDenseCP() 
	{
		runWeightedDivMMTest(TEST_NAME1, false, true, false, ExecType.CP);
	}
	
	@Test
	public void testWeightedDivMMLeftSparseCP() 
	{
		runWeightedDivMMTest(TEST_NAME1, true, true, false, ExecType.CP);
	}
	
	@Test
	public void testWeightedDivMMRightDenseCP() 
	{
		runWeightedDivMMTest(TEST_NAME2, false, true, false, ExecType.CP);
	}
	
	@Test
	public void testWeightedDivMMRightSparseCP() 
	{
		runWeightedDivMMTest(TEST_NAME2, true, true, false, ExecType.CP);
	}
	
	@Test
	public void testWeightedDivMMLeftDenseMR() 
	{
		runWeightedDivMMTest(TEST_NAME1, false, true, false, ExecType.MR);
	}
	
	@Test
	public void testWeightedDivMMLeftSparseMR() 
	{
		runWeightedDivMMTest(TEST_NAME1, true, true, false, ExecType.MR);
	}
	
	@Test
	public void testWeightedDivMMLeftDenseMRRep() 
	{
		runWeightedDivMMTest(TEST_NAME1, false, true, true, ExecType.MR);
	}
	
	@Test
	public void testWeightedDivMMRightDenseMR() 
	{
		runWeightedDivMMTest(TEST_NAME2, false, true, false, ExecType.MR);
	}
	
	@Test
	public void testWeightedDivMMRightSparseMR() 
	{
		runWeightedDivMMTest(TEST_NAME2, true, true, false, ExecType.MR);
	}
	
	@Test
	public void testWeightedDivMMRightDenseMRRep() 
	{
		runWeightedDivMMTest(TEST_NAME2, false, true, true, ExecType.MR);
	}
	
	@Test
	public void testWeightedDivMMLeftDenseSP() 
	{
		runWeightedDivMMTest(TEST_NAME1, false, true, false, ExecType.SPARK);
	}
	
	@Test
	public void testWeightedDivMMLeftSparseSP() 
	{
		runWeightedDivMMTest(TEST_NAME1, true, true, false, ExecType.SPARK);
	}
	
	@Test
	public void testWeightedDivMMLeftDenseSPRep() 
	{
		runWeightedDivMMTest(TEST_NAME1, false, true, true, ExecType.SPARK);
	}
	
	@Test
	public void testWeightedDivMMRightDenseSP() 
	{
		runWeightedDivMMTest(TEST_NAME2, false, true, false, ExecType.SPARK);
	}
	
	@Test
	public void testWeightedDivMMRightSparseSP() 
	{
		runWeightedDivMMTest(TEST_NAME2, true, true, false, ExecType.SPARK);
	}
	
	@Test
	public void testWeightedDivMMRightDenseSPRep() 
	{
		runWeightedDivMMTest(TEST_NAME2, false, true, true, ExecType.SPARK);
	}
	
	/**
	 * 
	 * @param sparseM1
	 * @param sparseM2
	 * @param instType
	 */
	private void runWeightedDivMMTest( String testname, boolean sparse, boolean rewrites, boolean rep, ExecType instType)
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
			boolean left = testname.equals(TEST_NAME1);
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
					                        HOME + OUTPUT_DIR + "R"    };
			fullRScriptName = HOME + TEST_NAME + ".R";
			rCmd = "Rscript" + " " + fullRScriptName + " " + 
			       HOME + INPUT_DIR + " " + HOME + EXPECTED_DIR;
			
			loadTestConfiguration(config);
	
			//generate actual dataset 
			double[][] X = getRandomMatrix(rows, cols, 0, 1, sparsity, 7); 
			writeInputMatrixWithMTD("X", X, true);
			double[][] U = getRandomMatrix(rows, rank, 0, 1, 1.0, 713); 
			writeInputMatrixWithMTD("U", U, true);
			double[][] V = getRandomMatrix(cols, rank, 0, 1, 1.0, 812); 
			writeInputMatrixWithMTD("V", V, true);
			
			runTest(true, false, null, -1); 
			runRScript(true); 
		
			//compare matrices 
			HashMap<CellIndex, Double> dmlfile = readDMLMatrixFromHDFS("R");
			HashMap<CellIndex, Double> rfile  = readRMatrixFromFS("R");
			TestUtils.compareMatrices(dmlfile, rfile, eps, "Stat-DML", "Stat-R");
			checkDMLMetaDataFile("R", new MatrixCharacteristics(left?cols:rows, rank, 1, 1));

			//check statistics for right operator in cp and spark
			if( instType == ExecType.CP && rewrites ) {
				Assert.assertTrue("Missing opcode wdivmm", Statistics.getCPHeavyHitterOpCodes().contains(WeightedDivMM.OPCODE_CP));
			}
			else if( instType == ExecType.SPARK && rewrites ) {
				Assert.assertTrue("Missing opcode sp_wdivmm",
						!rep && Statistics.getCPHeavyHitterOpCodes().contains(Instruction.SP_INST_PREFIX+WeightedDivMM.OPCODE)
					  || rep && Statistics.getCPHeavyHitterOpCodes().contains(Instruction.SP_INST_PREFIX+WeightedDivMMR.OPCODE) );
			}
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