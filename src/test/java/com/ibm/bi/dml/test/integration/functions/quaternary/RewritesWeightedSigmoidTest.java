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
import com.ibm.bi.dml.lops.LopProperties.ExecType;
import com.ibm.bi.dml.lops.WeightedSigmoid;
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
public class RewritesWeightedSigmoidTest extends AutomatedTestBase 
{
	
	private final static String TEST_NAME = "RewriteWeightedSigmoid";
	private final static String TEST_DIR = "functions/quaternary/";
	
	private final static double eps = 1e-10;
	
	private final static int rows1 = 1;
	private final static int rows2 = 5;
	private final static int rank = 10;
	private final static double spSparse = 0.001;
	private final static double spDense = 0.6;
	
	@Override
	public void setUp() 
	{
		TestUtils.clearAssertionInformation();
		addTestConfiguration(TEST_NAME,new TestConfiguration(TEST_DIR, TEST_NAME,new String[]{"C"}));
	}

	@Test
	public void testWSigmoidDenseRewritesCP() {
		runRewriteWeightedSigmoidTest(false, true, ExecType.CP);
	}
	
	@Test
	public void testWSigmoidSparseRewritesCP() {
		runRewriteWeightedSigmoidTest(true, true, ExecType.CP);
	}
	
	@Test
	public void testWSigmoidDenseRewritesMR() {
		runRewriteWeightedSigmoidTest(false, true, ExecType.MR);
	}
	
	@Test
	public void testWSigmoidSparseRewritesMR() {
		runRewriteWeightedSigmoidTest(true, true, ExecType.MR);
	}
	
	@Test
	public void testWSigmoidDenseRewritesSP() {
		runRewriteWeightedSigmoidTest(false, true, ExecType.SPARK);
	}
	
	@Test
	public void testWSigmoidSparseRewritesSP() {
		runRewriteWeightedSigmoidTest(true, true, ExecType.SPARK);
	}
	
	
	/**
	 * 
	 * @param sparseM1
	 * @param sparseM2
	 * @param instType
	 */
	private void runRewriteWeightedSigmoidTest( boolean sparse, boolean rewrites, ExecType instType)
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
		OptimizerUtils.ALLOW_ALGEBRAIC_SIMPLIFICATION = rewrites;
	    
		try
		{
			double sparsity = (sparse) ? spSparse : spDense;
			
			TestConfiguration config = getTestConfiguration(TEST_NAME);
			
			// This is for running the junit test the new way, i.e., construct the arguments directly
			String HOME = SCRIPT_DIR + TEST_DIR;
			fullDMLScriptName = HOME + TEST_NAME + ".dml";
			programArgs = new String[]{"-stats", "-explain", "runtime",
					                   "-args", 
					                        HOME + INPUT_DIR + "A",
					                        HOME + INPUT_DIR + "B",
					                        HOME + OUTPUT_DIR + "C"    };
			fullRScriptName = HOME + TEST_NAME + ".R";
			rCmd = "Rscript" + " " + fullRScriptName + " " + 
			       HOME + INPUT_DIR + " " + HOME + EXPECTED_DIR;
			
			loadTestConfiguration(config);
	
			//generate actual dataset 
			double[][] A = getRandomMatrix(rows1, rank, 0, 1, 1.0, 213); 
			writeInputMatrixWithMTD("A", A, true);
			double[][] B = getRandomMatrix(rows2, rank, 0, 1, sparsity, 312); 
			writeInputMatrixWithMTD("B", B, true);
			
			runTest(true, false, null, -1); 
			runRScript(true); 
			
			//compare matrices 
			HashMap<CellIndex, Double> dmlfile = readDMLMatrixFromHDFS("C");
			HashMap<CellIndex, Double> rfile  = readRMatrixFromFS("C");
			TestUtils.compareMatrices(dmlfile, rfile, eps, "Stat-DML", "Stat-R");
			checkDMLMetaDataFile("C", new MatrixCharacteristics(rows2,rank,1,1));

			//check statistics for right operator in cp
			if( instType == ExecType.CP )
				Assert.assertTrue(!Statistics.getCPHeavyHitterOpCodes().contains(WeightedSigmoid.OPCODE_CP));
			else if( instType == ExecType.SPARK )
				Assert.assertTrue(!Statistics.getCPHeavyHitterOpCodes().contains(Instruction.SP_INST_PREFIX+WeightedSigmoid.OPCODE));	
		}
		finally
		{
			rtplatform = platformOld;
			DMLScript.USE_LOCAL_SPARK_CONFIG = sparkConfigOld;
			OptimizerUtils.ALLOW_ALGEBRAIC_SIMPLIFICATION = rewritesOld;
		}
	}	
}