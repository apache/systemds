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

package org.apache.sysml.test.integration.functions.unary.matrix;

import java.util.HashMap;

import org.junit.Assert;
import org.junit.Test;

import org.apache.sysml.api.DMLScript;
import org.apache.sysml.api.DMLScript.RUNTIME_PLATFORM;
import org.apache.sysml.hops.OptimizerUtils;
import org.apache.sysml.lops.LopProperties.ExecType;
import org.apache.sysml.runtime.instructions.Instruction;
import org.apache.sysml.runtime.matrix.data.MatrixValue.CellIndex;
import org.apache.sysml.test.integration.AutomatedTestBase;
import org.apache.sysml.test.integration.TestConfiguration;
import org.apache.sysml.test.utils.TestUtils;
import org.apache.sysml.utils.Statistics;

/**
 * 
 * 
 */
public class FullSelectPosTest extends AutomatedTestBase 
{
	private final static String TEST_NAME = "SelPos";
	private final static String TEST_DIR = "functions/unary/matrix/";
	private static final String TEST_CLASS_DIR = TEST_DIR + FullSelectPosTest.class.getSimpleName() + "/";
	
	private final static double eps = 1e-10;
	
	private final static int rows = 1108;
	private final static int cols = 1001;
	private final static double spSparse = 0.05;
	private final static double spDense = 0.7;
	
	
	@Override
	public void setUp() 
	{
		addTestConfiguration(TEST_NAME,new TestConfiguration(TEST_CLASS_DIR, TEST_NAME,new String[]{"B"})); 
	}

	@Test
	public void testSelPosDenseCP() {
		runSelPosTest(false, ExecType.CP, false);
	}
	
	@Test
	public void testSelPosSparseCP() {
		runSelPosTest(true, ExecType.CP, false);
	}
	
	@Test
	public void testSelPosDenseMR() {
		runSelPosTest(false, ExecType.MR, false);
	}
	
	@Test
	public void testSelPosSparseMR() {
		runSelPosTest(true, ExecType.MR, false);
	}
	
	@Test
	public void testSelPosDenseSP() {
		runSelPosTest(false, ExecType.SPARK, false);
	}
	
	@Test
	public void testSelPosSparseSP() {
		runSelPosTest(true, ExecType.SPARK, false);
	}

	@Test
	public void testSelPosDenseRewriteCP() {
		runSelPosTest(false, ExecType.CP, true);
	}
	
	@Test
	public void testSelPosSparseRewriteCP() {
		runSelPosTest(true, ExecType.CP, true);
	}
	
	@Test
	public void testSelPosDenseRewriteMR() {
		runSelPosTest(false, ExecType.MR, true);
	}
	
	@Test
	public void testSelPosSparseRewriteMR() {
		runSelPosTest(true, ExecType.MR, true);
	}
	
	@Test
	public void testSelPosDenseRewriteSP() {
		runSelPosTest(false, ExecType.SPARK, true);
	}
	
	@Test
	public void testSelPosSparseRewriteSP() {
		runSelPosTest(true, ExecType.SPARK, true);
	}
	
	
	/**
	 * 
	 * @param sparseM1
	 * @param sparseM2
	 * @param instType
	 */
	private void runSelPosTest( boolean sparse, ExecType instType, boolean rewrites)
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

		//rewrites
		boolean oldFlagRewrites = OptimizerUtils.ALLOW_ALGEBRAIC_SIMPLIFICATION;
		OptimizerUtils.ALLOW_ALGEBRAIC_SIMPLIFICATION = rewrites;
		
		try
		{
			double sparsity = (sparse) ? spSparse : spDense;
			
			getAndLoadTestConfiguration(TEST_NAME);
			
			// This is for running the junit test the new way, i.e., construct the arguments directly
			String HOME = SCRIPT_DIR + TEST_DIR;
			fullDMLScriptName = HOME + TEST_NAME + ".dml";
			
			//stats parameter required for opcode check
			programArgs = new String[]{"-stats", "-args", input("A"), output("B") };
			
			fullRScriptName = HOME + TEST_NAME + ".R";
			rCmd = "Rscript" + " " + fullRScriptName + " " + inputDir() + " " + expectedDir();
	
			//generate actual dataset 
			double[][] A = getRandomMatrix(rows, cols, -1, 1, sparsity, 7); 
			writeInputMatrixWithMTD("A", A, true);
	
			runTest(true, false, null, -1); 
			runRScript(true); 
		
			//compare matrices 
			HashMap<CellIndex, Double> dmlfile = readDMLMatrixFromHDFS("B");
			HashMap<CellIndex, Double> rfile  = readRMatrixFromFS("B");
			TestUtils.compareMatrices(dmlfile, rfile, eps, "Stat-DML", "Stat-R");
			
			//check generated opcode
			if( rewrites ){
				if( instType == ExecType.CP )
					Assert.assertTrue("Missing opcode: sel+", Statistics.getCPHeavyHitterOpCodes().contains("sel+"));
				else if ( instType == ExecType.SPARK )
					Assert.assertTrue("Missing opcode: "+Instruction.SP_INST_PREFIX+"sel+", Statistics.getCPHeavyHitterOpCodes().contains(Instruction.SP_INST_PREFIX+"sel+"));	
			}
		}
		finally
		{
			rtplatform = platformOld;
			DMLScript.USE_LOCAL_SPARK_CONFIG = sparkConfigOld;
			OptimizerUtils.ALLOW_ALGEBRAIC_SIMPLIFICATION = oldFlagRewrites;
		}
	}	
}