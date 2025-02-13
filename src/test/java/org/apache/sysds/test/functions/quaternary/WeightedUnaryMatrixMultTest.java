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

package org.apache.sysds.test.functions.quaternary;

import java.util.HashMap;

import org.apache.sysds.common.Opcodes;
import org.junit.AfterClass;
import org.junit.Assert;
import org.junit.BeforeClass;
import org.junit.Test;
import org.apache.sysds.api.DMLScript;
import org.apache.sysds.common.Types.ExecMode;
import org.apache.sysds.hops.OptimizerUtils;
import org.apache.sysds.hops.QuaternaryOp;
import org.apache.sysds.lops.WeightedUnaryMM;
import org.apache.sysds.lops.WeightedUnaryMMR;
import org.apache.sysds.common.Types.ExecType;
import org.apache.sysds.runtime.instructions.Instruction;
import org.apache.sysds.runtime.matrix.data.MatrixValue.CellIndex;
import org.apache.sysds.runtime.meta.MatrixCharacteristics;
import org.apache.sysds.test.AutomatedTestBase;
import org.apache.sysds.test.TestConfiguration;
import org.apache.sysds.test.TestUtils;
import org.apache.sysds.utils.Statistics;

/**
 * 
 * 
 */
public class WeightedUnaryMatrixMultTest extends AutomatedTestBase 
{
	private final static String TEST_NAME1 = "WeightedUnaryMMExpMult";
	private final static String TEST_NAME2 = "WeightedUnaryMMExpDiv";	
	private final static String TEST_NAME3 = "WeightedUnaryMMPow2";
	private final static String TEST_NAME4 = "WeightedUnaryMMMult2";
	private final static String TEST_DIR = "functions/quaternary/";
	private final static String TEST_CLASS_DIR = TEST_DIR + WeightedUnaryMatrixMultTest.class.getSimpleName() + "/";
	
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
		addTestConfiguration(TEST_NAME1,new TestConfiguration(TEST_CLASS_DIR, TEST_NAME1,new String[]{"R"}));
		addTestConfiguration(TEST_NAME2,new TestConfiguration(TEST_CLASS_DIR, TEST_NAME2,new String[]{"R"}));
		addTestConfiguration(TEST_NAME3,new TestConfiguration(TEST_CLASS_DIR, TEST_NAME3,new String[]{"R"}));
		addTestConfiguration(TEST_NAME4,new TestConfiguration(TEST_CLASS_DIR, TEST_NAME4,new String[]{"R"}));
		
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

	//cp testcases
	
	@Test
	public void testWeightedUnaryMMExpMultDenseCP()  {
		runWeightedUnaryMMTest(TEST_NAME1, false, true, false, ExecType.CP);
	}
	
	@Test
	public void testWeightedUnaryMMExpDivDenseCP()  {
		runWeightedUnaryMMTest(TEST_NAME2, false, true, false, ExecType.CP);
	}
	
	@Test
	public void testWeightedUnaryMMPow2DenseCP()  {
		runWeightedUnaryMMTest(TEST_NAME3, false, true, false, ExecType.CP);
	}
	
	@Test
	public void testWeightedUnaryMMMult2DenseCP()  {
		runWeightedUnaryMMTest(TEST_NAME4, false, true, false, ExecType.CP);
	}
	
	@Test
	public void testWeightedUnaryMMExpMultSparseCP()  {
		runWeightedUnaryMMTest(TEST_NAME1, true, true, false, ExecType.CP);
	}
	
	@Test
	public void testWeightedUnaryMMExpDivSparseCP()  {
		runWeightedUnaryMMTest(TEST_NAME2, true, true, false, ExecType.CP);
	}
	
	@Test
	public void testWeightedUnaryMMPow2SparseCP()  {
		runWeightedUnaryMMTest(TEST_NAME3, true, true, false, ExecType.CP);
	}
	
	@Test
	public void testWeightedUnaryMMMult2SparseCP()  {
		runWeightedUnaryMMTest(TEST_NAME4, true, true, false, ExecType.CP);
	}
	
	//sp testcases
	
	@Test
	public void testWeightedUnaryMMExpMultDenseSP()  {
		runWeightedUnaryMMTest(TEST_NAME1, false, true, false, ExecType.SPARK);
	}
	
	@Test
	public void testWeightedUnaryMMExpDivDenseSP()  {
		runWeightedUnaryMMTest(TEST_NAME2, false, true, false, ExecType.SPARK);
	}
	
	@Test
	public void testWeightedUnaryMMPow2DenseSP()  {
		runWeightedUnaryMMTest(TEST_NAME3, false, true, false, ExecType.SPARK);
	}
	
	@Test
	public void testWeightedUnaryMMMult2DenseSP()  {
		runWeightedUnaryMMTest(TEST_NAME4, false, true, false, ExecType.SPARK);
	}
	
	@Test
	public void testWeightedUnaryMMExpMultSparseSP()  {
		runWeightedUnaryMMTest(TEST_NAME1, true, true, false, ExecType.SPARK);
	}
	
	@Test
	public void testWeightedUnaryMMExpDivSparseSP()  {
		runWeightedUnaryMMTest(TEST_NAME2, true, true, false, ExecType.SPARK);
	}
	
	@Test
	public void testWeightedUnaryMMPow2SparseSP()  {
		runWeightedUnaryMMTest(TEST_NAME3, true, true, false, ExecType.SPARK);
	}
	
	@Test
	public void testWeightedUnaryMMMult2SparseSP()  {
		runWeightedUnaryMMTest(TEST_NAME4, true, true, false, ExecType.SPARK);
	}
	
	@Test
	public void testWeightedUnaryMMExpMultDenseRepSP()  {
		runWeightedUnaryMMTest(TEST_NAME1, false, true, true, ExecType.SPARK);
	}
	
	private void runWeightedUnaryMMTest( String testname, boolean sparse, boolean rewrites, boolean rep, ExecType instType)
	{
		ExecMode platformOld = rtplatform;
		switch( instType ){
			case SPARK: rtplatform = ExecMode.SPARK; break;
			default: rtplatform = ExecMode.HYBRID; break;
		}
	
		boolean sparkConfigOld = DMLScript.USE_LOCAL_SPARK_CONFIG;
		if( rtplatform == ExecMode.SPARK )
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
			
			String TEST_CACHE_DIR = "";
			if (TEST_CACHE_ENABLED)
			{
				TEST_CACHE_DIR = TEST_NAME + "_" + sparsity + "/";
			}
			
			loadTestConfiguration(config, TEST_CACHE_DIR);
			
			// This is for running the junit test the new way, i.e., construct the arguments directly
			String HOME = SCRIPT_DIR + TEST_DIR;
			fullDMLScriptName = HOME + TEST_NAME + ".dml";
			programArgs = new String[]{"-stats", "-explain", "runtime", "-args",
				input("W"), input("U"), input("V"), output("R") };
			
			fullRScriptName = HOME + TEST_NAME + ".R";
			rCmd = "Rscript" + " " + fullRScriptName + " " + inputDir() + " " + expectedDir();
	
			//generate actual dataset 
			double[][] W = getRandomMatrix(rows, cols, 0, 1, sparsity, 7); 
			writeInputMatrixWithMTD("W", W, true);
			double[][] U = getRandomMatrix(rows, rank, 0, 1, 1.0, 713); 
			writeInputMatrixWithMTD("U", U, true);
			double[][] V = getRandomMatrix(cols, rank, 0, 1, 1.0, 812); 
			writeInputMatrixWithMTD("V", V, true);
			
			runTest(true, false, null, -1); 
			runRScript(true); 
		
			//compare matrices 
			HashMap<CellIndex, Double> dmlfile = readDMLMatrixFromOutputDir("R");
			HashMap<CellIndex, Double> rfile  = readRMatrixFromExpectedDir("R");
			TestUtils.compareMatrices(dmlfile, rfile, eps, "Stat-DML", "Stat-R");
			checkDMLMetaDataFile("R", new MatrixCharacteristics(rows, cols, 1, 1));

			//check statistics for right operator in cp and spark
			if( instType == ExecType.CP && rewrites ) {
				Assert.assertTrue("Missing opcode wumm", Statistics.getCPHeavyHitterOpCodes().contains(WeightedUnaryMM.OPCODE_CP));
			}
			else if( instType == ExecType.SPARK && rewrites ) {
				String opcode = Instruction.SP_INST_PREFIX + ((rep)? Opcodes.WEIGHTEDUNARYMMR.toString():Opcodes.WEIGHTEDUNARYMM.toString());
				Assert.assertTrue("Missing opcode sp_wumm", Statistics.getCPHeavyHitterOpCodes().contains(opcode) );
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