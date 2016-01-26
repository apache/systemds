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

package org.apache.sysml.test.integration.functions.quaternary;

import java.util.HashMap;

import org.junit.AfterClass;
import org.junit.Assert;
import org.junit.BeforeClass;
import org.junit.Test;

import org.apache.sysml.api.DMLScript;
import org.apache.sysml.api.DMLScript.RUNTIME_PLATFORM;
import org.apache.sysml.hops.OptimizerUtils;
import org.apache.sysml.hops.QuaternaryOp;
import org.apache.sysml.lops.LopProperties.ExecType;
import org.apache.sysml.lops.WeightedDivMM;
import org.apache.sysml.lops.WeightedDivMMR;
import org.apache.sysml.runtime.instructions.Instruction;
import org.apache.sysml.runtime.matrix.MatrixCharacteristics;
import org.apache.sysml.runtime.matrix.data.MatrixValue.CellIndex;
import org.apache.sysml.test.integration.AutomatedTestBase;
import org.apache.sysml.test.integration.TestConfiguration;
import org.apache.sysml.test.utils.TestUtils;
import org.apache.sysml.utils.Statistics;

/**
 * 
 * 
 */
public class WeightedDivMatrixMultTest extends AutomatedTestBase 
{
	private final static String TEST_NAME1 = "WeightedDivMMLeft";
	private final static String TEST_NAME2 = "WeightedDivMMRight";
	private final static String TEST_NAME3 = "WeightedDivMMMultBasic";
	private final static String TEST_NAME4 = "WeightedDivMMMultLeft";
	private final static String TEST_NAME5 = "WeightedDivMMMultRight";
	private final static String TEST_NAME6 = "WeightedDivMMMultMinusLeft";
	private final static String TEST_NAME7 = "WeightedDivMMMultMinusRight";
	private final static String TEST_NAME8 = "WeightedDivMM4MultMinusLeft";
	private final static String TEST_NAME9 = "WeightedDivMM4MultMinusRight";
	private final static String TEST_DIR = "functions/quaternary/";
	private final static String TEST_CLASS_DIR = TEST_DIR + WeightedDivMatrixMultTest.class.getSimpleName() + "/";
	
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
		addTestConfiguration(TEST_NAME5,new TestConfiguration(TEST_CLASS_DIR, TEST_NAME5,new String[]{"R"}));
		addTestConfiguration(TEST_NAME6,new TestConfiguration(TEST_CLASS_DIR, TEST_NAME6,new String[]{"R"}));
		addTestConfiguration(TEST_NAME7,new TestConfiguration(TEST_CLASS_DIR, TEST_NAME7,new String[]{"R"}));
		addTestConfiguration(TEST_NAME8,new TestConfiguration(TEST_CLASS_DIR, TEST_NAME8,new String[]{"R"}));
		addTestConfiguration(TEST_NAME9,new TestConfiguration(TEST_CLASS_DIR, TEST_NAME9,new String[]{"R"}));
	
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

	//a) testcases for wdivmm w/ DIVIDE LEFT/RIGHT
	
	@Test
	public void testWeightedDivMMLeftDenseCP() {
		runWeightedDivMMTest(TEST_NAME1, false, true, false, ExecType.CP);
	}
	
	@Test
	public void testWeightedDivMMLeftSparseCP() {
		runWeightedDivMMTest(TEST_NAME1, true, true, false, ExecType.CP);
	}
	
	@Test
	public void testWeightedDivMMRightDenseCP() {
		runWeightedDivMMTest(TEST_NAME2, false, true, false, ExecType.CP);
	}
	
	@Test
	public void testWeightedDivMMRightSparseCP() {
		runWeightedDivMMTest(TEST_NAME2, true, true, false, ExecType.CP);
	}
	
	@Test
	public void testWeightedDivMMLeftDenseMR() {
		runWeightedDivMMTest(TEST_NAME1, false, true, false, ExecType.MR);
	}
	
	@Test
	public void testWeightedDivMMLeftSparseMR() {
		runWeightedDivMMTest(TEST_NAME1, true, true, false, ExecType.MR);
	}
	
	@Test
	public void testWeightedDivMMLeftDenseMRRep() {
		runWeightedDivMMTest(TEST_NAME1, false, true, true, ExecType.MR);
	}
	
	@Test
	public void testWeightedDivMMRightDenseMR() {
		runWeightedDivMMTest(TEST_NAME2, false, true, false, ExecType.MR);
	}
	
	@Test
	public void testWeightedDivMMRightSparseMR() {
		runWeightedDivMMTest(TEST_NAME2, true, true, false, ExecType.MR);
	}
	
	@Test
	public void testWeightedDivMMRightDenseMRRep() {
		runWeightedDivMMTest(TEST_NAME2, false, true, true, ExecType.MR);
	}
	
	@Test
	public void testWeightedDivMMLeftDenseSP() {
		runWeightedDivMMTest(TEST_NAME1, false, true, false, ExecType.SPARK);
	}
	
	@Test
	public void testWeightedDivMMLeftSparseSP() {
		runWeightedDivMMTest(TEST_NAME1, true, true, false, ExecType.SPARK);
	}
	
	@Test
	public void testWeightedDivMMLeftDenseSPRep() {
		runWeightedDivMMTest(TEST_NAME1, false, true, true, ExecType.SPARK);
	}
	
	@Test
	public void testWeightedDivMMRightDenseSP() {
		runWeightedDivMMTest(TEST_NAME2, false, true, false, ExecType.SPARK);
	}
	
	@Test
	public void testWeightedDivMMRightSparseSP() {
		runWeightedDivMMTest(TEST_NAME2, true, true, false, ExecType.SPARK);
	}
	
	@Test
	public void testWeightedDivMMRightDenseSPRep() {
		runWeightedDivMMTest(TEST_NAME2, false, true, true, ExecType.SPARK);
	}

	//b) testcases for wdivmm w/ MULTIPLY BASIC/LEFT/RIGHT
	
	@Test
	public void testWeightedDivMMMultBasicDenseCP() {
		runWeightedDivMMTest(TEST_NAME3, false, true, false, ExecType.CP);
	}
	
	@Test
	public void testWeightedDivMMMultBasicSparseCP() {
		runWeightedDivMMTest(TEST_NAME3, true, true, false, ExecType.CP);
	}
	
	@Test
	public void testWeightedDivMMMultLeftDenseCP() {
		runWeightedDivMMTest(TEST_NAME4, false, true, false, ExecType.CP);
	}
	
	@Test
	public void testWeightedDivMMMultLeftSparseCP() {
		runWeightedDivMMTest(TEST_NAME4, true, true, false, ExecType.CP);
	}
	
	@Test
	public void testWeightedDivMMMultRightDenseCP() {
		runWeightedDivMMTest(TEST_NAME5, false, true, false, ExecType.CP);
	}
	
	@Test
	public void testWeightedDivMMMultRightSparseCP() {
		runWeightedDivMMTest(TEST_NAME5, true, true, false, ExecType.CP);
	}
	
	@Test
	public void testWeightedDivMMMultMinusLeftDenseCP() {
		runWeightedDivMMTest(TEST_NAME6, false, true, false, ExecType.CP);
	}
	
	@Test
	public void testWeightedDivMMMultMinusLeftSparseCP() {
		runWeightedDivMMTest(TEST_NAME6, true, true, false, ExecType.CP);
	}
	
	@Test
	public void testWeightedDivMMMultMinusRightDenseCP() {
		runWeightedDivMMTest(TEST_NAME7, false, true, false, ExecType.CP);
	}
	
	@Test
	public void testWeightedDivMMMultMinusRightSparseCP() {
		runWeightedDivMMTest(TEST_NAME7, true, true, false, ExecType.CP);
	}
	
	@Test
	public void testWeightedDivMM4MultMinusLeftDenseCP() {
		runWeightedDivMMTest(TEST_NAME8, false, true, false, ExecType.CP);
	}
	
	@Test
	public void testWeightedDivMM4MultMinusLeftSparseCP() {
		runWeightedDivMMTest(TEST_NAME8, true, true, false, ExecType.CP);
	}
	
	@Test
	public void testWeightedDivMM4MultMinusRightDenseCP() {
		runWeightedDivMMTest(TEST_NAME9, false, true, false, ExecType.CP);
	}
	
	@Test
	public void testWeightedDivMM4MultMinusRightSparseCP() {
		runWeightedDivMMTest(TEST_NAME9, true, true, false, ExecType.CP);
	}
	
	@Test
	public void testWeightedDivMMMultBasicDenseMR() {
		runWeightedDivMMTest(TEST_NAME3, false, true, false, ExecType.MR);
	}
	
	@Test
	public void testWeightedDivMMMultBasicSparseMR() {
		runWeightedDivMMTest(TEST_NAME3, true, true, false, ExecType.MR);
	}
	
	@Test
	public void testWeightedDivMMMultBasicDenseMRRep() {
		runWeightedDivMMTest(TEST_NAME3, false, true, true, ExecType.MR);
	}
	
	@Test
	public void testWeightedDivMMMultLeftDenseMR() {
		runWeightedDivMMTest(TEST_NAME4, false, true, false, ExecType.MR);
	}
	
	@Test
	public void testWeightedDivMMMultLeftSparseMR() {
		runWeightedDivMMTest(TEST_NAME4, true, true, false, ExecType.MR);
	}
	
	@Test
	public void testWeightedDivMMMultLeftDenseMRRep() {
		runWeightedDivMMTest(TEST_NAME4, false, true, true, ExecType.MR);
	}
	
	@Test
	public void testWeightedDivMMMultRightDenseMR() {
		runWeightedDivMMTest(TEST_NAME5, false, true, false, ExecType.MR);
	}
	
	@Test
	public void testWeightedDivMMMultRightSparseMR() {
		runWeightedDivMMTest(TEST_NAME5, true, true, false, ExecType.MR);
	}
	
	@Test
	public void testWeightedDivMMMultRightDenseMRRep() {
		runWeightedDivMMTest(TEST_NAME5, false, true, true, ExecType.MR);
	}
	
	@Test
	public void testWeightedDivMMMultMinusLeftDenseMR() {
		runWeightedDivMMTest(TEST_NAME6, false, true, false, ExecType.MR);
	}
	
	@Test
	public void testWeightedDivMMMultMinusLeftSparseMR() {
		runWeightedDivMMTest(TEST_NAME6, true, true, false, ExecType.MR);
	}
	
	@Test
	public void testWeightedDivMMMultMinusLeftDenseMRRep() {
		runWeightedDivMMTest(TEST_NAME6, false, true, true, ExecType.MR);
	}
	
	@Test
	public void testWeightedDivMMMultMinusRightDenseMR() {
		runWeightedDivMMTest(TEST_NAME7, false, true, false, ExecType.MR);
	}
	
	@Test
	public void testWeightedDivMMMultMinusRightSparseMR() {
		runWeightedDivMMTest(TEST_NAME7, true, true, false, ExecType.MR);
	}
	
	@Test
	public void testWeightedDivMMMultMinusRightDenseMRRep() {
		runWeightedDivMMTest(TEST_NAME7, false, true, true, ExecType.MR);
	}
	
	@Test
	public void testWeightedDivMM4MultMinusLeftDenseMR() {
		runWeightedDivMMTest(TEST_NAME8, false, true, false, ExecType.MR);
	}
	
	@Test
	public void testWeightedDivMM4MultMinusLeftSparseMR() {
		runWeightedDivMMTest(TEST_NAME8, true, true, false, ExecType.MR);
	}
	
	@Test
	public void testWeightedDivMM4MultMinusLeftDenseMRRep() {
		runWeightedDivMMTest(TEST_NAME8, false, true, true, ExecType.MR);
	}
	
	@Test
	public void testWeightedDivMM4MultMinusRightDenseMR() {
		runWeightedDivMMTest(TEST_NAME9, false, true, false, ExecType.MR);
	}
	
	@Test
	public void testWeightedDivMM4MultMinusRightSparseMR() {
		runWeightedDivMMTest(TEST_NAME9, true, true, false, ExecType.MR);
	}
	
	@Test
	public void testWeightedDivMM4MultMinusRightDenseMRRep() {
		runWeightedDivMMTest(TEST_NAME9, false, true, true, ExecType.MR);
	}
	
	@Test
	public void testWeightedDivMMMultBasicDenseSP() {
		runWeightedDivMMTest(TEST_NAME3, false, true, false, ExecType.SPARK);
	}
	
	@Test
	public void testWeightedDivMMMultBasicSparseSP() {
		runWeightedDivMMTest(TEST_NAME3, true, true, false, ExecType.SPARK);
	}
	
	@Test
	public void testWeightedDivMMMultBasicDenseSPRep() {
		runWeightedDivMMTest(TEST_NAME3, false, true, true, ExecType.SPARK);
	}
	
	@Test
	public void testWeightedDivMMMultLeftDenseSP() {
		runWeightedDivMMTest(TEST_NAME4, false, true, false, ExecType.SPARK);
	}
	
	@Test
	public void testWeightedDivMMMultLeftSparseSP() {
		runWeightedDivMMTest(TEST_NAME4, true, true, false, ExecType.SPARK);
	}
	
	@Test
	public void testWeightedDivMMMultLeftDenseSPRep() {
		runWeightedDivMMTest(TEST_NAME4, false, true, true, ExecType.SPARK);
	}
	
	@Test
	public void testWeightedDivMMMultRightDenseSP() {
		runWeightedDivMMTest(TEST_NAME5, false, true, false, ExecType.SPARK);
	}
	
	@Test
	public void testWeightedDivMMMultRightSparseSP() {
		runWeightedDivMMTest(TEST_NAME5, true, true, false, ExecType.SPARK);
	}
	
	@Test
	public void testWeightedDivMMMultRightDenseSPRep() {
		runWeightedDivMMTest(TEST_NAME5, false, true, true, ExecType.SPARK);
	}
	
	@Test
	public void testWeightedDivMMMultMinusLeftDenseSP() {
		runWeightedDivMMTest(TEST_NAME6, false, true, false, ExecType.SPARK);
	}
	
	@Test
	public void testWeightedDivMMMultMinusLeftSparseSP() {
		runWeightedDivMMTest(TEST_NAME6, true, true, false, ExecType.SPARK);
	}
	
	@Test
	public void testWeightedDivMMMultMinusLeftDenseSPRep() {
		runWeightedDivMMTest(TEST_NAME6, false, true, true, ExecType.SPARK);
	}
	
	@Test
	public void testWeightedDivMMMultMinusRightDenseSP() {
		runWeightedDivMMTest(TEST_NAME7, false, true, false, ExecType.SPARK);
	}
	
	@Test
	public void testWeightedDivMMMultMinusRightSparseSP() {
		runWeightedDivMMTest(TEST_NAME7, true, true, false, ExecType.SPARK);
	}
	
	@Test
	public void testWeightedDivMMMultMinusRightDenseSPRep() 
	{
		runWeightedDivMMTest(TEST_NAME7, false, true, true, ExecType.SPARK);
	}

	@Test
	public void testWeightedDivMM4MultMinusLeftDenseSP() {
		runWeightedDivMMTest(TEST_NAME8, false, true, false, ExecType.SPARK);
	}
	
	@Test
	public void testWeightedDivMM4MultMinusLeftSparseSP() {
		runWeightedDivMMTest(TEST_NAME8, true, true, false, ExecType.SPARK);
	}
	
	@Test
	public void testWeightedDivMM4MultMinusLeftDenseSPRep() {
		runWeightedDivMMTest(TEST_NAME8, false, true, true, ExecType.SPARK);
	}
	
	@Test
	public void testWeightedDivMM4MultMinusRightDenseSP()  {
		runWeightedDivMMTest(TEST_NAME9, false, true, false, ExecType.SPARK);
	}
	
	@Test
	public void testWeightedDivMM4MultMinusRightSparseSP() {
		runWeightedDivMMTest(TEST_NAME9, true, true, false, ExecType.SPARK);
	}
	
	@Test
	public void testWeightedDivMM4MultMinusRightDenseSPRep() {
		runWeightedDivMMTest(TEST_NAME9, false, true, true, ExecType.SPARK);
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
			boolean basic = testname.equals(TEST_NAME3);
			boolean left = testname.equals(TEST_NAME1) || testname.equals(TEST_NAME4) 
					|| testname.equals(TEST_NAME6) || testname.equals(TEST_NAME8);
			double sparsity = (sparse) ? spSparse : spDense;
			String TEST_NAME = testname;
			String TEST_CACHE_DIR = TEST_CACHE_ENABLED ? 
					TEST_CACHE_DIR = TEST_NAME + "_" + sparsity + "/" : "";
			
			TestConfiguration config = getTestConfiguration(TEST_NAME);
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
			HashMap<CellIndex, Double> dmlfile = readDMLMatrixFromHDFS("R");
			HashMap<CellIndex, Double> rfile  = readRMatrixFromFS("R");
			TestUtils.compareMatrices(dmlfile, rfile, eps, "Stat-DML", "Stat-R");
			checkDMLMetaDataFile("R", new MatrixCharacteristics(left?cols:rows, basic?cols:rank, 1, 1));

			//check statistics for right operator in cp and spark
			if( instType == ExecType.CP && rewrites ) {
				Assert.assertTrue("Missing opcode wdivmm", Statistics.getCPHeavyHitterOpCodes().contains(WeightedDivMM.OPCODE_CP));
			}
			else if( instType == ExecType.SPARK && rewrites ) {
				boolean reduce = rep || testname.equals(TEST_NAME8) || testname.equals(TEST_NAME9);
				String opcode = Instruction.SP_INST_PREFIX + ((reduce)?WeightedDivMMR.OPCODE:WeightedDivMM.OPCODE);
				Assert.assertTrue("Missing opcode sp_wdivmm", Statistics.getCPHeavyHitterOpCodes().contains(opcode) );
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