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
import org.apache.sysds.lops.WeightedCrossEntropy;
import org.apache.sysds.lops.WeightedCrossEntropyR;
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
public class WeightedCrossEntropyTest extends AutomatedTestBase 
{
	private final static String TEST_NAME = "WeightedCeMM";
	private final static String TEST_NAME2 = "WeightedCeMMEps";
	private final static String TEST_DIR = "functions/quaternary/";
	private final static String TEST_CLASS_DIR = TEST_DIR + WeightedCrossEntropyTest.class.getSimpleName() + "/";
	
	private final static double eps = 1e-6;
	private final static double log_eps = 0.1;
	
	private final static int rows = 1201;
	private final static int cols = 1103;
	private final static int rank = 10;
	private final static double spSparse = 0.002;
	private final static double spDense = 0.8;
	
	@Override
	public void setUp() 
	{
		TestUtils.clearAssertionInformation();
		addTestConfiguration(TEST_NAME,new TestConfiguration(TEST_CLASS_DIR, TEST_NAME,new String[]{"R"}));
		addTestConfiguration(TEST_NAME2,new TestConfiguration(TEST_CLASS_DIR, TEST_NAME2,new String[]{"R"}));
		
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
	
	@Test
	public void testCrossEntropyDenseCP() {
		runWeightedCrossEntropyTest(TEST_NAME, false, true, false, ExecType.CP);
	}
	
	@Test
	public void testCrossEntropySparseCP() {
		runWeightedCrossEntropyTest(TEST_NAME, true, true, false, ExecType.CP);
	}
	
	@Test
	public void testCrossEntropyDenseSP() {
		runWeightedCrossEntropyTest(TEST_NAME, false, true, false, ExecType.SPARK);
	}
	
	@Test
	public void testCrossEntropySparseSP() {
		runWeightedCrossEntropyTest(TEST_NAME, true, true, false, ExecType.SPARK);
	}
	
	@Test
	public void testCrossEntropyDenseSPRep() {
		runWeightedCrossEntropyTest(TEST_NAME, false, true, true, ExecType.SPARK);
	}
	
	// test cases for wcemm with Epsilon (sum(X*log(U%*%t(V) + eps)))
	
	@Test
	public void testCrossEntropyEpsDenseCP() {
		runWeightedCrossEntropyTest(TEST_NAME2, false, true, false, ExecType.CP);
	}
	
	@Test
	public void testCrossEntropyEpsSparseCP() {
		runWeightedCrossEntropyTest(TEST_NAME2, true, true, false, ExecType.CP);
	}
	
	@Test
	public void testCrossEntropyEpsDenseSP() {
		runWeightedCrossEntropyTest(TEST_NAME2, false, true, false, ExecType.SPARK);
	}
	
	@Test
	public void testCrossEntropyEpsSparseSP() {
		runWeightedCrossEntropyTest(TEST_NAME2, true, true, false, ExecType.SPARK);
	}
	
	@Test
	public void testCrossEntropyEpsDenseSPRep() {
		runWeightedCrossEntropyTest(TEST_NAME2, false, true, true, ExecType.SPARK);
	}
	
	
	/**
	 * 
	 * @param testname
	 * @param sparse
	 * @param rewrites
	 * @param rep
	 * @param instType
	 */
	private void runWeightedCrossEntropyTest( String testname, boolean sparse, boolean rewrites, boolean rep, ExecType instType)
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
				String eps = (TEST_NAME2.equals(testname)) ? "_" + log_eps : "";
				TEST_CACHE_DIR = sparsity + eps + "/";
			}
			
			loadTestConfiguration(config, TEST_CACHE_DIR);
			
			// This is for running the junit test the new way, i.e., construct the arguments directly
			String HOME = SCRIPT_DIR + TEST_DIR;
			fullDMLScriptName = HOME + TEST_NAME + ".dml";
			programArgs = new String[]{"-stats", "-explain", "runtime", "-args", 
				input("X"), input("U"), input("V"), output("R"), Double.toString(log_eps) };
			
			fullRScriptName = HOME + TEST_NAME + ".R";
			rCmd = "Rscript" + " " + fullRScriptName + " " + inputDir() + " " + expectedDir() + " " + log_eps;
	
			//generate actual dataset 
			double[][] X = getRandomMatrix(rows, cols, 0, 1, sparsity, 7); 
			writeInputMatrixWithMTD("X", X, true);
			double[][] U = getRandomMatrix(rows, rank, 0, 1, 1.0, 678); 
			writeInputMatrixWithMTD("U", U, true);
			double[][] V = getRandomMatrix(cols, rank, 0, 1, 1.0, 912); 
			writeInputMatrixWithMTD("V", V, true);
			
			//run the scripts
			runTest(true, false, null, -1); 
			runRScript(true); 
		
			//compare matrices 
			HashMap<CellIndex, Double> dmlfile = readDMLMatrixFromOutputDir("R");
			HashMap<CellIndex, Double> rfile  = readRMatrixFromExpectedDir("R");
			TestUtils.compareMatrices(dmlfile, rfile, eps, "Stat-DML", "Stat-R");
			checkDMLMetaDataFile("R", new MatrixCharacteristics(1,1,1,1));

			//check statistics for right operator in cp
			if( instType == ExecType.CP && rewrites )
				Assert.assertTrue("Missing opcode wcemm", Statistics.getCPHeavyHitterOpCodes().contains(WeightedCrossEntropy.OPCODE_CP));
			else if( instType == ExecType.SPARK && rewrites ) {
				Assert.assertTrue("Missing opcode sp_wcemm", 
						!rep && Statistics.getCPHeavyHitterOpCodes().contains(Instruction.SP_INST_PREFIX+Opcodes.WEIGHTEDCROSSENTROPY)
					  || rep && Statistics.getCPHeavyHitterOpCodes().contains(Instruction.SP_INST_PREFIX+ Opcodes.WEIGHTEDCROSSENTROPYR) );
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