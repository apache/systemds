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

package org.apache.sysds.test.functions.binary.matrix_full_other;

import java.util.HashMap;

import org.apache.sysds.common.Opcodes;
import org.junit.AfterClass;
import org.junit.Assert;
import org.junit.BeforeClass;
import org.junit.Test;
import org.apache.sysds.api.DMLScript;
import org.apache.sysds.common.Types.ExecMode;
import org.apache.sysds.hops.AggBinaryOp;
import org.apache.sysds.hops.AggBinaryOp.MMultMethod;
import org.apache.sysds.common.Types.ExecType;
import org.apache.sysds.lops.MMTSJ.MMTSJType;
import org.apache.sysds.runtime.instructions.Instruction;
import org.apache.sysds.runtime.matrix.data.MatrixValue.CellIndex;
import org.apache.sysds.test.AutomatedTestBase;
import org.apache.sysds.test.TestConfiguration;
import org.apache.sysds.test.TestUtils;
import org.apache.sysds.utils.Statistics;

/**
 * This testcase validates the new tsmm2 (spark only) instruction. We test against
 * all backends to ensure this operator is not mistakenly compiled for MR/CP. 
 * 
 */
public class FullMatrixMultiplicationTransposeSelf2Test extends AutomatedTestBase 
{
	private final static String TEST_NAME1 = "TransposeSelfMatrixMultiplication1";
	private final static String TEST_NAME2 = "TransposeSelfMatrixMultiplication2";
	private final static String TEST_DIR = "functions/binary/matrix_full_other/";
	private final static String TEST_CLASS_DIR = TEST_DIR + FullMatrixMultiplicationTransposeSelf2Test.class.getSimpleName() + "/";
	private final static double eps = 1e-10;
	
	private final static int rows1 = 2105;
	private final static int cols1 = 1003; //multi-block, but excess fits in broadcast
	
	private final static double sparsity1 = 0.64;
	private final static double sparsity2 = 0.07;
	
	
	@Override
	public void setUp() {
		TestUtils.clearAssertionInformation();
		addTestConfiguration(TEST_NAME1, new TestConfiguration(TEST_CLASS_DIR, TEST_NAME1, new String[] { "B" })); 
		addTestConfiguration(TEST_NAME2, new TestConfiguration(TEST_CLASS_DIR, TEST_NAME2, new String[] { "B" })); 
		if (TEST_CACHE_ENABLED)
			setOutAndExpectedDeletionDisabled(true);
	}

	@BeforeClass
	public static void init() {
		TestUtils.clearDirectory(TEST_DATA_DIR + TEST_CLASS_DIR);
	}

	@AfterClass
	public static void cleanUp() {
		if (TEST_CACHE_ENABLED)
			TestUtils.clearDirectory(TEST_DATA_DIR + TEST_CLASS_DIR);
	}

	@Test
	public void testTSMMLeftDenseCP() {
		runTransposeSelfMatrixMultiplicationTest(MMTSJType.LEFT, ExecType.CP, false);
	}
	
	@Test
	public void testTSMMLeftSparseCP() {
		runTransposeSelfMatrixMultiplicationTest(MMTSJType.LEFT, ExecType.CP, true);
	}
	
	@Test
	public void testTSMMRightDenseCP() {
		runTransposeSelfMatrixMultiplicationTest(MMTSJType.RIGHT, ExecType.CP, false);
	}
	
	@Test
	public void testTSMMRightSparseCP() {
		runTransposeSelfMatrixMultiplicationTest(MMTSJType.RIGHT, ExecType.CP, true);
	}
	
	@Test
	public void testTSMMLeftDenseSP() {
		runTransposeSelfMatrixMultiplicationTest(MMTSJType.LEFT, ExecType.SPARK, false);
	}
	
	@Test
	public void testTSMMLeftSparseSP() {
		runTransposeSelfMatrixMultiplicationTest(MMTSJType.LEFT, ExecType.SPARK, true);
	}
	
	@Test
	public void testTSMMRightDenseSP() {
		runTransposeSelfMatrixMultiplicationTest(MMTSJType.RIGHT, ExecType.SPARK, false);
	}
	
	@Test
	public void testTSMMRightSparseSP() {
		runTransposeSelfMatrixMultiplicationTest(MMTSJType.RIGHT, ExecType.SPARK, true);
	}
	
	private void runTransposeSelfMatrixMultiplicationTest( MMTSJType type, ExecType instType, boolean sparse )
	{
		ExecMode platformOld = rtplatform;
		switch( instType ){
			case SPARK: rtplatform = ExecMode.SPARK; break;
			default: rtplatform = ExecMode.HYBRID; break;
		}
	
		boolean sparkConfigOld = DMLScript.USE_LOCAL_SPARK_CONFIG;
		if( rtplatform == ExecMode.SPARK )
			DMLScript.USE_LOCAL_SPARK_CONFIG = true;

		if( instType == ExecType.SPARK ) //force tsmm2 to prevent mapmm
			AggBinaryOp.FORCED_MMULT_METHOD = MMultMethod.TSMM2;
		
		//setup exec type, rows, cols, caching dir
		int rows = (type == MMTSJType.LEFT) ? rows1 : cols1;
		int cols = (type == MMTSJType.LEFT) ? cols1 : rows1;
		double sparsity = sparse ? sparsity2 : sparsity1;
		String TEST_NAME = (type == MMTSJType.LEFT) ? TEST_NAME1 : TEST_NAME2;
		
		String TEST_CACHE_DIR = "";
		if (TEST_CACHE_ENABLED)
			TEST_CACHE_DIR = rows + "_" + cols + "_" + sparsity + "/";
		
		try
		{
			TestConfiguration config = getTestConfiguration(TEST_NAME);
			loadTestConfiguration(config, TEST_CACHE_DIR);
			
			/* This is for running the junit test the new way, i.e., construct the arguments directly */
			String HOME = SCRIPT_DIR + TEST_DIR;
			fullDMLScriptName = HOME + TEST_NAME + ".dml";
			programArgs = new String[]{"-stats", "-args", input("A"),
				Integer.toString(rows), Integer.toString(cols), output("B") };
			
			fullRScriptName = HOME + TEST_NAME + ".R";
			rCmd = "Rscript" + " " + fullRScriptName + " " + inputDir() + " " + expectedDir();
	
			//generate actual dataset
			double[][] A = getRandomMatrix(rows, cols, 0, 1, sparsity, 7); 
			writeInputMatrix("A", A, true);
	
			//run dml and R scripts
			
			runTest(true, false, null, -1); 
			runRScript(true); 
			
			//compare matrices 
			HashMap<CellIndex, Double> dmlfile = readDMLMatrixFromOutputDir("B");
			HashMap<CellIndex, Double> rfile  = readRMatrixFromExpectedDir("B");
			TestUtils.compareMatrices(dmlfile, rfile, eps, "Stat-DML", "Stat-R");
		
			//check for compiled tsmm instructions
			if( instType == ExecType.SPARK || instType == ExecType.CP ) {
				String opcode = (instType==ExecType.SPARK) ? Instruction.SP_INST_PREFIX + "tsmm2" : Opcodes.TSMM.toString();
				Assert.assertTrue("Missing opcode: "+opcode, Statistics.getCPHeavyHitterOpCodes().contains(opcode) );
			}
		}
		finally {
			DMLScript.USE_LOCAL_SPARK_CONFIG = sparkConfigOld;
			AggBinaryOp.FORCED_MMULT_METHOD = null;
			rtplatform = platformOld;
		}
	}
}