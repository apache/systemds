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

package org.apache.sysml.test.integration.functions.ternary;

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
 */
public class TernaryAggregateTest extends AutomatedTestBase 
{
	private final static String TEST_NAME1 = "TernaryAggregateRC";
	private final static String TEST_NAME2 = "TernaryAggregateC";
	
	private final static String TEST_DIR = "functions/ternary/";
	private final static String TEST_CLASS_DIR = TEST_DIR + TernaryAggregateTest.class.getSimpleName() + "/";
	private final static double eps = 1e-8;
	
	private final static int rows = 1111;
	private final static int cols = 1011;
	
	private final static double sparsity1 = 0.7;
	private final static double sparsity2 = 0.3;
	
	@Override
	public void setUp() {
		TestUtils.clearAssertionInformation();
		addTestConfiguration(TEST_NAME1, new TestConfiguration(TEST_CLASS_DIR, TEST_NAME1, new String[] { "R" }) ); 
		addTestConfiguration(TEST_NAME2, new TestConfiguration(TEST_CLASS_DIR, TEST_NAME2, new String[] { "R" }) ); 
	}

	@Test
	public void testTernaryAggregateRCDenseVectorCP() {
		runTernaryAggregateTest(TEST_NAME1, false, true, true, ExecType.CP);
	}
	
	@Test
	public void testTernaryAggregateRCSparseVectorCP() {
		runTernaryAggregateTest(TEST_NAME1, true, true, true, ExecType.CP);
	}
	
	@Test
	public void testTernaryAggregateRCDenseMatrixCP() {
		runTernaryAggregateTest(TEST_NAME1, false, false, true, ExecType.CP);
	}
	
	@Test
	public void testTernaryAggregateRCSparseMatrixCP() {
		runTernaryAggregateTest(TEST_NAME1, true, false, true, ExecType.CP);
	}
	
	@Test
	public void testTernaryAggregateRCDenseVectorSP() {
		runTernaryAggregateTest(TEST_NAME1, false, true, true, ExecType.SPARK);
	}
	
	@Test
	public void testTernaryAggregateRCSparseVectorSP() {
		runTernaryAggregateTest(TEST_NAME1, true, true, true, ExecType.SPARK);
	}
	
	@Test
	public void testTernaryAggregateRCDenseMatrixSP() {
		runTernaryAggregateTest(TEST_NAME1, false, false, true, ExecType.SPARK);
	}
	
	@Test
	public void testTernaryAggregateRCSparseMatrixSP() {
		runTernaryAggregateTest(TEST_NAME1, true, false, true, ExecType.SPARK);
	}
	
	@Test
	public void testTernaryAggregateRCDenseVectorMR() {
		runTernaryAggregateTest(TEST_NAME1, false, true, true, ExecType.MR);
	}
	
	@Test
	public void testTernaryAggregateRCSparseVectorMR() {
		runTernaryAggregateTest(TEST_NAME1, true, true, true, ExecType.MR);
	}
	
	@Test
	public void testTernaryAggregateRCDenseMatrixMR() {
		runTernaryAggregateTest(TEST_NAME1, false, false, true, ExecType.MR);
	}
	
	@Test
	public void testTernaryAggregateRCSparseMatrixMR() {
		runTernaryAggregateTest(TEST_NAME1, true, false, true, ExecType.MR);
	}
	
	@Test
	public void testTernaryAggregateCDenseVectorCP() {
		runTernaryAggregateTest(TEST_NAME2, false, true, true, ExecType.CP);
	}
	
	@Test
	public void testTernaryAggregateCSparseVectorCP() {
		runTernaryAggregateTest(TEST_NAME2, true, true, true, ExecType.CP);
	}
	
	@Test
	public void testTernaryAggregateCDenseMatrixCP() {
		runTernaryAggregateTest(TEST_NAME2, false, false, true, ExecType.CP);
	}
	
	@Test
	public void testTernaryAggregateCSparseMatrixCP() {
		runTernaryAggregateTest(TEST_NAME2, true, false, true, ExecType.CP);
	}
	
	@Test
	public void testTernaryAggregateCDenseVectorSP() {
		runTernaryAggregateTest(TEST_NAME2, false, true, true, ExecType.SPARK);
	}
	
	@Test
	public void testTernaryAggregateCSparseVectorSP() {
		runTernaryAggregateTest(TEST_NAME2, true, true, true, ExecType.SPARK);
	}
	
	@Test
	public void testTernaryAggregateCDenseMatrixSP() {
		runTernaryAggregateTest(TEST_NAME2, false, false, true, ExecType.SPARK);
	}
	
	@Test
	public void testTernaryAggregateCSparseMatrixSP() {
		runTernaryAggregateTest(TEST_NAME2, true, false, true, ExecType.SPARK);
	}
	
	//additional tests to check default without rewrites
	
	@Test
	public void testTernaryAggregateRCDenseVectorCPNoRewrite() {
		runTernaryAggregateTest(TEST_NAME1, false, true, false, ExecType.CP);
	}
	
	@Test
	public void testTernaryAggregateRCSparseVectorCPNoRewrite() {
		runTernaryAggregateTest(TEST_NAME1, true, true, false, ExecType.CP);
	}
	
	@Test
	public void testTernaryAggregateRCDenseMatrixCPNoRewrite() {
		runTernaryAggregateTest(TEST_NAME1, false, false, false, ExecType.CP);
	}
	
	@Test
	public void testTernaryAggregateRCSparseMatrixCPNoRewrite() {
		runTernaryAggregateTest(TEST_NAME1, true, false, false, ExecType.CP);
	}
	
	@Test
	public void testTernaryAggregateCDenseVectorCPNoRewrite() {
		runTernaryAggregateTest(TEST_NAME2, false, true, false, ExecType.CP);
	}
	
	@Test
	public void testTernaryAggregateCSparseVectorCPNoRewrite() {
		runTernaryAggregateTest(TEST_NAME2, true, true, false, ExecType.CP);
	}
	
	@Test
	public void testTernaryAggregateCDenseMatrixCPNoRewrite() {
		runTernaryAggregateTest(TEST_NAME2, false, false, false, ExecType.CP);
	}
	
	@Test
	public void testTernaryAggregateCSparseMatrixCPNoRewrite() {
		runTernaryAggregateTest(TEST_NAME2, true, false, false, ExecType.CP);
	}
	
	
	
	private void runTernaryAggregateTest(String testname, boolean sparse, boolean vectors, boolean rewrites, ExecType et)
	{
		//rtplatform for MR
		RUNTIME_PLATFORM platformOld = rtplatform;
		switch( et ){
			case MR: rtplatform = RUNTIME_PLATFORM.HADOOP; break;
			case SPARK: rtplatform = RUNTIME_PLATFORM.SPARK; break;
			default: rtplatform = RUNTIME_PLATFORM.HYBRID; break;
		}
	
		boolean sparkConfigOld = DMLScript.USE_LOCAL_SPARK_CONFIG;
		if( rtplatform == RUNTIME_PLATFORM.SPARK )
			DMLScript.USE_LOCAL_SPARK_CONFIG = true;
	
		boolean rewritesOld = OptimizerUtils.ALLOW_SUM_PRODUCT_REWRITES;
		
		try
		{
			TestConfiguration config = getTestConfiguration(testname);
			loadTestConfiguration(config);
			
			OptimizerUtils.ALLOW_SUM_PRODUCT_REWRITES = rewrites;
			
			String HOME = SCRIPT_DIR + TEST_DIR;
			fullDMLScriptName = HOME + testname + ".dml";
			programArgs = new String[]{"-stats","-args", input("A"), output("R")};
			
			fullRScriptName = HOME + testname + ".R";
			rCmd = "Rscript" + " " + fullRScriptName + " " + 
				inputDir() + " " + expectedDir();
	
			//generate actual dataset
			double sparsity = sparse ? sparsity2 : sparsity1;
			double[][] A = getRandomMatrix(vectors ? rows*cols : rows, 
					vectors ? 1 : cols, 0, 1, sparsity, 17); 
			writeInputMatrixWithMTD("A", A, true);
			
			//run test cases
			runTest(true, false, null, -1); 
			runRScript(true); 
			
			//compare output matrices 
			HashMap<CellIndex, Double> dmlfile = readDMLMatrixFromHDFS("R");
			HashMap<CellIndex, Double> rfile  = readRMatrixFromFS("R");
			TestUtils.compareMatrices(dmlfile, rfile, eps, "Stat-DML", "Stat-R");
			
			//check for rewritten patterns in statistics output
			if( rewrites && et != ExecType.MR ) {
				String opcode = ((et == ExecType.SPARK) ? Instruction.SP_INST_PREFIX : "") + 
					(((testname.equals(TEST_NAME1) || vectors ) ? "tak+*" : "tack+*"));
				Assert.assertEquals(new Boolean(true), new Boolean(
					Statistics.getCPHeavyHitterOpCodes().contains(opcode)));
			}
		}
		finally {
			rtplatform = platformOld;
			DMLScript.USE_LOCAL_SPARK_CONFIG = sparkConfigOld;
			OptimizerUtils.ALLOW_SUM_PRODUCT_REWRITES = rewritesOld;
		}
	}
}
