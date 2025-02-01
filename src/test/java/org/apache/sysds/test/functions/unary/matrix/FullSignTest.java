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

package org.apache.sysds.test.functions.unary.matrix;

import java.util.HashMap;

import org.apache.sysds.common.Opcodes;
import org.junit.AfterClass;
import org.junit.Assert;
import org.junit.BeforeClass;
import org.junit.Test;
import org.apache.sysds.api.DMLScript;
import org.apache.sysds.common.Types.ExecMode;
import org.apache.sysds.common.Types.ExecType;
import org.apache.sysds.runtime.instructions.Instruction;
import org.apache.sysds.runtime.matrix.data.MatrixValue.CellIndex;
import org.apache.sysds.test.AutomatedTestBase;
import org.apache.sysds.test.TestConfiguration;
import org.apache.sysds.test.TestUtils;
import org.apache.sysds.utils.Statistics;

/**
 * 
 * 
 */
public class FullSignTest extends AutomatedTestBase 
{
	private final static String TEST_NAME1 = "Sign1";
	private final static String TEST_NAME2 = "Sign2";
	private final static String TEST_DIR = "functions/unary/matrix/";
	private static final String TEST_CLASS_DIR = TEST_DIR + FullSignTest.class.getSimpleName() + "/";
	
	private final static int rows = 1108;
	private final static int cols = 1001;
	private final static double spSparse = 0.05;
	private final static double spDense = 0.7;
	
	@Override
	public void setUp() 
	{
		addTestConfiguration(TEST_NAME1, new TestConfiguration(TEST_CLASS_DIR, TEST_NAME1,new String[]{"B"})); 
		addTestConfiguration(TEST_NAME2, new TestConfiguration(TEST_CLASS_DIR, TEST_NAME2,new String[]{"B"})); 
		
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
	public void testSignDenseCP() {
		runSignTest(TEST_NAME1, false, ExecType.CP);
	}
	
	@Test
	public void testSignSparseCP() {
		runSignTest(TEST_NAME1, true, ExecType.CP);
	}
	
	@Test
	public void testSignDenseSP() {
		runSignTest(TEST_NAME1, false, ExecType.SPARK);
	}
	
	@Test
	public void testSignSparseSP() {
		runSignTest(TEST_NAME1, true, ExecType.SPARK);
	}

	@Test
	public void testRewriteSignDenseCP() {
		runSignTest(TEST_NAME2, false, ExecType.CP);
	}
	
	@Test
	public void testRewriteSignSparseCP() {
		runSignTest(TEST_NAME2, true, ExecType.CP);
	}
	
	@Test
	public void testRewriteSignDenseSP() {
		runSignTest(TEST_NAME2, false, ExecType.SPARK);
	}
	
	@Test
	public void testRewriteSignSparseSP() {
		runSignTest(TEST_NAME2, true, ExecType.SPARK);
	}
	
	private void runSignTest( String testname, boolean sparse, ExecType instType)
	{
		ExecMode platformOld = rtplatform;
		switch( instType ){
			case SPARK: rtplatform = ExecMode.SPARK; break;
			default: rtplatform = ExecMode.HYBRID; break;
		}
	
		boolean sparkConfigOld = DMLScript.USE_LOCAL_SPARK_CONFIG;
		if( rtplatform == ExecMode.SPARK )
			DMLScript.USE_LOCAL_SPARK_CONFIG = true;

		try
		{
			String TEST_NAME = testname;
			double sparsity = (sparse) ? spSparse : spDense;
			
			String TEST_CACHE_DIR = "";
			if (TEST_CACHE_ENABLED)
			{
				TEST_CACHE_DIR = sparsity + "/";
			}
			
			TestConfiguration config = getTestConfiguration(TEST_NAME);
			loadTestConfiguration(config, TEST_CACHE_DIR);
			
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
			HashMap<CellIndex, Double> dmlfile = readDMLMatrixFromOutputDir("B");
			HashMap<CellIndex, Double> rfile  = readRMatrixFromExpectedDir("B");
			TestUtils.compareMatrices(dmlfile, rfile, 0, "Stat-DML", "Stat-R");
			
			//check generated opcode
			if( instType == ExecType.CP ) {
				Assert.assertTrue("Missing opcode: sign", Statistics.getCPHeavyHitterOpCodes().contains(Opcodes.SIGN.toString()) ||
						Statistics.getCPHeavyHitterOpCodes().contains("gpu_sign") );
			}
			else if ( instType == ExecType.SPARK )
				Assert.assertTrue("Missing opcode: "+Instruction.SP_INST_PREFIX+Opcodes.SIGN.toString(), Statistics.getCPHeavyHitterOpCodes().contains(Instruction.SP_INST_PREFIX+Opcodes.SIGN.toString()));
		}
		finally {
			rtplatform = platformOld;
			DMLScript.USE_LOCAL_SPARK_CONFIG = sparkConfigOld;
		}
	}
}
