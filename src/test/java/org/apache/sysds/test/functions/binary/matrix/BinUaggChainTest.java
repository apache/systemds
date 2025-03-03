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

package org.apache.sysds.test.functions.binary.matrix;

import java.util.HashMap;

import org.apache.sysds.common.Opcodes;
import org.junit.Assert;
import org.junit.Test;
import org.apache.sysds.api.DMLScript;
import org.apache.sysds.common.Types.ExecMode;
import org.apache.sysds.common.Types.ExecType;
import org.apache.sysds.runtime.matrix.data.MatrixValue.CellIndex;
import org.apache.sysds.test.AutomatedTestBase;
import org.apache.sysds.test.TestConfiguration;
import org.apache.sysds.test.TestUtils;

public class BinUaggChainTest extends AutomatedTestBase 
{
	private final static String TEST_NAME1 = "BinUaggChain_Col";
	private final static String TEST_DIR = "functions/binary/matrix/";
	private final static String TEST_CLASS_DIR = TEST_DIR + BinUaggChainTest.class.getSimpleName() + "/";
	private final static double eps = 1e-8;
	
	private final static int rows = 1468;
	private final static int cols1 = 73; //single block
	private final static int cols2 = 1052; //multi block
	
	private final static double sparsity1 = 0.5; //dense
	private final static double sparsity2 = 0.1; //sparse
	
	@Override
	public void setUp() {
		TestUtils.clearAssertionInformation();
		addTestConfiguration(TEST_NAME1, new TestConfiguration(TEST_CLASS_DIR, TEST_NAME1, new String[] { "B" })); 
	}
	
	// -------------------------
	
	@Test
	public void testBinUaggChainColSingleDenseSP() {
		 runBinUaggTest(TEST_NAME1, true, false, ExecType.SPARK);
	}
	
	@Test
	public void testBinUaggChainColSingleSparseSP() {
		runBinUaggTest(TEST_NAME1, true, true, ExecType.SPARK);
	}
	
	@Test
	public void testBinUaggChainColMultiDenseSP() {
		runBinUaggTest(TEST_NAME1, false, false, ExecType.SPARK);
	}
	
	@Test
	public void testBinUaggChainColMultiSparseSP() {
		runBinUaggTest(TEST_NAME1, false, true, ExecType.SPARK);
	}
	
	private void runBinUaggTest( String testname, boolean singleBlock, boolean sparse, ExecType instType)
	{
		//rtplatform for MR
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
			getAndLoadTestConfiguration(TEST_NAME);
			
			String HOME = SCRIPT_DIR + TEST_DIR;
			fullDMLScriptName = HOME + TEST_NAME + ".dml";
			programArgs = new String[]{"-stats","-args", input("A"), output("B")};
			
			fullRScriptName = HOME + TEST_NAME + ".R";
			rCmd = "Rscript" + " " + fullRScriptName + " " + inputDir() + " " + expectedDir();
	
			//generate actual datasets
			double[][] A = getRandomMatrix(rows, singleBlock?cols1:cols2, -1, 1, sparse?sparsity2:sparsity1, 7);
			writeInputMatrixWithMTD("A", A, true);
			
			runTest(true, false, null, -1); 
			runRScript(true); 
			
			//compare matrices 
			HashMap<CellIndex, Double> dmlfile = readDMLMatrixFromOutputDir("B");
			HashMap<CellIndex, Double> rfile  = readRMatrixFromExpectedDir("B");
			TestUtils.compareMatrices(dmlfile, rfile, eps, "Stat-DML", "Stat-R");
			
			//check compiled/executed jobs
			int expectedNumCompiled = (singleBlock)?4:5; 
			int expectedNumExecuted = (singleBlock)?4:5; 
			checkNumCompiledSparkInst(expectedNumCompiled);
			checkNumExecutedSparkInst(expectedNumExecuted);
			Assert.assertEquals(singleBlock,
				heavyHittersContainsSubString(Opcodes.BINUAGGCHAIN.toString()));
		}
		finally {
			rtplatform = platformOld;
			DMLScript.USE_LOCAL_SPARK_CONFIG = sparkConfigOld;
		}
	}
}
