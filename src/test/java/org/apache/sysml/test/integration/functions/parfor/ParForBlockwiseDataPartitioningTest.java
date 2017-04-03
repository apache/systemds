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

package org.apache.sysml.test.integration.functions.parfor;

import java.util.HashMap;

import org.junit.Assert;
import org.junit.Test;

import org.apache.sysml.api.DMLScript;
import org.apache.sysml.api.DMLScript.RUNTIME_PLATFORM;
import org.apache.sysml.conf.CompilerConfig;
import org.apache.sysml.runtime.controlprogram.ParForProgramBlock.PDataPartitioner;
import org.apache.sysml.runtime.controlprogram.ParForProgramBlock.PExecMode;
import org.apache.sysml.runtime.matrix.data.MatrixValue.CellIndex;
import org.apache.sysml.test.integration.AutomatedTestBase;
import org.apache.sysml.test.integration.TestConfiguration;
import org.apache.sysml.test.utils.TestUtils;

public class ParForBlockwiseDataPartitioningTest extends AutomatedTestBase 
{	
	//positive test cases, i.e., test cases where row/column block partitioning applied
	private final static String TEST_NAME1 = "parfor_brdatapartitioning_pos";
	private final static String TEST_NAME2 = "parfor_bcdatapartitioning_pos";
	//negative test cases, i.e., test cases where row/column block partitioning not applied
	private final static String TEST_NAME3 = "parfor_brdatapartitioning_neg";
	private final static String TEST_NAME4 = "parfor_bcdatapartitioning_neg";
	
	private final static String TEST_DIR = "functions/parfor/";
	private final static String TEST_CLASS_DIR = TEST_DIR + ParForBlockwiseDataPartitioningTest.class.getSimpleName() + "/";
	private final static double eps = 1e-10;
	
	//moderate data size, force spark rix via unknowns 
	private final static int rows = (int)1014; 
	private final static int cols = (int)2024;
	private final static double sparsity1 = 0.7;
	private final static double sparsity2 = 0.01;
	
	@Override
	public void setUp() {
		TestUtils.clearAssertionInformation();
		addTestConfiguration(TEST_NAME1, new TestConfiguration(TEST_CLASS_DIR, TEST_NAME1, new String[] { "Rout" }) ); 
		addTestConfiguration(TEST_NAME2, new TestConfiguration(TEST_CLASS_DIR, TEST_NAME2, new String[] { "Rout" }) ); 
		addTestConfiguration(TEST_NAME3, new TestConfiguration(TEST_CLASS_DIR, TEST_NAME3, new String[] { "Rout" }) ); 
		addTestConfiguration(TEST_NAME4, new TestConfiguration(TEST_CLASS_DIR, TEST_NAME4, new String[] { "Rout" }) ); 
	}

	
	@Test
	public void testParForRowBlockPartitioningLocalLocalDense() {
		runParForDataPartitioningTest(TEST_NAME1, PDataPartitioner.LOCAL, PExecMode.LOCAL, false);
	}

	@Test
	public void testParForRowBlockPartitioningLocalRemoteDense() {
		runParForDataPartitioningTest(TEST_NAME1, PDataPartitioner.LOCAL, PExecMode.REMOTE_SPARK, false);
	}	

	@Test
	public void testParForRowBlockPartitioningRemoteLocalDense() {
		runParForDataPartitioningTest(TEST_NAME1, PDataPartitioner.REMOTE_SPARK, PExecMode.LOCAL, false);
	}

	@Test
	public void testParForRowBlockPartitioningRemoteRemoteDense() {
		runParForDataPartitioningTest(TEST_NAME1, PDataPartitioner.REMOTE_SPARK, PExecMode.REMOTE_SPARK, false);
	}

	@Test
	public void testParForRowBlockPartitioningLocalLocalSparse() {
		runParForDataPartitioningTest(TEST_NAME1, PDataPartitioner.LOCAL, PExecMode.LOCAL, true);
	}

	@Test
	public void testParForRowBlockPartitioningLocalRemoteSparse() {
		runParForDataPartitioningTest(TEST_NAME1, PDataPartitioner.LOCAL, PExecMode.REMOTE_SPARK, true);
	}	

	@Test
	public void testParForRowBlockPartitioningRemoteLocalSparse() {
		runParForDataPartitioningTest(TEST_NAME1, PDataPartitioner.REMOTE_SPARK, PExecMode.LOCAL, true);
	}

	@Test
	public void testParForRowBlockPartitioningRemoteRemoteSparse() {
		runParForDataPartitioningTest(TEST_NAME1, PDataPartitioner.REMOTE_SPARK, PExecMode.REMOTE_SPARK, true);
	}

	@Test
	public void testParForColBlockPartitioningLocalLocalDense() {
		runParForDataPartitioningTest(TEST_NAME2, PDataPartitioner.LOCAL, PExecMode.LOCAL, false);
	}

	@Test
	public void testParForColBlockPartitioningLocalRemoteDense() {
		runParForDataPartitioningTest(TEST_NAME2, PDataPartitioner.LOCAL, PExecMode.REMOTE_SPARK, false);
	}	

	@Test
	public void testParForColBlockPartitioningRemoteLocalDense() {
		runParForDataPartitioningTest(TEST_NAME2, PDataPartitioner.REMOTE_SPARK, PExecMode.LOCAL, false);
	}

	@Test
	public void testParForColBlockPartitioningRemoteRemoteDense() {
		runParForDataPartitioningTest(TEST_NAME2, PDataPartitioner.REMOTE_SPARK, PExecMode.REMOTE_SPARK, false);
	}

	@Test
	public void testParForColBlockPartitioningLocalLocalSparse() {
		runParForDataPartitioningTest(TEST_NAME2, PDataPartitioner.LOCAL, PExecMode.LOCAL, true);
	}

	@Test
	public void testParForColBlockPartitioningLocalRemoteSparse() {
		runParForDataPartitioningTest(TEST_NAME2, PDataPartitioner.LOCAL, PExecMode.REMOTE_SPARK, true);
	}	

	@Test
	public void testParForColBlockPartitioningRemoteLocalSparse() {
		runParForDataPartitioningTest(TEST_NAME2, PDataPartitioner.REMOTE_SPARK, PExecMode.LOCAL, true);
	}

	@Test
	public void testParForColBlockPartitioningRemoteRemoteSparse() {
		runParForDataPartitioningTest(TEST_NAME2, PDataPartitioner.REMOTE_SPARK, PExecMode.REMOTE_SPARK, true);
	}
	
	//fused data partition execute
	
	@Test
	public void testParForRowBlockPartitioningRemoteRemoteFusedDense() {
		runParForDataPartitioningTest(TEST_NAME1, PDataPartitioner.UNSPECIFIED, PExecMode.REMOTE_SPARK_DP, false);
	}

	@Test
	public void testParForColBlockPartitioningRemoteRemoteFusedDense() {
		runParForDataPartitioningTest(TEST_NAME2, PDataPartitioner.UNSPECIFIED, PExecMode.REMOTE_SPARK_DP, false);
	}

	
	//negative examples
	
	@Test
	public void testParForRowBlockPartitioningRemoteLocalSparseNegative() {
		runParForDataPartitioningTest(TEST_NAME3, PDataPartitioner.REMOTE_SPARK, PExecMode.LOCAL, true);
	}
	
	@Test
	public void testParForRowBlockPartitioningRemoteRemoteSparseNegative() {
		runParForDataPartitioningTest(TEST_NAME3, PDataPartitioner.REMOTE_SPARK, PExecMode.REMOTE_SPARK, true);
	}
	
	@Test
	public void testParForColBlockPartitioningRemoteLocalSparseNegative() {
		runParForDataPartitioningTest(TEST_NAME4, PDataPartitioner.REMOTE_SPARK, PExecMode.LOCAL, true);
	}
	
	@Test
	public void testParForColBlockPartitioningRemoteRemoteSparseNegative() {
		runParForDataPartitioningTest(TEST_NAME4, PDataPartitioner.REMOTE_SPARK, PExecMode.REMOTE_SPARK, true);
	}
	
	private void runParForDataPartitioningTest( String testname, PDataPartitioner partitioner, PExecMode mode, boolean sparse )
	{
		RUNTIME_PLATFORM oldRT = rtplatform;
		boolean oldUseSparkConfig = DMLScript.USE_LOCAL_SPARK_CONFIG;
		boolean oldDynRecompile = CompilerConfig.FLAG_DYN_RECOMPILE;
		
		//run always in spark execution mode
		DMLScript.USE_LOCAL_SPARK_CONFIG = true;
		rtplatform = RUNTIME_PLATFORM.HYBRID_SPARK;
		
		try
		{
			TestConfiguration config = getTestConfiguration(testname);
			loadTestConfiguration(config);
			
			CompilerConfig.FLAG_DYN_RECOMPILE = false;
			
			String HOME = SCRIPT_DIR + TEST_DIR;
			fullDMLScriptName = HOME + testname + ".dml";
			programArgs = new String[]{"-stats", "-args", input("V"), 
				partitioner.name(), mode.name(), output("R") };
			
			fullRScriptName = HOME + testname + ".R";
			rCmd = "Rscript" + " " + fullRScriptName + " " + inputDir() + " " + expectedDir();
	
			//generate input data
			int lrows = testname.equals(TEST_NAME1) || testname.equals(TEST_NAME3) ? rows : cols;
			int lcols = testname.equals(TEST_NAME1) || testname.equals(TEST_NAME3) ? cols : rows;
			double lsparsity = sparse ? sparsity2 : sparsity1;
			double[][] V = getRandomMatrix(lrows, lcols, 0, 1, lsparsity, System.nanoTime());
			writeInputMatrixWithMTD("V", V, true);
	
			//run test
			runTest(true, false, null, -1);
			runRScript(true);
			
			//compare matrices
			HashMap<CellIndex, Double> dmlfile = readDMLMatrixFromHDFS("R");
			HashMap<CellIndex, Double> rfile  = readRMatrixFromFS("Rout");
			TestUtils.compareMatrices(dmlfile, rfile, eps, "DML", "R");
			
			//test for correct plan
			boolean pos = testname.equals(TEST_NAME1) || testname.equals(TEST_NAME2);
			Assert.assertEquals(pos, heavyHittersContainsSubString("ParFor-DPSP") 
					|| heavyHittersContainsSubString("ParFor-DPESP"));
		}
		finally
		{
			rtplatform = oldRT;
			DMLScript.USE_LOCAL_SPARK_CONFIG = oldUseSparkConfig;
			CompilerConfig.FLAG_DYN_RECOMPILE = oldDynRecompile;
		}
	}
}
