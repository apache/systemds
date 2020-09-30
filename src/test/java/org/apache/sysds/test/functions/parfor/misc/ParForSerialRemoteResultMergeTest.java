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

package org.apache.sysds.test.functions.parfor.misc;

import java.util.HashMap;

import org.junit.Assert;
import org.junit.Test;
import org.apache.sysds.api.DMLScript;
import org.apache.sysds.common.Types.ExecMode;
import org.apache.sysds.runtime.matrix.data.MatrixValue.CellIndex;
import org.apache.sysds.test.AutomatedTestBase;
import org.apache.sysds.test.TestConfiguration;
import org.apache.sysds.test.TestUtils;
import org.apache.sysds.utils.Statistics;

public class ParForSerialRemoteResultMergeTest extends AutomatedTestBase 
{
	private final static String TEST_NAME1 = "parfor_pr_resultmerge1a"; //MR w/o compare
	private final static String TEST_NAME2 = "parfor_pr_resultmerge1b"; //MR w/ compare
	private final static String TEST_NAME3 = "parfor_pr_resultmerge1c"; //Spark w/o compare
	private final static String TEST_NAME4 = "parfor_pr_resultmerge1d"; //Spark w/ compare
	
	private final static String TEST_DIR = "functions/parfor/";
	private final static String TEST_CLASS_DIR = TEST_DIR + ParForSerialRemoteResultMergeTest.class.getSimpleName() + "/";
	private final static double eps = 1e-10;
	
	private final static int rows = 1100;
	private final static int cols = 70;
	
	private final static double sparsity1 = 0.7;
	private final static double sparsity2 = 0.1d;
	
	
	@Override
	public void setUp() {
		addTestConfiguration(TEST_NAME1,
			new TestConfiguration(TEST_CLASS_DIR, TEST_NAME1, new String[] { "R" }) );
	
		addTestConfiguration(TEST_NAME2,
			new TestConfiguration(TEST_CLASS_DIR, TEST_NAME2, new String[] { "R" }) );
		
		addTestConfiguration(TEST_NAME3,
			new TestConfiguration(TEST_CLASS_DIR, TEST_NAME3, new String[] { "R" }) );
	
		addTestConfiguration(TEST_NAME4,
			new TestConfiguration(TEST_CLASS_DIR, TEST_NAME4, new String[] { "R" }) );
	}

	@Test
	public void testSingleResultMergeDenseCP() {
		runParallelRemoteResultMerge(TEST_NAME1, false);
	}
	
	@Test
	public void testSingleResultMergeSparseCP() {
		runParallelRemoteResultMerge(TEST_NAME1, true);
	}
	
	@Test
	public void testSingleResultMergeCompareDenseCP() {
		runParallelRemoteResultMerge(TEST_NAME2, false);
	}
	
	@Test
	public void testSingleResultMergeCompareSparseCP() {
		runParallelRemoteResultMerge(TEST_NAME2, true);
	}
	
	@Test
	public void testSingleResultMergeDenseSP() {
		runParallelRemoteResultMerge(TEST_NAME3, false);
	}
	
	@Test
	public void testSingleResultMergeSparseSP() {
		runParallelRemoteResultMerge(TEST_NAME3, true);
	}
	
	@Test
	public void testSingleResultMergeCompareDenseSP() {
		runParallelRemoteResultMerge(TEST_NAME4, false);
	}
	
	@Test
	public void testSingleResultMergeCompareSparseSP() {
		runParallelRemoteResultMerge(TEST_NAME4, true);
	}
	
	private void runParallelRemoteResultMerge( String test_name, boolean sparse )
	{
		ExecMode oldRT = rtplatform;
		boolean oldUseSparkConfig = DMLScript.USE_LOCAL_SPARK_CONFIG;
		
		if( test_name.equals(TEST_NAME3) || test_name.equals(TEST_NAME4)  )
			DMLScript.USE_LOCAL_SPARK_CONFIG = true;
		rtplatform = ExecMode.HYBRID;
		
		//inst exec type, influenced via rows
		String TEST_NAME = test_name;
		
		try
		{
			//script
			TestConfiguration config = getTestConfiguration(TEST_NAME);
			config.addVariable("rows", rows);
			config.addVariable("cols", cols);
			loadTestConfiguration(config);
			
			String HOME = SCRIPT_DIR + TEST_DIR;
			fullDMLScriptName = HOME + TEST_NAME + ".dml";
			programArgs = new String[]{"-explain", "-args", input("V"), 
				Integer.toString(rows), Integer.toString(cols), output("R") };
			
			fullRScriptName = HOME + TEST_NAME + ".R";
			rCmd = "Rscript" + " " + fullRScriptName + " " + inputDir() + " " + expectedDir();
	
			long seed = System.nanoTime();
			double sparsity = sparse ? sparsity2 : sparsity1;
			double[][] V = getRandomMatrix(rows, cols, 0, 1, sparsity, seed);
			writeInputMatrix("V", V, true);
	
			boolean exceptionExpected = false;
			runTest(true, exceptionExpected, null, -1);
			runRScript(true);
			
			//compare num Spark jobs
			int expectedJobs = ( test_name.equals(TEST_NAME3) || test_name.equals(TEST_NAME4)  ) ? 2 : 0; 
			Assert.assertEquals("Unexpected number of executed Spark jobs.", expectedJobs, Statistics.getNoOfExecutedSPInst());
			
			//compare matrices
			HashMap<CellIndex, Double> dmlfile = readDMLMatrixFromHDFS("R");
			HashMap<CellIndex, Double> rfile  = readRMatrixFromFS("Rout");
			TestUtils.compareMatrices(dmlfile, rfile, eps, "DML", "R");
		}
		finally {
			rtplatform = oldRT;
			DMLScript.USE_LOCAL_SPARK_CONFIG = oldUseSparkConfig;
		}
	}
}