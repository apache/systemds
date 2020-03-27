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

package org.apache.sysds.test.functions.parfor;

import java.util.HashMap;

import org.junit.AfterClass;
import org.junit.Assert;
import org.junit.BeforeClass;
import org.junit.Test;
import org.apache.sysds.common.Types.ExecMode;
import org.apache.sysds.hops.OptimizerUtils;
import org.apache.sysds.lops.LopProperties.ExecType;
import org.apache.sysds.runtime.controlprogram.ParForProgramBlock;
import org.apache.sysds.runtime.controlprogram.parfor.stat.InfrastructureAnalyzer;
import org.apache.sysds.runtime.matrix.data.MatrixValue.CellIndex;
import org.apache.sysds.test.AutomatedTestBase;
import org.apache.sysds.test.TestConfiguration;
import org.apache.sysds.test.TestUtils;
import org.apache.sysds.utils.Statistics;

public class ParForRepeatedOptimizationTest extends AutomatedTestBase 
{
	private final static String TEST_NAME1 = "parfor_repeatedopt1";
	private final static String TEST_NAME2 = "parfor_repeatedopt2";
	private final static String TEST_NAME3 = "parfor_repeatedopt3";
	private final static String TEST_DIR = "functions/parfor/";
	private final static String TEST_CLASS_DIR = TEST_DIR + ParForRepeatedOptimizationTest.class.getSimpleName() + "/";
	private final static double eps = 1e-8;
	
	private final static int rows = 1000000;
	private final static int cols = 10;
	private final static double sparsity = 0.7;
	
	@Override
	public void setUp() 
	{
		addTestConfiguration( TEST_NAME1, new TestConfiguration(TEST_CLASS_DIR, TEST_NAME1, new String[]{"R"}) ); 
		addTestConfiguration( TEST_NAME2, new TestConfiguration(TEST_CLASS_DIR, TEST_NAME2, new String[]{"R"}) ); 
		addTestConfiguration( TEST_NAME3, new TestConfiguration(TEST_CLASS_DIR, TEST_NAME3, new String[]{"R"}) ); 
		
		if (TEST_CACHE_ENABLED) {
			setOutAndExpectedDeletionDisabled(true);
		}
	}
	
	@BeforeClass
	public static void init() {
		TestUtils.clearDirectory(TEST_DATA_DIR + TEST_CLASS_DIR);
	}

	@AfterClass
	public static void cleanUp() {
		if (TEST_CACHE_ENABLED) {
			TestUtils.clearDirectory(TEST_DATA_DIR + TEST_CLASS_DIR);
		}
	}

	@Test
	public void testParForRepeatedOptNoReuseNoUpdateCP() {
		int numExpectedMRJobs = 1+8; //reblock, 3*partition, 4*checkpoints, 1
		runParForRepeatedOptTest( false, false, false, ExecType.CP, numExpectedMRJobs );
	}
	
	@Test
	public void testParForRepeatedOptNoReuseUpdateCP() {
		int numExpectedMRJobs = 1+3+6; //reblock, 3*partition, 4*checkpoints, 2
 		runParForRepeatedOptTest( false, true, false, ExecType.CP, numExpectedMRJobs );
	}
	
	@Test
	public void testParForRepeatedOptNoReuseChangedDimCP() {
		int numExpectedMRJobs = 1+3+7; //reblock, 3*partition, 4*checkpoints, 3
 		runParForRepeatedOptTest( false, false, true, ExecType.CP, numExpectedMRJobs );
	}
	
	@Test
	public void testParForRepeatedOptReuseNoUpdateCP() {
		int numExpectedMRJobs = 1+1 + 5; //reblock, partition, ?
		runParForRepeatedOptTest( true, false, false, ExecType.CP, numExpectedMRJobs );
	}
	
	@Test
	public void testParForRepeatedOptReuseUpdateCP() {
		int numExpectedMRJobs = 1+3+6; //reblock, 3*partition, 4*checkpoint, 2
		runParForRepeatedOptTest( true, true, false, ExecType.CP, numExpectedMRJobs );
	}
	
	@Test
	public void testParForRepeatedOptReuseChangedDimCP() {
		int numExpectedMRJobs = 1+3+7; //reblock, 3*partition, 4*checkpoints, 3
		runParForRepeatedOptTest( true, false, true, ExecType.CP, numExpectedMRJobs );
	}
	
	
	/**
	 * update, refers to changing data
	 * changed dim, refers to changing dimensions and changing parfor predicate
	 * 
	 * 
	 * @param outer execution mode of outer parfor loop
	 * @param inner execution mode of inner parfor loop
	 * @param instType execution mode of instructions
	 */
	private void runParForRepeatedOptTest( boolean reusePartitionedData, boolean update, boolean changedDim, ExecType et, int numExpectedMR )
	{
		ExecMode platformOld = rtplatform;
		double memfactorOld = OptimizerUtils.MEM_UTIL_FACTOR;
		boolean reuseOld = ParForProgramBlock.ALLOW_REUSE_PARTITION_VARS;
		
		String TEST_NAME = update ? TEST_NAME2 : ( changedDim ? TEST_NAME3 : TEST_NAME1);
		
		TestConfiguration config = getTestConfiguration(TEST_NAME);
		config.addVariable("rows", rows);
		config.addVariable("cols", cols);
		
		String TEST_CACHE_DIR = "";
		if (TEST_CACHE_ENABLED)
		{
			TEST_CACHE_DIR = TEST_NAME +  "/";
		}
		
		loadTestConfiguration(config, TEST_CACHE_DIR);
		
		try
		{
			rtplatform = ExecMode.HYBRID;
			OptimizerUtils.MEM_UTIL_FACTOR = computeMemoryUtilFactor( 70 ); //force partitioning
			ParForProgramBlock.ALLOW_REUSE_PARTITION_VARS = reusePartitionedData;
			
			String HOME = SCRIPT_DIR + TEST_DIR;
			fullDMLScriptName = HOME + TEST_NAME + ".dml";
			programArgs = new String[]{"-stats","-args", input("V"), 
				Integer.toString(rows), Integer.toString(cols),
				output("R"),
				Integer.toString((update||changedDim)?1:0)};
			
			fullRScriptName = HOME + TEST_NAME + ".R";
			rCmd = "Rscript" + " " + fullRScriptName + " " + 
				inputDir() + " " + expectedDir() + " " + Integer.toString((update||changedDim)?1:0);
	
			double[][] V = getRandomMatrix(rows, cols, 0, 1, sparsity, 7);
			writeInputMatrix("V", V, true);
	
			runTest(true, false, null, -1);
			runRScript(true);
			
			Assert.assertEquals("Unexpected number of executed Spark jobs.",
				numExpectedMR, Statistics.getNoOfExecutedSPInst());
			
			//compare matrices
			HashMap<CellIndex, Double> dmlfile = readDMLMatrixFromHDFS("R");
			HashMap<CellIndex, Double> rfile  = readRMatrixFromFS("R");
			TestUtils.compareMatrices(dmlfile, rfile, eps, "DML", "R");
		}
		finally {
			//reset optimizer flags to pre-test configuration
			rtplatform = platformOld;
			OptimizerUtils.MEM_UTIL_FACTOR = memfactorOld;
			ParForProgramBlock.ALLOW_REUSE_PARTITION_VARS = reuseOld;
		}
	}
	
	private static double computeMemoryUtilFactor( int mb ) {
		return Math.min(1, ((double)1024*1024*mb)/InfrastructureAnalyzer.getLocalMaxMemory());
	}
}
