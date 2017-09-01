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
import org.apache.sysml.lops.LopProperties.ExecType;
import org.apache.sysml.runtime.controlprogram.parfor.stat.InfrastructureAnalyzer;
import org.apache.sysml.runtime.matrix.data.MatrixValue.CellIndex;
import org.apache.sysml.test.integration.AutomatedTestBase;
import org.apache.sysml.test.integration.TestConfiguration;
import org.apache.sysml.test.utils.TestUtils;

public class ParForDataPartitionExecuteTest extends AutomatedTestBase 
{
	private final static String TEST_NAME = "DataPartitionExecute";
	private final static String TEST_DIR = "functions/parfor/";
	private final static String TEST_CLASS_DIR = TEST_DIR + ParForDataPartitionExecuteTest.class.getSimpleName() + "/";
	private final static double eps = 1e-10;
	
	private final static int dim1 = 2001;
	private final static int dim2 = 101;
	private final static double sparsity1 = 0.7;
	private final static double sparsity2 = 0.3d;
	
	@Override
	public void setUp() {
		addTestConfiguration(TEST_NAME, new TestConfiguration(TEST_CLASS_DIR, TEST_NAME, new String[]{"R"}));
	}
	
	@Test
	public void testFusedDataPartitionExecuteRowDenseMR() {
		runFusedDataPartitionExecuteTest(false, true, ExecType.MR);
	}
	
	@Test
	public void testFusedDataPartitionExecuteColDenseMR() {
		runFusedDataPartitionExecuteTest(false, false, ExecType.MR);
	}
	
	@Test
	public void testFusedDataPartitionExecuteRowSparseMR() {
		runFusedDataPartitionExecuteTest(true, true, ExecType.MR);
	}
	
	@Test
	public void testFusedDataPartitionExecuteColSparseMR() {
		runFusedDataPartitionExecuteTest(true, false, ExecType.MR);
	}
	
	@Test
	public void testFusedDataPartitionExecuteRowDenseSpark() {
		runFusedDataPartitionExecuteTest(false, true, ExecType.SPARK);
	}
	
	@Test
	public void testFusedDataPartitionExecuteColDenseSpark() {
		runFusedDataPartitionExecuteTest(false, false, ExecType.SPARK);
	}
	
	@Test
	public void testFusedDataPartitionExecuteRowSparseSpark() {
		runFusedDataPartitionExecuteTest(true, true, ExecType.SPARK);
	}
	
	@Test
	public void testFusedDataPartitionExecuteColSparseSpark() {
		runFusedDataPartitionExecuteTest(true, false, ExecType.SPARK);
	}
	
	private void runFusedDataPartitionExecuteTest(boolean sparse, boolean row, ExecType et)
	{
		RUNTIME_PLATFORM platformOld = rtplatform;
		switch( et ){
			case MR: rtplatform = RUNTIME_PLATFORM.HYBRID; break;
			case SPARK: rtplatform = RUNTIME_PLATFORM.HYBRID_SPARK; break;
			default: throw new RuntimeException("Unsupported exec type: "+et.name());
		}
		boolean sparkConfigOld = DMLScript.USE_LOCAL_SPARK_CONFIG;
		if( et == ExecType.SPARK )
			DMLScript.USE_LOCAL_SPARK_CONFIG = true;
		
		//modify memory budget to trigger fused datapartition-execute
		long oldmem = InfrastructureAnalyzer.getLocalMaxMemory();
		InfrastructureAnalyzer.setLocalMaxMemory(1*1024*1024); //1MB
		
		try
		{
			int rows = row ? dim2 : dim1;
			int cols = row ? dim1 : dim2;
			
			TestConfiguration config = getTestConfiguration(TEST_NAME);
			config.addVariable("rows", rows);
			config.addVariable("cols", cols);
			loadTestConfiguration(config);
			
			String HOME = SCRIPT_DIR + TEST_DIR;
			fullDMLScriptName = HOME + TEST_NAME + ".dml";
			programArgs = new String[]{"-stats", "-args", input("X"), 
				String.valueOf(et == ExecType.SPARK).toUpperCase(),
				String.valueOf(row).toUpperCase(), output("R") };
			
			fullRScriptName = HOME + TEST_NAME + ".R";
			rCmd = "Rscript" + " " + fullRScriptName + " " + inputDir() 
				+ " " + String.valueOf(row).toUpperCase() + " " + expectedDir();
			
			//generate input data
			double sparsity = sparse ? sparsity2 : sparsity1;
			double[][] X = getRandomMatrix(rows, cols, 0, 1, sparsity, 7);
			writeInputMatrixWithMTD("X", X, true);
			
			//run test case
			runTest(true, false, null, -1);
			runRScript(true);
			
			//compare matrices
			HashMap<CellIndex, Double> dmlfile = readDMLMatrixFromHDFS("R");
			HashMap<CellIndex, Double> rfile  = readRMatrixFromFS("R");
			TestUtils.compareMatrices(dmlfile, rfile, eps, "DML", "R");
			
			//check for compiled datapartition-execute
			Assert.assertTrue(heavyHittersContainsSubString(
				(et == ExecType.SPARK) ? "ParFor-DPESP" : "MR-Job_ParFor-DPEMR"));
		}
		finally {
			rtplatform = platformOld;
			DMLScript.USE_LOCAL_SPARK_CONFIG = sparkConfigOld;
			InfrastructureAnalyzer.setLocalMaxMemory(oldmem); //1MB
		}
	}
}
