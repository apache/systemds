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

package org.apache.sysml.test.integration.functions.reorg;

import java.util.HashMap;

import org.junit.Test;

import org.apache.sysml.api.DMLScript;
import org.apache.sysml.api.DMLScript.RUNTIME_PLATFORM;
import org.apache.sysml.lops.LopProperties.ExecType;
import org.apache.sysml.runtime.matrix.data.MatrixValue.CellIndex;
import org.apache.sysml.test.integration.AutomatedTestBase;
import org.apache.sysml.test.integration.TestConfiguration;
import org.apache.sysml.test.utils.TestUtils;

public class VectorReshapeTest extends AutomatedTestBase 
{
	private final static String TEST_NAME = "VectorReshape";
	private final static String TEST_DIR = "functions/reorg/";
	private static final String TEST_CLASS_DIR = TEST_DIR + VectorReshapeTest.class.getSimpleName() + "/";

	private final static int rows1 = 1;
	private final static int cols1 = 802816;
	private final static int rows2 = 64;
	private final static int cols2 = 12544;
	
	private final static double sparsityDense = 0.9;
	private final static double sparsitySparse = 0.1;
	
	@Override
	public void setUp() {
		addTestConfiguration(TEST_NAME, new TestConfiguration(
			TEST_CLASS_DIR, TEST_NAME, new String[] { "R" }) );
	}
	
	@Test
	public void testVectorReshapeDenseCP() {
		runVectorReshape(false, ExecType.CP);
	}
	
	@Test
	public void testVectorReshapeSparseCP() {
		runVectorReshape(true, ExecType.CP);
	}
	
	@Test
	public void testVectorReshapeDenseMR() {
		runVectorReshape(false, ExecType.MR);
	}
	
	@Test
	public void testVectorReshapeSparseMR() {
		runVectorReshape(true, ExecType.MR);
	}
	
	@Test
	public void testVectorReshapeDenseSpark() {
		runVectorReshape(false, ExecType.SPARK);
	}
	
	@Test
	public void testVectorReshapeSparseSpark() {
		runVectorReshape(true, ExecType.SPARK);
	}
	
	private void runVectorReshape(boolean sparse, ExecType et)
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
		
		try
		{
			//register test configuration
			TestConfiguration config = getTestConfiguration(TEST_NAME);
			loadTestConfiguration(config);
			
			String HOME = SCRIPT_DIR + TEST_DIR;
			fullDMLScriptName = HOME + TEST_NAME + ".dml";
			programArgs = new String[]{"-args", input("X"), 
				String.valueOf(rows2), String.valueOf(cols2), output("R") };
			
			fullRScriptName = HOME + TEST_NAME + ".R";
			rCmd = "Rscript" + " " + fullRScriptName + " " + 
		           inputDir() + " " + rows2 + " " + cols2 + " " + expectedDir();
			
			double sparsity = sparse ? sparsitySparse : sparsityDense;
			double[][] X = getRandomMatrix(rows1, cols1, 0, 1, sparsity, 7);
			writeInputMatrixWithMTD("X", X, true); 
			
			runTest(true, false, null, -1);
			runRScript(true); 
			
			//compare matrices 
			HashMap<CellIndex, Double> dmlfile = readDMLMatrixFromHDFS("R");
			HashMap<CellIndex, Double> rfile  = readRMatrixFromFS("R");
			TestUtils.compareMatrices(dmlfile, rfile, 10e-10, "Stat-DML", "Stat-R");
		}
		finally {
			//reset platform for additional tests
			rtplatform = platformOld;
			DMLScript.USE_LOCAL_SPARK_CONFIG = sparkConfigOld;
		}
	}	
}
