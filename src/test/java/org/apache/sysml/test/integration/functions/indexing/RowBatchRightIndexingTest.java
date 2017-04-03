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

package org.apache.sysml.test.integration.functions.indexing;

import java.util.HashMap;
import org.junit.Test;

import org.apache.sysml.api.DMLScript;
import org.apache.sysml.api.DMLScript.RUNTIME_PLATFORM;
import org.apache.sysml.lops.LopProperties.ExecType;
import org.apache.sysml.runtime.matrix.data.MatrixValue.CellIndex;
import org.apache.sysml.test.integration.AutomatedTestBase;
import org.apache.sysml.test.integration.TestConfiguration;
import org.apache.sysml.test.utils.TestUtils;

public class RowBatchRightIndexingTest extends AutomatedTestBase
{	
	private final static String TEST_NAME = "RowBatchIndexingTest";
	private final static String TEST_DIR = "functions/indexing/";
	private final static String TEST_CLASS_DIR = TEST_DIR + RowBatchRightIndexingTest.class.getSimpleName() + "/";
	
	private final static double epsilon=0.0000000001;
	private final static int rows = 1500; //multiple of 500
	private final static int cols = 1050;
	
	private final static double sparsity1 = 0.5;
	private final static double sparsity2 = 0.01;
	
	
	@Override
	public void setUp() {
		TestUtils.clearAssertionInformation();
		addTestConfiguration(TEST_NAME, new TestConfiguration(TEST_CLASS_DIR, TEST_NAME, new String[] {"B"}));
	}
	
	@Test
	public void testRightIndexingDenseCP() {
		runRightIndexingTest(ExecType.CP, false);
	}
	
	@Test
	public void testRightIndexingDenseSP() {
		runRightIndexingTest(ExecType.SPARK, false);
	}
	
	@Test
	public void testRightIndexingDenseMR() {
		runRightIndexingTest(ExecType.MR, false);
	}
	
	@Test
	public void testRightIndexingSparseCP() {
		runRightIndexingTest(ExecType.CP, true);
	}
	
	@Test
	public void testRightIndexingSparseSP() {
		runRightIndexingTest(ExecType.SPARK, true);
	}
	
	@Test
	public void testRightIndexingSparseMR() {
		runRightIndexingTest(ExecType.MR, true);
	}
	
	/**
	 * 
	 * @param et
	 * @param sparse
	 */
	public void runRightIndexingTest( ExecType et, boolean sparse ) 
	{
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
		    TestConfiguration config = getTestConfiguration(TEST_NAME);
			loadTestConfiguration(config);

		    double sparsity = sparse ? sparsity2 : sparsity1;
		    
	        config.addVariable("rows", rows);
	        config.addVariable("cols", cols);
	        
	        String RI_HOME = SCRIPT_DIR + TEST_DIR;
			fullDMLScriptName = RI_HOME + TEST_NAME + ".dml";
			programArgs = new String[]{"-args",  input("A"), output("B") };
			
			fullRScriptName = RI_HOME + TEST_NAME + ".R";
			rCmd = "Rscript" + " " + fullRScriptName + " " + 
				inputDir() + " " + expectedDir();
	
			double[][] A = getRandomMatrix(rows, cols, 0, 1, sparsity, 23);
	        writeInputMatrixWithMTD("A", A, true);
	        
	        //run tests
	        runTest(true, false, null, -1);
			runRScript(true);
			
			//compare output aggregate
			HashMap<CellIndex, Double> dmlfile = readDMLMatrixFromHDFS("B");
			HashMap<CellIndex, Double> rfile = readRMatrixFromFS("B");
			TestUtils.compareMatrices(dmlfile, rfile, epsilon, "DML", "R");
		}
		finally {
			rtplatform = platformOld;
			DMLScript.USE_LOCAL_SPARK_CONFIG = sparkConfigOld;
		}
	}
}
