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
import java.util.Random;

import org.junit.Test;

import org.apache.sysml.api.DMLScript;
import org.apache.sysml.api.DMLScript.RUNTIME_PLATFORM;
import org.apache.sysml.lops.LopProperties.ExecType;
import org.apache.sysml.runtime.matrix.data.MatrixValue.CellIndex;
import org.apache.sysml.test.integration.AutomatedTestBase;
import org.apache.sysml.test.integration.TestConfiguration;
import org.apache.sysml.test.utils.TestUtils;



public class RightIndexingVectorTest extends AutomatedTestBase
{
	
	private final static String TEST_NAME = "RightIndexingVectorTest";
	private final static String TEST_DIR = "functions/indexing/";
	private final static String TEST_CLASS_DIR = TEST_DIR + RightIndexingVectorTest.class.getSimpleName() + "/";
	
	private final static double epsilon=0.0000000001;
	private final static int rows = 2279;
	private final static int cols = 1050;
	private final static int min=0;
	private final static int max=100;
	
	@Override
	public void setUp() {
		addTestConfiguration(TEST_NAME, new TestConfiguration(TEST_CLASS_DIR, TEST_NAME, 
				new String[] {"B", "C", "D"}));
	}
	
	@Test
	public void testRightIndexingCP() 
	{
		runRightIndexingTest(ExecType.CP);
	}
	
	@Test
	public void testRightIndexingSP() 
	{
		runRightIndexingTest(ExecType.SPARK);
	}
	
	@Test
	public void testRightIndexingMR() 
	{
		runRightIndexingTest(ExecType.MR);
	}
	
	public void runRightIndexingTest( ExecType et ) 
	{
		RUNTIME_PLATFORM oldRTP = rtplatform;
				
		boolean sparkConfigOld = DMLScript.USE_LOCAL_SPARK_CONFIG;
		
		try
		{
		    TestConfiguration config = getTestConfiguration(TEST_NAME);
		    if(et == ExecType.SPARK) {
		    	rtplatform = RUNTIME_PLATFORM.SPARK;
		    }
		    else {
		    	rtplatform = (et==ExecType.MR)? RUNTIME_PLATFORM.HADOOP : RUNTIME_PLATFORM.SINGLE_NODE;
		    }
		    if( rtplatform == RUNTIME_PLATFORM.SPARK )
				DMLScript.USE_LOCAL_SPARK_CONFIG = true;
		    
		    config.addVariable("rows", rows);
	        config.addVariable("cols", cols);
	
			loadTestConfiguration(config);
	        
	        String RI_HOME = SCRIPT_DIR + TEST_DIR;
			fullDMLScriptName = RI_HOME + TEST_NAME + ".dml";
			programArgs = new String[]{"-args",  input("A"),
				Long.toString(rows), Long.toString(cols),
				output("B"), output("C"), output("D") };
			
			fullRScriptName = RI_HOME + TEST_NAME + ".R";
			rCmd = "Rscript" + " " + fullRScriptName + " " + inputDir() + " " + expectedDir();
			
			Random rand=new Random(System.currentTimeMillis());
	        double sparsity=rand.nextDouble();
	        double[][] A = getRandomMatrix(rows, cols, min, max, sparsity, System.currentTimeMillis());
	        writeInputMatrix("A", A, true);
	        
	        boolean exceptionExpected = false;
			int expectedNumberOfJobs = -1;
			runTest(true, exceptionExpected, null, expectedNumberOfJobs);
			
			runRScript(true);
			//disableOutAndExpectedDeletion();
		
			for(String file: config.getOutputFiles())
			{
				HashMap<CellIndex, Double> dmlfile = readDMLMatrixFromHDFS(file);
				HashMap<CellIndex, Double> rfile = readRMatrixFromFS(file);
				TestUtils.compareMatrices(dmlfile, rfile, epsilon, file+"-DML", file+"-R");
			}
		}
		finally
		{
			rtplatform = oldRTP;
			DMLScript.USE_LOCAL_SPARK_CONFIG = sparkConfigOld;
		}
	}
}
