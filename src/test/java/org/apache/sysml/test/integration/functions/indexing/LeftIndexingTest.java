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
import org.apache.sysml.hops.LeftIndexingOp;
import org.apache.sysml.hops.LeftIndexingOp.LeftIndexingMethod;
import org.apache.sysml.lops.LopProperties.ExecType;
import org.apache.sysml.runtime.matrix.data.MatrixValue.CellIndex;
import org.apache.sysml.test.integration.AutomatedTestBase;
import org.apache.sysml.test.integration.TestConfiguration;
import org.apache.sysml.test.utils.TestUtils;

public class LeftIndexingTest extends AutomatedTestBase
{
	
	private final static String TEST_DIR = "functions/indexing/";
	private final static String TEST_CLASS_DIR = TEST_DIR + LeftIndexingTest.class.getSimpleName() + "/";
	
	private final static double epsilon=0.0000000001;
	private final static int rows = 1279;
	private final static int cols = 1050;
	private final static int min=0;
	private final static int max=100;
	
	@Override
	public void setUp() {
		addTestConfiguration("LeftIndexingTest", new TestConfiguration(TEST_CLASS_DIR, "LeftIndexingTest", 
				new String[] {"AB", "AC", "AD"}));
	}
	
	@Test
	public void testLeftIndexing() {
		runTestLeftIndexing(ExecType.MR, null);
	}
	
	@Test
	public void testMapLeftIndexingSP() {
		runTestLeftIndexing(ExecType.SPARK, LeftIndexingMethod.SP_MLEFTINDEX);
	}
	
	@Test
	public void testGeneralLeftIndexingSP() {
		runTestLeftIndexing(ExecType.SPARK, LeftIndexingMethod.SP_GLEFTINDEX);
	}
	
	private void runTestLeftIndexing(ExecType et, LeftIndexingOp.LeftIndexingMethod indexingMethod) {
		
		boolean sparkConfigOld = DMLScript.USE_LOCAL_SPARK_CONFIG;
		RUNTIME_PLATFORM oldRTP = rtplatform;
		TestConfiguration config = getTestConfiguration("LeftIndexingTest");
		try
		{
			if(indexingMethod != null) {
				LeftIndexingOp.FORCED_LEFT_INDEXING = indexingMethod;
			}
			
			if(et == ExecType.SPARK) {
		    	rtplatform = RUNTIME_PLATFORM.SPARK;
		    }
			else {
				// rtplatform = (et==ExecType.MR)? RUNTIME_PLATFORM.HADOOP : RUNTIME_PLATFORM.SINGLE_NODE;
			    rtplatform = RUNTIME_PLATFORM.HYBRID;
			}
			if( rtplatform == RUNTIME_PLATFORM.SPARK )
				DMLScript.USE_LOCAL_SPARK_CONFIG = true;
			
		    
	        config.addVariable("rows", rows);
	        config.addVariable("cols", cols);
	      
	        long rowstart=816, rowend=1229, colstart=967, colend=1009;
	      //  long rowstart=2, rowend=4, colstart=9, colend=10;
	        /*
	        Random rand=new Random(System.currentTimeMillis());
	        rowstart=(long)(rand.nextDouble()*((double)rows))+1;
	        rowend=(long)(rand.nextDouble()*((double)(rows-rowstart+1)))+rowstart;
	        colstart=(long)(rand.nextDouble()*((double)cols))+1;
	        colend=(long)(rand.nextDouble()*((double)(cols-colstart+1)))+colstart;
	        */
	        config.addVariable("rowstart", rowstart);
	        config.addVariable("rowend", rowend);
	        config.addVariable("colstart", colstart);
	        config.addVariable("colend", colend);
			loadTestConfiguration(config);
	        
			/* This is for running the junit test the new way, i.e., construct the arguments directly */
			String LI_HOME = SCRIPT_DIR + TEST_DIR;
			fullDMLScriptName = LI_HOME + "LeftIndexingTest" + ".dml";
			programArgs = new String[]{"-args",  input("A"),
				Long.toString(rows), Long.toString(cols),
				Long.toString(rowstart), Long.toString(rowend),
				Long.toString(colstart), Long.toString(colend),
				output("AB"), output("AC"), output("AD"),
				input("B"), input("C"), input("D"),
				Long.toString(rowend-rowstart+1), 
				Long.toString(colend-colstart+1),
				Long.toString(cols-colstart+1)};
			
			fullRScriptName = LI_HOME + "LeftIndexingTest" + ".R";
			rCmd = "Rscript" + " " + fullRScriptName + " " + 
				inputDir() + " " + rowstart + " " + rowend + " " + colstart + " " + colend + " " + expectedDir();
	
			double sparsity=1.0;//rand.nextDouble(); 
	        double[][] A = getRandomMatrix(rows, cols, min, max, sparsity, System.currentTimeMillis());
	        writeInputMatrix("A", A, true);
	        
	        sparsity=0.1;//rand.nextDouble();
	        double[][] B = getRandomMatrix((int)(rowend-rowstart+1), (int)(colend-colstart+1), min, max, sparsity, System.currentTimeMillis());
	        writeInputMatrix("B", B, true);
	        
	        sparsity=0.5;//rand.nextDouble();
	        double[][] C = getRandomMatrix((int)(rowend), (int)(cols-colstart+1), min, max, sparsity, System.currentTimeMillis());
	        writeInputMatrix("C", C, true);
	        
	        sparsity=0.01;//rand.nextDouble();
	        double[][] D = getRandomMatrix(rows, (int)(colend-colstart+1), min, max, sparsity, System.currentTimeMillis());
	        writeInputMatrix("D", D, true);
	
			/*
			 * Expected number of jobs:
			 * Reblock - 1 job 
			 * While loop iteration - 10 jobs
			 * Final output write - 1 job
			 */
	        //boolean exceptionExpected = false;
			//int expectedNumberOfJobs = 12;
			//runTest(exceptionExpected, null, expectedNumberOfJobs);
	        boolean exceptionExpected = false;
			int expectedNumberOfJobs = -1;
			runTest(true, exceptionExpected, null, expectedNumberOfJobs);
		}
		finally
		{
			rtplatform = oldRTP;
			DMLScript.USE_LOCAL_SPARK_CONFIG = sparkConfigOld;
			LeftIndexingOp.FORCED_LEFT_INDEXING = null;
		}
		
		runRScript(true);
		//disableOutAndExpectedDeletion();
	
		for(String file: config.getOutputFiles())
		{
			HashMap<CellIndex, Double> dmlfile = readDMLMatrixFromHDFS(file);
			HashMap<CellIndex, Double> rfile = readRMatrixFromFS(file);
		//	System.out.println(file+"-DML: "+dmlfile);
		//	System.out.println(file+"-R: "+rfile);
			TestUtils.compareMatrices(dmlfile, rfile, epsilon, file+"-DML", file+"-R");
		}
	}
}

