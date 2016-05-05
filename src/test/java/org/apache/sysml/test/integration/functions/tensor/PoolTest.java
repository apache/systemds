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
package org.apache.sysml.test.integration.functions.tensor;

import java.util.HashMap;

import org.apache.sysml.api.DMLScript;
import org.apache.sysml.api.DMLScript.RUNTIME_PLATFORM;
import org.apache.sysml.lops.LopProperties.ExecType;
import org.apache.sysml.runtime.matrix.data.MatrixValue.CellIndex;
import org.apache.sysml.test.integration.AutomatedTestBase;
import org.apache.sysml.test.integration.TestConfiguration;
import org.apache.sysml.test.utils.TestUtils;
import org.junit.Test;

public class PoolTest extends AutomatedTestBase
{
	
	private final static String TEST_NAME = "PoolTest";
	private final static String TEST_DIR = "functions/tensor/";
	private final static String TEST_CLASS_DIR = TEST_DIR + Conv2DTest.class.getSimpleName() + "/";
	private final static double epsilon=0.0000000001;
	
	@Override
	public void setUp() {
		addTestConfiguration(TEST_NAME, new TestConfiguration(TEST_CLASS_DIR, TEST_NAME, 
				new String[] {"B"}));
	}
	
	HashMap<CellIndex, Double> bHM = new HashMap<CellIndex, Double>();
	
	@Test
	public void testMaxPool2DDense1() 
	{
		int numImg = 1; int imgSize = 6; int numChannels = 1;  int stride = 2; int pad = 0; int poolSize1 = 2; int poolSize2 = 2;
		fillMaxPoolTest1HM();
		runConv2DTest(ExecType.CP, imgSize, numImg, numChannels, stride, pad, poolSize1, poolSize2, "max");
	}
	
	@Test
	public void testMaxPool2DDense2() 
	{
		int numImg = 2; int imgSize = 6; int numChannels = 1;  int stride = 1; int pad = 0; int poolSize1 = 2; int poolSize2 = 2;
		fillMaxPoolTest2HM();
		runConv2DTest(ExecType.CP, imgSize, numImg, numChannels, stride, pad, poolSize1, poolSize2, "max");
	}
	
	
	@Test
	public void testMaxPool2DDense3() 
	{
		int numImg = 3; int imgSize = 7; int numChannels = 2;  int stride = 2; int pad = 0; int poolSize1 = 3; int poolSize2 = 3;
		fillMaxPoolTest3HM();
		runConv2DTest(ExecType.CP, imgSize, numImg, numChannels, stride, pad, poolSize1, poolSize2, "max");
	}
	
	@Test
	public void testMaxPool2DDense4() 
	{
		int numImg = 2; int imgSize = 4; int numChannels = 2;  int stride = 1; int pad = 0; int poolSize1 = 3; int poolSize2 = 3;
		fillMaxPoolTest4HM();
		runConv2DTest(ExecType.CP, imgSize, numImg, numChannels, stride, pad, poolSize1, poolSize2, "max");
	}
	
	/**
	 * 
	 * @param et
	 * @param sparse
	 */
	public void runConv2DTest( ExecType et, int imgSize, int numImg, int numChannels, int stride, 
			int pad, int poolSize1, int poolSize2, String poolMode) 
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
			
			loadTestConfiguration(config);
	        
			/* This is for running the junit test the new way, i.e., construct the arguments directly */
			String RI_HOME = SCRIPT_DIR + TEST_DIR;
			fullDMLScriptName = RI_HOME + TEST_NAME + ".dml";
			
			programArgs = new String[]{"-explain", "-args",  "" + imgSize, "" + numImg, 
					"" + numChannels, "" + poolSize1, "" + poolSize2, 
					"" + stride, "" + pad, poolMode, 
					output("B")};
			        
			boolean exceptionExpected = false;
			int expectedNumberOfJobs = -1;
			runTest(true, exceptionExpected, null, expectedNumberOfJobs);
			
			// Uncomment this after fixing following R error:
//			Error in matrix(0, C * Hf * Wf, Hout * Wout, byrow = TRUE) : 
//				  invalid 'nrow' value (too large or NA)
//				Calls: max_pool -> im2col -> matrix
			// ------------------------------------------
//			fullRScriptName = RI_HOME + TEST_NAME + ".R";
//			rCmd = "Rscript" + " " + fullRScriptName + " " + imgSize + " " + numImg + 
//					" " + numChannels + " " + poolSize1 + 
//					" " + poolSize2 + " " + stride + " " + pad + " " + expectedDir(); 
//			
//			// Run comparison R script
//			runRScript(true);
//			HashMap<CellIndex, Double> bHM = readRMatrixFromFS("B");
			
			HashMap<CellIndex, Double> dmlfile = readDMLMatrixFromHDFS("B");
			TestUtils.compareMatrices(dmlfile, bHM, epsilon, "B-DML", "NumPy");
			
		}
		finally
		{
			rtplatform = oldRTP;
			DMLScript.USE_LOCAL_SPARK_CONFIG = sparkConfigOld;
			bHM.clear();
		}
	}
	
	private void fillMaxPoolTest1HM() {
		bHM.put(new CellIndex(1, 1), 8.0); bHM.put(new CellIndex(1, 2), 10.0); bHM.put(new CellIndex(1, 3), 12.0); bHM.put(new CellIndex(1, 4), 20.0); 
		bHM.put(new CellIndex(1, 5), 22.0); bHM.put(new CellIndex(1, 6), 24.0); bHM.put(new CellIndex(1, 7), 32.0); bHM.put(new CellIndex(1, 8), 34.0); 
		bHM.put(new CellIndex(1, 9), 36.0); 
	}
	
	private void fillMaxPoolTest2HM() {
		bHM.put(new CellIndex(1, 1), 8.0); bHM.put(new CellIndex(2, 1), 44.0); bHM.put(new CellIndex(1, 2), 9.0); bHM.put(new CellIndex(2, 2), 45.0); 
		bHM.put(new CellIndex(1, 3), 10.0); bHM.put(new CellIndex(2, 3), 46.0); bHM.put(new CellIndex(1, 4), 11.0); bHM.put(new CellIndex(2, 4), 47.0); 
		bHM.put(new CellIndex(1, 5), 12.0); bHM.put(new CellIndex(2, 5), 48.0); bHM.put(new CellIndex(1, 6), 14.0); bHM.put(new CellIndex(2, 6), 50.0); 
		bHM.put(new CellIndex(1, 7), 15.0); bHM.put(new CellIndex(2, 7), 51.0); bHM.put(new CellIndex(1, 8), 16.0); bHM.put(new CellIndex(2, 8), 52.0); 
		bHM.put(new CellIndex(1, 9), 17.0); bHM.put(new CellIndex(2, 9), 53.0); bHM.put(new CellIndex(1, 10), 18.0); bHM.put(new CellIndex(2, 10), 54.0); 
		bHM.put(new CellIndex(1, 11), 20.0); bHM.put(new CellIndex(2, 11), 56.0); bHM.put(new CellIndex(1, 12), 21.0); bHM.put(new CellIndex(2, 12), 57.0); 
		bHM.put(new CellIndex(1, 13), 22.0); bHM.put(new CellIndex(2, 13), 58.0); bHM.put(new CellIndex(1, 14), 23.0); bHM.put(new CellIndex(2, 14), 59.0); 
		bHM.put(new CellIndex(1, 15), 24.0); bHM.put(new CellIndex(2, 15), 60.0); bHM.put(new CellIndex(1, 16), 26.0); bHM.put(new CellIndex(2, 16), 62.0); 
		bHM.put(new CellIndex(1, 17), 27.0); bHM.put(new CellIndex(2, 17), 63.0); bHM.put(new CellIndex(1, 18), 28.0); bHM.put(new CellIndex(2, 18), 64.0); 
		bHM.put(new CellIndex(1, 19), 29.0); bHM.put(new CellIndex(2, 19), 65.0); bHM.put(new CellIndex(1, 20), 30.0); bHM.put(new CellIndex(2, 20), 66.0); 
		bHM.put(new CellIndex(1, 21), 32.0); bHM.put(new CellIndex(2, 21), 68.0); bHM.put(new CellIndex(1, 22), 33.0); bHM.put(new CellIndex(2, 22), 69.0); 
		bHM.put(new CellIndex(1, 23), 34.0); bHM.put(new CellIndex(2, 23), 70.0); bHM.put(new CellIndex(1, 24), 35.0); bHM.put(new CellIndex(2, 24), 71.0); 
		bHM.put(new CellIndex(1, 25), 36.0); bHM.put(new CellIndex(2, 25), 72.0);
	}
	
	private void fillMaxPoolTest3HM() {
		bHM.put(new CellIndex(1, 1), 17.0); bHM.put(new CellIndex(1, 2), 19.0); bHM.put(new CellIndex(1, 3), 21.0); bHM.put(new CellIndex(1, 4), 31.0); 
		bHM.put(new CellIndex(1, 5), 33.0); bHM.put(new CellIndex(1, 6), 35.0); bHM.put(new CellIndex(1, 7), 45.0); bHM.put(new CellIndex(1, 8), 47.0); 
		bHM.put(new CellIndex(1, 9), 49.0); bHM.put(new CellIndex(1, 10), 66.0); bHM.put(new CellIndex(1, 11), 68.0); bHM.put(new CellIndex(1, 12), 70.0); 
		bHM.put(new CellIndex(1, 13), 80.0); bHM.put(new CellIndex(1, 14), 82.0); bHM.put(new CellIndex(1, 15), 84.0); bHM.put(new CellIndex(1, 16), 94.0); 
		bHM.put(new CellIndex(1, 17), 96.0); bHM.put(new CellIndex(1, 18), 98.0); bHM.put(new CellIndex(2, 1), 115.0); bHM.put(new CellIndex(2, 2), 117.0); 
		bHM.put(new CellIndex(2, 3), 119.0); bHM.put(new CellIndex(2, 4), 129.0); bHM.put(new CellIndex(2, 5), 131.0); bHM.put(new CellIndex(2, 6), 133.0); 
		bHM.put(new CellIndex(2, 7), 143.0); bHM.put(new CellIndex(2, 8), 145.0); bHM.put(new CellIndex(2, 9), 147.0); bHM.put(new CellIndex(2, 10), 164.0); 
		bHM.put(new CellIndex(2, 11), 166.0); bHM.put(new CellIndex(2, 12), 168.0); bHM.put(new CellIndex(2, 13), 178.0); bHM.put(new CellIndex(2, 14), 180.0); 
		bHM.put(new CellIndex(2, 15), 182.0); bHM.put(new CellIndex(2, 16), 192.0); bHM.put(new CellIndex(2, 17), 194.0); bHM.put(new CellIndex(2, 18), 196.0); 
		bHM.put(new CellIndex(3, 1), 213.0); bHM.put(new CellIndex(3, 2), 215.0); bHM.put(new CellIndex(3, 3), 217.0); bHM.put(new CellIndex(3, 4), 227.0); 
		bHM.put(new CellIndex(3, 5), 229.0); bHM.put(new CellIndex(3, 6), 231.0); bHM.put(new CellIndex(3, 7), 241.0); bHM.put(new CellIndex(3, 8), 243.0); 
		bHM.put(new CellIndex(3, 9), 245.0); bHM.put(new CellIndex(3, 10), 262.0); bHM.put(new CellIndex(3, 11), 264.0); bHM.put(new CellIndex(3, 12), 266.0); 
		bHM.put(new CellIndex(3, 13), 276.0); bHM.put(new CellIndex(3, 14), 278.0); bHM.put(new CellIndex(3, 15), 280.0); bHM.put(new CellIndex(3, 16), 290.0); 
		bHM.put(new CellIndex(3, 17), 292.0); bHM.put(new CellIndex(3, 18), 294.0); 
	}
	
	private void fillMaxPoolTest4HM() {
		bHM.put(new CellIndex(1, 1), 11.0); bHM.put(new CellIndex(1, 2), 12.0); bHM.put(new CellIndex(1, 3), 15.0); bHM.put(new CellIndex(1, 4), 16.0); 
		bHM.put(new CellIndex(1, 5), 27.0); bHM.put(new CellIndex(1, 6), 28.0); bHM.put(new CellIndex(1, 7), 31.0); bHM.put(new CellIndex(1, 8), 32.0); 
		bHM.put(new CellIndex(2, 1), 43.0); bHM.put(new CellIndex(2, 2), 44.0); bHM.put(new CellIndex(2, 3), 47.0); bHM.put(new CellIndex(2, 4), 48.0); 
		bHM.put(new CellIndex(2, 5), 59.0); bHM.put(new CellIndex(2, 6), 60.0); bHM.put(new CellIndex(2, 7), 63.0); bHM.put(new CellIndex(2, 8), 64.0); 
	}
}
