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
import org.apache.sysml.runtime.util.ConvolutionUtils;
import org.apache.sysml.test.integration.AutomatedTestBase;
import org.apache.sysml.test.integration.TestConfiguration;
import org.apache.sysml.test.utils.TestUtils;
import org.junit.Test;

public class PoolBackwardTest extends AutomatedTestBase
{
	
	private final static String TEST_NAME = "PoolBackwardTest";
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
	public void testMaxPool2DBackwardDense1() 
	{
		int numImg = 1; int imgSize = 4; int numChannels = 1;  int stride = 2; int pad = 0; int poolSize1 = 2; int poolSize2 = 2;
		fillMaxPoolTest1HM();
		runConv2DTest(ExecType.CP, imgSize, numImg, numChannels, stride, pad, poolSize1, poolSize2, "max");
	}
	
	@Test
	public void testMaxPool2DBackwardDense2() 
	{
		int numImg = 3; int imgSize = 6; int numChannels = 3;  int stride = 1; int pad = 0; int poolSize1 = 2; int poolSize2 = 2;
		fillMaxPoolTest2HM();
		runConv2DTest(ExecType.CP, imgSize, numImg, numChannels, stride, pad, poolSize1, poolSize2, "max");
	}
	
	@Test
	public void testMaxPool2DBackwardDense3() 
	{
		int numImg = 2; int imgSize = 7; int numChannels = 2;  int stride = 2; int pad = 0; int poolSize1 = 3; int poolSize2 = 3;
		fillMaxPoolTest3HM();
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
			
			long P = ConvolutionUtils.getP(imgSize, poolSize1, stride, pad);
			programArgs = new String[]{"-explain", "-args",  "" + imgSize, "" + numImg, 
					"" + numChannels, "" + poolSize1, "" + poolSize2, 
					"" + stride, "" + pad, poolMode, 
					"" + P, "" + P, 
					output("B")};
			        
			boolean exceptionExpected = false;
			int expectedNumberOfJobs = -1;
			runTest(true, exceptionExpected, null, expectedNumberOfJobs);
			
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
		bHM.put(new CellIndex(1, 1), 0.0); bHM.put(new CellIndex(1, 2), 0.0); bHM.put(new CellIndex(1, 3), 0.0); bHM.put(new CellIndex(1, 4), 0.0); 
		bHM.put(new CellIndex(1, 5), 0.0); bHM.put(new CellIndex(1, 6), 1.0); bHM.put(new CellIndex(1, 7), 0.0); bHM.put(new CellIndex(1, 8), 2.0); 
		bHM.put(new CellIndex(1, 9), 0.0); bHM.put(new CellIndex(1, 10), 0.0); bHM.put(new CellIndex(1, 11), 0.0); bHM.put(new CellIndex(1, 12), 0.0); 
		bHM.put(new CellIndex(1, 13), 0.0); bHM.put(new CellIndex(1, 14), 3.0); bHM.put(new CellIndex(1, 15), 0.0); bHM.put(new CellIndex(1, 16), 4.0);  
	}
	
	private void fillMaxPoolTest2HM() {
		bHM.put(new CellIndex(1, 1), 0.0); bHM.put(new CellIndex(1, 2), 0.0); bHM.put(new CellIndex(1, 3), 0.0); bHM.put(new CellIndex(1, 4), 0.0); 
		bHM.put(new CellIndex(1, 5), 0.0); bHM.put(new CellIndex(1, 6), 0.0); bHM.put(new CellIndex(1, 7), 0.0); bHM.put(new CellIndex(1, 8), 1.0); 
		bHM.put(new CellIndex(1, 9), 2.0); bHM.put(new CellIndex(1, 10), 3.0); bHM.put(new CellIndex(1, 11), 4.0); bHM.put(new CellIndex(1, 12), 5.0); 
		bHM.put(new CellIndex(1, 13), 0.0); bHM.put(new CellIndex(1, 14), 6.0); bHM.put(new CellIndex(1, 15), 7.0); bHM.put(new CellIndex(1, 16), 8.0); 
		bHM.put(new CellIndex(1, 17), 9.0); bHM.put(new CellIndex(1, 18), 10.0); bHM.put(new CellIndex(1, 19), 0.0); bHM.put(new CellIndex(1, 20), 11.0); 
		bHM.put(new CellIndex(1, 21), 12.0); bHM.put(new CellIndex(1, 22), 13.0); bHM.put(new CellIndex(1, 23), 14.0); bHM.put(new CellIndex(1, 24), 15.0); 
		bHM.put(new CellIndex(1, 25), 0.0); bHM.put(new CellIndex(1, 26), 16.0); bHM.put(new CellIndex(1, 27), 17.0); bHM.put(new CellIndex(1, 28), 18.0); 
		bHM.put(new CellIndex(1, 29), 19.0); bHM.put(new CellIndex(1, 30), 20.0); bHM.put(new CellIndex(1, 31), 0.0); bHM.put(new CellIndex(1, 32), 21.0); 
		bHM.put(new CellIndex(1, 33), 22.0); bHM.put(new CellIndex(1, 34), 23.0); bHM.put(new CellIndex(1, 35), 24.0); bHM.put(new CellIndex(1, 36), 25.0); 
		bHM.put(new CellIndex(1, 37), 0.0); bHM.put(new CellIndex(1, 38), 0.0); bHM.put(new CellIndex(1, 39), 0.0); bHM.put(new CellIndex(1, 40), 0.0); 
		bHM.put(new CellIndex(1, 41), 0.0); bHM.put(new CellIndex(1, 42), 0.0); bHM.put(new CellIndex(1, 43), 0.0); bHM.put(new CellIndex(1, 44), 26.0); 
		bHM.put(new CellIndex(1, 45), 27.0); bHM.put(new CellIndex(1, 46), 28.0); bHM.put(new CellIndex(1, 47), 29.0); bHM.put(new CellIndex(1, 48), 30.0); 
		bHM.put(new CellIndex(1, 49), 0.0); bHM.put(new CellIndex(1, 50), 31.0); bHM.put(new CellIndex(1, 51), 32.0); bHM.put(new CellIndex(1, 52), 33.0); 
		bHM.put(new CellIndex(1, 53), 34.0); bHM.put(new CellIndex(1, 54), 35.0); bHM.put(new CellIndex(1, 55), 0.0); bHM.put(new CellIndex(1, 56), 36.0); 
		bHM.put(new CellIndex(1, 57), 37.0); bHM.put(new CellIndex(1, 58), 38.0); bHM.put(new CellIndex(1, 59), 39.0); bHM.put(new CellIndex(1, 60), 40.0); 
		bHM.put(new CellIndex(1, 61), 0.0); bHM.put(new CellIndex(1, 62), 41.0); bHM.put(new CellIndex(1, 63), 42.0); bHM.put(new CellIndex(1, 64), 43.0); 
		bHM.put(new CellIndex(1, 65), 44.0); bHM.put(new CellIndex(1, 66), 45.0); bHM.put(new CellIndex(1, 67), 0.0); bHM.put(new CellIndex(1, 68), 46.0); 
		bHM.put(new CellIndex(1, 69), 47.0); bHM.put(new CellIndex(1, 70), 48.0); bHM.put(new CellIndex(1, 71), 49.0); bHM.put(new CellIndex(1, 72), 50.0); 
		bHM.put(new CellIndex(1, 73), 0.0); bHM.put(new CellIndex(1, 74), 0.0); bHM.put(new CellIndex(1, 75), 0.0); bHM.put(new CellIndex(1, 76), 0.0); 
		bHM.put(new CellIndex(1, 77), 0.0); bHM.put(new CellIndex(1, 78), 0.0); bHM.put(new CellIndex(1, 79), 0.0); bHM.put(new CellIndex(1, 80), 51.0); 
		bHM.put(new CellIndex(1, 81), 52.0); bHM.put(new CellIndex(1, 82), 53.0); bHM.put(new CellIndex(1, 83), 54.0); bHM.put(new CellIndex(1, 84), 55.0); 
		bHM.put(new CellIndex(1, 85), 0.0); bHM.put(new CellIndex(1, 86), 56.0); bHM.put(new CellIndex(1, 87), 57.0); bHM.put(new CellIndex(1, 88), 58.0); 
		bHM.put(new CellIndex(1, 89), 59.0); bHM.put(new CellIndex(1, 90), 60.0); bHM.put(new CellIndex(1, 91), 0.0); bHM.put(new CellIndex(1, 92), 61.0); 
		bHM.put(new CellIndex(1, 93), 62.0); bHM.put(new CellIndex(1, 94), 63.0); bHM.put(new CellIndex(1, 95), 64.0); bHM.put(new CellIndex(1, 96), 65.0); 
		bHM.put(new CellIndex(1, 97), 0.0); bHM.put(new CellIndex(1, 98), 66.0); bHM.put(new CellIndex(1, 99), 67.0); bHM.put(new CellIndex(1, 100), 68.0); 
		bHM.put(new CellIndex(1, 101), 69.0); bHM.put(new CellIndex(1, 102), 70.0); bHM.put(new CellIndex(1, 103), 0.0); bHM.put(new CellIndex(1, 104), 71.0); 
		bHM.put(new CellIndex(1, 105), 72.0); bHM.put(new CellIndex(1, 106), 73.0); bHM.put(new CellIndex(1, 107), 74.0); bHM.put(new CellIndex(1, 108), 75.0); 
		bHM.put(new CellIndex(2, 1), 0.0); bHM.put(new CellIndex(2, 2), 0.0); bHM.put(new CellIndex(2, 3), 0.0); bHM.put(new CellIndex(2, 4), 0.0); 
		bHM.put(new CellIndex(2, 5), 0.0); bHM.put(new CellIndex(2, 6), 0.0); bHM.put(new CellIndex(2, 7), 0.0); bHM.put(new CellIndex(2, 8), 76.0); 
		bHM.put(new CellIndex(2, 9), 77.0); bHM.put(new CellIndex(2, 10), 78.0); bHM.put(new CellIndex(2, 11), 79.0); bHM.put(new CellIndex(2, 12), 80.0); 
		bHM.put(new CellIndex(2, 13), 0.0); bHM.put(new CellIndex(2, 14), 81.0); bHM.put(new CellIndex(2, 15), 82.0); bHM.put(new CellIndex(2, 16), 83.0); 
		bHM.put(new CellIndex(2, 17), 84.0); bHM.put(new CellIndex(2, 18), 85.0); bHM.put(new CellIndex(2, 19), 0.0); bHM.put(new CellIndex(2, 20), 86.0); 
		bHM.put(new CellIndex(2, 21), 87.0); bHM.put(new CellIndex(2, 22), 88.0); bHM.put(new CellIndex(2, 23), 89.0); bHM.put(new CellIndex(2, 24), 90.0); 
		bHM.put(new CellIndex(2, 25), 0.0); bHM.put(new CellIndex(2, 26), 91.0); bHM.put(new CellIndex(2, 27), 92.0); bHM.put(new CellIndex(2, 28), 93.0); 
		bHM.put(new CellIndex(2, 29), 94.0); bHM.put(new CellIndex(2, 30), 95.0); bHM.put(new CellIndex(2, 31), 0.0); bHM.put(new CellIndex(2, 32), 96.0); 
		bHM.put(new CellIndex(2, 33), 97.0); bHM.put(new CellIndex(2, 34), 98.0); bHM.put(new CellIndex(2, 35), 99.0); bHM.put(new CellIndex(2, 36), 100.0); 
		bHM.put(new CellIndex(2, 37), 0.0); bHM.put(new CellIndex(2, 38), 0.0); bHM.put(new CellIndex(2, 39), 0.0); bHM.put(new CellIndex(2, 40), 0.0); 
		bHM.put(new CellIndex(2, 41), 0.0); bHM.put(new CellIndex(2, 42), 0.0); bHM.put(new CellIndex(2, 43), 0.0); bHM.put(new CellIndex(2, 44), 101.0); 
		bHM.put(new CellIndex(2, 45), 102.0); bHM.put(new CellIndex(2, 46), 103.0); bHM.put(new CellIndex(2, 47), 104.0); bHM.put(new CellIndex(2, 48), 105.0); 
		bHM.put(new CellIndex(2, 49), 0.0); bHM.put(new CellIndex(2, 50), 106.0); bHM.put(new CellIndex(2, 51), 107.0); bHM.put(new CellIndex(2, 52), 108.0); 
		bHM.put(new CellIndex(2, 53), 109.0); bHM.put(new CellIndex(2, 54), 110.0); bHM.put(new CellIndex(2, 55), 0.0); bHM.put(new CellIndex(2, 56), 111.0); 
		bHM.put(new CellIndex(2, 57), 112.0); bHM.put(new CellIndex(2, 58), 113.0); bHM.put(new CellIndex(2, 59), 114.0); bHM.put(new CellIndex(2, 60), 115.0); 
		bHM.put(new CellIndex(2, 61), 0.0); bHM.put(new CellIndex(2, 62), 116.0); bHM.put(new CellIndex(2, 63), 117.0); bHM.put(new CellIndex(2, 64), 118.0); 
		bHM.put(new CellIndex(2, 65), 119.0); bHM.put(new CellIndex(2, 66), 120.0); bHM.put(new CellIndex(2, 67), 0.0); bHM.put(new CellIndex(2, 68), 121.0); 
		bHM.put(new CellIndex(2, 69), 122.0); bHM.put(new CellIndex(2, 70), 123.0); bHM.put(new CellIndex(2, 71), 124.0); bHM.put(new CellIndex(2, 72), 125.0); 
		bHM.put(new CellIndex(2, 73), 0.0); bHM.put(new CellIndex(2, 74), 0.0); bHM.put(new CellIndex(2, 75), 0.0); bHM.put(new CellIndex(2, 76), 0.0); 
		bHM.put(new CellIndex(2, 77), 0.0); bHM.put(new CellIndex(2, 78), 0.0); bHM.put(new CellIndex(2, 79), 0.0); bHM.put(new CellIndex(2, 80), 126.0); 
		bHM.put(new CellIndex(2, 81), 127.0); bHM.put(new CellIndex(2, 82), 128.0); bHM.put(new CellIndex(2, 83), 129.0); bHM.put(new CellIndex(2, 84), 130.0); 
		bHM.put(new CellIndex(2, 85), 0.0); bHM.put(new CellIndex(2, 86), 131.0); bHM.put(new CellIndex(2, 87), 132.0); bHM.put(new CellIndex(2, 88), 133.0); 
		bHM.put(new CellIndex(2, 89), 134.0); bHM.put(new CellIndex(2, 90), 135.0); bHM.put(new CellIndex(2, 91), 0.0); bHM.put(new CellIndex(2, 92), 136.0); 
		bHM.put(new CellIndex(2, 93), 137.0); bHM.put(new CellIndex(2, 94), 138.0); bHM.put(new CellIndex(2, 95), 139.0); bHM.put(new CellIndex(2, 96), 140.0); 
		bHM.put(new CellIndex(2, 97), 0.0); bHM.put(new CellIndex(2, 98), 141.0); bHM.put(new CellIndex(2, 99), 142.0); bHM.put(new CellIndex(2, 100), 143.0); 
		bHM.put(new CellIndex(2, 101), 144.0); bHM.put(new CellIndex(2, 102), 145.0); bHM.put(new CellIndex(2, 103), 0.0); bHM.put(new CellIndex(2, 104), 146.0); 
		bHM.put(new CellIndex(2, 105), 147.0); bHM.put(new CellIndex(2, 106), 148.0); bHM.put(new CellIndex(2, 107), 149.0); bHM.put(new CellIndex(2, 108), 150.0); 
		bHM.put(new CellIndex(3, 1), 0.0); bHM.put(new CellIndex(3, 2), 0.0); bHM.put(new CellIndex(3, 3), 0.0); bHM.put(new CellIndex(3, 4), 0.0); 
		bHM.put(new CellIndex(3, 5), 0.0); bHM.put(new CellIndex(3, 6), 0.0); bHM.put(new CellIndex(3, 7), 0.0); bHM.put(new CellIndex(3, 8), 151.0); 
		bHM.put(new CellIndex(3, 9), 152.0); bHM.put(new CellIndex(3, 10), 153.0); bHM.put(new CellIndex(3, 11), 154.0); bHM.put(new CellIndex(3, 12), 155.0); 
		bHM.put(new CellIndex(3, 13), 0.0); bHM.put(new CellIndex(3, 14), 156.0); bHM.put(new CellIndex(3, 15), 157.0); bHM.put(new CellIndex(3, 16), 158.0); 
		bHM.put(new CellIndex(3, 17), 159.0); bHM.put(new CellIndex(3, 18), 160.0); bHM.put(new CellIndex(3, 19), 0.0); bHM.put(new CellIndex(3, 20), 161.0); 
		bHM.put(new CellIndex(3, 21), 162.0); bHM.put(new CellIndex(3, 22), 163.0); bHM.put(new CellIndex(3, 23), 164.0); bHM.put(new CellIndex(3, 24), 165.0); 
		bHM.put(new CellIndex(3, 25), 0.0); bHM.put(new CellIndex(3, 26), 166.0); bHM.put(new CellIndex(3, 27), 167.0); bHM.put(new CellIndex(3, 28), 168.0); 
		bHM.put(new CellIndex(3, 29), 169.0); bHM.put(new CellIndex(3, 30), 170.0); bHM.put(new CellIndex(3, 31), 0.0); bHM.put(new CellIndex(3, 32), 171.0); 
		bHM.put(new CellIndex(3, 33), 172.0); bHM.put(new CellIndex(3, 34), 173.0); bHM.put(new CellIndex(3, 35), 174.0); bHM.put(new CellIndex(3, 36), 175.0); 
		bHM.put(new CellIndex(3, 37), 0.0); bHM.put(new CellIndex(3, 38), 0.0); bHM.put(new CellIndex(3, 39), 0.0); bHM.put(new CellIndex(3, 40), 0.0); 
		bHM.put(new CellIndex(3, 41), 0.0); bHM.put(new CellIndex(3, 42), 0.0); bHM.put(new CellIndex(3, 43), 0.0); bHM.put(new CellIndex(3, 44), 176.0); 
		bHM.put(new CellIndex(3, 45), 177.0); bHM.put(new CellIndex(3, 46), 178.0); bHM.put(new CellIndex(3, 47), 179.0); bHM.put(new CellIndex(3, 48), 180.0); 
		bHM.put(new CellIndex(3, 49), 0.0); bHM.put(new CellIndex(3, 50), 181.0); bHM.put(new CellIndex(3, 51), 182.0); bHM.put(new CellIndex(3, 52), 183.0); 
		bHM.put(new CellIndex(3, 53), 184.0); bHM.put(new CellIndex(3, 54), 185.0); bHM.put(new CellIndex(3, 55), 0.0); bHM.put(new CellIndex(3, 56), 186.0); 
		bHM.put(new CellIndex(3, 57), 187.0); bHM.put(new CellIndex(3, 58), 188.0); bHM.put(new CellIndex(3, 59), 189.0); bHM.put(new CellIndex(3, 60), 190.0); 
		bHM.put(new CellIndex(3, 61), 0.0); bHM.put(new CellIndex(3, 62), 191.0); bHM.put(new CellIndex(3, 63), 192.0); bHM.put(new CellIndex(3, 64), 193.0); 
		bHM.put(new CellIndex(3, 65), 194.0); bHM.put(new CellIndex(3, 66), 195.0); bHM.put(new CellIndex(3, 67), 0.0); bHM.put(new CellIndex(3, 68), 196.0); 
		bHM.put(new CellIndex(3, 69), 197.0); bHM.put(new CellIndex(3, 70), 198.0); bHM.put(new CellIndex(3, 71), 199.0); bHM.put(new CellIndex(3, 72), 200.0); 
		bHM.put(new CellIndex(3, 73), 0.0); bHM.put(new CellIndex(3, 74), 0.0); bHM.put(new CellIndex(3, 75), 0.0); bHM.put(new CellIndex(3, 76), 0.0); 
		bHM.put(new CellIndex(3, 77), 0.0); bHM.put(new CellIndex(3, 78), 0.0); bHM.put(new CellIndex(3, 79), 0.0); bHM.put(new CellIndex(3, 80), 201.0); 
		bHM.put(new CellIndex(3, 81), 202.0); bHM.put(new CellIndex(3, 82), 203.0); bHM.put(new CellIndex(3, 83), 204.0); bHM.put(new CellIndex(3, 84), 205.0); 
		bHM.put(new CellIndex(3, 85), 0.0); bHM.put(new CellIndex(3, 86), 206.0); bHM.put(new CellIndex(3, 87), 207.0); bHM.put(new CellIndex(3, 88), 208.0); 
		bHM.put(new CellIndex(3, 89), 209.0); bHM.put(new CellIndex(3, 90), 210.0); bHM.put(new CellIndex(3, 91), 0.0); bHM.put(new CellIndex(3, 92), 211.0); 
		bHM.put(new CellIndex(3, 93), 212.0); bHM.put(new CellIndex(3, 94), 213.0); bHM.put(new CellIndex(3, 95), 214.0); bHM.put(new CellIndex(3, 96), 215.0); 
		bHM.put(new CellIndex(3, 97), 0.0); bHM.put(new CellIndex(3, 98), 216.0); bHM.put(new CellIndex(3, 99), 217.0); bHM.put(new CellIndex(3, 100), 218.0); 
		bHM.put(new CellIndex(3, 101), 219.0); bHM.put(new CellIndex(3, 102), 220.0); bHM.put(new CellIndex(3, 103), 0.0); bHM.put(new CellIndex(3, 104), 221.0); 
		bHM.put(new CellIndex(3, 105), 222.0); bHM.put(new CellIndex(3, 106), 223.0); bHM.put(new CellIndex(3, 107), 224.0); bHM.put(new CellIndex(3, 108), 225.0); 
	}
	
	private void fillMaxPoolTest3HM() {
		bHM.put(new CellIndex(1, 1), 0.0); bHM.put(new CellIndex(1, 2), 0.0); bHM.put(new CellIndex(1, 3), 0.0); bHM.put(new CellIndex(1, 4), 0.0); 
		bHM.put(new CellIndex(1, 5), 0.0); bHM.put(new CellIndex(1, 6), 0.0); bHM.put(new CellIndex(1, 7), 0.0); bHM.put(new CellIndex(1, 8), 0.0); 
		bHM.put(new CellIndex(1, 9), 0.0); bHM.put(new CellIndex(1, 10), 0.0); bHM.put(new CellIndex(1, 11), 0.0); bHM.put(new CellIndex(1, 12), 0.0); 
		bHM.put(new CellIndex(1, 13), 0.0); bHM.put(new CellIndex(1, 14), 0.0); bHM.put(new CellIndex(1, 15), 0.0); bHM.put(new CellIndex(1, 16), 0.0); 
		bHM.put(new CellIndex(1, 17), 1.0); bHM.put(new CellIndex(1, 18), 0.0); bHM.put(new CellIndex(1, 19), 2.0); bHM.put(new CellIndex(1, 20), 0.0); 
		bHM.put(new CellIndex(1, 21), 3.0); bHM.put(new CellIndex(1, 22), 0.0); bHM.put(new CellIndex(1, 23), 0.0); bHM.put(new CellIndex(1, 24), 0.0); 
		bHM.put(new CellIndex(1, 25), 0.0); bHM.put(new CellIndex(1, 26), 0.0); bHM.put(new CellIndex(1, 27), 0.0); bHM.put(new CellIndex(1, 28), 0.0); 
		bHM.put(new CellIndex(1, 29), 0.0); bHM.put(new CellIndex(1, 30), 0.0); bHM.put(new CellIndex(1, 31), 4.0); bHM.put(new CellIndex(1, 32), 0.0); 
		bHM.put(new CellIndex(1, 33), 5.0); bHM.put(new CellIndex(1, 34), 0.0); bHM.put(new CellIndex(1, 35), 6.0); bHM.put(new CellIndex(1, 36), 0.0); 
		bHM.put(new CellIndex(1, 37), 0.0); bHM.put(new CellIndex(1, 38), 0.0); bHM.put(new CellIndex(1, 39), 0.0); bHM.put(new CellIndex(1, 40), 0.0); 
		bHM.put(new CellIndex(1, 41), 0.0); bHM.put(new CellIndex(1, 42), 0.0); bHM.put(new CellIndex(1, 43), 0.0); bHM.put(new CellIndex(1, 44), 0.0); 
		bHM.put(new CellIndex(1, 45), 7.0); bHM.put(new CellIndex(1, 46), 0.0); bHM.put(new CellIndex(1, 47), 8.0); bHM.put(new CellIndex(1, 48), 0.0); 
		bHM.put(new CellIndex(1, 49), 9.0); bHM.put(new CellIndex(1, 50), 0.0); bHM.put(new CellIndex(1, 51), 0.0); bHM.put(new CellIndex(1, 52), 0.0); 
		bHM.put(new CellIndex(1, 53), 0.0); bHM.put(new CellIndex(1, 54), 0.0); bHM.put(new CellIndex(1, 55), 0.0); bHM.put(new CellIndex(1, 56), 0.0); 
		bHM.put(new CellIndex(1, 57), 0.0); bHM.put(new CellIndex(1, 58), 0.0); bHM.put(new CellIndex(1, 59), 0.0); bHM.put(new CellIndex(1, 60), 0.0); 
		bHM.put(new CellIndex(1, 61), 0.0); bHM.put(new CellIndex(1, 62), 0.0); bHM.put(new CellIndex(1, 63), 0.0); bHM.put(new CellIndex(1, 64), 0.0); 
		bHM.put(new CellIndex(1, 65), 0.0); bHM.put(new CellIndex(1, 66), 10.0); bHM.put(new CellIndex(1, 67), 0.0); bHM.put(new CellIndex(1, 68), 11.0); 
		bHM.put(new CellIndex(1, 69), 0.0); bHM.put(new CellIndex(1, 70), 12.0); bHM.put(new CellIndex(1, 71), 0.0); bHM.put(new CellIndex(1, 72), 0.0); 
		bHM.put(new CellIndex(1, 73), 0.0); bHM.put(new CellIndex(1, 74), 0.0); bHM.put(new CellIndex(1, 75), 0.0); bHM.put(new CellIndex(1, 76), 0.0); 
		bHM.put(new CellIndex(1, 77), 0.0); bHM.put(new CellIndex(1, 78), 0.0); bHM.put(new CellIndex(1, 79), 0.0); bHM.put(new CellIndex(1, 80), 13.0); 
		bHM.put(new CellIndex(1, 81), 0.0); bHM.put(new CellIndex(1, 82), 14.0); bHM.put(new CellIndex(1, 83), 0.0); bHM.put(new CellIndex(1, 84), 15.0); 
		bHM.put(new CellIndex(1, 85), 0.0); bHM.put(new CellIndex(1, 86), 0.0); bHM.put(new CellIndex(1, 87), 0.0); bHM.put(new CellIndex(1, 88), 0.0); 
		bHM.put(new CellIndex(1, 89), 0.0); bHM.put(new CellIndex(1, 90), 0.0); bHM.put(new CellIndex(1, 91), 0.0); bHM.put(new CellIndex(1, 92), 0.0); 
		bHM.put(new CellIndex(1, 93), 0.0); bHM.put(new CellIndex(1, 94), 16.0); bHM.put(new CellIndex(1, 95), 0.0); bHM.put(new CellIndex(1, 96), 17.0); 
		bHM.put(new CellIndex(1, 97), 0.0); bHM.put(new CellIndex(1, 98), 18.0); bHM.put(new CellIndex(2, 1), 0.0); bHM.put(new CellIndex(2, 2), 0.0); 
		bHM.put(new CellIndex(2, 3), 0.0); bHM.put(new CellIndex(2, 4), 0.0); bHM.put(new CellIndex(2, 5), 0.0); bHM.put(new CellIndex(2, 6), 0.0); 
		bHM.put(new CellIndex(2, 7), 0.0); bHM.put(new CellIndex(2, 8), 0.0); bHM.put(new CellIndex(2, 9), 0.0); bHM.put(new CellIndex(2, 10), 0.0); 
		bHM.put(new CellIndex(2, 11), 0.0); bHM.put(new CellIndex(2, 12), 0.0); bHM.put(new CellIndex(2, 13), 0.0); bHM.put(new CellIndex(2, 14), 0.0); 
		bHM.put(new CellIndex(2, 15), 0.0); bHM.put(new CellIndex(2, 16), 0.0); bHM.put(new CellIndex(2, 17), 19.0); bHM.put(new CellIndex(2, 18), 0.0); 
		bHM.put(new CellIndex(2, 19), 20.0); bHM.put(new CellIndex(2, 20), 0.0); bHM.put(new CellIndex(2, 21), 21.0); bHM.put(new CellIndex(2, 22), 0.0); 
		bHM.put(new CellIndex(2, 23), 0.0); bHM.put(new CellIndex(2, 24), 0.0); bHM.put(new CellIndex(2, 25), 0.0); bHM.put(new CellIndex(2, 26), 0.0); 
		bHM.put(new CellIndex(2, 27), 0.0); bHM.put(new CellIndex(2, 28), 0.0); bHM.put(new CellIndex(2, 29), 0.0); bHM.put(new CellIndex(2, 30), 0.0); 
		bHM.put(new CellIndex(2, 31), 22.0); bHM.put(new CellIndex(2, 32), 0.0); bHM.put(new CellIndex(2, 33), 23.0); bHM.put(new CellIndex(2, 34), 0.0); 
		bHM.put(new CellIndex(2, 35), 24.0); bHM.put(new CellIndex(2, 36), 0.0); bHM.put(new CellIndex(2, 37), 0.0); bHM.put(new CellIndex(2, 38), 0.0); 
		bHM.put(new CellIndex(2, 39), 0.0); bHM.put(new CellIndex(2, 40), 0.0); bHM.put(new CellIndex(2, 41), 0.0); bHM.put(new CellIndex(2, 42), 0.0); 
		bHM.put(new CellIndex(2, 43), 0.0); bHM.put(new CellIndex(2, 44), 0.0); bHM.put(new CellIndex(2, 45), 25.0); bHM.put(new CellIndex(2, 46), 0.0); 
		bHM.put(new CellIndex(2, 47), 26.0); bHM.put(new CellIndex(2, 48), 0.0); bHM.put(new CellIndex(2, 49), 27.0); bHM.put(new CellIndex(2, 50), 0.0); 
		bHM.put(new CellIndex(2, 51), 0.0); bHM.put(new CellIndex(2, 52), 0.0); bHM.put(new CellIndex(2, 53), 0.0); bHM.put(new CellIndex(2, 54), 0.0); 
		bHM.put(new CellIndex(2, 55), 0.0); bHM.put(new CellIndex(2, 56), 0.0); bHM.put(new CellIndex(2, 57), 0.0); bHM.put(new CellIndex(2, 58), 0.0); 
		bHM.put(new CellIndex(2, 59), 0.0); bHM.put(new CellIndex(2, 60), 0.0); bHM.put(new CellIndex(2, 61), 0.0); bHM.put(new CellIndex(2, 62), 0.0); 
		bHM.put(new CellIndex(2, 63), 0.0); bHM.put(new CellIndex(2, 64), 0.0); bHM.put(new CellIndex(2, 65), 0.0); bHM.put(new CellIndex(2, 66), 28.0); 
		bHM.put(new CellIndex(2, 67), 0.0); bHM.put(new CellIndex(2, 68), 29.0); bHM.put(new CellIndex(2, 69), 0.0); bHM.put(new CellIndex(2, 70), 30.0); 
		bHM.put(new CellIndex(2, 71), 0.0); bHM.put(new CellIndex(2, 72), 0.0); bHM.put(new CellIndex(2, 73), 0.0); bHM.put(new CellIndex(2, 74), 0.0); 
		bHM.put(new CellIndex(2, 75), 0.0); bHM.put(new CellIndex(2, 76), 0.0); bHM.put(new CellIndex(2, 77), 0.0); bHM.put(new CellIndex(2, 78), 0.0); 
		bHM.put(new CellIndex(2, 79), 0.0); bHM.put(new CellIndex(2, 80), 31.0); bHM.put(new CellIndex(2, 81), 0.0); bHM.put(new CellIndex(2, 82), 32.0); 
		bHM.put(new CellIndex(2, 83), 0.0); bHM.put(new CellIndex(2, 84), 33.0); bHM.put(new CellIndex(2, 85), 0.0); bHM.put(new CellIndex(2, 86), 0.0); 
		bHM.put(new CellIndex(2, 87), 0.0); bHM.put(new CellIndex(2, 88), 0.0); bHM.put(new CellIndex(2, 89), 0.0); bHM.put(new CellIndex(2, 90), 0.0); 
		bHM.put(new CellIndex(2, 91), 0.0); bHM.put(new CellIndex(2, 92), 0.0); bHM.put(new CellIndex(2, 93), 0.0); bHM.put(new CellIndex(2, 94), 34.0); 
		bHM.put(new CellIndex(2, 95), 0.0); bHM.put(new CellIndex(2, 96), 35.0); bHM.put(new CellIndex(2, 97), 0.0); bHM.put(new CellIndex(2, 98), 36.0); 
	}
}
