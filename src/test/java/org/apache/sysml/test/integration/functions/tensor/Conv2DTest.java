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
import org.apache.sysml.hops.ConvolutionOp;
import org.apache.sysml.lops.LopProperties.ExecType;
import org.apache.sysml.runtime.matrix.data.MatrixValue.CellIndex;
import org.apache.sysml.test.integration.AutomatedTestBase;
import org.apache.sysml.test.integration.TestConfiguration;
import org.apache.sysml.test.utils.TestUtils;
import org.junit.Test;

public class Conv2DTest extends AutomatedTestBase
{
	
	private final static String TEST_NAME = "Conv2DTest";
	private final static String TEST_DIR = "functions/tensor/";
	private final static String TEST_CLASS_DIR = TEST_DIR + Conv2DTest.class.getSimpleName() + "/";
	private final static double epsilon=0.0000000001;
	
	@Override
	public void setUp() {
		addTestConfiguration(TEST_NAME, new TestConfiguration(TEST_CLASS_DIR, TEST_NAME, 
				new String[] {"B"}));
	}
	
	@Test
	public void testConv2DDense1() 
	{
		int numImg = 5; int imgSize = 3; int numChannels = 3; int numFilters = 6; int filterSize = 2; int stride = 1; int pad = 0;
		runConv2DTest(ExecType.CP, imgSize, numImg, numChannels, numFilters, filterSize, stride, pad, false);
	}
	
	@Test
	public void testConv2DDense2() 
	{
		int numImg = 1; int imgSize = 10; int numChannels = 4; int numFilters = 3; int filterSize = 4; int stride = 2; int pad = 0;
		runConv2DTest(ExecType.CP, imgSize, numImg, numChannels, numFilters, filterSize, stride, pad, false);
	}
	
	@Test
	public void testConv2DDense3() 
	{
		int numImg = 1; int imgSize = 10; int numChannels = 4; int numFilters = 3; int filterSize = 4; int stride = 2; int pad = 1;
		runConv2DTest(ExecType.CP, imgSize, numImg, numChannels, numFilters, filterSize, stride, pad, false);
	}
	
	@Test
	public void testConv2DDense4() 
	{
		int numImg = 3; int imgSize = 10; int numChannels = 1; int numFilters = 3; int filterSize = 2; int stride = 2; int pad = 1;
		runConv2DTest(ExecType.CP, imgSize, numImg, numChannels, numFilters, filterSize, stride, pad, false);
	}
	
	@Test
	public void testConv2DDense5() 
	{
		int numImg = 3; int imgSize = 8; int numChannels = 2; int numFilters = 3; int filterSize = 3; int stride = 1; int pad = 2;
		runConv2DTest(ExecType.CP, imgSize, numImg, numChannels, numFilters, filterSize, stride, pad, false);
	}
	
	@Test
	public void testConv2DDense6() 
	{
		int numImg = 5; int imgSize = 3; int numChannels = 3; int numFilters = 6; int filterSize = 2; int stride = 1; int pad = 0;
		runConv2DTest(ExecType.CP, imgSize, numImg, numChannels, numFilters, filterSize, stride, pad, true);
	}
	
	@Test
	public void testConv2DDense7() 
	{
		int numImg = 1; int imgSize = 10; int numChannels = 4; int numFilters = 3; int filterSize = 4; int stride = 2; int pad = 0;
		runConv2DTest(ExecType.CP, imgSize, numImg, numChannels, numFilters, filterSize, stride, pad, true);
	}
	
	@Test
	public void testConv2DDense8() 
	{
		int numImg = 1; int imgSize = 10; int numChannels = 4; int numFilters = 3; int filterSize = 4; int stride = 2; int pad = 1;
		runConv2DTest(ExecType.CP, imgSize, numImg, numChannels, numFilters, filterSize, stride, pad, true);
	}
	
	@Test
	public void testConv2DDense9() 
	{
		int numImg = 3; int imgSize = 10; int numChannels = 1; int numFilters = 3; int filterSize = 2; int stride = 2; int pad = 1;
		runConv2DTest(ExecType.CP, imgSize, numImg, numChannels, numFilters, filterSize, stride, pad, true);
	}
	
	@Test
	public void testConv2DDense10() 
	{
		int numImg = 3; int imgSize = 8; int numChannels = 2; int numFilters = 3; int filterSize = 3; int stride = 1; int pad = 2;
		runConv2DTest(ExecType.CP, imgSize, numImg, numChannels, numFilters, filterSize, stride, pad, true);
	}
	
	
	/**
	 * 
	 * @param et
	 * @param sparse
	 */
	public void runConv2DTest( ExecType et, int imgSize, int numImg, int numChannels, int numFilters, 
			int filterSize, int stride, int pad, boolean FORCE_NON_IM2COL) 
	{
		RUNTIME_PLATFORM oldRTP = rtplatform;
			
		boolean sparkConfigOld = DMLScript.USE_LOCAL_SPARK_CONFIG;
		
		boolean oldForceNonIm2col = ConvolutionOp.FORCE_NON_IM2COL; 
		ConvolutionOp.FORCE_NON_IM2COL = FORCE_NON_IM2COL;
		
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
				"" + numChannels, "" + numFilters, 
				"" + filterSize, "" + stride, "" + pad, 
				output("B")};
			
			fullRScriptName = RI_HOME + TEST_NAME + ".R";
			rCmd = "Rscript" + " " + fullRScriptName + " " + imgSize + " " + numImg + 
					" " + numChannels + " " + numFilters + 
					" " + filterSize + " " + stride + " " + pad + " " + expectedDir(); 
			
			boolean exceptionExpected = false;
			int expectedNumberOfJobs = -1;
			runTest(true, exceptionExpected, null, expectedNumberOfJobs);

			// Run comparison R script
			runRScript(true);
			HashMap<CellIndex, Double> bHM = readRMatrixFromFS("B");
			
			HashMap<CellIndex, Double> dmlfile = readDMLMatrixFromHDFS("B");
			TestUtils.compareMatrices(dmlfile, bHM, epsilon, "B-DML", "B-R");
			
		}
		finally
		{
			rtplatform = oldRTP;
			DMLScript.USE_LOCAL_SPARK_CONFIG = sparkConfigOld;
			ConvolutionOp.FORCE_NON_IM2COL = oldForceNonIm2col;
		}
	}
}

