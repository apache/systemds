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
	private final static String TEST_CLASS_DIR = TEST_DIR + PoolBackwardTest.class.getSimpleName() + "/";
	private final static double epsilon=0.0000000001;
	
	@Override
	public void setUp() {
		addTestConfiguration(TEST_NAME, new TestConfiguration(TEST_CLASS_DIR, TEST_NAME, new String[] {"B"}));
	}
	
	@Test
	public void testMaxPool2DBackwardDense1() {
		int numImg = 1; int imgSize = 4; int numChannels = 1;  int stride = 2; int pad = 0; int poolSize1 = 2; int poolSize2 = 2;
		runPoolTest(ExecType.CP, imgSize, numImg, numChannels, stride, pad, poolSize1, poolSize2, "max", false, false);
	}
	
	@Test
	public void testMaxPool2DBackwardDense2() {
		int numImg = 3; int imgSize = 6; int numChannels = 3;  int stride = 1; int pad = 0; int poolSize1 = 2; int poolSize2 = 2;
		runPoolTest(ExecType.CP, imgSize, numImg, numChannels, stride, pad, poolSize1, poolSize2, "max", false, false);
	}
	
	@Test
	public void testMaxPool2DBackwardDense3() {
		int numImg = 2; int imgSize = 7; int numChannels = 2;  int stride = 2; int pad = 0; int poolSize1 = 3; int poolSize2 = 3;
		runPoolTest(ExecType.CP, imgSize, numImg, numChannels, stride, pad, poolSize1, poolSize2, "max", false, false);
	}
	
	@Test
	public void testMaxPool2DBackwardSparse1() {
		int numImg = 1; int imgSize = 4; int numChannels = 1;  int stride = 2; int pad = 0; int poolSize1 = 2; int poolSize2 = 2;
		runPoolTest(ExecType.CP, imgSize, numImg, numChannels, stride, pad, poolSize1, poolSize2, "max", true, false);
	}
	
	@Test
	public void testMaxPool2DBackwardSparse2() {
		int numImg = 3; int imgSize = 6; int numChannels = 3;  int stride = 1; int pad = 0; int poolSize1 = 2; int poolSize2 = 2;
		runPoolTest(ExecType.CP, imgSize, numImg, numChannels, stride, pad, poolSize1, poolSize2, "max", true, false);
	}
	
	@Test
	public void testMaxPool2DBackwardSparse3() {
		int numImg = 2; int imgSize = 7; int numChannels = 2;  int stride = 2; int pad = 0; int poolSize1 = 3; int poolSize2 = 3;
		runPoolTest(ExecType.CP, imgSize, numImg, numChannels, stride, pad, poolSize1, poolSize2, "max", true, false);
	}
	
	@Test
	public void testMaxPool2DBackwardSparse4() {
		int numImg = 1; int imgSize = 4; int numChannels = 1;  int stride = 2; int pad = 0; int poolSize1 = 2; int poolSize2 = 2;
		runPoolTest(ExecType.CP, imgSize, numImg, numChannels, stride, pad, poolSize1, poolSize2, "max", true, true);
	}
	
	@Test
	public void testMaxPool2DBackwardSparse5() {
		int numImg = 3; int imgSize = 6; int numChannels = 3;  int stride = 1; int pad = 0; int poolSize1 = 2; int poolSize2 = 2;
		runPoolTest(ExecType.CP, imgSize, numImg, numChannels, stride, pad, poolSize1, poolSize2, "max", true, true);
	}
	
	@Test
	public void testMaxPool2DBackwardSparse6() {
		int numImg = 2; int imgSize = 7; int numChannels = 2;  int stride = 2; int pad = 0; int poolSize1 = 3; int poolSize2 = 3;
		runPoolTest(ExecType.CP, imgSize, numImg, numChannels, stride, pad, poolSize1, poolSize2, "max", true, true);
	}
	
	@Test
	public void testMaxPool2DBackwardSparse7() {
		int numImg = 1; int imgSize = 4; int numChannels = 1;  int stride = 2; int pad = 0; int poolSize1 = 2; int poolSize2 = 2;
		runPoolTest(ExecType.CP, imgSize, numImg, numChannels, stride, pad, poolSize1, poolSize2, "max", false, true);
	}
	
	@Test
	public void testMaxPool2DBackwardSparse8() {
		int numImg = 3; int imgSize = 6; int numChannels = 3;  int stride = 1; int pad = 0; int poolSize1 = 2; int poolSize2 = 2;
		runPoolTest(ExecType.CP, imgSize, numImg, numChannels, stride, pad, poolSize1, poolSize2, "max", false, true);
	}
	
	@Test
	public void testMaxPool2DBackwardSparse9() {
		int numImg = 2; int imgSize = 7; int numChannels = 2;  int stride = 2; int pad = 0; int poolSize1 = 3; int poolSize2 = 3;
		runPoolTest(ExecType.CP, imgSize, numImg, numChannels, stride, pad, poolSize1, poolSize2, "max", false, true);
	}
	
	@Test
	public void testMaxPool2DBackwardSparse10() {
		int numImg = 10; int imgSize = 28; int numChannels = 1;  int stride = 1; int pad = 0; int poolSize1 = 5; int poolSize2 = 5;
		runPoolTest(ExecType.CP, imgSize, numImg, numChannels, stride, pad, poolSize1, poolSize2, "max", true, false);
	}
	
	@Test
	public void testMaxPool2DBackwardSparse11() {
		int numImg = 10; int imgSize = 28; int numChannels = 1;  int stride = 1; int pad = 0; int poolSize1 = 5; int poolSize2 = 5;
		runPoolTest(ExecType.CP, imgSize, numImg, numChannels, stride, pad, poolSize1, poolSize2, "max", true, true);
	}
	
	@Test
	public void testMaxPool2DBackwardSparse12() {
		int numImg = 10; int imgSize = 28; int numChannels = 4;  int stride = 1; int pad = 0; int poolSize1 = 5; int poolSize2 = 5;
		runPoolTest(ExecType.CP, imgSize, numImg, numChannels, stride, pad, poolSize1, poolSize2, "max", true, true);
	}
	
	public void runPoolTest( ExecType et, int imgSize, int numImg, int numChannels, int stride, 
			int pad, int poolSize1, int poolSize2, String poolMode, boolean sparse1, boolean sparse2) 
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
			String sparseVal1 = String.valueOf(sparse1).toUpperCase();
			String sparseVal2 = String.valueOf(sparse2).toUpperCase();
			TestConfiguration config = getTestConfiguration(TEST_NAME);
			loadTestConfiguration(config);
			
			long P = ConvolutionUtils.getP(imgSize, poolSize1, stride, pad);
			
			String RI_HOME = SCRIPT_DIR + TEST_DIR;
			fullDMLScriptName = RI_HOME + TEST_NAME + ".dml";
			programArgs = new String[]{"-explain", "-args",  String.valueOf(imgSize), String.valueOf(numImg),
					String.valueOf(numChannels), String.valueOf(poolSize1), String.valueOf(poolSize2),
					String.valueOf(stride), String.valueOf(pad), String.valueOf(poolMode),
					String.valueOf(P), String.valueOf(P), output("B"), sparseVal1, sparseVal2};
			
			fullRScriptName = RI_HOME + TEST_NAME + ".R";
			rCmd = "Rscript" + " " + fullRScriptName + " " + imgSize + " " + numImg + " " + numChannels + 
					" " + poolSize1 + " " + poolSize2 + " " + stride + " " + pad + " " +  P + " " + P + 
					" " + expectedDir() + " " + sparseVal1 + " " + sparseVal2; 
			
			// run dml and r scripts
			runTest(true, false, null, -1);
			runRScript(true);
			
			// compare results
			HashMap<CellIndex, Double> bHM = readRMatrixFromFS("B");
			HashMap<CellIndex, Double> dmlfile = readDMLMatrixFromHDFS("B");
			TestUtils.compareMatrices(dmlfile, bHM, epsilon, "B-DML", "R");
		}
		finally {
			rtplatform = platformOld;
			DMLScript.USE_LOCAL_SPARK_CONFIG = sparkConfigOld;
		}
	}
}
