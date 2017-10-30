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

public class ChannelSumTest extends AutomatedTestBase
{
	
	private final static String TEST_NAME = "ChannelSumTest";
	private final static String TEST_DIR = "functions/tensor/";
	private final static String TEST_CLASS_DIR = TEST_DIR + PoolTest.class.getSimpleName() + "/";
	private final static double epsilon=0.0000000001;
	
	@Override
	public void setUp() {
		addTestConfiguration(TEST_NAME, new TestConfiguration(TEST_CLASS_DIR, TEST_NAME, 
				new String[] {"B"}));
	}
	
	@Test
	public void testChannelSumDense1() 
	{
		int numImg = 10; int imgSize = 9; int numChannels = 5; 
		runChannelSumTest(ExecType.CP, imgSize, numImg, numChannels, false);
	}
	
	@Test
	public void testChannelSumDense2() 
	{
		int numImg = 2; int imgSize = 5; int numChannels = 3; 
		runChannelSumTest(ExecType.CP, imgSize, numImg, numChannels, false);
	}
	
	@Test
	public void testChannelSumDense3() 
	{
		int numImg = 9; int imgSize = 4; int numChannels = 11; 
		runChannelSumTest(ExecType.CP, imgSize, numImg, numChannels, false);
	}
	
	@Test
	public void testChannelSumDense4() 
	{
		int numImg = 7; int imgSize = 8; int numChannels = 12; 
		runChannelSumTest(ExecType.CP, imgSize, numImg, numChannels, false);
	}
	
	@Test
	public void testChannelSumSparse1() 
	{
		int numImg = 4; int imgSize = 10; int numChannels = 5; 
		runChannelSumTest(ExecType.CP, imgSize, numImg, numChannels, true);
	}
	
	@Test
	public void testChannelSumSparse2() 
	{
		int numImg = 2; int imgSize = 10; int numChannels = 8; 
		runChannelSumTest(ExecType.CP, imgSize, numImg, numChannels, true);
	}
	
	@Test
	public void testChannelSumSparse3() 
	{
		int numImg = 4; int imgSize = 10; int numChannels = 11; 
		runChannelSumTest(ExecType.CP, imgSize, numImg, numChannels, true);
	}
	
	@Test
	public void testChannelSumSparse4() 
	{
		int numImg = 9; int imgSize = 6; int numChannels = 8; 
		runChannelSumTest(ExecType.CP, imgSize, numImg, numChannels, true);
	}
	
	public void runChannelSumTest( ExecType et, int imgSize, int numImg, int numChannels, boolean sparse) 
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
			String sparseVal = String.valueOf(sparse).toUpperCase();
			
			TestConfiguration config = getTestConfiguration(TEST_NAME);
			loadTestConfiguration(config);
	
			String RI_HOME = SCRIPT_DIR + TEST_DIR;
			fullDMLScriptName = RI_HOME + TEST_NAME + ".dml";
			programArgs = new String[]{"-explain", "hops", "-args", String.valueOf(imgSize), 
				String.valueOf(numImg), String.valueOf(numChannels),
				output("B"), sparseVal};
			
			fullRScriptName = RI_HOME + TEST_NAME + ".R";
			rCmd = "Rscript" + " " + fullRScriptName + " " + imgSize + " " + numImg + 
				" " + numChannels + " " + expectedDir() + " " + sparseVal; 
			
			// run scripts
			runTest(true, false, null, -1);
			runRScript(true);
			
			//compare results
			HashMap<CellIndex, Double> bHM = readRMatrixFromFS("B");
			HashMap<CellIndex, Double> dmlfile = readDMLMatrixFromHDFS("B");
			TestUtils.compareMatrices(dmlfile, bHM, epsilon, "B-DML", "NumPy");
		}
		finally {
			rtplatform = platformOld;
			DMLScript.USE_LOCAL_SPARK_CONFIG = sparkConfigOld;
		}
	}
	
}
