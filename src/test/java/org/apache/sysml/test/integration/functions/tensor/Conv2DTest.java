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
	
	HashMap<CellIndex, Double> bHM = new HashMap<CellIndex, Double>();
	
	@Test
	public void testConv2DDense1() 
	{
		int numImg = 5; int imgSize = 3; int numChannels = 3; int numFilters = 6; int filterSize = 2; int stride = 1; int pad = 0;
		fillTest1HM();
		runConv2DTest(ExecType.CP, imgSize, numImg, numChannels, numFilters, filterSize, stride, pad);
	}
	
	@Test
	public void testConv2DDense2() 
	{
		int numImg = 1; int imgSize = 10; int numChannels = 4; int numFilters = 3; int filterSize = 4; int stride = 2; int pad = 0;
		fillTest2HM();
		runConv2DTest(ExecType.CP, imgSize, numImg, numChannels, numFilters, filterSize, stride, pad);
	}
	
	@Test
	public void testConv2DDense3() 
	{
		int numImg = 1; int imgSize = 10; int numChannels = 4; int numFilters = 3; int filterSize = 4; int stride = 2; int pad = 1;
		fillTest3HM();
		runConv2DTest(ExecType.CP, imgSize, numImg, numChannels, numFilters, filterSize, stride, pad);
	}
	
	@Test
	public void testConv2DDense4() 
	{
		int numImg = 3; int imgSize = 10; int numChannels = 1; int numFilters = 3; int filterSize = 2; int stride = 2; int pad = 1;
		fillTest4HM();
		runConv2DTest(ExecType.CP, imgSize, numImg, numChannels, numFilters, filterSize, stride, pad);
	}
	
	/**
	 * 
	 * @param et
	 * @param sparse
	 */
	public void runConv2DTest( ExecType et, int imgSize, int numImg, int numChannels, int numFilters, 
			int filterSize, int stride, int pad) 
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
				"" + numChannels, "" + numFilters, 
				"" + filterSize, "" + stride, "" + pad, 
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
	
	private void fillTest1HM() {
		bHM.put(new CellIndex(1, 1), 1245.0); bHM.put(new CellIndex(1, 2), 1323.0); bHM.put(new CellIndex(1, 3), 1479.0); bHM.put(new CellIndex(1, 4), 1557.0); 
		bHM.put(new CellIndex(1, 5), 2973.0); bHM.put(new CellIndex(1, 6), 3195.0); bHM.put(new CellIndex(1, 7), 3639.0); bHM.put(new CellIndex(1, 8), 3861.0); 
		bHM.put(new CellIndex(1, 9), 4701.0); bHM.put(new CellIndex(1, 10), 5067.0); bHM.put(new CellIndex(1, 11), 5799.0); bHM.put(new CellIndex(1, 12), 6165.0); 
		bHM.put(new CellIndex(1, 13), 6429.0); bHM.put(new CellIndex(1, 14), 6939.0); bHM.put(new CellIndex(1, 15), 7959.0); bHM.put(new CellIndex(1, 16), 8469.0); 
		bHM.put(new CellIndex(1, 17), 8157.0); bHM.put(new CellIndex(1, 18), 8811.0); bHM.put(new CellIndex(1, 19), 10119.0); bHM.put(new CellIndex(1, 20), 10773.0); 
		bHM.put(new CellIndex(1, 21), 9885.0); bHM.put(new CellIndex(1, 22), 10683.0); bHM.put(new CellIndex(1, 23), 12279.0); bHM.put(new CellIndex(1, 24), 13077.0); 
		bHM.put(new CellIndex(2, 1), 3351.0); bHM.put(new CellIndex(2, 2), 3429.0); bHM.put(new CellIndex(2, 3), 3585.0); bHM.put(new CellIndex(2, 4), 3663.0); 
		bHM.put(new CellIndex(2, 5), 8967.0); bHM.put(new CellIndex(2, 6), 9189.0); bHM.put(new CellIndex(2, 7), 9633.0); bHM.put(new CellIndex(2, 8), 9855.0); 
		bHM.put(new CellIndex(2, 9), 14583.0); bHM.put(new CellIndex(2, 10), 14949.0); bHM.put(new CellIndex(2, 11), 15681.0); bHM.put(new CellIndex(2, 12), 16047.0); 
		bHM.put(new CellIndex(2, 13), 20199.0); bHM.put(new CellIndex(2, 14), 20709.0); bHM.put(new CellIndex(2, 15), 21729.0); bHM.put(new CellIndex(2, 16), 22239.0); 
		bHM.put(new CellIndex(2, 17), 25815.0); bHM.put(new CellIndex(2, 18), 26469.0); bHM.put(new CellIndex(2, 19), 27777.0); bHM.put(new CellIndex(2, 20), 28431.0); 
		bHM.put(new CellIndex(2, 21), 31431.0); bHM.put(new CellIndex(2, 22), 32229.0); bHM.put(new CellIndex(2, 23), 33825.0); bHM.put(new CellIndex(2, 24), 34623.0); 
		bHM.put(new CellIndex(3, 1), 5457.0); bHM.put(new CellIndex(3, 2), 5535.0); bHM.put(new CellIndex(3, 3), 5691.0); bHM.put(new CellIndex(3, 4), 5769.0); 
		bHM.put(new CellIndex(3, 5), 14961.0); bHM.put(new CellIndex(3, 6), 15183.0); bHM.put(new CellIndex(3, 7), 15627.0); bHM.put(new CellIndex(3, 8), 15849.0); 
		bHM.put(new CellIndex(3, 9), 24465.0); bHM.put(new CellIndex(3, 10), 24831.0); bHM.put(new CellIndex(3, 11), 25563.0); bHM.put(new CellIndex(3, 12), 25929.0); 
		bHM.put(new CellIndex(3, 13), 33969.0); bHM.put(new CellIndex(3, 14), 34479.0); bHM.put(new CellIndex(3, 15), 35499.0); bHM.put(new CellIndex(3, 16), 36009.0); 
		bHM.put(new CellIndex(3, 17), 43473.0); bHM.put(new CellIndex(3, 18), 44127.0); bHM.put(new CellIndex(3, 19), 45435.0); bHM.put(new CellIndex(3, 20), 46089.0); 
		bHM.put(new CellIndex(3, 21), 52977.0); bHM.put(new CellIndex(3, 22), 53775.0); bHM.put(new CellIndex(3, 23), 55371.0); bHM.put(new CellIndex(3, 24), 56169.0); 
		bHM.put(new CellIndex(4, 1), 7563.0); bHM.put(new CellIndex(4, 2), 7641.0); bHM.put(new CellIndex(4, 3), 7797.0); bHM.put(new CellIndex(4, 4), 7875.0); 
		bHM.put(new CellIndex(4, 5), 20955.0); bHM.put(new CellIndex(4, 6), 21177.0); bHM.put(new CellIndex(4, 7), 21621.0); bHM.put(new CellIndex(4, 8), 21843.0); 
		bHM.put(new CellIndex(4, 9), 34347.0); bHM.put(new CellIndex(4, 10), 34713.0); bHM.put(new CellIndex(4, 11), 35445.0); bHM.put(new CellIndex(4, 12), 35811.0); 
		bHM.put(new CellIndex(4, 13), 47739.0); bHM.put(new CellIndex(4, 14), 48249.0); bHM.put(new CellIndex(4, 15), 49269.0); bHM.put(new CellIndex(4, 16), 49779.0); 
		bHM.put(new CellIndex(4, 17), 61131.0); bHM.put(new CellIndex(4, 18), 61785.0); bHM.put(new CellIndex(4, 19), 63093.0); bHM.put(new CellIndex(4, 20), 63747.0); 
		bHM.put(new CellIndex(4, 21), 74523.0); bHM.put(new CellIndex(4, 22), 75321.0); bHM.put(new CellIndex(4, 23), 76917.0); bHM.put(new CellIndex(4, 24), 77715.0); 
		bHM.put(new CellIndex(5, 1), 9669.0); bHM.put(new CellIndex(5, 2), 9747.0); bHM.put(new CellIndex(5, 3), 9903.0); bHM.put(new CellIndex(5, 4), 9981.0); 
		bHM.put(new CellIndex(5, 5), 26949.0); bHM.put(new CellIndex(5, 6), 27171.0); bHM.put(new CellIndex(5, 7), 27615.0); bHM.put(new CellIndex(5, 8), 27837.0); 
		bHM.put(new CellIndex(5, 9), 44229.0); bHM.put(new CellIndex(5, 10), 44595.0); bHM.put(new CellIndex(5, 11), 45327.0); bHM.put(new CellIndex(5, 12), 45693.0); 
		bHM.put(new CellIndex(5, 13), 61509.0); bHM.put(new CellIndex(5, 14), 62019.0); bHM.put(new CellIndex(5, 15), 63039.0); bHM.put(new CellIndex(5, 16), 63549.0); 
		bHM.put(new CellIndex(5, 17), 78789.0); bHM.put(new CellIndex(5, 18), 79443.0); bHM.put(new CellIndex(5, 19), 80751.0); bHM.put(new CellIndex(5, 20), 81405.0); 
		bHM.put(new CellIndex(5, 21), 96069.0); bHM.put(new CellIndex(5, 22), 96867.0); bHM.put(new CellIndex(5, 23), 98463.0); bHM.put(new CellIndex(5, 24), 99261.0); 
	}
	
	private void fillTest2HM() {
		bHM.put(new CellIndex(1, 1), 479680.0); bHM.put(new CellIndex(1, 2), 483840.0); bHM.put(new CellIndex(1, 3), 488000.0); bHM.put(new CellIndex(1, 4), 492160.0); 
		bHM.put(new CellIndex(1, 5), 521280.0); bHM.put(new CellIndex(1, 6), 525440.0); bHM.put(new CellIndex(1, 7), 529600.0); bHM.put(new CellIndex(1, 8), 533760.0); 
		bHM.put(new CellIndex(1, 9), 562880.0); bHM.put(new CellIndex(1, 10), 567040.0); bHM.put(new CellIndex(1, 11), 571200.0); bHM.put(new CellIndex(1, 12), 575360.0); 
		bHM.put(new CellIndex(1, 13), 604480.0); bHM.put(new CellIndex(1, 14), 608640.0); bHM.put(new CellIndex(1, 15), 612800.0); bHM.put(new CellIndex(1, 16), 616960.0); 
		bHM.put(new CellIndex(1, 17), 1165760.0); bHM.put(new CellIndex(1, 18), 1178112.0); bHM.put(new CellIndex(1, 19), 1190464.0); bHM.put(new CellIndex(1, 20), 1202816.0); 
		bHM.put(new CellIndex(1, 21), 1289280.0); bHM.put(new CellIndex(1, 22), 1301632.0); bHM.put(new CellIndex(1, 23), 1313984.0); bHM.put(new CellIndex(1, 24), 1326336.0); 
		bHM.put(new CellIndex(1, 25), 1412800.0); bHM.put(new CellIndex(1, 26), 1425152.0); bHM.put(new CellIndex(1, 27), 1437504.0); bHM.put(new CellIndex(1, 28), 1449856.0); 
		bHM.put(new CellIndex(1, 29), 1536320.0); bHM.put(new CellIndex(1, 30), 1548672.0); bHM.put(new CellIndex(1, 31), 1561024.0); bHM.put(new CellIndex(1, 32), 1573376.0); 
		bHM.put(new CellIndex(1, 33), 1851840.0); bHM.put(new CellIndex(1, 34), 1872384.0); bHM.put(new CellIndex(1, 35), 1892928.0); bHM.put(new CellIndex(1, 36), 1913472.0); 
		bHM.put(new CellIndex(1, 37), 2057280.0); bHM.put(new CellIndex(1, 38), 2077824.0); bHM.put(new CellIndex(1, 39), 2098368.0); bHM.put(new CellIndex(1, 40), 2118912.0); 
		bHM.put(new CellIndex(1, 41), 2262720.0); bHM.put(new CellIndex(1, 42), 2283264.0); bHM.put(new CellIndex(1, 43), 2303808.0); bHM.put(new CellIndex(1, 44), 2324352.0); 
		bHM.put(new CellIndex(1, 45), 2468160.0); bHM.put(new CellIndex(1, 46), 2488704.0); bHM.put(new CellIndex(1, 47), 2509248.0); bHM.put(new CellIndex(1, 48), 2529792.0);  
	}
	
	private void fillTest3HM() {
		bHM.put(new CellIndex(1, 1), 277104.0); bHM.put(new CellIndex(1, 2), 368096.0); bHM.put(new CellIndex(1, 3), 371408.0); bHM.put(new CellIndex(1, 4), 374720.0); 
		bHM.put(new CellIndex(1, 5), 279840.0); bHM.put(new CellIndex(1, 6), 378800.0); bHM.put(new CellIndex(1, 7), 502560.0); bHM.put(new CellIndex(1, 8), 506720.0); 
		bHM.put(new CellIndex(1, 9), 510880.0); bHM.put(new CellIndex(1, 10), 381056.0); bHM.put(new CellIndex(1, 11), 410480.0); bHM.put(new CellIndex(1, 12), 544160.0); 
		bHM.put(new CellIndex(1, 13), 548320.0); bHM.put(new CellIndex(1, 14), 552480.0); bHM.put(new CellIndex(1, 15), 411776.0); bHM.put(new CellIndex(1, 16), 442160.0); 
		bHM.put(new CellIndex(1, 17), 585760.0); bHM.put(new CellIndex(1, 18), 589920.0); bHM.put(new CellIndex(1, 19), 594080.0); bHM.put(new CellIndex(1, 20), 442496.0); 
		bHM.put(new CellIndex(1, 21), 331896.0); bHM.put(new CellIndex(1, 22), 439184.0); bHM.put(new CellIndex(1, 23), 442112.0); bHM.put(new CellIndex(1, 24), 445040.0); 
		bHM.put(new CellIndex(1, 25), 331104.0); bHM.put(new CellIndex(1, 26), 650352.0); bHM.put(new CellIndex(1, 27), 870368.0); bHM.put(new CellIndex(1, 28), 879824.0); 
		bHM.put(new CellIndex(1, 29), 889280.0); bHM.put(new CellIndex(1, 30), 669216.0); bHM.put(new CellIndex(1, 31), 922544.0); bHM.put(new CellIndex(1, 32), 1233696.0); 
		bHM.put(new CellIndex(1, 33), 1246048.0); bHM.put(new CellIndex(1, 34), 1258400.0); bHM.put(new CellIndex(1, 35), 946304.0); bHM.put(new CellIndex(1, 36), 1015664.0); 
		bHM.put(new CellIndex(1, 37), 1357216.0); bHM.put(new CellIndex(1, 38), 1369568.0); bHM.put(new CellIndex(1, 39), 1381920.0); bHM.put(new CellIndex(1, 40), 1038464.0); 
		bHM.put(new CellIndex(1, 41), 1108784.0); bHM.put(new CellIndex(1, 42), 1480736.0); bHM.put(new CellIndex(1, 43), 1493088.0); bHM.put(new CellIndex(1, 44), 1505440.0); 
		bHM.put(new CellIndex(1, 45), 1130624.0); bHM.put(new CellIndex(1, 46), 866424.0); bHM.put(new CellIndex(1, 47), 1156496.0); bHM.put(new CellIndex(1, 48), 1165568.0); 
		bHM.put(new CellIndex(1, 49), 1174640.0); bHM.put(new CellIndex(1, 50), 881760.0); bHM.put(new CellIndex(1, 51), 1023600.0); bHM.put(new CellIndex(1, 52), 1372640.0); 
		bHM.put(new CellIndex(1, 53), 1388240.0); bHM.put(new CellIndex(1, 54), 1403840.0); bHM.put(new CellIndex(1, 55), 1058592.0); bHM.put(new CellIndex(1, 56), 1466288.0); 
		bHM.put(new CellIndex(1, 57), 1964832.0); bHM.put(new CellIndex(1, 58), 1985376.0); bHM.put(new CellIndex(1, 59), 2005920.0); bHM.put(new CellIndex(1, 60), 1511552.0); 
		bHM.put(new CellIndex(1, 61), 1620848.0); bHM.put(new CellIndex(1, 62), 2170272.0); bHM.put(new CellIndex(1, 63), 2190816.0); bHM.put(new CellIndex(1, 64), 2211360.0); 
		bHM.put(new CellIndex(1, 65), 1665152.0); bHM.put(new CellIndex(1, 66), 1775408.0); bHM.put(new CellIndex(1, 67), 2375712.0); bHM.put(new CellIndex(1, 68), 2396256.0); 
		bHM.put(new CellIndex(1, 69), 2416800.0); bHM.put(new CellIndex(1, 70), 1818752.0); bHM.put(new CellIndex(1, 71), 1400952.0); bHM.put(new CellIndex(1, 72), 1873808.0); 
		bHM.put(new CellIndex(1, 73), 1889024.0); bHM.put(new CellIndex(1, 74), 1904240.0); bHM.put(new CellIndex(1, 75), 1432416.0); 
	}
	
	private void fillTest4HM() {
		bHM.put(new CellIndex(1, 1), 4.0); bHM.put(new CellIndex(1, 2), 18.0); bHM.put(new CellIndex(1, 3), 32.0); bHM.put(new CellIndex(1, 4), 46.0); 
		bHM.put(new CellIndex(1, 5), 60.0); bHM.put(new CellIndex(1, 6), 30.0); bHM.put(new CellIndex(1, 7), 106.0); bHM.put(new CellIndex(1, 8), 196.0); 
		bHM.put(new CellIndex(1, 9), 216.0); bHM.put(new CellIndex(1, 10), 236.0); bHM.put(new CellIndex(1, 11), 256.0); bHM.put(new CellIndex(1, 12), 110.0); 
		bHM.put(new CellIndex(1, 13), 226.0); bHM.put(new CellIndex(1, 14), 396.0); bHM.put(new CellIndex(1, 15), 416.0); bHM.put(new CellIndex(1, 16), 436.0); 
		bHM.put(new CellIndex(1, 17), 456.0); bHM.put(new CellIndex(1, 18), 190.0); bHM.put(new CellIndex(1, 19), 346.0); bHM.put(new CellIndex(1, 20), 596.0); 
		bHM.put(new CellIndex(1, 21), 616.0); bHM.put(new CellIndex(1, 22), 636.0); bHM.put(new CellIndex(1, 23), 656.0); bHM.put(new CellIndex(1, 24), 270.0); 
		bHM.put(new CellIndex(1, 25), 466.0); bHM.put(new CellIndex(1, 26), 796.0); bHM.put(new CellIndex(1, 27), 816.0); bHM.put(new CellIndex(1, 28), 836.0); 
		bHM.put(new CellIndex(1, 29), 856.0); bHM.put(new CellIndex(1, 30), 350.0); bHM.put(new CellIndex(1, 31), 182.0); bHM.put(new CellIndex(1, 32), 278.0); 
		bHM.put(new CellIndex(1, 33), 284.0); bHM.put(new CellIndex(1, 34), 290.0); bHM.put(new CellIndex(1, 35), 296.0); bHM.put(new CellIndex(1, 36), 100.0); 
		bHM.put(new CellIndex(1, 37), 8.0); bHM.put(new CellIndex(1, 38), 38.0); bHM.put(new CellIndex(1, 39), 68.0); bHM.put(new CellIndex(1, 40), 98.0); 
		bHM.put(new CellIndex(1, 41), 128.0); bHM.put(new CellIndex(1, 42), 70.0); bHM.put(new CellIndex(1, 43), 234.0); bHM.put(new CellIndex(1, 44), 476.0); 
		bHM.put(new CellIndex(1, 45), 528.0); bHM.put(new CellIndex(1, 46), 580.0); bHM.put(new CellIndex(1, 47), 632.0); bHM.put(new CellIndex(1, 48), 310.0); 
		bHM.put(new CellIndex(1, 49), 514.0); bHM.put(new CellIndex(1, 50), 996.0); bHM.put(new CellIndex(1, 51), 1048.0); bHM.put(new CellIndex(1, 52), 1100.0); 
		bHM.put(new CellIndex(1, 53), 1152.0); bHM.put(new CellIndex(1, 54), 550.0); bHM.put(new CellIndex(1, 55), 794.0); bHM.put(new CellIndex(1, 56), 1516.0); 
		bHM.put(new CellIndex(1, 57), 1568.0); bHM.put(new CellIndex(1, 58), 1620.0); bHM.put(new CellIndex(1, 59), 1672.0); bHM.put(new CellIndex(1, 60), 790.0); 
		bHM.put(new CellIndex(1, 61), 1074.0); bHM.put(new CellIndex(1, 62), 2036.0); bHM.put(new CellIndex(1, 63), 2088.0); bHM.put(new CellIndex(1, 64), 2140.0); 
		bHM.put(new CellIndex(1, 65), 2192.0); bHM.put(new CellIndex(1, 66), 1030.0); bHM.put(new CellIndex(1, 67), 546.0); bHM.put(new CellIndex(1, 68), 1018.0); 
		bHM.put(new CellIndex(1, 69), 1040.0); bHM.put(new CellIndex(1, 70), 1062.0); bHM.put(new CellIndex(1, 71), 1084.0); bHM.put(new CellIndex(1, 72), 500.0); 
		bHM.put(new CellIndex(1, 73), 12.0); bHM.put(new CellIndex(1, 74), 58.0); bHM.put(new CellIndex(1, 75), 104.0); bHM.put(new CellIndex(1, 76), 150.0); 
		bHM.put(new CellIndex(1, 77), 196.0); bHM.put(new CellIndex(1, 78), 110.0); bHM.put(new CellIndex(1, 79), 362.0); bHM.put(new CellIndex(1, 80), 756.0); 
		bHM.put(new CellIndex(1, 81), 840.0); bHM.put(new CellIndex(1, 82), 924.0); bHM.put(new CellIndex(1, 83), 1008.0); bHM.put(new CellIndex(1, 84), 510.0); 
		bHM.put(new CellIndex(1, 85), 802.0); bHM.put(new CellIndex(1, 86), 1596.0); bHM.put(new CellIndex(1, 87), 1680.0); bHM.put(new CellIndex(1, 88), 1764.0); 
		bHM.put(new CellIndex(1, 89), 1848.0); bHM.put(new CellIndex(1, 90), 910.0); bHM.put(new CellIndex(1, 91), 1242.0); bHM.put(new CellIndex(1, 92), 2436.0); 
		bHM.put(new CellIndex(1, 93), 2520.0); bHM.put(new CellIndex(1, 94), 2604.0); bHM.put(new CellIndex(1, 95), 2688.0); bHM.put(new CellIndex(1, 96), 1310.0); 
		bHM.put(new CellIndex(1, 97), 1682.0); bHM.put(new CellIndex(1, 98), 3276.0); bHM.put(new CellIndex(1, 99), 3360.0); bHM.put(new CellIndex(1, 100), 3444.0); 
		bHM.put(new CellIndex(1, 101), 3528.0); bHM.put(new CellIndex(1, 102), 1710.0); bHM.put(new CellIndex(1, 103), 910.0); bHM.put(new CellIndex(1, 104), 1758.0); 
		bHM.put(new CellIndex(1, 105), 1796.0); bHM.put(new CellIndex(1, 106), 1834.0); bHM.put(new CellIndex(1, 107), 1872.0); bHM.put(new CellIndex(1, 108), 900.0); 
		bHM.put(new CellIndex(2, 1), 404.0); bHM.put(new CellIndex(2, 2), 718.0); bHM.put(new CellIndex(2, 3), 732.0); bHM.put(new CellIndex(2, 4), 746.0); 
		bHM.put(new CellIndex(2, 5), 760.0); bHM.put(new CellIndex(2, 6), 330.0); bHM.put(new CellIndex(2, 7), 706.0); bHM.put(new CellIndex(2, 8), 1196.0); 
		bHM.put(new CellIndex(2, 9), 1216.0); bHM.put(new CellIndex(2, 10), 1236.0); bHM.put(new CellIndex(2, 11), 1256.0); bHM.put(new CellIndex(2, 12), 510.0); 
		bHM.put(new CellIndex(2, 13), 826.0); bHM.put(new CellIndex(2, 14), 1396.0); bHM.put(new CellIndex(2, 15), 1416.0); bHM.put(new CellIndex(2, 16), 1436.0); 
		bHM.put(new CellIndex(2, 17), 1456.0); bHM.put(new CellIndex(2, 18), 590.0); bHM.put(new CellIndex(2, 19), 946.0); bHM.put(new CellIndex(2, 20), 1596.0); 
		bHM.put(new CellIndex(2, 21), 1616.0); bHM.put(new CellIndex(2, 22), 1636.0); bHM.put(new CellIndex(2, 23), 1656.0); bHM.put(new CellIndex(2, 24), 670.0); 
		bHM.put(new CellIndex(2, 25), 1066.0); bHM.put(new CellIndex(2, 26), 1796.0); bHM.put(new CellIndex(2, 27), 1816.0); bHM.put(new CellIndex(2, 28), 1836.0); 
		bHM.put(new CellIndex(2, 29), 1856.0); bHM.put(new CellIndex(2, 30), 750.0); bHM.put(new CellIndex(2, 31), 382.0); bHM.put(new CellIndex(2, 32), 578.0); 
		bHM.put(new CellIndex(2, 33), 584.0); bHM.put(new CellIndex(2, 34), 590.0); bHM.put(new CellIndex(2, 35), 596.0); bHM.put(new CellIndex(2, 36), 200.0); 
		bHM.put(new CellIndex(2, 37), 808.0); bHM.put(new CellIndex(2, 38), 1538.0); bHM.put(new CellIndex(2, 39), 1568.0); bHM.put(new CellIndex(2, 40), 1598.0); 
		bHM.put(new CellIndex(2, 41), 1628.0); bHM.put(new CellIndex(2, 42), 770.0); bHM.put(new CellIndex(2, 43), 1634.0); bHM.put(new CellIndex(2, 44), 3076.0); 
		bHM.put(new CellIndex(2, 45), 3128.0); bHM.put(new CellIndex(2, 46), 3180.0); bHM.put(new CellIndex(2, 47), 3232.0); bHM.put(new CellIndex(2, 48), 1510.0); 
		bHM.put(new CellIndex(2, 49), 1914.0); bHM.put(new CellIndex(2, 50), 3596.0); bHM.put(new CellIndex(2, 51), 3648.0); bHM.put(new CellIndex(2, 52), 3700.0); 
		bHM.put(new CellIndex(2, 53), 3752.0); bHM.put(new CellIndex(2, 54), 1750.0); bHM.put(new CellIndex(2, 55), 2194.0); bHM.put(new CellIndex(2, 56), 4116.0); 
		bHM.put(new CellIndex(2, 57), 4168.0); bHM.put(new CellIndex(2, 58), 4220.0); bHM.put(new CellIndex(2, 59), 4272.0); bHM.put(new CellIndex(2, 60), 1990.0); 
		bHM.put(new CellIndex(2, 61), 2474.0); bHM.put(new CellIndex(2, 62), 4636.0); bHM.put(new CellIndex(2, 63), 4688.0); bHM.put(new CellIndex(2, 64), 4740.0); 
		bHM.put(new CellIndex(2, 65), 4792.0); bHM.put(new CellIndex(2, 66), 2230.0); bHM.put(new CellIndex(2, 67), 1146.0); bHM.put(new CellIndex(2, 68), 2118.0); 
		bHM.put(new CellIndex(2, 69), 2140.0); bHM.put(new CellIndex(2, 70), 2162.0); bHM.put(new CellIndex(2, 71), 2184.0); bHM.put(new CellIndex(2, 72), 1000.0); 
		bHM.put(new CellIndex(2, 73), 1212.0); bHM.put(new CellIndex(2, 74), 2358.0); bHM.put(new CellIndex(2, 75), 2404.0); bHM.put(new CellIndex(2, 76), 2450.0); 
		bHM.put(new CellIndex(2, 77), 2496.0); bHM.put(new CellIndex(2, 78), 1210.0); bHM.put(new CellIndex(2, 79), 2562.0); bHM.put(new CellIndex(2, 80), 4956.0); 
		bHM.put(new CellIndex(2, 81), 5040.0); bHM.put(new CellIndex(2, 82), 5124.0); bHM.put(new CellIndex(2, 83), 5208.0); bHM.put(new CellIndex(2, 84), 2510.0); 
		bHM.put(new CellIndex(2, 85), 3002.0); bHM.put(new CellIndex(2, 86), 5796.0); bHM.put(new CellIndex(2, 87), 5880.0); bHM.put(new CellIndex(2, 88), 5964.0); 
		bHM.put(new CellIndex(2, 89), 6048.0); bHM.put(new CellIndex(2, 90), 2910.0); bHM.put(new CellIndex(2, 91), 3442.0); bHM.put(new CellIndex(2, 92), 6636.0); 
		bHM.put(new CellIndex(2, 93), 6720.0); bHM.put(new CellIndex(2, 94), 6804.0); bHM.put(new CellIndex(2, 95), 6888.0); bHM.put(new CellIndex(2, 96), 3310.0); 
		bHM.put(new CellIndex(2, 97), 3882.0); bHM.put(new CellIndex(2, 98), 7476.0); bHM.put(new CellIndex(2, 99), 7560.0); bHM.put(new CellIndex(2, 100), 7644.0); 
		bHM.put(new CellIndex(2, 101), 7728.0); bHM.put(new CellIndex(2, 102), 3710.0); bHM.put(new CellIndex(2, 103), 1910.0); bHM.put(new CellIndex(2, 104), 3658.0); 
		bHM.put(new CellIndex(2, 105), 3696.0); bHM.put(new CellIndex(2, 106), 3734.0); bHM.put(new CellIndex(2, 107), 3772.0); bHM.put(new CellIndex(2, 108), 1800.0); 
		bHM.put(new CellIndex(3, 1), 804.0); bHM.put(new CellIndex(3, 2), 1418.0); bHM.put(new CellIndex(3, 3), 1432.0); bHM.put(new CellIndex(3, 4), 1446.0); 
		bHM.put(new CellIndex(3, 5), 1460.0); bHM.put(new CellIndex(3, 6), 630.0); bHM.put(new CellIndex(3, 7), 1306.0); bHM.put(new CellIndex(3, 8), 2196.0); 
		bHM.put(new CellIndex(3, 9), 2216.0); bHM.put(new CellIndex(3, 10), 2236.0); bHM.put(new CellIndex(3, 11), 2256.0); bHM.put(new CellIndex(3, 12), 910.0); 
		bHM.put(new CellIndex(3, 13), 1426.0); bHM.put(new CellIndex(3, 14), 2396.0); bHM.put(new CellIndex(3, 15), 2416.0); bHM.put(new CellIndex(3, 16), 2436.0); 
		bHM.put(new CellIndex(3, 17), 2456.0); bHM.put(new CellIndex(3, 18), 990.0); bHM.put(new CellIndex(3, 19), 1546.0); bHM.put(new CellIndex(3, 20), 2596.0); 
		bHM.put(new CellIndex(3, 21), 2616.0); bHM.put(new CellIndex(3, 22), 2636.0); bHM.put(new CellIndex(3, 23), 2656.0); bHM.put(new CellIndex(3, 24), 1070.0); 
		bHM.put(new CellIndex(3, 25), 1666.0); bHM.put(new CellIndex(3, 26), 2796.0); bHM.put(new CellIndex(3, 27), 2816.0); bHM.put(new CellIndex(3, 28), 2836.0); 
		bHM.put(new CellIndex(3, 29), 2856.0); bHM.put(new CellIndex(3, 30), 1150.0); bHM.put(new CellIndex(3, 31), 582.0); bHM.put(new CellIndex(3, 32), 878.0); 
		bHM.put(new CellIndex(3, 33), 884.0); bHM.put(new CellIndex(3, 34), 890.0); bHM.put(new CellIndex(3, 35), 896.0); bHM.put(new CellIndex(3, 36), 300.0); 
		bHM.put(new CellIndex(3, 37), 1608.0); bHM.put(new CellIndex(3, 38), 3038.0); bHM.put(new CellIndex(3, 39), 3068.0); bHM.put(new CellIndex(3, 40), 3098.0); 
		bHM.put(new CellIndex(3, 41), 3128.0); bHM.put(new CellIndex(3, 42), 1470.0); bHM.put(new CellIndex(3, 43), 3034.0); bHM.put(new CellIndex(3, 44), 5676.0); 
		bHM.put(new CellIndex(3, 45), 5728.0); bHM.put(new CellIndex(3, 46), 5780.0); bHM.put(new CellIndex(3, 47), 5832.0); bHM.put(new CellIndex(3, 48), 2710.0); 
		bHM.put(new CellIndex(3, 49), 3314.0); bHM.put(new CellIndex(3, 50), 6196.0); bHM.put(new CellIndex(3, 51), 6248.0); bHM.put(new CellIndex(3, 52), 6300.0); 
		bHM.put(new CellIndex(3, 53), 6352.0); bHM.put(new CellIndex(3, 54), 2950.0); bHM.put(new CellIndex(3, 55), 3594.0); bHM.put(new CellIndex(3, 56), 6716.0); 
		bHM.put(new CellIndex(3, 57), 6768.0); bHM.put(new CellIndex(3, 58), 6820.0); bHM.put(new CellIndex(3, 59), 6872.0); bHM.put(new CellIndex(3, 60), 3190.0); 
		bHM.put(new CellIndex(3, 61), 3874.0); bHM.put(new CellIndex(3, 62), 7236.0); bHM.put(new CellIndex(3, 63), 7288.0); bHM.put(new CellIndex(3, 64), 7340.0); 
		bHM.put(new CellIndex(3, 65), 7392.0); bHM.put(new CellIndex(3, 66), 3430.0); bHM.put(new CellIndex(3, 67), 1746.0); bHM.put(new CellIndex(3, 68), 3218.0); 
		bHM.put(new CellIndex(3, 69), 3240.0); bHM.put(new CellIndex(3, 70), 3262.0); bHM.put(new CellIndex(3, 71), 3284.0); bHM.put(new CellIndex(3, 72), 1500.0); 
		bHM.put(new CellIndex(3, 73), 2412.0); bHM.put(new CellIndex(3, 74), 4658.0); bHM.put(new CellIndex(3, 75), 4704.0); bHM.put(new CellIndex(3, 76), 4750.0); 
		bHM.put(new CellIndex(3, 77), 4796.0); bHM.put(new CellIndex(3, 78), 2310.0); bHM.put(new CellIndex(3, 79), 4762.0); bHM.put(new CellIndex(3, 80), 9156.0); 
		bHM.put(new CellIndex(3, 81), 9240.0); bHM.put(new CellIndex(3, 82), 9324.0); bHM.put(new CellIndex(3, 83), 9408.0); bHM.put(new CellIndex(3, 84), 4510.0); 
		bHM.put(new CellIndex(3, 85), 5202.0); bHM.put(new CellIndex(3, 86), 9996.0); bHM.put(new CellIndex(3, 87), 10080.0); bHM.put(new CellIndex(3, 88), 10164.0); 
		bHM.put(new CellIndex(3, 89), 10248.0); bHM.put(new CellIndex(3, 90), 4910.0); bHM.put(new CellIndex(3, 91), 5642.0); bHM.put(new CellIndex(3, 92), 10836.0); 
		bHM.put(new CellIndex(3, 93), 10920.0); bHM.put(new CellIndex(3, 94), 11004.0); bHM.put(new CellIndex(3, 95), 11088.0); bHM.put(new CellIndex(3, 96), 5310.0); 
		bHM.put(new CellIndex(3, 97), 6082.0); bHM.put(new CellIndex(3, 98), 11676.0); bHM.put(new CellIndex(3, 99), 11760.0); bHM.put(new CellIndex(3, 100), 11844.0); 
		bHM.put(new CellIndex(3, 101), 11928.0); bHM.put(new CellIndex(3, 102), 5710.0); bHM.put(new CellIndex(3, 103), 2910.0); bHM.put(new CellIndex(3, 104), 5558.0); 
		bHM.put(new CellIndex(3, 105), 5596.0); bHM.put(new CellIndex(3, 106), 5634.0); bHM.put(new CellIndex(3, 107), 5672.0); bHM.put(new CellIndex(3, 108), 2700.0); 
	}
}

