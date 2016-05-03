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

public class Conv2DBackwardTest extends AutomatedTestBase
{
	
	private final static String TEST_NAME = "Conv2DBackwardTest";
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
	public void testConv2DBackwardFilterDense1() 
	{
		int numImg = 3; int imgSize = 3; int numChannels = 3; int numFilters = 1; int filterSize = 2; int stride = 1; int pad = 0;
		fillTest1HM();
		runConv2DBackwardFilterTest(ExecType.CP, imgSize, numImg, numChannels, numFilters, filterSize, stride, pad);
	}
	
	@Test
	public void testConv2DBackwardFilterDense2() 
	{
		int numImg = 3; int imgSize = 3; int numChannels = 3; int numFilters = 4; int filterSize = 2; int stride = 1; int pad = 0;
		fillTest2HM();
		runConv2DBackwardFilterTest(ExecType.CP, imgSize, numImg, numChannels, numFilters, filterSize, stride, pad);
	}
	
	@Test
	public void testConv2DBackwardFilterDense3() 
	{
		int numImg = 3; int imgSize = 10; int numChannels = 4; int numFilters = 3; int filterSize = 2; int stride = 2; int pad = 1;
		fillTest3HM();
		runConv2DBackwardFilterTest(ExecType.CP, imgSize, numImg, numChannels, numFilters, filterSize, stride, pad);
	}
	
	@Test
	public void testConv2DBackwardFilterDense4() 
	{
		int numImg = 3; int imgSize = 10; int numChannels = 4; int numFilters = 3; int filterSize = 3; int stride = 1; int pad = 1;
		fillTest4HM();
		runConv2DBackwardFilterTest(ExecType.CP, imgSize, numImg, numChannels, numFilters, filterSize, stride, pad);
	}
	
	@Test
	public void testConv2DBackwardFilterDense5() 
	{
		int numImg = 3; int imgSize = 10; int numChannels = 2; int numFilters = 3; int filterSize = 3; int stride = 3; int pad = 1;
		fillTest5HM();
		runConv2DBackwardFilterTest(ExecType.CP, imgSize, numImg, numChannels, numFilters, filterSize, stride, pad);
	}
	
	/**
	 * 
	 * @param et
	 * @param sparse
	 */
	public void runConv2DBackwardFilterTest( ExecType et, int imgSize, int numImg, int numChannels, int numFilters, 
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
			
			
			long P = ConvolutionUtils.getP(imgSize, filterSize, stride, pad);
			
			programArgs = new String[]{"-explain", "-args",  "" + imgSize, "" + numImg, 
				"" + numChannels, "" + numFilters, 
				"" + filterSize, "" + stride, "" + pad,
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
	
	private void fillTest1HM() {
		bHM.put(new CellIndex(1, 1), 3225.0); bHM.put(new CellIndex(1, 2), 3303.0); bHM.put(new CellIndex(1, 3), 3459.0); bHM.put(new CellIndex(1, 4), 3537.0); 
		bHM.put(new CellIndex(1, 5), 3927.0); bHM.put(new CellIndex(1, 6), 4005.0); bHM.put(new CellIndex(1, 7), 4161.0); bHM.put(new CellIndex(1, 8), 4239.0); 
		bHM.put(new CellIndex(1, 9), 4629.0); bHM.put(new CellIndex(1, 10), 4707.0); bHM.put(new CellIndex(1, 11), 4863.0); bHM.put(new CellIndex(1, 12), 4941.0);    
	}
	
	private void fillTest2HM() {
		bHM.put(new CellIndex(1, 1), 10137.0); bHM.put(new CellIndex(1, 2), 10359.0); bHM.put(new CellIndex(1, 3), 10803.0); bHM.put(new CellIndex(1, 4), 11025.0); 
		bHM.put(new CellIndex(1, 5), 12135.0); bHM.put(new CellIndex(1, 6), 12357.0); bHM.put(new CellIndex(1, 7), 12801.0); bHM.put(new CellIndex(1, 8), 13023.0); 
		bHM.put(new CellIndex(1, 9), 14133.0); bHM.put(new CellIndex(1, 10), 14355.0); bHM.put(new CellIndex(1, 11), 14799.0); bHM.put(new CellIndex(1, 12), 15021.0); 
		bHM.put(new CellIndex(2, 1), 11577.0); bHM.put(new CellIndex(2, 2), 11847.0); bHM.put(new CellIndex(2, 3), 12387.0); bHM.put(new CellIndex(2, 4), 12657.0); 
		bHM.put(new CellIndex(2, 5), 14007.0); bHM.put(new CellIndex(2, 6), 14277.0); bHM.put(new CellIndex(2, 7), 14817.0); bHM.put(new CellIndex(2, 8), 15087.0); 
		bHM.put(new CellIndex(2, 9), 16437.0); bHM.put(new CellIndex(2, 10), 16707.0); bHM.put(new CellIndex(2, 11), 17247.0); bHM.put(new CellIndex(2, 12), 17517.0); 
		bHM.put(new CellIndex(3, 1), 13017.0); bHM.put(new CellIndex(3, 2), 13335.0); bHM.put(new CellIndex(3, 3), 13971.0); bHM.put(new CellIndex(3, 4), 14289.0); 
		bHM.put(new CellIndex(3, 5), 15879.0); bHM.put(new CellIndex(3, 6), 16197.0); bHM.put(new CellIndex(3, 7), 16833.0); bHM.put(new CellIndex(3, 8), 17151.0); 
		bHM.put(new CellIndex(3, 9), 18741.0); bHM.put(new CellIndex(3, 10), 19059.0); bHM.put(new CellIndex(3, 11), 19695.0); bHM.put(new CellIndex(3, 12), 20013.0); 
		bHM.put(new CellIndex(4, 1), 14457.0); bHM.put(new CellIndex(4, 2), 14823.0); bHM.put(new CellIndex(4, 3), 15555.0); bHM.put(new CellIndex(4, 4), 15921.0); 
		bHM.put(new CellIndex(4, 5), 17751.0); bHM.put(new CellIndex(4, 6), 18117.0); bHM.put(new CellIndex(4, 7), 18849.0); bHM.put(new CellIndex(4, 8), 19215.0); 
		bHM.put(new CellIndex(4, 9), 21045.0); bHM.put(new CellIndex(4, 10), 21411.0); bHM.put(new CellIndex(4, 11), 22143.0); bHM.put(new CellIndex(4, 12), 22509.0); 
	}
	
	private void fillTest3HM() {
		bHM.put(new CellIndex(1, 1), 6624300.0); bHM.put(new CellIndex(1, 2), 6580425.0); bHM.put(new CellIndex(1, 3), 6326100.0); bHM.put(new CellIndex(1, 4), 6283425.0); 
		bHM.put(new CellIndex(1, 5), 7599300.0); bHM.put(new CellIndex(1, 6), 7547925.0); bHM.put(new CellIndex(1, 7), 7256100.0); bHM.put(new CellIndex(1, 8), 7205925.0); 
		bHM.put(new CellIndex(1, 9), 8574300.0); bHM.put(new CellIndex(1, 10), 8515425.0); bHM.put(new CellIndex(1, 11), 8186100.0); bHM.put(new CellIndex(1, 12), 8128425.0); 
		bHM.put(new CellIndex(1, 13), 9549300.0); bHM.put(new CellIndex(1, 14), 9482925.0); bHM.put(new CellIndex(1, 15), 9116100.0); bHM.put(new CellIndex(1, 16), 9050925.0); 
		bHM.put(new CellIndex(2, 1), 7855500.0); bHM.put(new CellIndex(2, 2), 7808925.0); bHM.put(new CellIndex(2, 3), 7530300.0); bHM.put(new CellIndex(2, 4), 7484925.0); 
		bHM.put(new CellIndex(2, 5), 9100500.0); bHM.put(new CellIndex(2, 6), 9046425.0); bHM.put(new CellIndex(2, 7), 8730300.0); bHM.put(new CellIndex(2, 8), 8677425.0); 
		bHM.put(new CellIndex(2, 9), 10345500.0); bHM.put(new CellIndex(2, 10), 10283925.0); bHM.put(new CellIndex(2, 11), 9930300.0); bHM.put(new CellIndex(2, 12), 9869925.0); 
		bHM.put(new CellIndex(2, 13), 11590500.0); bHM.put(new CellIndex(2, 14), 11521425.0); bHM.put(new CellIndex(2, 15), 11130300.0); bHM.put(new CellIndex(2, 16), 11062425.0); 
		bHM.put(new CellIndex(3, 1), 9086700.0); bHM.put(new CellIndex(3, 2), 9037425.0); bHM.put(new CellIndex(3, 3), 8734500.0); bHM.put(new CellIndex(3, 4), 8686425.0); 
		bHM.put(new CellIndex(3, 5), 10601700.0); bHM.put(new CellIndex(3, 6), 10544925.0); bHM.put(new CellIndex(3, 7), 10204500.0); bHM.put(new CellIndex(3, 8), 10148925.0); 
		bHM.put(new CellIndex(3, 9), 12116700.0); bHM.put(new CellIndex(3, 10), 12052425.0); bHM.put(new CellIndex(3, 11), 11674500.0); bHM.put(new CellIndex(3, 12), 11611425.0); 
		bHM.put(new CellIndex(3, 13), 13631700.0); bHM.put(new CellIndex(3, 14), 13559925.0); bHM.put(new CellIndex(3, 15), 13144500.0); bHM.put(new CellIndex(3, 16), 13073925.0); 
	}
	
	private void fillTest4HM() {
		bHM.put(new CellIndex(1, 1), 58099680.0); bHM.put(new CellIndex(1, 2), 64543545.0); bHM.put(new CellIndex(1, 3), 58077810.0); bHM.put(new CellIndex(1, 4), 64471050.0); 
		bHM.put(new CellIndex(1, 5), 71620050.0); bHM.put(new CellIndex(1, 6), 64444050.0); bHM.put(new CellIndex(1, 7), 57859110.0); bHM.put(new CellIndex(1, 8), 64273545.0); 
		bHM.put(new CellIndex(1, 9), 57832380.0); bHM.put(new CellIndex(1, 10), 66750480.0); bHM.put(new CellIndex(1, 11), 74142045.0); bHM.put(new CellIndex(1, 12), 66704310.0); 
		bHM.put(new CellIndex(1, 13), 73948050.0); bHM.put(new CellIndex(1, 14), 82135050.0); bHM.put(new CellIndex(1, 15), 73894050.0); bHM.put(new CellIndex(1, 16), 66266910.0); 
		bHM.put(new CellIndex(1, 17), 73602045.0); bHM.put(new CellIndex(1, 18), 66215880.0); bHM.put(new CellIndex(1, 19), 75401280.0); bHM.put(new CellIndex(1, 20), 83740545.0); 
		bHM.put(new CellIndex(1, 21), 75330810.0); bHM.put(new CellIndex(1, 22), 83425050.0); bHM.put(new CellIndex(1, 23), 92650050.0); bHM.put(new CellIndex(1, 24), 83344050.0); 
		bHM.put(new CellIndex(1, 25), 74674710.0); bHM.put(new CellIndex(1, 26), 82930545.0); bHM.put(new CellIndex(1, 27), 74599380.0); bHM.put(new CellIndex(1, 28), 84052080.0); 
		bHM.put(new CellIndex(1, 29), 93339045.0); bHM.put(new CellIndex(1, 30), 83957310.0); bHM.put(new CellIndex(1, 31), 92902050.0); bHM.put(new CellIndex(1, 32), 103165050.0); 
		bHM.put(new CellIndex(1, 33), 92794050.0); bHM.put(new CellIndex(1, 34), 83082510.0); bHM.put(new CellIndex(1, 35), 92259045.0); bHM.put(new CellIndex(1, 36), 82982880.0); 
		bHM.put(new CellIndex(2, 1), 68913180.0); bHM.put(new CellIndex(2, 2), 76572045.0); bHM.put(new CellIndex(2, 3), 68915610.0); bHM.put(new CellIndex(2, 4), 76621050.0); 
		bHM.put(new CellIndex(2, 5), 85135050.0); bHM.put(new CellIndex(2, 6), 76621050.0); bHM.put(new CellIndex(2, 7), 68915610.0); bHM.put(new CellIndex(2, 8), 76572045.0); 
		bHM.put(new CellIndex(2, 9), 68913180.0); bHM.put(new CellIndex(2, 10), 79993980.0); bHM.put(new CellIndex(2, 11), 88870545.0); bHM.put(new CellIndex(2, 12), 79972110.0); 
		bHM.put(new CellIndex(2, 13), 88798050.0); bHM.put(new CellIndex(2, 14), 98650050.0); bHM.put(new CellIndex(2, 15), 88771050.0); bHM.put(new CellIndex(2, 16), 79753410.0); 
		bHM.put(new CellIndex(2, 17), 88600545.0); bHM.put(new CellIndex(2, 18), 79726680.0); bHM.put(new CellIndex(2, 19), 91074780.0); bHM.put(new CellIndex(2, 20), 101169045.0); 
		bHM.put(new CellIndex(2, 21), 91028610.0); bHM.put(new CellIndex(2, 22), 100975050.0); bHM.put(new CellIndex(2, 23), 112165050.0); bHM.put(new CellIndex(2, 24), 100921050.0); 
		bHM.put(new CellIndex(2, 25), 90591210.0); bHM.put(new CellIndex(2, 26), 100629045.0); bHM.put(new CellIndex(2, 27), 90540180.0); bHM.put(new CellIndex(2, 28), 102155580.0); 
		bHM.put(new CellIndex(2, 29), 113467545.0); bHM.put(new CellIndex(2, 30), 102085110.0); bHM.put(new CellIndex(2, 31), 113152050.0); bHM.put(new CellIndex(2, 32), 125680050.0); 
		bHM.put(new CellIndex(2, 33), 113071050.0); bHM.put(new CellIndex(2, 34), 101429010.0); bHM.put(new CellIndex(2, 35), 112657545.0); bHM.put(new CellIndex(2, 36), 101353680.0); 
		bHM.put(new CellIndex(3, 1), 79726680.0); bHM.put(new CellIndex(3, 2), 88600545.0); bHM.put(new CellIndex(3, 3), 79753410.0); bHM.put(new CellIndex(3, 4), 88771050.0); 
		bHM.put(new CellIndex(3, 5), 98650050.0); bHM.put(new CellIndex(3, 6), 88798050.0); bHM.put(new CellIndex(3, 7), 79972110.0); bHM.put(new CellIndex(3, 8), 88870545.0); 
		bHM.put(new CellIndex(3, 9), 79993980.0); bHM.put(new CellIndex(3, 10), 93237480.0); bHM.put(new CellIndex(3, 11), 103599045.0); bHM.put(new CellIndex(3, 12), 93239910.0); 
		bHM.put(new CellIndex(3, 13), 103648050.0); bHM.put(new CellIndex(3, 14), 115165050.0); bHM.put(new CellIndex(3, 15), 103648050.0); bHM.put(new CellIndex(3, 16), 93239910.0); 
		bHM.put(new CellIndex(3, 17), 103599045.0); bHM.put(new CellIndex(3, 18), 93237480.0); bHM.put(new CellIndex(3, 19), 106748280.0); bHM.put(new CellIndex(3, 20), 118597545.0); 
		bHM.put(new CellIndex(3, 21), 106726410.0); bHM.put(new CellIndex(3, 22), 118525050.0); bHM.put(new CellIndex(3, 23), 131680050.0); bHM.put(new CellIndex(3, 24), 118498050.0); 
		bHM.put(new CellIndex(3, 25), 106507710.0); bHM.put(new CellIndex(3, 26), 118327545.0); bHM.put(new CellIndex(3, 27), 106480980.0); bHM.put(new CellIndex(3, 28), 120259080.0); 
		bHM.put(new CellIndex(3, 29), 133596045.0); bHM.put(new CellIndex(3, 30), 120212910.0); bHM.put(new CellIndex(3, 31), 133402050.0); bHM.put(new CellIndex(3, 32), 148195050.0); 
		bHM.put(new CellIndex(3, 33), 133348050.0); bHM.put(new CellIndex(3, 34), 119775510.0); bHM.put(new CellIndex(3, 35), 133056045.0); bHM.put(new CellIndex(3, 36), 119724480.0); 
	}
	
	private void fillTest5HM() {
		bHM.put(new CellIndex(1, 1), 582822.0); bHM.put(new CellIndex(1, 2), 771498.0); bHM.put(new CellIndex(1, 3), 574344.0); bHM.put(new CellIndex(1, 4), 750924.0); 
		bHM.put(new CellIndex(1, 5), 993936.0); bHM.put(new CellIndex(1, 6), 739872.0); bHM.put(new CellIndex(1, 7), 540324.0); bHM.put(new CellIndex(1, 8), 715086.0); 
		bHM.put(new CellIndex(1, 9), 532224.0); bHM.put(new CellIndex(1, 10), 742122.0); bHM.put(new CellIndex(1, 11), 982098.0); bHM.put(new CellIndex(1, 12), 730944.0); 
		bHM.put(new CellIndex(1, 13), 956124.0); bHM.put(new CellIndex(1, 14), 1265136.0); bHM.put(new CellIndex(1, 15), 941472.0); bHM.put(new CellIndex(1, 16), 688824.0); 
		bHM.put(new CellIndex(1, 17), 911286.0); bHM.put(new CellIndex(1, 18), 678024.0); bHM.put(new CellIndex(2, 1), 693414.0); bHM.put(new CellIndex(2, 2), 918666.0); 
		bHM.put(new CellIndex(2, 3), 684504.0); bHM.put(new CellIndex(2, 4), 895500.0); bHM.put(new CellIndex(2, 5), 1186320.0); bHM.put(new CellIndex(2, 6), 883872.0); 
		bHM.put(new CellIndex(2, 7), 646596.0); bHM.put(new CellIndex(2, 8), 856494.0); bHM.put(new CellIndex(2, 9), 638064.0); bHM.put(new CellIndex(2, 10), 895914.0); 
		bHM.put(new CellIndex(2, 11), 1186866.0); bHM.put(new CellIndex(2, 12), 884304.0); bHM.put(new CellIndex(2, 13), 1158300.0); bHM.put(new CellIndex(2, 14), 1534320.0); 
		bHM.put(new CellIndex(2, 15), 1143072.0); bHM.put(new CellIndex(2, 16), 838296.0); bHM.put(new CellIndex(2, 17), 1110294.0); bHM.put(new CellIndex(2, 18), 827064.0); 
		bHM.put(new CellIndex(3, 1), 804006.0); bHM.put(new CellIndex(3, 2), 1065834.0); bHM.put(new CellIndex(3, 3), 794664.0); bHM.put(new CellIndex(3, 4), 1040076.0); 
		bHM.put(new CellIndex(3, 5), 1378704.0); bHM.put(new CellIndex(3, 6), 1027872.0); bHM.put(new CellIndex(3, 7), 752868.0); bHM.put(new CellIndex(3, 8), 997902.0); 
		bHM.put(new CellIndex(3, 9), 743904.0); bHM.put(new CellIndex(3, 10), 1049706.0); bHM.put(new CellIndex(3, 11), 1391634.0); bHM.put(new CellIndex(3, 12), 1037664.0); 
		bHM.put(new CellIndex(3, 13), 1360476.0); bHM.put(new CellIndex(3, 14), 1803504.0); bHM.put(new CellIndex(3, 15), 1344672.0); bHM.put(new CellIndex(3, 16), 987768.0); 
		bHM.put(new CellIndex(3, 17), 1309302.0); bHM.put(new CellIndex(3, 18), 976104.0); 
	}
}
