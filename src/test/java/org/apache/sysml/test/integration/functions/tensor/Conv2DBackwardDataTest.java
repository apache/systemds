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

public class Conv2DBackwardDataTest extends AutomatedTestBase
{
	
	private final static String TEST_NAME = "Conv2DBackwardDataTest";
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
		int numImg = 2; int imgSize = 10; int numChannels = 3; int numFilters = 2; int filterSize = 2; int stride = 1; int pad = 0;
		fillTest1HM();
		runConv2DTest(ExecType.CP, imgSize, numImg, numChannels, numFilters, filterSize, stride, pad);
	}
	
	@Test
	public void testConv2DDense2() 
	{
		int numImg = 5; int imgSize = 3; int numChannels = 2; int numFilters = 3; int filterSize = 3; int stride = 1; int pad = 1;
		fillTest2HM();
		runConv2DTest(ExecType.CP, imgSize, numImg, numChannels, numFilters, filterSize, stride, pad);
	}
	
	@Test
	public void testConv2DDense3() 
	{
		int numImg = 5; int imgSize = 3; int numChannels = 2; int numFilters = 3; int filterSize = 3; int stride = 2; int pad = 1;
		fillTest3HM();
		runConv2DTest(ExecType.CP, imgSize, numImg, numChannels, numFilters, filterSize, stride, pad);
	}
	
	@Test
	public void testConv2DDense4() 
	{
		int numImg = 5; int imgSize = 10; int numChannels = 2; int numFilters = 3; int filterSize = 2; int stride = 2; int pad = 1;
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
		bHM.put(new CellIndex(1, 1), 1067.0); bHM.put(new CellIndex(1, 2), 2231.0); bHM.put(new CellIndex(1, 3), 2261.0); bHM.put(new CellIndex(1, 4), 2291.0); 
		bHM.put(new CellIndex(1, 5), 2321.0); bHM.put(new CellIndex(1, 6), 2351.0); bHM.put(new CellIndex(1, 7), 2381.0); bHM.put(new CellIndex(1, 8), 2411.0); 
		bHM.put(new CellIndex(1, 9), 2441.0); bHM.put(new CellIndex(1, 10), 1278.0); bHM.put(new CellIndex(1, 11), 2426.0); bHM.put(new CellIndex(1, 12), 5068.0); 
		bHM.put(new CellIndex(1, 13), 5136.0); bHM.put(new CellIndex(1, 14), 5204.0); bHM.put(new CellIndex(1, 15), 5272.0); bHM.put(new CellIndex(1, 16), 5340.0); 
		bHM.put(new CellIndex(1, 17), 5408.0); bHM.put(new CellIndex(1, 18), 5476.0); bHM.put(new CellIndex(1, 19), 5544.0); bHM.put(new CellIndex(1, 20), 2898.0); 
		bHM.put(new CellIndex(1, 21), 2714.0); bHM.put(new CellIndex(1, 22), 5680.0); bHM.put(new CellIndex(1, 23), 5748.0); bHM.put(new CellIndex(1, 24), 5816.0); 
		bHM.put(new CellIndex(1, 25), 5884.0); bHM.put(new CellIndex(1, 26), 5952.0); bHM.put(new CellIndex(1, 27), 6020.0); bHM.put(new CellIndex(1, 28), 6088.0); 
		bHM.put(new CellIndex(1, 29), 6156.0); bHM.put(new CellIndex(1, 30), 3222.0); bHM.put(new CellIndex(1, 31), 3002.0); bHM.put(new CellIndex(1, 32), 6292.0); 
		bHM.put(new CellIndex(1, 33), 6360.0); bHM.put(new CellIndex(1, 34), 6428.0); bHM.put(new CellIndex(1, 35), 6496.0); bHM.put(new CellIndex(1, 36), 6564.0); 
		bHM.put(new CellIndex(1, 37), 6632.0); bHM.put(new CellIndex(1, 38), 6700.0); bHM.put(new CellIndex(1, 39), 6768.0); bHM.put(new CellIndex(1, 40), 3546.0); 
		bHM.put(new CellIndex(1, 41), 3290.0); bHM.put(new CellIndex(1, 42), 6904.0); bHM.put(new CellIndex(1, 43), 6972.0); bHM.put(new CellIndex(1, 44), 7040.0); 
		bHM.put(new CellIndex(1, 45), 7108.0); bHM.put(new CellIndex(1, 46), 7176.0); bHM.put(new CellIndex(1, 47), 7244.0); bHM.put(new CellIndex(1, 48), 7312.0); 
		bHM.put(new CellIndex(1, 49), 7380.0); bHM.put(new CellIndex(1, 50), 3870.0); bHM.put(new CellIndex(1, 51), 3578.0); bHM.put(new CellIndex(1, 52), 7516.0); 
		bHM.put(new CellIndex(1, 53), 7584.0); bHM.put(new CellIndex(1, 54), 7652.0); bHM.put(new CellIndex(1, 55), 7720.0); bHM.put(new CellIndex(1, 56), 7788.0); 
		bHM.put(new CellIndex(1, 57), 7856.0); bHM.put(new CellIndex(1, 58), 7924.0); bHM.put(new CellIndex(1, 59), 7992.0); bHM.put(new CellIndex(1, 60), 4194.0); 
		bHM.put(new CellIndex(1, 61), 3866.0); bHM.put(new CellIndex(1, 62), 8128.0); bHM.put(new CellIndex(1, 63), 8196.0); bHM.put(new CellIndex(1, 64), 8264.0); 
		bHM.put(new CellIndex(1, 65), 8332.0); bHM.put(new CellIndex(1, 66), 8400.0); bHM.put(new CellIndex(1, 67), 8468.0); bHM.put(new CellIndex(1, 68), 8536.0); 
		bHM.put(new CellIndex(1, 69), 8604.0); bHM.put(new CellIndex(1, 70), 4518.0); bHM.put(new CellIndex(1, 71), 4154.0); bHM.put(new CellIndex(1, 72), 8740.0); 
		bHM.put(new CellIndex(1, 73), 8808.0); bHM.put(new CellIndex(1, 74), 8876.0); bHM.put(new CellIndex(1, 75), 8944.0); bHM.put(new CellIndex(1, 76), 9012.0); 
		bHM.put(new CellIndex(1, 77), 9080.0); bHM.put(new CellIndex(1, 78), 9148.0); bHM.put(new CellIndex(1, 79), 9216.0); bHM.put(new CellIndex(1, 80), 4842.0); 
		bHM.put(new CellIndex(1, 81), 4442.0); bHM.put(new CellIndex(1, 82), 9352.0); bHM.put(new CellIndex(1, 83), 9420.0); bHM.put(new CellIndex(1, 84), 9488.0); 
		bHM.put(new CellIndex(1, 85), 9556.0); bHM.put(new CellIndex(1, 86), 9624.0); bHM.put(new CellIndex(1, 87), 9692.0); bHM.put(new CellIndex(1, 88), 9760.0); 
		bHM.put(new CellIndex(1, 89), 9828.0); bHM.put(new CellIndex(1, 90), 5166.0); bHM.put(new CellIndex(1, 91), 2529.0); bHM.put(new CellIndex(1, 92), 5303.0); 
		bHM.put(new CellIndex(1, 93), 5341.0); bHM.put(new CellIndex(1, 94), 5379.0); bHM.put(new CellIndex(1, 95), 5417.0); bHM.put(new CellIndex(1, 96), 5455.0); 
		bHM.put(new CellIndex(1, 97), 5493.0); bHM.put(new CellIndex(1, 98), 5531.0); bHM.put(new CellIndex(1, 99), 5569.0); bHM.put(new CellIndex(1, 100), 2916.0); 
		bHM.put(new CellIndex(1, 101), 1399.0); bHM.put(new CellIndex(1, 102), 2903.0); bHM.put(new CellIndex(1, 103), 2949.0); bHM.put(new CellIndex(1, 104), 2995.0); 
		bHM.put(new CellIndex(1, 105), 3041.0); bHM.put(new CellIndex(1, 106), 3087.0); bHM.put(new CellIndex(1, 107), 3133.0); bHM.put(new CellIndex(1, 108), 3179.0); 
		bHM.put(new CellIndex(1, 109), 3225.0); bHM.put(new CellIndex(1, 110), 1674.0); bHM.put(new CellIndex(1, 111), 3162.0); bHM.put(new CellIndex(1, 112), 6556.0); 
		bHM.put(new CellIndex(1, 113), 6656.0); bHM.put(new CellIndex(1, 114), 6756.0); bHM.put(new CellIndex(1, 115), 6856.0); bHM.put(new CellIndex(1, 116), 6956.0); 
		bHM.put(new CellIndex(1, 117), 7056.0); bHM.put(new CellIndex(1, 118), 7156.0); bHM.put(new CellIndex(1, 119), 7256.0); bHM.put(new CellIndex(1, 120), 3762.0); 
		bHM.put(new CellIndex(1, 121), 3594.0); bHM.put(new CellIndex(1, 122), 7456.0); bHM.put(new CellIndex(1, 123), 7556.0); bHM.put(new CellIndex(1, 124), 7656.0); 
		bHM.put(new CellIndex(1, 125), 7756.0); bHM.put(new CellIndex(1, 126), 7856.0); bHM.put(new CellIndex(1, 127), 7956.0); bHM.put(new CellIndex(1, 128), 8056.0); 
		bHM.put(new CellIndex(1, 129), 8156.0); bHM.put(new CellIndex(1, 130), 4230.0); bHM.put(new CellIndex(1, 131), 4026.0); bHM.put(new CellIndex(1, 132), 8356.0); 
		bHM.put(new CellIndex(1, 133), 8456.0); bHM.put(new CellIndex(1, 134), 8556.0); bHM.put(new CellIndex(1, 135), 8656.0); bHM.put(new CellIndex(1, 136), 8756.0); 
		bHM.put(new CellIndex(1, 137), 8856.0); bHM.put(new CellIndex(1, 138), 8956.0); bHM.put(new CellIndex(1, 139), 9056.0); bHM.put(new CellIndex(1, 140), 4698.0); 
		bHM.put(new CellIndex(1, 141), 4458.0); bHM.put(new CellIndex(1, 142), 9256.0); bHM.put(new CellIndex(1, 143), 9356.0); bHM.put(new CellIndex(1, 144), 9456.0); 
		bHM.put(new CellIndex(1, 145), 9556.0); bHM.put(new CellIndex(1, 146), 9656.0); bHM.put(new CellIndex(1, 147), 9756.0); bHM.put(new CellIndex(1, 148), 9856.0); 
		bHM.put(new CellIndex(1, 149), 9956.0); bHM.put(new CellIndex(1, 150), 5166.0); bHM.put(new CellIndex(1, 151), 4890.0); bHM.put(new CellIndex(1, 152), 10156.0); 
		bHM.put(new CellIndex(1, 153), 10256.0); bHM.put(new CellIndex(1, 154), 10356.0); bHM.put(new CellIndex(1, 155), 10456.0); bHM.put(new CellIndex(1, 156), 10556.0); 
		bHM.put(new CellIndex(1, 157), 10656.0); bHM.put(new CellIndex(1, 158), 10756.0); bHM.put(new CellIndex(1, 159), 10856.0); bHM.put(new CellIndex(1, 160), 5634.0); 
		bHM.put(new CellIndex(1, 161), 5322.0); bHM.put(new CellIndex(1, 162), 11056.0); bHM.put(new CellIndex(1, 163), 11156.0); bHM.put(new CellIndex(1, 164), 11256.0); 
		bHM.put(new CellIndex(1, 165), 11356.0); bHM.put(new CellIndex(1, 166), 11456.0); bHM.put(new CellIndex(1, 167), 11556.0); bHM.put(new CellIndex(1, 168), 11656.0); 
		bHM.put(new CellIndex(1, 169), 11756.0); bHM.put(new CellIndex(1, 170), 6102.0); bHM.put(new CellIndex(1, 171), 5754.0); bHM.put(new CellIndex(1, 172), 11956.0); 
		bHM.put(new CellIndex(1, 173), 12056.0); bHM.put(new CellIndex(1, 174), 12156.0); bHM.put(new CellIndex(1, 175), 12256.0); bHM.put(new CellIndex(1, 176), 12356.0); 
		bHM.put(new CellIndex(1, 177), 12456.0); bHM.put(new CellIndex(1, 178), 12556.0); bHM.put(new CellIndex(1, 179), 12656.0); bHM.put(new CellIndex(1, 180), 6570.0); 
		bHM.put(new CellIndex(1, 181), 6186.0); bHM.put(new CellIndex(1, 182), 12856.0); bHM.put(new CellIndex(1, 183), 12956.0); bHM.put(new CellIndex(1, 184), 13056.0); 
		bHM.put(new CellIndex(1, 185), 13156.0); bHM.put(new CellIndex(1, 186), 13256.0); bHM.put(new CellIndex(1, 187), 13356.0); bHM.put(new CellIndex(1, 188), 13456.0); 
		bHM.put(new CellIndex(1, 189), 13556.0); bHM.put(new CellIndex(1, 190), 7038.0); bHM.put(new CellIndex(1, 191), 3437.0); bHM.put(new CellIndex(1, 192), 7127.0); 
		bHM.put(new CellIndex(1, 193), 7181.0); bHM.put(new CellIndex(1, 194), 7235.0); bHM.put(new CellIndex(1, 195), 7289.0); bHM.put(new CellIndex(1, 196), 7343.0); 
		bHM.put(new CellIndex(1, 197), 7397.0); bHM.put(new CellIndex(1, 198), 7451.0); bHM.put(new CellIndex(1, 199), 7505.0); bHM.put(new CellIndex(1, 200), 3888.0); 
		bHM.put(new CellIndex(1, 201), 1731.0); bHM.put(new CellIndex(1, 202), 3575.0); bHM.put(new CellIndex(1, 203), 3637.0); bHM.put(new CellIndex(1, 204), 3699.0); 
		bHM.put(new CellIndex(1, 205), 3761.0); bHM.put(new CellIndex(1, 206), 3823.0); bHM.put(new CellIndex(1, 207), 3885.0); bHM.put(new CellIndex(1, 208), 3947.0); 
		bHM.put(new CellIndex(1, 209), 4009.0); bHM.put(new CellIndex(1, 210), 2070.0); bHM.put(new CellIndex(1, 211), 3898.0); bHM.put(new CellIndex(1, 212), 8044.0); 
		bHM.put(new CellIndex(1, 213), 8176.0); bHM.put(new CellIndex(1, 214), 8308.0); bHM.put(new CellIndex(1, 215), 8440.0); bHM.put(new CellIndex(1, 216), 8572.0); 
		bHM.put(new CellIndex(1, 217), 8704.0); bHM.put(new CellIndex(1, 218), 8836.0); bHM.put(new CellIndex(1, 219), 8968.0); bHM.put(new CellIndex(1, 220), 4626.0); 
		bHM.put(new CellIndex(1, 221), 4474.0); bHM.put(new CellIndex(1, 222), 9232.0); bHM.put(new CellIndex(1, 223), 9364.0); bHM.put(new CellIndex(1, 224), 9496.0); 
		bHM.put(new CellIndex(1, 225), 9628.0); bHM.put(new CellIndex(1, 226), 9760.0); bHM.put(new CellIndex(1, 227), 9892.0); bHM.put(new CellIndex(1, 228), 10024.0); 
		bHM.put(new CellIndex(1, 229), 10156.0); bHM.put(new CellIndex(1, 230), 5238.0); bHM.put(new CellIndex(1, 231), 5050.0); bHM.put(new CellIndex(1, 232), 10420.0); 
		bHM.put(new CellIndex(1, 233), 10552.0); bHM.put(new CellIndex(1, 234), 10684.0); bHM.put(new CellIndex(1, 235), 10816.0); bHM.put(new CellIndex(1, 236), 10948.0); 
		bHM.put(new CellIndex(1, 237), 11080.0); bHM.put(new CellIndex(1, 238), 11212.0); bHM.put(new CellIndex(1, 239), 11344.0); bHM.put(new CellIndex(1, 240), 5850.0); 
		bHM.put(new CellIndex(1, 241), 5626.0); bHM.put(new CellIndex(1, 242), 11608.0); bHM.put(new CellIndex(1, 243), 11740.0); bHM.put(new CellIndex(1, 244), 11872.0); 
		bHM.put(new CellIndex(1, 245), 12004.0); bHM.put(new CellIndex(1, 246), 12136.0); bHM.put(new CellIndex(1, 247), 12268.0); bHM.put(new CellIndex(1, 248), 12400.0); 
		bHM.put(new CellIndex(1, 249), 12532.0); bHM.put(new CellIndex(1, 250), 6462.0); bHM.put(new CellIndex(1, 251), 6202.0); bHM.put(new CellIndex(1, 252), 12796.0); 
		bHM.put(new CellIndex(1, 253), 12928.0); bHM.put(new CellIndex(1, 254), 13060.0); bHM.put(new CellIndex(1, 255), 13192.0); bHM.put(new CellIndex(1, 256), 13324.0); 
		bHM.put(new CellIndex(1, 257), 13456.0); bHM.put(new CellIndex(1, 258), 13588.0); bHM.put(new CellIndex(1, 259), 13720.0); bHM.put(new CellIndex(1, 260), 7074.0); 
		bHM.put(new CellIndex(1, 261), 6778.0); bHM.put(new CellIndex(1, 262), 13984.0); bHM.put(new CellIndex(1, 263), 14116.0); bHM.put(new CellIndex(1, 264), 14248.0); 
		bHM.put(new CellIndex(1, 265), 14380.0); bHM.put(new CellIndex(1, 266), 14512.0); bHM.put(new CellIndex(1, 267), 14644.0); bHM.put(new CellIndex(1, 268), 14776.0); 
		bHM.put(new CellIndex(1, 269), 14908.0); bHM.put(new CellIndex(1, 270), 7686.0); bHM.put(new CellIndex(1, 271), 7354.0); bHM.put(new CellIndex(1, 272), 15172.0); 
		bHM.put(new CellIndex(1, 273), 15304.0); bHM.put(new CellIndex(1, 274), 15436.0); bHM.put(new CellIndex(1, 275), 15568.0); bHM.put(new CellIndex(1, 276), 15700.0); 
		bHM.put(new CellIndex(1, 277), 15832.0); bHM.put(new CellIndex(1, 278), 15964.0); bHM.put(new CellIndex(1, 279), 16096.0); bHM.put(new CellIndex(1, 280), 8298.0); 
		bHM.put(new CellIndex(1, 281), 7930.0); bHM.put(new CellIndex(1, 282), 16360.0); bHM.put(new CellIndex(1, 283), 16492.0); bHM.put(new CellIndex(1, 284), 16624.0); 
		bHM.put(new CellIndex(1, 285), 16756.0); bHM.put(new CellIndex(1, 286), 16888.0); bHM.put(new CellIndex(1, 287), 17020.0); bHM.put(new CellIndex(1, 288), 17152.0); 
		bHM.put(new CellIndex(1, 289), 17284.0); bHM.put(new CellIndex(1, 290), 8910.0); bHM.put(new CellIndex(1, 291), 4345.0); bHM.put(new CellIndex(1, 292), 8951.0); 
		bHM.put(new CellIndex(1, 293), 9021.0); bHM.put(new CellIndex(1, 294), 9091.0); bHM.put(new CellIndex(1, 295), 9161.0); bHM.put(new CellIndex(1, 296), 9231.0); 
		bHM.put(new CellIndex(1, 297), 9301.0); bHM.put(new CellIndex(1, 298), 9371.0); bHM.put(new CellIndex(1, 299), 9441.0); bHM.put(new CellIndex(1, 300), 4860.0); 
		bHM.put(new CellIndex(2, 1), 3335.0); bHM.put(new CellIndex(2, 2), 7091.0); bHM.put(new CellIndex(2, 3), 7121.0); bHM.put(new CellIndex(2, 4), 7151.0); 
		bHM.put(new CellIndex(2, 5), 7181.0); bHM.put(new CellIndex(2, 6), 7211.0); bHM.put(new CellIndex(2, 7), 7241.0); bHM.put(new CellIndex(2, 8), 7271.0); 
		bHM.put(new CellIndex(2, 9), 7301.0); bHM.put(new CellIndex(2, 10), 3870.0); bHM.put(new CellIndex(2, 11), 7610.0); bHM.put(new CellIndex(2, 12), 16084.0); 
		bHM.put(new CellIndex(2, 13), 16152.0); bHM.put(new CellIndex(2, 14), 16220.0); bHM.put(new CellIndex(2, 15), 16288.0); bHM.put(new CellIndex(2, 16), 16356.0); 
		bHM.put(new CellIndex(2, 17), 16424.0); bHM.put(new CellIndex(2, 18), 16492.0); bHM.put(new CellIndex(2, 19), 16560.0); bHM.put(new CellIndex(2, 20), 8730.0); 
		bHM.put(new CellIndex(2, 21), 7898.0); bHM.put(new CellIndex(2, 22), 16696.0); bHM.put(new CellIndex(2, 23), 16764.0); bHM.put(new CellIndex(2, 24), 16832.0); 
		bHM.put(new CellIndex(2, 25), 16900.0); bHM.put(new CellIndex(2, 26), 16968.0); bHM.put(new CellIndex(2, 27), 17036.0); bHM.put(new CellIndex(2, 28), 17104.0); 
		bHM.put(new CellIndex(2, 29), 17172.0); bHM.put(new CellIndex(2, 30), 9054.0); bHM.put(new CellIndex(2, 31), 8186.0); bHM.put(new CellIndex(2, 32), 17308.0); 
		bHM.put(new CellIndex(2, 33), 17376.0); bHM.put(new CellIndex(2, 34), 17444.0); bHM.put(new CellIndex(2, 35), 17512.0); bHM.put(new CellIndex(2, 36), 17580.0); 
		bHM.put(new CellIndex(2, 37), 17648.0); bHM.put(new CellIndex(2, 38), 17716.0); bHM.put(new CellIndex(2, 39), 17784.0); bHM.put(new CellIndex(2, 40), 9378.0); 
		bHM.put(new CellIndex(2, 41), 8474.0); bHM.put(new CellIndex(2, 42), 17920.0); bHM.put(new CellIndex(2, 43), 17988.0); bHM.put(new CellIndex(2, 44), 18056.0); 
		bHM.put(new CellIndex(2, 45), 18124.0); bHM.put(new CellIndex(2, 46), 18192.0); bHM.put(new CellIndex(2, 47), 18260.0); bHM.put(new CellIndex(2, 48), 18328.0); 
		bHM.put(new CellIndex(2, 49), 18396.0); bHM.put(new CellIndex(2, 50), 9702.0); bHM.put(new CellIndex(2, 51), 8762.0); bHM.put(new CellIndex(2, 52), 18532.0); 
		bHM.put(new CellIndex(2, 53), 18600.0); bHM.put(new CellIndex(2, 54), 18668.0); bHM.put(new CellIndex(2, 55), 18736.0); bHM.put(new CellIndex(2, 56), 18804.0); 
		bHM.put(new CellIndex(2, 57), 18872.0); bHM.put(new CellIndex(2, 58), 18940.0); bHM.put(new CellIndex(2, 59), 19008.0); bHM.put(new CellIndex(2, 60), 10026.0); 
		bHM.put(new CellIndex(2, 61), 9050.0); bHM.put(new CellIndex(2, 62), 19144.0); bHM.put(new CellIndex(2, 63), 19212.0); bHM.put(new CellIndex(2, 64), 19280.0); 
		bHM.put(new CellIndex(2, 65), 19348.0); bHM.put(new CellIndex(2, 66), 19416.0); bHM.put(new CellIndex(2, 67), 19484.0); bHM.put(new CellIndex(2, 68), 19552.0); 
		bHM.put(new CellIndex(2, 69), 19620.0); bHM.put(new CellIndex(2, 70), 10350.0); bHM.put(new CellIndex(2, 71), 9338.0); bHM.put(new CellIndex(2, 72), 19756.0); 
		bHM.put(new CellIndex(2, 73), 19824.0); bHM.put(new CellIndex(2, 74), 19892.0); bHM.put(new CellIndex(2, 75), 19960.0); bHM.put(new CellIndex(2, 76), 20028.0); 
		bHM.put(new CellIndex(2, 77), 20096.0); bHM.put(new CellIndex(2, 78), 20164.0); bHM.put(new CellIndex(2, 79), 20232.0); bHM.put(new CellIndex(2, 80), 10674.0); 
		bHM.put(new CellIndex(2, 81), 9626.0); bHM.put(new CellIndex(2, 82), 20368.0); bHM.put(new CellIndex(2, 83), 20436.0); bHM.put(new CellIndex(2, 84), 20504.0); 
		bHM.put(new CellIndex(2, 85), 20572.0); bHM.put(new CellIndex(2, 86), 20640.0); bHM.put(new CellIndex(2, 87), 20708.0); bHM.put(new CellIndex(2, 88), 20776.0); 
		bHM.put(new CellIndex(2, 89), 20844.0); bHM.put(new CellIndex(2, 90), 10998.0); bHM.put(new CellIndex(2, 91), 5445.0); bHM.put(new CellIndex(2, 92), 11459.0); 
		bHM.put(new CellIndex(2, 93), 11497.0); bHM.put(new CellIndex(2, 94), 11535.0); bHM.put(new CellIndex(2, 95), 11573.0); bHM.put(new CellIndex(2, 96), 11611.0); 
		bHM.put(new CellIndex(2, 97), 11649.0); bHM.put(new CellIndex(2, 98), 11687.0); bHM.put(new CellIndex(2, 99), 11725.0); bHM.put(new CellIndex(2, 100), 6156.0); 
		bHM.put(new CellIndex(2, 101), 4963.0); bHM.put(new CellIndex(2, 102), 10355.0); bHM.put(new CellIndex(2, 103), 10401.0); bHM.put(new CellIndex(2, 104), 10447.0); 
		bHM.put(new CellIndex(2, 105), 10493.0); bHM.put(new CellIndex(2, 106), 10539.0); bHM.put(new CellIndex(2, 107), 10585.0); bHM.put(new CellIndex(2, 108), 10631.0); 
		bHM.put(new CellIndex(2, 109), 10677.0); bHM.put(new CellIndex(2, 110), 5562.0); bHM.put(new CellIndex(2, 111), 10938.0); bHM.put(new CellIndex(2, 112), 22756.0); 
		bHM.put(new CellIndex(2, 113), 22856.0); bHM.put(new CellIndex(2, 114), 22956.0); bHM.put(new CellIndex(2, 115), 23056.0); bHM.put(new CellIndex(2, 116), 23156.0); 
		bHM.put(new CellIndex(2, 117), 23256.0); bHM.put(new CellIndex(2, 118), 23356.0); bHM.put(new CellIndex(2, 119), 23456.0); bHM.put(new CellIndex(2, 120), 12186.0); 
		bHM.put(new CellIndex(2, 121), 11370.0); bHM.put(new CellIndex(2, 122), 23656.0); bHM.put(new CellIndex(2, 123), 23756.0); bHM.put(new CellIndex(2, 124), 23856.0); 
		bHM.put(new CellIndex(2, 125), 23956.0); bHM.put(new CellIndex(2, 126), 24056.0); bHM.put(new CellIndex(2, 127), 24156.0); bHM.put(new CellIndex(2, 128), 24256.0); 
		bHM.put(new CellIndex(2, 129), 24356.0); bHM.put(new CellIndex(2, 130), 12654.0); bHM.put(new CellIndex(2, 131), 11802.0); bHM.put(new CellIndex(2, 132), 24556.0); 
		bHM.put(new CellIndex(2, 133), 24656.0); bHM.put(new CellIndex(2, 134), 24756.0); bHM.put(new CellIndex(2, 135), 24856.0); bHM.put(new CellIndex(2, 136), 24956.0); 
		bHM.put(new CellIndex(2, 137), 25056.0); bHM.put(new CellIndex(2, 138), 25156.0); bHM.put(new CellIndex(2, 139), 25256.0); bHM.put(new CellIndex(2, 140), 13122.0); 
		bHM.put(new CellIndex(2, 141), 12234.0); bHM.put(new CellIndex(2, 142), 25456.0); bHM.put(new CellIndex(2, 143), 25556.0); bHM.put(new CellIndex(2, 144), 25656.0); 
		bHM.put(new CellIndex(2, 145), 25756.0); bHM.put(new CellIndex(2, 146), 25856.0); bHM.put(new CellIndex(2, 147), 25956.0); bHM.put(new CellIndex(2, 148), 26056.0); 
		bHM.put(new CellIndex(2, 149), 26156.0); bHM.put(new CellIndex(2, 150), 13590.0); bHM.put(new CellIndex(2, 151), 12666.0); bHM.put(new CellIndex(2, 152), 26356.0); 
		bHM.put(new CellIndex(2, 153), 26456.0); bHM.put(new CellIndex(2, 154), 26556.0); bHM.put(new CellIndex(2, 155), 26656.0); bHM.put(new CellIndex(2, 156), 26756.0); 
		bHM.put(new CellIndex(2, 157), 26856.0); bHM.put(new CellIndex(2, 158), 26956.0); bHM.put(new CellIndex(2, 159), 27056.0); bHM.put(new CellIndex(2, 160), 14058.0); 
		bHM.put(new CellIndex(2, 161), 13098.0); bHM.put(new CellIndex(2, 162), 27256.0); bHM.put(new CellIndex(2, 163), 27356.0); bHM.put(new CellIndex(2, 164), 27456.0); 
		bHM.put(new CellIndex(2, 165), 27556.0); bHM.put(new CellIndex(2, 166), 27656.0); bHM.put(new CellIndex(2, 167), 27756.0); bHM.put(new CellIndex(2, 168), 27856.0); 
		bHM.put(new CellIndex(2, 169), 27956.0); bHM.put(new CellIndex(2, 170), 14526.0); bHM.put(new CellIndex(2, 171), 13530.0); bHM.put(new CellIndex(2, 172), 28156.0); 
		bHM.put(new CellIndex(2, 173), 28256.0); bHM.put(new CellIndex(2, 174), 28356.0); bHM.put(new CellIndex(2, 175), 28456.0); bHM.put(new CellIndex(2, 176), 28556.0); 
		bHM.put(new CellIndex(2, 177), 28656.0); bHM.put(new CellIndex(2, 178), 28756.0); bHM.put(new CellIndex(2, 179), 28856.0); bHM.put(new CellIndex(2, 180), 14994.0); 
		bHM.put(new CellIndex(2, 181), 13962.0); bHM.put(new CellIndex(2, 182), 29056.0); bHM.put(new CellIndex(2, 183), 29156.0); bHM.put(new CellIndex(2, 184), 29256.0); 
		bHM.put(new CellIndex(2, 185), 29356.0); bHM.put(new CellIndex(2, 186), 29456.0); bHM.put(new CellIndex(2, 187), 29556.0); bHM.put(new CellIndex(2, 188), 29656.0); 
		bHM.put(new CellIndex(2, 189), 29756.0); bHM.put(new CellIndex(2, 190), 15462.0); bHM.put(new CellIndex(2, 191), 7649.0); bHM.put(new CellIndex(2, 192), 15875.0); 
		bHM.put(new CellIndex(2, 193), 15929.0); bHM.put(new CellIndex(2, 194), 15983.0); bHM.put(new CellIndex(2, 195), 16037.0); bHM.put(new CellIndex(2, 196), 16091.0); 
		bHM.put(new CellIndex(2, 197), 16145.0); bHM.put(new CellIndex(2, 198), 16199.0); bHM.put(new CellIndex(2, 199), 16253.0); bHM.put(new CellIndex(2, 200), 8424.0); 
		bHM.put(new CellIndex(2, 201), 6591.0); bHM.put(new CellIndex(2, 202), 13619.0); bHM.put(new CellIndex(2, 203), 13681.0); bHM.put(new CellIndex(2, 204), 13743.0); 
		bHM.put(new CellIndex(2, 205), 13805.0); bHM.put(new CellIndex(2, 206), 13867.0); bHM.put(new CellIndex(2, 207), 13929.0); bHM.put(new CellIndex(2, 208), 13991.0); 
		bHM.put(new CellIndex(2, 209), 14053.0); bHM.put(new CellIndex(2, 210), 7254.0); bHM.put(new CellIndex(2, 211), 14266.0); bHM.put(new CellIndex(2, 212), 29428.0); 
		bHM.put(new CellIndex(2, 213), 29560.0); bHM.put(new CellIndex(2, 214), 29692.0); bHM.put(new CellIndex(2, 215), 29824.0); bHM.put(new CellIndex(2, 216), 29956.0); 
		bHM.put(new CellIndex(2, 217), 30088.0); bHM.put(new CellIndex(2, 218), 30220.0); bHM.put(new CellIndex(2, 219), 30352.0); bHM.put(new CellIndex(2, 220), 15642.0); 
		bHM.put(new CellIndex(2, 221), 14842.0); bHM.put(new CellIndex(2, 222), 30616.0); bHM.put(new CellIndex(2, 223), 30748.0); bHM.put(new CellIndex(2, 224), 30880.0); 
		bHM.put(new CellIndex(2, 225), 31012.0); bHM.put(new CellIndex(2, 226), 31144.0); bHM.put(new CellIndex(2, 227), 31276.0); bHM.put(new CellIndex(2, 228), 31408.0); 
		bHM.put(new CellIndex(2, 229), 31540.0); bHM.put(new CellIndex(2, 230), 16254.0); bHM.put(new CellIndex(2, 231), 15418.0); bHM.put(new CellIndex(2, 232), 31804.0); 
		bHM.put(new CellIndex(2, 233), 31936.0); bHM.put(new CellIndex(2, 234), 32068.0); bHM.put(new CellIndex(2, 235), 32200.0); bHM.put(new CellIndex(2, 236), 32332.0); 
		bHM.put(new CellIndex(2, 237), 32464.0); bHM.put(new CellIndex(2, 238), 32596.0); bHM.put(new CellIndex(2, 239), 32728.0); bHM.put(new CellIndex(2, 240), 16866.0); 
		bHM.put(new CellIndex(2, 241), 15994.0); bHM.put(new CellIndex(2, 242), 32992.0); bHM.put(new CellIndex(2, 243), 33124.0); bHM.put(new CellIndex(2, 244), 33256.0); 
		bHM.put(new CellIndex(2, 245), 33388.0); bHM.put(new CellIndex(2, 246), 33520.0); bHM.put(new CellIndex(2, 247), 33652.0); bHM.put(new CellIndex(2, 248), 33784.0); 
		bHM.put(new CellIndex(2, 249), 33916.0); bHM.put(new CellIndex(2, 250), 17478.0); bHM.put(new CellIndex(2, 251), 16570.0); bHM.put(new CellIndex(2, 252), 34180.0); 
		bHM.put(new CellIndex(2, 253), 34312.0); bHM.put(new CellIndex(2, 254), 34444.0); bHM.put(new CellIndex(2, 255), 34576.0); bHM.put(new CellIndex(2, 256), 34708.0); 
		bHM.put(new CellIndex(2, 257), 34840.0); bHM.put(new CellIndex(2, 258), 34972.0); bHM.put(new CellIndex(2, 259), 35104.0); bHM.put(new CellIndex(2, 260), 18090.0); 
		bHM.put(new CellIndex(2, 261), 17146.0); bHM.put(new CellIndex(2, 262), 35368.0); bHM.put(new CellIndex(2, 263), 35500.0); bHM.put(new CellIndex(2, 264), 35632.0); 
		bHM.put(new CellIndex(2, 265), 35764.0); bHM.put(new CellIndex(2, 266), 35896.0); bHM.put(new CellIndex(2, 267), 36028.0); bHM.put(new CellIndex(2, 268), 36160.0); 
		bHM.put(new CellIndex(2, 269), 36292.0); bHM.put(new CellIndex(2, 270), 18702.0); bHM.put(new CellIndex(2, 271), 17722.0); bHM.put(new CellIndex(2, 272), 36556.0); 
		bHM.put(new CellIndex(2, 273), 36688.0); bHM.put(new CellIndex(2, 274), 36820.0); bHM.put(new CellIndex(2, 275), 36952.0); bHM.put(new CellIndex(2, 276), 37084.0); 
		bHM.put(new CellIndex(2, 277), 37216.0); bHM.put(new CellIndex(2, 278), 37348.0); bHM.put(new CellIndex(2, 279), 37480.0); bHM.put(new CellIndex(2, 280), 19314.0); 
		bHM.put(new CellIndex(2, 281), 18298.0); bHM.put(new CellIndex(2, 282), 37744.0); bHM.put(new CellIndex(2, 283), 37876.0); bHM.put(new CellIndex(2, 284), 38008.0); 
		bHM.put(new CellIndex(2, 285), 38140.0); bHM.put(new CellIndex(2, 286), 38272.0); bHM.put(new CellIndex(2, 287), 38404.0); bHM.put(new CellIndex(2, 288), 38536.0); 
		bHM.put(new CellIndex(2, 289), 38668.0); bHM.put(new CellIndex(2, 290), 19926.0); bHM.put(new CellIndex(2, 291), 9853.0); bHM.put(new CellIndex(2, 292), 20291.0); 
		bHM.put(new CellIndex(2, 293), 20361.0); bHM.put(new CellIndex(2, 294), 20431.0); bHM.put(new CellIndex(2, 295), 20501.0); bHM.put(new CellIndex(2, 296), 20571.0); 
		bHM.put(new CellIndex(2, 297), 20641.0); bHM.put(new CellIndex(2, 298), 20711.0); bHM.put(new CellIndex(2, 299), 20781.0); bHM.put(new CellIndex(2, 300), 10692.0); 
	}
	
	private void fillTest2HM() {
		bHM.put(new CellIndex(1, 1), 4290.0); bHM.put(new CellIndex(1, 2), 6729.0); bHM.put(new CellIndex(1, 3), 4698.0); bHM.put(new CellIndex(1, 4), 7299.0); 
		bHM.put(new CellIndex(1, 5), 11430.0); bHM.put(new CellIndex(1, 6), 7965.0); bHM.put(new CellIndex(1, 7), 5586.0); bHM.put(new CellIndex(1, 8), 8727.0); 
		bHM.put(new CellIndex(1, 9), 6066.0); bHM.put(new CellIndex(1, 10), 5586.0); bHM.put(new CellIndex(1, 11), 8754.0); bHM.put(new CellIndex(1, 12), 6102.0); 
		bHM.put(new CellIndex(1, 13), 9486.0); bHM.put(new CellIndex(1, 14), 14832.0); bHM.put(new CellIndex(1, 15), 10314.0); bHM.put(new CellIndex(1, 16), 7206.0); 
		bHM.put(new CellIndex(1, 17), 11238.0); bHM.put(new CellIndex(1, 18), 7794.0); bHM.put(new CellIndex(2, 1), 11094.0); bHM.put(new CellIndex(2, 2), 17178.0); 
		bHM.put(new CellIndex(2, 3), 11826.0); bHM.put(new CellIndex(2, 4), 18234.0); bHM.put(new CellIndex(2, 5), 28197.0); bHM.put(new CellIndex(2, 6), 19386.0); 
		bHM.put(new CellIndex(2, 7), 13362.0); bHM.put(new CellIndex(2, 8), 20634.0); bHM.put(new CellIndex(2, 9), 14166.0); bHM.put(new CellIndex(2, 10), 15306.0); 
		bHM.put(new CellIndex(2, 11), 23577.0); bHM.put(new CellIndex(2, 12), 16146.0); bHM.put(new CellIndex(2, 13), 24795.0); bHM.put(new CellIndex(2, 14), 38160.0); 
		bHM.put(new CellIndex(2, 15), 26109.0); bHM.put(new CellIndex(2, 16), 17898.0); bHM.put(new CellIndex(2, 17), 27519.0); bHM.put(new CellIndex(2, 18), 18810.0); 
		bHM.put(new CellIndex(3, 1), 17898.0); bHM.put(new CellIndex(3, 2), 27627.0); bHM.put(new CellIndex(3, 3), 18954.0); bHM.put(new CellIndex(3, 4), 29169.0); 
		bHM.put(new CellIndex(3, 5), 44964.0); bHM.put(new CellIndex(3, 6), 30807.0); bHM.put(new CellIndex(3, 7), 21138.0); bHM.put(new CellIndex(3, 8), 32541.0); 
		bHM.put(new CellIndex(3, 9), 22266.0); bHM.put(new CellIndex(3, 10), 25026.0); bHM.put(new CellIndex(3, 11), 38400.0); bHM.put(new CellIndex(3, 12), 26190.0); 
		bHM.put(new CellIndex(3, 13), 40104.0); bHM.put(new CellIndex(3, 14), 61488.0); bHM.put(new CellIndex(3, 15), 41904.0); bHM.put(new CellIndex(3, 16), 28590.0); 
		bHM.put(new CellIndex(3, 17), 43800.0); bHM.put(new CellIndex(3, 18), 29826.0); bHM.put(new CellIndex(4, 1), 24702.0); bHM.put(new CellIndex(4, 2), 38076.0); 
		bHM.put(new CellIndex(4, 3), 26082.0); bHM.put(new CellIndex(4, 4), 40104.0); bHM.put(new CellIndex(4, 5), 61731.0); bHM.put(new CellIndex(4, 6), 42228.0); 
		bHM.put(new CellIndex(4, 7), 28914.0); bHM.put(new CellIndex(4, 8), 44448.0); bHM.put(new CellIndex(4, 9), 30366.0); bHM.put(new CellIndex(4, 10), 34746.0); 
		bHM.put(new CellIndex(4, 11), 53223.0); bHM.put(new CellIndex(4, 12), 36234.0); bHM.put(new CellIndex(4, 13), 55413.0); bHM.put(new CellIndex(4, 14), 84816.0); 
		bHM.put(new CellIndex(4, 15), 57699.0); bHM.put(new CellIndex(4, 16), 39282.0); bHM.put(new CellIndex(4, 17), 60081.0); bHM.put(new CellIndex(4, 18), 40842.0); 
		bHM.put(new CellIndex(5, 1), 31506.0); bHM.put(new CellIndex(5, 2), 48525.0); bHM.put(new CellIndex(5, 3), 33210.0); bHM.put(new CellIndex(5, 4), 51039.0); 
		bHM.put(new CellIndex(5, 5), 78498.0); bHM.put(new CellIndex(5, 6), 53649.0); bHM.put(new CellIndex(5, 7), 36690.0); bHM.put(new CellIndex(5, 8), 56355.0); 
		bHM.put(new CellIndex(5, 9), 38466.0); bHM.put(new CellIndex(5, 10), 44466.0); bHM.put(new CellIndex(5, 11), 68046.0); bHM.put(new CellIndex(5, 12), 46278.0); 
		bHM.put(new CellIndex(5, 13), 70722.0); bHM.put(new CellIndex(5, 14), 108144.0); bHM.put(new CellIndex(5, 15), 73494.0); bHM.put(new CellIndex(5, 16), 49974.0); 
		bHM.put(new CellIndex(5, 17), 76362.0); bHM.put(new CellIndex(5, 18), 51858.0);
	}
	
	private void fillTest3HM() {
		bHM.put(new CellIndex(1, 1), 489.0); bHM.put(new CellIndex(1, 2), 1044.0); bHM.put(new CellIndex(1, 3), 558.0); bHM.put(new CellIndex(1, 4), 1098.0); 
		bHM.put(new CellIndex(1, 5), 2328.0); bHM.put(new CellIndex(1, 6), 1236.0); bHM.put(new CellIndex(1, 7), 627.0); bHM.put(new CellIndex(1, 8), 1320.0); 
		bHM.put(new CellIndex(1, 9), 696.0); bHM.put(new CellIndex(1, 10), 624.0); bHM.put(new CellIndex(1, 11), 1341.0); bHM.put(new CellIndex(1, 12), 720.0); 
		bHM.put(new CellIndex(1, 13), 1422.0); bHM.put(new CellIndex(1, 14), 3030.0); bHM.put(new CellIndex(1, 15), 1614.0); bHM.put(new CellIndex(1, 16), 816.0); 
		bHM.put(new CellIndex(1, 17), 1725.0); bHM.put(new CellIndex(1, 18), 912.0); bHM.put(new CellIndex(2, 1), 1317.0); bHM.put(new CellIndex(2, 2), 2700.0); 
		bHM.put(new CellIndex(2, 3), 1386.0); bHM.put(new CellIndex(2, 4), 2754.0); bHM.put(new CellIndex(2, 5), 5640.0); bHM.put(new CellIndex(2, 6), 2892.0); 
		bHM.put(new CellIndex(2, 7), 1455.0); bHM.put(new CellIndex(2, 8), 2976.0); bHM.put(new CellIndex(2, 9), 1524.0); bHM.put(new CellIndex(2, 10), 1776.0); 
		bHM.put(new CellIndex(2, 11), 3645.0); bHM.put(new CellIndex(2, 12), 1872.0); bHM.put(new CellIndex(2, 13), 3726.0); bHM.put(new CellIndex(2, 14), 7638.0); 
		bHM.put(new CellIndex(2, 15), 3918.0); bHM.put(new CellIndex(2, 16), 1968.0); bHM.put(new CellIndex(2, 17), 4029.0); bHM.put(new CellIndex(2, 18), 2064.0); 
		bHM.put(new CellIndex(3, 1), 2145.0); bHM.put(new CellIndex(3, 2), 4356.0); bHM.put(new CellIndex(3, 3), 2214.0); bHM.put(new CellIndex(3, 4), 4410.0); 
		bHM.put(new CellIndex(3, 5), 8952.0); bHM.put(new CellIndex(3, 6), 4548.0); bHM.put(new CellIndex(3, 7), 2283.0); bHM.put(new CellIndex(3, 8), 4632.0); 
		bHM.put(new CellIndex(3, 9), 2352.0); bHM.put(new CellIndex(3, 10), 2928.0); bHM.put(new CellIndex(3, 11), 5949.0); bHM.put(new CellIndex(3, 12), 3024.0); 
		bHM.put(new CellIndex(3, 13), 6030.0); bHM.put(new CellIndex(3, 14), 12246.0); bHM.put(new CellIndex(3, 15), 6222.0); bHM.put(new CellIndex(3, 16), 3120.0); 
		bHM.put(new CellIndex(3, 17), 6333.0); bHM.put(new CellIndex(3, 18), 3216.0); bHM.put(new CellIndex(4, 1), 2973.0); bHM.put(new CellIndex(4, 2), 6012.0); 
		bHM.put(new CellIndex(4, 3), 3042.0); bHM.put(new CellIndex(4, 4), 6066.0); bHM.put(new CellIndex(4, 5), 12264.0); bHM.put(new CellIndex(4, 6), 6204.0); 
		bHM.put(new CellIndex(4, 7), 3111.0); bHM.put(new CellIndex(4, 8), 6288.0); bHM.put(new CellIndex(4, 9), 3180.0); bHM.put(new CellIndex(4, 10), 4080.0); 
		bHM.put(new CellIndex(4, 11), 8253.0); bHM.put(new CellIndex(4, 12), 4176.0); bHM.put(new CellIndex(4, 13), 8334.0); bHM.put(new CellIndex(4, 14), 16854.0); 
		bHM.put(new CellIndex(4, 15), 8526.0); bHM.put(new CellIndex(4, 16), 4272.0); bHM.put(new CellIndex(4, 17), 8637.0); bHM.put(new CellIndex(4, 18), 4368.0); 
		bHM.put(new CellIndex(5, 1), 3801.0); bHM.put(new CellIndex(5, 2), 7668.0); bHM.put(new CellIndex(5, 3), 3870.0); bHM.put(new CellIndex(5, 4), 7722.0); 
		bHM.put(new CellIndex(5, 5), 15576.0); bHM.put(new CellIndex(5, 6), 7860.0); bHM.put(new CellIndex(5, 7), 3939.0); bHM.put(new CellIndex(5, 8), 7944.0); 
		bHM.put(new CellIndex(5, 9), 4008.0); bHM.put(new CellIndex(5, 10), 5232.0); bHM.put(new CellIndex(5, 11), 10557.0); bHM.put(new CellIndex(5, 12), 5328.0); 
		bHM.put(new CellIndex(5, 13), 10638.0); bHM.put(new CellIndex(5, 14), 21462.0); bHM.put(new CellIndex(5, 15), 10830.0); bHM.put(new CellIndex(5, 16), 5424.0); 
		bHM.put(new CellIndex(5, 17), 10941.0); bHM.put(new CellIndex(5, 18), 5520.0); 
	}
	
	private void fillTest4HM() {
		bHM.put(new CellIndex(1, 1), 1908.0); bHM.put(new CellIndex(1, 2), 1830.0); bHM.put(new CellIndex(1, 3), 1944.0); bHM.put(new CellIndex(1, 4), 1863.0); 
		bHM.put(new CellIndex(1, 5), 1980.0); bHM.put(new CellIndex(1, 6), 1896.0); bHM.put(new CellIndex(1, 7), 2016.0); bHM.put(new CellIndex(1, 8), 1929.0); 
		bHM.put(new CellIndex(1, 9), 2052.0); bHM.put(new CellIndex(1, 10), 1962.0); bHM.put(new CellIndex(1, 11), 1866.0); bHM.put(new CellIndex(1, 12), 1764.0); 
		bHM.put(new CellIndex(1, 13), 1896.0); bHM.put(new CellIndex(1, 14), 1791.0); bHM.put(new CellIndex(1, 15), 1926.0); bHM.put(new CellIndex(1, 16), 1818.0); 
		bHM.put(new CellIndex(1, 17), 1956.0); bHM.put(new CellIndex(1, 18), 1845.0); bHM.put(new CellIndex(1, 19), 1986.0); bHM.put(new CellIndex(1, 20), 1872.0); 
		bHM.put(new CellIndex(1, 21), 2124.0); bHM.put(new CellIndex(1, 22), 2028.0); bHM.put(new CellIndex(1, 23), 2160.0); bHM.put(new CellIndex(1, 24), 2061.0); 
		bHM.put(new CellIndex(1, 25), 2196.0); bHM.put(new CellIndex(1, 26), 2094.0); bHM.put(new CellIndex(1, 27), 2232.0); bHM.put(new CellIndex(1, 28), 2127.0); 
		bHM.put(new CellIndex(1, 29), 2268.0); bHM.put(new CellIndex(1, 30), 2160.0); bHM.put(new CellIndex(1, 31), 2046.0); bHM.put(new CellIndex(1, 32), 1926.0); 
		bHM.put(new CellIndex(1, 33), 2076.0); bHM.put(new CellIndex(1, 34), 1953.0); bHM.put(new CellIndex(1, 35), 2106.0); bHM.put(new CellIndex(1, 36), 1980.0); 
		bHM.put(new CellIndex(1, 37), 2136.0); bHM.put(new CellIndex(1, 38), 2007.0); bHM.put(new CellIndex(1, 39), 2166.0); bHM.put(new CellIndex(1, 40), 2034.0); 
		bHM.put(new CellIndex(1, 41), 2340.0); bHM.put(new CellIndex(1, 42), 2226.0); bHM.put(new CellIndex(1, 43), 2376.0); bHM.put(new CellIndex(1, 44), 2259.0); 
		bHM.put(new CellIndex(1, 45), 2412.0); bHM.put(new CellIndex(1, 46), 2292.0); bHM.put(new CellIndex(1, 47), 2448.0); bHM.put(new CellIndex(1, 48), 2325.0); 
		bHM.put(new CellIndex(1, 49), 2484.0); bHM.put(new CellIndex(1, 50), 2358.0); bHM.put(new CellIndex(1, 51), 2226.0); bHM.put(new CellIndex(1, 52), 2088.0); 
		bHM.put(new CellIndex(1, 53), 2256.0); bHM.put(new CellIndex(1, 54), 2115.0); bHM.put(new CellIndex(1, 55), 2286.0); bHM.put(new CellIndex(1, 56), 2142.0); 
		bHM.put(new CellIndex(1, 57), 2316.0); bHM.put(new CellIndex(1, 58), 2169.0); bHM.put(new CellIndex(1, 59), 2346.0); bHM.put(new CellIndex(1, 60), 2196.0); 
		bHM.put(new CellIndex(1, 61), 2556.0); bHM.put(new CellIndex(1, 62), 2424.0); bHM.put(new CellIndex(1, 63), 2592.0); bHM.put(new CellIndex(1, 64), 2457.0); 
		bHM.put(new CellIndex(1, 65), 2628.0); bHM.put(new CellIndex(1, 66), 2490.0); bHM.put(new CellIndex(1, 67), 2664.0); bHM.put(new CellIndex(1, 68), 2523.0); 
		bHM.put(new CellIndex(1, 69), 2700.0); bHM.put(new CellIndex(1, 70), 2556.0); bHM.put(new CellIndex(1, 71), 2406.0); bHM.put(new CellIndex(1, 72), 2250.0); 
		bHM.put(new CellIndex(1, 73), 2436.0); bHM.put(new CellIndex(1, 74), 2277.0); bHM.put(new CellIndex(1, 75), 2466.0); bHM.put(new CellIndex(1, 76), 2304.0); 
		bHM.put(new CellIndex(1, 77), 2496.0); bHM.put(new CellIndex(1, 78), 2331.0); bHM.put(new CellIndex(1, 79), 2526.0); bHM.put(new CellIndex(1, 80), 2358.0); 
		bHM.put(new CellIndex(1, 81), 2772.0); bHM.put(new CellIndex(1, 82), 2622.0); bHM.put(new CellIndex(1, 83), 2808.0); bHM.put(new CellIndex(1, 84), 2655.0); 
		bHM.put(new CellIndex(1, 85), 2844.0); bHM.put(new CellIndex(1, 86), 2688.0); bHM.put(new CellIndex(1, 87), 2880.0); bHM.put(new CellIndex(1, 88), 2721.0); 
		bHM.put(new CellIndex(1, 89), 2916.0); bHM.put(new CellIndex(1, 90), 2754.0); bHM.put(new CellIndex(1, 91), 2586.0); bHM.put(new CellIndex(1, 92), 2412.0); 
		bHM.put(new CellIndex(1, 93), 2616.0); bHM.put(new CellIndex(1, 94), 2439.0); bHM.put(new CellIndex(1, 95), 2646.0); bHM.put(new CellIndex(1, 96), 2466.0); 
		bHM.put(new CellIndex(1, 97), 2676.0); bHM.put(new CellIndex(1, 98), 2493.0); bHM.put(new CellIndex(1, 99), 2706.0); bHM.put(new CellIndex(1, 100), 2520.0); 
		bHM.put(new CellIndex(1, 101), 2352.0); bHM.put(new CellIndex(1, 102), 2286.0); bHM.put(new CellIndex(1, 103), 2400.0); bHM.put(new CellIndex(1, 104), 2331.0); 
		bHM.put(new CellIndex(1, 105), 2448.0); bHM.put(new CellIndex(1, 106), 2376.0); bHM.put(new CellIndex(1, 107), 2496.0); bHM.put(new CellIndex(1, 108), 2421.0); 
		bHM.put(new CellIndex(1, 109), 2544.0); bHM.put(new CellIndex(1, 110), 2466.0); bHM.put(new CellIndex(1, 111), 2382.0); bHM.put(new CellIndex(1, 112), 2292.0); 
		bHM.put(new CellIndex(1, 113), 2424.0); bHM.put(new CellIndex(1, 114), 2331.0); bHM.put(new CellIndex(1, 115), 2466.0); bHM.put(new CellIndex(1, 116), 2370.0); 
		bHM.put(new CellIndex(1, 117), 2508.0); bHM.put(new CellIndex(1, 118), 2409.0); bHM.put(new CellIndex(1, 119), 2550.0); bHM.put(new CellIndex(1, 120), 2448.0); 
		bHM.put(new CellIndex(1, 121), 2640.0); bHM.put(new CellIndex(1, 122), 2556.0); bHM.put(new CellIndex(1, 123), 2688.0); bHM.put(new CellIndex(1, 124), 2601.0); 
		bHM.put(new CellIndex(1, 125), 2736.0); bHM.put(new CellIndex(1, 126), 2646.0); bHM.put(new CellIndex(1, 127), 2784.0); bHM.put(new CellIndex(1, 128), 2691.0); 
		bHM.put(new CellIndex(1, 129), 2832.0); bHM.put(new CellIndex(1, 130), 2736.0); bHM.put(new CellIndex(1, 131), 2634.0); bHM.put(new CellIndex(1, 132), 2526.0); 
		bHM.put(new CellIndex(1, 133), 2676.0); bHM.put(new CellIndex(1, 134), 2565.0); bHM.put(new CellIndex(1, 135), 2718.0); bHM.put(new CellIndex(1, 136), 2604.0); 
		bHM.put(new CellIndex(1, 137), 2760.0); bHM.put(new CellIndex(1, 138), 2643.0); bHM.put(new CellIndex(1, 139), 2802.0); bHM.put(new CellIndex(1, 140), 2682.0); 
		bHM.put(new CellIndex(1, 141), 2928.0); bHM.put(new CellIndex(1, 142), 2826.0); bHM.put(new CellIndex(1, 143), 2976.0); bHM.put(new CellIndex(1, 144), 2871.0); 
		bHM.put(new CellIndex(1, 145), 3024.0); bHM.put(new CellIndex(1, 146), 2916.0); bHM.put(new CellIndex(1, 147), 3072.0); bHM.put(new CellIndex(1, 148), 2961.0); 
		bHM.put(new CellIndex(1, 149), 3120.0); bHM.put(new CellIndex(1, 150), 3006.0); bHM.put(new CellIndex(1, 151), 2886.0); bHM.put(new CellIndex(1, 152), 2760.0); 
		bHM.put(new CellIndex(1, 153), 2928.0); bHM.put(new CellIndex(1, 154), 2799.0); bHM.put(new CellIndex(1, 155), 2970.0); bHM.put(new CellIndex(1, 156), 2838.0); 
		bHM.put(new CellIndex(1, 157), 3012.0); bHM.put(new CellIndex(1, 158), 2877.0); bHM.put(new CellIndex(1, 159), 3054.0); bHM.put(new CellIndex(1, 160), 2916.0); 
		bHM.put(new CellIndex(1, 161), 3216.0); bHM.put(new CellIndex(1, 162), 3096.0); bHM.put(new CellIndex(1, 163), 3264.0); bHM.put(new CellIndex(1, 164), 3141.0); 
		bHM.put(new CellIndex(1, 165), 3312.0); bHM.put(new CellIndex(1, 166), 3186.0); bHM.put(new CellIndex(1, 167), 3360.0); bHM.put(new CellIndex(1, 168), 3231.0); 
		bHM.put(new CellIndex(1, 169), 3408.0); bHM.put(new CellIndex(1, 170), 3276.0); bHM.put(new CellIndex(1, 171), 3138.0); bHM.put(new CellIndex(1, 172), 2994.0); 
		bHM.put(new CellIndex(1, 173), 3180.0); bHM.put(new CellIndex(1, 174), 3033.0); bHM.put(new CellIndex(1, 175), 3222.0); bHM.put(new CellIndex(1, 176), 3072.0); 
		bHM.put(new CellIndex(1, 177), 3264.0); bHM.put(new CellIndex(1, 178), 3111.0); bHM.put(new CellIndex(1, 179), 3306.0); bHM.put(new CellIndex(1, 180), 3150.0); 
		bHM.put(new CellIndex(1, 181), 3504.0); bHM.put(new CellIndex(1, 182), 3366.0); bHM.put(new CellIndex(1, 183), 3552.0); bHM.put(new CellIndex(1, 184), 3411.0); 
		bHM.put(new CellIndex(1, 185), 3600.0); bHM.put(new CellIndex(1, 186), 3456.0); bHM.put(new CellIndex(1, 187), 3648.0); bHM.put(new CellIndex(1, 188), 3501.0); 
		bHM.put(new CellIndex(1, 189), 3696.0); bHM.put(new CellIndex(1, 190), 3546.0); bHM.put(new CellIndex(1, 191), 3390.0); bHM.put(new CellIndex(1, 192), 3228.0); 
		bHM.put(new CellIndex(1, 193), 3432.0); bHM.put(new CellIndex(1, 194), 3267.0); bHM.put(new CellIndex(1, 195), 3474.0); bHM.put(new CellIndex(1, 196), 3306.0); 
		bHM.put(new CellIndex(1, 197), 3516.0); bHM.put(new CellIndex(1, 198), 3345.0); bHM.put(new CellIndex(1, 199), 3558.0); bHM.put(new CellIndex(1, 200), 3384.0); 
		bHM.put(new CellIndex(2, 1), 5796.0); bHM.put(new CellIndex(2, 2), 5394.0); bHM.put(new CellIndex(2, 3), 5832.0); bHM.put(new CellIndex(2, 4), 5427.0); 
		bHM.put(new CellIndex(2, 5), 5868.0); bHM.put(new CellIndex(2, 6), 5460.0); bHM.put(new CellIndex(2, 7), 5904.0); bHM.put(new CellIndex(2, 8), 5493.0); 
		bHM.put(new CellIndex(2, 9), 5940.0); bHM.put(new CellIndex(2, 10), 5526.0); bHM.put(new CellIndex(2, 11), 5106.0); bHM.put(new CellIndex(2, 12), 4680.0); 
		bHM.put(new CellIndex(2, 13), 5136.0); bHM.put(new CellIndex(2, 14), 4707.0); bHM.put(new CellIndex(2, 15), 5166.0); bHM.put(new CellIndex(2, 16), 4734.0); 
		bHM.put(new CellIndex(2, 17), 5196.0); bHM.put(new CellIndex(2, 18), 4761.0); bHM.put(new CellIndex(2, 19), 5226.0); bHM.put(new CellIndex(2, 20), 4788.0); 
		bHM.put(new CellIndex(2, 21), 6012.0); bHM.put(new CellIndex(2, 22), 5592.0); bHM.put(new CellIndex(2, 23), 6048.0); bHM.put(new CellIndex(2, 24), 5625.0); 
		bHM.put(new CellIndex(2, 25), 6084.0); bHM.put(new CellIndex(2, 26), 5658.0); bHM.put(new CellIndex(2, 27), 6120.0); bHM.put(new CellIndex(2, 28), 5691.0); 
		bHM.put(new CellIndex(2, 29), 6156.0); bHM.put(new CellIndex(2, 30), 5724.0); bHM.put(new CellIndex(2, 31), 5286.0); bHM.put(new CellIndex(2, 32), 4842.0); 
		bHM.put(new CellIndex(2, 33), 5316.0); bHM.put(new CellIndex(2, 34), 4869.0); bHM.put(new CellIndex(2, 35), 5346.0); bHM.put(new CellIndex(2, 36), 4896.0); 
		bHM.put(new CellIndex(2, 37), 5376.0); bHM.put(new CellIndex(2, 38), 4923.0); bHM.put(new CellIndex(2, 39), 5406.0); bHM.put(new CellIndex(2, 40), 4950.0); 
		bHM.put(new CellIndex(2, 41), 6228.0); bHM.put(new CellIndex(2, 42), 5790.0); bHM.put(new CellIndex(2, 43), 6264.0); bHM.put(new CellIndex(2, 44), 5823.0); 
		bHM.put(new CellIndex(2, 45), 6300.0); bHM.put(new CellIndex(2, 46), 5856.0); bHM.put(new CellIndex(2, 47), 6336.0); bHM.put(new CellIndex(2, 48), 5889.0); 
		bHM.put(new CellIndex(2, 49), 6372.0); bHM.put(new CellIndex(2, 50), 5922.0); bHM.put(new CellIndex(2, 51), 5466.0); bHM.put(new CellIndex(2, 52), 5004.0); 
		bHM.put(new CellIndex(2, 53), 5496.0); bHM.put(new CellIndex(2, 54), 5031.0); bHM.put(new CellIndex(2, 55), 5526.0); bHM.put(new CellIndex(2, 56), 5058.0); 
		bHM.put(new CellIndex(2, 57), 5556.0); bHM.put(new CellIndex(2, 58), 5085.0); bHM.put(new CellIndex(2, 59), 5586.0); bHM.put(new CellIndex(2, 60), 5112.0); 
		bHM.put(new CellIndex(2, 61), 6444.0); bHM.put(new CellIndex(2, 62), 5988.0); bHM.put(new CellIndex(2, 63), 6480.0); bHM.put(new CellIndex(2, 64), 6021.0); 
		bHM.put(new CellIndex(2, 65), 6516.0); bHM.put(new CellIndex(2, 66), 6054.0); bHM.put(new CellIndex(2, 67), 6552.0); bHM.put(new CellIndex(2, 68), 6087.0); 
		bHM.put(new CellIndex(2, 69), 6588.0); bHM.put(new CellIndex(2, 70), 6120.0); bHM.put(new CellIndex(2, 71), 5646.0); bHM.put(new CellIndex(2, 72), 5166.0); 
		bHM.put(new CellIndex(2, 73), 5676.0); bHM.put(new CellIndex(2, 74), 5193.0); bHM.put(new CellIndex(2, 75), 5706.0); bHM.put(new CellIndex(2, 76), 5220.0); 
		bHM.put(new CellIndex(2, 77), 5736.0); bHM.put(new CellIndex(2, 78), 5247.0); bHM.put(new CellIndex(2, 79), 5766.0); bHM.put(new CellIndex(2, 80), 5274.0); 
		bHM.put(new CellIndex(2, 81), 6660.0); bHM.put(new CellIndex(2, 82), 6186.0); bHM.put(new CellIndex(2, 83), 6696.0); bHM.put(new CellIndex(2, 84), 6219.0); 
		bHM.put(new CellIndex(2, 85), 6732.0); bHM.put(new CellIndex(2, 86), 6252.0); bHM.put(new CellIndex(2, 87), 6768.0); bHM.put(new CellIndex(2, 88), 6285.0); 
		bHM.put(new CellIndex(2, 89), 6804.0); bHM.put(new CellIndex(2, 90), 6318.0); bHM.put(new CellIndex(2, 91), 5826.0); bHM.put(new CellIndex(2, 92), 5328.0); 
		bHM.put(new CellIndex(2, 93), 5856.0); bHM.put(new CellIndex(2, 94), 5355.0); bHM.put(new CellIndex(2, 95), 5886.0); bHM.put(new CellIndex(2, 96), 5382.0); 
		bHM.put(new CellIndex(2, 97), 5916.0); bHM.put(new CellIndex(2, 98), 5409.0); bHM.put(new CellIndex(2, 99), 5946.0); bHM.put(new CellIndex(2, 100), 5436.0); 
		bHM.put(new CellIndex(2, 101), 7536.0); bHM.put(new CellIndex(2, 102), 7146.0); bHM.put(new CellIndex(2, 103), 7584.0); bHM.put(new CellIndex(2, 104), 7191.0); 
		bHM.put(new CellIndex(2, 105), 7632.0); bHM.put(new CellIndex(2, 106), 7236.0); bHM.put(new CellIndex(2, 107), 7680.0); bHM.put(new CellIndex(2, 108), 7281.0); 
		bHM.put(new CellIndex(2, 109), 7728.0); bHM.put(new CellIndex(2, 110), 7326.0); bHM.put(new CellIndex(2, 111), 6918.0); bHM.put(new CellIndex(2, 112), 6504.0); 
		bHM.put(new CellIndex(2, 113), 6960.0); bHM.put(new CellIndex(2, 114), 6543.0); bHM.put(new CellIndex(2, 115), 7002.0); bHM.put(new CellIndex(2, 116), 6582.0); 
		bHM.put(new CellIndex(2, 117), 7044.0); bHM.put(new CellIndex(2, 118), 6621.0); bHM.put(new CellIndex(2, 119), 7086.0); bHM.put(new CellIndex(2, 120), 6660.0); 
		bHM.put(new CellIndex(2, 121), 7824.0); bHM.put(new CellIndex(2, 122), 7416.0); bHM.put(new CellIndex(2, 123), 7872.0); bHM.put(new CellIndex(2, 124), 7461.0); 
		bHM.put(new CellIndex(2, 125), 7920.0); bHM.put(new CellIndex(2, 126), 7506.0); bHM.put(new CellIndex(2, 127), 7968.0); bHM.put(new CellIndex(2, 128), 7551.0); 
		bHM.put(new CellIndex(2, 129), 8016.0); bHM.put(new CellIndex(2, 130), 7596.0); bHM.put(new CellIndex(2, 131), 7170.0); bHM.put(new CellIndex(2, 132), 6738.0); 
		bHM.put(new CellIndex(2, 133), 7212.0); bHM.put(new CellIndex(2, 134), 6777.0); bHM.put(new CellIndex(2, 135), 7254.0); bHM.put(new CellIndex(2, 136), 6816.0); 
		bHM.put(new CellIndex(2, 137), 7296.0); bHM.put(new CellIndex(2, 138), 6855.0); bHM.put(new CellIndex(2, 139), 7338.0); bHM.put(new CellIndex(2, 140), 6894.0); 
		bHM.put(new CellIndex(2, 141), 8112.0); bHM.put(new CellIndex(2, 142), 7686.0); bHM.put(new CellIndex(2, 143), 8160.0); bHM.put(new CellIndex(2, 144), 7731.0); 
		bHM.put(new CellIndex(2, 145), 8208.0); bHM.put(new CellIndex(2, 146), 7776.0); bHM.put(new CellIndex(2, 147), 8256.0); bHM.put(new CellIndex(2, 148), 7821.0); 
		bHM.put(new CellIndex(2, 149), 8304.0); bHM.put(new CellIndex(2, 150), 7866.0); bHM.put(new CellIndex(2, 151), 7422.0); bHM.put(new CellIndex(2, 152), 6972.0); 
		bHM.put(new CellIndex(2, 153), 7464.0); bHM.put(new CellIndex(2, 154), 7011.0); bHM.put(new CellIndex(2, 155), 7506.0); bHM.put(new CellIndex(2, 156), 7050.0); 
		bHM.put(new CellIndex(2, 157), 7548.0); bHM.put(new CellIndex(2, 158), 7089.0); bHM.put(new CellIndex(2, 159), 7590.0); bHM.put(new CellIndex(2, 160), 7128.0); 
		bHM.put(new CellIndex(2, 161), 8400.0); bHM.put(new CellIndex(2, 162), 7956.0); bHM.put(new CellIndex(2, 163), 8448.0); bHM.put(new CellIndex(2, 164), 8001.0); 
		bHM.put(new CellIndex(2, 165), 8496.0); bHM.put(new CellIndex(2, 166), 8046.0); bHM.put(new CellIndex(2, 167), 8544.0); bHM.put(new CellIndex(2, 168), 8091.0); 
		bHM.put(new CellIndex(2, 169), 8592.0); bHM.put(new CellIndex(2, 170), 8136.0); bHM.put(new CellIndex(2, 171), 7674.0); bHM.put(new CellIndex(2, 172), 7206.0); 
		bHM.put(new CellIndex(2, 173), 7716.0); bHM.put(new CellIndex(2, 174), 7245.0); bHM.put(new CellIndex(2, 175), 7758.0); bHM.put(new CellIndex(2, 176), 7284.0); 
		bHM.put(new CellIndex(2, 177), 7800.0); bHM.put(new CellIndex(2, 178), 7323.0); bHM.put(new CellIndex(2, 179), 7842.0); bHM.put(new CellIndex(2, 180), 7362.0); 
		bHM.put(new CellIndex(2, 181), 8688.0); bHM.put(new CellIndex(2, 182), 8226.0); bHM.put(new CellIndex(2, 183), 8736.0); bHM.put(new CellIndex(2, 184), 8271.0); 
		bHM.put(new CellIndex(2, 185), 8784.0); bHM.put(new CellIndex(2, 186), 8316.0); bHM.put(new CellIndex(2, 187), 8832.0); bHM.put(new CellIndex(2, 188), 8361.0); 
		bHM.put(new CellIndex(2, 189), 8880.0); bHM.put(new CellIndex(2, 190), 8406.0); bHM.put(new CellIndex(2, 191), 7926.0); bHM.put(new CellIndex(2, 192), 7440.0); 
		bHM.put(new CellIndex(2, 193), 7968.0); bHM.put(new CellIndex(2, 194), 7479.0); bHM.put(new CellIndex(2, 195), 8010.0); bHM.put(new CellIndex(2, 196), 7518.0); 
		bHM.put(new CellIndex(2, 197), 8052.0); bHM.put(new CellIndex(2, 198), 7557.0); bHM.put(new CellIndex(2, 199), 8094.0); bHM.put(new CellIndex(2, 200), 7596.0); 
		bHM.put(new CellIndex(3, 1), 9684.0); bHM.put(new CellIndex(3, 2), 8958.0); bHM.put(new CellIndex(3, 3), 9720.0); bHM.put(new CellIndex(3, 4), 8991.0); 
		bHM.put(new CellIndex(3, 5), 9756.0); bHM.put(new CellIndex(3, 6), 9024.0); bHM.put(new CellIndex(3, 7), 9792.0); bHM.put(new CellIndex(3, 8), 9057.0); 
		bHM.put(new CellIndex(3, 9), 9828.0); bHM.put(new CellIndex(3, 10), 9090.0); bHM.put(new CellIndex(3, 11), 8346.0); bHM.put(new CellIndex(3, 12), 7596.0); 
		bHM.put(new CellIndex(3, 13), 8376.0); bHM.put(new CellIndex(3, 14), 7623.0); bHM.put(new CellIndex(3, 15), 8406.0); bHM.put(new CellIndex(3, 16), 7650.0); 
		bHM.put(new CellIndex(3, 17), 8436.0); bHM.put(new CellIndex(3, 18), 7677.0); bHM.put(new CellIndex(3, 19), 8466.0); bHM.put(new CellIndex(3, 20), 7704.0); 
		bHM.put(new CellIndex(3, 21), 9900.0); bHM.put(new CellIndex(3, 22), 9156.0); bHM.put(new CellIndex(3, 23), 9936.0); bHM.put(new CellIndex(3, 24), 9189.0); 
		bHM.put(new CellIndex(3, 25), 9972.0); bHM.put(new CellIndex(3, 26), 9222.0); bHM.put(new CellIndex(3, 27), 10008.0); bHM.put(new CellIndex(3, 28), 9255.0); 
		bHM.put(new CellIndex(3, 29), 10044.0); bHM.put(new CellIndex(3, 30), 9288.0); bHM.put(new CellIndex(3, 31), 8526.0); bHM.put(new CellIndex(3, 32), 7758.0); 
		bHM.put(new CellIndex(3, 33), 8556.0); bHM.put(new CellIndex(3, 34), 7785.0); bHM.put(new CellIndex(3, 35), 8586.0); bHM.put(new CellIndex(3, 36), 7812.0); 
		bHM.put(new CellIndex(3, 37), 8616.0); bHM.put(new CellIndex(3, 38), 7839.0); bHM.put(new CellIndex(3, 39), 8646.0); bHM.put(new CellIndex(3, 40), 7866.0); 
		bHM.put(new CellIndex(3, 41), 10116.0); bHM.put(new CellIndex(3, 42), 9354.0); bHM.put(new CellIndex(3, 43), 10152.0); bHM.put(new CellIndex(3, 44), 9387.0); 
		bHM.put(new CellIndex(3, 45), 10188.0); bHM.put(new CellIndex(3, 46), 9420.0); bHM.put(new CellIndex(3, 47), 10224.0); bHM.put(new CellIndex(3, 48), 9453.0); 
		bHM.put(new CellIndex(3, 49), 10260.0); bHM.put(new CellIndex(3, 50), 9486.0); bHM.put(new CellIndex(3, 51), 8706.0); bHM.put(new CellIndex(3, 52), 7920.0); 
		bHM.put(new CellIndex(3, 53), 8736.0); bHM.put(new CellIndex(3, 54), 7947.0); bHM.put(new CellIndex(3, 55), 8766.0); bHM.put(new CellIndex(3, 56), 7974.0); 
		bHM.put(new CellIndex(3, 57), 8796.0); bHM.put(new CellIndex(3, 58), 8001.0); bHM.put(new CellIndex(3, 59), 8826.0); bHM.put(new CellIndex(3, 60), 8028.0); 
		bHM.put(new CellIndex(3, 61), 10332.0); bHM.put(new CellIndex(3, 62), 9552.0); bHM.put(new CellIndex(3, 63), 10368.0); bHM.put(new CellIndex(3, 64), 9585.0); 
		bHM.put(new CellIndex(3, 65), 10404.0); bHM.put(new CellIndex(3, 66), 9618.0); bHM.put(new CellIndex(3, 67), 10440.0); bHM.put(new CellIndex(3, 68), 9651.0); 
		bHM.put(new CellIndex(3, 69), 10476.0); bHM.put(new CellIndex(3, 70), 9684.0); bHM.put(new CellIndex(3, 71), 8886.0); bHM.put(new CellIndex(3, 72), 8082.0); 
		bHM.put(new CellIndex(3, 73), 8916.0); bHM.put(new CellIndex(3, 74), 8109.0); bHM.put(new CellIndex(3, 75), 8946.0); bHM.put(new CellIndex(3, 76), 8136.0); 
		bHM.put(new CellIndex(3, 77), 8976.0); bHM.put(new CellIndex(3, 78), 8163.0); bHM.put(new CellIndex(3, 79), 9006.0); bHM.put(new CellIndex(3, 80), 8190.0); 
		bHM.put(new CellIndex(3, 81), 10548.0); bHM.put(new CellIndex(3, 82), 9750.0); bHM.put(new CellIndex(3, 83), 10584.0); bHM.put(new CellIndex(3, 84), 9783.0); 
		bHM.put(new CellIndex(3, 85), 10620.0); bHM.put(new CellIndex(3, 86), 9816.0); bHM.put(new CellIndex(3, 87), 10656.0); bHM.put(new CellIndex(3, 88), 9849.0); 
		bHM.put(new CellIndex(3, 89), 10692.0); bHM.put(new CellIndex(3, 90), 9882.0); bHM.put(new CellIndex(3, 91), 9066.0); bHM.put(new CellIndex(3, 92), 8244.0); 
		bHM.put(new CellIndex(3, 93), 9096.0); bHM.put(new CellIndex(3, 94), 8271.0); bHM.put(new CellIndex(3, 95), 9126.0); bHM.put(new CellIndex(3, 96), 8298.0); 
		bHM.put(new CellIndex(3, 97), 9156.0); bHM.put(new CellIndex(3, 98), 8325.0); bHM.put(new CellIndex(3, 99), 9186.0); bHM.put(new CellIndex(3, 100), 8352.0); 
		bHM.put(new CellIndex(3, 101), 12720.0); bHM.put(new CellIndex(3, 102), 12006.0); bHM.put(new CellIndex(3, 103), 12768.0); bHM.put(new CellIndex(3, 104), 12051.0); 
		bHM.put(new CellIndex(3, 105), 12816.0); bHM.put(new CellIndex(3, 106), 12096.0); bHM.put(new CellIndex(3, 107), 12864.0); bHM.put(new CellIndex(3, 108), 12141.0); 
		bHM.put(new CellIndex(3, 109), 12912.0); bHM.put(new CellIndex(3, 110), 12186.0); bHM.put(new CellIndex(3, 111), 11454.0); bHM.put(new CellIndex(3, 112), 10716.0); 
		bHM.put(new CellIndex(3, 113), 11496.0); bHM.put(new CellIndex(3, 114), 10755.0); bHM.put(new CellIndex(3, 115), 11538.0); bHM.put(new CellIndex(3, 116), 10794.0); 
		bHM.put(new CellIndex(3, 117), 11580.0); bHM.put(new CellIndex(3, 118), 10833.0); bHM.put(new CellIndex(3, 119), 11622.0); bHM.put(new CellIndex(3, 120), 10872.0); 
		bHM.put(new CellIndex(3, 121), 13008.0); bHM.put(new CellIndex(3, 122), 12276.0); bHM.put(new CellIndex(3, 123), 13056.0); bHM.put(new CellIndex(3, 124), 12321.0); 
		bHM.put(new CellIndex(3, 125), 13104.0); bHM.put(new CellIndex(3, 126), 12366.0); bHM.put(new CellIndex(3, 127), 13152.0); bHM.put(new CellIndex(3, 128), 12411.0); 
		bHM.put(new CellIndex(3, 129), 13200.0); bHM.put(new CellIndex(3, 130), 12456.0); bHM.put(new CellIndex(3, 131), 11706.0); bHM.put(new CellIndex(3, 132), 10950.0); 
		bHM.put(new CellIndex(3, 133), 11748.0); bHM.put(new CellIndex(3, 134), 10989.0); bHM.put(new CellIndex(3, 135), 11790.0); bHM.put(new CellIndex(3, 136), 11028.0); 
		bHM.put(new CellIndex(3, 137), 11832.0); bHM.put(new CellIndex(3, 138), 11067.0); bHM.put(new CellIndex(3, 139), 11874.0); bHM.put(new CellIndex(3, 140), 11106.0); 
		bHM.put(new CellIndex(3, 141), 13296.0); bHM.put(new CellIndex(3, 142), 12546.0); bHM.put(new CellIndex(3, 143), 13344.0); bHM.put(new CellIndex(3, 144), 12591.0); 
		bHM.put(new CellIndex(3, 145), 13392.0); bHM.put(new CellIndex(3, 146), 12636.0); bHM.put(new CellIndex(3, 147), 13440.0); bHM.put(new CellIndex(3, 148), 12681.0); 
		bHM.put(new CellIndex(3, 149), 13488.0); bHM.put(new CellIndex(3, 150), 12726.0); bHM.put(new CellIndex(3, 151), 11958.0); bHM.put(new CellIndex(3, 152), 11184.0); 
		bHM.put(new CellIndex(3, 153), 12000.0); bHM.put(new CellIndex(3, 154), 11223.0); bHM.put(new CellIndex(3, 155), 12042.0); bHM.put(new CellIndex(3, 156), 11262.0); 
		bHM.put(new CellIndex(3, 157), 12084.0); bHM.put(new CellIndex(3, 158), 11301.0); bHM.put(new CellIndex(3, 159), 12126.0); bHM.put(new CellIndex(3, 160), 11340.0); 
		bHM.put(new CellIndex(3, 161), 13584.0); bHM.put(new CellIndex(3, 162), 12816.0); bHM.put(new CellIndex(3, 163), 13632.0); bHM.put(new CellIndex(3, 164), 12861.0); 
		bHM.put(new CellIndex(3, 165), 13680.0); bHM.put(new CellIndex(3, 166), 12906.0); bHM.put(new CellIndex(3, 167), 13728.0); bHM.put(new CellIndex(3, 168), 12951.0); 
		bHM.put(new CellIndex(3, 169), 13776.0); bHM.put(new CellIndex(3, 170), 12996.0); bHM.put(new CellIndex(3, 171), 12210.0); bHM.put(new CellIndex(3, 172), 11418.0); 
		bHM.put(new CellIndex(3, 173), 12252.0); bHM.put(new CellIndex(3, 174), 11457.0); bHM.put(new CellIndex(3, 175), 12294.0); bHM.put(new CellIndex(3, 176), 11496.0); 
		bHM.put(new CellIndex(3, 177), 12336.0); bHM.put(new CellIndex(3, 178), 11535.0); bHM.put(new CellIndex(3, 179), 12378.0); bHM.put(new CellIndex(3, 180), 11574.0); 
		bHM.put(new CellIndex(3, 181), 13872.0); bHM.put(new CellIndex(3, 182), 13086.0); bHM.put(new CellIndex(3, 183), 13920.0); bHM.put(new CellIndex(3, 184), 13131.0); 
		bHM.put(new CellIndex(3, 185), 13968.0); bHM.put(new CellIndex(3, 186), 13176.0); bHM.put(new CellIndex(3, 187), 14016.0); bHM.put(new CellIndex(3, 188), 13221.0); 
		bHM.put(new CellIndex(3, 189), 14064.0); bHM.put(new CellIndex(3, 190), 13266.0); bHM.put(new CellIndex(3, 191), 12462.0); bHM.put(new CellIndex(3, 192), 11652.0); 
		bHM.put(new CellIndex(3, 193), 12504.0); bHM.put(new CellIndex(3, 194), 11691.0); bHM.put(new CellIndex(3, 195), 12546.0); bHM.put(new CellIndex(3, 196), 11730.0); 
		bHM.put(new CellIndex(3, 197), 12588.0); bHM.put(new CellIndex(3, 198), 11769.0); bHM.put(new CellIndex(3, 199), 12630.0); bHM.put(new CellIndex(3, 200), 11808.0); 
		bHM.put(new CellIndex(4, 1), 13572.0); bHM.put(new CellIndex(4, 2), 12522.0); bHM.put(new CellIndex(4, 3), 13608.0); bHM.put(new CellIndex(4, 4), 12555.0); 
		bHM.put(new CellIndex(4, 5), 13644.0); bHM.put(new CellIndex(4, 6), 12588.0); bHM.put(new CellIndex(4, 7), 13680.0); bHM.put(new CellIndex(4, 8), 12621.0); 
		bHM.put(new CellIndex(4, 9), 13716.0); bHM.put(new CellIndex(4, 10), 12654.0); bHM.put(new CellIndex(4, 11), 11586.0); bHM.put(new CellIndex(4, 12), 10512.0); 
		bHM.put(new CellIndex(4, 13), 11616.0); bHM.put(new CellIndex(4, 14), 10539.0); bHM.put(new CellIndex(4, 15), 11646.0); bHM.put(new CellIndex(4, 16), 10566.0); 
		bHM.put(new CellIndex(4, 17), 11676.0); bHM.put(new CellIndex(4, 18), 10593.0); bHM.put(new CellIndex(4, 19), 11706.0); bHM.put(new CellIndex(4, 20), 10620.0); 
		bHM.put(new CellIndex(4, 21), 13788.0); bHM.put(new CellIndex(4, 22), 12720.0); bHM.put(new CellIndex(4, 23), 13824.0); bHM.put(new CellIndex(4, 24), 12753.0); 
		bHM.put(new CellIndex(4, 25), 13860.0); bHM.put(new CellIndex(4, 26), 12786.0); bHM.put(new CellIndex(4, 27), 13896.0); bHM.put(new CellIndex(4, 28), 12819.0); 
		bHM.put(new CellIndex(4, 29), 13932.0); bHM.put(new CellIndex(4, 30), 12852.0); bHM.put(new CellIndex(4, 31), 11766.0); bHM.put(new CellIndex(4, 32), 10674.0); 
		bHM.put(new CellIndex(4, 33), 11796.0); bHM.put(new CellIndex(4, 34), 10701.0); bHM.put(new CellIndex(4, 35), 11826.0); bHM.put(new CellIndex(4, 36), 10728.0); 
		bHM.put(new CellIndex(4, 37), 11856.0); bHM.put(new CellIndex(4, 38), 10755.0); bHM.put(new CellIndex(4, 39), 11886.0); bHM.put(new CellIndex(4, 40), 10782.0); 
		bHM.put(new CellIndex(4, 41), 14004.0); bHM.put(new CellIndex(4, 42), 12918.0); bHM.put(new CellIndex(4, 43), 14040.0); bHM.put(new CellIndex(4, 44), 12951.0); 
		bHM.put(new CellIndex(4, 45), 14076.0); bHM.put(new CellIndex(4, 46), 12984.0); bHM.put(new CellIndex(4, 47), 14112.0); bHM.put(new CellIndex(4, 48), 13017.0); 
		bHM.put(new CellIndex(4, 49), 14148.0); bHM.put(new CellIndex(4, 50), 13050.0); bHM.put(new CellIndex(4, 51), 11946.0); bHM.put(new CellIndex(4, 52), 10836.0); 
		bHM.put(new CellIndex(4, 53), 11976.0); bHM.put(new CellIndex(4, 54), 10863.0); bHM.put(new CellIndex(4, 55), 12006.0); bHM.put(new CellIndex(4, 56), 10890.0); 
		bHM.put(new CellIndex(4, 57), 12036.0); bHM.put(new CellIndex(4, 58), 10917.0); bHM.put(new CellIndex(4, 59), 12066.0); bHM.put(new CellIndex(4, 60), 10944.0); 
		bHM.put(new CellIndex(4, 61), 14220.0); bHM.put(new CellIndex(4, 62), 13116.0); bHM.put(new CellIndex(4, 63), 14256.0); bHM.put(new CellIndex(4, 64), 13149.0); 
		bHM.put(new CellIndex(4, 65), 14292.0); bHM.put(new CellIndex(4, 66), 13182.0); bHM.put(new CellIndex(4, 67), 14328.0); bHM.put(new CellIndex(4, 68), 13215.0); 
		bHM.put(new CellIndex(4, 69), 14364.0); bHM.put(new CellIndex(4, 70), 13248.0); bHM.put(new CellIndex(4, 71), 12126.0); bHM.put(new CellIndex(4, 72), 10998.0); 
		bHM.put(new CellIndex(4, 73), 12156.0); bHM.put(new CellIndex(4, 74), 11025.0); bHM.put(new CellIndex(4, 75), 12186.0); bHM.put(new CellIndex(4, 76), 11052.0); 
		bHM.put(new CellIndex(4, 77), 12216.0); bHM.put(new CellIndex(4, 78), 11079.0); bHM.put(new CellIndex(4, 79), 12246.0); bHM.put(new CellIndex(4, 80), 11106.0); 
		bHM.put(new CellIndex(4, 81), 14436.0); bHM.put(new CellIndex(4, 82), 13314.0); bHM.put(new CellIndex(4, 83), 14472.0); bHM.put(new CellIndex(4, 84), 13347.0); 
		bHM.put(new CellIndex(4, 85), 14508.0); bHM.put(new CellIndex(4, 86), 13380.0); bHM.put(new CellIndex(4, 87), 14544.0); bHM.put(new CellIndex(4, 88), 13413.0); 
		bHM.put(new CellIndex(4, 89), 14580.0); bHM.put(new CellIndex(4, 90), 13446.0); bHM.put(new CellIndex(4, 91), 12306.0); bHM.put(new CellIndex(4, 92), 11160.0); 
		bHM.put(new CellIndex(4, 93), 12336.0); bHM.put(new CellIndex(4, 94), 11187.0); bHM.put(new CellIndex(4, 95), 12366.0); bHM.put(new CellIndex(4, 96), 11214.0); 
		bHM.put(new CellIndex(4, 97), 12396.0); bHM.put(new CellIndex(4, 98), 11241.0); bHM.put(new CellIndex(4, 99), 12426.0); bHM.put(new CellIndex(4, 100), 11268.0); 
		bHM.put(new CellIndex(4, 101), 17904.0); bHM.put(new CellIndex(4, 102), 16866.0); bHM.put(new CellIndex(4, 103), 17952.0); bHM.put(new CellIndex(4, 104), 16911.0); 
		bHM.put(new CellIndex(4, 105), 18000.0); bHM.put(new CellIndex(4, 106), 16956.0); bHM.put(new CellIndex(4, 107), 18048.0); bHM.put(new CellIndex(4, 108), 17001.0); 
		bHM.put(new CellIndex(4, 109), 18096.0); bHM.put(new CellIndex(4, 110), 17046.0); bHM.put(new CellIndex(4, 111), 15990.0); bHM.put(new CellIndex(4, 112), 14928.0); 
		bHM.put(new CellIndex(4, 113), 16032.0); bHM.put(new CellIndex(4, 114), 14967.0); bHM.put(new CellIndex(4, 115), 16074.0); bHM.put(new CellIndex(4, 116), 15006.0); 
		bHM.put(new CellIndex(4, 117), 16116.0); bHM.put(new CellIndex(4, 118), 15045.0); bHM.put(new CellIndex(4, 119), 16158.0); bHM.put(new CellIndex(4, 120), 15084.0); 
		bHM.put(new CellIndex(4, 121), 18192.0); bHM.put(new CellIndex(4, 122), 17136.0); bHM.put(new CellIndex(4, 123), 18240.0); bHM.put(new CellIndex(4, 124), 17181.0); 
		bHM.put(new CellIndex(4, 125), 18288.0); bHM.put(new CellIndex(4, 126), 17226.0); bHM.put(new CellIndex(4, 127), 18336.0); bHM.put(new CellIndex(4, 128), 17271.0); 
		bHM.put(new CellIndex(4, 129), 18384.0); bHM.put(new CellIndex(4, 130), 17316.0); bHM.put(new CellIndex(4, 131), 16242.0); bHM.put(new CellIndex(4, 132), 15162.0); 
		bHM.put(new CellIndex(4, 133), 16284.0); bHM.put(new CellIndex(4, 134), 15201.0); bHM.put(new CellIndex(4, 135), 16326.0); bHM.put(new CellIndex(4, 136), 15240.0); 
		bHM.put(new CellIndex(4, 137), 16368.0); bHM.put(new CellIndex(4, 138), 15279.0); bHM.put(new CellIndex(4, 139), 16410.0); bHM.put(new CellIndex(4, 140), 15318.0); 
		bHM.put(new CellIndex(4, 141), 18480.0); bHM.put(new CellIndex(4, 142), 17406.0); bHM.put(new CellIndex(4, 143), 18528.0); bHM.put(new CellIndex(4, 144), 17451.0); 
		bHM.put(new CellIndex(4, 145), 18576.0); bHM.put(new CellIndex(4, 146), 17496.0); bHM.put(new CellIndex(4, 147), 18624.0); bHM.put(new CellIndex(4, 148), 17541.0); 
		bHM.put(new CellIndex(4, 149), 18672.0); bHM.put(new CellIndex(4, 150), 17586.0); bHM.put(new CellIndex(4, 151), 16494.0); bHM.put(new CellIndex(4, 152), 15396.0); 
		bHM.put(new CellIndex(4, 153), 16536.0); bHM.put(new CellIndex(4, 154), 15435.0); bHM.put(new CellIndex(4, 155), 16578.0); bHM.put(new CellIndex(4, 156), 15474.0); 
		bHM.put(new CellIndex(4, 157), 16620.0); bHM.put(new CellIndex(4, 158), 15513.0); bHM.put(new CellIndex(4, 159), 16662.0); bHM.put(new CellIndex(4, 160), 15552.0); 
		bHM.put(new CellIndex(4, 161), 18768.0); bHM.put(new CellIndex(4, 162), 17676.0); bHM.put(new CellIndex(4, 163), 18816.0); bHM.put(new CellIndex(4, 164), 17721.0); 
		bHM.put(new CellIndex(4, 165), 18864.0); bHM.put(new CellIndex(4, 166), 17766.0); bHM.put(new CellIndex(4, 167), 18912.0); bHM.put(new CellIndex(4, 168), 17811.0); 
		bHM.put(new CellIndex(4, 169), 18960.0); bHM.put(new CellIndex(4, 170), 17856.0); bHM.put(new CellIndex(4, 171), 16746.0); bHM.put(new CellIndex(4, 172), 15630.0); 
		bHM.put(new CellIndex(4, 173), 16788.0); bHM.put(new CellIndex(4, 174), 15669.0); bHM.put(new CellIndex(4, 175), 16830.0); bHM.put(new CellIndex(4, 176), 15708.0); 
		bHM.put(new CellIndex(4, 177), 16872.0); bHM.put(new CellIndex(4, 178), 15747.0); bHM.put(new CellIndex(4, 179), 16914.0); bHM.put(new CellIndex(4, 180), 15786.0); 
		bHM.put(new CellIndex(4, 181), 19056.0); bHM.put(new CellIndex(4, 182), 17946.0); bHM.put(new CellIndex(4, 183), 19104.0); bHM.put(new CellIndex(4, 184), 17991.0); 
		bHM.put(new CellIndex(4, 185), 19152.0); bHM.put(new CellIndex(4, 186), 18036.0); bHM.put(new CellIndex(4, 187), 19200.0); bHM.put(new CellIndex(4, 188), 18081.0); 
		bHM.put(new CellIndex(4, 189), 19248.0); bHM.put(new CellIndex(4, 190), 18126.0); bHM.put(new CellIndex(4, 191), 16998.0); bHM.put(new CellIndex(4, 192), 15864.0); 
		bHM.put(new CellIndex(4, 193), 17040.0); bHM.put(new CellIndex(4, 194), 15903.0); bHM.put(new CellIndex(4, 195), 17082.0); bHM.put(new CellIndex(4, 196), 15942.0); 
		bHM.put(new CellIndex(4, 197), 17124.0); bHM.put(new CellIndex(4, 198), 15981.0); bHM.put(new CellIndex(4, 199), 17166.0); bHM.put(new CellIndex(4, 200), 16020.0); 
		bHM.put(new CellIndex(5, 1), 17460.0); bHM.put(new CellIndex(5, 2), 16086.0); bHM.put(new CellIndex(5, 3), 17496.0); bHM.put(new CellIndex(5, 4), 16119.0); 
		bHM.put(new CellIndex(5, 5), 17532.0); bHM.put(new CellIndex(5, 6), 16152.0); bHM.put(new CellIndex(5, 7), 17568.0); bHM.put(new CellIndex(5, 8), 16185.0); 
		bHM.put(new CellIndex(5, 9), 17604.0); bHM.put(new CellIndex(5, 10), 16218.0); bHM.put(new CellIndex(5, 11), 14826.0); bHM.put(new CellIndex(5, 12), 13428.0); 
		bHM.put(new CellIndex(5, 13), 14856.0); bHM.put(new CellIndex(5, 14), 13455.0); bHM.put(new CellIndex(5, 15), 14886.0); bHM.put(new CellIndex(5, 16), 13482.0); 
		bHM.put(new CellIndex(5, 17), 14916.0); bHM.put(new CellIndex(5, 18), 13509.0); bHM.put(new CellIndex(5, 19), 14946.0); bHM.put(new CellIndex(5, 20), 13536.0); 
		bHM.put(new CellIndex(5, 21), 17676.0); bHM.put(new CellIndex(5, 22), 16284.0); bHM.put(new CellIndex(5, 23), 17712.0); bHM.put(new CellIndex(5, 24), 16317.0); 
		bHM.put(new CellIndex(5, 25), 17748.0); bHM.put(new CellIndex(5, 26), 16350.0); bHM.put(new CellIndex(5, 27), 17784.0); bHM.put(new CellIndex(5, 28), 16383.0); 
		bHM.put(new CellIndex(5, 29), 17820.0); bHM.put(new CellIndex(5, 30), 16416.0); bHM.put(new CellIndex(5, 31), 15006.0); bHM.put(new CellIndex(5, 32), 13590.0); 
		bHM.put(new CellIndex(5, 33), 15036.0); bHM.put(new CellIndex(5, 34), 13617.0); bHM.put(new CellIndex(5, 35), 15066.0); bHM.put(new CellIndex(5, 36), 13644.0); 
		bHM.put(new CellIndex(5, 37), 15096.0); bHM.put(new CellIndex(5, 38), 13671.0); bHM.put(new CellIndex(5, 39), 15126.0); bHM.put(new CellIndex(5, 40), 13698.0); 
		bHM.put(new CellIndex(5, 41), 17892.0); bHM.put(new CellIndex(5, 42), 16482.0); bHM.put(new CellIndex(5, 43), 17928.0); bHM.put(new CellIndex(5, 44), 16515.0); 
		bHM.put(new CellIndex(5, 45), 17964.0); bHM.put(new CellIndex(5, 46), 16548.0); bHM.put(new CellIndex(5, 47), 18000.0); bHM.put(new CellIndex(5, 48), 16581.0); 
		bHM.put(new CellIndex(5, 49), 18036.0); bHM.put(new CellIndex(5, 50), 16614.0); bHM.put(new CellIndex(5, 51), 15186.0); bHM.put(new CellIndex(5, 52), 13752.0); 
		bHM.put(new CellIndex(5, 53), 15216.0); bHM.put(new CellIndex(5, 54), 13779.0); bHM.put(new CellIndex(5, 55), 15246.0); bHM.put(new CellIndex(5, 56), 13806.0); 
		bHM.put(new CellIndex(5, 57), 15276.0); bHM.put(new CellIndex(5, 58), 13833.0); bHM.put(new CellIndex(5, 59), 15306.0); bHM.put(new CellIndex(5, 60), 13860.0); 
		bHM.put(new CellIndex(5, 61), 18108.0); bHM.put(new CellIndex(5, 62), 16680.0); bHM.put(new CellIndex(5, 63), 18144.0); bHM.put(new CellIndex(5, 64), 16713.0); 
		bHM.put(new CellIndex(5, 65), 18180.0); bHM.put(new CellIndex(5, 66), 16746.0); bHM.put(new CellIndex(5, 67), 18216.0); bHM.put(new CellIndex(5, 68), 16779.0); 
		bHM.put(new CellIndex(5, 69), 18252.0); bHM.put(new CellIndex(5, 70), 16812.0); bHM.put(new CellIndex(5, 71), 15366.0); bHM.put(new CellIndex(5, 72), 13914.0); 
		bHM.put(new CellIndex(5, 73), 15396.0); bHM.put(new CellIndex(5, 74), 13941.0); bHM.put(new CellIndex(5, 75), 15426.0); bHM.put(new CellIndex(5, 76), 13968.0); 
		bHM.put(new CellIndex(5, 77), 15456.0); bHM.put(new CellIndex(5, 78), 13995.0); bHM.put(new CellIndex(5, 79), 15486.0); bHM.put(new CellIndex(5, 80), 14022.0); 
		bHM.put(new CellIndex(5, 81), 18324.0); bHM.put(new CellIndex(5, 82), 16878.0); bHM.put(new CellIndex(5, 83), 18360.0); bHM.put(new CellIndex(5, 84), 16911.0); 
		bHM.put(new CellIndex(5, 85), 18396.0); bHM.put(new CellIndex(5, 86), 16944.0); bHM.put(new CellIndex(5, 87), 18432.0); bHM.put(new CellIndex(5, 88), 16977.0); 
		bHM.put(new CellIndex(5, 89), 18468.0); bHM.put(new CellIndex(5, 90), 17010.0); bHM.put(new CellIndex(5, 91), 15546.0); bHM.put(new CellIndex(5, 92), 14076.0); 
		bHM.put(new CellIndex(5, 93), 15576.0); bHM.put(new CellIndex(5, 94), 14103.0); bHM.put(new CellIndex(5, 95), 15606.0); bHM.put(new CellIndex(5, 96), 14130.0); 
		bHM.put(new CellIndex(5, 97), 15636.0); bHM.put(new CellIndex(5, 98), 14157.0); bHM.put(new CellIndex(5, 99), 15666.0); bHM.put(new CellIndex(5, 100), 14184.0); 
		bHM.put(new CellIndex(5, 101), 23088.0); bHM.put(new CellIndex(5, 102), 21726.0); bHM.put(new CellIndex(5, 103), 23136.0); bHM.put(new CellIndex(5, 104), 21771.0); 
		bHM.put(new CellIndex(5, 105), 23184.0); bHM.put(new CellIndex(5, 106), 21816.0); bHM.put(new CellIndex(5, 107), 23232.0); bHM.put(new CellIndex(5, 108), 21861.0); 
		bHM.put(new CellIndex(5, 109), 23280.0); bHM.put(new CellIndex(5, 110), 21906.0); bHM.put(new CellIndex(5, 111), 20526.0); bHM.put(new CellIndex(5, 112), 19140.0); 
		bHM.put(new CellIndex(5, 113), 20568.0); bHM.put(new CellIndex(5, 114), 19179.0); bHM.put(new CellIndex(5, 115), 20610.0); bHM.put(new CellIndex(5, 116), 19218.0); 
		bHM.put(new CellIndex(5, 117), 20652.0); bHM.put(new CellIndex(5, 118), 19257.0); bHM.put(new CellIndex(5, 119), 20694.0); bHM.put(new CellIndex(5, 120), 19296.0); 
		bHM.put(new CellIndex(5, 121), 23376.0); bHM.put(new CellIndex(5, 122), 21996.0); bHM.put(new CellIndex(5, 123), 23424.0); bHM.put(new CellIndex(5, 124), 22041.0); 
		bHM.put(new CellIndex(5, 125), 23472.0); bHM.put(new CellIndex(5, 126), 22086.0); bHM.put(new CellIndex(5, 127), 23520.0); bHM.put(new CellIndex(5, 128), 22131.0); 
		bHM.put(new CellIndex(5, 129), 23568.0); bHM.put(new CellIndex(5, 130), 22176.0); bHM.put(new CellIndex(5, 131), 20778.0); bHM.put(new CellIndex(5, 132), 19374.0); 
		bHM.put(new CellIndex(5, 133), 20820.0); bHM.put(new CellIndex(5, 134), 19413.0); bHM.put(new CellIndex(5, 135), 20862.0); bHM.put(new CellIndex(5, 136), 19452.0); 
		bHM.put(new CellIndex(5, 137), 20904.0); bHM.put(new CellIndex(5, 138), 19491.0); bHM.put(new CellIndex(5, 139), 20946.0); bHM.put(new CellIndex(5, 140), 19530.0); 
		bHM.put(new CellIndex(5, 141), 23664.0); bHM.put(new CellIndex(5, 142), 22266.0); bHM.put(new CellIndex(5, 143), 23712.0); bHM.put(new CellIndex(5, 144), 22311.0); 
		bHM.put(new CellIndex(5, 145), 23760.0); bHM.put(new CellIndex(5, 146), 22356.0); bHM.put(new CellIndex(5, 147), 23808.0); bHM.put(new CellIndex(5, 148), 22401.0); 
		bHM.put(new CellIndex(5, 149), 23856.0); bHM.put(new CellIndex(5, 150), 22446.0); bHM.put(new CellIndex(5, 151), 21030.0); bHM.put(new CellIndex(5, 152), 19608.0); 
		bHM.put(new CellIndex(5, 153), 21072.0); bHM.put(new CellIndex(5, 154), 19647.0); bHM.put(new CellIndex(5, 155), 21114.0); bHM.put(new CellIndex(5, 156), 19686.0); 
		bHM.put(new CellIndex(5, 157), 21156.0); bHM.put(new CellIndex(5, 158), 19725.0); bHM.put(new CellIndex(5, 159), 21198.0); bHM.put(new CellIndex(5, 160), 19764.0); 
		bHM.put(new CellIndex(5, 161), 23952.0); bHM.put(new CellIndex(5, 162), 22536.0); bHM.put(new CellIndex(5, 163), 24000.0); bHM.put(new CellIndex(5, 164), 22581.0); 
		bHM.put(new CellIndex(5, 165), 24048.0); bHM.put(new CellIndex(5, 166), 22626.0); bHM.put(new CellIndex(5, 167), 24096.0); bHM.put(new CellIndex(5, 168), 22671.0); 
		bHM.put(new CellIndex(5, 169), 24144.0); bHM.put(new CellIndex(5, 170), 22716.0); bHM.put(new CellIndex(5, 171), 21282.0); bHM.put(new CellIndex(5, 172), 19842.0); 
		bHM.put(new CellIndex(5, 173), 21324.0); bHM.put(new CellIndex(5, 174), 19881.0); bHM.put(new CellIndex(5, 175), 21366.0); bHM.put(new CellIndex(5, 176), 19920.0); 
		bHM.put(new CellIndex(5, 177), 21408.0); bHM.put(new CellIndex(5, 178), 19959.0); bHM.put(new CellIndex(5, 179), 21450.0); bHM.put(new CellIndex(5, 180), 19998.0); 
		bHM.put(new CellIndex(5, 181), 24240.0); bHM.put(new CellIndex(5, 182), 22806.0); bHM.put(new CellIndex(5, 183), 24288.0); bHM.put(new CellIndex(5, 184), 22851.0); 
		bHM.put(new CellIndex(5, 185), 24336.0); bHM.put(new CellIndex(5, 186), 22896.0); bHM.put(new CellIndex(5, 187), 24384.0); bHM.put(new CellIndex(5, 188), 22941.0); 
		bHM.put(new CellIndex(5, 189), 24432.0); bHM.put(new CellIndex(5, 190), 22986.0); bHM.put(new CellIndex(5, 191), 21534.0); bHM.put(new CellIndex(5, 192), 20076.0); 
		bHM.put(new CellIndex(5, 193), 21576.0); bHM.put(new CellIndex(5, 194), 20115.0); bHM.put(new CellIndex(5, 195), 21618.0); bHM.put(new CellIndex(5, 196), 20154.0); 
		bHM.put(new CellIndex(5, 197), 21660.0); bHM.put(new CellIndex(5, 198), 20193.0); bHM.put(new CellIndex(5, 199), 21702.0); bHM.put(new CellIndex(5, 200), 20232.0); 
	}
}
