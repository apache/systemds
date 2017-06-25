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

package org.apache.sysml.test.integration.functions.transform;

import org.junit.Assert;
import org.junit.Test;
import org.apache.sysml.api.DMLScript;
import org.apache.sysml.api.DMLScript.RUNTIME_PLATFORM;
import org.apache.sysml.runtime.io.FrameReader;
import org.apache.sysml.runtime.io.FrameReaderFactory;
import org.apache.sysml.runtime.matrix.data.CSVFileFormatProperties;
import org.apache.sysml.runtime.matrix.data.FrameBlock;
import org.apache.sysml.runtime.matrix.data.InputInfo;
import org.apache.sysml.runtime.util.DataConverter;
import org.apache.sysml.test.integration.AutomatedTestBase;
import org.apache.sysml.test.integration.TestConfiguration;
import org.apache.sysml.test.utils.TestUtils;
import org.apache.sysml.utils.Statistics;

/**
 * This test is similar to TransformFrameEncodeDecode test but specifically
 * tests for various problematic tokens such as ", ', etc.
 * 
 */
public class TransformFrameEncodeDecodeTokenTest extends AutomatedTestBase 
{
	private final static String TEST_NAME1 = "TransformFrameEncodeDecode";
	private final static String TEST_DIR = "functions/transform/";
	private final static String TEST_CLASS_DIR = TEST_DIR + TransformFrameEncodeDecodeTokenTest.class.getSimpleName() + "/";
	
	//dataset and transform tasks without missing values
	private final static String DATASET1 	= "20news/20news_subset.csv";
	private final static String SPEC1 		= "20news/20news.tfspec_recode.json";
	
	@Override
	public void setUp()  {
		TestUtils.clearAssertionInformation();
		addTestConfiguration(TEST_NAME1, new TestConfiguration(TEST_CLASS_DIR, TEST_NAME1, new String[] { "F2" }) );
	}
	
	@Test
	public void test20newsRecodeSingleNodeCSV() {
		runTransformTest(RUNTIME_PLATFORM.SINGLE_NODE, "csv");
	}
	
	@Test
	public void test20newsRecodeSparkCSV() {
		runTransformTest(RUNTIME_PLATFORM.SPARK, "csv");
	}
	
	@Test
	public void test20newsRecodeHybridCSV() {
		runTransformTest(RUNTIME_PLATFORM.HYBRID_SPARK, "csv");
	}
	
	/**
	 * 
	 * @param rt
	 * @param ofmt
	 * @param dataset
	 */
	private void runTransformTest( RUNTIME_PLATFORM rt, String ofmt )
	{
		//set runtime platform
		RUNTIME_PLATFORM rtold = rtplatform;
		rtplatform = rt;

		boolean sparkConfigOld = DMLScript.USE_LOCAL_SPARK_CONFIG;
		if( rtplatform == RUNTIME_PLATFORM.SPARK || rtplatform == RUNTIME_PLATFORM.HYBRID_SPARK)
			DMLScript.USE_LOCAL_SPARK_CONFIG = true;

		if( !ofmt.equals("csv") )
			throw new RuntimeException("Unsupported test output format");
		
		try
		{
			getAndLoadTestConfiguration(TEST_NAME1);
			
			String HOME = SCRIPT_DIR + TEST_DIR;
			fullDMLScriptName = HOME + TEST_NAME1 + ".dml";
			programArgs = new String[]{"-explain","-nvargs", 
				"DATA=" + HOME + "input/" + DATASET1,
				"TFSPEC=" + HOME + "input/" + SPEC1,
				"TFDATA=" + output("tfout"), "SEP= ",
				"OFMT=" + ofmt, "OSEP= " };
	
			runTest(true, false, null, -1); 
			
			//read input/output and compare
			FrameReader reader1 = FrameReaderFactory.createFrameReader(InputInfo.CSVInputInfo, 
					new CSVFileFormatProperties(false, " ", false));
			FrameBlock fb1 = reader1.readFrameFromHDFS(HOME + "input/" + DATASET1, -1L, -1L);
			FrameReader reader2 = FrameReaderFactory.createFrameReader(InputInfo.CSVInputInfo,
					new CSVFileFormatProperties(false, " ", false));
			FrameBlock fb2 = reader2.readFrameFromHDFS(output("tfout"), -1L, -1L);
			String[][] R1 = DataConverter.convertToStringFrame(fb1);
			String[][] R2 = DataConverter.convertToStringFrame(fb2);
			TestUtils.compareFrames(R1, R2, R1.length, R1[0].length);			
			
			if( rt == RUNTIME_PLATFORM.HYBRID_SPARK ) {
				Assert.assertEquals("Wrong number of executed Spark instructions: " + 
					Statistics.getNoOfExecutedSPInst(), new Long(2), new Long(Statistics.getNoOfExecutedSPInst()));
			}
		}
		catch(Exception ex) {
			throw new RuntimeException(ex);
		}
		finally {
			rtplatform = rtold;
			DMLScript.USE_LOCAL_SPARK_CONFIG = sparkConfigOld;
		}
	}
}