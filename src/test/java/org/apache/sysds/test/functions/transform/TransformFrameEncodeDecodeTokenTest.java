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

package org.apache.sysds.test.functions.transform;

import org.junit.Assert;
import org.junit.Test;
import org.apache.sysds.api.DMLScript;
import org.apache.sysds.common.Types.ExecMode;
import org.apache.sysds.common.Types.FileFormat;
import org.apache.sysds.runtime.frame.data.FrameBlock;
import org.apache.sysds.runtime.io.FileFormatPropertiesCSV;
import org.apache.sysds.runtime.io.FrameReader;
import org.apache.sysds.runtime.io.FrameReaderFactory;
import org.apache.sysds.runtime.util.DataConverter;
import org.apache.sysds.test.AutomatedTestBase;
import org.apache.sysds.test.TestConfiguration;
import org.apache.sysds.test.TestUtils;
import org.apache.sysds.utils.Statistics;

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
		runTransformTest(ExecMode.SINGLE_NODE, "csv");
	}
	
	@Test
	public void test20newsRecodeSparkCSV() {
		runTransformTest(ExecMode.SPARK, "csv");
	}
	
	@Test
	public void test20newsRecodeHybridCSV() {
		runTransformTest(ExecMode.HYBRID, "csv");
	}
	
	private void runTransformTest( ExecMode rt, String ofmt )
	{
		//set runtime platform
		ExecMode rtold = rtplatform;
		rtplatform = rt;

		boolean sparkConfigOld = DMLScript.USE_LOCAL_SPARK_CONFIG;
		if( rtplatform == ExecMode.SPARK || rtplatform == ExecMode.HYBRID)
			DMLScript.USE_LOCAL_SPARK_CONFIG = true;

		if( !ofmt.equals("csv") )
			throw new RuntimeException("Unsupported test output format");
		
		try
		{
			getAndLoadTestConfiguration(TEST_NAME1);
			
			String HOME = SCRIPT_DIR + TEST_DIR;
			fullDMLScriptName = HOME + TEST_NAME1 + ".dml";
			programArgs = new String[]{"-explain","-nvargs", 
				"DATA=" + DATASET_DIR + DATASET1,
				"TFSPEC=" + DATASET_DIR+ SPEC1,
				"TFDATA=" + output("tfout"), "SEP= ",
				"OFMT=" + ofmt, "OSEP= " };
	
			runTest(true, false, null, -1);
			
			//read input/output and compare
			FrameReader reader1 = FrameReaderFactory.createFrameReader(FileFormat.CSV, 
					new FileFormatPropertiesCSV(false, " ", false));
			FrameBlock fb1 = reader1.readFrameFromHDFS(DATASET_DIR + DATASET1, -1L, -1L);
			FrameReader reader2 = FrameReaderFactory.createFrameReader(FileFormat.CSV,
					new FileFormatPropertiesCSV(false, " ", false));
			FrameBlock fb2 = reader2.readFrameFromHDFS(output("tfout"), -1L, -1L);
			String[][] R1 = DataConverter.convertToStringFrame(fb1);
			String[][] R2 = DataConverter.convertToStringFrame(fb2);
			TestUtils.compareFrames(R1, R2, R1.length, R1[0].length);
			
			if( rt == ExecMode.HYBRID ) {
				Assert.assertEquals("Wrong number of executed Spark instructions: " + 
					Statistics.getNoOfExecutedSPInst(), Long.valueOf(0),
					Long.valueOf(Statistics.getNoOfExecutedSPInst()));
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