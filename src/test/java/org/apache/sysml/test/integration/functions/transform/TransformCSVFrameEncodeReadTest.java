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

import org.junit.Test;
import org.apache.sysml.api.DMLScript;
import org.apache.sysml.api.DMLScript.RUNTIME_PLATFORM;
import org.apache.sysml.runtime.io.FrameReader;
import org.apache.sysml.runtime.io.FrameReaderTextCSV;
import org.apache.sysml.runtime.io.FrameReaderTextCSVParallel;
import org.apache.sysml.runtime.matrix.data.CSVFileFormatProperties;
import org.apache.sysml.runtime.matrix.data.FrameBlock;
import org.apache.sysml.runtime.util.DataConverter;
import org.apache.sysml.test.integration.AutomatedTestBase;
import org.apache.sysml.test.integration.TestConfiguration;
import org.apache.sysml.test.utils.TestUtils;

public class TransformCSVFrameEncodeReadTest extends AutomatedTestBase 
{
	private final static String TEST_NAME1 = "TransformCSVFrameEncodeRead";
	private final static String TEST_DIR = "functions/transform/";
	private final static String TEST_CLASS_DIR = TEST_DIR + TransformCSVFrameEncodeReadTest.class.getSimpleName() + "/";
	
	//dataset and transform tasks without missing values
	private final static String DATASET 	= "csv_mix/quotes1.csv";
	
	@Override
	public void setUp()  {
		TestUtils.clearAssertionInformation();
		addTestConfiguration(TEST_NAME1, 
			new TestConfiguration(TEST_CLASS_DIR, TEST_NAME1, new String[] { "R" }) );
	}
	
	@Test
	public void testFrameReadMetaSingleNodeCSV() {
		runTransformTest(RUNTIME_PLATFORM.SINGLE_NODE, "csv", false, false);
	}
	
	@Test
	public void testFrameReadMetaSparkCSV() {
		runTransformTest(RUNTIME_PLATFORM.SPARK, "csv", false, false);
	}
	
	@Test
	public void testFrameReadMetaHybridCSV() {
		runTransformTest(RUNTIME_PLATFORM.HYBRID_SPARK, "csv", false, false);
	}
	
	@Test
	public void testFrameParReadMetaSingleNodeCSV() {
		runTransformTest(RUNTIME_PLATFORM.SINGLE_NODE, "csv", false, true);
	}
	
	@Test
	public void testFrameParReadMetaSparkCSV() {
		runTransformTest(RUNTIME_PLATFORM.SPARK, "csv", false, true);
	}
	
	@Test
	public void testFrameParReadMetaHybridCSV() {
		runTransformTest(RUNTIME_PLATFORM.HYBRID_SPARK, "csv", false, true);
	}

	@Test
	public void testFrameReadSubMetaSingleNodeCSV() {
		runTransformTest(RUNTIME_PLATFORM.SINGLE_NODE, "csv", true, false);
	}
	
	@Test
	public void testFrameReadSubMetaSparkCSV() {
		runTransformTest(RUNTIME_PLATFORM.SPARK, "csv", true, false);
	}
	
	@Test
	public void testFrameReadSubMetaHybridCSV() {
		runTransformTest(RUNTIME_PLATFORM.HYBRID_SPARK, "csv", true, false);
	}
	
	@Test
	public void testFrameParReadSubMetaSingleNodeCSV() {
		runTransformTest(RUNTIME_PLATFORM.SINGLE_NODE, "csv", true, true);
	}
	
	@Test
	public void testFrameParReadSubMetaSparkCSV() {
		runTransformTest(RUNTIME_PLATFORM.SPARK, "csv", true, true);
	}
	
	@Test
	public void testFrameParReadSubMetaHybridCSV() {
		runTransformTest(RUNTIME_PLATFORM.HYBRID_SPARK, "csv", true, true);
	}

	
	/**
	 * 
	 * @param rt
	 * @param ofmt
	 * @param dataset
	 */
	private void runTransformTest( RUNTIME_PLATFORM rt, String ofmt, boolean subset, boolean parRead )
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
			int nrows = subset ? 4 : 13;
			fullDMLScriptName = HOME + TEST_NAME1 + ".dml";
			programArgs = new String[]{"-explain", "-stats","-args", 
				HOME + "input/" + DATASET, String.valueOf(nrows), output("R") };
			
			runTest(true, false, null, -1); 
			
			//read input/output and compare
			FrameReader reader2 = parRead ? 
				new FrameReaderTextCSVParallel( new CSVFileFormatProperties() ) : 
				new FrameReaderTextCSV( new CSVFileFormatProperties()  );
			FrameBlock fb2 = reader2.readFrameFromHDFS(output("R"), -1L, -1L);
			System.out.println(DataConverter.toString(fb2));
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