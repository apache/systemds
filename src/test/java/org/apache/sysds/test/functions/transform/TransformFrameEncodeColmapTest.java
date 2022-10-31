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

public class TransformFrameEncodeColmapTest extends AutomatedTestBase 
{
	private final static String TEST_NAME1 = "TransformFrameEncodeColmap1";
	private final static String TEST_NAME2 = "TransformFrameEncodeColmap2";
	
	private final static String TEST_DIR = "functions/transform/";
	private final static String TEST_CLASS_DIR = TEST_DIR + TransformFrameEncodeColmapTest.class.getSimpleName() + "/";
	
	//dataset and transform tasks without missing values
	private final static String DATASET1 = "homes3/homes.csv";
	private final static String SPEC1    = "homes3/homes.tfspec_colmap1.json"; 
	private final static String SPEC1b   = "homes3/homes.tfspec_colmap2.json"; 
	
	@Override
	public void setUp()  {
		TestUtils.clearAssertionInformation();
		addTestConfiguration(TEST_NAME1, new TestConfiguration(TEST_CLASS_DIR, TEST_NAME1, new String[] { "y" }) );
		addTestConfiguration(TEST_NAME2, new TestConfiguration(TEST_CLASS_DIR, TEST_NAME2, new String[] { "y" }) );
	}
	
	@Test
	public void testHomesIDsSingleNode1() {
		runTransformTest(TEST_NAME1, ExecMode.SINGLE_NODE, "csv", false);
	}
	
	@Test
	public void testHomesColnamesSingleNode1() {
		runTransformTest(TEST_NAME1, ExecMode.SINGLE_NODE, "csv", true);
	}
	
	@Test
	public void testHomesIDsSpark1() {
		runTransformTest(TEST_NAME1, ExecMode.SPARK, "csv", false);
	}
	
	@Test
	public void testHomesColnamesSpark1() {
		runTransformTest(TEST_NAME1, ExecMode.SPARK, "csv", true);
	}
	
	@Test
	public void testHomesIDsSingleNode2() {
		runTransformTest(TEST_NAME2, ExecMode.SINGLE_NODE, "csv", false);
	}
	
	@Test
	public void testHomesColnamesSingleNode2() {
		runTransformTest(TEST_NAME2, ExecMode.SINGLE_NODE, "csv", true);
	}
	
	@Test
	public void testHomesIDsSpark2() {
		runTransformTest(TEST_NAME2, ExecMode.SPARK, "csv", false);
	}
	
	@Test
	public void testHomesColnamesSpark2() {
		runTransformTest(TEST_NAME2, ExecMode.SPARK, "csv", true);
	}
	
	private void runTransformTest( String testname, ExecMode rt, String ofmt, boolean colnames )
	{
		//set runtime platform
		ExecMode rtold = rtplatform;
		rtplatform = rt;

		boolean sparkConfigOld = DMLScript.USE_LOCAL_SPARK_CONFIG;
		if( rtplatform == ExecMode.SPARK || rtplatform == ExecMode.HYBRID)
			DMLScript.USE_LOCAL_SPARK_CONFIG = true;

		//set transform specification
		String DATASET = DATASET1;
		String SPEC = colnames?SPEC1b:SPEC1;

		if( !ofmt.equals("csv") )
			throw new RuntimeException("Unsupported test output format");
		
		try
		{
			getAndLoadTestConfiguration(testname);
			
			String HOME = SCRIPT_DIR + TEST_DIR;
			fullDMLScriptName = HOME + testname + ".dml";
			programArgs = new String[]{"-explain","-nvargs", 
				"DATA=" + DATASET_DIR + DATASET,
				"TFSPEC=" + DATASET_DIR + SPEC,
				"TFDATA=" + output("tfout"), 
				"OFMT=" + ofmt, "OSEP=," };
			
			runTest(true, false, null, -1); 
			
			//read input/output and compare
			FrameReader reader1 = FrameReaderFactory.createFrameReader(FileFormat.CSV, 
				new FileFormatPropertiesCSV(true, ",", false));
			FrameBlock fb1 = reader1.readFrameFromHDFS(DATASET_DIR + DATASET, -1L, -1L);
			FrameReader reader2 = FrameReaderFactory.createFrameReader(FileFormat.CSV);
			FrameBlock fb2 = reader2.readFrameFromHDFS(output("tfout"), -1L, -1L);
			String[][] R1 = DataConverter.convertToStringFrame(fb1);
			String[][] R2 = DataConverter.convertToStringFrame(fb2);
			TestUtils.compareFrames(R1, R2, R1.length, R1[0].length);
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