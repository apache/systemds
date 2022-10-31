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

public class TransformFrameEncodeDecodeTest extends AutomatedTestBase 
{
	private final static String TEST_NAME1 = "TransformFrameEncodeDecode";
	private final static String TEST_DIR = "functions/transform/";
	private final static String TEST_CLASS_DIR = TEST_DIR + TransformFrameEncodeDecodeTest.class.getSimpleName() + "/";
	
	//dataset and transform tasks without missing values
	private final static String DATASET1 = "homes3/homes.csv";
	private final static String SPEC1    = "homes3/homes.tfspec_recode.json"; 
	private final static String SPEC1b   = "homes3/homes.tfspec_recode2.json"; 
	private final static String SPEC2    = "homes3/homes.tfspec_dummy.json";
	private final static String SPEC2b   = "homes3/homes.tfspec_dummy2.json";
	
	public enum TransformType {
		RECODE,
		DUMMY,
		BIN,
		IMPUTE,
		OMIT,
	}
	
	@Override
	public void setUp()  {
		TestUtils.clearAssertionInformation();
		addTestConfiguration(TEST_NAME1, 
			new TestConfiguration(TEST_CLASS_DIR, TEST_NAME1, new String[] { "y" }) );
	}
	
	@Test
	public void testHomesRecodeIDsSingleNodeCSV() {
		runTransformTest(ExecMode.SINGLE_NODE, "csv", TransformType.RECODE, false);
	}
	
	@Test
	public void testHomesRecodeIDsSparkCSV() {
		runTransformTest(ExecMode.SPARK, "csv", TransformType.RECODE, false);
	}
	
	@Test
	public void testHomesRecodeIDsHybridCSV() {
		runTransformTest(ExecMode.HYBRID, "csv", TransformType.RECODE, false);
	}
	
	@Test
	public void testHomesDummycodeIDsSingleNodeCSV() {
		runTransformTest(ExecMode.SINGLE_NODE, "csv", TransformType.DUMMY, false);
	}
	
	@Test
	public void testHomesDummycodeIDsSparkCSV() {
		runTransformTest(ExecMode.SPARK, "csv", TransformType.DUMMY, false);
	}
	
	@Test
	public void testHomesDummycodeIDsHybridCSV() {
		runTransformTest(ExecMode.HYBRID, "csv", TransformType.DUMMY, false);
	}
	
	@Test
	public void testHomesRecodeColnamesSingleNodeCSV() {
		runTransformTest(ExecMode.SINGLE_NODE, "csv", TransformType.RECODE, true);
	}
	
	@Test
	public void testHomesRecodeColnamesSparkCSV() {
		runTransformTest(ExecMode.SPARK, "csv", TransformType.RECODE, true);
	}
	
	@Test
	public void testHomesRecodeColnamesHybridCSV() {
		runTransformTest(ExecMode.HYBRID, "csv", TransformType.RECODE, true);
	}
	
	@Test
	public void testHomesDummycodeColnamesSingleNodeCSV() {
		runTransformTest(ExecMode.SINGLE_NODE, "csv", TransformType.DUMMY, true);
	}
	
	@Test
	public void testHomesDummycodeColnamesSparkCSV() {
		runTransformTest(ExecMode.SPARK, "csv", TransformType.DUMMY, true);
	}
	
	@Test
	public void testHomesDummycodeColnamesHybridCSV() {
		runTransformTest(ExecMode.HYBRID, "csv", TransformType.DUMMY, true);
	}
	
	private void runTransformTest( ExecMode rt, String ofmt, TransformType type, boolean colnames )
	{
		//set runtime platform
		ExecMode rtold = rtplatform;
		rtplatform = rt;

		boolean sparkConfigOld = DMLScript.USE_LOCAL_SPARK_CONFIG;
		if( rtplatform == ExecMode.SPARK || rtplatform == ExecMode.HYBRID)
			DMLScript.USE_LOCAL_SPARK_CONFIG = true;

		//set transform specification
		String SPEC = null; String DATASET = null;
		switch( type ) {
			case RECODE: SPEC = colnames?SPEC1b:SPEC1; DATASET = DATASET1; break;
			case DUMMY:  SPEC = colnames?SPEC2b:SPEC2; DATASET = DATASET1; break;
			default: throw new RuntimeException("Unsupported transform type for encode/decode test.");
		}

		if( !ofmt.equals("csv") )
			throw new RuntimeException("Unsupported test output format");
		
		try
		{
			getAndLoadTestConfiguration(TEST_NAME1);
			
			String HOME = SCRIPT_DIR + TEST_DIR;
			fullDMLScriptName = HOME + TEST_NAME1 + ".dml";
			programArgs = new String[]{"-explain","-nvargs", 
				"DATA=" + DATASET_DIR + DATASET,
				"TFSPEC=" + DATASET_DIR + SPEC,
				"TFDATA=" + output("tfout"), "SEP=,",
				"OFMT=" + ofmt, "OSEP=," };

			// Originally OSEP was set to
			// OSEP=","
			// Apache Commons CLI strips away the leading and trailing quotes, leaving us with
			// OSEP=",
			// This is just a feature/bug and is reported in CLI-262,
			// though even a fix is unlikely to be backported to 1.2

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
