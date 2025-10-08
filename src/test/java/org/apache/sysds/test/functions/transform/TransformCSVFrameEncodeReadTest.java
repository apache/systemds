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

import static org.junit.Assert.fail;

import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;
import org.apache.sysds.api.DMLScript;
import org.apache.sysds.common.Types.ExecMode;
import org.apache.sysds.runtime.frame.data.FrameBlock;
import org.apache.sysds.runtime.io.FileFormatPropertiesCSV;
import org.apache.sysds.runtime.io.FrameReader;
import org.apache.sysds.runtime.io.FrameReaderTextCSV;
import org.apache.sysds.runtime.io.FrameReaderTextCSVParallel;
import org.apache.sysds.runtime.util.DataConverter;
import org.apache.sysds.test.AutomatedTestBase;
import org.apache.sysds.test.TestConfiguration;
import org.apache.sysds.test.TestUtils;
import org.junit.Test;


public class TransformCSVFrameEncodeReadTest extends AutomatedTestBase {
	protected static final Log LOG = LogFactory.getLog(TransformCSVFrameEncodeReadTest.class.getName());

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
		runTransformTest(ExecMode.SINGLE_NODE, "csv", false, false);
	}
	
	@Test
	public void testFrameReadMetaSparkCSV() {
		runTransformTest(ExecMode.SPARK, "csv", false, false);
	}
	
	@Test
	public void testFrameReadMetaHybridCSV() {
		runTransformTest(ExecMode.HYBRID, "csv", false, false);
	}
	
	@Test
	public void testFrameParReadMetaSingleNodeCSV() {
		runTransformTest(ExecMode.SINGLE_NODE, "csv", false, true);
	}
	
	@Test
	public void testFrameParReadMetaSparkCSV() {
		runTransformTest(ExecMode.SPARK, "csv", false, true);
	}
	
	@Test
	public void testFrameParReadMetaHybridCSV() {
		runTransformTest(ExecMode.HYBRID, "csv", false, true);
	}

	@Test
	public void testFrameReadSubMetaSingleNodeCSV() {
		runTransformTest(ExecMode.SINGLE_NODE, "csv", true, false);
	}
	
	@Test
	public void testFrameReadSubMetaSparkCSV() {
		runTransformTest(ExecMode.SPARK, "csv", true, false);
	}
	
	@Test
	public void testFrameReadSubMetaHybridCSV() {
		runTransformTest(ExecMode.HYBRID, "csv", true, false);
	}
	
	@Test
	public void testFrameParReadSubMetaSingleNodeCSV() {
		runTransformTest(ExecMode.SINGLE_NODE, "csv", true, true);
	}
	
	@Test
	public void testFrameParReadSubMetaSparkCSV() {
		runTransformTest(ExecMode.SPARK, "csv", true, true);
	}
	
	@Test
	public void testFrameParReadSubMetaHybridCSV() {
		runTransformTest(ExecMode.HYBRID, "csv", true, true);
	}
	
	private void runTransformTest( ExecMode rt, String ofmt, boolean subset, boolean parRead )
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
			setOutputBuffering(true);
			
			String HOME = SCRIPT_DIR + TEST_DIR;
			int nrows = subset ? 4 : 13;
			fullDMLScriptName = HOME + TEST_NAME1 + ".dml";
			programArgs = new String[]{"-args", 
				DATASET_DIR + DATASET, String.valueOf(nrows), output("R") };
			
			String stdOut = runTest(null).toString();

			//read input/output and compare
			FrameReader reader2 = parRead ? 
				new FrameReaderTextCSVParallel( new FileFormatPropertiesCSV() ) : 
				new FrameReaderTextCSV( new FileFormatPropertiesCSV()  );
			FrameBlock fb2 = reader2.readFrameFromHDFS(output("R"), -1L, -1L);
			String[] fromDisk = DataConverter.toString(fb2).split("\n");
			String[] printed = stdOut.split("\n");
			boolean equal = true;
			StringBuilder err = new StringBuilder();
			for(int i = 0; i < fromDisk.length; i++){
				if(! fromDisk[i].strip().equals(printed[i].strip())){
					err.append("\n not equal: \n'"+ (fromDisk[i] + "'\n'" + printed[i] + "'"));
					equal = false;
				}
				
			}
			if(!equal)
				fail(err.toString());
			
		}
		catch(Exception ex) {
			ex.printStackTrace();
			throw new RuntimeException(ex);
		}
		finally {
			rtplatform = rtold;
			DMLScript.USE_LOCAL_SPARK_CONFIG = sparkConfigOld;
		}
	}
}