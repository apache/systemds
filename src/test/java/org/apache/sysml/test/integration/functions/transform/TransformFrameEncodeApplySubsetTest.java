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
import org.apache.sysml.runtime.matrix.data.MatrixValue.CellIndex;
import org.apache.sysml.test.integration.AutomatedTestBase;
import org.apache.sysml.test.integration.TestConfiguration;
import org.apache.sysml.test.utils.TestUtils;

public class TransformFrameEncodeApplySubsetTest extends AutomatedTestBase 
{
	private final static String TEST_NAME1 = "TransformFrameEncodeApplySubset1";
	private final static String TEST_NAME2 = "TransformFrameEncodeApplySubset2";
	
	private final static String TEST_DIR = "functions/transform/";
	private final static String TEST_CLASS_DIR = TEST_DIR + TransformFrameEncodeApplySubsetTest.class.getSimpleName() + "/";
	
	//dataset and transform tasks without missing values
	private final static String DATASET1 	= "homes3/homes.csv";
	
	@Override
	public void setUp()  {
		TestUtils.clearAssertionInformation();
		addTestConfiguration(TEST_NAME1, new TestConfiguration(TEST_CLASS_DIR, TEST_NAME1, new String[] { "y" }) );
		addTestConfiguration(TEST_NAME2, new TestConfiguration(TEST_CLASS_DIR, TEST_NAME2, new String[] { "y" }) );
	}
	
	@Test
	public void testHomesRecodeColnames1SingleNodeCSV() {
		runTransformTest(TEST_NAME1, RUNTIME_PLATFORM.SINGLE_NODE, "csv", true);
	}
	
	@Test
	public void testHomesRecodeColnames1SparkCSV() {
		runTransformTest(TEST_NAME1, RUNTIME_PLATFORM.SPARK, "csv", true);
	}
	
	@Test
	public void testHomesRecodeColnames1HybridCSV() {
		runTransformTest(TEST_NAME1, RUNTIME_PLATFORM.HYBRID_SPARK, "csv", true);
	}
	
	@Test
	public void testHomesRecodeColnames2SingleNodeCSV() {
		runTransformTest(TEST_NAME2, RUNTIME_PLATFORM.SINGLE_NODE, "csv", true);
	}
	
	@Test
	public void testHomesRecodeColnames2SparkCSV() {
		runTransformTest(TEST_NAME2, RUNTIME_PLATFORM.SPARK, "csv", true);
	}
	
	@Test
	public void testHomesRecodeColnames2HybridCSV() {
		runTransformTest(TEST_NAME2, RUNTIME_PLATFORM.HYBRID_SPARK, "csv", true);
	}
	
	
	/**
	 * 
	 * @param rt
	 * @param ofmt
	 * @param dataset
	 */
	private void runTransformTest(String testname, RUNTIME_PLATFORM rt, String ofmt, boolean colnames)
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
			getAndLoadTestConfiguration(testname);
			
			String HOME = SCRIPT_DIR + TEST_DIR;
			fullDMLScriptName = HOME + testname + ".dml";
			programArgs = new String[]{"-explain", "recompile_hops", "-args", 
				HOME + "input/" + DATASET1, output("R") };
	
			runTest(true, false, null, -1); 
			
			//check output 
			Assert.assertEquals(Double.valueOf(148), 
				readDMLMatrixFromHDFS("R").get(new CellIndex(1,1)));
		}
		finally {
			rtplatform = rtold;
			DMLScript.USE_LOCAL_SPARK_CONFIG = sparkConfigOld;
		}
	}
}