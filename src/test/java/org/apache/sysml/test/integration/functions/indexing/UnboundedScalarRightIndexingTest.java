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

package org.apache.sysml.test.integration.functions.indexing;

import org.junit.Test;
import org.apache.sysml.api.DMLException;
import org.apache.sysml.api.DMLScript;
import org.apache.sysml.api.DMLScript.RUNTIME_PLATFORM;
import org.apache.sysml.lops.LopProperties.ExecType;
import org.apache.sysml.test.integration.AutomatedTestBase;
import org.apache.sysml.test.integration.TestConfiguration;


public class UnboundedScalarRightIndexingTest extends AutomatedTestBase
{
	private final static String TEST_NAME = "UnboundedScalarRightIndexingTest";
	private final static String TEST_DIR = "functions/indexing/";
	private final static String TEST_CLASS_DIR = TEST_DIR + UnboundedScalarRightIndexingTest.class.getSimpleName() + "/";
	
	@Override
	public void setUp() {
		addTestConfiguration(TEST_NAME, new TestConfiguration(TEST_CLASS_DIR, TEST_NAME, new String[] {}));
	}
	
	@Test
	public void testRightIndexingCPNonzero() {
		runRightIndexingTest(ExecType.CP, 7);
	}
	
	@Test
	public void testRightIndexingSPNonzero() {
		runRightIndexingTest(ExecType.SPARK, 7);
	}
	
	@Test
	public void testRightIndexingMRNonzero() {
		runRightIndexingTest(ExecType.MR, 7);
	}
	
	@Test
	public void testRightIndexingCPZero() {
		runRightIndexingTest(ExecType.CP, 0);
	}
	
	@Test
	public void testRightIndexingSPZero() {
		runRightIndexingTest(ExecType.SPARK, 0);
	}
	
	@Test
	public void testRightIndexingMRZero() {
		runRightIndexingTest(ExecType.MR, 0);
	}
	
	/**
	 * 
	 * @param et
	 */
	public void runRightIndexingTest( ExecType et, int val ) 
	{
		RUNTIME_PLATFORM platformOld = rtplatform;
		switch( et ){
			case MR: rtplatform = RUNTIME_PLATFORM.HADOOP; break;
			case SPARK: rtplatform = RUNTIME_PLATFORM.SPARK; break;
			default: rtplatform = RUNTIME_PLATFORM.HYBRID; break;
		}
	
		boolean sparkConfigOld = DMLScript.USE_LOCAL_SPARK_CONFIG;
		if( rtplatform == RUNTIME_PLATFORM.SPARK )
			DMLScript.USE_LOCAL_SPARK_CONFIG = true;

		try {
		    TestConfiguration config = getTestConfiguration(TEST_NAME);
		    loadTestConfiguration(config);
	        
	        String RI_HOME = SCRIPT_DIR + TEST_DIR;
			fullDMLScriptName = RI_HOME + TEST_NAME + ".dml";
			programArgs = new String[]{ "-args", String.valueOf(val) };
			fullRScriptName = RI_HOME + TEST_NAME + ".R";
			
			//run test (expected runtime exception)
			runTest(true, true, DMLException.class, -1);
		}
		finally
		{
			rtplatform = platformOld;
			DMLScript.USE_LOCAL_SPARK_CONFIG = sparkConfigOld;
		}
	}
}
