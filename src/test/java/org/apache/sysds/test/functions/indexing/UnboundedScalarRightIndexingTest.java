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

package org.apache.sysds.test.functions.indexing;

import org.apache.sysds.api.DMLScript;
import org.apache.sysds.common.Types.ExecMode;
import org.apache.sysds.lops.LopProperties.ExecType;
import org.apache.sysds.runtime.DMLRuntimeException;
import org.apache.sysds.test.AutomatedTestBase;
import org.apache.sysds.test.TestConfiguration;
import org.junit.Test;


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
	public void testRightIndexingCPZero() {
		runRightIndexingTest(ExecType.CP, 0);
	}
	
	@Test
	public void testRightIndexingSPZero() {
		runRightIndexingTest(ExecType.SPARK, 0);
	}
	
	public void runRightIndexingTest( ExecType et, int val ) 
	{
		ExecMode platformOld = rtplatform;
		switch( et ){
			case SPARK: rtplatform = ExecMode.SPARK; break;
			default: rtplatform = ExecMode.HYBRID; break;
		}
	
		boolean sparkConfigOld = DMLScript.USE_LOCAL_SPARK_CONFIG;
		if( rtplatform == ExecMode.SPARK )
			DMLScript.USE_LOCAL_SPARK_CONFIG = true;

		try {
		    TestConfiguration config = getTestConfiguration(TEST_NAME);
		    loadTestConfiguration(config);
	        
	        String RI_HOME = SCRIPT_DIR + TEST_DIR;
			fullDMLScriptName = RI_HOME + TEST_NAME + ".dml";
			programArgs = new String[]{ "-args", String.valueOf(val) };
			fullRScriptName = RI_HOME + TEST_NAME + ".R";
			
			//run test (expected runtime exception)
			runTest(true, true, DMLRuntimeException.class, -1);
		}
		finally
		{
			rtplatform = platformOld;
			DMLScript.USE_LOCAL_SPARK_CONFIG = sparkConfigOld;
		}
	}
}
