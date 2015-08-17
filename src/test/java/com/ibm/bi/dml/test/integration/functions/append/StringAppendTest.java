/**
 * (C) Copyright IBM Corp. 2010, 2015
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * 
 */

package com.ibm.bi.dml.test.integration.functions.append;


import org.junit.Test;

import com.ibm.bi.dml.api.DMLException;
import com.ibm.bi.dml.api.DMLScript;
import com.ibm.bi.dml.api.DMLScript.RUNTIME_PLATFORM;
import com.ibm.bi.dml.lops.LopProperties.ExecType;
import com.ibm.bi.dml.test.integration.AutomatedTestBase;
import com.ibm.bi.dml.test.integration.TestConfiguration;
import com.ibm.bi.dml.test.utils.TestUtils;

public class StringAppendTest extends AutomatedTestBase
{
	
	private final static String TEST_NAME1 = "basic_string_append";
	private final static String TEST_NAME2 = "loop_string_append";
		
	private final static String TEST_DIR = "functions/append/";

	@Override
	public void setUp() {
		TestUtils.clearAssertionInformation();
		addTestConfiguration(TEST_NAME1, new TestConfiguration(TEST_DIR, TEST_NAME1, new String[] {"S"}));
		addTestConfiguration(TEST_NAME2, new TestConfiguration(TEST_DIR, TEST_NAME2, new String[] {"S"}));
	}

	@Test
	public void testBasicStringAppendCP() {
		runStringAppendTest(TEST_NAME1, -1, false, ExecType.CP);
	}
	
	@Test
	public void testLoopStringAppendCP() {
		runStringAppendTest(TEST_NAME2, 100, false, ExecType.CP);
	}
	
	@Test
	public void testLoopStringAppendErrorCP() {
		runStringAppendTest(TEST_NAME2, 10000, true, ExecType.CP);
	}
	
	// -------------------------------------------------------

	@Test
	public void testBasicStringAppendSP() {
		runStringAppendTest(TEST_NAME1, -1, false, ExecType.SPARK);
	}
	
	@Test
	public void testLoopStringAppendSP() {
		runStringAppendTest(TEST_NAME2, 100, false, ExecType.SPARK);
	}
	
	@Test
	public void testLoopStringAppendErrorSP() {
		runStringAppendTest(TEST_NAME2, 10000, true, ExecType.SPARK);
	}
	
	// -------------------------------------------------------
	
	//note: there should be no difference to running in MR because scalar operation
	
	@Test
	public void testBasicStringAppendMR() {
		runStringAppendTest(TEST_NAME1, -1, false, ExecType.CP);
	}
	
	@Test
	public void testLoopStringAppendMR() {
		runStringAppendTest(TEST_NAME2, 100, false, ExecType.CP);
	}
	
	@Test
	public void testLoopStringAppendErrorMR() {
		runStringAppendTest(TEST_NAME2, 10000, true, ExecType.CP);
	}
	
	/**
	 * 
	 * @param platform
	 * @param rows
	 * @param cols1
	 * @param cols2
	 * @param cols3
	 */
	public void runStringAppendTest(String TEST_NAME, int iters, boolean exceptionExpected, ExecType et)
	{
		RUNTIME_PLATFORM oldPlatform = rtplatform;		

	    if(et == ExecType.SPARK) {
	    	rtplatform = RUNTIME_PLATFORM.SPARK;
	    }
	    else {
			rtplatform = (et==ExecType.MR) ? RUNTIME_PLATFORM.HADOOP : RUNTIME_PLATFORM.HYBRID;
	    }
		boolean sparkConfigOld = DMLScript.USE_LOCAL_SPARK_CONFIG;
		if( rtplatform == RUNTIME_PLATFORM.SPARK )
			DMLScript.USE_LOCAL_SPARK_CONFIG = true;
		
		try
		{
			TestConfiguration config = getTestConfiguration(TEST_NAME);			
			loadTestConfiguration(config);
			
			String RI_HOME = SCRIPT_DIR + TEST_DIR;
			fullDMLScriptName = RI_HOME + TEST_NAME + ".dml";
			programArgs = new String[]{"-args",  Integer.toString(iters),
					                             RI_HOME + OUTPUT_DIR + "C" };
			
			runTest(true, exceptionExpected, DMLException.class, 0);
		}
		catch(Exception ex)
		{
			ex.printStackTrace();
			throw new RuntimeException(ex);
		}
		finally
		{
			rtplatform = oldPlatform;	
			DMLScript.USE_LOCAL_SPARK_CONFIG = sparkConfigOld;
		}
		
	}
}
