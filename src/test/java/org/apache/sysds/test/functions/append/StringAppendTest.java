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

package org.apache.sysds.test.functions.append;


import org.junit.Test;
import org.apache.sysds.api.DMLException;
import org.apache.sysds.common.Types.ExecMode;
import org.apache.sysds.lops.LopProperties.ExecType;
import org.apache.sysds.test.AutomatedTestBase;
import org.apache.sysds.test.TestConfiguration;
import org.apache.sysds.test.TestUtils;

public class StringAppendTest extends AutomatedTestBase
{
	private final static String TEST_NAME1 = "basic_string_append";
	private final static String TEST_NAME2 = "loop_string_append";
	
	private final static String TEST_DIR = "functions/append/";
	private final static String TEST_CLASS_DIR = TEST_DIR + StringAppendTest.class.getSimpleName() + "/";

	@Override
	public void setUp() {
		TestUtils.clearAssertionInformation();
		addTestConfiguration(TEST_NAME1, new TestConfiguration(TEST_CLASS_DIR, TEST_NAME1, new String[] {"S"}));
		addTestConfiguration(TEST_NAME2, new TestConfiguration(TEST_CLASS_DIR, TEST_NAME2, new String[] {"S"}));
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
	
	public void runStringAppendTest(String TEST_NAME, int iters, boolean exceptionExpected, ExecType et)
	{
		ExecMode oldPlatform = setExecMode(et);
		try {
			getAndLoadTestConfiguration(TEST_NAME);
			
			String RI_HOME = SCRIPT_DIR + TEST_DIR;
			fullDMLScriptName = RI_HOME + TEST_NAME + ".dml";
			programArgs = new String[]{
				"-args", Integer.toString(iters), output("C") };
			
			runTest(true, exceptionExpected, DMLException.class, 0);
		}
		catch(Exception ex) {
			ex.printStackTrace();
			throw new RuntimeException(ex);
		}
		finally {
			resetExecMode(oldPlatform);
		}
	}
}
