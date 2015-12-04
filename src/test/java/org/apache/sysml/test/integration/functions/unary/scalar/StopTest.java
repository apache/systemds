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

package org.apache.sysml.test.integration.functions.unary.scalar;

import org.junit.Test;

import org.apache.sysml.test.integration.AutomatedTestBase;
import org.apache.sysml.test.integration.TestConfiguration;


/**
 * Tests the print function
 */
public class StopTest extends AutomatedTestBase 
{
	
	private final static String TEST_DIR = "functions/unary/scalar/";
	private final static String TEST_STOP = "StopTest";
	
	@Override
	public void setUp() {
		baseDirectory = SCRIPT_DIR + "functions/unary/scalar/";

		availableTestConfigurations.put(TEST_STOP, new TestConfiguration(TEST_DIR, TEST_STOP, new String[] {}));
	}

	String errMessage = "M is all 0 matrix.";
	String outMessage = "Sum(M2) = 10.0";
	
	@Test
	public void testStop1_0() {
		runStopTest(1,0);
	}
	
	@Test
	public void testStop1_1() {
		runStopTest(1,1);
	}
	
	public void runStopTest(int test_num, int val) {
		
		TestConfiguration config = availableTestConfigurations.get("StopTest");
		
		String STOP_HOME = SCRIPT_DIR + TEST_DIR;
		fullDMLScriptName = STOP_HOME + TEST_STOP + test_num + ".dml";
		programArgs = new String[]{"-args", Integer.toString(val), errMessage};
		
		loadTestConfiguration(config);
		boolean exceptionExpected = false;
		int expectedNumberOfJobs = 0;
		
		if ( val == 0 )
			setExpectedStdErr(errMessage);
		else 
			setExpectedStdOut(outMessage);
			
		runTest(true, exceptionExpected, null, expectedNumberOfJobs); 
	}
	
}
