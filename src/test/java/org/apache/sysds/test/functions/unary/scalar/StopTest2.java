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

package org.apache.sysds.test.functions.unary.scalar;

import static org.junit.Assert.assertTrue;

import java.io.ByteArrayOutputStream;

import org.apache.sysds.runtime.DMLScriptException;
import org.apache.sysds.test.AutomatedTestBase;
import org.apache.sysds.test.TestConfiguration;
import org.junit.Test;


/**
 * Tests the print function
 */
public class StopTest2 extends AutomatedTestBase 
{
	
	private final static String TEST_DIR = "functions/unary/scalar/";
	private final static String TEST_CLASS_DIR = TEST_DIR + StopTest2.class.getSimpleName() + "/";
	private final static String TEST_STOP = "StopTest";
	
	@Override
	public void setUp() {
		setOutputBuffering(true);
		availableTestConfigurations.put(TEST_STOP, new TestConfiguration(TEST_CLASS_DIR, TEST_STOP, new String[] {}));
	}

	String errMessage = "Stop Here.";
	String outMessage = "10.0";
	
	@Test
	public void testStop2() {
		runStopTest(2);
	}
	
	public void runStopTest(int test_num) {
		
		TestConfiguration config = availableTestConfigurations.get("StopTest");
		
		String STOP_HOME = SCRIPT_DIR + TEST_DIR;
		fullDMLScriptName = STOP_HOME + TEST_STOP + test_num + ".dml";
		programArgs = new String[]{"-args", errMessage};
		
		loadTestConfiguration(config);

		ByteArrayOutputStream stdOut = runTest(true, true, DMLScriptException.class, -1); 

		assertTrue(bufferContainsString(stdOut, outMessage));
	}
	
}
