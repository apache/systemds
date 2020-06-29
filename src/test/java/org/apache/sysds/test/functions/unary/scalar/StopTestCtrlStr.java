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

import org.apache.sysds.common.Types.ExecMode;
import org.apache.sysds.runtime.DMLScriptException;
import org.apache.sysds.test.AutomatedTestBase;
import org.apache.sysds.test.TestConfiguration;
import org.junit.Test;


/**
 * Tests the print function
 */
public class StopTestCtrlStr extends AutomatedTestBase 
{
	
	private final static String TEST_DIR = "functions/unary/scalar/";
	private final static String TEST_CLASS_DIR = TEST_DIR + StopTestCtrlStr.class.getSimpleName() + "/";
	private final static String TEST_NAME = "StopTestLoops";
	
	private final static int rows = 100;
	private final static String inputName = "in";
	private final static double cutoff = 0.6;
	
	@Override
	public void setUp() {
		addTestConfiguration(TEST_NAME, new TestConfiguration(TEST_CLASS_DIR, TEST_NAME, new String[] {}));
	}

	@Test
	public void testStopParfor() {
		
		getAndLoadTestConfiguration(TEST_NAME);
		
		String STOP_HOME = SCRIPT_DIR + TEST_DIR;
		fullDMLScriptName = STOP_HOME + TEST_NAME + "_parfor.dml";
		programArgs = new String[]{};
		
		boolean exceptionExpected = true;
		int expectedNumberOfJobs = 0;
		
		runTest(true, exceptionExpected, null, expectedNumberOfJobs); 
	}
	
	@Test
	public void testStopFor() {
		testLoop("for", ExecMode.HYBRID);
	}
	
	@Test
	public void testStopWhile() {
		testLoop("while", ExecMode.HYBRID);
	}
	
	@Test
	public void testStopFunction() {
		testLoop("fn", ExecMode.HYBRID);
	}
	
	private void testLoop(String loop, ExecMode rt) {
		
		ExecMode oldRT = rtplatform;
		rtplatform = rt;
		
		getAndLoadTestConfiguration(TEST_NAME);
		String STOP_HOME = SCRIPT_DIR + TEST_DIR;
		
		fullDMLScriptName = STOP_HOME + TEST_NAME + "_" + loop + ".dml";
		programArgs = new String[]{"-args", input(inputName), Integer.toString(rows), Double.toString(cutoff)};
		
        double[][] vector = getRandomMatrix(rows, 1, 0, 1, 1, System.currentTimeMillis());
        writeInputMatrix(inputName, vector);

		
		int cutoffIndex = findIndexAtCutoff(vector, cutoff);
		String expextedStdErr = "";
        if(cutoffIndex <= rows) {
			// Expected exception
			expextedStdErr = "Element " + cutoffIndex + ".";
			runTest(DMLScriptException.class, expextedStdErr,-1);
        }
        else {
			// No exception
        	String outMessage = "None made to cutoff.";
			ByteArrayOutputStream stdOut = runTest(true, false, null, -1);
			assertTrue(bufferContainsString(stdOut, outMessage));
        }
		
		
		rtplatform = oldRT;
	}
	
	private static int findIndexAtCutoff(double[][] vector, double cutoff) {
		int i=1;
		while(i<=vector.length) {
			if(vector[i-1][0] > cutoff)
				break;
			i++;
		}
		return i;
	}
}
