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

package org.apache.sysds.test.functions.builtin;

import java.util.ArrayList;
import java.util.List;

import org.junit.Test;
import org.apache.sysds.test.AutomatedTestBase;
import org.apache.sysds.test.TestConfiguration;


public class BuiltinCVLmTest extends AutomatedTestBase
{
	private final static String TEST_NAME = "cvlm";
	private final static String TEST_DIR = "functions/builtin/";
	private final static String TEST_CLASS_DIR = TEST_DIR + BuiltinCVLmTest.class.getSimpleName() + "/";
	
	private final static int rows = 100;
	private final static int cols = 10;
	
	@Override
	public void setUp() {
		addTestConfiguration(TEST_NAME,new TestConfiguration(TEST_CLASS_DIR, TEST_NAME,new String[]{"B"})); 
	}
	
	@Test
	public void testlmCV() {
		runtestlmCV();
	}
	
	private void runtestlmCV()
	{
		loadTestConfiguration(getTestConfiguration(TEST_NAME));
		String HOME = SCRIPT_DIR + TEST_DIR;
		fullDMLScriptName = HOME + TEST_NAME + ".dml";
		List<String> proArgs = new ArrayList<>();
		
		int k = 3;
		proArgs.add("-args");
		proArgs.add(input("X"));
		proArgs.add(input("y"));
		proArgs.add(String.valueOf(k));
		proArgs.add(output("y_predict"));
		proArgs.add(output("beta"));
		programArgs = proArgs.toArray(new String[proArgs.size()]);
		double[][] X = getRandomMatrix(rows, cols, 0, 1, 0.8, -1);
		double[][] y = getRandomMatrix(rows, 1, 0, 1, 0.8, -1);
		writeInputMatrixWithMTD("X", X, true);
		writeInputMatrixWithMTD("y", y, true);
		
		runTest(true, EXCEPTION_NOT_EXPECTED, null, -1);
	}
}
