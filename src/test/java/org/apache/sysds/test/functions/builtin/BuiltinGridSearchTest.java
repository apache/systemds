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

import org.junit.Assert;
import org.junit.Test;

import org.apache.sysds.common.Types.ExecMode;
import org.apache.sysds.lops.LopProperties.ExecType;
import org.apache.sysds.test.AutomatedTestBase;
import org.apache.sysds.test.TestConfiguration;
import org.apache.sysds.test.TestUtils;


public class BuiltinGridSearchTest extends AutomatedTestBase
{
	private final static String TEST_NAME = "GridSearchLM";
	private final static String TEST_DIR = "functions/builtin/";
	private final static String TEST_CLASS_DIR = TEST_DIR + BuiltinGridSearchTest.class.getSimpleName() + "/";
	
	private final static int rows = 300;
	private final static int cols = 20;
	
	@Override
	public void setUp() {
		addTestConfiguration(TEST_NAME,new TestConfiguration(TEST_CLASS_DIR, TEST_NAME,new String[]{"R"})); 
	}
	
	@Test
	public void testGridSearchCP() {
		runGridSearch(ExecType.CP);
	}
	
	@Test
	public void testGridSearchSpark() {
		runGridSearch(ExecType.SPARK);
	}
	
	private void runGridSearch(ExecType et)
	{
		ExecMode modeOld = setExecMode(et);
		try {
			loadTestConfiguration(getTestConfiguration(TEST_NAME));
			String HOME = SCRIPT_DIR + TEST_DIR;
	
			fullDMLScriptName = HOME + TEST_NAME + ".dml";
			programArgs = new String[] {"-args", input("X"), input("y"), output("R")};
			double[][] X = getRandomMatrix(rows, cols, 0, 1, 0.8, -1);
			double[][] y = getRandomMatrix(rows, 1, 0, 1, 0.8, -1);
			writeInputMatrixWithMTD("X", X, true);
			writeInputMatrixWithMTD("y", y, true);
			
			runTest(true, EXCEPTION_NOT_EXPECTED, null, -1);
			
			//expected loss smaller than default invocation
			Assert.assertTrue(TestUtils.readDMLBoolean(output("R")));
		}
		finally {
			resetExecMode(modeOld);
		}
	}
}
