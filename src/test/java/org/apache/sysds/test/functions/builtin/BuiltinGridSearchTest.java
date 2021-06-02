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
import org.apache.sysds.test.AutomatedTestBase;
import org.apache.sysds.test.TestConfiguration;
import org.apache.sysds.test.TestUtils;
import org.apache.sysds.utils.Statistics;

public class BuiltinGridSearchTest extends AutomatedTestBase
{
	private final static String TEST_NAME1 = "GridSearchLM";
	private final static String TEST_NAME2 = "GridSearchMLogreg";
	private final static String TEST_NAME3 = "GridSearchLM2";
	private final static String TEST_DIR = "functions/builtin/";
	private final static String TEST_CLASS_DIR = TEST_DIR + BuiltinGridSearchTest.class.getSimpleName() + "/";
	
	private final static int rows = 300;
	private final static int cols = 20;
	
	@Override
	public void setUp() {
		addTestConfiguration(TEST_NAME1,new TestConfiguration(TEST_CLASS_DIR, TEST_NAME1,new String[]{"R"}));
		addTestConfiguration(TEST_NAME2,new TestConfiguration(TEST_CLASS_DIR, TEST_NAME2,new String[]{"R"}));
		addTestConfiguration(TEST_NAME3,new TestConfiguration(TEST_CLASS_DIR, TEST_NAME3,new String[]{"R"}));
	}
	
	@Test
	public void testGridSearchLmCP() {
		runGridSearch(TEST_NAME1, ExecMode.SINGLE_NODE);
	}
	
	@Test
	public void testGridSearchLmHybrid() {
		runGridSearch(TEST_NAME1, ExecMode.HYBRID);
	}
	
	@Test
	public void testGridSearchLmSpark() {
		runGridSearch(TEST_NAME1, ExecMode.SPARK);
	}
	
	@Test
	public void testGridSearchMLogregCP() {
		runGridSearch(TEST_NAME2, ExecMode.SINGLE_NODE);
	}
	
	@Test
	public void testGridSearchMLogregHybrid() {
		runGridSearch(TEST_NAME2, ExecMode.HYBRID);
	}
	
	@Test
	public void testGridSearchLm2CP() {
		runGridSearch(TEST_NAME3, ExecMode.SINGLE_NODE);
	}
	
	@Test
	public void testGridSearchLm2Hybrid() {
		runGridSearch(TEST_NAME3, ExecMode.HYBRID);
	}
	
	private void runGridSearch(String testname, ExecMode et)
	{
		ExecMode modeOld = setExecMode(et);
		try {
			loadTestConfiguration(getTestConfiguration(testname));
			String HOME = SCRIPT_DIR + TEST_DIR;
	
			fullDMLScriptName = HOME + testname + ".dml";
			programArgs = new String[] {"-args", input("X"), input("y"), output("R")};
			double[][] X = getRandomMatrix(rows, cols, 0, 1, 0.8, 7);
			double[][] y = getRandomMatrix(rows, 1, 1, 2, 1, 1);
			writeInputMatrixWithMTD("X", X, true);
			writeInputMatrixWithMTD("y", y, true);
			
			runTest(true, EXCEPTION_NOT_EXPECTED, null, -1);
			
			//expected loss smaller than default invocation
			Assert.assertTrue(TestUtils.readDMLBoolean(output("R")));
			if( et != ExecMode.SPARK )
				Assert.assertEquals(0, Statistics.getNoOfExecutedSPInst());
		}
		finally {
			resetExecMode(modeOld);
		}
	}
}
