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

package org.apache.sysds.test.functions.builtin.part1;

import org.apache.sysds.common.Opcodes;
import org.junit.Assert;
import org.junit.Test;

import java.io.File;

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
	private final static String TEST_NAME4 = "GridSearchLMCV";
	private final static String TEST_DIR = "functions/builtin/";
	private final static String TEST_CLASS_DIR = TEST_DIR + BuiltinGridSearchTest.class.getSimpleName() + "/";
	
	private final static int _rows = 400;
	private final static int _cols = 20;
	private boolean _codegen = false;
	
	@Override
	public void setUp() {
		TestUtils.clearAssertionInformation();
		addTestConfiguration(TEST_NAME1,new TestConfiguration(TEST_CLASS_DIR, TEST_NAME1,new String[]{"R"}));
		addTestConfiguration(TEST_NAME2,new TestConfiguration(TEST_CLASS_DIR, TEST_NAME2,new String[]{"R"}));
		addTestConfiguration(TEST_NAME3,new TestConfiguration(TEST_CLASS_DIR, TEST_NAME3,new String[]{"R"}));
		addTestConfiguration(TEST_NAME4,new TestConfiguration(TEST_CLASS_DIR, TEST_NAME4,new String[]{"R"}));
	}
	
	@Test
	public void testGridSearchLmCP() {
		runGridSearch(TEST_NAME1, ExecMode.SINGLE_NODE, false);
	}
	
	@Test
	public void testGridSearchLmHybrid() {
		runGridSearch(TEST_NAME1, ExecMode.HYBRID, false);
	}
	
	@Test
	public void testGridSearchLmCodegenCP() {
		runGridSearch(TEST_NAME1, ExecMode.SINGLE_NODE, true);
	}
	
	@Test
	public void testGridSearchLmCodegenHybrid() {
		runGridSearch(TEST_NAME1, ExecMode.HYBRID, true);
	}
	
	@Test
	public void testGridSearchLmVerboseCP() {
		runGridSearch(TEST_NAME1, ExecMode.SINGLE_NODE, false, true);
	}
	
	@Test
	public void testGridSearchLmVerboseHybrid() {
		runGridSearch(TEST_NAME1, ExecMode.HYBRID, false, true);
	}
	
	@Test
	public void testGridSearchLmSpark() {
		runGridSearch(TEST_NAME1, ExecMode.SPARK, false);
	}
	
	@Test
	public void testGridSearchMLogregCP() {
		runGridSearch(TEST_NAME2, ExecMode.SINGLE_NODE, false);
	}
	
	@Test
	public void testGridSearchMLogregHybrid() {
		runGridSearch(TEST_NAME2, ExecMode.HYBRID, false);
	}
	
	@Test
	public void testGridSearchMLogregVerboseCP() {
		//verbose default
		runGridSearch(TEST_NAME2, ExecMode.SINGLE_NODE, false, true);
	}
	
	@Test
	public void testGridSearchMLogregVerboseHybrid() {
		//verbose default
		runGridSearch(TEST_NAME2, ExecMode.HYBRID, false, true);
	}
	
	
	@Test
	public void testGridSearchLm2CP() {
		runGridSearch(TEST_NAME3, ExecMode.SINGLE_NODE, false);
	}
	
	@Test
	public void testGridSearchLm2Hybrid() {
		runGridSearch(TEST_NAME3, ExecMode.HYBRID, false);
	}
	
	@Test
	public void testGridSearchLmCvCP() {
		runGridSearch(TEST_NAME4, ExecMode.SINGLE_NODE, false);
	}
	
	@Test
	public void testGridSearchLmCvHybrid() {
		runGridSearch(TEST_NAME4, ExecMode.HYBRID, false);
	}
	
	@Test
	public void testGridSearchMLogreg4CP() {
		runGridSearch(TEST_NAME2, ExecMode.SINGLE_NODE, 10, 4, false, false);
	}
	
	@Test
	public void testGridSearchMLogreg4Hybrid() {
		runGridSearch(TEST_NAME2, ExecMode.HYBRID, 10, 4, false, false);
	}
	
	
	private void runGridSearch(String testname, ExecMode et, boolean codegen) {
		runGridSearch(testname, et, _cols, 2, codegen, false); //binary classification
	}
	
	private void runGridSearch(String testname, ExecMode et, boolean codegen, boolean verbose) {
		runGridSearch(testname, et, _cols, 2, codegen, verbose); //binary classification
	}
	
	private void runGridSearch(String testname, ExecMode et, int cols, int nc, boolean codegen, boolean verbose)
	{
		ExecMode modeOld = setExecMode(et);
		_codegen = codegen;
		
		try {
			loadTestConfiguration(getTestConfiguration(testname));
			String HOME = SCRIPT_DIR + TEST_DIR;
	
			fullDMLScriptName = HOME + testname + ".dml";
			programArgs = new String[] {"-stats", "100", "-args",
				input("X"), input("y"), output("R"), String.valueOf(verbose).toUpperCase()};
			double max = testname.equals(TEST_NAME2) ? nc : 2;
			double[][] X = getRandomMatrix(_rows, cols, 0, 1, 0.8, 7);
			double[][] y = getRandomMatrix(_rows, 1, 1, max, 1, 1);
			writeInputMatrixWithMTD("X", X, true);
			writeInputMatrixWithMTD("y", y, true);
			
			runTest(true, EXCEPTION_NOT_EXPECTED, null, -1);
			
			//expected loss smaller than default invocation
			Assert.assertTrue(TestUtils.readDMLBoolean(output("R")));
			
			//correct handling of verbose flag
			if( verbose ) // 2 prints outside, if verbose more
				Assert.assertTrue(Statistics.getCPHeavyHitterCount(Opcodes.PRINT.toString())>100);
			
			//Assert.assertEquals(0, Statistics.getNoOfExecutedSPInst());
			//TODO analyze influence of multiple subsequent tests
		}
		finally {
			resetExecMode(modeOld);
		}
	}
	
	/**
	 * Override default configuration with custom test configuration to ensure
	 * scratch space and local temporary directory locations are also updated.
	 */
	@Override
	protected File getConfigTemplateFile() {
		return !_codegen ? super.getConfigTemplateFile() :
			getCodegenConfigFile(SCRIPT_DIR + "functions/codegenalg/", CodegenTestType.DEFAULT);
	}
}
